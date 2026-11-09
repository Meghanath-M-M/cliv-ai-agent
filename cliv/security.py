"""Filesystem safety rules shared by every tool.

The agent is confined to the project directory (the current working
directory by default).  Resolution happens with ``Path.resolve()`` so a
symlink cannot be used to escape the root.  Sensitive files such as
``.env`` and private keys are blocked even inside the project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

# Names that should never be read or modified by the agent.  ``.env``
# variants are handled separately so that ``.env.example`` stays readable.
SENSITIVE_NAMES = {
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}

SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}

# Path components that indicate a private/configuration directory.
SENSITIVE_PARTS = {".aws", ".azure", ".gnupg", ".ssh"}

# Environment files that are safe because they are templates.
ENV_TEMPLATES = {
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.dist",
}

# Writing inside these directories is never allowed.
PROTECTED_WRITE_PARTS = {".git"}


def project_root() -> Path:
    """Return the effective project root.

    ``CLIV_PROJECT_ROOT`` exists for tests and embedding, but normal CLI
    usage simply uses the directory in which ``cliv`` was started.
    """

    configured = os.environ.get("CLIV_PROJECT_ROOT")
    root = Path(configured).expanduser() if configured else Path.cwd()
    return root.resolve()


def is_sensitive_path(path: Path) -> bool:
    """Return True when *path* points at a credential or secret file."""

    name = path.name.lower()
    if name in {item.lower() for item in SENSITIVE_NAMES}:
        return True
    if name in ENV_TEMPLATES:
        return False
    if name.startswith(".env"):
        return True
    if path.suffix.lower() in SENSITIVE_SUFFIXES:
        return True
    lowered_parts = [part.lower() for part in path.parts]
    if any(part in SENSITIVE_PARTS for part in lowered_parts):
        return True
    # The agent's own config/history/log/embedding cache can contain keys
    # and conversation text.  Block it when a user starts cliv in $HOME.
    if ".config" in lowered_parts and "cliv" in lowered_parts:
        return True
    return False


def resolve_project_path(
    path: str,
    *,
    for_write: bool = False,
    allow_sensitive: bool = False,
    root: Optional[Path] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve *path* and enforce the project-root boundary.

    Returns ``(resolved_path, None)`` on success or
    ``(None, error_message)`` on failure.
    """

    root = (root or project_root()).resolve()
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        # Relative paths are relative to the project root, not necessarily
        # to the process's current directory.
        raw = root / raw

    try:
        resolved = raw.resolve()
    except (OSError, RuntimeError) as exc:
        return None, f"Error: Could not resolve path '{path}': {exc}"

    try:
        resolved.relative_to(root)
    except ValueError:
        return (
            None,
            f"Error: Path is outside the project directory and is blocked: {path}",
        )

    if not allow_sensitive and is_sensitive_path(resolved):
        return None, f"Error: Access to sensitive file is blocked: {path}"

    if for_write:
        rel_parts = set(resolved.relative_to(root).parts)
        if rel_parts & PROTECTED_WRITE_PARTS:
            return None, f"Error: Writing inside protected directory is blocked: {path}"
        if resolved.exists() and not resolved.is_file():
            return None, f"Error: Path is not a regular file: {path}"

    return resolved, None


def is_within_root(path: Path, root: Optional[Path] = None) -> bool:
    """Convenience predicate used by tests and callers."""

    root = (root or project_root()).resolve()
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False
