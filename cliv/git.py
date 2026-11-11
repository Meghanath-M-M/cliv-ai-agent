"""Safe, first-class git integration for cliv.

All operations run git directly with ``shell=False`` and a conservative
allowlist of subcommands.  Nothing is committed or branched without an
explicit opt-in flag, and auto-commit only ever touches files that cliv
itself modified during the session -- never unrelated working-tree changes.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Subcommands considered safe to run automatically.  Destructive history
# rewriting (reset, rebase, filter-branch, etc.) is deliberately excluded.
ALLOWED_GIT_SUBCOMMANDS = {
    "add",
    "branch",
    "checkout",
    "commit",
    "diff",
    "log",
    "merge-base",
    "rev-parse",
    "show",
    "status",
    "switch",
}


class GitError(RuntimeError):
    """Raised when a git operation cannot be completed safely."""


def _run(args: Sequence[str], *, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a git subcommand, raising ``GitError`` on failure."""

    if not args or args[0] not in ALLOWED_GIT_SUBCOMMANDS:
        raise GitError(f"git subcommand not allowlisted: {args[0] if args else '(none)'}")
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitError("git is not installed or not on PATH") from exc
    except OSError as exc:  # pragma: no cover - environment error
        raise GitError(f"could not run git: {exc}") from exc


def is_git_repo(*, cwd: Optional[str] = None) -> bool:
    """Return True when *cwd* is inside a git repository."""

    result = _run(["rev-parse", "--is-inside-work-tree"], cwd=cwd)
    return result.returncode == 0 and result.stdout.strip() == "true"


def current_branch(*, cwd: Optional[str] = None) -> Optional[str]:
    """Return the name of the current branch, or ``None`` when detached."""

    result = _run(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    if result.returncode != 0:
        return None
    name = result.stdout.strip()
    return name if name != "HEAD" else None


def branch_exists(name: str, *, cwd: Optional[str] = None) -> bool:
    """Return True when a local branch named *name* exists."""

    result = _run(["rev-parse", "--verify", f"refs/heads/{name}"], cwd=cwd)
    return result.returncode == 0


def create_branch(name: str, *, cwd: Optional[str] = None, from_branch: Optional[str] = None) -> None:
    """Create a new branch named *name*, failing if it already exists."""

    if branch_exists(name, cwd=cwd):
        raise GitError(f"branch already exists: {name}")
    args = ["checkout", "-b", name]
    if from_branch:
        args.append(from_branch)
    result = _run(args, cwd=cwd)
    if result.returncode != 0:
        raise GitError(f"failed to create branch {name}: {result.stderr.strip()}")


def switch_branch(name: str, *, cwd: Optional[str] = None) -> None:
    """Switch to an existing branch named *name*."""

    if not branch_exists(name, cwd=cwd):
        raise GitError(f"branch does not exist: {name}")
    result = _run(["switch", name], cwd=cwd)
    if result.returncode != 0:
        raise GitError(f"failed to switch to branch {name}: {result.stderr.strip()}")


def ensure_branch(name: str, *, cwd: Optional[str] = None, from_branch: Optional[str] = None) -> bool:
    """Ensure *name* exists and is checked out. Returns True if a branch was created."""

    if branch_exists(name, cwd=cwd):
        switch_branch(name, cwd=cwd)
        return False
    create_branch(name, cwd=cwd, from_branch=from_branch)
    return True


def status_short(*, cwd: Optional[str] = None) -> str:
    """Return ``git status -s`` output."""

    result = _run(["status", "-s"], cwd=cwd)
    if result.returncode != 0:
        raise GitError(f"git status failed: {result.stderr.strip()}")
    return result.stdout.strip()


def recent_commits(limit: int = 5, *, cwd: Optional[str] = None) -> List[str]:
    """Return the last *limit* commits as one-line summaries."""

    if not is_git_repo(cwd=cwd):
        return []
    result = _run(["log", f"-{limit}", "--oneline"], cwd=cwd)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]


def add(paths: Sequence[str], *, cwd: Optional[str] = None) -> None:
    """Stage *paths* for commit."""

    if not paths:
        return
    result = _run(["add", *paths], cwd=cwd)
    if result.returncode != 0:
        raise GitError(f"git add failed: {result.stderr.strip()}")


def commit(message: str, paths: Sequence[str], *, cwd: Optional[str] = None) -> str:
    """Stage and commit *paths* with *message*, returning the new commit id."""

    add(paths, cwd=cwd)
    result = _run(
        ["commit", "-m", message, "--", *paths],
        cwd=cwd,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "nothing to commit" in stderr or "nothing added" in stderr:
            return ""
        raise GitError(f"git commit failed: {stderr}")
    rev = _run(["rev-parse", "HEAD"], cwd=cwd)
    return rev.stdout.strip() if rev.returncode == 0 else ""


@dataclass(frozen=True)
class GitInfo:
    repo: bool
    branch: Optional[str]
    dirty: bool
