"""Conservative risk assessment for file changes.

The same classifier is used by the approval preview and by the safety
snapshot/undo layer.  It is deliberately heuristic: the goal is to make a
reviewer look twice, not to pretend that a path name is a security proof.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from cliv.security import project_root

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".cxx",
    ".hpp",
    ".cs",
    ".swift",
    ".scala",
    ".sh",
    ".bash",
    ".ps1",
    ".sql",
}

CONFIG_SUFFIXES = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".env",
    ".properties",
}

DOC_SUFFIXES = {".md", ".rst", ".txt", ".adoc", ".license"}

DEPENDENCY_NAMES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "requirements.txt",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "cargo.toml",
    "cargo.lock",
    "go.mod",
    "go.sum",
    "gemfile",
    "gemfile.lock",
    "composer.json",
    "composer.lock",
}

CI_NAMES = {
    "jenkinsfile",
    ".gitlab-ci.yml",
    ".travis.yml",
    "azure-pipelines.yml",
    "appveyor.yml",
}

DEPLOYMENT_NAMES = {
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "procfile",
    "makefile",
}

SECURITY_WORDS = {
    "auth",
    "authentication",
    "authorization",
    "credential",
    "credentials",
    "crypto",
    "login",
    "password",
    "permission",
    "permissions",
    "secret",
    "security",
    "token",
}

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

UNSAFE_CODE_PATTERNS = (
    (re.compile(r"\beval\s*\("), "adds eval()"),
    (re.compile(r"\bexec\s*\("), "adds exec()"),
    (re.compile(r"\bos\.system\s*\("), "adds os.system()"),
    (re.compile(r"\bsubprocess\b"), "adds subprocess usage"),
    (re.compile(r"\bshutil\.rmtree\s*\("), "adds recursive deletion"),
    (re.compile(r"\bpickle\.loads?\s*\("), "adds pickle deserialization"),
    (re.compile(r"\byaml\.load\s*\("), "adds unsafe YAML loading"),
    (re.compile(r"shell\s*=\s*True"), "enables shell execution"),
)

NETWORK_PATTERNS = (
    re.compile(r"\brequests\.(?:get|post|put|delete|request)\s*\("),
    re.compile(r"\bhttpx\.(?:get|post|put|delete|request)\s*\("),
    re.compile(r"\burllib\.request\b"),
    re.compile(r"\bsocket\.socket\s*\("),
)

SECRET_WORDS = re.compile(
    r"\b(?:api[_-]?key|password|secret|token|private[_-]?key)\b", re.IGNORECASE
)

TEST_FUNCTION = re.compile(r"(?m)^\s*(?:async\s+)?def\s+(test_\w+)\s*\(")
ASSERTION = re.compile(r"(?m)^\s*assert\b")

MAX_REASONS = 12


@dataclass(frozen=True)
class RiskAssessment:
    """A risk level plus the concrete reasons that produced it."""

    level: str = "low"
    reasons: Tuple[str, ...] = ()

    @property
    def score(self) -> int:
        return RISK_ORDER.get(self.level, 0)

    @property
    def label(self) -> str:
        return self.level.upper()

    def render(self, *, path: Optional[str] = None) -> str:
        """Render a plain-text header suitable for an approval preview."""

        if path is not None:
            detail = "; ".join(self.reasons)
            suffix = f": {detail}" if detail else ""
            return f"[risk] {self.label} {path}{suffix}"

        lines = [f"RISK: {self.label}"]
        if self.reasons:
            if len(self.reasons) == 1 and len(self.reasons[0]) <= 90:
                lines.append(f"Why: {self.reasons[0]}")
            else:
                lines.append("Why:")
                for reason in self.reasons:
                    lines.append(f"  - {reason}")
        return "\n".join(lines)


def _relative_parts(path: Path, root: Path) -> List[str]:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return [part.lower() for part in path.parts]
    return [part.lower() for part in relative.parts]


def _is_test_path(parts: Sequence[str], name: str) -> bool:
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "tests" in parts
        or "test" in parts
    )


def _is_security_path(parts: Sequence[str], name: str) -> bool:
    stem = Path(name).stem
    words = set(parts) | {stem}
    return bool(words & SECURITY_WORDS) or any(
        word in stem for word in ("auth", "security", "secret", "token", "password")
    )


def _is_sensitive_path(path: Path) -> bool:
    name = path.name.lower()
    if name in {item.lower() for item in SENSITIVE_NAMES}:
        return True
    if name in {".env.example", ".env.sample", ".env.template", ".env.dist"}:
        return False
    if name.startswith(".env"):
        return True
    return path.suffix.lower() in SENSITIVE_SUFFIXES


def _merge(level: str, reasons: Iterable[str]) -> Tuple[str, List[str]]:
    """Return the highest level and a de-duplicated reason list."""

    best = level
    result: List[str] = []
    for reason in reasons:
        if reason and reason not in result:
            result.append(reason)
    return best, result[:MAX_REASONS]


def _raise_level(current: str, candidate: str) -> str:
    if RISK_ORDER.get(candidate, 0) > RISK_ORDER.get(current, 0):
        return candidate
    return current


def classify_path(
    path: Path,
    *,
    root: Optional[Path] = None,
    operation: str = "edit",
) -> RiskAssessment:
    """Classify a path by its role in the repository."""

    root = (root or project_root()).resolve()
    parts = _relative_parts(path, root)
    name = path.name.lower()
    suffix = path.suffix.lower()
    level = "low"
    reasons: List[str] = []

    if _is_sensitive_path(path) or any(part in {".git", ".ssh", ".aws"} for part in parts):
        level = "critical"
        reasons.append("touches a credential, secret, or protected control file")

    if operation == "delete":
        level = _raise_level(level, "high")
        reasons.append("deletes a file")

    if ".github" in parts and "workflows" in parts:
        level = _raise_level(level, "high")
        reasons.append("modifies CI workflow configuration")
    elif name in CI_NAMES or ".circleci" in parts:
        level = _raise_level(level, "high")
        reasons.append("modifies CI configuration")

    if name in DEPENDENCY_NAMES or (
        name.startswith("requirements") and name.endswith(".txt")
    ):
        level = _raise_level(level, "high")
        reasons.append("modifies dependency configuration")

    if _is_security_path(parts, name):
        level = _raise_level(level, "high")
        reasons.append("touches authentication or security-sensitive code")

    if any(part in {"migrations", "alembic"} for part in parts) or name == "schema.sql":
        level = _raise_level(level, "high")
        reasons.append("modifies database schema or migration code")

    if name in DEPLOYMENT_NAMES:
        level = _raise_level(level, "high" if name == "makefile" else "medium")
        reasons.append("modifies build or deployment configuration")

    if suffix in CONFIG_SUFFIXES and not reasons:
        level = _raise_level(level, "medium")
        reasons.append("modifies configuration")

    if suffix in SOURCE_SUFFIXES:
        level = _raise_level(level, "medium")
        if _is_test_path(parts, name):
            reasons.append("modifies tests")
        else:
            reasons.append("modifies source code")

    if suffix in DOC_SUFFIXES or name in {"license", "notice", "changelog"}:
        if not reasons:
            reasons.append("documentation update")

    if not reasons:
        reasons.append("small repository change")

    return RiskAssessment(level=level, reasons=tuple(reasons))


def _changed_text(old: str, new: str) -> Tuple[int, int, str]:
    """Return (deleted_lines, added_lines, added_text)."""

    old_lines = old.splitlines()
    new_lines = new.splitlines()
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    deleted = 0
    added = 0
    added_parts: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            deleted += max(0, i2 - i1)
        if tag in ("insert", "replace"):
            added += max(0, j2 - j1)
            added_parts.extend(new_lines[j1:j2])
    return deleted, added, "\n".join(added_parts)


def assess_edit(
    path: Path,
    old: str,
    new: str,
    *,
    root: Optional[Path] = None,
    operation: str = "edit",
) -> RiskAssessment:
    """Assess a concrete edit, including the size and shape of the change."""

    base = classify_path(path, root=root, operation=operation)
    level = base.level
    reasons = list(base.reasons)

    old_count = len(old.splitlines())
    new_count = len(new.splitlines())
    deleted, added, added_text = _changed_text(old, new)

    existed = bool(old) or path.exists()
    if not existed and new:
        level = _raise_level(level, "medium" if path.suffix.lower() in SOURCE_SUFFIXES else "low")
        reasons.append("creates a new file")

    if existed and old and not new:
        level = _raise_level(level, "high")
        reasons.append("empties an existing file")

    if deleted:
        ratio = deleted / max(1, old_count)
        if deleted >= 200 or (ratio >= 0.5 and deleted >= 20):
            level = _raise_level(level, "high")
            reasons.append(
                f"large deletion: {deleted} lines ({ratio:.0%} of the file)"
            )
        elif deleted >= 50:
            level = _raise_level(level, "medium")
            reasons.append(f"removes {deleted} lines")

    old_tests = len(TEST_FUNCTION.findall(old))
    new_tests = len(TEST_FUNCTION.findall(new))
    if new_tests < old_tests:
        level = _raise_level(level, "high")
        reasons.append(f"removes {old_tests - new_tests} test function(s)")

    old_asserts = len(ASSERTION.findall(old))
    new_asserts = len(ASSERTION.findall(new))
    if new_asserts < old_asserts:
        level = _raise_level(level, "medium")
        reasons.append(f"removes {old_asserts - new_asserts} assertion(s)")

    if added_text:
        for pattern, reason in UNSAFE_CODE_PATTERNS:
            if pattern.search(added_text):
                level = _raise_level(level, "high")
                reasons.append(reason)
                break
        if any(pattern.search(added_text) for pattern in NETWORK_PATTERNS):
            level = _raise_level(level, "medium")
            reasons.append("adds network access")
        if SECRET_WORDS.search(added_text):
            level = _raise_level(level, "medium")
            reasons.append("mentions secret-like identifiers")

    if added > 0 and deleted == 0 and not reasons:
        reasons.append(f"adds {added} lines")

    final_level, final_reasons = _merge(level, reasons)
    return RiskAssessment(level=final_level, reasons=tuple(final_reasons))


def assess_delete(path: Path, *, root: Optional[Path] = None) -> RiskAssessment:
    """Assess a file deletion without loading its contents."""

    return classify_path(path, root=root, operation="delete")


def assess_changes(
    changes: Sequence[Tuple[Path, str, str]],
    *,
    root: Optional[Path] = None,
    operation: str = "edit",
) -> RiskAssessment:
    """Aggregate risk for one or more concrete edits."""

    root = (root or project_root()).resolve()
    level = "low"
    reasons: List[str] = []
    multiple = len(changes) > 1
    for path, old, new in changes:
        assessment = assess_edit(path, old, new, root=root, operation=operation)
        level = _raise_level(level, assessment.level)
        try:
            label = str(path.resolve().relative_to(root)).replace("\\", "/")
        except (OSError, ValueError):
            label = str(path)
        if multiple:
            reasons.extend(
                f"{label}: {reason}" for reason in assessment.reasons
            )
        else:
            reasons.extend(assessment.reasons)
    final_level, final_reasons = _merge(level, reasons)
    return RiskAssessment(level=final_level, reasons=tuple(final_reasons))
