"""Project-scoped safety snapshots and time-travel undo.

A snapshot is captured immediately before an approved write.  It stores the
original bytes of every affected file outside the repository, plus a small
JSON manifest.  Undo verifies that the files still match the state produced
by the original operation, so it refuses to silently overwrite later edits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cliv.config import SNAPSHOT_DIR
from cliv.security import project_root, resolve_project_path

SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
MAX_SUMMARY_CHARS = 400


class SnapshotError(Exception):
    """Raised when a snapshot cannot be created or restored safely."""


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    existed: bool
    mode: Optional[int] = None
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None
    backup: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SnapshotFile":
        return cls(
            path=str(value.get("path", "")),
            existed=bool(value.get("existed", False)),
            mode=(
                int(value["mode"])
                if value.get("mode") is not None
                else None
            ),
            before_hash=value.get("before_hash"),
            after_hash=value.get("after_hash"),
            backup=value.get("backup"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "existed": self.existed,
            "mode": self.mode,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "backup": self.backup,
        }


@dataclass(frozen=True)
class Snapshot:
    id: str
    created_at: str
    project_root: str
    session_id: str
    tool: str
    summary: str
    status: str
    files: Tuple[SnapshotFile, ...] = ()
    rolled_back_at: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Snapshot":
        raw_files = value.get("files", [])
        files = tuple(
            SnapshotFile.from_dict(item)
            for item in raw_files
            if isinstance(item, dict)
        )
        return cls(
            id=str(value.get("id", "")),
            created_at=str(value.get("created_at", "")),
            project_root=str(value.get("project_root", "")),
            session_id=str(value.get("session_id", "")),
            tool=str(value.get("tool", "")),
            summary=str(value.get("summary", "")),
            status=str(value.get("status", "captured")),
            files=files,
            rolled_back_at=value.get("rolled_back_at"),
        )

    @property
    def file_count(self) -> int:
        return len(self.files)

    def render(self) -> str:
        lines = [
            f"Snapshot: {self.id}",
            f"Created:  {self.created_at}",
            f"Tool:     {self.tool}",
            f"Status:   {self.status}",
            f"Project:  {self.project_root}",
        ]
        if self.summary:
            lines.append(f"Summary:  {self.summary}")
        if self.rolled_back_at:
            lines.append(f"Undone:   {self.rolled_back_at}")
        lines.append("Files:")
        for entry in self.files:
            marker = "M" if entry.existed else "A"
            lines.append(f"  {marker} {entry.path}")
        return "\n".join(lines)


@dataclass(frozen=True)
class RollbackResult:
    success: bool
    message: str
    snapshot: Optional[Snapshot] = None
    conflicts: Tuple[str, ...] = ()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return _hash_bytes(path.read_bytes())
    except OSError as exc:
        raise SnapshotError(f"Could not read {path}: {exc}") from exc


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        os.replace(temp, path)
    finally:
        try:
            if temp.exists():
                temp.unlink()
        except OSError:
            pass


def _relative_label(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


class SnapshotStore:
    """Capture and restore pre-write file state for one project."""

    def __init__(self, path: Optional[Path] = None):
        configured = os.environ.get("CLIV_SNAPSHOT_DIR")
        if path is not None:
            self.path = Path(path).expanduser()
        elif configured:
            self.path = Path(configured).expanduser()
        else:
            self.path = SNAPSHOT_DIR
        self.path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path, 0o700)
        except OSError:
            pass

    def _project_dir(self, root: Path) -> Path:
        key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        directory = self.path / key
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        return directory

    def _snapshot_dir(self, root: Path, snapshot_id: str) -> Path:
        return self._project_dir(root) / "snapshots" / snapshot_id

    def _manifest_path(self, root: Path, snapshot_id: str) -> Path:
        return self._snapshot_dir(root, snapshot_id) / "manifest.json"

    def _read_manifest(
        self, snapshot_id: str, *, project: Optional[Path] = None
    ) -> Tuple[Dict[str, Any], Path, Path]:
        if not SNAPSHOT_ID_RE.fullmatch(snapshot_id):
            raise SnapshotError(f"Invalid snapshot id: {snapshot_id}")
        root = (project or project_root()).resolve()
        snapshot_dir = self._snapshot_dir(root, snapshot_id)
        manifest_path = self._manifest_path(root, snapshot_id)
        if not manifest_path.exists():
            raise SnapshotError(f"Snapshot not found: {snapshot_id}")
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"Could not read snapshot {snapshot_id}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SnapshotError(f"Invalid snapshot manifest: {snapshot_id}")
        return payload, snapshot_dir, root

    def _load_snapshot(
        self, snapshot_id: str, *, project: Optional[Path] = None
    ) -> Tuple[Snapshot, Path, Path]:
        payload, snapshot_dir, root = self._read_manifest(
            snapshot_id, project=project
        )
        snapshot = Snapshot.from_dict(payload)
        if snapshot.id != snapshot_id:
            raise SnapshotError(f"Snapshot id mismatch for {snapshot_id}")
        if snapshot.project_root != str(root):
            raise SnapshotError(f"Snapshot project mismatch for {snapshot_id}")
        return snapshot, snapshot_dir, root

    def capture(
        self,
        paths: Sequence[Path],
        *,
        tool: str,
        summary: str = "",
        project: Optional[Path] = None,
        session_id: str = "",
    ) -> Snapshot:
        """Capture the current bytes of *paths* before a write."""

        root = (project or project_root()).resolve()
        unique: List[Path] = []
        seen: set[Path] = set()
        for path in paths:
            resolved, error = resolve_project_path(
                str(path), for_write=True, root=root
            )
            if error:
                raise SnapshotError(error)
            assert resolved is not None
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)

        if not unique:
            raise SnapshotError("No files were selected for the snapshot.")

        snapshot_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        snapshot_dir = self._snapshot_dir(root, snapshot_id)
        (snapshot_dir / "files").mkdir(parents=True, exist_ok=False)
        try:
            entries: List[SnapshotFile] = []
            for index, path in enumerate(unique):
                existed = path.exists()
                if existed and not path.is_file():
                    raise SnapshotError(f"Not a regular file: {path}")
                data = path.read_bytes() if existed else None
                backup_name: Optional[str] = None
                mode: Optional[int] = None
                before_hash: Optional[str] = None
                if data is not None:
                    before_hash = _hash_bytes(data)
                    backup_name = f"files/{index}.bin"
                    backup_path = snapshot_dir / backup_name
                    backup_path.write_bytes(data)
                    try:
                        os.chmod(backup_path, 0o600)
                    except OSError:
                        pass
                    try:
                        mode = stat.S_IMODE(path.stat().st_mode)
                    except OSError:
                        mode = None
                entries.append(
                    SnapshotFile(
                        path=_relative_label(path, root),
                        existed=existed,
                        mode=mode,
                        before_hash=before_hash,
                        backup=backup_name,
                    )
                )

            payload = {
                "id": snapshot_id,
                "created_at": _now(),
                "project_root": str(root),
                "session_id": str(session_id)[:64],
                "tool": str(tool)[:64],
                "summary": " ".join(str(summary).split())[:MAX_SUMMARY_CHARS],
                "status": "captured",
                "files": [entry.to_dict() for entry in entries],
            }
            _atomic_write_json(snapshot_dir / "manifest.json", payload)
            return Snapshot.from_dict(payload)
        except Exception:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            raise

    def mark_applied(
        self, snapshot_id: str, *, project: Optional[Path] = None
    ) -> Snapshot:
        """Record the post-write state so undo can detect later edits."""

        payload, snapshot_dir, root = self._read_manifest(
            snapshot_id, project=project
        )
        files = payload.get("files", [])
        for entry in files:
            if not isinstance(entry, dict):
                continue
            path, error = resolve_project_path(
                str(entry.get("path", "")), for_write=True, root=root
            )
            if error:
                raise SnapshotError(error)
            assert path is not None
            entry["after_hash"] = _hash_file(path)
        payload["status"] = "applied"
        payload["applied_at"] = _now()
        _atomic_write_json(snapshot_dir / "manifest.json", payload)
        return Snapshot.from_dict(payload)

    def mark_failed(
        self, snapshot_id: str, *, project: Optional[Path] = None
    ) -> Snapshot:
        payload, snapshot_dir, _ = self._read_manifest(
            snapshot_id, project=project
        )
        payload["status"] = "failed"
        payload["failed_at"] = _now()
        _atomic_write_json(snapshot_dir / "manifest.json", payload)
        return Snapshot.from_dict(payload)

    def get(
        self, snapshot_id: str, *, project: Optional[Path] = None
    ) -> Snapshot:
        snapshot, _, _ = self._load_snapshot(snapshot_id, project=project)
        return snapshot

    def recent(
        self,
        limit: int = 20,
        *,
        project: Optional[Path] = None,
        status: Optional[str] = None,
    ) -> List[Snapshot]:
        root = (project or project_root()).resolve()
        snapshots_dir = self._project_dir(root) / "snapshots"
        if not snapshots_dir.exists():
            return []
        found: List[Snapshot] = []
        for directory in snapshots_dir.iterdir():
            if not directory.is_dir():
                continue
            manifest = directory / "manifest.json"
            if not manifest.exists():
                continue
            try:
                with open(manifest, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                snapshot = Snapshot.from_dict(payload)
            except (OSError, json.JSONDecodeError):
                continue
            if snapshot.project_root != str(root):
                continue
            if status is not None and snapshot.status != status:
                continue
            found.append(snapshot)
        found.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return found[: max(1, min(int(limit), 100))]

    def latest_applied(self, *, project: Optional[Path] = None) -> Optional[Snapshot]:
        snapshots = self.recent(1, project=project, status="applied")
        return snapshots[0] if snapshots else None

    def rollback(
        self,
        snapshot_id: str,
        *,
        project: Optional[Path] = None,
        force: bool = False,
    ) -> RollbackResult:
        """Restore a snapshot, refusing to clobber later edits by default."""

        try:
            snapshot, snapshot_dir, root = self._load_snapshot(
                snapshot_id, project=project
            )
        except SnapshotError as exc:
            return RollbackResult(False, str(exc))

        if snapshot.status == "rolled_back":
            return RollbackResult(False, "Snapshot is already rolled back.", snapshot)
        if snapshot.status != "applied" and not force:
            return RollbackResult(
                False,
                f"Snapshot is {snapshot.status}; only applied snapshots can be undone.",
                snapshot,
            )

        conflicts: List[str] = []
        prepared: List[Tuple[Path, SnapshotFile]] = []
        for entry in snapshot.files:
            path, error = resolve_project_path(
                entry.path, for_write=True, root=root
            )
            if error:
                conflicts.append(f"{entry.path}: {error}")
                continue
            assert path is not None
            prepared.append((path, entry))
            try:
                current_hash = _hash_file(path)
            except SnapshotError as exc:
                conflicts.append(f"{entry.path}: {exc}")
                continue
            if current_hash != entry.after_hash:
                conflicts.append(
                    f"{entry.path}: changed since this snapshot "
                    "(expected the post-edit state)"
                )

        if conflicts and not force:
            return RollbackResult(
                False,
                "Undo stopped because the files changed after this snapshot.",
                snapshot,
                tuple(conflicts),
            )

        try:
            for path, entry in prepared:
                if entry.existed:
                    if not entry.backup:
                        raise SnapshotError(
                            f"Missing backup data for {entry.path}"
                        )
                    backup = snapshot_dir / entry.backup
                    if not backup.exists():
                        raise SnapshotError(
                            f"Missing backup file for {entry.path}"
                        )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(backup.read_bytes())
                    if entry.mode is not None:
                        try:
                            os.chmod(path, entry.mode)
                        except OSError:
                            pass
                elif path.exists():
                    path.unlink()
        except (OSError, SnapshotError) as exc:
            return RollbackResult(
                False,
                f"Undo failed while restoring files: {exc}",
                snapshot,
            )

        payload, _, _ = self._read_manifest(snapshot_id, project=root)
        payload["status"] = "rolled_back"
        payload["rolled_back_at"] = _now()
        _atomic_write_json(snapshot_dir / "manifest.json", payload)
        restored = Snapshot.from_dict(payload)
        return RollbackResult(
            True,
            f"Restored {len(snapshot.files)} file(s) from snapshot {snapshot.id}.",
            restored,
        )


def render_snapshot(snapshot: Snapshot) -> str:
    return snapshot.render()


def render_snapshots(
    snapshots: Iterable[Snapshot], *, project: Optional[Path] = None
) -> str:
    root = (project or project_root()).resolve()
    lines = ["CLIV SAFETY SNAPSHOTS", "", f"Project: {root}"]
    items = list(snapshots)
    if not items:
        lines.append("No snapshots recorded yet.")
        return "\n".join(lines)
    lines.append("")
    lines.append(
        f"{'ID':<28} {'CREATED (UTC)':<20} {'STATUS':<12} {'TOOL':<14} FILES"
    )
    for snapshot in items:
        lines.append(
            f"{snapshot.id:<28} "
            f"{snapshot.created_at.replace('T', ' ')[:19]:<20} "
            f"{snapshot.status:<12} "
            f"{snapshot.tool:<14} "
            f"{snapshot.file_count}"
        )
    return "\n".join(lines)
