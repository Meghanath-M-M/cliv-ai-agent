"""Regression tests for the project-root path policy."""

from pathlib import Path

import pytest

from cliv.security import (
    is_sensitive_path,
    is_within_root,
    project_root,
    resolve_project_path,
)


def test_prefix_sibling_is_not_treated_as_inside(project_dir):
    evil = project_dir.parent / f"{project_dir.name}-evil"
    evil.mkdir()
    target = evil / "file.txt"
    target.write_text("nope")

    resolved, error = resolve_project_path(str(target), root=project_dir)

    assert resolved is None
    assert "blocked" in error


def test_relative_path_inside_project(project_dir):
    target = project_dir / "nested" / "file.txt"
    target.parent.mkdir()
    target.write_text("ok")

    resolved, error = resolve_project_path("nested/file.txt", root=project_dir)

    assert error is None
    assert resolved == target.resolve()


def test_path_outside_project_is_blocked(project_dir):
    outside = project_dir.parent / "outside.txt"
    resolved, error = resolve_project_path(str(outside), root=project_dir)
    assert resolved is None
    assert "outside" in error


def test_env_is_sensitive_but_example_is_allowed(project_dir):
    assert is_sensitive_path(project_dir / ".env")
    assert is_sensitive_path(project_dir / ".env.local")
    assert not is_sensitive_path(project_dir / ".env.example")
    assert is_sensitive_path(project_dir / ".ssh" / "id_rsa")
    assert is_sensitive_path(project_dir / "server.pem")
    assert is_sensitive_path(
        Path.home() / ".config" / "cliv" / "config.json"
    )


def test_git_writes_are_blocked(project_dir):
    (project_dir / ".git").mkdir()
    resolved, error = resolve_project_path(".git/config", for_write=True, root=project_dir)
    assert resolved is None
    assert "protected" in error


def test_project_root_respects_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIV_PROJECT_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_is_within_root(project_dir):
    assert is_within_root(project_dir / "x.txt", project_dir)
    assert not is_within_root(project_dir.parent / "x.txt", project_dir)


def test_symlink_escape_is_blocked(project_dir):
    outside = project_dir.parent / "outside-secret.txt"
    outside.write_text("secret")
    link = project_dir / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform")

    resolved, error = resolve_project_path("link.txt", root=project_dir)
    assert resolved is None
    assert "outside" in error
