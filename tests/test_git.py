"""Tests for safe, first-class git integration (cliv/git.py)."""

import os
import subprocess

import pytest

from cliv.git import (
    GitError,
    add,
    branch_exists,
    commit,
    create_branch,
    current_branch,
    ensure_branch,
    is_git_repo,
    status_short,
    switch_branch,
)


def _init_repo(path) -> None:
    """Initialize a git repo with a default user so commits work."""

    subprocess.run(["git", "init"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "cliv test"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "cliv@test.invalid"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("# hello\n")
    subprocess.run(
        ["git", "add", "README.md"], cwd=str(path), check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=str(path), check=True, capture_output=True
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_repo(repo_dir)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "cliv test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "cliv@test.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "cliv test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "cliv@test.invalid")
    return repo_dir


def test_is_git_repo(repo):
    assert is_git_repo(cwd=str(repo)) is True
    assert is_git_repo(cwd=str(repo.parent)) is False


def test_current_branch(repo):
    branch = current_branch(cwd=str(repo))
    assert branch in {"master", "main"}


def test_create_and_switch_branch(repo):
    assert create_branch("feature-x", cwd=str(repo)) is None
    assert branch_exists("feature-x", cwd=str(repo)) is True
    assert current_branch(cwd=str(repo)) == "feature-x"

    for candidate in ("master", "main"):
        if branch_exists(candidate, cwd=str(repo)):
            switch_branch(candidate, cwd=str(repo))
            break
    assert current_branch(cwd=str(repo)) in {"master", "main"}


def test_ensure_branch_creates_and_switches(repo):
    created = ensure_branch("feature-y", cwd=str(repo))
    assert created is True
    assert current_branch(cwd=str(repo)) == "feature-y"

    created_again = ensure_branch("feature-y", cwd=str(repo))
    assert created_again is False


def test_create_branch_duplicate_raises(repo):
    create_branch("feature-z", cwd=str(repo))
    with pytest.raises(GitError):
        create_branch("feature-z", cwd=str(repo))


def test_status_short_and_commit(repo):
    assert status_short(cwd=str(repo)) == ""
    (repo / "new.txt").write_text("new file\n")
    cid = commit("add new file", ["new.txt"], cwd=str(repo))
    assert cid
    assert status_short(cwd=str(repo)) == ""
    assert current_branch(cwd=str(repo)) in {"master", "main"}


def test_add_and_commit_only_staged_paths(repo):
    (repo / "a.txt").write_text("a\n")
    (repo / "b.txt").write_text("b\n")
    add(["a.txt"], cwd=str(repo))
    cid = commit("only a", ["a.txt"], cwd=str(repo))
    assert cid
    # b.txt remains untracked / not committed
    status = status_short(cwd=str(repo))
    assert "b.txt" in status


def test_git_not_a_repo(tmp_path):
    assert is_git_repo(cwd=str(tmp_path)) is False
    assert current_branch(cwd=str(tmp_path)) is None
    assert branch_exists("anything", cwd=str(tmp_path)) is False
    with pytest.raises(GitError):
        switch_branch("anything", cwd=str(tmp_path))
