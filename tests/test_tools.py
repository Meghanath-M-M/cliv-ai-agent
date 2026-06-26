"""Tests for cliv tools."""

import pytest
from pathlib import Path
from cliv.tools.read_file import ReadFileTool
from cliv.tools.list_files import ListFilesTool
from cliv.tools.edit_file import EditFileTool
from cliv.tools.remove_file import RemoveFileTool


class TestReadFile:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        tool = ReadFileTool()
        result = tool.execute(path=str(f))
        assert result == "hello world"

    def test_file_not_found(self):
        tool = ReadFileTool()
        result = tool.execute(path="/nonexistent/file.txt")
        assert "not found" in result

    def test_truncates_large_files(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 200_000)
        tool = ReadFileTool()
        result = tool.execute(path=str(f))
        assert "truncated" in result
        assert len(result) < 150_000


class TestListFiles:
    def test_lists_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b").mkdir()
        tool = ListFilesTool()
        result = tool.execute(path=str(tmp_path))
        assert "[FILE] a.txt" in result
        assert "[DIR] b" in result

    def test_empty_directory(self, tmp_path):
        tool = ListFilesTool()
        result = tool.execute(path=str(tmp_path))
        assert result == "(empty directory)"

    def test_not_found(self):
        tool = ListFilesTool()
        result = tool.execute(path="/nonexistent")
        assert "not found" in result


class TestEditFile:
    def test_overwrite_file(self, tmp_path):
        f = tmp_path / "out.txt"
        tool = EditFileTool()
        result = tool.execute(path=str(f), new_string="new content")
        assert "Successfully wrote" in result
        assert f.read_text() == "new content"

    def test_patch_file(self, tmp_path):
        f = tmp_path / "patch.txt"
        f.write_text("old line\nsecond line")
        tool = EditFileTool()
        result = tool.execute(path=str(f), old_string="old line", new_string="new line")
        assert "Successfully patched" in result
        assert "new line" in f.read_text()

    def test_patch_old_string_not_found(self, tmp_path):
        f = tmp_path / "patch.txt"
        f.write_text("content")
        tool = EditFileTool()
        result = tool.execute(path=str(f), old_string="missing", new_string="new")
        assert "not found" in result
        assert f.read_text() == "content"

    def test_blocks_outside_allowed_dirs(self, tmp_path):
        tool = EditFileTool()
        result = tool.execute(path="/etc/passwd", new_string="hack")
        assert "blocked" in result


class TestRemoveFile:
    def test_removes_existing_file(self, tmp_path):
        f = tmp_path / "delete_me.txt"
        f.write_text("bye")
        tool = RemoveFileTool()
        result = tool.execute(path=str(f))
        assert "Successfully removed" in result
        assert not f.exists()

    def test_file_not_found(self, tmp_path):
        tool = RemoveFileTool()
        result = tool.execute(path=str(tmp_path / "nope.txt"))
        assert "not found" in result

    def test_blocks_outside_allowed_dirs(self):
        tool = RemoveFileTool()
        result = tool.execute(path="/etc/passwd")
        assert "blocked" in result
