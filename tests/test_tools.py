"""Tests for cliv tools and the project-root safety policy."""

import pytest

from cliv.tools.edit_file import EditFileTool
from cliv.tools.list_files import ListFilesTool
from cliv.tools.multi_edit import MultiEditTool
from cliv.tools.read_file import ReadFileTool
from cliv.tools.remove_file import RemoveFileTool
from cliv.tools.run_command import RunCommandTool
from cliv.tools.run_test import RunTestsTool


class TestReadFile:
    def test_reads_existing_file(self, project_dir):
        (project_dir / "test.txt").write_text("hello world")
        result = ReadFileTool().execute(path="test.txt")
        assert result == "hello world"

    def test_file_not_found(self, project_dir):
        result = ReadFileTool().execute(path="missing.txt")
        assert "not found" in result

    def test_truncates_large_files(self, project_dir):
        (project_dir / "big.txt").write_text("x" * 200_000)
        result = ReadFileTool().execute(path="big.txt")
        assert "truncated" in result
        assert len(result) < 150_000

    def test_blocks_sensitive_env(self, project_dir):
        (project_dir / ".env").write_text("GROQ_API_KEY=secret")
        result = ReadFileTool().execute(path=".env")
        assert "blocked" in result
        assert "secret" not in result

    def test_blocks_path_outside_project(self, project_dir):
        outside = project_dir.parent / "outside.txt"
        outside.write_text("nope")
        result = ReadFileTool().execute(path=str(outside))
        assert "blocked" in result


class TestListFiles:
    def test_lists_directory(self, project_dir):
        (project_dir / "a.txt").write_text("a")
        (project_dir / "b").mkdir()
        result = ListFilesTool().execute(path=".")
        assert "[FILE] a.txt" in result
        assert "[DIR] b" in result

    def test_empty_directory(self, project_dir):
        result = ListFilesTool().execute(path=".")
        assert result == "(empty directory)"

    def test_not_found(self, project_dir):
        result = ListFilesTool().execute(path="missing-dir")
        assert "not found" in result

    def test_hides_sensitive_files(self, project_dir):
        (project_dir / ".env").write_text("secret")
        (project_dir / "visible.txt").write_text("ok")
        result = ListFilesTool().execute(path=".")
        assert ".env" not in result
        assert "visible.txt" in result


class TestEditFile:
    def test_overwrite_file(self, project_dir):
        tool = EditFileTool()
        result = tool.execute(path="out.txt", new_string="new content")
        assert "Successfully wrote" in result
        assert (project_dir / "out.txt").read_text() == "new content"

    def test_patch_file(self, project_dir):
        path = project_dir / "patch.txt"
        path.write_text("old line\nsecond line")
        result = EditFileTool().execute(
            path="patch.txt", old_string="old line", new_string="new line"
        )
        assert "Successfully patched" in result
        assert "new line" in path.read_text()

    def test_patch_old_string_not_found(self, project_dir):
        path = project_dir / "patch.txt"
        path.write_text("content")
        result = EditFileTool().execute(
            path="patch.txt", old_string="missing", new_string="new"
        )
        assert "not found" in result
        assert path.read_text() == "content"

    def test_blocks_outside_allowed_dirs(self, project_dir):
        outside = project_dir.parent / "evil.txt"
        result = EditFileTool().execute(path=str(outside), new_string="hack")
        assert "blocked" in result
        assert not outside.exists()

    def test_blocks_sensitive_file(self, project_dir):
        result = EditFileTool().execute(path=".env", new_string="hack")
        assert "blocked" in result

    def test_preview_is_a_diff(self, project_dir):
        (project_dir / "app.py").write_text("print('old')\n")
        preview = EditFileTool().preview(
            {"path": "app.py", "old_string": "old", "new_string": "new"}
        )
        assert "-print('old')" in preview
        assert "+print('new')" in preview


class TestRemoveFile:
    def test_removes_existing_file(self, project_dir):
        path = project_dir / "delete_me.txt"
        path.write_text("bye")
        result = RemoveFileTool().execute(path="delete_me.txt")
        assert "Successfully removed" in result
        assert not path.exists()

    def test_file_not_found(self, project_dir):
        result = RemoveFileTool().execute(path="nope.txt")
        assert "not found" in result

    def test_blocks_outside_allowed_dirs(self, project_dir):
        result = RemoveFileTool().execute(path="/etc/passwd")
        assert "blocked" in result

    def test_blocks_git_directory(self, project_dir):
        (project_dir / ".git").mkdir()
        (project_dir / ".git" / "config").write_text("secret")
        result = RemoveFileTool().execute(path=".git/config")
        assert "blocked" in result


class TestMultiEdit:
    def test_applies_edits_to_multiple_files(self, project_dir):
        (project_dir / "a.txt").write_text("one")
        (project_dir / "b.txt").write_text("two")
        result = MultiEditTool().execute(
            edits=[
                {"path": "a.txt", "old_string": "one", "new_string": "ONE"},
                {"path": "b.txt", "old_string": "two", "new_string": "TWO"},
            ]
        )
        assert "Successfully applied 2" in result
        assert (project_dir / "a.txt").read_text() == "ONE"
        assert (project_dir / "b.txt").read_text() == "TWO"

    def test_preview_shows_combined_diff(self, project_dir):
        (project_dir / "a.txt").write_text("old")
        preview = MultiEditTool().preview(
            {"edits": [{"path": "a.txt", "old_string": "old", "new_string": "new"}]}
        )
        assert "-old" in preview
        assert "+new" in preview

    def test_rejects_missing_old_string_without_writing(self, project_dir):
        (project_dir / "a.txt").write_text("old")
        result = MultiEditTool().execute(
            edits=[{"path": "a.txt", "old_string": "missing", "new_string": "new"}]
        )
        assert "not found" in result
        assert (project_dir / "a.txt").read_text() == "old"


class TestRunCommand:
    def test_runs_allowlisted_command(self, project_dir):
        result = RunCommandTool().execute(
            command="python", args=["-c", "print('hello from command')"]
        )
        assert "hello from command" in result
        assert "Exit code: 0" in result

    def test_blocks_disallowed_command(self, project_dir):
        result = RunCommandTool().execute(command="rm", args=["-rf", "."])
        assert "not allowlisted" in result

    def test_blocks_path_as_executable(self, project_dir):
        result = RunCommandTool().execute(command="./script.sh")
        assert "bare executable" in result

    def test_blocks_cwd_outside_project(self, project_dir):
        result = RunCommandTool().execute(
            command="python", args=["-c", "print(1)"], cwd=str(project_dir.parent)
        )
        assert "blocked" in result


class TestApprovalMetadata:
    @pytest.mark.parametrize(
        "tool",
        [EditFileTool(), MultiEditTool(), RemoveFileTool(), RunCommandTool(), RunTestsTool()],
    )
    def test_mutating_or_code_running_tools_require_approval(self, tool):
        assert tool.requires_approval is True

    @pytest.mark.parametrize("tool", [ReadFileTool(), ListFilesTool()])
    def test_read_only_tools_do_not_require_approval(self, tool):
        assert tool.requires_approval is False


class TestRunTests:
    def test_preview_describes_command(self, project_dir):
        preview = RunTestsTool().preview({"target": ""})
        assert "pytest" in preview

    def test_rejects_test_target_outside_project(self, project_dir):
        result = RunTestsTool().execute(target=str(project_dir.parent))
        assert "blocked" in result
