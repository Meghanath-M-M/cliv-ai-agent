import subprocess
from .base import BaseTool


class GitDiffTool(BaseTool):
    def __init__(self):
        self.name = "git_diff"
        self.description = "Get the git status and git diff of the current repository to see what code has changed."
        self.input_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, **kwargs) -> str:
        try:
            # Get the current git status
            status_result = subprocess.run(
                ["git", "status", "-s"], capture_output=True, text=True, check=True
            )

            # Get the actual code changes
            diff_result = subprocess.run(
                ["git", "diff"], capture_output=True, text=True, check=True
            )

            status_output = status_result.stdout.strip()
            diff_output = diff_result.stdout.strip()

            if not status_output and not diff_output:
                return "No changes found. The working tree is clean."

            return f"--- GIT STATUS ---\n{status_output}\n\n--- GIT DIFF ---\n{diff_output}"

        except FileNotFoundError:
            return "Error: Git is not installed or not found in the system path."
        except subprocess.CalledProcessError as e:
            return f"Error executing git command. Is this a git repository?\nDetails: {e.stderr}"
        except Exception as e:
            return f"Unexpected error running git: {str(e)}"
