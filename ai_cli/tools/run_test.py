import subprocess
from .base import BaseTool


class RunTestsTool(BaseTool):
    def __init__(self):
        self.name = "run_tests"
        self.description = "Execute the pytest test suite and return the results, including any error traces."
        self.input_schema = {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Specific test file or directory to run (leave empty to run all tests)",
                }
            },
            "required": [],
        }

    def execute(self, target: str = "", **kwargs) -> str:
        try:
            command = ["pytest"]
            if target:
                command.append(target)

            # We don't use check=True here because pytest returns a non-zero exit code if tests fail,
            # and we actually WANT to capture and return that failure text to the AI!
            result = subprocess.run(command, capture_output=True, text=True)

            output = result.stdout.strip()
            if result.stderr:
                output += f"\nErrors:\n{result.stderr.strip()}"

            return f"Pytest Results:\n{output}"

        except FileNotFoundError:
            return "Error: pytest is not installed. Please install it using `uv add pytest` or `pip install pytest`."
        except Exception as e:
            return f"Unexpected error running pytest: {str(e)}"
