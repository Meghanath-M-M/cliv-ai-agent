# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "groq",
#     "pydantic",
#     "python-dotenv",
# ]
# ///

import os
import sys
import json
from typing import List, Dict, Any
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]  # Groq/OpenAI format expects 'parameters'


class AIAgent:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.messages: List[Dict[str, Any]] = []
        self.tools: List[Tool] = []
        self._setup_tools()

    def _setup_tools(self):
        self.tools = [
            Tool(
                name="read_file",
                description="Read the contents of a file at the specified path",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="list_files",
                description="List all files and directories in the specified path",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path to list (defaults to current directory)",
                        }
                    },
                    "required": [],
                },
            ),
            Tool(
                name="edit_file",
                description="Edit a file by replacing old_text with new_text. Creates the file if it doesn't exist.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to edit",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "The text to search for and replace (leave empty to create new file)",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "The text to replace old_text with",
                        },
                    },
                    "required": ["path", "new_text"],
                },
            ),
        ]

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        try:
            if tool_name == "read_file":
                return self._read_file(tool_input["path"])
            elif tool_name == "list_files":
                return self._list_files(tool_input.get("path", "."))
            elif tool_name == "edit_file":
                return self._edit_file(
                    tool_input["path"],
                    tool_input.get("old_text", ""),
                    tool_input["new_text"],
                )
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _read_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"File contents of {path}:\n{content}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _list_files(self, path: str) -> str:
        try:
            if not os.path.exists(path):
                return f"Path not found: {path}"

            items = []
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    items.append(f"[DIR]  {item}/")
                else:
                    items.append(f"[FILE] {item}")

            if not items:
                return f"Empty directory: {path}"

            return f"Contents of {path}:\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            if os.path.exists(path) and old_text:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                if old_text not in content:
                    return f"Text not found in file: {old_text}"

                content = content.replace(old_text, new_text)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

                return f"Successfully edited {path}"
            else:
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)

                return f"Successfully created {path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        # Map schemas into OpenAI/Groq function syntax
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]

        while True:
            try:
                # Swapped to the correct chat completions router
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=4096,
                    messages=self.messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                # Crucial step: append the model's call payload directly back into history
                self.messages.append(response_message)

                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        # Parse out the arguments from their returned serialized string state
                        tool_input = json.loads(tool_call.function.arguments)

                        result = self._execute_tool(tool_name, tool_input)

                        # Deliver tool resolution contexts back to Groq under 'tool' role identifiers
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": result,
                            }
                        )
                    # Re-loop to send execution logs back up for final phrasing evaluation
                    continue
                else:
                    return response_message.content if response_message.content else ""

            except Exception as e:
                return f"Error: {str(e)}"


if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        sys.exit(1)

    agent = AIAgent(api_key)
    # Executing localized context test run
    response = agent.chat("What files are in the current directory?")
    print(response)
