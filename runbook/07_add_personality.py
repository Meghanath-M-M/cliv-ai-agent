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
import re
import argparse
import logging
import time
from typing import List, Dict, Any
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("agent.log")],
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_RETRIES = 5  # cap for hallucination / syntax-error retries
MAX_RATE_LIMIT_RETRIES = 10  # cap for 429 retries


class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class AIAgent:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.messages: List[Dict[str, Any]] = []
        self.tools: List[Tool] = []
        self._setup_tools()

    def _setup_tools(self):
        self.tools = [
            Tool(
                name="list_files",
                description="List all files and directories in a path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list. Use '.' for the current directory.",
                        }
                    },
                    "required": [],
                },
            ),
            Tool(
                name="read_file",
                description="Read the contents of a file at the specified path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read.",
                        }
                    },
                    "required": ["path"],
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
                            "description": "The path to the file to edit.",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "The text to find and replace. Leave empty to create a new file.",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "The replacement text.",
                        },
                    },
                    "required": ["path", "new_text"],
                },
            ),
        ]

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        try:
            if tool_name == "list_files":
                return self._list_files(tool_input.get("path", "."))
            elif tool_name == "read_file":
                return self._read_file(tool_input["path"])
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

    def _list_files(self, path: str) -> str:
        try:
            if not os.path.exists(path):
                return f"Path not found: {path}"
            items = []
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                items.append(
                    f"[DIR]  {item}/" if os.path.isdir(item_path) else f"[FILE] {item}"
                )
            return (
                f"Contents of {path}:\n" + "\n".join(items)
                if items
                else f"Empty directory: {path}"
            )
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def _read_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f"File contents of {path}:\n{f.read()}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            if os.path.exists(path) and old_text:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                if old_text not in content:
                    return f"Text not found in file: {old_text}"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content.replace(old_text, new_text))
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

    def _prune_history(self):
        """Drop old messages to stay under TPM limit.

        Only prunes at a 'user' turn not immediately followed by a 'tool'
        message, so we never break a tool-call chain the API requires intact.
        """
        if len(self.messages) <= 6:
            return
        safe_index = len(self.messages) - 6
        while safe_index < len(self.messages):
            msg = self.messages[safe_index]
            if msg.get("role") == "user":
                next_msg = (
                    self.messages[safe_index + 1]
                    if safe_index + 1 < len(self.messages)
                    else None
                )
                if next_msg is None or next_msg.get("role") != "tool":
                    break
            safe_index += 1
        if safe_index < len(self.messages):
            self.messages = [self.messages[0]] + self.messages[safe_index:]
            logging.info("Auto-pruned conversation history.")

    def _is_hallucinated_tool_call(self, content: str, allowed: set) -> bool:
        """Return True if the model printed a tool call as raw text."""
        # XML-style: <function=list_files> or truncated function=list_files>
        if "function=" in content and any(name in content for name in allowed):
            return True
        # Raw tool name at the start of the response
        if any(content.startswith(name) for name in allowed):
            return True
        # Raw JSON blob: [{"name": ...}] or {"name": ..., "parameters": ...}
        if content.startswith("[") or content.startswith("{"):
            if any(name in content for name in allowed):
                return True
            if '"parameters"' in content and '"name"' in content:
                return True
        return False

    def chat(self, user_input: str) -> str:
        if not self.messages:
            self.messages.append(
                {
                    "role": "system",
                    "content": (
                        "You are a file assistant. Use only these tools via the native tool_calls API: "
                        "list_files, read_file, edit_file. "
                        "Never output raw XML, JSON, or text function calls. "
                        "Always pass arguments as a JSON dict. "
                        "For the current directory use path '.'."
                    ),
                }
            )

        self.messages.append({"role": "user", "content": user_input})
        self._prune_history()

        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.tools
        ]
        allowed_tool_names = {t.name for t in self.tools}

        model_retry_count = 0  # hallucination / syntax errors
        rate_limit_retries = 0  # 429 rate limit errors

        while True:
            if model_retry_count >= MAX_RETRIES:
                logging.error(f"Exceeded max model retries ({MAX_RETRIES}).")
                return "Sorry, I was unable to complete this request after several attempts. Please try rephrasing."

            try:
                response = self.client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",  # 30K TPM on free tier
                    max_tokens=1024,
                    messages=self.messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                msg_dict: Dict[str, Any] = {"role": "assistant"}
                if response_message.content:
                    msg_dict["content"] = response_message.content

                # --- Hallucination catcher ---
                if not tool_calls and response_message.content:
                    content_str = response_message.content.strip()
                    if self._is_hallucinated_tool_call(content_str, allowed_tool_names):
                        logging.warning(
                            f"Hallucination intercepted (attempt {model_retry_count + 1}): {content_str}"
                        )
                        self.messages.append(msg_dict)
                        self.messages.append(
                            {
                                "role": "user",
                                "content": "System Error: You printed a function call as raw text. Use the native tool_calls API instead. Try again.",
                            }
                        )
                        model_retry_count += 1
                        time.sleep(1.5)
                        continue

                # --- Process valid tool calls ---
                valid_tool_calls = []
                if tool_calls:
                    msg_dict["tool_calls"] = []
                    for tc in tool_calls:
                        if tc.function.name in allowed_tool_names:
                            msg_dict["tool_calls"].append(
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                            )
                            valid_tool_calls.append(tc)
                    if not msg_dict["tool_calls"]:
                        del msg_dict["tool_calls"]

                self.messages.append(msg_dict)

                if valid_tool_calls:
                    model_retry_count = 0
                    rate_limit_retries = 0
                    if response_message.content:
                        print(response_message.content)
                    for tc in valid_tool_calls:
                        tool_input = json.loads(tc.function.arguments)
                        logging.info(f"Executing {tc.function.name} with {tool_input}")
                        result = self._execute_tool(tc.function.name, tool_input)
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tc.function.name,
                                "content": result,
                            }
                        )
                    time.sleep(1)
                    continue
                else:
                    return response_message.content if response_message.content else ""

            except Exception as e:
                error_str = str(e)

                # --- Rate limit (429) ---
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    if rate_limit_retries >= MAX_RATE_LIMIT_RETRIES:
                        logging.error("Exceeded max rate-limit retries.")
                        return "Rate limit still active after several retries. Please wait a minute and try again."

                    logging.warning(f"Rate limit hit: {error_str}")
                    wait_seconds = 62.0
                    try:
                        retry_after = e.response.headers.get("retry-after")  # type: ignore[attr-defined]
                        if retry_after:
                            wait_seconds = float(retry_after) + 2
                    except AttributeError:
                        pass
                    if wait_seconds == 62.0:
                        m = re.search(r"try again in ([\d.]+)s", error_str)
                        if m:
                            wait_seconds = float(m.group(1)) + 2

                    rate_limit_retries += 1
                    logging.warning(
                        f"Waiting {wait_seconds:.1f}s (attempt {rate_limit_retries}/{MAX_RATE_LIMIT_RETRIES})"
                    )
                    print(
                        f"\n[Rate limit — waiting {wait_seconds:.0f}s... (attempt {rate_limit_retries}/{MAX_RATE_LIMIT_RETRIES})]",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                    continue

                # --- API syntax rejection ---
                if "failed_generation" in error_str or "tool_use_failed" in error_str:
                    logging.warning(
                        f"API syntax rejection (attempt {model_retry_count + 1})"
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "System Error: Invalid tool call arguments. Use a valid JSON dict. Try again.",
                        }
                    )
                    model_retry_count += 1
                    time.sleep(2)
                    continue

                return f"API Error: {error_str}"


def main():
    parser = argparse.ArgumentParser(description="AI Code Assistant")
    parser.add_argument("--api-key", help="Groq API key (or set GROQ_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "Error: Provide an API key via --api-key or GROQ_API_KEY environment variable"
        )
        sys.exit(1)

    agent = AIAgent(api_key)

    print("AI Code Assistant")
    print("=================")
    print("Tools: list_files, read_file, edit_file")
    print("Commands: /clear  (reset memory)  |  exit / quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if user_input.lower() == "/clear":
                if agent.messages:
                    agent.messages = [agent.messages[0]]
                print("Memory cleared.\n")
                continue
            if not user_input:
                continue
            print("\nAssistant: ", end="", flush=True)
            response = agent.chat(user_input)
            if response:
                print(response)
            print()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nTerminal Error: {str(e)}\n")


if __name__ == "__main__":
    main()
