import os
import logging
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import ollama
except ImportError:
    ollama = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("agent.log")],
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

class Tool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]

class AIAgent:
    def __init__(self, api_key: str = None, local_model: str = "qwen2.5-coder:3b"):
        self.api_key = api_key
        self.local_model = local_model
        self.messages: List[Dict[str, Any]] = []
        self.tools: List[Tool] = []
        self._setup_tools()

        if self.api_key and Groq:
            self.mode = "online"
            self.client = Groq(api_key=self.api_key)
            self.model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        else:
            self.mode = "offline"
            if not ollama:
                logging.warning("Ollama library not found. Offline mode may fail.")

    def _setup_tools(self):
        self.tools = [
            Tool(
                name="read_file",
                description="Read the contents of a file at the specified path",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file to read"}
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="list_files",
                description="List all files and directories in the specified path",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The directory path to list"}
                    },
                    "required": [],
                },
            ),
            Tool(
                name="edit_file",
                description="Edit a file by replacing old_text with new_text. Creates the file if it doesn't exist.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file to edit"},
                        "old_text": {"type": "string", "description": "The text to search for and replace (leave empty to create)"},
                        "new_text": {"type": "string", "description": "The text to replace old_text with"},
                    },
                    "required": ["path", "new_text"],
                },
            ),
        ]

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        logging.info(f"Executing tool: {tool_name} with input: {tool_input}")
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
            logging.error(f"Error executing {tool_name}: {str(e)}")
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
        # SAFETY RAIL: Require manual confirmation before modifying disk
        print(f"\n[SYSTEM WARNING] The AI wants to modify the file: '{path}'")
        confirm = input("Allow this change? [y/N]: ").strip().lower()
        if confirm != 'y':
            return f"Operation blocked: User denied permission to edit {path}."

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
        logging.info(f"User input: {user_input}")
        self.messages.append({"role": "user", "content": user_input})

        tool_schemas = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
        } for tool in self.tools]

        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful coding assistant operating in a terminal environment.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. When the user says a simple greeting like 'hi' or 'hello', you MUST respond ONLY with plain English text.\n"
                "   Example User: 'hi'\n"
                "   Example Assistant: 'Hello! I am your AI code assistant. How can I help you with your files today?'\n"
                "2. NEVER output raw JSON in your text responses.\n"
                "3. To interact with files, use the provided tools silently in the background."
            ),
        }
        while True:
            try:
                normalized_tool_calls = []
                content = None

                if self.mode == "online":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        max_tokens=4096,
                        messages=[system_msg] + self.messages,
                        tools=tool_schemas,
                        tool_choice="auto",
                    )
                    message = response.choices[0].message
                    content = message.content

                    if message.tool_calls:
                        for tc in message.tool_calls:
                            normalized_tool_calls.append({
                                "id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            })

                else:
                    # Fetch response from Ollama
                    response = ollama.chat(
                        model=self.local_model,
                        messages=[system_msg] + self.messages,
                        tools=tool_schemas,
                    )
                    
                    # Robust handling for both dict and object structures across library versions
                    if hasattr(response, 'message') or isinstance(response, dict):
                        msg_obj = response.message if hasattr(response, 'message') else response.get("message", {})
                        
                        if hasattr(msg_obj, 'content'):
                            content = msg_obj.content
                            raw_tool_calls = msg_obj.tool_calls or []
                        else:
                            content = msg_obj.get("content")
                            raw_tool_calls = msg_obj.get("tool_calls") or []
                    else:
                        content = ""
                        raw_tool_calls = []

                    for i, tc in enumerate(raw_tool_calls):
                        # Handle if tool call is an object or a dictionary
                        if hasattr(tc, 'function'):
                            tc_name = tc.function.name
                            args = tc.function.arguments
                        else:
                            tc_name = tc["function"]["name"]
                            args = tc["function"]["arguments"]

                        args_str = json.dumps(args) if isinstance(args, dict) else args
                        normalized_tool_calls.append({
                            "id": f"call_{i}",
                            "name": tc_name,
                            "arguments": args_str
                        })

                # Record assistant response
                assistant_message = {"role": "assistant"}
                if content is not None:
                    assistant_message["content"] = content
                
                if normalized_tool_calls:
                    assistant_message["tool_calls"] = [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    } for tc in normalized_tool_calls]

                self.messages.append(assistant_message)

                # Process tools
                if normalized_tool_calls:
                    for tc in normalized_tool_calls:
                        try:
                            function_args = json.loads(tc["arguments"])
                        except json.JSONDecodeError:
                            function_args = {}

                        result = self._execute_tool(tc["name"], function_args)
                        logging.info(f"Tool result: {result[:500]}...")

                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": result,
                        })
                else:
                    return content if content else ""

            except Exception as e:
                return f"Error [{self.mode.upper()} Mode]: {str(e)}"