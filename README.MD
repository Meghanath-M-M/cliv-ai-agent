# > cliv_

**Terminal-native intelligence for your local codebase.**

A powerful, terminal-based AI coding assistant that autonomously reads, edits, and debugs your local files — built for speed, privacy, and accuracy with a hybrid Online/Offline architecture.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-orange)](https://console.groq.com/)
[![Ollama](https://img.shields.io/badge/Offline-Ollama-green)](https://ollama.com/)

---

## ✨ Features

- 🌐 **Hybrid Execution** — Run blazingly fast in the cloud via the Groq API (`llama-4-scout`), or 100% privately offline using your local hardware via Ollama (`qwen2.5-coder:3b`). No internet required in offline mode.

- ⚡ **Single-Shot or Interactive** — Run a one-off command directly from your shell (`cliv "fix the bug in main.py"`), or drop into a full conversational REPL session that remembers context across turns.

- 🛠️ **Autonomous File Operations** — Reads codebases, lists directory contents, and writes/edits files from natural language commands. No copy-pasting code back and forth.

- 🛡️ **Human-in-the-Loop Safety** — Before any file write or edit, the agent pauses and explicitly asks for permission. Your filesystem is never modified without your approval.

- 🧠 **Defensive Parsing Shield** — Smaller local models sometimes hallucinate raw JSON text instead of proper function calls. The shield intercepts, normalizes, and executes these silently in the background — so offline mode stays reliable even when the model misbehaves.

- 📡 **Automatic Network Fallback** — Pings the network before every session and mid-conversation. If your connection drops, `cliv` silently falls back to local Ollama instead of crashing.

- 💾 **Persistent Memory** — Conversation history is saved to disk (`~/.config/cliv/history.json`) and restored automatically. Resume exactly where you left off. Type `clear` to wipe and start fresh.

- 💸 **Live Session Telemetry** — Real-time token count and estimated cost ticker after every response in online mode, so you always know what a session is costing you.

- 📦 **Reproducible Builds** — Packaged with `uv` for conflict-free, locked dependency management.

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Meghanath-M-M/cliv-ai-agent.git
cd cliv-ai-agent

# 2. Install dependencies (using uv — recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. (Optional) Pull the offline model
ollama pull llama3.1:8b                

# 4. Run
cliv
```

> Alternatively, install with pip: `pip install -e .`

**Prerequisites:** Python 3.11+, [`uv`](https://astral.sh/uv) (recommended) or `pip`. For online mode, a free [Groq API key](https://console.groq.com/keys). For offline mode, [Ollama](https://ollama.com/) installed and running.

---

## 💻 Usage

### Interactive mode

```bash
cliv
```

```
> cliv_
==================
Mode: ONLINE
Terminal-native intelligence for your local codebase.
Type 'exit', 'quit', or 'clear' to manage the session.

> list files
cliv:
Action complete: Contents of .: [FILE] .env  [DIR] .git/ ...

Session: 5,781 tokens · ~$0.00029
```

### Single-shot mode

Skip the REPL entirely and run a command directly from your shell — perfect for scripting or quick one-off tasks:

```bash
cliv "fix the bug in main.py"
cliv "list the files in this directory"
cliv "create a hello.py that prints 'Hello, World!'"
```

### Force offline mode

```bash
cliv --offline
```

### First-Run Setup

On first launch, you'll be prompted for a Groq API key:

- **Online mode** → Paste your key. Optionally save it for future sessions.
- **Offline mode** → Press `Enter` without a key. `cliv` auto-detects Ollama and routes locally.

If you're online but your connection drops mid-session, `cliv` detects it automatically and switches to Ollama without interrupting your workflow.

---

## 🏗️ Architecture

```
cliv/
├── cli.py        ← Terminal UI (Rich), startup flow, routing, streaming output
├── agent.py      ← Core brain: tool routing, LLM calls, memory, parsing shield
├── config.py     ← API key management (env var → config file fallback)
└── tools/        ← Dynamically-loaded tool plugins (read_file, list_files, edit_file)
```

### How the Agent Loop Works

```
User Input
    ↓
agent.chat()  ←──────────────────────────────────────┐
    ↓                                                 │
Check network → online (Groq) or offline (Ollama)    │
    ↓                                                 │
Tool call returned?                                   │
  ├── Yes → execute_tool() → append result → loop ───┘
  └── No  → Defensive Shield: is response raw JSON?
                ├── Yes → intercept & execute tool silently
                └── No  → stream final text to user
```

### The Defensive Parsing Shield

Small local models (3B parameters) often hallucinate tool calls as plain text JSON instead of using the proper function-calling API — for example, outputting:

```json
{"name": "read_file", "arguments": {"path": "main.py"}}
```

`cliv` detects this pattern, strips markdown fences, parses the JSON, and executes the tool — turning a crash into a successful operation, entirely behind the scenes. This makes `qwen2.5-coder:3b` reliably usable for real agentic workflows without ever surfacing the recovery to the user.

---

## 🛠️ Tool Reference

| Tool | Description | Requires Approval |
|---|---|---|
| `read_file` | Read any file's contents | No |
| `list_files` | List directory contents | No |
| `edit_file` | Write or patch a file on disk | **Yes — always** |

Tools are dynamically loaded — any class inheriting from `BaseTool` and dropped into `cliv/tools/` is automatically discovered and registered at startup. Adding a new capability is a one-file change.

```python
class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file at the specified path"
    input_schema = { ... }

    def execute(self, path: str, **kwargs) -> str:
        ...
```

---

## ⚙️ Configuration

| Method | Variable / Path | Priority |
|---|---|---|
| CLI flag | `cliv --api-key YOUR_KEY` | Highest |
| Environment variable | `GROQ_API_KEY` | High |
| Saved config | `~/.config/cliv/config.json` | Medium |
| Interactive prompt | On first launch | Fallback |
| Force offline | `cliv --offline` | Overrides all of the above |

History and logs live alongside the config: `~/.config/cliv/history.json` and `~/.config/cliv/agent.log`.

---

## 🗺️ Roadmap

- [ ] `run_tests` — execute `pytest` and feed results back to the agent for self-correcting debug loops
- [ ] `git_diff` — read uncommitted changes and ask the agent to review or explain them
- [ ] `search_codebase` — local semantic search using vector embeddings (Ollama `nomic-embed-text` + numpy cosine similarity)
- [ ] `pyproject.toml` script entry for global `cliv` install via `pip install cliv-cli`
- [ ] GitHub Actions CI with pytest coverage badge

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Check the [issues page](https://github.com/Meghanath-M-M/cliv-ai-agent/issues).

1. Fork the repo
2. Create your branch: `git checkout -b feature/my-new-tool`
3. Commit your changes: `git commit -m 'feat: add run_tests tool'`
4. Push to the branch: `git push origin feature/my-new-tool`
5. Open a pull request

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
