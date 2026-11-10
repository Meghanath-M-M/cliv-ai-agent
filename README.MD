# > cliv_

**Terminal-native intelligence for your local codebase.**

A terminal-based AI coding assistant that reads, edits, and debugs files in your project — built for speed, privacy, and a small, auditable codebase. It uses Groq for fast online tool calling and Ollama for local execution.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Meghanath-M-M/cliv-ai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Meghanath-M-M/cliv-ai-agent/actions/workflows/ci.yml)
[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-orange)](https://console.groq.com/)
[![Ollama](https://img.shields.io/badge/Offline-Ollama-green)](https://ollama.com/)

---

## ✨ Features

- 🌐 **Hybrid Execution** — Use Groq (`gpt-oss-120b`) when online, or run locally with Ollama (`llama3.1:8b`). The agent checks connectivity at startup and between turns, then falls back to Ollama if the network drops.

- 🔁 **Model Validation & Fallback** — At startup `cliv` checks the configured Groq model against the Models API, then probes it with the agent's real tool schemas. If it was retired or cannot use tools, `cliv` tries configured fallbacks in order, tells you which model it selected, and finally falls back to Ollama. A discontinued model no longer breaks the CLI.

- ⚡ **Single-Shot or Interactive** — Run a one-off command (`cliv "fix the bug in main.py"`) or use the conversational REPL. History is persisted and restored between sessions.

- 🛠️ **Tool-Using Agent Loop** — Reads files, lists directories, writes/edits files, applies multi-file patches, deletes files, runs tests, and runs allowlisted commands. Online and tool-capable offline models use the same loop.

- 🔎 **Semantic Codebase Search (RAG)** — Chunks the project, embeds chunks with Ollama's `nomic-embed-text`, and ranks results with a vectorized cosine-similarity calculation. Embeddings are cached by chunk hash, so only new or changed chunks are re-embedded.

- 🧬 **Project DNA** — Scans the repository, detects languages, extracts Python/JS/TS classes, functions, and imports, resolves internal dependencies, and builds an architecture report with components, entry points, cycles, duplicate functions, and unused dependencies. Run `cliv --project-dna`, or simply ask `Analyse this project.` It is read-only and also works through the offline intent router.

- 🩺 **Self-Healing Test Loop** — Ask `Fix all failing tests.` CLIV runs pytest, parses the failures, asks the selected model for a small JSON repair plan, applies it through the normal diff/approval gate, and reruns the suite for up to two repair attempts. It works online or with Ollama, never edits files directly, and reports every attempt.

- 🕒 **Development Timeline** — An append-only local SQLite log of sessions, requests, approvals, tool results, test outcomes, self-heal runs, snapshots, undo operations, and Security Guardian decisions, scoped by project. Run `cliv --timeline` for a compact recent-history view, or let the agent call the read-only `read_timeline` tool. The log stores summaries rather than file contents, redacts recognized secrets, and lives at `~/.config/cliv/timeline.db`.

- 🎨 **Risk-Aware Visual Diffs** — Every edit preview carries a LOW/MEDIUM/HIGH/CRITICAL risk assessment with concrete reasons. Wide terminals get a syntax-highlighted side-by-side diff; narrow terminals and logs keep a colored unified diff. Dependency files, CI workflows, security code, test removals, large deletions, and unsafe code additions are called out before approval.

- ⏪ **Safety Snapshots & Time-Travel Undo** — CLIV captures the original bytes of every file immediately before an approved write or deletion. Run `cliv undo` to restore the most recent applied operation, or `cliv snapshots` to inspect the safety history. Undo refuses to overwrite later edits unless you explicitly force it.

- 🛡️ **Security Guardian** — A policy layer runs before every tool call and after every result. It supports `read`, `write`, `execute`, and `network` permission tiers; classifies commands such as destructive Git operations, package installs, and arbitrary Python; inherits the same operation-aware file risk shown in approval diffs; redacts recognized secrets before output reaches the model or timeline; warns the model when tool output contains instruction-like text; and writes append-only audit events. Permission denials happen before approval, so `--yes` cannot bypass them.

- 🙋 **Human-in-the-Loop Safety** — Writes, edits, multi-file patches, deletions, test runs, and commands require explicit approval. Edits show a risk-aware diff first. `--yes` auto-approves; `--dry-run` previews without applying changes.

- 🔒 **Project-Root Confinement** — Every file tool resolves paths against the project root and uses `Path.relative_to()` instead of string prefixes. Symlinks cannot escape the root. `.env`, private keys, `.ssh`, and similar secrets are blocked, and `.git` is write-protected.

- 🧠 **Defensive Parsing Shield** — Small local models sometimes print raw JSON instead of calling a function. The shield can recover **read-only** tool calls from that text. Write operations are deliberately refused by the shield so they always go through the approval gate.

- 📡 **Automatic Network Fallback** — If the connection drops mid-session, `cliv` switches to Ollama when it is available.

- 💾 **Persistent, Budgeted Memory** — History is saved to `~/.config/cliv/history.json` and trimmed by approximate token budget, preserving complete user → assistant → tool turns instead of cutting the last N messages blindly.

- 💸 **Live Session Telemetry** — Token counts and estimated Groq cost after each response.

- 📦 **Reproducible Builds** — Packaged with `uv` and locked dependencies.

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Meghanath-M-M/cliv-ai-agent.git
cd cliv-ai-agent

# 2. Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install the mode you want
uv sync --extra online      # Groq support
uv sync --extra offline     # Ollama + NumPy
uv sync --extra all         # both

# 4. (Offline/RAG only) pull the models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 5. Run
cliv
```

With pip, install the same extras:

```bash
pip install -e ".[online]"    # or ".[offline]" / ".[all]"
```

**Prerequisites:** Python 3.11+, [`uv`](https://astral.sh/uv) (recommended) or `pip`. Online mode needs a free [Groq API key](https://console.groq.com/keys). Offline mode and semantic search need [Ollama](https://ollama.com/) and the models shown above.

> A plain `pip install -e .` installs only the core CLI. It does **not** install Groq, Ollama, or NumPy.

---

## 💻 Usage

### Interactive mode

```bash
cliv
```

```
> cliv_
Mode: ONLINE
Type 'exit', 'quit', or 'clear' to manage the session.

> list files
cliv:
Action complete: Contents of .: [FILE] README.MD  [DIR] .git/ ...

Session: 5,781 tokens · ~$0.00029
```

### Single-shot mode

```bash
cliv "fix the bug in main.py"
cliv "list the files in this directory"
cliv "create a hello.py that prints 'Hello, World!'"
```

### Project DNA

Analyse the current repository without an API key:

```bash
cliv --project-dna
```

Or use the same pipeline through the agent, online or offline:

```bash
cliv "Analyse this project."
cliv --offline "Understand this codebase."
```

The report includes detected languages, architecture roles, components, entry points, declared and imported dependencies, internal dependency edges, cycles, and structural issues. Python is parsed with `ast`; JavaScript and TypeScript use a conservative import/symbol extractor. The tool is read-only and confined to the project root.

### Self-healing tests

Run the bounded repair loop directly from the prompt:

```bash
cliv "Fix all failing tests."
cliv "Run tests and repair failures."
cliv --yes "Make the tests pass."
```

CLIV runs `pytest` first, extracts failing test names and relevant source files, and sends a compact failure report to the model. The model must return a JSON plan with at most five edits. Each plan is validated against the project root and sensitive-file policy, then applied through the same combined diff and approval flow as `multi_edit`. Tests are rerun after each attempt, with a maximum of two repair attempts. If a run needs approval and you decline, the loop stops without changing anything.

The loop is intentionally narrow: it does not run arbitrary commands, does not bypass approvals, and does not hide failures. `--dry-run` reports that self-healing is skipped because it needs to execute tests and apply edits.

### Development timeline

Inspect what CLIV has done in this project:

```bash
cliv --timeline
```

The timeline is an append-only SQLite database at `~/.config/cliv/timeline.db` (override with `CLIV_TIMELINE_DB`). It records session starts, requests, approval decisions, write/test tool results, self-heal summaries, safety snapshots, and undo operations. Events are scoped by project root, so a repository only shows its own history. Database triggers reject updates and deletes. The agent can also query it through the read-only `read_timeline` tool, for example: `What have we done recently?`

### Review and undo changes

Before an approved write or deletion, CLIV captures the affected files in a local safety snapshot. The approval prompt then shows:

- the risk level and the reasons behind it;
- a syntax-highlighted side-by-side diff on wide terminals;
- a colored unified diff on narrow terminals and in logs.

After an operation, inspect or reverse it:

```bash
cliv snapshots              # list recent snapshots
cliv undo                   # restore the last applied snapshot
cliv undo 20261110T...      # restore a specific snapshot
cliv undo --yes             # skip the confirmation prompt
```

Snapshots live outside the repository at `~/.config/cliv/snapshots/` (override with `CLIV_SNAPSHOT_DIR`). Undo verifies that each file still matches the state produced by the original operation. If another edit changed a file in the meantime, CLIV stops instead of clobbering it. Use `--force` only when you deliberately want to discard those later edits.

### Security permissions

By default CLIV grants all four capabilities: `read`, `write`, `execute`, and `network`. Restrict them per session when you want a narrower agent:

```bash
cliv --permissions read "Summarize this repository."
cliv --permissions read,write "Update the docstrings in cli.py."
cliv --permissions read,write,execute "Run the tests and fix failures."
```

Use `CLIV_PERMISSIONS` or the saved config key `permissions` for a persistent policy. Permission checks run **before** the approval prompt; `--yes` does not bypass them.

The Guardian also applies command-specific checks. For example, `git push`, `git fetch`, and `python -m pip install` require `network`; `git reset --hard` is marked critical; and arbitrary Python or pytest execution is marked high risk. These command checks are deliberately heuristic. If you need a hard network boundary, run CLIV inside a container or OS-level sandbox.

Tool output is treated as untrusted data. Recognized credentials are replaced with `[REDACTED]` before they are sent back to the model or written to the timeline. If a file or command result looks like it is trying to instruct the assistant, CLIV prepends a security notice and records a `prompt_injection` audit event. The content is preserved for inspection, but the model is told not to follow instructions embedded inside it.

Allowed writes and executions, plus every permission denial, are available in the timeline as `security_decision` and `permission_denied` events. Redactions and injection warnings are recorded as `redaction` and `prompt_injection` events. Mutating tools are snapshot-backed: if a safety snapshot cannot be created, the write is blocked rather than applied without a rollback point.

### Offline mode

```bash
cliv --offline
```

Offline mode first tries Ollama's native tool calling. If the selected model does not support tools, `cliv` falls back to a deterministic regex intent router and uses Ollama only for generation. That fallback is useful, but it is intentionally limited. To force it from the start:

```bash
cliv --offline --legacy-offline
```

### Safety flags

```bash
cliv --dry-run "delete old.py"   # show what would happen; apply nothing
cliv --yes "fix main.py"         # auto-approve every tool (use with care)
```

`--yes` and `--dry-run` are mutually exclusive.

### First-run setup

On first launch, you will be prompted for a Groq API key:

- **Online mode** → paste your key. Optionally save it for future sessions.
- **Offline mode** → press `Enter` without a key, or pass `--offline`.

Saved keys live in `~/.config/cliv/config.json` with mode `0600` on POSIX (the directory is `0700`). Prefer the `GROQ_API_KEY` environment variable if you do not want a key on disk.

---

## 🏗️ Architecture

```
cliv/
├── cli.py          ← terminal UI, startup flow, REPL/single-shot routing
├── agent.py        ← agent loop, tool dispatch, approvals
├── config.py       ← API key/config storage with restrictive permissions
├── history.py      ← persistent history + token-budget trimming
├── intent.py       ← regex fallback for local models without tool calling
├── offline.py      ← deterministic offline file workflows
├── sanitizer.py    ← defensive parsing shield
├── security.py     ← project-root and sensitive-file policy
├── risk.py         ← path-aware risk classification for changes
├── guardian.py     ← permissions, command risk, redaction, injection defense
├── diff_view.py    ← risk-aware, syntax-highlighted approval rendering
├── snapshots.py    ← pre-write file snapshots, conflict checks, and undo
├── ui.py           ← UI boundary (Rich implementation + NullUI for tests)
├── intelligence/   ← Project DNA: scanner, parser, dependency graph, report
├── healing/        ← bounded pytest repair loop and structured failure parsing
├── timeline/       ← redacted, append-only SQLite development event log
├── llm/            ← provider backends, model validation, fallback router
└── tools/          ← dynamically loaded tool plugins
```

### How the Agent Loop Works

```
User Input
    ↓
agent.chat()
    ↓
Check network → validate Groq model → Groq (online) or Ollama (offline)
    ↓
Build system prompt + context within token budget
    ↓
Call backend with tool schemas
    ↓
Tool calls returned?
  ├── Yes → Guardian permission/risk check
  │        ├── Denied → append block reason → loop
  │        └── Allowed → check approval → snapshot → execute
  │                      → redact/scan result → append result → loop
  └── No  → sanitize response
              ├── Recovered read-only tool call → execute → loop
              └── Plain text → return to user
```

The loop is capped at `MAX_ITERATIONS = 10` so a confused model cannot spin forever. Tool results are appended as `role: "tool"` messages with matching `tool_call_id`s.

### Offline routing

Ollama models such as `llama3.1:8b` support tool calling and use the normal loop. Smaller or older models that reject the `tools` parameter trigger a fallback:

```
Ollama tool call fails / unsupported
    ↓
classify_intent() in code
    ↓
deterministic read / create / fix / delete / rename flow
    ↓
Ollama generates content and summaries
```

This is a deliberate workaround for weak local models, not a claim that every offline model is fully autonomous.

### The Defensive Parsing Shield

Small models sometimes emit a tool call as plain text:

```json
{"name": "read_file", "arguments": {"path": "main.py"}}
```

The shield strips markdown fences, parses the JSON, and recovers **read-only** calls such as `read_file`, `list_files`, and `search_codebase`. If the leaked call targets a write/command tool, the shield refuses and asks the user to rephrase so the normal approval path is used.

---

## 🛠️ Tool Reference

| Tool | Description | Permission | Requires Approval |
|---|---|---|---|
| `read_file` | Read a file inside the project (sensitive files blocked) | Read | No |
| `list_files` | List directory contents (sensitive entries hidden) | Read | No |
| `analyse_project` | Read-only Project DNA scan, symbol extraction, and dependency report | Read | No |
| `read_timeline` | Read the recent append-only development timeline for this project | Read | No |
| `search_codebase` | Cached semantic search with local Ollama embeddings | Read | No |
| `run_tests` | Run pytest in the project root | Execute | **Yes** |
| `run_command` | Run an allowlisted executable without a shell | Execute; Network for commands such as `git push` | **Yes** |
| `git_diff` | Show repository status and uncommitted changes | Read | No |
| `edit_file` | Write or patch a file; shows a risk-aware diff and captures a snapshot | Read + Write | **Yes** |
| `multi_edit` | Apply several edits; shows one combined risk-aware diff and captures all files | Read + Write | **Yes** |
| `remove_file` | Delete a file; captures a snapshot and shows deletion risk | Read + Write | **Yes** |

`run_tests` and `run_command` require approval because pytest imports project code and allowlisted tools such as Python and Git can still have side effects. `run_command` rejects shell metacharacters by construction: it takes a bare executable name plus an argument list and calls `subprocess.run(..., shell=False)`.

### Adding a tool

Drop a class inheriting from `BaseTool` into `cliv/tools/`. It is discovered at startup.

```python
class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file at the specified path"
    input_schema = { ... }
    requires_approval = False
    permissions = frozenset({"read"})
    risk_level = "low"

    def execute(self, path: str, **kwargs) -> str:
        ...

    # Optional: return a diff/preview shown before approval.
    def preview(self, tool_input: dict) -> str | None:
        ...

    # Optional: return operation-specific risk for the Guardian audit.
    def assess_risk(self, tool_input: dict) -> RiskAssessment | None:
        ...
```

Declare `permissions` and `risk_level` on every tool. The base class is fail-closed: an undeclared tool requires all four permissions and is treated as high risk, so a plugin cannot silently bypass a restricted policy. `requires_approval` controls the human confirmation prompt; it is separate from Guardian permissions.

### Security model

- Tools resolve paths with `Path.resolve()` and reject anything not relative to the project root.
- `Path.relative_to()` avoids the `/home/me/proj-evil` prefix bug.
- `.env` (except `.env.example`), `.ssh`, `.aws`, `.gnupg`, private keys, and similar files are blocked even inside the project.
- `.git` is write-protected.
- `read_file` and `list_files` no longer have unrestricted access to `$HOME`.
- Commands use an explicit allowlist (`CLIV_ALLOWED_COMMANDS` can extend it) and run without a shell.
- API keys are saved with `0600` permissions on POSIX.
- The timeline database is created with `0600` permissions on POSIX and rejects updates/deletes through append-only triggers.
- Before an approved file write or deletion, the agent captures a safety snapshot of the affected files. Snapshots are stored outside the project with restrictive permissions and are scoped by project root.
- Undo checks the post-operation file hashes first and refuses to overwrite files that changed after the snapshot unless `--force` is supplied.
- Approval previews classify changes as LOW/MEDIUM/HIGH/CRITICAL and call out dependency files, CI configuration, security code, deleted tests, large deletions, and unsafe code patterns.
- The Security Guardian enforces `read`, `write`, `execute`, and `network` permission tiers before approval. Restricted policies cannot be bypassed with `--yes` or by a prompt-injection attempt.
- Commands are classified separately: network-capable Git operations require `network`, destructive repository rewrites are high/critical, and arbitrary Python/pytest execution is high risk. These are heuristics, not an OS sandbox.
- Tool output is scanned for prompt-injection patterns and recognizable credentials. Injection warnings mark the content as untrusted; secrets are replaced with `[REDACTED]` before reaching the model or the timeline.
- Security decisions, denials, redactions, and prompt-injection detections are recorded as append-only timeline events. The timeline applies the same redaction pass to summaries and details.
- Mutating tools are snapshot-backed: a write is blocked if the pre-write snapshot cannot be created, so every applied mutation has a rollback point.

---

## ⚙️ Configuration

| Method | Variable / Path | Priority |
|---|---|---|
| CLI flag | `cliv --api-key YOUR_KEY` | Highest |
| Environment variable | `GROQ_API_KEY` | High |
| Saved config | `~/.config/cliv/config.json` | Medium |
| Interactive prompt | On first launch | Fallback |
| Force offline | `cliv --offline` | Overrides all of the above |

History and logs live alongside the config: `~/.config/cliv/history.json` and `~/.config/cliv/agent.log`. The embedding cache is `~/.config/cliv/embeddings.json`. The development timeline is `~/.config/cliv/timeline.db`; set `CLIV_TIMELINE_DB` to use another path. Safety snapshots live under `~/.config/cliv/snapshots/`; set `CLIV_SNAPSHOT_DIR` to move them. Logs rotate at 5 MB with three backups.

Security permissions can be set per invocation with `cliv --permissions read,write`, persistently with `CLIV_PERMISSIONS=read,write,execute`, or with the saved config key `permissions`. The default is all four tiers.

### Model selection and fallback

The Groq model is validated before the first turn:

1. The configured primary model is checked against `client.models.list()`.
2. If it is unavailable, configured fallbacks are tried in order.
3. Each candidate is probed with CLIV's actual tool schemas before it is accepted.
4. Every switch is logged and shown in the terminal as a `Model notice`; nothing is switched silently.
5. If no Groq model passes, CLIV tries Ollama. Tool-less local models still work through the deterministic intent router.

Configure the primary model and fallback order with environment variables:

```bash
export GROQ_MODEL="openai/gpt-oss-120b"
export GROQ_FALLBACK_MODELS="openai/gpt-oss-20b,llama-3.3-70b-versatile"
```

or the saved config keys `groq_model` and `groq_fallback_models`:

```bash
cliv  # first-run key setup
# ~/.config/cliv/config.json
{
  "groq_model": "openai/gpt-oss-120b",
  "groq_fallback_models": ["openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
}
```

Environment variables win over saved config. The built-in defaults are used when neither is set.

---

## 🧪 Testing

```bash
uv sync --extra dev
uv run pytest
```

The suite covers the agent loop, approval flow, path security, risk classification, diff rendering, safety snapshots and undo, Security Guardian permissions/command risk/redaction/prompt-injection handling, history trimming, backends, tool behavior, Project DNA scanning/graphs, the self-healing repair loop, the append-only development timeline, offline intent routing, RAG caching, config permissions, and the eval harness. CI runs it on Python 3.11, 3.12, and 3.13.

Run with coverage:

```bash
uv run pytest --cov=cliv --cov-report=term-missing
```

---

## 📊 Evaluation Set

`evals/` contains 15 fixed tasks with machine-checkable pass/fail criteria: create a file, fix syntax, fix a logic bug, read a value, analyse a project architecture, list files, delete, rename, multi-file edit, create a package, run tests, repair failing tests, semantic search, refuse a secret, and inspect a git diff.

```bash
python -m evals.run --list
python -m evals.run --offline
python -m evals.run --task create_file
python -m evals.run --keep --task fix_logic
```

Each task runs in a fresh temporary project. The runner prints per-task PASS/FAIL/SKIP and a final score, and exits non-zero if any task fails. The semantic-search task needs Ollama, NumPy, and a running Ollama server; it is skipped when the packages or server are unavailable.

---

## 🗺️ Roadmap

- [x] `pyproject.toml` script entry: `pip install .` exposes the `cliv` command
- [x] GitHub Actions CI with pytest coverage
- [x] Project-root path confinement and sensitive-file blocking
- [x] Approval-gated test runs and allowlisted command execution
- [x] Cached/incremental embeddings and vectorized similarity
- [x] Groq model validation, ordered fallbacks, and tool-schema probing
- [x] Project DNA: repository scanner, symbol parser, dependency graph, and architecture report
- [x] Self-healing test loop with structured pytest parsing, bounded JSON repairs, and approval-gated reruns
- [x] Development Intelligence Timeline: append-only SQLite event log, CLI view, and read-only agent tool
- [x] Risk-aware visual diffs: syntax-highlighted side-by-side review with risk levels and concrete reasons
- [x] Safety snapshots and time-travel undo: pre-write capture, conflict-aware rollback, and `cliv undo`
- [x] Security Guardian: permission tiers, command/tool risk, secret redaction, prompt-injection defense, and audit events
- [x] Context-window budget instead of a fixed message count
- [x] Multi-file edits with a combined diff before approval
- [x] Reproducible evaluation set with pass/fail scoring

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. See the [issues page](https://github.com/Meghanath-M-M/cliv-ai-agent/issues).

1. Fork the repo
2. Create your branch: `git checkout -b feature/my-new-tool`
3. Add tests for new behavior
4. Run `pytest`
5. Open a pull request

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
