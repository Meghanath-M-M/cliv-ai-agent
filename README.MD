# Single-File AI Agent Tutorial

Build a complete AI agent in 218 lines of Python. No frameworks, no complexity. Just clean code that shows exactly how agents work.

This tutorial demonstrates a working agent that can read, write, and edit files through natural conversation with Claude. Everything runs from a single Python file using `uv`'s inline dependencies: no virtual environments, no pip installs, no dependency conflicts.

You'll understand how AI agents parse responses, execute tools, and maintain conversation context. The mechanics laid bare, without abstractions.

## Credits

This repository is forked from Francis Beeson's implementation: [Single-File AI Agent Tutorial](https://github.com/leobeeson/single-file-ai-agent-tutorial)

The code is based on the tutorial by Thorsten Ball in ["How to Build an Agent"](https://ampcode.com/how-to-build-an-agent) from ampcode.com. Thank you, Thorsten. Brilliant guide.

## Features

- Single-file execution.
- No virtual environment or manual dependency installation required (except for `uv`).
- The AI agent can:
  - Read file contents.
  - List directory contents.
  - Edit existing files or create new ones.
- Interactive chat interface.
- Error handling and feedback.
- Logging of agent tool usage.

## Requirements

- `uv` package manager (see installation instructions below)
- Python 3.12 or higher (though `uv` will [handle this automatically](https://docs.astral.sh/uv/concepts/python-versions/))
- Anthropic API key (you can get one from [Anthropic's Console](https://console.anthropic.com/settings/keys))

## Installing uv

`uv` is an extremely fast Python package manager that simplifies running Python scripts with inline dependencies.

It allows you to run Python scripts [without needing to create a virtual environment or manually install dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies).

To [install](https://docs.astral.sh/uv/getting-started/installation/) `uv`::

### Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, verify it's working:

```bash
uv --version
```

## Installing the Agent

The tool can be installed globally on your system using `uv`. From the repository root, run:

```bash
uv tool install .
```

If you plan to modify the code, you can install it in editable mode:
```bash
uv tool install -e .
```

## Running the Agent

Once installed, you can start the assistant from any terminal using:

```bash
ai-assist
```

The first time you run it, the agent will prompt you to enter your Anthropic API key, which it will securely save to `~/.config/ai_cli/config.json`. Alternatively, you can pass it via an environment variable or flag:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
ai-assist
```

Or:

```bash
ai-assist --api-key "your-api-key-here"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
