# 🤖 cliv

A powerful, terminal-based AI coding assistant that can safely read, list, and edit your local files. Built for speed, privacy, and accuracy with a hybrid Online/Offline architecture.

## ✨ Features

- 🌐 **Hybrid Execution**: Choose your engine. Run blazingly fast in the cloud using the Groq API, or run 100% privately offline using your local hardware via Ollama.
- 🛠️ **Autonomous File Operations**: The agent can autonomously read codebases, list directory contents, and write/edit files based on your natural language commands.
- 🛡️ **Safety Rails**: Prevents accidental overwrites. Any time the AI attempts to modify or create a file, it pauses and explicitly requests user permission [y/N].
- 🧠 **Defensive Parsing**: Built with custom JSON shields to catch and gracefully handle hallucinations and formatting quirks typical of smaller local models.
- 📦 **Reproducible Builds**: Packaged with uv for ultra-fast, conflict-free dependency management.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- `uv` (Recommended) or `pip`
- For **Online Mode**: A free [Groq API Key](https://console.groq.com/keys)
- For **Offline Mode**: [Ollama](https://ollama.com/) installed on your machine

### Installation

1. **Clone the repository**

   ```bash
git clone https://github.com/yourusername/cliv.git
    cd cliv
   ```

2. **Sync the environment**

   This project uses `uv` to guarantee exact package versions and reproducible builds.

   ```bash
   # Install uv if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Sync dependencies from the uv.lock file
   uv sync
   ```

   *(Alternatively, you can install via standard pip: `pip install -e .`)*

3. **Download the Offline Model (Optional but Recommended)**

   If you plan to run the agent without internet access, you must pull the local AI model to your machine:

   ```bash
   ollama pull qwen2.5-coder:3b
   ```

## 💻 Usage

To start the assistant, simply run the global command from anywhere in your terminal:

```bash
cliv
```

### First Run & Configuration

On startup, the CLI will ask for your Groq API key:

- **To run Online**: Paste your API key and press enter. You can choose to save this key locally so you don't have to enter it again.
- **To run Offline**: Simply press Enter without typing a key. The agent will automatically detect Ollama and pivot to your local offline model.

### Example Commands

Once the chat starts, try issuing natural language commands:

- `"List the files in my current directory."`
- `"Read main.py and tell me what it does."`
- `"Create a new file called hello.py that prints 'Hello World'."`
- `"Edit config.py and change the default port to 8080."`

## ⚙️ How It Works (Under The Hood)

The assistant uses an array of predefined JSON tool schemas (`read_file`, `list_files`, `edit_file`) which the AI uses to interact with your operating system.

- **Online Mode (Groq)**: Utilizes Meta's incredibly fast `llama-4-scout` architecture for instantaneous, highly accurate tool-calling.
- **Offline Mode (Ollama)**: Relies on Alibaba's `qwen2.5-coder:3b`. Because smaller models can sometimes struggle with strict JSON formatting, the offline engine wraps the interaction in a custom **Defensive Shield** to unwrap markdown blocks, enforce strict system rules, and prevent JSON leakage into the terminal output.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/cliv/issues).

## 📝 License

This project is licensed under the MIT License.
