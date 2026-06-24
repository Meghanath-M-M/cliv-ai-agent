import os
import logging

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]
import ollama
from rich.console import Console
from typing import Any
from .base import BaseTool

console = Console()


class SearchCodebaseTool(BaseTool):
    def __init__(self):
        self.name = "search_codebase"
        self.description = "Search the entire project directory for code snippets relevant to a semantic query (RAG)."
        self.input_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g., 'where is the database connection?'",
                }
            },
            "required": ["query"],
        }
        self.collection = []
        self.is_indexed = False

    def _build_index(self):
        """Chunks the codebase and generates vector embeddings."""
        if not np:
            return "Error: numpy is required. Run `uv add numpy`."

        console.print(
            "\n[bold yellow][SYSTEM][/bold yellow] Indexing codebase for semantic search... (This takes a moment the first time)"
        )
        ignore_dirs = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}

        for root, dirs, files in os.walk("."):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for file in files:
                if file.endswith((".py", ".md", ".txt", ".json")):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Simple chunking: split into ~1000 character blocks
                        chunks = [
                            content[i : i + 1000] for i in range(0, len(content), 1000)
                        ]

                        for chunk in chunks:
                            # Convert text into a mathematical vector
                            response = ollama.embeddings(
                                model="nomic-embed-text", prompt=chunk
                            )
                            embedding = response["embedding"]
                            self.collection.append(
                                {
                                    "filepath": filepath,
                                    "content": chunk,
                                    "embedding": embedding,
                                }
                            )
                    except Exception as e:
                        logging.warning(f"Failed to index {filepath}: {e}")

        self.is_indexed = True
        console.print("[bold green][SYSTEM] Codebase index complete![/bold green]")

    def execute(self, query: str, *args: Any, **kwargs: Any) -> str:
        if not np:
            return "Error: numpy is required for vector math."

        # Build the vector database on the first search
        if not self.is_indexed:
            self._build_index()

        if not self.collection:
            return "No code files found to index."

        # Embed the user's search query
        query_res = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_vec = np.array(query_res["embedding"])

        # Calculate cosine similarity between the query and all code chunks
        results = []
        for item in self.collection:
            doc_vec = np.array(item["embedding"])
            # Cosine similarity formula: (A dot B) / (||A|| * ||B||)
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            results.append((similarity, item))

        # Sort by highest similarity
        results.sort(key=lambda x: x[0], reverse=True)

        # Return the top 3 most relevant code chunks
        top_results = results[:3]

        output = f"Top 3 codebase matches for '{query}':\n\n"
        for score, item in top_results:
            output += (
                f"--- File: {item['filepath']} (Relevance Score: {score:.2f}) ---\n"
            )
            output += f"{item['content']}\n\n"

        return output
