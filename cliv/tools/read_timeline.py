"""Read-only access to the local development timeline."""

from cliv.timeline import TimelineStore, render_timeline
from cliv.tools.base import BaseTool


class ReadTimelineTool(BaseTool):
    name = "read_timeline"
    description = (
        "Show the recent local development timeline: requests, approvals, "
        "tool results, test outcomes, and self-healing runs for this project. "
        "Read-only."
    )
    requires_approval = False
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "How many recent events to show (1-100)",
            }
        },
        "required": [],
    }

    def execute(self, limit: int = 20, **kwargs) -> str:
        try:
            count = max(1, min(int(limit), 100))
            store = TimelineStore()
            return render_timeline(store.recent(limit=count))
        except Exception as exc:  # noqa: BLE001 - return a model-readable error
            return f"Error reading timeline: {exc}"
