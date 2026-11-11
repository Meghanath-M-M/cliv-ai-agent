"""Unit tests for parallel tool execution in AIAgent."""

import time
from threading import current_thread
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from cliv.agent import AIAgent
from cliv.models import ToolCall
from cliv.tools.base import BaseTool


class SlowReadTool(BaseTool):
    name = "slow_read"
    description = "Simulates a slow file read."
    permissions = frozenset({"read"})
    risk_level = "low"
    requires_approval = False

    def __init__(self, delay=0.08):
        self.delay = delay
        self.executed_threads = []

    def execute(self, path: str = "", **kwargs) -> str:
        time.sleep(self.delay)
        t_name = current_thread().name
        self.executed_threads.append(t_name)
        return f"content of {path} via {t_name}"


class MutatingWriteTool(BaseTool):
    name = "mutating_write"
    description = "Simulates a file write."
    permissions = frozenset({"write"})
    risk_level = "high"
    requires_approval = False

    def execute(self, path: str = "", **kwargs) -> str:
        return f"wrote to {path}"


class FailingReadTool(BaseTool):
    name = "failing_read"
    description = "Simulates a read tool that throws an unexpected exception."
    permissions = frozenset({"read"})
    risk_level = "low"
    requires_approval = False

    def execute(self, **kwargs) -> str:
        raise RuntimeError("Disk read error in thread")


def test_is_parallelizable():
    agent = AIAgent(force_offline=True)
    assert agent._is_parallelizable("read_file") is True
    assert agent._is_parallelizable("list_files") is True
    assert agent._is_parallelizable("search_codebase") is True

    # Mutating tools must never be parallelized
    assert agent._is_parallelizable("edit_file") is False
    assert agent._is_parallelizable("multi_edit") is False
    assert agent._is_parallelizable("remove_file") is False
    assert agent._is_parallelizable("run_command") is False
    assert agent._is_parallelizable("run_tests") is False


def test_parallel_execution_speeds_up_independent_reads(tmp_path):
    agent = AIAgent(force_offline=True)
    slow_tool = SlowReadTool(delay=0.08)
    agent.tools["slow_read"] = slow_tool

    tool_calls = [
        ToolCall(id="c1", name="slow_read", arguments={"path": "file1.py"}),
        ToolCall(id="c2", name="slow_read", arguments={"path": "file2.py"}),
        ToolCall(id="c3", name="slow_read", arguments={"path": "file3.py"}),
    ]

    start = time.perf_counter()
    results = agent._execute_tool_calls(tool_calls)
    elapsed = time.perf_counter() - start

    assert len(results) == 3
    # If sequential, 3 * 0.08s = 0.24s. With 3 threads in parallel, elapsed < 0.20s
    assert elapsed < 0.22

    # Verify results preserve original order
    assert results[0][0].id == "c1"
    assert "file1.py" in results[0][1]
    assert results[1][0].id == "c2"
    assert "file2.py" in results[1][1]
    assert results[2][0].id == "c3"
    assert "file3.py" in results[2][1]


def test_mixed_read_and_write_batches_preserve_order():
    agent = AIAgent(force_offline=True)
    slow_tool = SlowReadTool(delay=0.01)
    write_tool = MutatingWriteTool()
    agent.tools["slow_read"] = slow_tool
    agent.tools["mutating_write"] = write_tool

    tool_calls = [
        ToolCall(id="c1", name="slow_read", arguments={"path": "first.txt"}),
        ToolCall(id="c2", name="mutating_write", arguments={"path": "out.txt"}),
        ToolCall(id="c3", name="slow_read", arguments={"path": "third.txt"}),
    ]

    results = agent._execute_tool_calls(tool_calls)
    assert len(results) == 3
    assert results[0][0].id == "c1"
    assert "first.txt" in results[0][1]
    assert results[1][0].id == "c2"
    assert "wrote to out.txt" in results[1][1]
    assert results[2][0].id == "c3"
    assert "third.txt" in results[2][1]


def test_parallel_tools_disabled():
    agent = AIAgent(force_offline=True, parallel_tools=False)
    slow_tool = SlowReadTool(delay=0.04)
    agent.tools["slow_read"] = slow_tool

    tool_calls = [
        ToolCall(id="c1", name="slow_read", arguments={"path": "f1.txt"}),
        ToolCall(id="c2", name="slow_read", arguments={"path": "f2.txt"}),
    ]

    start = time.perf_counter()
    results = agent._execute_tool_calls(tool_calls)
    elapsed = time.perf_counter() - start

    assert len(results) == 2
    # Sequential: >= 0.075s
    assert elapsed >= 0.07


def test_concurrent_exception_is_isolated():
    agent = AIAgent(force_offline=True)
    agent.tools["slow_read"] = SlowReadTool(delay=0.01)
    agent.tools["failing_read"] = FailingReadTool()

    tool_calls = [
        ToolCall(id="c1", name="slow_read", arguments={"path": "f1.txt"}),
        ToolCall(id="c2", name="failing_read", arguments={}),
        ToolCall(id="c3", name="slow_read", arguments={"path": "f3.txt"}),
    ]

    results = agent._execute_tool_calls(tool_calls)
    assert len(results) == 3
    assert "content of f1.txt" in results[0][1]
    assert "Error" in results[1][1]
    assert "content of f3.txt" in results[2][1]
