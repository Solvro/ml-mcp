"""`kg` has to report a failed tool call, not dump a traceback.

Once #64 made the server raise ToolError instead of returning "Error: ..." as text, the CLI's
uncaught exception became the user-visible output. The answer goes to stdout, so a failure has
to go somewhere else and exit non-zero - otherwise anything piping `kg` reads an error as an
answer, which is the same confusion the issue is about, one process further out.
"""

import importlib
import os
import sys
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError


@pytest.fixture
def cli():
    """Import the CLI module with a throwaway key in the environment.

    `client.py` builds its LLM client at import time and raises without a key, so the module
    cannot be imported at collection on a machine that has none - CI, for one. The key is
    confined to the import and the module is dropped afterwards, so nothing leaks into tests
    that reason about which providers are configured.
    """
    module_name = "src.mcp_client.client"
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


@pytest.mark.parametrize(
    "error",
    [
        ToolError("Knowledge graph unavailable: the server has no initialized RAG."),
        TimeoutError("The language model request exceeded the maximum allowed wait time."),
    ],
    ids=["tool-error", "timeout"],
)
def test_failure_is_reported_on_stderr_with_a_nonzero_exit(cli, capsys, error) -> None:
    with patch.object(sys, "argv", ["kg", "Kto wyklada analize?"]):
        with patch.object(cli, "query_knowledge_graph", side_effect=error):
            with pytest.raises(SystemExit) as exit_info:
                cli.call_knowledge_graph_tool()

    captured = capsys.readouterr()

    assert exit_info.value.code == 1
    assert str(error) in captured.err
    assert captured.out == "", "stdout carries the answer; an error there would be read as one"


def test_a_working_call_prints_nothing_extra(cli, capsys) -> None:
    with patch.object(sys, "argv", ["kg", "Kto wyklada analize?"]):
        with patch.object(cli, "query_knowledge_graph", return_value=None) as queried:
            cli.call_knowledge_graph_tool()

    assert queried.called
    assert capsys.readouterr().err == ""


def test_usage_is_shown_without_a_question(cli, capsys) -> None:
    with patch.object(sys, "argv", ["kg"]):
        with pytest.raises(SystemExit) as exit_info:
            cli.call_knowledge_graph_tool()

    assert exit_info.value.code == 1
    assert "Usage: kg" in capsys.readouterr().out
