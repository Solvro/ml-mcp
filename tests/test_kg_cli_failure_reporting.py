"""`kg` has to report a failed tool call, not dump a traceback.

Once #64 made the server raise ToolError instead of returning "Error: ..." as text, the CLI's
uncaught exception became the user-visible output. The answer goes to stdout, so a failure has
to go somewhere else and exit non-zero - otherwise anything piping `kg` reads an error as an
answer, which is the same confusion the issue is about, one process further out.
"""

import sys
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from src.mcp_client.client import call_knowledge_graph_tool


@pytest.mark.parametrize(
    "error",
    [
        ToolError("Knowledge graph unavailable: the server has no initialized RAG."),
        TimeoutError("The language model request exceeded the maximum allowed wait time."),
    ],
    ids=["tool-error", "timeout"],
)
def test_failure_is_reported_on_stderr_with_a_nonzero_exit(capsys, error) -> None:
    with patch.object(sys, "argv", ["kg", "Kto wyklada analize?"]):
        with patch("src.mcp_client.client.query_knowledge_graph", side_effect=error):
            with pytest.raises(SystemExit) as exit_info:
                call_knowledge_graph_tool()

    captured = capsys.readouterr()

    assert exit_info.value.code == 1
    assert str(error) in captured.err
    assert captured.out == "", "stdout carries the answer; an error there would be read as one"


def test_a_working_call_prints_nothing_extra(capsys) -> None:
    with patch.object(sys, "argv", ["kg", "Kto wyklada analize?"]):
        with patch("src.mcp_client.client.query_knowledge_graph", return_value=None) as queried:
            call_knowledge_graph_tool()

    assert queried.called
    assert capsys.readouterr().err == ""
