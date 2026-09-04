"""`src/` logs, it does not print.

Issue #63 replaced every print with a logger so that LOG_LEVEL actually controls what a service
says. A print added later would be invisible to that switch and would bypass the log format, so
this walks the package and fails on one. The check is AST-based: "print(" inside a string or a
comment is not a call and must not trip it.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# The kg CLI writes the answer, and its usage help, to stdout. That is the command's output
# rather than a log line, and it has to keep working under LOG_LEVEL=WARNING.
ALLOWED_FILES = {Path("mcp_client") / "client.py"}


def _print_calls(path: Path) -> list[int]:
    """Line numbers of every call to the print builtin in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


@pytest.mark.parametrize(
    "path",
    sorted(p for p in SRC.rglob("*.py") if p.relative_to(SRC) not in ALLOWED_FILES),
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_module_logs_instead_of_printing(path: Path) -> None:
    lines = _print_calls(path)

    assert not lines, (
        f"{path.relative_to(SRC)} calls print on line(s) {lines}; "
        "use logging.getLogger(__name__) so LOG_LEVEL can control it"
    )


def test_the_allowlist_still_describes_real_files() -> None:
    """An allowlisted file that moved would silently stop being checked either way."""
    for relative in ALLOWED_FILES:
        assert (SRC / relative).is_file(), f"allowlisted {relative} no longer exists"
