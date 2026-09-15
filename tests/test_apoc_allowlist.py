import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMPOSE = REPO / "docker" / "compose.stack.yml"
APOC_NAME_RE = re.compile(r"apoc\.[a-zA-Z_]+(?:\.[a-zA-Z_]+)*")

DRIVER_SCHEMA_CALLS = {
    "apoc.meta.data",
    "apoc.meta.graph",
    "apoc.schema.nodes",
    "apoc.schema.properties.distinct",
    "apoc.any.property",
}


def _setting(name: str) -> list[str]:
    match = re.search(
        rf"NEO4J_dbms_security_procedures_{name}=(\S+)", COMPOSE.read_text(encoding="utf-8")
    )
    assert match, f"{name} is not set in compose.stack.yml"
    return match.group(1).split(",")


def _covered(name: str, entries: list[str]) -> bool:
    for entry in entries:
        if entry == name:
            return True
        if entry.endswith(".*") and name.startswith(entry[:-1]):
            return True
    return False


def _names_in_src() -> set[str]:
    names: set[str] = set()
    for path in (REPO / "src").rglob("*.py"):
        names.update(APOC_NAME_RE.findall(path.read_text(encoding="utf-8")))
    return names


@pytest.mark.parametrize("name", sorted(_names_in_src()))
def test_every_apoc_name_in_src_is_allowlisted(name: str) -> None:
    assert _covered(name, _setting("allowlist")), (
        f"{name} is called in src/ but not in NEO4J_dbms_security_procedures_allowlist; "
        "Neo4j will answer 'There is no procedure with the name' at runtime"
    )


@pytest.mark.parametrize("name", sorted(DRIVER_SCHEMA_CALLS))
def test_the_schema_refresh_procedures_are_allowlisted_and_unsandboxed(name: str) -> None:
    """These read database internals; without `unrestricted` the server never starts."""
    assert _covered(name, _setting("allowlist"))
    assert _covered(name, _setting("unrestricted"))


def test_unrestricted_is_a_subset_of_the_allowlist() -> None:
    """Lifting the sandbox for something that is not loaded is a typo waiting to be trusted."""
    allowlist = _setting("allowlist")
    for entry in _setting("unrestricted"):
        assert entry in allowlist, f"{entry} is unrestricted but not allowlisted"


def test_no_apoc_cypher_procedure_is_loaded() -> None:
    for setting in ("allowlist", "unrestricted"):
        entries = [e for e in _setting(setting) if e.startswith("apoc.cypher")]
        assert entries == [], f"apoc.cypher.* is back in {setting}: {entries}"
    assert not _covered("apoc.cypher.runFirstColumnMany", _setting("allowlist"))
    assert not _covered("apoc.cypher.runFile", _setting("allowlist"))
    assert "apoc.cypher.runFile" not in _names_in_src()
