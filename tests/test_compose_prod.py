import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """A throw-away copy of the two compose files, plus a planted "developer" root `.env`.

    This never touches the real checkout's root `.env` (a developer may have a real one there
    with real secrets). Instead it plants a fake one at the same relative location
    (`docker/compose.stack.yml`'s mcp-server reads `../.env`, i.e. this sandbox's own root) with
    a unique sentinel key and a `NEO4J_PASSWORD` that conflicts with the VM file's, so tests can
    prove the sentinel never reaches the container and the VM value wins the conflict -- not just
    that rendering succeeded.
    """
    if shutil.which("docker") is None:
        pytest.skip("needs the docker CLI for `docker compose config`")
    project_dir = tmp_path_factory.mktemp("ml-mcp-sandbox")
    (project_dir / "docker").mkdir()
    shutil.copy(ROOT / "docker/compose.stack.yml", project_dir / "docker/compose.stack.yml")
    shutil.copy(ROOT / "docker/compose.prod.yml", project_dir / "docker/compose.prod.yml")
    (project_dir / ".env").write_text("DEV_ONLY_SENTINEL=leaked\nNEO4J_PASSWORD=dev\n")

    vm_env_file = tmp_path_factory.mktemp("vm") / ".env"
    vm_env_file.write_text(
        (ROOT / ".env.prod.example").read_text().replace("NEO4J_PASSWORD=", "NEO4J_PASSWORD=x")
    )
    return project_dir, vm_env_file


@pytest.fixture(scope="module")
def rendered(sandbox):
    project_dir, vm_env_file = sandbox
    out = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            "ml-mcp",
            "--env-file",
            str(vm_env_file),
            "-f",
            str(project_dir / "docker/compose.stack.yml"),
            "-f",
            str(project_dir / "docker/compose.prod.yml"),
            "config",
            "--format",
            "json",
        ],
        env={**os.environ, "RELEASE_TAG": "sha-check", "ML_MCP_ENV_FILE": str(vm_env_file)},
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)["services"], vm_env_file


@pytest.fixture(scope="module")
def services(rendered) -> dict:
    return rendered[0]


@pytest.fixture(scope="module")
def base_mcp_server_env_keys(sandbox) -> set:
    """Keys compose.stack.yml itself hard-codes under mcp-server's `environment:` (not via
    env_file) -- computed from the actual base file rather than hard-coded, since the overlay
    only replaces `env_file` and must leave these alone."""
    project_dir, _ = sandbox
    base = yaml.safe_load((project_dir / "docker/compose.stack.yml").read_text())
    raw_env = base["services"]["mcp-server"].get("environment") or []
    if isinstance(raw_env, dict):
        return set(raw_env)
    return {entry.split("=", 1)[0] for entry in raw_env}


def test_mcp_server_runs_the_published_image(services):
    assert services["mcp-server"]["image"] == "ghcr.io/solvro/ml-mcp-server:sha-check"


def test_mcp_server_reads_only_the_vm_env_file(rendered, base_mcp_server_env_keys):
    services, vm_env_file = rendered
    mcp = services["mcp-server"]

    env_file_entries = mcp.get("env_file")
    if env_file_entries is not None:
        # Older compose prints env_file entries as strings, newer as {"path": ..., "required": ...}.
        paths = [e["path"] if isinstance(e, dict) else e for e in env_file_entries]
        assert paths == [str(vm_env_file)]

    # This CLI (confirmed on the Docker Desktop 2.30.3, Homebrew 2.31.0 and upstream 2.39.4
    # builds available here) always folds env_file into `environment` and drops the field from
    # `config` output regardless of the branch above, so the real proof has to be in the
    # resolved values: the sandbox plants a fake developer `.env` (see the `sandbox` fixture)
    # with a unique sentinel key and a `NEO4J_PASSWORD` that conflicts with the VM file's. If
    # `!override` ever regressed to a plain list, compose would *merge* both env files instead
    # of replacing one with the other -- the sentinel would leak into the resolved environment
    # even though the conflicting key still happened to read the VM's value (verified locally:
    # dropping `!override` here reproduces exactly that leak). So this asserts all three: the
    # sentinel never crosses over, the VM value wins the conflict, and the resolved key set is
    # exactly the VM file's keys plus what compose.stack.yml itself hard-codes -- nothing else.
    resolved = mcp["environment"]
    vm_keys = {
        line.split("=", 1)[0]
        for line in vm_env_file.read_text().splitlines()
        if line and not line.startswith("#")
    }
    assert "DEV_ONLY_SENTINEL" not in resolved
    assert resolved["NEO4J_PASSWORD"] == "x"
    assert set(resolved) == vm_keys | base_mcp_server_env_keys


def test_nothing_publishes_a_host_port(services):
    assert all(not svc.get("ports") for svc in services.values())


def test_every_service_rotates_its_logs(services):
    for name, svc in services.items():
        assert svc["logging"]["driver"] == "json-file", name
        assert svc["logging"]["options"]["max-size"] == "10m", name


def test_example_lists_only_what_serving_needs():
    keys = {
        line.split("=", 1)[0]
        for line in (ROOT / ".env.prod.example").read_text().splitlines()
        if line and not line.startswith("#")
    }
    assert {"NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OPENAI_API_KEY", "LOG_LEVEL"} <= keys
    assert not any(k.startswith(("PREFECT_", "DATA_PIPELINE_", "OCR_", "PIPELINE_")) for k in keys)
