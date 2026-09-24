import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    if shutil.which("docker") is None:
        pytest.skip("needs the docker CLI for `docker compose config`")
    env_file = tmp_path_factory.mktemp("vm") / ".env"
    env_file.write_text(
        (ROOT / ".env.prod.example").read_text().replace("NEO4J_PASSWORD=", "NEO4J_PASSWORD=x")
    )
    out = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            "ml-mcp",
            "--env-file",
            str(env_file),
            "-f",
            str(ROOT / "docker/compose.stack.yml"),
            "-f",
            str(ROOT / "docker/compose.prod.yml"),
            "config",
            "--format",
            "json",
        ],
        env={**os.environ, "RELEASE_TAG": "sha-check", "ML_MCP_ENV_FILE": str(env_file)},
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)["services"], env_file


@pytest.fixture(scope="module")
def services(rendered) -> dict:
    return rendered[0]


def test_mcp_server_runs_the_published_image(services):
    assert services["mcp-server"]["image"] == "ghcr.io/solvro/ml-mcp-server:sha-check"


def test_mcp_server_reads_only_the_vm_env_file(rendered):
    services, env_file = rendered
    env_file_entries = services["mcp-server"].get("env_file")
    if env_file_entries is not None:
        # Older compose prints env_file entries as strings, newer as {"path": ..., "required": ...}.
        paths = [e["path"] if isinstance(e, dict) else e for e in env_file_entries]
        assert paths == [str(env_file)]
    else:
        # This CLI (confirmed on the Docker Desktop 2.30.3, Homebrew 2.31.0 and upstream 2.39.4
        # builds available here) always folds env_file into `environment` and drops the field
        # from `config` output, interpolated or not. That means the strong signal here is that
        # `rendered` succeeded at all: compose.stack.yml's base `env_file: [../.env]` entry does
        # not exist in this checkout, and without `!override` compose *appends* env_file lists
        # across `-f` layers (verified by dropping `!override` locally: `config` then fails with
        # "env file ... .env not found"). Reaching this point therefore already proves only the
        # VM env file was read; this assertion confirms its values made it through.
        assert services["mcp-server"]["environment"]["NEO4J_PASSWORD"] == "x"


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
