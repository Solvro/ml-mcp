import sys
import tomllib
from pathlib import Path

import pytest

from src.data_pipeline import cli as cli_module
from src.data_pipeline.pipeline import PipelineOutcome


def _stub_pipeline_cli_runtime(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
    monkeypatch.setattr(cli_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "data_pipeline_flow",
        lambda: PipelineOutcome(
            processed={"file://docs/a.pdf"},
            deleted={"file://docs/b.pdf"},
        ),
    )


def test_prefect_pipeline_main_returns_none(monkeypatch) -> None:
    _stub_pipeline_cli_runtime(monkeypatch)
    result = cli_module.prefect_pipeline_main()
    assert result is None


def test_prefect_pipeline_main_exits_zero_when_wrapped_by_sys_exit(monkeypatch) -> None:
    _stub_pipeline_cli_runtime(monkeypatch)
    with pytest.raises(SystemExit) as exc_info:
        sys.exit(cli_module.prefect_pipeline_main())
    assert exc_info.value.code in (None, 0)


def test_prefect_pipeline_script_points_to_cli_wrapper() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["prefect_pipeline"]
        == "src.data_pipeline.cli:prefect_pipeline_main"
    )
