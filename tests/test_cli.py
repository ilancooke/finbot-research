from __future__ import annotations

from typer.testing import CliRunner

from finbot_research.cli import app


def test_cli_exposes_feature_label_diagnostics_subcommand() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "feature-label-diagnostics" in result.output
    assert "summarize-diagnostics" in result.output
