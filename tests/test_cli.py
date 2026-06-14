from __future__ import annotations

from typer.testing import CliRunner

from finbot_research.cli import app


def test_cli_exposes_feature_label_diagnostics_subcommand() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "bucket-signal-diagnostics" in result.output
    assert "feature-label-diagnostics" in result.output
    assert "fundamental-research-v0" in result.output
    assert "price-strength-equity-curve-backtest" in result.output
    assert "price-strength-equity-curve-robustness" in result.output
    assert "price-strength-horizon-sensitivity" in result.output
    assert "price-strength-turnover-cost-efficiency" in result.output
    assert "price-strength-holding-period-simulation" in result.output
    assert "price-strength-portfolio-simulation" in result.output
    assert "price-strength-rebalance-feasibility" in result.output
    assert "price-strength-scorecard-v0" in result.output
    assert "price-strength-scorecard-v1" in result.output
    assert "summarize-diagnostics" in result.output
