from __future__ import annotations

import typer

from finbot_research.config import get_data_root
from finbot_research.diagnostic_summary import build_diagnostic_summary, terminal_summary
from finbot_research.feature_label_diagnostics import build_feature_label_diagnostics

app = typer.Typer(help="Run Finbot research diagnostics.")


@app.callback()
def root() -> None:
    """Run Finbot research diagnostics."""


@app.command("feature-label-diagnostics")
def feature_label_diagnostics_command() -> None:
    """Evaluate feature signal against forward-return labels."""

    data_root = get_data_root()
    output_path, metadata_path, summary = build_feature_label_diagnostics(data_root)
    typer.echo(
        "Wrote "
        f"{output_path} rows={summary['joined_rows']} "
        f"features={len(summary['selected_feature_columns'])}"
    )
    if summary["skipped_feature_columns"]:
        typer.echo(f"Skipped missing features: {', '.join(summary['skipped_feature_columns'])}")
    typer.echo(f"Wrote {metadata_path}")


@app.command("summarize-diagnostics")
def summarize_diagnostics_command(
    include_partial_current_year_in_stability: bool = typer.Option(
        False,
        "--include-partial-current-year-in-stability",
        help="Include the max calendar year in completed-year stability and lookback calculations.",
    ),
) -> None:
    """Summarize feature diagnostics into ranked research report outputs."""

    data_root = get_data_root()
    paths, stats = build_diagnostic_summary(
        data_root,
        include_partial_current_year_in_stability=include_partial_current_year_in_stability,
    )
    typer.echo(terminal_summary(paths, stats))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
