from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DOTENV_FILE = ".env"


def parse_dotenv_value(key: str, dotenv_path: Path | None = None) -> str | None:
    dotenv_path = dotenv_path or Path.cwd() / DEFAULT_DOTENV_FILE
    if not dotenv_path.exists():
        return None

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        env_key, env_value = line.split("=", 1)
        if env_key.strip() == key:
            return env_value.strip().strip('"').strip("'")
    return None


def get_env(key: str, default: str = "", dotenv_path: Path | None = None) -> str:
    value = os.getenv(key)
    if value:
        return value

    dotenv_value = parse_dotenv_value(key, dotenv_path=dotenv_path)
    if dotenv_value:
        return dotenv_value

    return default


def get_data_root() -> Path:
    value = get_env("FINBOT_DATA_ROOT")
    if not value:
        raise RuntimeError("Missing FINBOT_DATA_ROOT. Set it to the shared Finbot data directory.")
    return Path(value).expanduser().resolve()


def price_features_path(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "features" / "equity_price_features.parquet"


def relative_features_path(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "features" / "equity_relative_features.parquet"


def labels_path(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "labels" / "equity_forward_return_labels.parquet"


def feature_label_diagnostics_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "feature_label_diagnostics"


def feature_label_diagnostics_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_diagnostics.parquet"


def feature_label_summary_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_summary.parquet"


def feature_label_summary_csv_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_summary.csv"


def feature_label_report_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_report.md"


def feature_label_year_spreads_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_year_spreads.parquet"


def feature_label_year_spreads_csv_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_year_spreads.csv"


def feature_label_lookback_summary_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_lookback_summary.parquet"


def feature_label_lookback_summary_csv_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_lookback_summary.csv"


def feature_label_current_year_snapshot_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_current_year_snapshot.parquet"


def feature_label_current_year_snapshot_csv_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_current_year_snapshot.csv"
