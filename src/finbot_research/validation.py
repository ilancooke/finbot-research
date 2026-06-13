from __future__ import annotations

import pandas as pd


class ValidationError(ValueError):
    """Raised when a research input dataset is not usable."""


def require_non_empty(dataframe: pd.DataFrame, name: str) -> None:
    if dataframe.empty:
        raise ValidationError(f"{name} is empty")


def require_columns(dataframe: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValidationError(f"{name} missing required columns: {missing}")


def require_unique_symbol_dates(dataframe: pd.DataFrame, name: str) -> None:
    if not {"symbol", "date"}.issubset(dataframe.columns):
        return
    duplicates = dataframe.duplicated(subset=["symbol", "date"], keep=False)
    if duplicates.any():
        raise ValidationError(f"{name} contains duplicate symbol/date rows")


def validate_joinable_dataset(dataframe: pd.DataFrame, name: str) -> None:
    require_non_empty(dataframe, name)
    require_columns(dataframe, ("symbol", "date"), name)
    require_unique_symbol_dates(dataframe, name)
