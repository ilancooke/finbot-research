from __future__ import annotations

import pandas as pd
import pytest

from finbot_research.validation import ValidationError, validate_joinable_dataset


def test_validate_joinable_dataset_rejects_missing_columns() -> None:
    with pytest.raises(ValidationError, match="missing required columns"):
        validate_joinable_dataset(pd.DataFrame({"symbol": ["AAPL"]}), "features")


def test_validate_joinable_dataset_rejects_duplicate_symbol_dates() -> None:
    frame = pd.DataFrame({"symbol": ["AAPL", "AAPL"], "date": ["2026-01-02", "2026-01-02"]})

    with pytest.raises(ValidationError, match="duplicate symbol/date"):
        validate_joinable_dataset(frame, "features")
