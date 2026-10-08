"""Shared filesystem paths for research notebooks and pipeline stages."""

from datetime import date
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

EDA_DATA_PATH = PROCESSED_DATA_DIR / "eda_data.csv"
FTE_DATA_PATH = PROCESSED_DATA_DIR / "fte_data.csv"
LABELED_DATA_PATH = PROCESSED_DATA_DIR / "labeled_data.csv"
SPLIT_MANIFEST_PATH = PROCESSED_DATA_DIR / "split_manifest.csv"
SPLIT_SUMMARY_PATH = PROCESSED_DATA_DIR / "split_summary.csv"
VALIDATION_PLAN_PATH = PROCESSED_DATA_DIR / "validation_plan.json"

class ExperimentConfig(BaseModel):
    """Validated, immutable settings for one research experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbols: tuple[str, ...]
    start_date: date
    end_date: date
    validation_start: date
    test_start: date
    target_horizon: PositiveInt = 5

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, symbols: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(symbol.strip().upper() for symbol in symbols)
        if not normalized or any(not symbol for symbol in normalized):
            raise ValueError("symbols must contain at least one non-empty ticker")
        if len(normalized) != len(set(normalized)):
            raise ValueError("symbols must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_dates(self) -> Self:
        if not self.start_date < self.validation_start < self.test_start < self.end_date:
            raise ValueError(
                "dates must satisfy start_date < validation_start < test_start < end_date"
            )
        return self
