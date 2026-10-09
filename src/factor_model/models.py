"""Structured data models used by the research pipeline."""

from datetime import date
from typing import Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator, model_validator
from sklearn.pipeline import Pipeline


class ValidationSplits(BaseModel):
    """Chronological data tables for one experiment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
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


class RidgeSelection(BaseModel):
    """Fitted validation winner and the schema needed to score new rows."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    best_alpha: float
    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    target_column: str
    comparison: pd.DataFrame


class ModelEvaluation(BaseModel):
    """Aligned stock scores, daily correlations, and model/baseline summaries."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scored_rows: pd.DataFrame
    daily_ic: pd.DataFrame
    summary: pd.DataFrame
