"""Structured data models used by the research pipeline."""

import pandas as pd
from pydantic import BaseModel, ConfigDict


class ValidationSplits(BaseModel):
    """Chronological data tables for one experiment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
