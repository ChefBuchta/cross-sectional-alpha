"""Shared filesystem paths for research notebooks and pipeline stages."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

EDA_DATA_PATH = PROCESSED_DATA_DIR / "eda_data.csv"
FTE_DATA_PATH = PROCESSED_DATA_DIR / "fte_data.csv"
LABELED_DATA_PATH = PROCESSED_DATA_DIR / "labeled_data.csv"
