"""Tests for validated experiment settings."""

import unittest
from datetime import date

from pydantic import ValidationError

from factor_model.models import ExperimentConfig


class ExperimentConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = {
            "symbols": ("aapl", "MSFT"),
            "start_date": date(2018, 1, 1),
            "end_date": date(2025, 1, 1),
            "validation_start": date(2022, 1, 1),
            "test_start": date(2023, 1, 1),
        }

    def test_normalizes_symbols_and_keeps_configuration_frozen(self) -> None:
        config = ExperimentConfig(**self.settings)

        self.assertEqual(config.symbols, ("AAPL", "MSFT"))
        self.assertEqual(config.target_horizon, 5)
        with self.assertRaises(ValidationError):
            config.target_horizon = 20

    def test_rejects_empty_and_duplicate_symbols(self) -> None:
        for symbols in [(), ("AAPL", " "), ("AAPL", "aapl")]:
            with self.subTest(symbols=symbols), self.assertRaises(ValidationError):
                ExperimentConfig(**{**self.settings, "symbols": symbols})

    def test_rejects_invalid_date_order(self) -> None:
        with self.assertRaisesRegex(ValidationError, "dates must satisfy"):
            ExperimentConfig(
                **{**self.settings, "validation_start": date(2025, 1, 1)}
            )

    def test_rejects_nonpositive_horizon(self) -> None:
        with self.assertRaises(ValidationError):
            ExperimentConfig(**{**self.settings, "target_horizon": 0})


if __name__ == "__main__":
    unittest.main()
