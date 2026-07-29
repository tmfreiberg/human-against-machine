"""Unit tests for :mod:`ham10000.reporting`."""

from __future__ import annotations

import numpy as np
import pytest

from ham10000.evaluation.metrics import evaluate
from ham10000.reporting import comparison_table, confusion_with_recall, format_report

CODES = {0: "nv", 1: "mel"}
PERFECT = evaluate(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
MAJORITY = evaluate(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]))


class TestFormatReport:
    def test_produces_one_row_named_for_the_model(self) -> None:
        assert format_report(PERFECT, "rn18").index.tolist() == ["rn18"]

    def test_carries_the_expected_column_labels(self) -> None:
        """Results tables in the notebooks use these names."""
        columns = set(format_report(PERFECT).columns)

        assert {"ACC", "BACC", "F1", "F2", "MCC"} <= columns


class TestComparisonTable:
    def test_sorted_by_balanced_accuracy_descending(self) -> None:
        table = comparison_table({"majority": MAJORITY, "perfect": PERFECT})

        assert table.index.tolist() == ["perfect", "majority"]

    def test_default_sort_is_not_plain_accuracy(self) -> None:
        """Sorting by ACC would rank an always-majority model competitively."""
        target = np.array([0] * 9 + [1])
        skewed = evaluate(target, np.zeros(10, dtype=int))
        balanced = evaluate(target, np.array([0] * 7 + [1, 1, 1]))

        table = comparison_table({"skewed": skewed, "balanced": balanced})

        assert table.index[0] == "balanced"
        assert table.loc["skewed", "ACC"] > table.loc["balanced", "ACC"]

    def test_alternative_sort_column(self) -> None:
        table = comparison_table(
            {"majority": MAJORITY, "perfect": PERFECT}, sort_by="F2"
        )

        assert table.index[0] == "perfect"

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="No reports"):
            comparison_table({})

    def test_unknown_sort_column_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            comparison_table({"a": PERFECT}, sort_by="AUROC")


class TestConfusionWithRecall:
    def test_recall_sits_beside_the_matrix(self) -> None:
        table = confusion_with_recall(
            np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), CODES
        )

        assert table["recall"].tolist() == [1.0, 0.5]

    def test_support_is_reported(self) -> None:
        """A recall of 1.0 over two lesions should not read as a strong result."""
        table = confusion_with_recall(
            np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1]), CODES
        )

        assert table.loc["mel", "support"] == 1

    def test_classes_never_predicted_still_appear(self) -> None:
        table = confusion_with_recall(np.array([0, 0]), np.array([0, 0]), CODES)

        assert "mel" in table.index
        assert table.loc["mel", "recall"] != table.loc["mel", "recall"]  # nan

    def test_row_sums_match_support(self) -> None:
        table = confusion_with_recall(
            np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), CODES
        )

        assert table.loc["nv", ["nv", "mel"]].sum() == table.loc["nv", "support"]
