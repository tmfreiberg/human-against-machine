"""Unit tests for :mod:`ham10000.evaluation.metrics`.

Two themes. First, that the metrics behave correctly under the imbalance this
dataset actually has -- an always-majority classifier must be visibly bad.
Second, that failures raise rather than becoming `nan`, since an earlier approach had
fourteen bare handlers each converting an error into missing data.
"""

from __future__ import annotations

import numpy as np
import pytest

from ham10000.evaluation.metrics import (
    confusion_frame,
    evaluate,
    per_class_recall,
    weighted_fbeta,
)

CODES = {0: "nv", 1: "mel"}


class TestBalancedAccuracy:
    def test_always_majority_classifier_is_exposed(self) -> None:
        """The reason balanced accuracy is the headline metric."""
        target = np.array([0] * 90 + [1] * 10)
        prediction = np.zeros(100, dtype=int)

        report = evaluate(target, prediction)

        assert report.accuracy == pytest.approx(0.90)
        assert report.balanced_accuracy == pytest.approx(0.50)

    def test_perfect_prediction_scores_one(self) -> None:
        target = np.array([0, 1, 0, 1])

        assert evaluate(target, target).balanced_accuracy == 1.0

    def test_balanced_accuracy_is_mean_per_class_recall(self) -> None:
        target = np.array([0, 0, 1, 1])
        prediction = np.array([0, 0, 0, 1])

        report = evaluate(target, prediction)
        recalls = per_class_recall(target, prediction, CODES)

        assert report.balanced_accuracy == pytest.approx(recalls.mean())


class TestFBeta:
    def test_beta_above_one_rewards_recall(self) -> None:
        """F2 must prefer catching melanomas over avoiding false alarms."""
        high_recall = weighted_fbeta(np.array([0.3]), np.array([0.9]), beta=2.0)
        high_precision = weighted_fbeta(np.array([0.9]), np.array([0.3]), beta=2.0)

        assert high_recall > high_precision

    def test_beta_below_one_reverses_the_preference(self) -> None:
        high_recall = weighted_fbeta(np.array([0.3]), np.array([0.9]), beta=0.5)
        high_precision = weighted_fbeta(np.array([0.9]), np.array([0.3]), beta=0.5)

        assert high_precision > high_recall

    def test_a_degenerate_class_contributes_zero_not_nan(self) -> None:
        """Regression: `precision.all() == 0` nan-ed the whole score.

        `ndarray.all()` returns a bool, so such a guard fires when *any*
        class had zero precision -- routine for a class with 87 lesions -- and
        returned nan for the entire weighted F-score.
        """
        result = weighted_fbeta(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

        assert result == pytest.approx(0.5)
        assert not np.isnan(result)

    def test_weights_shift_the_average(self) -> None:
        precision = np.array([1.0, 0.0])
        recall = np.array([1.0, 0.0])

        assert weighted_fbeta(
            precision, recall, weights=np.array([1.0, 0.0])
        ) == pytest.approx(1.0)
        assert weighted_fbeta(
            precision, recall, weights=np.array([0.0, 1.0])
        ) == pytest.approx(0.0)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            weighted_fbeta(np.array([1.0, 1.0]), np.array([1.0]))

    def test_zero_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="sum to zero"):
            weighted_fbeta(np.array([1.0]), np.array([1.0]), weights=np.array([0.0]))

    @pytest.mark.parametrize("beta", [0.0, -1.0])
    def test_non_positive_beta_raises(self, beta: float) -> None:
        with pytest.raises(ValueError, match="beta must be positive"):
            weighted_fbeta(np.array([1.0]), np.array([1.0]), beta=beta)


class TestPerClassRecall:
    def test_recall_is_computed_per_class(self) -> None:
        target = np.array([0, 0, 1, 1])
        prediction = np.array([0, 0, 0, 1])

        recalls = per_class_recall(target, prediction, CODES)

        assert recalls["nv"] == 1.0
        assert recalls["mel"] == 0.5

    def test_a_never_predicted_class_scores_zero_not_nan(self) -> None:
        target = np.array([0, 0, 1])
        prediction = np.array([0, 0, 0])

        assert per_class_recall(target, prediction, CODES)["mel"] == 0.0


class TestRocAuc:
    def test_probabilities_enable_roc_auc(self) -> None:
        target = np.array([0, 0, 1, 1])
        prediction = np.array([0, 0, 1, 1])
        probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])

        report = evaluate(target, prediction, probabilities)

        assert report.roc_auc_macro == pytest.approx(1.0)

    def test_absent_probabilities_yield_nan(self) -> None:
        report = evaluate(np.array([0, 1]), np.array([0, 1]))

        assert np.isnan(report.roc_auc_macro)

    def test_single_class_target_yields_nan_by_design(self) -> None:
        """Genuinely undefined, detected explicitly rather than caught."""
        target = np.zeros(4, dtype=int)
        probabilities = np.tile([0.9, 0.1], (4, 1))

        report = evaluate(target, np.zeros(4, dtype=int), probabilities)

        assert np.isnan(report.roc_auc_macro)


class TestFailuresRaise:
    def test_mismatched_lengths_raise(self) -> None:
        """A naive implementation returned a table of nan."""
        with pytest.raises(ValueError, match="same length"):
            evaluate(np.array([0, 1]), np.array([0]))

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            evaluate(np.array([], dtype=int), np.array([], dtype=int))

    def test_probability_row_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="probabilities has"):
            evaluate(np.array([0, 1]), np.array([0, 1]), np.array([[0.5, 0.5]]))


class TestConfusionFrame:
    def test_rows_are_true_and_columns_predicted(self) -> None:
        frame = confusion_frame(np.array([0, 0, 1]), np.array([0, 0, 0]), CODES)

        assert frame.loc["nv", "nv"] == 2
        assert frame.loc["mel", "nv"] == 1
        assert frame.index.name == "true"

    def test_never_predicted_classes_still_appear(self) -> None:
        """Replaces `pad_df`, which reconstructed them after the fact."""
        frame = confusion_frame(np.array([0, 0]), np.array([0, 0]), CODES)

        assert list(frame.columns) == ["nv", "mel"]
        assert frame.loc["mel", "mel"] == 0

    def test_numpy_integer_labels_are_handled(self) -> None:
        """Regression: `pad_df` used isinstance(x, int), false for np.int64.

        That sent every label into the non-integer bucket, leaving the integer
        set empty and raising `min() arg is an empty sequence`.
        """
        target = np.array([0, 1], dtype=np.int64)

        frame = confusion_frame(target, target, CODES)

        assert frame.shape == (2, 2)

    def test_row_sums_equal_class_support(self) -> None:
        target = np.array([0, 0, 0, 1])
        prediction = np.array([0, 1, 0, 1])

        frame = confusion_frame(target, prediction, CODES)

        assert frame.loc["nv"].sum() == 3
        assert frame.loc["mel"].sum() == 1


class TestReportSerialisation:
    def test_to_series_has_the_expected_column_names(self) -> None:
        """The results tables in the notebooks use these labels."""
        series = evaluate(np.array([0, 1]), np.array([0, 1])).to_series()

        assert set(series.index) >= {"ACC", "BACC", "F1", "F2", "MCC"}

    def test_reports_are_frozen(self) -> None:
        report = evaluate(np.array([0, 1]), np.array([0, 1]))

        with pytest.raises(AttributeError):
            report.accuracy = 0.0  # type: ignore[misc]
