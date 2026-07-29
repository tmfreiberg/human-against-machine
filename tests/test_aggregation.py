"""Unit tests for :mod:`ham10000.models.aggregation`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ham10000.models.aggregation import (
    aggregate_predictions,
    aggregate_probabilities,
    class_name_from_column,
    majority_vote,
    predicted_label,
)


@pytest.fixture
def probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lesion_id": ["L1", "L1", "L2"],
            "image_id": ["I1", "I2", "I3"],
            "prob_mel": [0.2, 0.8, 0.1],
            "prob_nv": [0.8, 0.2, 0.9],
        }
    )


class TestClassNameFromColumn:
    def test_strips_the_prefix(self) -> None:
        assert class_name_from_column("prob_mel") == "mel"

    def test_class_names_containing_underscores_survive(self) -> None:
        """Regression: an earlier approach used split('_')[1] and returned 'high'.

        Class names come from user-supplied grouping keys, so `high_risk` is a
        perfectly ordinary specification that silently broke the argmax.
        """
        assert class_name_from_column("prob_high_risk") == "high_risk"

    def test_missing_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="does not start with prefix"):
            class_name_from_column("mel")


class TestAggregateProbabilities:
    def test_max_lifts_every_image_of_a_lesion(
        self, probabilities: pd.DataFrame
    ) -> None:
        result = aggregate_probabilities(probabilities, {"max": ["mel"]})

        assert result["prob_mel"].tolist() == [0.8, 0.8, 0.1]

    def test_min_and_mean_are_supported(self, probabilities: pd.DataFrame) -> None:
        assert aggregate_probabilities(probabilities, {"min": ["mel"]})[
            "prob_mel"
        ].tolist() == [0.2, 0.2, 0.1]
        assert aggregate_probabilities(probabilities, {"mean": ["nv"]})[
            "prob_nv"
        ].tolist() == [0.5, 0.5, 0.9]

    def test_unmentioned_classes_are_untouched(
        self, probabilities: pd.DataFrame
    ) -> None:
        result = aggregate_probabilities(probabilities, {"max": ["mel"]})

        assert result["prob_nv"].tolist() == [0.8, 0.2, 0.9]

    def test_the_specification_is_not_mutated(
        self, probabilities: pd.DataFrame
    ) -> None:
        """Regression: repeated calls produced `prob_prob_mel` and silently no-opped."""
        spec = {"max": ["mel"]}

        first = aggregate_probabilities(probabilities, spec)
        second = aggregate_probabilities(probabilities, spec)

        assert spec == {"max": ["mel"]}
        pd.testing.assert_frame_equal(first, second)

    def test_input_frame_is_not_modified(self, probabilities: pd.DataFrame) -> None:
        before = probabilities.copy()

        aggregate_probabilities(probabilities, {"max": ["mel"]})

        pd.testing.assert_frame_equal(probabilities, before)

    def test_no_method_returns_an_independent_copy(
        self, probabilities: pd.DataFrame
    ) -> None:
        result = aggregate_probabilities(probabilities, None)
        result.loc[0, "prob_mel"] = 99.0

        assert probabilities.loc[0, "prob_mel"] == 0.2

    def test_unknown_aggregation_raises(self, probabilities: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate_probabilities(probabilities, {"median": ["mel"]})

    def test_missing_group_column_raises(self) -> None:
        """A naive implementation printed a message and returned None."""
        frame = pd.DataFrame({"prob_mel": [0.5]})

        with pytest.raises(KeyError, match="No grouping column"):
            aggregate_probabilities(frame, {"max": ["mel"]})

    def test_lesion_id_is_preferred_over_image_id(
        self, probabilities: pd.DataFrame
    ) -> None:
        """Grouping by image_id would make aggregation a no-op."""
        result = aggregate_probabilities(probabilities, {"max": ["mel"]})

        assert result["prob_mel"].tolist()[:2] == [0.8, 0.8]


class TestPredictedLabel:
    def test_returns_class_names_without_codes(
        self, probabilities: pd.DataFrame
    ) -> None:
        assert predicted_label(probabilities).tolist() == ["nv", "mel", "nv"]

    def test_returns_integer_labels_with_codes(
        self, probabilities: pd.DataFrame
    ) -> None:
        assert predicted_label(probabilities, {0: "nv", 1: "mel"}).tolist() == [0, 1, 0]

    def test_unknown_class_raises(self, probabilities: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="absent from label_codes"):
            predicted_label(probabilities, {0: "nv"})

    def test_no_probability_columns_raises(self) -> None:
        with pytest.raises(KeyError, match="No columns with prefix"):
            predicted_label(pd.DataFrame({"lesion_id": ["L1"]}))


class TestMajorityVote:
    def test_returns_the_most_common_value(self) -> None:
        assert majority_vote(pd.Series([1, 1, 2])) == 1

    def test_ties_are_broken_randomly_not_by_order(self) -> None:
        """Regression: the random branch was unreachable.

        `Series.mode()` returns every tied value, so it is non-empty exactly
        when a mode exists. Ties always resolved to the lowest label, which
        biases toward the low-indexed benign classes.
        """
        rng = np.random.RandomState(0)

        outcomes = {majority_vote(pd.Series([1, 2]), rng) for _ in range(100)}

        assert outcomes == {1, 2}

    def test_empty_series_yields_nan(self) -> None:
        assert np.isnan(majority_vote(pd.Series([], dtype=float)))


class TestAggregatePredictions:
    def test_majority_wins_within_a_lesion(self) -> None:
        frame = pd.DataFrame(
            {"lesion_id": ["L1", "L1", "L1", "L2"], "pred": [1, 1, 0, 0]}
        )

        assert aggregate_predictions(frame)["pred_final"].tolist() == [1, 1, 1, 0]

    def test_result_is_constant_within_a_lesion(self) -> None:
        frame = pd.DataFrame({"lesion_id": ["L1"] * 3, "pred": [1, 0, 1]})

        assert aggregate_predictions(frame)["pred_final"].nunique() == 1

    def test_seeding_makes_tie_breaks_reproducible(self) -> None:
        frame = pd.DataFrame({"lesion_id": ["L1", "L1"], "pred": [0, 1]})

        first = aggregate_predictions(frame, seed=7)["pred_final"].tolist()
        second = aggregate_predictions(frame, seed=7)["pred_final"].tolist()

        assert first == second

    def test_input_frame_is_not_modified(self) -> None:
        frame = pd.DataFrame({"lesion_id": ["L1"], "pred": [1]})
        before = frame.copy()

        aggregate_predictions(frame)

        pd.testing.assert_frame_equal(frame, before)

    def test_missing_prediction_column_raises(self) -> None:
        """The bare except silently switched to a different algorithm."""
        frame = pd.DataFrame({"lesion_id": ["L1"]})

        with pytest.raises(KeyError, match="Prediction column"):
            aggregate_predictions(frame)
