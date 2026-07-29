"""Unit tests for :mod:`ham10000.models.thresholds`.

`TestReferenceEquivalence` is the load-bearing class. A deliberately naive,
row-by-row transcription of the rule is written out below, and the vectorised
implementation is checked against it over two thousand random probability
vectors. The two must agree exactly: every thresholded figure this project
reports depends on the rule behaving as described.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from ham10000.models.aggregation import predicted_label
from ham10000.models.thresholds import (
    apply_cost_sensitive_weights,
    apply_priority_thresholds,
)

# The five-class task as configured in the notebooks.
CLASSES = ["other", "akiec", "bcc", "mel", "nv"]
CODES = dict(enumerate(CLASSES))
PROMOTE = [("mel", 0.4), ("bcc", 0.4), ("akiec", 0.4)]
DEMOTE = [("nv", 0.6)]


def reference_threshold(
    probabilities: pd.Series,
    threshold_dict_help: OrderedDict[str, float] | None,
    threshold_dict_hinder: OrderedDict[str, float] | None,
    prefix: str = "prob_",
) -> pd.Series:
    """Transcription of the  `multiclass_models.threshold`.

    Reproduced in behaviour including the in-place mutation and the `break`
    that limits promotion to a single class.
    """
    probabilities = probabilities.copy()
    if isinstance(threshold_dict_help, OrderedDict):
        for dx, thres in threshold_dict_help.items():
            if (
                prefix + dx in probabilities.index
                and probabilities[prefix + dx] > thres
            ):
                probabilities[prefix + dx] = 1
                break
    if isinstance(threshold_dict_hinder, OrderedDict):
        for dx, thres in threshold_dict_hinder.items():
            if (
                prefix + dx in probabilities.index
                and probabilities[prefix + dx] < thres
            ):
                probabilities[prefix + dx] = 0
                break
    return probabilities


@pytest.fixture
def probabilities() -> pd.DataFrame:
    """Two thousand realistic probability vectors over five classes."""
    rng = np.random.RandomState(0)
    values = rng.dirichlet(np.full(len(CLASSES), 0.5), size=2000)
    return pd.DataFrame(values, columns=[f"prob_{c}" for c in CLASSES])


class TestReferenceEquivalence:
    def test_matches_the_reference_implementation(
        self, probabilities: pd.DataFrame
    ) -> None:
        migrated = apply_priority_thresholds(
            probabilities, promote=PROMOTE, demote=DEMOTE
        )

        expected = probabilities.apply(
            lambda row: reference_threshold(
                row, OrderedDict(PROMOTE), OrderedDict(DEMOTE)
            ),
            axis=1,
        )

        pd.testing.assert_frame_equal(migrated, expected, check_dtype=False)

    def test_matches_with_promotion_only(self, probabilities: pd.DataFrame) -> None:
        migrated = apply_priority_thresholds(probabilities, promote=PROMOTE)
        expected = probabilities.apply(
            lambda row: reference_threshold(row, OrderedDict(PROMOTE), None), axis=1
        )

        pd.testing.assert_frame_equal(migrated, expected, check_dtype=False)

    def test_matches_with_demotion_only(self, probabilities: pd.DataFrame) -> None:
        migrated = apply_priority_thresholds(probabilities, demote=DEMOTE)
        expected = probabilities.apply(
            lambda row: reference_threshold(row, None, OrderedDict(DEMOTE)), axis=1
        )

        pd.testing.assert_frame_equal(migrated, expected, check_dtype=False)

    def test_predictions_match_too(self, probabilities: pd.DataFrame) -> None:
        """Equivalence at the level that actually matters: the class chosen."""
        migrated = predicted_label(
            apply_priority_thresholds(probabilities, promote=PROMOTE, demote=DEMOTE),
            CODES,
        )
        expected = predicted_label(
            probabilities.apply(
                lambda row: reference_threshold(
                    row, OrderedDict(PROMOTE), OrderedDict(DEMOTE)
                ),
                axis=1,
            ),
            CODES,
        )

        assert (migrated == expected).all()


class TestRuleA:
    def test_promotion_overturns_a_larger_score(self) -> None:
        frame = pd.DataFrame({"prob_mel": [0.45], "prob_nv": [0.50]})

        adjusted = apply_priority_thresholds(frame, promote=[("mel", 0.4)])

        assert predicted_label(adjusted).iloc[0] == "mel"

    def test_only_the_first_qualifying_class_is_promoted(self) -> None:
        """The ordering is the clinical priority, and the `break` enforces it."""
        frame = pd.DataFrame(
            {"prob_mel": [0.42], "prob_bcc": [0.45], "prob_nv": [0.13]}
        )

        adjusted = apply_priority_thresholds(
            frame, promote=[("mel", 0.4), ("bcc", 0.4)]
        )

        assert adjusted["prob_mel"].iloc[0] == 1.0
        assert adjusted["prob_bcc"].iloc[0] == 0.45
        assert predicted_label(adjusted).iloc[0] == "mel"

    def test_reordering_changes_the_outcome(self) -> None:
        """If order did not matter, the OrderedDict would be pointless."""
        frame = pd.DataFrame({"prob_mel": [0.42], "prob_bcc": [0.45]})

        mel_first = apply_priority_thresholds(
            frame, promote=[("mel", 0.4), ("bcc", 0.4)]
        )
        bcc_first = apply_priority_thresholds(
            frame, promote=[("bcc", 0.4), ("mel", 0.4)]
        )

        assert predicted_label(mel_first).iloc[0] == "mel"
        assert predicted_label(bcc_first).iloc[0] == "bcc"

    def test_nothing_happens_below_the_bar(self) -> None:
        """Rule A is a gate: no crossing, no effect."""
        frame = pd.DataFrame({"prob_mel": [0.35], "prob_nv": [0.65]})

        adjusted = apply_priority_thresholds(frame, promote=[("mel", 0.4)])

        pd.testing.assert_frame_equal(adjusted, frame)

    def test_threshold_is_strict(self) -> None:
        """Strict comparison: a probability equal to the bar does not clear it."""
        frame = pd.DataFrame({"prob_mel": [0.4], "prob_nv": [0.6]})

        adjusted = apply_priority_thresholds(frame, promote=[("mel", 0.4)])

        assert adjusted["prob_mel"].iloc[0] == 0.4

    def test_demotion_removes_a_class_from_contention(self) -> None:
        frame = pd.DataFrame({"prob_mel": [0.45], "prob_nv": [0.55]})

        adjusted = apply_priority_thresholds(frame, demote=[("nv", 0.6)])

        assert adjusted["prob_nv"].iloc[0] == 0.0
        assert predicted_label(adjusted).iloc[0] == "mel"

    def test_unknown_class_names_are_ignored(self) -> None:
        frame = pd.DataFrame({"prob_mel": [0.45], "prob_nv": [0.55]})

        adjusted = apply_priority_thresholds(frame, promote=[("unicorn", 0.1)])

        pd.testing.assert_frame_equal(adjusted, frame)

    def test_input_frame_is_not_modified(self) -> None:
        """A naive implementation mutated the caller's Series in place."""
        frame = pd.DataFrame({"prob_mel": [0.45], "prob_nv": [0.55]})
        before = frame.copy()

        apply_priority_thresholds(frame, promote=[("mel", 0.4)], demote=[("nv", 0.6)])

        pd.testing.assert_frame_equal(frame, before)

    def test_non_probability_columns_are_preserved(self) -> None:
        frame = pd.DataFrame(
            {"lesion_id": ["L1"], "prob_mel": [0.45], "prob_nv": [0.55]}
        )

        adjusted = apply_priority_thresholds(frame, promote=[("mel", 0.4)])

        assert adjusted["lesion_id"].tolist() == ["L1"]

    def test_no_rules_is_a_no_op(self, probabilities: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            apply_priority_thresholds(probabilities), probabilities
        )


class TestRuleC:
    def test_boost_overturns_a_larger_raw_score(self) -> None:
        frame = pd.DataFrame({"prob_mel": [0.35], "prob_nv": [0.65]})

        adjusted = apply_cost_sensitive_weights(frame, thresholds=[("mel", 0.4)])

        assert predicted_label(adjusted).iloc[0] == "mel"

    def test_equal_thresholds_reduce_to_plain_argmax(
        self, probabilities: pd.DataFrame
    ) -> None:
        uniform = [(c, 0.5) for c in CLASSES]

        adjusted = apply_cost_sensitive_weights(probabilities, thresholds=uniform)

        assert (predicted_label(adjusted) == predicted_label(probabilities)).all()

    def test_ordering_is_irrelevant(self) -> None:
        """Unlike Rule A, priority must be encoded in the values."""
        frame = pd.DataFrame({"prob_mel": [0.42], "prob_bcc": [0.45]})

        one = apply_cost_sensitive_weights(
            frame, thresholds=[("mel", 0.4), ("bcc", 0.4)]
        )
        two = apply_cost_sensitive_weights(
            frame, thresholds=[("bcc", 0.4), ("mel", 0.4)]
        )

        pd.testing.assert_frame_equal(one, two)

    def test_priority_can_be_encoded_in_the_values(self) -> None:
        frame = pd.DataFrame({"prob_mel": [0.42], "prob_bcc": [0.45]})

        adjusted = apply_cost_sensitive_weights(
            frame, thresholds=[("mel", 0.3), ("bcc", 0.4)]
        )

        assert predicted_label(adjusted).iloc[0] == "mel"

    def test_ranking_is_a_monotone_transform(self) -> None:
        """Scaling by positive constants cannot reorder within a class."""
        frame = pd.DataFrame({"prob_mel": [0.1, 0.3], "prob_nv": [0.9, 0.7]})

        adjusted = apply_cost_sensitive_weights(frame, thresholds=[("mel", 0.4)])

        assert adjusted["prob_mel"].is_monotonic_increasing

    def test_non_positive_threshold_raises(self) -> None:
        frame = pd.DataFrame({"prob_mel": [0.5]})

        with pytest.raises(ValueError, match="must be positive"):
            apply_cost_sensitive_weights(frame, thresholds=[("mel", 0.0)])

    def test_input_frame_is_not_modified(self) -> None:
        frame = pd.DataFrame({"prob_mel": [0.35], "prob_nv": [0.65]})
        before = frame.copy()

        apply_cost_sensitive_weights(frame, thresholds=[("mel", 0.4)])

        pd.testing.assert_frame_equal(frame, before)


class TestRulesDiffer:
    def test_the_two_rules_are_not_equivalent(
        self, probabilities: pd.DataFrame
    ) -> None:
        """Documented behaviour: A is a gate, C is a reweighting."""
        a = predicted_label(
            apply_priority_thresholds(probabilities, promote=PROMOTE, demote=DEMOTE),
            CODES,
        )
        c = predicted_label(
            apply_cost_sensitive_weights(probabilities, thresholds=PROMOTE), CODES
        )

        agreement = (a == c).mean()
        assert 0.7 < agreement < 0.95

    def test_rule_c_calls_melanoma_more_often(
        self, probabilities: pd.DataFrame
    ) -> None:
        """C's effective condition is p_mel > 0.4 * p_other, far weaker."""
        plain = (predicted_label(probabilities, CODES) == 3).mean()
        a = (
            predicted_label(
                apply_priority_thresholds(probabilities, promote=PROMOTE), CODES
            )
            == 3
        ).mean()
        c = (
            predicted_label(
                apply_cost_sensitive_weights(probabilities, thresholds=PROMOTE), CODES
            )
            == 3
        ).mean()

        assert plain < a < c

    def test_they_diverge_when_nothing_crosses_a_bar(self) -> None:
        frame = pd.DataFrame({"prob_mel": [0.35], "prob_nv": [0.65]})

        a = apply_priority_thresholds(frame, promote=[("mel", 0.4)])
        c = apply_cost_sensitive_weights(frame, thresholds=[("mel", 0.4)])

        assert predicted_label(a).iloc[0] == "nv"
        assert predicted_label(c).iloc[0] == "mel"
