"""Unit tests for :mod:`ham10000.data.labels`.

The invariants here are not cosmetic. A model's output layer is indexed by
`LabelScheme.codes`, and stored probability CSVs are named from it, so a change
in ordering silently mismatches saved artefacts with the code that reads them.
"""

from __future__ import annotations

import pytest

from ham10000.data.labels import OTHER, LabelScheme

ALL_DX = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


class TestFromDiagnoses:
    def test_exhaustive_specification_has_no_other_class(self) -> None:
        scheme = LabelScheme.from_diagnoses(sorted(ALL_DX), ALL_DX)

        assert OTHER not in scheme.codes.values()
        assert scheme.n_classes == 7

    def test_partial_specification_gains_other_at_index_zero(self) -> None:
        scheme = LabelScheme.from_diagnoses(["mel", "nv"], ALL_DX)

        assert scheme.codes[0] == OTHER
        assert scheme.codes == {0: OTHER, 1: "mel", 2: "nv"}

    def test_uncovered_diagnoses_all_map_to_other(self) -> None:
        scheme = LabelScheme.from_diagnoses(["mel"], ALL_DX)

        assert {scheme.mapping[dx] for dx in ALL_DX - {"mel"}} == {0}

    def test_ordering_is_stable_regardless_of_input_order(self) -> None:
        """Saved model heads depend on this; shuffling must not renumber."""
        first = LabelScheme.from_diagnoses(["nv", "mel", "bcc"], ALL_DX)
        second = LabelScheme.from_diagnoses(["bcc", "nv", "mel"], ALL_DX)

        assert first.codes == second.codes

    def test_absent_diagnoses_are_dropped(self) -> None:
        scheme = LabelScheme.from_diagnoses(["mel", "unicorn"], ALL_DX)

        assert "unicorn" not in scheme.mapping
        assert scheme.codes == {0: OTHER, 1: "mel"}

    def test_specification_matching_nothing_raises(self) -> None:
        """A naive implementation built an empty scheme and failed later, elsewhere."""
        with pytest.raises(ValueError, match="None of the requested"):
            LabelScheme.from_diagnoses(["unicorn"], ALL_DX)


class TestFromGroups:
    def test_exhaustive_grouping_has_no_other(self) -> None:
        scheme = LabelScheme.from_groups(
            {
                "malignant": ["mel", "bcc", "akiec"],
                "benign": ["nv", "bkl", "df", "vasc"],
            },
            ALL_DX,
        )

        assert scheme.codes == {0: "benign", 1: "malignant"}

    def test_partial_grouping_shifts_indices_up(self) -> None:
        scheme = LabelScheme.from_groups(
            {"malignant": ["mel"], "benign": ["nv"]}, ALL_DX
        )

        assert scheme.codes == {0: OTHER, 1: "benign", 2: "malignant"}

    def test_grouped_diagnoses_share_an_index(self) -> None:
        scheme = LabelScheme.from_groups(
            {"malignant": ["mel", "bcc"], "benign": ["nv", "bkl"]},
            {"mel", "bcc", "nv", "bkl"},
        )

        assert scheme.mapping["mel"] == scheme.mapping["bcc"]
        assert scheme.mapping["nv"] != scheme.mapping["mel"]

    def test_empty_groups_are_discarded(self) -> None:
        scheme = LabelScheme.from_groups(
            {"malignant": ["mel"], "imaginary": ["unicorn"]}, {"mel", "nv"}
        )

        assert "imaginary" not in scheme.codes.values()

    def test_overlapping_groups_are_rejected(self) -> None:
        """New check: an earlier approach let a later group silently win."""
        with pytest.raises(ValueError, match="appears in both"):
            LabelScheme.from_groups(
                {"malignant": ["mel"], "high_risk": ["mel"]}, {"mel", "nv"}
            )

    def test_no_surviving_group_raises(self) -> None:
        with pytest.raises(ValueError, match="No group contains"):
            LabelScheme.from_groups({"imaginary": ["unicorn"]}, ALL_DX)


class TestSchemeInterface:
    def test_probability_columns_follow_model_output_order(self) -> None:
        scheme = LabelScheme.from_diagnoses(["mel", "nv"], {"mel", "nv"})

        assert scheme.probability_columns == ["prob_mel", "prob_nv"]

    def test_name_of_unknown_index_raises(self) -> None:
        """A naive implementation returned the index itself, producing mixed types."""
        scheme = LabelScheme.from_diagnoses(["mel", "nv"], {"mel", "nv"})

        with pytest.raises(KeyError):
            scheme.name_of(99)

    def test_every_index_is_contiguous_from_zero(self) -> None:
        """A model head of width n needs classes numbered 0..n-1."""
        scheme = LabelScheme.from_diagnoses(["mel", "nv"], ALL_DX)

        assert sorted(scheme.codes) == list(range(scheme.n_classes))

    def test_mapping_and_codes_are_consistent(self) -> None:
        scheme = LabelScheme.from_diagnoses(["mel", "nv"], ALL_DX)

        assert set(scheme.mapping.values()) == set(scheme.codes)

    def test_build_dispatches_on_type(self) -> None:
        from_list = LabelScheme.build(["mel", "nv"], {"mel", "nv"})
        from_dict = LabelScheme.build({"m": ["mel"], "n": ["nv"]}, {"mel", "nv"})

        assert from_list.n_classes == from_dict.n_classes == 2
