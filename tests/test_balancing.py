"""Unit tests for :mod:`ham10000.data.balancing`.

The critical property is that balancing touches training data only. Repeating a
validation lesion inflates apparent performance, because the same lesion is
then scored several times and its errors or successes are counted twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ham10000.data.balancing import balance, expand_validation, resample_class


@pytest.fixture
def training() -> pd.DataFrame:
    """Four nevus lesions and one melanoma lesion with two images."""
    return pd.DataFrame(
        {
            "lesion_id": ["N1", "N2", "N3", "N4", "M1", "M1"],
            "image_id": ["I1", "I2", "I3", "I4", "I5", "I6"],
            "num_images": [1, 1, 1, 1, 2, 2],
            "dx": ["nv", "nv", "nv", "nv", "mel", "mel"],
            "set": ["t1", "t1", "t1", "t1", "t1", "ta"],
        }
    )


class TestResampleClass:
    @pytest.mark.parametrize("target", [1, 3, 6, 7, 100])
    def test_counts_sum_to_the_target_exactly(self, target: int) -> None:
        frame = pd.DataFrame(
            {
                "lesion_id": ["L1", "L1", "L2"],
                "image_id": ["I1", "I2", "I3"],
                "num_images": [2, 2, 1],
            }
        )

        counts = resample_class(frame, target, rng=np.random.RandomState(0))

        assert counts.sum() == target

    def test_draws_are_spread_evenly_across_lesions(self) -> None:
        """A lesion with five images must not get five times the influence."""
        frame = pd.DataFrame(
            {
                "lesion_id": ["L1"] * 4 + ["L2"],
                "image_id": [f"I{i}" for i in range(5)],
                "num_images": [4, 4, 4, 4, 1],
            }
        )

        counts = resample_class(frame, 8, rng=np.random.RandomState(0))
        per_lesion = (
            frame.assign(n=frame["image_id"].map(counts))
            .groupby("lesion_id")["n"]
            .sum()
        )

        assert per_lesion["L1"] == per_lesion["L2"] == 4

    def test_undersampling_draws_from_distinct_lesions(self) -> None:
        """N < D means Q == 0: N lesions contribute one image each."""
        frame = pd.DataFrame(
            {
                "lesion_id": [f"L{i}" for i in range(10)],
                "image_id": [f"I{i}" for i in range(10)],
                "num_images": [1] * 10,
            }
        )

        counts = resample_class(frame, 3, rng=np.random.RandomState(0))

        assert counts.sum() == 3
        assert (counts <= 1).all()

    def test_one_image_per_lesion_uses_a_single_image(self) -> None:
        frame = pd.DataFrame(
            {
                "lesion_id": ["L1", "L1", "L1"],
                "image_id": ["I1", "I2", "I3"],
                "num_images": [3, 3, 3],
            }
        )

        counts = resample_class(
            frame, 4, rng=np.random.RandomState(0), one_image_per_lesion=True
        )

        assert counts.sum() == 4
        assert (counts > 0).sum() == 1

    def test_seeding_makes_allocation_reproducible(self) -> None:
        frame = pd.DataFrame(
            {
                "lesion_id": ["L1", "L2", "L3"],
                "image_id": ["I1", "I2", "I3"],
                "num_images": [1, 1, 1],
            }
        )

        first = resample_class(frame, 2, rng=np.random.RandomState(7))
        second = resample_class(frame, 2, rng=np.random.RandomState(7))

        pd.testing.assert_series_equal(first, second)

    def test_empty_class_raises(self) -> None:
        empty = pd.DataFrame({"lesion_id": [], "image_id": [], "num_images": []})

        with pytest.raises(ValueError, match="empty class"):
            resample_class(empty, 5, rng=np.random.RandomState(0))

    def test_negative_target_raises(self, training: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            resample_class(training, -1, rng=np.random.RandomState(0))


class TestBalance:
    def test_class_counts_hit_their_targets(self, training: pd.DataFrame) -> None:
        balanced = balance(training, {"nv": 6, "mel": 6})

        assert balanced["dx"].value_counts().to_dict() == {"nv": 6, "mel": 6}

    def test_minority_class_is_oversampled(self, training: pd.DataFrame) -> None:
        balanced = balance(training, {"nv": 4, "mel": 4})

        assert (balanced["dx"] == "mel").sum() == 4

    def test_majority_class_is_undersampled(self, training: pd.DataFrame) -> None:
        balanced = balance(training, {"nv": 2, "mel": 2})

        assert (balanced["dx"] == "nv").sum() == 2

    def test_no_new_lesions_are_invented(self, training: pd.DataFrame) -> None:
        balanced = balance(training, {"nv": 8, "mel": 8})

        assert set(balanced["lesion_id"]) <= set(training["lesion_id"])

    def test_unmentioned_classes_pass_through(self, training: pd.DataFrame) -> None:
        balanced = balance(training, {"mel": 4})

        assert (balanced["dx"] == "nv").sum() == 4

    def test_validation_rows_are_refused(self, training: pd.DataFrame) -> None:
        """The property that keeps reported performance honest."""
        contaminated = pd.concat(
            [training, training.head(1).assign(set="v1", lesion_id="V1")]
        )

        with pytest.raises(ValueError, match="validation row"):
            balance(contaminated, {"nv": 4})

    def test_absent_class_raises_rather_than_silently_skipping(
        self, training: pd.DataFrame
    ) -> None:
        """The  `except: pass` dropped the class without a word."""
        with pytest.raises(ValueError, match="absent from column"):
            balance(training, {"unicorn": 10})

    def test_input_frame_is_not_modified(self, training: pd.DataFrame) -> None:
        before = training.copy()

        balance(training, {"nv": 6, "mel": 6})

        pd.testing.assert_frame_equal(training, before)

    def test_multiplicity_columns_are_recorded(self, training: pd.DataFrame) -> None:
        balanced = balance(training, {"nv": 6, "mel": 6})

        assert "img_mult" in balanced.columns
        assert "lesion_mult" in balanced.columns
        assert balanced["lesion_mult"].min() >= 1

    def test_seeding_makes_balancing_reproducible(self, training: pd.DataFrame) -> None:
        first = balance(training, {"nv": 3, "mel": 3}, seed=1)
        second = balance(training, {"nv": 3, "mel": 3}, seed=1)

        pd.testing.assert_frame_equal(first, second)


class TestExpandValidation:
    def test_every_lesion_is_repeated_equally(self) -> None:
        """Unequal repetition would silently reweight the validation set."""
        frame = pd.DataFrame(
            {
                "lesion_id": ["L1", "L1", "L2"],
                "image_id": ["I1", "I2", "I3"],
                "num_images": [2, 2, 1],
            }
        )

        expanded = expand_validation(frame, 4)

        assert expanded["lesion_id"].value_counts().to_dict() == {"L1": 4, "L2": 4}

    def test_factor_of_one_preserves_lesion_count(self) -> None:
        frame = pd.DataFrame(
            {"lesion_id": ["L1", "L2"], "image_id": ["I1", "I2"], "num_images": [1, 1]}
        )

        assert len(expand_validation(frame, 1)) == 2

    def test_zero_factor_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {"lesion_id": ["L1"], "image_id": ["I1"], "num_images": [1]}
        )

        with pytest.raises(ValueError, match="at least 1"):
            expand_validation(frame, 0)

    def test_multiple_images_are_rotated_through(self) -> None:
        """With a stochastic transform, varied source images add diversity."""
        frame = pd.DataFrame(
            {
                "lesion_id": ["L1", "L1"],
                "image_id": ["I1", "I2"],
                "num_images": [2, 2],
            }
        )

        expanded = expand_validation(frame, 2)

        assert set(expanded["image_id"]) == {"I1", "I2"}
