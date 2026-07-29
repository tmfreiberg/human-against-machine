"""Unit tests for :mod:`ham10000.data.splitting`.

Two of these carry more weight than the rest.

`TestReferenceEquivalence` holds an independent, deliberately naive
transcription of the splitting algorithm and asserts the vectorised
implementation agrees with it exactly. Any figure this project reports depends
on the split, so a refactor that quietly changed which lesions land where would
invalidate every result without failing anything else.

`TestNoLeakage` checks the property the whole design exists to guarantee, on
data constructed so that an image-level split would visibly violate it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ham10000.data.splitting import (
    SplitConfig,
    assign_splits,
    lesion_overlap,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Metadata with deliberately uneven lesion multiplicity.

    Twelve lesions across two classes; four have multiple images. Any
    image-level split of this frame would separate images of a shared lesion.
    """
    rows: list[dict[str, object]] = []
    multiplicity = {"L1": 3, "L2": 2, "L3": 1, "L4": 1, "L5": 2, "L6": 1}
    image = 0
    for label in (0, 1):
        for lesion, count in multiplicity.items():
            for _ in range(count):
                rows.append(
                    {
                        "lesion_id": f"{lesion}_c{label}",
                        "image_id": f"I{image:03d}",
                        "dx": "mel" if label else "nv",
                        "label": label,
                    }
                )
                image += 1
    return pd.DataFrame(rows)


def reference_assign(
    df: pd.DataFrame, tvr: int, seed: int, keep_first: bool, stratified: bool
) -> pd.Series:
    """Transcription of `processing.process.train_val_split` ().

    Reproduced verbatim in behaviour, including the global RNG seeding and the
    in-place assignment, so that the migrated implementation can be checked
    against it rather than against a description of it.
    """
    df = df.copy()
    tvr_multiplier = tvr / (tvr + 1)

    if stratified:
        train_size = dict(
            df.drop_duplicates(subset=["lesion_id"], keep="first")[
                "label"
            ].value_counts()
        )
        train_size = {k: int(tvr_multiplier * v) for k, v in train_size.items()}
        np.random.seed(seed)
        for label, t_size in train_size.items():
            distinct_lesions = df[df["label"] == label]["lesion_id"].unique()
            t = np.random.choice(distinct_lesions, t_size, replace=False)
            v = distinct_lesions[~np.isin(distinct_lesions, t)]
            if keep_first:
                t1 = df[df["lesion_id"].isin(t)].drop_duplicates(
                    subset=["lesion_id"], keep="first"
                )["image_id"]
                v1 = df[df["lesion_id"].isin(v)].drop_duplicates(
                    subset=["lesion_id"], keep="first"
                )["image_id"]
            else:
                t1 = (
                    df[df["lesion_id"].isin(t)]
                    .sample(frac=1, random_state=seed)
                    .drop_duplicates(subset=["lesion_id"], keep="first")["image_id"]
                )
                v1 = (
                    df[df["lesion_id"].isin(v)]
                    .sample(frac=1, random_state=seed)
                    .drop_duplicates(subset=["lesion_id"], keep="first")["image_id"]
                )
            ta = df[(df["lesion_id"].isin(t)) & ~(df["image_id"].isin(t1))]["image_id"]
            va = df[(df["lesion_id"].isin(v)) & ~(df["image_id"].isin(v1))]["image_id"]
            df.loc[df["image_id"].isin(t1), "set"] = "t1"
            df.loc[df["image_id"].isin(v1), "set"] = "v1"
            df.loc[df["image_id"].isin(ta), "set"] = "ta"
            df.loc[df["image_id"].isin(va), "set"] = "va"
    else:
        distinct_lesions = df["lesion_id"].unique()
        n = int(tvr_multiplier * distinct_lesions.shape[0])
        np.random.seed(seed)
        t = np.random.choice(distinct_lesions, n, replace=False)
        v = distinct_lesions[~np.isin(distinct_lesions, t)]
        if keep_first:
            t1 = df[df["lesion_id"].isin(t)].drop_duplicates(
                subset=["lesion_id"], keep="first"
            )["image_id"]
            v1 = df[df["lesion_id"].isin(v)].drop_duplicates(
                subset=["lesion_id"], keep="first"
            )["image_id"]
        else:
            t1 = (
                df[df["lesion_id"].isin(t)]
                .sample(frac=1, random_state=seed)
                .drop_duplicates(subset=["lesion_id"], keep="first")["image_id"]
            )
            v1 = (
                df[df["lesion_id"].isin(v)]
                .sample(frac=1, random_state=seed)
                .drop_duplicates(subset=["lesion_id"], keep="first")["image_id"]
            )
        ta = df[(df["lesion_id"].isin(t)) & ~(df["image_id"].isin(t1))]["image_id"]
        va = df[(df["lesion_id"].isin(v)) & ~(df["image_id"].isin(v1))]["image_id"]
        df.loc[df["image_id"].isin(t1), "set"] = "t1"
        df.loc[df["image_id"].isin(v1), "set"] = "v1"
        df.loc[df["image_id"].isin(ta), "set"] = "ta"
        df.loc[df["image_id"].isin(va), "set"] = "va"

    return df["set"]


class TestReferenceEquivalence:
    @pytest.mark.parametrize("stratified", [True, False])
    @pytest.mark.parametrize("keep_first", [True, False])
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_matches_the_reference_algorithm(
        self, frame: pd.DataFrame, stratified: bool, keep_first: bool, seed: int
    ) -> None:
        config = SplitConfig(
            train_val_ratio=3, seed=seed, keep_first=keep_first, stratified=stratified
        )

        migrated = assign_splits(frame, config)
        reference = reference_assign(
            frame, tvr=3, seed=seed, keep_first=keep_first, stratified=stratified
        )

        pd.testing.assert_series_equal(
            migrated.sets.astype(str), reference.astype(str), check_names=False
        )

    def test_does_not_disturb_the_global_random_stream(
        self, frame: pd.DataFrame
    ) -> None:
        """Seeding the global NumPy RNG would silently reseed the whole process."""
        np.random.seed(12345)
        expected = np.random.random()

        np.random.seed(12345)
        assign_splits(frame, SplitConfig(seed=7))
        actual = np.random.random()

        assert actual == expected


class TestNoLeakage:
    @pytest.mark.parametrize("stratified", [True, False])
    @pytest.mark.parametrize("seed", range(5))
    def test_no_lesion_appears_on_both_sides(
        self, frame: pd.DataFrame, stratified: bool, seed: int
    ) -> None:
        """The guarantee the lesion-level design exists to provide."""
        assignment = assign_splits(frame, SplitConfig(seed=seed, stratified=stratified))

        assert assignment.is_disjoint

        annotated = frame.assign(set=assignment.sets)
        assert lesion_overlap(annotated) == set()

    def test_every_image_of_a_lesion_shares_a_side(self, frame: pd.DataFrame) -> None:
        assignment = assign_splits(frame)
        sides = frame.assign(side=assignment.sets.str[0])

        per_lesion = sides.groupby("lesion_id")["side"].nunique()

        assert (per_lesion == 1).all()

    def test_detects_leakage_when_it_exists(self) -> None:
        """`lesion_overlap` must actually catch a bad split, not just pass."""
        leaky = pd.DataFrame(
            {"lesion_id": ["L1", "L1", "L2"], "set": ["t1", "v1", "va"]}
        )

        assert lesion_overlap(leaky) == {"L1"}


class TestAssignment:
    def test_every_image_is_assigned(self, frame: pd.DataFrame) -> None:
        assignment = assign_splits(frame)

        assert assignment.sets.notna().all()

    def test_input_frame_is_not_modified(self, frame: pd.DataFrame) -> None:
        before = frame.copy()

        assign_splits(frame)

        pd.testing.assert_frame_equal(frame, before)

    def test_exactly_one_primary_image_per_lesion(self, frame: pd.DataFrame) -> None:
        """`t1` and `v1` designate a single representative image per lesion."""
        assignment = assign_splits(frame)
        primary = frame[assignment.sets.isin(["t1", "v1"])]

        assert not primary["lesion_id"].duplicated().any()
        assert set(primary["lesion_id"]) == set(frame["lesion_id"])

    def test_all_images_union_equals_the_lesion_images(
        self, frame: pd.DataFrame
    ) -> None:
        """`t1 | ta` must be exactly the images of the training lesions."""
        assignment = assign_splits(frame)
        annotated = frame.assign(set=assignment.sets)

        train_images = set(
            annotated.loc[annotated["set"].isin(["t1", "ta"]), "image_id"]
        )
        expected = set(
            annotated.loc[
                annotated["lesion_id"].isin(assignment.train_lesions), "image_id"
            ]
        )

        assert train_images == expected

    def test_stratification_preserves_class_proportions(self) -> None:
        """Unstratified splitting can misrepresent a rare class by chance."""
        rows = [
            {"lesion_id": f"L{i}", "image_id": f"I{i}", "label": 0 if i < 90 else 1}
            for i in range(100)
        ]
        frame = pd.DataFrame(rows)

        assignment = assign_splits(frame, SplitConfig(stratified=True))
        annotated = frame.assign(side=assignment.sets.str[0])

        rare_train = ((annotated["label"] == 1) & (annotated["side"] == "t")).sum()

        # 10 rare lesions, ratio 3 => int(0.75 * 10) == 7
        assert rare_train == 7


class TestConfig:
    def test_train_fraction_derives_from_ratio(self) -> None:
        assert SplitConfig(train_val_ratio=3).train_fraction == 0.75
        assert SplitConfig(train_val_ratio=1).train_fraction == 0.5

    def test_zero_ratio_is_rejected(self) -> None:
        """The  `tvr == 0` branch referenced an unassigned name."""
        with pytest.raises(ValueError, match="at least 1"):
            SplitConfig(train_val_ratio=0)

    def test_config_is_frozen(self) -> None:
        config = SplitConfig()

        with pytest.raises(AttributeError):
            config.seed = 99  # type: ignore[misc]


class TestValidation:
    def test_missing_column_names_itself(self) -> None:
        frame = pd.DataFrame({"lesion_id": ["L1"], "image_id": ["I1"]})

        with pytest.raises(KeyError, match="label"):
            assign_splits(frame, SplitConfig(stratified=True))

    def test_unstratified_does_not_require_a_label_column(self) -> None:
        frame = pd.DataFrame(
            {"lesion_id": ["L1", "L2", "L3", "L4"], "image_id": list("ABCD")}
        )

        assignment = assign_splits(frame, SplitConfig(stratified=False))

        assert assignment.sets.notna().all()
