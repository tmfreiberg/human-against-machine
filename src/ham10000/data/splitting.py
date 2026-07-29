"""Lesion-level train/validation splitting.

This is the project's most consequential methodological decision, so it gets
the most scrutiny.

Why lesion-level
----------------
HAM10000 contains 10,015 images of 7,470 lesions: 1,956 lesions are
photographed more than once, some up to five times. Two images of the same
lesion are near-duplicates. Splitting at the image level would therefore put
near-copies of the same lesion on both sides of the boundary, and the resulting
validation score would measure memorisation rather than generalisation.
Splitting at the lesion level -- assigning every image of a lesion to the same
side -- is the control that prevents this.

The `set` column
----------------
Each image is assigned one of four values. Stored artefacts depend on these
exact names, so they are fixed:

===========  ================================================================
Value        Meaning
===========  ================================================================
``t1``       Training, and the single designated image for its lesion
``ta``       Training, an additional image of a lesion already represented
``v1``       Validation, and the single designated image for its lesion
``va``       Validation, an additional image of a lesion already represented
===========  ================================================================

This encodes two experiments in one column. The one-image-per-lesion setting
uses `t1`/`v1` alone; the all-images setting uses `t1 | ta` and `v1 | va`.
Since `ta` is defined as images of training lesions excluding `t1`, the union
is exactly the images of the training lesions, with no double counting.

Two properties worth stating
----------------------------
:func:`assign_splits` returns a Series rather than modifying its input, so a
caller can compare two splits, or discard one, without having damaged the frame
it started from.

Randomness comes from a local `numpy.random.RandomState` rather than the global
NumPy stream, so calling this function cannot perturb any other consumer of
randomness in the same process. `tests/test_splitting.py` pins the exact
assignment produced for a given seed, so a refactor cannot quietly renumber
existing splits.

Warning
-------
This function produces a train/validation split only. Every figure reported by
this project comes from the validation set, and thresholds and aggregation
rules were chosen on that same set, so those figures are model-selection scores
rather than held-out estimates and are optimistic by an unknown amount.

A held-out test set does exist for this data, the ISIC 2018 Task 3 set, and
evaluating against it would give a cleaner number. This project does not use
it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["SplitAssignment", "SplitConfig", "assign_splits", "lesion_overlap"]

TRAIN_PRIMARY = "t1"
TRAIN_ADDITIONAL = "ta"
VAL_PRIMARY = "v1"
VAL_ADDITIONAL = "va"

#: Values appearing in the `set` column, in a stable order.
SPLIT_VALUES = (TRAIN_PRIMARY, TRAIN_ADDITIONAL, VAL_PRIMARY, VAL_ADDITIONAL)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Parameters governing the train/validation split.

    Parameters
    ----------
    train_val_ratio:
        Ratio of training lesions to validation lesions. The default of 3
        yields a 75/25 split, since the training fraction is
        ``ratio / (ratio + 1)``.
    seed:
        Seed for lesion selection and for choosing which image represents a
        lesion.
    keep_first:
        If True, a lesion's first image in file order is its designated image.
        If False (the default), a random image is chosen. Random is
        preferable: file order in HAM10000 is not arbitrary with respect to
        acquisition, so "first" is a weak but real source of bias.
    stratified:
        If True, apply the ratio within each class, so class proportions are
        preserved across the split. Given that nevi are over 70% of the data
        and dermatofibroma under 2%, an unstratified split can leave a rare
        class badly represented on one side by chance.

    Examples
    --------
    >>> SplitConfig().train_fraction
    0.75
    >>> SplitConfig(train_val_ratio=4).train_fraction
    0.8
    """

    train_val_ratio: int = 3
    seed: int = 0
    keep_first: bool = False
    stratified: bool = True

    def __post_init__(self) -> None:
        """Validate the ratio.

        A ratio below 1 would mean more validation lesions than training
        ones, and a ratio of 0 no training set at all. Rejecting the value is
        clearer than producing a degenerate split.
        """
        if self.train_val_ratio < 1:
            raise ValueError(
                f"train_val_ratio must be at least 1, got {self.train_val_ratio}."
            )

    @property
    def train_fraction(self) -> float:
        """Fraction of lesions assigned to training."""
        return self.train_val_ratio / (self.train_val_ratio + 1)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """The result of a split: a `set` Series plus the lesion partition.

    Exposing the lesion partition alongside the per-image labels is what makes
    the disjointness guarantee checkable by a caller rather than merely
    asserted in a docstring.

    Parameters
    ----------
    sets:
        Series aligned to the input index, valued in :data:`SPLIT_VALUES`.
    train_lesions:
        Lesion identifiers assigned to training.
    val_lesions:
        Lesion identifiers assigned to validation.
    """

    sets: pd.Series
    train_lesions: frozenset[str]
    val_lesions: frozenset[str]

    @property
    def is_disjoint(self) -> bool:
        """Whether no lesion appears on both sides. Always True if correct."""
        return not (self.train_lesions & self.val_lesions)


def _designated_images(
    frame: pd.DataFrame, lesions: np.ndarray, *, keep_first: bool, seed: int
) -> pd.Index:
    """Pick exactly one image per lesion from the given lesion set.

    Returns
    -------
    pd.Index
        Index labels of the chosen rows.
    """
    subset = frame[frame["lesion_id"].isin(lesions)]
    if not keep_first:
        # Shuffle before dropping duplicates, so the surviving row is a random
        # representative rather than whichever came first in file order.
        # `random_state` here is independent of the lesion-selection stream.
        subset = subset.sample(frac=1, random_state=seed)
    return subset.drop_duplicates(subset=["lesion_id"], keep="first").index


def assign_splits(
    frame: pd.DataFrame,
    config: SplitConfig | None = None,
    *,
    label_column: str = "label",
) -> SplitAssignment:
    """Assign every image to a training or validation split, lesion-wise.

    Parameters
    ----------
    frame:
        Metadata with at least `lesion_id` and `image_id` columns, plus
        `label_column` when `config.stratified` is True. Not modified.
    config:
        Split parameters. Defaults to :class:`SplitConfig`.
    label_column:
        Column holding integer class labels, used only when stratifying.

    Returns
    -------
    SplitAssignment

    Raises
    ------
    KeyError
        If a required column is missing, naming the column rather than failing
        deep inside pandas.

    Examples
    --------
    >>> frame = pd.DataFrame(
    ...     {
    ...         "lesion_id": ["L1", "L1", "L2", "L3", "L4", "L5", "L6", "L7"],
    ...         "image_id": [f"I{i}" for i in range(8)],
    ...         "label": [0, 0, 0, 0, 1, 1, 1, 1],
    ...     }
    ... )
    >>> assignment = assign_splits(frame, SplitConfig(seed=0))
    >>> assignment.is_disjoint
    True
    >>> sorted(assignment.sets.unique())
    ['t1', 'ta', 'v1']

    Both images of lesion `L1` land on the same side:

    >>> sides = assignment.sets[frame["lesion_id"] == "L1"].str[0]
    >>> sides.nunique()
    1

    The input is untouched:

    >>> "set" in frame.columns
    False
    """
    config = config or SplitConfig()

    required = {"lesion_id", "image_id"}
    if config.stratified:
        required.add(label_column)
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            f"Metadata is missing required column(s): {sorted(missing)}. "
            f"Present: {sorted(frame.columns)}."
        )

    # A local generator rather than np.random.seed. Same algorithm and stream,
    # no global state.
    rng = np.random.RandomState(config.seed)

    if config.stratified:
        lesion_rows = frame.drop_duplicates(subset=["lesion_id"], keep="first")
        groups = [
            (frame[frame[label_column] == label], count)
            for label, count in lesion_rows[label_column].value_counts().items()
        ]
    else:
        groups = [(frame, frame["lesion_id"].nunique())]

    sets = pd.Series(pd.NA, index=frame.index, dtype="object")
    train_lesions: set[str] = set()
    val_lesions: set[str] = set()

    for group, lesion_count in groups:
        distinct = group["lesion_id"].unique()
        n_train = int(config.train_fraction * lesion_count)

        chosen = rng.choice(distinct, n_train, replace=False)
        held_out = distinct[~np.isin(distinct, chosen)]
        train_lesions.update(chosen.tolist())
        val_lesions.update(held_out.tolist())

        train_primary = _designated_images(
            group, chosen, keep_first=config.keep_first, seed=config.seed
        )
        val_primary = _designated_images(
            group, held_out, keep_first=config.keep_first, seed=config.seed
        )

        in_train = group["lesion_id"].isin(chosen)
        in_val = group["lesion_id"].isin(held_out)

        # Order matters: assign the broad category first, then overwrite the
        # designated rows, which avoids constructing the complement
        # explicitly.
        sets.loc[group.index[in_train]] = TRAIN_ADDITIONAL
        sets.loc[group.index[in_val]] = VAL_ADDITIONAL
        sets.loc[train_primary] = TRAIN_PRIMARY
        sets.loc[val_primary] = VAL_PRIMARY

    return SplitAssignment(
        sets=sets,
        train_lesions=frozenset(train_lesions),
        val_lesions=frozenset(val_lesions),
    )


def lesion_overlap(frame: pd.DataFrame, set_column: str = "set") -> set[str]:
    """Return lesions appearing on both sides of the split.

    A non-empty result means leakage. Intended for use as an assertion in
    notebooks and tests, where it converts an invisible methodological error
    into a visible one.

    Parameters
    ----------
    frame:
        Metadata including `lesion_id` and a split column.
    set_column:
        Column holding the split labels.

    Returns
    -------
    set of str
        Lesion identifiers found in both training and validation. Empty when
        the split is sound.

    Examples
    --------
    >>> clean = pd.DataFrame(
    ...     {"lesion_id": ["L1", "L1", "L2"], "set": ["t1", "ta", "v1"]}
    ... )
    >>> lesion_overlap(clean)
    set()

    An image-level split leaks:

    >>> leaky = pd.DataFrame(
    ...     {"lesion_id": ["L1", "L1", "L2"], "set": ["t1", "v1", "v1"]}
    ... )
    >>> lesion_overlap(leaky)
    {'L1'}
    """
    side = frame[set_column].astype(str).str[0]
    train = set(frame.loc[side == "t", "lesion_id"])
    val = set(frame.loc[side == "v", "lesion_id"])
    return train & val
