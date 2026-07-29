"""Class balancing by lesion-aware resampling, and validation-set expansion.

The problem
-----------
HAM10000 is severely imbalanced: nevi are 6,705 of 10,015 images (67%) while
dermatofibroma is 115 (1.1%). A classifier trained on that distribution can
reach 67% accuracy by answering "nevus" unconditionally, and will, because
cross-entropy rewards it. Resampling the training set toward uniform class
counts removes that incentive.

The allocation scheme
---------------------
Resampling here is not naive row-level sampling with replacement. It preserves
the lesion structure, and the arithmetic is worth stating because it is the
most carefully considered part of this pipeline.

To draw ``N`` images of a class spread over its ``D`` distinct lesions, write
``N = Q*D + R``. Every lesion contributes ``Q`` images, and ``R`` randomly
chosen lesions contribute one more. Within a lesion holding ``k`` images, write
``Q = q*k + r``: each of its images is taken ``q`` times, and ``r`` of them once
more.

The effect is that sampling is as uniform as integer arithmetic allows at both
levels -- across lesions within a class, and across images within a lesion. A
lesion photographed five times does not thereby get five times the influence of
one photographed once.

The same arithmetic covers both directions. When ``N < D`` we get ``Q = 0``, so
no lesion contributes a base image and exactly ``R = N`` lesions contribute one
each -- undersampling by drawing from distinct lesions rather than dropping
rows arbitrarily. That is what reduces the nevi.

Failures raise
--------------
The allocation is one function applied per class, and errors propagate. Wrapping
a per-class body in `except: pass` would leave a class silently absent from the
balanced set, producing an unbalanced "balanced" training set with nothing to
indicate anything had gone wrong.

Warning
-------
Resampling multiplies *training* rows only. Applying it to validation data
would inflate apparent performance, since the same lesion would be scored
repeatedly. :func:`balance` therefore refuses frames containing validation
rows.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

__all__ = ["balance", "expand_validation", "resample_class"]


def _multiplicity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach `lesion_mult`, the number of rows each lesion now occupies."""
    output = frame.copy()
    counts = output["lesion_id"].value_counts()
    output.insert(1, "lesion_mult", output["lesion_id"].map(counts))
    return output


def resample_class(
    frame: pd.DataFrame,
    target: int,
    *,
    rng: np.random.RandomState,
    one_image_per_lesion: bool = False,
) -> pd.Series:
    """Return per-image repeat counts realising `target` draws for one class.

    Parameters
    ----------
    frame:
        Rows for a single class, with `lesion_id`, `image_id` and `num_images`.
    target:
        Number of images wanted for this class.
    rng:
        Random state, for choosing which lesions receive the remainder.
    one_image_per_lesion:
        Treat each lesion as holding exactly one image, matching the
        `train_one_img_per_lesion` setting.

    Returns
    -------
    pd.Series
        Indexed by `image_id`, valued by how many times that image is drawn.
        Sums to `target`.

    Raises
    ------
    ValueError
        If `target` is negative, or the frame is empty.

    Examples
    --------
    Two lesions, three images, asking for six draws gives each lesion three:

    >>> frame = pd.DataFrame(
    ...     {
    ...         "lesion_id": ["L1", "L1", "L2"],
    ...         "image_id": ["I1", "I2", "I3"],
    ...         "num_images": [2, 2, 1],
    ...     }
    ... )
    >>> counts = resample_class(frame, 6, rng=np.random.RandomState(0))
    >>> int(counts.sum())
    6
    >>> int(counts["I3"])
    3

    Undersampling draws from distinct lesions rather than dropping rows:

    >>> counts = resample_class(frame, 1, rng=np.random.RandomState(0))
    >>> int(counts.sum())
    1
    """
    if target < 0:
        raise ValueError(f"target must be non-negative, got {target}.")
    if frame.empty:
        raise ValueError("Cannot resample an empty class.")

    lesions = frame["lesion_id"].unique()
    n_lesions = lesions.size
    per_lesion, remainder = divmod(target, n_lesions)

    counts: dict[str, int] = dict.fromkeys(frame["image_id"], 0)

    # Base allocation: `per_lesion` draws for every lesion, spread over its
    # images as evenly as integer division allows.
    for lesion in lesions:
        images = frame.loc[frame["lesion_id"] == lesion, "image_id"].tolist()
        available = 1 if one_image_per_lesion else len(images)
        each, extra = divmod(per_lesion, available)

        for image in images[:available]:
            counts[image] += each
        if extra:
            for image in rng.choice(images[:available], extra, replace=False):
                counts[str(image)] += 1

    # Remainder: one further image from each of `remainder` distinct lesions.
    if remainder:
        chosen = rng.choice(lesions, remainder, replace=False)
        for lesion in chosen:
            images = frame.loc[frame["lesion_id"] == lesion, "image_id"].tolist()
            counts[str(rng.choice(images))] += 1

    return pd.Series(counts, name="img_mult")


def balance(
    frame: pd.DataFrame,
    sample_size: Mapping[str, int],
    *,
    class_column: str = "dx",
    seed: int = 0,
    one_image_per_lesion: bool = False,
) -> pd.DataFrame:
    """Resample a training frame toward the requested per-class counts.

    Parameters
    ----------
    frame:
        Training rows only. Not modified.
    sample_size:
        Class value to the number of images wanted. Classes not mentioned are
        passed through at their natural frequency; classes mentioned but absent
        from the frame raise.
    class_column:
        Column holding the class, typically `dx` or `label`.
    seed:
        Random seed.
    one_image_per_lesion:
        Treat each lesion as holding one image.

    Returns
    -------
    pd.DataFrame
        Rows repeated per the allocation, with `img_mult` and `lesion_mult`
        columns recording the multiplicities.

    Raises
    ------
    ValueError
        If the frame contains validation rows, or a requested class is absent.

    Examples
    --------
    >>> frame = pd.DataFrame(
    ...     {
    ...         "lesion_id": ["L1", "L2", "L3", "L4"],
    ...         "image_id": ["I1", "I2", "I3", "I4"],
    ...         "num_images": [1, 1, 1, 1],
    ...         "dx": ["nv", "nv", "nv", "mel"],
    ...         "set": ["t1", "t1", "t1", "t1"],
    ...     }
    ... )
    >>> balanced = balance(frame, {"nv": 2, "mel": 2})
    >>> dict(sorted(balanced["dx"].value_counts().items()))
    {'mel': 2, 'nv': 2}

    The minority class is oversampled by repeating its lesion:

    >>> balanced.loc[balanced["dx"] == "mel", "image_id"].tolist()
    ['I4', 'I4']
    """
    if "set" in frame.columns:
        validation = frame["set"].astype(str).str.startswith("v")
        if validation.any():
            raise ValueError(
                f"Frame contains {int(validation.sum())} validation row(s). "
                "Balancing must be applied to training data only, or "
                "validation performance is inflated by repeated lesions."
            )

    missing = set(sample_size) - set(frame[class_column].unique())
    if missing:
        raise ValueError(
            f"Requested class(es) {sorted(missing)} absent from column "
            f"{class_column!r}. Present: {sorted(frame[class_column].unique())}."
        )

    rng = np.random.RandomState(seed)
    pieces: list[pd.DataFrame] = []

    for value in frame[class_column].unique():
        class_frame = frame[frame[class_column] == value]

        if value not in sample_size:
            pieces.append(class_frame.assign(img_mult=1))
            continue

        counts = resample_class(
            class_frame,
            sample_size[value],
            rng=rng,
            one_image_per_lesion=one_image_per_lesion,
        )
        drawn = class_frame.assign(
            img_mult=class_frame["image_id"].map(counts).fillna(0).astype(int)
        )
        drawn = drawn[drawn["img_mult"] > 0]
        pieces.append(drawn.loc[drawn.index.repeat(drawn["img_mult"])])

    combined = pd.concat(pieces, ignore_index=True)
    return _multiplicity_columns(combined).reset_index(drop=True)


def expand_validation(
    frame: pd.DataFrame,
    factor: int,
    *,
    seed: int = 0,
    one_image_per_lesion: bool = False,
) -> pd.DataFrame:
    """Repeat each validation lesion `factor` times for test-time augmentation.

    Parameters
    ----------
    frame:
        Validation rows. Not modified.
    factor:
        Predictions to make per lesion, later combined into one verdict.
    seed:
        Random seed, used when spreading draws over a lesion's images.
    one_image_per_lesion:
        Repeat a single designated image rather than rotating through the
        lesion's images.

    Returns
    -------
    pd.DataFrame
        Rows repeated, with `img_mult` and `lesion_mult`.

    Notes
    -----
    This only makes sense because the evaluation transform is stochastic:
    repeating a lesion yields `factor` differently augmented views, which are
    then aggregated. With a deterministic transform it would produce `factor`
    identical predictions and change nothing but the runtime.

    Every lesion is repeated the same number of times, so the expansion cannot
    reweight the validation set across classes.

    Examples
    --------
    >>> frame = pd.DataFrame(
    ...     {
    ...         "lesion_id": ["L1", "L2"],
    ...         "image_id": ["I1", "I2"],
    ...         "num_images": [1, 1],
    ...     }
    ... )
    >>> expanded = expand_validation(frame, 3)
    >>> len(expanded)
    6
    >>> expanded["lesion_id"].value_counts().to_dict()
    {'L1': 3, 'L2': 3}
    """
    if factor < 1:
        raise ValueError(f"factor must be at least 1, got {factor}.")

    rng = np.random.RandomState(seed)
    pieces: list[pd.DataFrame] = []

    for lesion in frame["lesion_id"].unique():
        lesion_frame = frame[frame["lesion_id"] == lesion]
        counts = resample_class(
            lesion_frame,
            factor,
            rng=rng,
            one_image_per_lesion=one_image_per_lesion,
        )
        drawn = lesion_frame.assign(
            img_mult=lesion_frame["image_id"].map(counts).fillna(0).astype(int)
        )
        drawn = drawn[drawn["img_mult"] > 0]
        pieces.append(drawn.loc[drawn.index.repeat(drawn["img_mult"])])

    combined = pd.concat(pieces, ignore_index=True)
    return _multiplicity_columns(combined).reset_index(drop=True)
