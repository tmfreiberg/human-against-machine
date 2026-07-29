"""Combining per-image predictions into per-lesion verdicts.

A lesion may be photographed several times, and the model scores each image
independently. Something has to reconcile those scores into one answer per
lesion, because a lesion is what a clinician is deciding about.

Two reconciliation points exist, and they are not equivalent:

* :func:`aggregate_probabilities` combines the *probabilities* first, then takes
  a single argmax. Choosing `max` for melanoma here means "if any view of this
  lesion looked malignant, treat the lesion as that likely to be malignant" --
  an explicitly sensitivity-favouring rule for a screening context.
* :func:`aggregate_predictions` lets each image vote and combines the *labels*.

Thresholding is deliberately absent from this module; it lives in
:mod:`ham10000.models.thresholds`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PROBABILITY_PREFIX",
    "aggregate_predictions",
    "aggregate_probabilities",
    "class_name_from_column",
    "majority_vote",
    "predicted_label",
]

PROBABILITY_PREFIX = "prob_"


def class_name_from_column(column: str, prefix: str = PROBABILITY_PREFIX) -> str:
    """Recover a class name from its probability column.

    Notes
    -----
    The prefix is stripped rather than the name split on underscores. Splitting
    works only for class names without underscores, and class names come from
    the keys of a user-supplied grouping dict, so a specification such as
    ``{"high_risk": [...]}`` would yield ``prob_high_risk`` -> ``"high"`` and a
    `KeyError` deep inside the argmax.

    Examples
    --------
    >>> class_name_from_column("prob_mel")
    'mel'
    >>> class_name_from_column("prob_high_risk")
    'high_risk'

    >>> class_name_from_column("mel")
    Traceback (most recent call last):
        ...
    ValueError: Column 'mel' does not start with prefix 'prob_'.
    """
    if not column.startswith(prefix):
        raise ValueError(f"Column {column!r} does not start with prefix {prefix!r}.")
    return column[len(prefix) :]


def aggregate_probabilities(
    frame: pd.DataFrame,
    method: Mapping[str, Sequence[str]] | None = None,
    *,
    group_column: str | None = None,
    prefix: str = PROBABILITY_PREFIX,
) -> pd.DataFrame:
    """Replace each image's class probabilities with a per-lesion aggregate.

    Parameters
    ----------
    frame:
        Probability table with `prob_<class>` columns and a grouping column.
        Not modified.
    method:
        Aggregation name (`mean`, `max`, `min`) to the classes it applies to.
        Classes not mentioned keep their per-image values.
    group_column:
        Column to group on. Defaults to `lesion_id` when present, else
        `image_id`.
    prefix:
        Probability column prefix.

    Returns
    -------
    pd.DataFrame
        A copy with the selected columns replaced by their group aggregate.

    Raises
    ------
    KeyError
        If no grouping column can be found. Returning `None` instead would
        hand the caller a value it does not expect and fail one line later.
    ValueError
        On an unrecognised aggregation name.

    Notes
    -----
    The `method` mapping is not mutated. Rewriting the caller's dict in place
    to attach column prefixes would mean a second call with the same dict looks
    for `prob_prob_mel`, matches nothing, and silently performs no aggregation:
    a failure with no error and no visible difference except wrong numbers.

    Examples
    --------
    >>> frame = pd.DataFrame(
    ...     {
    ...         "lesion_id": ["L1", "L1", "L2"],
    ...         "prob_mel": [0.2, 0.8, 0.1],
    ...         "prob_nv": [0.8, 0.2, 0.9],
    ...     }
    ... )
    >>> aggregate_probabilities(frame, {"max": ["mel"]})["prob_mel"].tolist()
    [0.8, 0.8, 0.1]

    Unmentioned classes are untouched:

    >>> aggregate_probabilities(frame, {"max": ["mel"]})["prob_nv"].tolist()
    [0.8, 0.2, 0.9]

    The caller's specification survives intact:

    >>> spec = {"max": ["mel"]}
    >>> _ = aggregate_probabilities(frame, spec)
    >>> _ = aggregate_probabilities(frame, spec)
    >>> spec
    {'max': ['mel']}
    """
    if not method:
        return frame.copy()

    valid = {"mean", "max", "min"}
    unknown = set(method) - valid
    if unknown:
        raise ValueError(
            f"Unknown aggregation(s): {sorted(unknown)}. Expected {sorted(valid)}."
        )

    if group_column is None:
        for candidate in ("lesion_id", "image_id"):
            if candidate in frame.columns:
                group_column = candidate
                break
        else:
            raise KeyError(
                "No grouping column found. Expected 'lesion_id' or 'image_id'; "
                f"present: {sorted(frame.columns)}."
            )

    output = frame.copy()
    probability_columns = [c for c in output.columns if c.startswith(prefix)]

    for name, classes in method.items():
        columns = [f"{prefix}{c}" for c in classes]
        columns = [c for c in columns if c in probability_columns]
        if not columns:
            continue
        aggregated = output.groupby(group_column)[columns].agg(name)
        for column in columns:
            output[column] = output[group_column].map(aggregated[column])

    return output


def predicted_label(
    frame: pd.DataFrame,
    label_codes: Mapping[int, str] | None = None,
    *,
    prefix: str = PROBABILITY_PREFIX,
) -> pd.Series:
    """Return the argmax class for each row.

    Parameters
    ----------
    frame:
        Probability table.
    label_codes:
        Index to class name. When given, integer labels are returned; when
        omitted, class names are.
    prefix:
        Probability column prefix.

    Returns
    -------
    pd.Series
        Integer labels or class names, aligned to `frame.index`.

    Notes
    -----
    Vectorised via `idxmax` over the probability block rather than
    `DataFrame.apply(..., axis=1)`, which invoked a Python function per row.

    Examples
    --------
    >>> frame = pd.DataFrame({"prob_mel": [0.2, 0.8], "prob_nv": [0.8, 0.2]})
    >>> predicted_label(frame).tolist()
    ['nv', 'mel']
    >>> predicted_label(frame, {0: "nv", 1: "mel"}).tolist()
    [0, 1]
    """
    columns = [c for c in frame.columns if c.startswith(prefix)]
    if not columns:
        raise KeyError(
            f"No columns with prefix {prefix!r}. Present: {sorted(frame.columns)}."
        )

    names = (
        frame[columns]
        .astype(float)
        .idxmax(axis=1)
        .map(lambda column: class_name_from_column(column, prefix))
    )
    if label_codes is None:
        return names

    inverse = {name: index for index, name in label_codes.items()}
    missing = set(names) - set(inverse)
    if missing:
        raise KeyError(
            f"Predicted class(es) {sorted(missing)} absent from label_codes "
            f"{sorted(inverse)}."
        )
    return names.map(inverse)


def majority_vote(
    values: pd.Series, rng: np.random.RandomState | None = None
) -> object:
    """Return the most common value, breaking ties at random.

    Notes
    -----
    Note that `Series.mode()` returns *every* tied value rather than one, so a
    guard of the form ``if not modes.empty: return modes[0]`` never reaches its
    alternative branch and resolves every tie by sorted order. For integer
    class labels that means the lowest index, and since `other` and `nv` occupy
    low indices it would bias tied lesions toward the benign class. In a
    melanoma-screening context that is the wrong direction to be biased, which
    is why the tie-break here is genuinely random and seeded.

    Examples
    --------
    >>> int(majority_vote(pd.Series([1, 1, 2])))
    1

    A tie is resolved by chance, not by ordering:

    >>> rng = np.random.RandomState(0)
    >>> outcomes = {int(majority_vote(pd.Series([1, 2]), rng)) for _ in range(50)}
    >>> outcomes == {1, 2}
    True

    >>> majority_vote(pd.Series([], dtype=float))
    nan
    """
    modes = values.mode()
    if modes.empty:
        return np.nan
    if len(modes) == 1:
        return modes.iloc[0]
    rng = rng if rng is not None else np.random.RandomState()
    return modes.iloc[rng.randint(len(modes))]


def aggregate_predictions(
    frame: pd.DataFrame,
    *,
    group_column: str | None = None,
    prediction_column: str = "pred",
    output_column: str = "pred_final",
    seed: int | None = None,
) -> pd.DataFrame:
    """Combine per-image predictions into one prediction per lesion.

    Parameters
    ----------
    frame:
        Table with a prediction column and a grouping column. Not modified.
    group_column:
        Defaults to `lesion_id` when present, else `image_id`.
    prediction_column:
        Column holding per-image predictions.
    output_column:
        Column to write the per-lesion verdict into.
    seed:
        Seed for tie-breaking, making the result reproducible.

    Returns
    -------
    pd.DataFrame
        A copy with `output_column` added.

    Notes
    -----
    Errors propagate. Catching them and falling back to a different
    aggregation would mean a missing column or a dtype problem silently
    switches algorithm, with no way for the caller to tell which one produced
    the answer.

    Examples
    --------
    >>> frame = pd.DataFrame(
    ...     {"lesion_id": ["L1", "L1", "L1", "L2"], "pred": [1, 1, 0, 0]}
    ... )
    >>> aggregate_predictions(frame)["pred_final"].tolist()
    [1, 1, 1, 0]
    """
    if group_column is None:
        for candidate in ("lesion_id", "image_id"):
            if candidate in frame.columns:
                group_column = candidate
                break
        else:
            raise KeyError(
                "No grouping column found. Expected 'lesion_id' or 'image_id'; "
                f"present: {sorted(frame.columns)}."
            )

    if prediction_column not in frame.columns:
        raise KeyError(
            f"Prediction column {prediction_column!r} not found. "
            f"Present: {sorted(frame.columns)}."
        )

    rng = np.random.RandomState(seed) if seed is not None else None
    verdicts = {
        key: majority_vote(group, rng)
        for key, group in frame.groupby(group_column)[prediction_column]
    }

    output = frame.copy()
    output[output_column] = output[group_column].map(verdicts)
    return output
