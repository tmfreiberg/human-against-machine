"""Sensitivity-biased decision rules over class probabilities.

Plain `argmax` is the wrong decision rule for melanoma screening. A lesion
scored 45% melanoma against 50% nevus is called a nevus, and the two errors are
not symmetric: a missed melanoma delays treatment of a cancer with a sharp
survival gradient in time to diagnosis, while a false alarm costs a biopsy.

This module provides two rules that bias the decision toward the dangerous
classes. They are different rules, not two implementations of one, and they
disagree on roughly one lesion in six under the thresholds used here.

Rule A -- priority-ordered promotion (default)
----------------------------------------------
:func:`apply_priority_thresholds`. Walk the promotion list in order; the
**first** class whose probability clears its bar has its score set to 1.0, and
the walk stops. Then walk the demotion list; the first class below its bar is
set to 0.0.

The ordering carries the clinical priority: melanoma is checked before basal
cell carcinoma because it matters more. At most one class is promoted.

This is the rule every thresholded figure in this project comes from.

A tempting variant, not implemented
-----------------------------------
Promoting *every* class that clears its bar, rather than stopping at the first,
looks like a simplification. It is not. The ordering becomes decorative, and
when two classes are both promoted to 1.0 the tie falls to `idxmax`, which
takes the first column, and columns are ordered alphabetically by class name.
So `bcc` would beat `mel` because "b" precedes "m": an arbitrary rule standing
where a clinical one belongs. That is why Rule A stops at the first match.

Rule C -- cost-sensitive reweighting
-------------------------------------
:func:`apply_cost_sensitive_weights`. Divide each probability by its threshold
and take the argmax: ``argmax_c (p_c / t_c)``. This is the standard
cost-sensitive decision rule. It needs no tie-break, it degenerates to plain
argmax when all thresholds are equal, and it never leaves a row unnormalised.

It is *not* a reformulation of Rule A. Rule A is a conditional gate that does
nothing until a probability crosses a bar; Rule C is an unconditional
reweighting that shifts every boundary. With a melanoma threshold of 0.4 the
effective condition becomes ``p_mel > 0.4 * p_nv`` rather than
``p_mel > 0.4``, which is far weaker, so Rule C calls melanoma noticeably more
often. Rule C also cannot express the "demote only if below" condition, since a
weight applies unconditionally.

Which to use
------------
Rule A is the default. It is easy to reason about, and its priority ordering
states which class matters most rather than leaving that implicit in a set of
weights. Rule C is the more standard method and is available as an alternative;
switching would change every reported figure, so the two are kept separate
rather than swapped.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ham10000.models.aggregation import PROBABILITY_PREFIX

__all__ = [
    "apply_cost_sensitive_weights",
    "apply_priority_thresholds",
]

#: An ordered list of (class name, threshold) pairs. A sequence of tuples
#: rather than a dict, because the ordering is load-bearing and a plain dict
#: gives no signal that it is.
ThresholdList = Sequence[tuple[str, float]]


def _probability_block(
    frame: pd.DataFrame, prefix: str
) -> tuple[list[str], np.ndarray]:
    """Return the probability column names and their values as a float array."""
    columns = [c for c in frame.columns if c.startswith(prefix)]
    if not columns:
        raise KeyError(
            f"No columns with prefix {prefix!r}. Present: {sorted(frame.columns)}."
        )
    return columns, frame[columns].to_numpy(dtype=float)


def _first_crossing(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row, the index of the first True in `mask`, and whether any."""
    any_true = mask.any(axis=1)
    # argmax on a boolean array returns the first True, or 0 when all False --
    # hence the companion mask.
    first = mask.argmax(axis=1)
    return first, any_true


def apply_priority_thresholds(
    frame: pd.DataFrame,
    *,
    promote: ThresholdList | None = None,
    demote: ThresholdList | None = None,
    prefix: str = PROBABILITY_PREFIX,
) -> pd.DataFrame:
    """Apply Rule A: promote the first class clearing its bar, then stop.

    Parameters
    ----------
    frame:
        Probability table. Not modified.
    promote:
        Ordered `(class, threshold)` pairs. The first class whose probability
        *exceeds* its threshold is set to 1.0. Order is clinical priority.
    demote:
        Ordered `(class, threshold)` pairs. The first class whose probability
        falls *below* its threshold is set to 0.0. Applied after promotion, as
        after promotion.
    prefix:
        Probability column prefix.

    Returns
    -------
    pd.DataFrame
        A copy with adjusted values in the probability columns.

    Warnings
    --------
    The returned values are **scores, not probabilities**: overwriting an entry
    with 1.0 or 0.0 leaves the row no longer summing to one. They are fit for
    `argmax` and unfit for anything that assumes a distribution, such as
    ROC-AUC or a calibration plot. Score the *unadjusted* probabilities for
    those, using the unadjusted probabilities instead.

    Notes
    -----
    The caller's frame is not modified, and the implementation is vectorised
    rather than applying a Python function per row.

    Examples
    --------
    Melanoma clears its bar and is promoted, overtaking a larger nevus score:

    >>> frame = pd.DataFrame({"prob_mel": [0.45], "prob_nv": [0.50]})
    >>> adjusted = apply_priority_thresholds(frame, promote=[("mel", 0.4)])
    >>> adjusted["prob_mel"].tolist()
    [1.0]

    Only the first qualifying class is promoted -- the ordering is the priority:

    >>> frame = pd.DataFrame({"prob_mel": [0.42], "prob_bcc": [0.45]})
    >>> adjusted = apply_priority_thresholds(
    ...     frame, promote=[("mel", 0.4), ("bcc", 0.4)]
    ... )
    >>> adjusted[["prob_mel", "prob_bcc"]].iloc[0].tolist()
    [1.0, 0.45]

    Below the bar, nothing happens at all:

    >>> frame = pd.DataFrame({"prob_mel": [0.35], "prob_nv": [0.65]})
    >>> apply_priority_thresholds(frame, promote=[("mel", 0.4)])["prob_mel"].tolist()
    [0.35]

    Demotion pushes a class out of contention:

    >>> frame = pd.DataFrame({"prob_mel": [0.45], "prob_nv": [0.55]})
    >>> adjusted = apply_priority_thresholds(frame, demote=[("nv", 0.6)])
    >>> adjusted["prob_nv"].tolist()
    [0.0]
    """
    columns, values = _probability_block(frame, prefix)
    adjusted = values.copy()

    for rules, comparison, replacement in (
        (promote, np.greater, 1.0),
        (demote, np.less, 0.0),
    ):
        if not rules:
            continue

        indices = []
        bars = []
        for name, threshold in rules:
            column = f"{prefix}{name}"
            if column in columns:
                indices.append(columns.index(column))
                bars.append(threshold)
        if not indices:
            continue

        candidate = adjusted[:, indices]
        crossings = comparison(candidate, np.asarray(bars))
        first, any_true = _first_crossing(crossings)

        rows = np.flatnonzero(any_true)
        chosen = np.asarray(indices)[first[rows]]
        adjusted[rows, chosen] = replacement

    output = frame.copy()
    output[columns] = adjusted
    return output


def apply_cost_sensitive_weights(
    frame: pd.DataFrame,
    *,
    thresholds: ThresholdList | None = None,
    prefix: str = PROBABILITY_PREFIX,
) -> pd.DataFrame:
    """Apply Rule C: divide each probability by its threshold.

    Parameters
    ----------
    frame:
        Probability table. Not modified.
    thresholds:
        `(class, threshold)` pairs. A class with threshold `t` is boosted by
        `1/t`, so a smaller `t` means a stronger preference. Classes not listed
        are unchanged, equivalent to `t = 1`. Order is irrelevant here, unlike
        Rule A.
    prefix:
        Probability column prefix.

    Returns
    -------
    pd.DataFrame
        A copy with reweighted values.

    Raises
    ------
    ValueError
        On a non-positive threshold.

    Warnings
    --------
    As with Rule A the result is scores, not probabilities: dividing by
    per-class constants does not preserve the sum. Unlike Rule A the *ranking*
    is a monotone transform of the input, so the rule is well behaved.

    Notes
    -----
    To express a clinical priority under this rule, encode it in the threshold
    *values* rather than the ordering: `t_mel = 0.3` outranks `t_bcc = 0.4`.
    There is no equivalent of Rule A's conditional demotion, because a weight
    applies unconditionally.

    Examples
    --------
    A 2.5x boost overturns a larger raw score:

    >>> frame = pd.DataFrame({"prob_mel": [0.35], "prob_nv": [0.65]})
    >>> adjusted = apply_cost_sensitive_weights(frame, thresholds=[("mel", 0.4)])
    >>> bool(adjusted["prob_mel"].iloc[0] > adjusted["prob_nv"].iloc[0])
    True

    Equal thresholds reduce to plain argmax:

    >>> frame = pd.DataFrame({"prob_mel": [0.3], "prob_nv": [0.7]})
    >>> adjusted = apply_cost_sensitive_weights(
    ...     frame, thresholds=[("mel", 0.5), ("nv", 0.5)]
    ... )
    >>> bool(adjusted["prob_mel"].iloc[0] < adjusted["prob_nv"].iloc[0])
    True

    Unlike Rule A, the rule acts even when nothing crosses a bar:

    >>> frame = pd.DataFrame({"prob_mel": [0.35], "prob_nv": [0.65]})
    >>> a = apply_priority_thresholds(frame, promote=[("mel", 0.4)])
    >>> c = apply_cost_sensitive_weights(frame, thresholds=[("mel", 0.4)])
    >>> float(a["prob_mel"].iloc[0]), round(float(c["prob_mel"].iloc[0]), 4)
    (0.35, 0.875)
    """
    columns, values = _probability_block(frame, prefix)
    weights = np.ones(len(columns))

    for name, threshold in thresholds or ():
        if threshold <= 0:
            raise ValueError(
                f"Threshold for {name!r} must be positive, got {threshold}."
            )
        column = f"{prefix}{name}"
        if column in columns:
            weights[columns.index(column)] = 1.0 / threshold

    output = frame.copy()
    output[columns] = values * weights
    return output
