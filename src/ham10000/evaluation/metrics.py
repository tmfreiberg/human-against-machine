"""Classification metrics for imbalanced, sensitivity-critical problems.

Why balanced accuracy is the headline
-------------------------------------
Nevi are 67% of HAM10000. A classifier that answers "nevus" unconditionally
scores 0.67 plain accuracy while being clinically worthless. Balanced accuracy
-- the mean of per-class recall -- scores that same classifier at
``1/n_classes``, which is the honest number. Every summary in this project
leads with it.

Why F-beta with beta > 1
------------------------
Missing a melanoma and over-calling a nevus are not symmetric errors. The first
delays treatment of a cancer with a sharp survival gradient in time to
diagnosis; the second costs a biopsy. `F2` weights recall four times as heavily
as precision (`beta**2`), which is the direction a screening tool should err
in. `F1/2` is reported alongside for contrast, not because it is appropriate.

Failures raise
--------------
A metric wrapped in a bare `try/except` that assigns `nan` on failure is
indistinguishable, in the results table, from a metric that is legitimately
undefined for its input. Failures therefore propagate here. The two cases where
a metric genuinely is undefined, ROC-AUC with fewer than two classes present
and Matthews correlation with one, are detected explicitly rather than caught.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "ClassificationReport",
    "evaluate",
    "per_class_recall",
    "weighted_fbeta",
]


def weighted_fbeta(
    precision: np.ndarray,
    recall: np.ndarray,
    *,
    beta: float = 1.0,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted average of per-class F-beta scores.

    Parameters
    ----------
    precision, recall:
        Per-class values, same length.
    beta:
        Recall is weighted `beta**2` times as heavily as precision.
    weights:
        Per-class weights. Uniform when omitted. Inverse class frequency is the
        usual choice, which upweights the rare classes that matter clinically.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        On mismatched lengths, non-positive beta, or weights summing to zero.

    Notes
    -----
    A class with zero precision and zero recall contributes 0 to the average,
    which is what sklearn does and what the metric means.

    Beware a guard of the form ``if precision.all() == 0``. `ndarray.all()`
    returns a boolean, so that expression is true exactly when *at least one*
    class has zero precision, not when all do. On this dataset a rare class
    scoring zero precision is unremarkable, so such a guard would turn the
    whole weighted F-score into `nan` whenever one class was degenerate.

    Examples
    --------
    >>> precision = np.array([0.5, 1.0])
    >>> recall = np.array([1.0, 0.5])
    >>> round(weighted_fbeta(precision, recall, beta=1.0), 4)
    0.6667

    A degenerate class contributes zero rather than nan-ing the result:

    >>> round(weighted_fbeta(np.array([0.0, 1.0]), np.array([0.0, 1.0])), 4)
    0.5

    Beta above 1 rewards recall:

    >>> low_recall = weighted_fbeta(np.array([0.9]), np.array([0.3]), beta=2.0)
    >>> low_precision = weighted_fbeta(np.array([0.3]), np.array([0.9]), beta=2.0)
    >>> low_precision > low_recall
    True
    """
    precision = np.ravel(np.asarray(precision, dtype=float))
    recall = np.ravel(np.asarray(recall, dtype=float))

    if precision.shape != recall.shape:
        raise ValueError(
            f"precision and recall must be the same length, got "
            f"{precision.shape} and {recall.shape}."
        )
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}.")

    if weights is None:
        weights = np.ones_like(precision)
    weights = np.ravel(np.asarray(weights, dtype=float))
    if weights.shape != precision.shape:
        raise ValueError(
            f"weights must be the same length as precision, got "
            f"{weights.shape} and {precision.shape}."
        )
    total_weight = weights.sum()
    if total_weight == 0:
        raise ValueError("weights sum to zero.")

    denominator = (beta**2) * precision + recall
    scores = np.divide(
        (1 + beta**2) * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator > 0,
    )
    return float(np.sum(weights * scores) / total_weight)


def per_class_recall(
    target: np.ndarray,
    prediction: np.ndarray,
    label_codes: Mapping[int, str] | None = None,
) -> pd.Series:
    """Recall for each class, i.e. sensitivity.

    The single most important table for this problem: balanced accuracy is
    their mean, and melanoma recall is the number a clinician would ask for
    first.

    Examples
    --------
    >>> target = np.array([0, 0, 1, 1])
    >>> prediction = np.array([0, 0, 0, 1])
    >>> per_class_recall(target, prediction, {0: "nv", 1: "mel"}).to_dict()
    {'nv': 1.0, 'mel': 0.5}
    """
    labels = sorted(set(np.unique(target)) | set(np.unique(prediction)))
    scores = recall_score(
        target, prediction, labels=labels, average=None, zero_division=0
    )
    index = [label_codes[label] if label_codes else label for label in labels]
    return pd.Series(scores, index=index, name="recall")


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Metrics for one set of predictions.

    Attributes
    ----------
    accuracy:
        Plain accuracy. Reported for completeness and misleading on its own.
    balanced_accuracy:
        Mean per-class recall. The headline metric.
    precision, recall:
        Macro averages.
    f_half, f1, f2:
        Macro F-beta at beta = 0.5, 1, 2.
    mcc:
        Matthews correlation, which is informative under imbalance.
    roc_auc_macro, roc_auc_weighted, roc_auc_balanced:
        Threshold-free ranking quality. `nan` when probabilities are not
        supplied. `roc_auc_balanced` uses inverse-frequency sample weights.
    n_samples:
        Number of rows scored.

    Examples
    --------
    >>> report = evaluate(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]))
    >>> report.balanced_accuracy
    0.75
    >>> report.n_samples
    4
    """

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f_half: float
    f1: float
    f2: float
    mcc: float
    roc_auc_macro: float
    roc_auc_weighted: float
    roc_auc_balanced: float
    n_samples: int

    def to_series(self) -> pd.Series:
        """Return the report as a Series, for stacking into a results table.

        Examples
        --------
        >>> report = evaluate(np.array([0, 1]), np.array([0, 1]))
        >>> float(report.to_series()["BACC"])
        1.0
        """
        return pd.Series(
            {
                "ACC": self.accuracy,
                "BACC": self.balanced_accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "F1/2": self.f_half,
                "F1": self.f1,
                "F2": self.f2,
                "MCC": self.mcc,
                "ROC-AUC mac": self.roc_auc_macro,
                "ROC-AUC wt": self.roc_auc_weighted,
                "ROC-AUC wt*": self.roc_auc_balanced,
                "n": self.n_samples,
            }
        )


def _roc_auc_scores(
    target: np.ndarray, probabilities: np.ndarray | None
) -> tuple[float, float, float]:
    """Return macro, weighted, and inverse-frequency-weighted ROC-AUC.

    ROC-AUC is genuinely undefined when only one class is present, so that case
    is detected and returns `nan` rather than being caught from an exception.
    """
    nan_triple = (float("nan"),) * 3
    if probabilities is None:
        return nan_triple

    classes = np.unique(target)
    if classes.size < 2:
        return nan_triple

    scores = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
    multi_class = {} if probabilities.shape[1] == 2 else {"multi_class": "ovr"}

    counts = np.bincount(target, minlength=int(target.max()) + 1)
    with np.errstate(divide="ignore"):
        inverse = np.where(counts > 0, 1 / np.maximum(counts, 1), 0.0)

    return (
        float(roc_auc_score(target, scores, average="macro", **multi_class)),
        float(roc_auc_score(target, scores, average="weighted", **multi_class)),
        float(
            roc_auc_score(
                target,
                scores,
                average="weighted",
                sample_weight=inverse[target],
                **multi_class,
            )
        ),
    )


def evaluate(
    target: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> ClassificationReport:
    """Compute the full metric set for one set of predictions.

    Parameters
    ----------
    target:
        True integer labels.
    prediction:
        Predicted integer labels.
    probabilities:
        Optional `(n_samples, n_classes)` probability matrix, enabling the
        ROC-AUC metrics.

    Returns
    -------
    ClassificationReport

    Raises
    ------
    ValueError
        On length mismatches or empty input, rather than returning a table of
        `nan`.

    Examples
    --------
    >>> target = np.array([0, 0, 0, 1])
    >>> report = evaluate(target, np.array([0, 0, 0, 0]))

    Plain accuracy flatters the always-majority classifier:

    >>> report.accuracy
    0.75

    Balanced accuracy does not:

    >>> report.balanced_accuracy
    0.5
    """
    target = np.asarray(target)
    prediction = np.asarray(prediction)

    if target.shape != prediction.shape:
        raise ValueError(
            f"target and prediction must be the same length, got "
            f"{target.shape} and {prediction.shape}."
        )
    if target.size == 0:
        raise ValueError("Cannot evaluate an empty set of predictions.")
    if probabilities is not None and len(probabilities) != len(target):
        raise ValueError(
            f"probabilities has {len(probabilities)} rows but target has {len(target)}."
        )

    macro = {"average": "macro", "zero_division": 0}

    # Matthews correlation is undefined when only one class is present:
    # sklearn warns and returns 0.0, which reads as "no better than chance"
    # for what may be a perfect prediction. Report it as undefined instead.
    observed = set(np.unique(target)) | set(np.unique(prediction))
    mcc = (
        float(matthews_corrcoef(target, prediction))
        if len(observed) > 1
        else float("nan")
    )

    roc_macro, roc_weighted, roc_balanced = _roc_auc_scores(target, probabilities)

    with warnings.catch_warnings():
        # A degenerate single-class input makes sklearn warn about confusion
        # matrix shape from inside balanced_accuracy_score. The metrics that
        # are genuinely undefined here (MCC, ROC-AUC) are already reported as
        # nan above, so the warning adds nothing. Scoped to this call only.
        warnings.filterwarnings("ignore", message="A single label was found")
        balanced = float(balanced_accuracy_score(target, prediction))

    return ClassificationReport(
        accuracy=float(accuracy_score(target, prediction)),
        balanced_accuracy=balanced,
        precision=float(precision_score(target, prediction, **macro)),
        recall=float(recall_score(target, prediction, **macro)),
        f_half=float(fbeta_score(target, prediction, beta=0.5, **macro)),
        f1=float(fbeta_score(target, prediction, beta=1.0, **macro)),
        f2=float(fbeta_score(target, prediction, beta=2.0, **macro)),
        mcc=mcc,
        roc_auc_macro=roc_macro,
        roc_auc_weighted=roc_weighted,
        roc_auc_balanced=roc_balanced,
        n_samples=int(target.size),
    )


def confusion_frame(
    target: np.ndarray,
    prediction: np.ndarray,
    label_codes: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame, rows true and columns predicted.

    Every class in `label_codes` appears, whether or not it was predicted. That
    avoids reconstructing missing rows and columns after the fact from the
    labels that happen to be present. Such reconstruction typically uses
    `isinstance(x, int)`, which is `False` for `np.int64` and so finds no
    integer labels at all.

    Examples
    --------
    >>> target = np.array([0, 0, 1])
    >>> prediction = np.array([0, 0, 0])
    >>> confusion_frame(target, prediction, {0: "nv", 1: "mel"})
    ... # doctest: +NORMALIZE_WHITESPACE
    pred  nv  mel
    true
    nv     2    0
    mel    1    0
    """
    if label_codes is not None:
        labels = sorted(label_codes)
        names = [label_codes[label] for label in labels]
    else:
        labels = sorted(set(np.unique(target)) | set(np.unique(prediction)))
        names = [str(label) for label in labels]

    matrix = confusion_matrix(target, prediction, labels=labels)
    frame = pd.DataFrame(matrix, index=names, columns=names)
    frame.index.name = "true"
    frame.columns.name = "pred"
    return frame
