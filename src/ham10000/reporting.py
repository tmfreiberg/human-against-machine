"""Presentation helpers for notebook and report output.

Measurement lives in :mod:`ham10000.evaluation`; presentation lives here.
Keeping them apart matters because a metric should be computable in a test
without producing output, and a table should be formattable without recomputing
anything.

Everything here returns a value rather than printing it, which keeps it usable
from a script, testable, and able to feed a rendered document.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ham10000.evaluation.metrics import ClassificationReport, confusion_frame

__all__ = ["comparison_table", "confusion_with_recall", "format_report"]


def format_report(report: ClassificationReport, name: str = "model") -> pd.DataFrame:
    """Render one report as a single-row table.

    Examples
    --------
    >>> import numpy as np
    >>> from ham10000.evaluation.metrics import evaluate
    >>> table = format_report(evaluate(np.array([0, 1]), np.array([0, 1])), "demo")
    >>> table.index.tolist()
    ['demo']
    >>> float(table.loc["demo", "BACC"])
    1.0
    """
    return report.to_series().to_frame(name).T


def comparison_table(
    reports: Mapping[str, ClassificationReport],
    *,
    sort_by: str = "BACC",
) -> pd.DataFrame:
    """Stack several reports into one table, best first.

    This is the artefact that replaces `print_model_evaluation`, which printed
    each model's metrics in sequence and left the reader to compare by eye
    across screens of output.

    Parameters
    ----------
    reports:
        Model name to report.
    sort_by:
        Column to sort descending. Balanced accuracy by default, since plain
        accuracy is misleading under this dataset's imbalance.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If `reports` is empty, or `sort_by` is not a reported metric.

    Examples
    --------
    >>> import numpy as np
    >>> from ham10000.evaluation.metrics import evaluate
    >>> good = evaluate(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    >>> poor = evaluate(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]))
    >>> comparison_table({"poor": poor, "good": good}).index.tolist()
    ['good', 'poor']
    """
    if not reports:
        raise ValueError("No reports to compare.")

    table = pd.concat(
        [report.to_series().to_frame(name).T for name, report in reports.items()]
    )
    if sort_by not in table.columns:
        raise ValueError(
            f"Unknown metric {sort_by!r}. Available: {sorted(table.columns)}."
        )
    return table.sort_values(sort_by, ascending=False)


def confusion_with_recall(
    target: Sequence[int] | np.ndarray,
    prediction: Sequence[int] | np.ndarray,
    label_codes: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Confusion matrix with a per-class recall column appended.

    Recall belongs beside the matrix rather than in a separate table: reading a
    row and its recall together is how you see that a class is being missed,
    and melanoma's row is the one that matters clinically.

    Examples
    --------
    >>> import numpy as np
    >>> table = confusion_with_recall(
    ...     np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), {0: "nv", 1: "mel"}
    ... )
    >>> table["recall"].tolist()
    [1.0, 0.5]

    The support column makes a small denominator visible, so a recall of 1.0
    over two lesions is not mistaken for a strong result:

    >>> table["support"].tolist()
    [2, 2]
    """
    matrix = confusion_frame(np.asarray(target), np.asarray(prediction), label_codes)
    support = matrix.sum(axis=1)
    correct = pd.Series(np.diag(matrix), index=matrix.index)

    output = matrix.copy()
    output["support"] = support
    # numpy division so a class with no support yields nan, not pd.NA -- the
    # latter cannot be cast to float and raised on any unrepresented class.
    output["recall"] = np.round(
        np.divide(
            correct.to_numpy(dtype=float),
            support.to_numpy(dtype=float),
            out=np.full(len(support), np.nan),
            where=support.to_numpy() > 0,
        ),
        4,
    )
    return output
