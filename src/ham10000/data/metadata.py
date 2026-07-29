"""Loading and filtering of the HAM10000 metadata table.

Replaces the loading portion of `processing.process.__init__` and the
duplicate implementation in `explore.data_handling.process_metadata_csv`.

Failures raise. A missing file or a malformed table stops the pipeline at the
point of the problem, rather than printing a message and leaving a later
statement to fail somewhere unrelated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

__all__ = ["REQUIRED_COLUMNS", "exclude", "load_metadata", "restrict"]

#: Columns the pipeline depends on. `dx_type`, `age`, `sex` and `localization`
#: are present in HAM10000 but not required by any modelling step.
REQUIRED_COLUMNS = ("lesion_id", "image_id", "dx")


def load_metadata(path: Path | str, *, add_num_images: bool = True) -> pd.DataFrame:
    r"""Read the metadata CSV and optionally annotate lesion multiplicity.

    Parameters
    ----------
    path:
        Path to `metadata.csv`.
    add_num_images:
        Insert a `num_images` column giving, for each row, how many images
        exist of that row's lesion. Placed immediately right of `lesion_id`,
        which keeps the lesion's identity and its multiplicity adjacent.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a required column is absent, or if any `image_id` repeats.

    Notes
    -----
    The duplicate-`image_id` check is new. Nothing downstream would fail
    loudly on a repeated identifier, but a repeat would silently corrupt the
    split -- the same image could be designated for a lesion twice.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     csv = Path(tmp) / "metadata.csv"
    ...     _ = csv.write_text(
    ...         "lesion_id,image_id,dx\nL1,I1,mel\nL1,I2,mel\nL2,I3,nv\n"
    ...     )
    ...     frame = load_metadata(csv)
    ...     list(frame.columns)
    ['lesion_id', 'num_images', 'image_id', 'dx']

    `num_images` counts images of the lesion, not rows in the table:

    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     csv = Path(tmp) / "metadata.csv"
    ...     _ = csv.write_text(
    ...         "lesion_id,image_id,dx\nL1,I1,mel\nL1,I2,mel\nL2,I3,nv\n"
    ...     )
    ...     load_metadata(csv)["num_images"].tolist()
    [2, 2, 1]
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    frame = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}. "
            f"Present: {sorted(frame.columns)}."
        )

    duplicated = frame["image_id"].duplicated()
    if duplicated.any():
        examples = frame.loc[duplicated, "image_id"].unique()[:5].tolist()
        raise ValueError(
            f"{path} contains {int(duplicated.sum())} duplicate image_id "
            f"value(s), e.g. {examples}. Each image must appear once."
        )

    if add_num_images:
        counts = frame["lesion_id"].map(frame["lesion_id"].value_counts())
        frame.insert(1, "num_images", counts)

    return frame


def restrict(
    frame: pd.DataFrame, criteria: Mapping[str, Sequence[object]]
) -> pd.DataFrame:
    """Keep only rows matching every criterion.

    Parameters
    ----------
    frame:
        Metadata. Not modified.
    criteria:
        Column name to permitted values. Columns absent from the frame are
        ignored.

    Returns
    -------
    pd.DataFrame
        A copy, not a view. Returning a view and then inserting columns into it
        is the classic source of `SettingWithCopyWarning`, and of edits that
        silently do not stick.

    Examples
    --------
    >>> frame = pd.DataFrame({"dx": ["mel", "nv", "bkl"], "sex": ["m", "f", "m"]})
    >>> restrict(frame, {"dx": ["mel", "nv"]})["dx"].tolist()
    ['mel', 'nv']

    Criteria combine conjunctively:

    >>> restrict(frame, {"dx": ["mel", "nv"], "sex": ["m"]})["dx"].tolist()
    ['mel']

    An empty specification is a no-op:

    >>> len(restrict(frame, {}))
    3
    """
    mask = pd.Series(True, index=frame.index)
    for column, values in criteria.items():
        if column in frame.columns:
            mask &= frame[column].isin(list(values))
    return frame[mask].copy()


def exclude(
    frame: pd.DataFrame, criteria: Mapping[str, Sequence[object]]
) -> pd.DataFrame:
    """Drop rows matching any criterion.

    Parameters
    ----------
    frame:
        Metadata. Not modified.
    criteria:
        Column name to values whose presence causes a row to be dropped.

    Returns
    -------
    pd.DataFrame
        A copy.

    Notes
    -----
    Rows are dropped if they match *any* criterion, not only rows matching
    every one. The distinction is invisible with a single criterion and
    inverts the meaning with two, so it is worth stating explicitly.

    Examples
    --------
    >>> frame = pd.DataFrame({"dx": ["mel", "nv", "bkl"], "sex": ["m", "f", "m"]})
    >>> exclude(frame, {"dx": ["mel"]})["dx"].tolist()
    ['nv', 'bkl']

    Two criteria drop rows matching either, not only rows matching both:

    >>> exclude(frame, {"dx": ["mel"], "sex": ["f"]})["dx"].tolist()
    ['bkl']
    """
    mask = pd.Series(False, index=frame.index)
    for column, values in criteria.items():
        if column in frame.columns:
            mask |= frame[column].isin(list(values))
    return frame[~mask].copy()
