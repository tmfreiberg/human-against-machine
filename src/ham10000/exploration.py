"""Descriptive statistics and image grids for dataset exploration.

Loading and splitting live in :mod:`ham10000.data`; this module holds only what
is specific to looking at the data. Keeping a second implementation of either
here is how the two drift apart.

`matplotlib` is imported inside the plotting functions rather than at module
scope, so importing this module costs nothing and works without the `analysis`
extra installed. Only the grid functions need it.

Artefacts worth looking for
---------------------------
The grids exist to make dataset problems visible rather than assumed. HAM10000
images contain dermatologist ink markings, adhesive scale rulers, hair, and
vignetting from the dermatoscope aperture. Several of these correlate with
diagnosis through acquisition practice rather than pathology -- a lesion
someone bothered to mark and measure was a lesion someone was worried about --
so a model can score well by learning the artefact. Viewing many images
per class is the cheapest available check on that.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "describe_image",
    "describe_predictions",
    "diagnosis_grid",
    "frequencies",
    "image_grid",
    "lesion_grid",
]

#: Metadata fields shown beneath each image in a grid.
CAPTION_FIELDS = (
    "lesion_id",
    "num_images",
    "image_id",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
)


def frequencies(
    frame: pd.DataFrame,
    column: str,
    *,
    where: tuple[str, object] | None = None,
) -> pd.DataFrame:
    """Tabulate absolute and relative frequencies of a column's values.

    Parameters
    ----------
    frame:
        Data to summarise.
    column:
        Column whose values are counted.
    where:
        Optional `(column, value)` restriction applied before counting, for
        questions of the form "among melanomas, how are body sites
        distributed?".

    Returns
    -------
    pd.DataFrame
        Two rows, `freq` and `%`, with one column per distinct value.

    Notes
    -----
    Missing values are counted rather than dropped. HAM10000 has 57 rows with
    no recorded age, and a summary that silently omits them misstates the
    denominator.

    The restriction is a named parameter rather than a variadic positional
    one, so it is self-documenting at the call site and checkable by a type
    checker.

    Examples
    --------
    >>> frame = pd.DataFrame(
    ...     {"dx": ["mel", "nv", "nv", "nv"], "sex": ["m", "f", "f", "m"]}
    ... )
    >>> frequencies(frame, "dx")
    dx      nv   mel
    freq   3.0   1.0
    %     75.0  25.0

    Restricted to nevi, how do sexes distribute?

    >>> frequencies(frame, "sex", where=("dx", "nv"))
    sex       f      m
    freq   2.00   1.00
    %     66.67  33.33
    """
    if column not in frame.columns:
        raise KeyError(
            f"Column {column!r} not found. Available: {sorted(frame.columns)}."
        )

    if where is not None:
        filter_column, value = where
        if filter_column not in frame.columns:
            raise KeyError(
                f"Column {filter_column!r} not found. "
                f"Available: {sorted(frame.columns)}."
            )
        frame = frame[frame[filter_column] == value]

    values = frame[column]
    table = pd.concat(
        [
            values.value_counts(dropna=False),
            values.value_counts(normalize=True, dropna=False).mul(100).round(2),
        ],
        axis=1,
        keys=["freq", "%"],
    )
    table.index.names = [column]
    return table.T


def describe_image(image_id: str, frame: pd.DataFrame) -> str:
    """Build a multi-line caption of an image's metadata.

    Fields absent from the frame are skipped rather than raising, since the
    grids are used with partial frames during exploration.

    Parameters
    ----------
    image_id:
        Identifier to look up.
    frame:
        Metadata containing an `image_id` column.

    Returns
    -------
    str
        One `field: value` line per available field, newline-terminated.

    Raises
    ------
    KeyError
        If the image is not in the frame. Swallowing the lookup failure would
        produce an empty caption, indistinguishable from a metadata gap.

    Examples
    --------
    >>> frame = pd.DataFrame(
    ...     {"image_id": ["I1"], "lesion_id": ["L1"], "dx": ["mel"]}
    ... )
    >>> print(describe_image("I1", frame))
    lesion_id: L1
    image_id: I1
    dx: mel
    <BLANKLINE>

    >>> describe_image("nope", frame)
    Traceback (most recent call last):
        ...
    KeyError: "Image 'nope' not found in metadata."
    """
    rows = frame.loc[frame["image_id"] == image_id]
    if rows.empty:
        raise KeyError(f"Image {image_id!r} not found in metadata.")

    row = rows.iloc[0]
    lines = [
        f"{field}: {row[field]}" for field in CAPTION_FIELDS if field in frame.columns
    ]
    return "\n".join(lines) + "\n"


def describe_predictions(
    image_id: str,
    frames: dict[str, pd.DataFrame],
    diagnoses: tuple[str, ...] = ("nv", "bkl", "mel", "bcc", "akiec", "vasc", "df"),
) -> str:
    """Build a caption comparing per-class probabilities across models.

    Parameters
    ----------
    image_id:
        Identifier to look up.
    frames:
        Model name to a probability table with `image_id` and `prob_<dx>`
        columns.
    diagnoses:
        Classes to report, in display order.

    Returns
    -------
    str
        One line per diagnosis, probabilities separated by a bar, with a
        placeholder where a model has no prediction for that image.

    Notes
    -----
    The probability lookup happens once per diagnosis, inside the loop that
    builds the labels. Doing it after the loop would use the loop variable's
    final value and produce a caption with every label but only one
    probability.

    Examples
    --------
    >>> one = pd.DataFrame({"image_id": ["I1"], "prob_mel": [0.25]})
    >>> two = pd.DataFrame({"image_id": ["I1"], "prob_mel": [0.80]})
    >>> print(describe_predictions("I1", {"a": one, "b": two}, ("mel",)))
    mel: 25.00% | 80.00%
    <BLANKLINE>

    A model lacking the class shows a placeholder rather than failing:

    >>> print(describe_predictions("I1", {"a": one}, ("mel", "nv")))
    mel: 25.00%
    nv: ▯
    <BLANKLINE>
    """
    lines = []
    for dx in diagnoses:
        cells = []
        for table in frames.values():
            column = f"prob_{dx}"
            match = table.loc[table["image_id"] == image_id]
            if column in table.columns and not match.empty:
                cells.append(f"{100 * float(match[column].iloc[0]):.2f}%")
            else:
                cells.append("▯")
        lines.append(f"{dx}: {' | '.join(cells)}")
    return "\n".join(lines) + "\n"


def _render_grid(
    image_ids: list[list[str | None]],
    frame: pd.DataFrame,
    image_dir: Path,
    *,
    caption_extra: dict[str, str] | None = None,
    fontsize: int = 8,
    cell_size: tuple[float, float] = (2.4, 2.4),
    captions: bool = True,
) -> Figure:
    """Render a rectangular grid of images with metadata captions.

    Shared by :func:`lesion_grid` and :func:`diagnosis_grid`, which differ only
    in how they choose the identifiers.

    A `None` entry, or a file that cannot be opened, leaves an empty cell. The
    Checking explicitly, rather than wrapping the cell body in a bare
    `except`, keeps genuine errors such as a malformed caption visible.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    n_rows = len(image_ids)
    n_cols = max(len(row) for row in image_ids)

    # Cells are sized in inches. The caption sits below each image, so a cell
    # needs extra height when captions are on -- roughly 0.22 inches per line
    # at the default font size.
    caption_lines = len(CAPTION_FIELDS) if captions else 0
    width, height = cell_size
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        squeeze=False,
        figsize=(width * n_cols, (height + 0.22 * caption_lines) * n_rows),
    )

    for row_index, row in enumerate(image_ids):
        for col_index in range(n_cols):
            axis = axes[row_index][col_index]
            axis.set_xticks([])
            axis.set_yticks([])

            image_id = row[col_index] if col_index < len(row) else None
            if image_id is None:
                axis.axis("off")
                continue

            path = image_dir / f"{image_id}.jpg"
            if not path.is_file():
                axis.axis("off")
                continue

            parts = []
            if captions:
                parts.append(describe_image(image_id, frame))
            if caption_extra and image_id in caption_extra:
                parts.append(caption_extra[image_id])
            if parts:
                axis.set_xlabel("\n".join(parts), fontsize=fontsize, loc="left")
            axis.imshow(Image.open(path))

    return figure


def lesion_grid(
    lesion_ids: Sequence[str],
    frame: pd.DataFrame,
    image_dir: Path,
    *,
    max_rows: int = 10,
    max_cols: int = 6,
    seed: int | None = 0,
    cell_size: tuple[float, float] = (2.4, 2.4),
    captions: bool = True,
) -> Figure:
    """Show one lesion per row, its images across the columns.

    This is the view that makes the case for lesion-level splitting concrete:
    the images in a row are near-duplicates, and separating them across a
    train/validation boundary would be self-evidently wrong.

    Parameters
    ----------
    lesion_ids:
        Lesions to display. Duplicates are removed. If more than `max_rows`
        remain, a random subset is drawn.
    frame:
        Metadata with `lesion_id` and `image_id` columns.
    image_dir:
        Directory containing `<image_id>.jpg` files.
    max_rows, max_cols:
        Grid limits.
    seed:
        Seed for subsetting. `None` uses fresh randomness, in which case a
        notebook re-run will not reproduce its own figures.

    Returns
    -------
    Figure
        Returned rather than shown, so a caller can save it. Calling
        `plt.show()` and returning `None` would make the figure impossible to
        write to disk for a docs build.
    """
    rng = np.random.RandomState(seed)
    unique_lesions = np.unique(np.asarray(lesion_ids))

    if unique_lesions.size > max_rows:
        unique_lesions = rng.choice(unique_lesions, max_rows, replace=False)

    grid: list[list[str | None]] = []
    for lesion in unique_lesions:
        images = frame.loc[frame["lesion_id"] == lesion, "image_id"].tolist()
        if len(images) > max_cols:
            images = rng.choice(images, max_cols, replace=False).tolist()
        grid.append(list(images))

    return _render_grid(grid, frame, image_dir, cell_size=cell_size, captions=captions)


def diagnosis_grid(
    diagnoses: Sequence[str],
    frame: pd.DataFrame,
    image_dir: Path,
    *,
    n_rows: int = 10,
    seed: int | None = 0,
    cell_size: tuple[float, float] = (2.4, 2.4),
    captions: bool = True,
) -> Figure:
    """Show one diagnosis per column, sampled lesions down the rows.

    The view for comparing classes side by side, and the one where shared
    acquisition artefacts within a class become apparent.

    Parameters
    ----------
    diagnoses:
        Classes to display, one per column.
    frame:
        Metadata with `lesion_id`, `image_id`, and `dx` columns.
    image_dir:
        Directory containing `<image_id>.jpg` files.
    n_rows:
        Lesions sampled per class. Classes with fewer lesions than this are
        shown in full rather than raising: dermatofibroma has only 115 lesions
        in the full dataset and far fewer in a filtered subset.
    seed:
        Seed for sampling.

    Returns
    -------
    Figure
    """
    rng = np.random.RandomState(seed)

    columns: dict[str, list[str]] = {}
    for dx in diagnoses:
        lesions = frame.loc[frame["dx"] == dx, "lesion_id"].unique()
        take = min(n_rows, lesions.size)
        chosen = rng.choice(lesions, take, replace=False)
        columns[dx] = [
            str(
                rng.choice(
                    frame.loc[frame["lesion_id"] == lesion, "image_id"].to_numpy()
                )
            )
            for lesion in chosen
        ]

    grid: list[list[str | None]] = [
        [columns[dx][row] if row < len(columns[dx]) else None for dx in diagnoses]
        for row in range(n_rows)
    ]
    return _render_grid(grid, frame, image_dir, cell_size=cell_size, captions=captions)


def image_grid(
    image_ids: Sequence[str],
    frame: pd.DataFrame,
    image_dir: Path,
    *,
    n_cols: int = 4,
    cell_size: tuple[float, float] = (2.4, 2.4),
    captions: bool = False,
    notes: dict[str, str] | None = None,
) -> Figure:
    """Show specific images, named explicitly.

    The counterpart to :func:`lesion_grid` and :func:`diagnosis_grid`, which
    sample. Sampling is right for "what does this class look like in general";
    it is wrong for "here are four images with ink markings", because the
    figure then depends on a seed and silently changes if the sampling code is
    ever touched. For a figure making a specific point, name the images.

    Parameters
    ----------
    image_ids:
        Images to show, in order.
    frame:
        Metadata, used only for captions.
    image_dir:
        Directory containing `<image_id>.jpg`.
    n_cols:
        Images per row.
    cell_size:
        Inches per cell, before caption space.
    captions:
        Whether to print metadata beneath each image. Off by default: when the
        point is a visual one, eight lines of metadata under each thumbnail
        get in the way.
    notes:
        Optional short label per image, e.g. `{"ISIC_0024468": "ink marking"}`.
        Shown beneath the image regardless of `captions`.

    Returns
    -------
    Figure

    Examples
    --------
    A figure about acquisition artefacts, with the examples pinned::

        >>> figure = image_grid(                       # doctest: +SKIP
        ...     ["ISIC_0024468", "ISIC_0025803"],
        ...     metadata,
        ...     settings.images,
        ...     notes={"ISIC_0024468": "ink marking", "ISIC_0025803": "ruler"},
        ... )
    """
    ids = list(image_ids)
    if not ids:
        raise ValueError("No image ids given.")

    n_rows = -(-len(ids) // n_cols)  # ceiling division
    grid: list[list[str | None]] = [
        [
            ids[row * n_cols + col] if row * n_cols + col < len(ids) else None
            for col in range(n_cols)
        ]
        for row in range(n_rows)
    ]
    return _render_grid(
        grid,
        frame,
        image_dir,
        caption_extra=notes,
        cell_size=cell_size,
        captions=captions,
    )
