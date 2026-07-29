"""Batched inference producing per-image class probabilities.

Two properties are worth stating, because getting either wrong fails silently.

**Failures raise.** Catching them and returning the input frame would give the
caller a DataFrame with no probability columns, or worse, one still carrying
columns from an earlier call, and an evaluation would then run against stale
numbers.

**Probabilities are aligned positionally, and the alignment is verified.**
``DataLoader(shuffle=False)`` yields rows in frame order, so the *i*-th
prediction belongs to the *i*-th row. Merging on the index instead would be
correct only when the frame's index happens to run ``0..n-1``, and would
silently attach probabilities to the wrong images for any filtered or sliced
frame. Joining on ``image_id`` is not an alternative: the expanded validation
set repeats each lesion deliberately, so an identifier join blows up
combinatorially. The identifiers returned by the loader are therefore checked
against the frame rather than assumed to match.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from ham10000.data.labels import LabelScheme
from ham10000.models.dataset import LesionImageDataset

__all__ = ["predict_probabilities", "select_device"]


def select_device(prefer: str | None = None) -> torch.device:
    """Return the compute device, preferring CUDA then MPS then CPU.

    Parameters
    ----------
    prefer:
        Explicit device string. Bypasses detection when given.

    Returns
    -------
    torch.device

    Notes
    -----
    Apple Silicon via `mps` is included, since much of this project is meant
    to run on a laptop.

    Examples
    --------
    >>> select_device("cpu")
    device(type='cpu')
    >>> isinstance(select_device(), torch.device)
    True
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_probabilities(
    frame: pd.DataFrame,
    model: nn.Module,
    scheme: LabelScheme,
    *,
    image_dir: Path,
    transform: Callable[[Image.Image], torch.Tensor],
    batch_size: int = 32,
    device: torch.device | str | None = None,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Score every row of `frame` and append per-class probability columns.

    Parameters
    ----------
    frame:
        Metadata with an `image_id` column. Not modified.
    model:
        A model whose head width matches `scheme.n_classes`. Set to eval mode
        internally; the caller's training/eval state is not restored.
    scheme:
        Label scheme, which fixes both the column names and their order.
    image_dir:
        Directory containing `<image_id>.jpg`.
    transform:
        Transform applied to each image. If this is stochastic, repeated rows
        yield different views -- which is the mechanism behind the expanded
        validation set.
    batch_size, num_workers:
        DataLoader settings.
    device:
        Compute device. Detected when omitted.

    Returns
    -------
    pd.DataFrame
        A copy of `frame` with one `prob_<class>` column per class, carrying
        the original index.

    Raises
    ------
    ValueError
        If the model's output width disagrees with the scheme, or if the
        identifiers returned by the loader do not match the input frame.

    Notes
    -----
    Probabilities are softmax over the model's logits, so each row sums to one
    across classes.

    The output-width check matters because the failure is silent otherwise. A
    checkpoint trained under a five-class scheme and scored under a binary one
    would yield a two-column probability table built from the first two of five
    logits: numbers that look entirely plausible and are meaningless.
    """
    # An explicit torch.device must be honoured, not discarded and re-detected.
    resolved = torch.device(device) if device is not None else select_device()

    dataset = LesionImageDataset(
        frame,
        n_classes=scheme.n_classes,
        image_dir=image_dir,
        transform=transform,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.eval()
    model.to(resolved)

    chunks: list[torch.Tensor] = []
    identifiers: list[str] = []

    for images, _, image_ids in loader:
        logits = model(images.to(resolved))
        if logits.shape[1] != scheme.n_classes:
            raise ValueError(
                f"Model produces {logits.shape[1]} outputs but the label scheme "
                f"has {scheme.n_classes} classes. The checkpoint was probably "
                "trained under a different scheme."
            )
        chunks.append(torch.softmax(logits, dim=1).cpu())
        identifiers.extend(image_ids)

    probabilities = torch.cat(chunks).numpy()

    # Positional alignment is correct because shuffle=False preserves frame
    # order, but verify rather than assume -- misalignment here is invisible.
    if identifiers != frame["image_id"].astype(str).tolist():
        raise ValueError(
            "Image identifiers returned by the DataLoader do not match the "
            "input frame. Probabilities cannot be aligned safely."
        )

    output = frame.copy()
    for index, column in enumerate(scheme.probability_columns):
        output[column] = probabilities[:, index]
    return output
