"""PyTorch dataset over lesion images.

Replaces `multiclass_models.image_n_label`, whose name reflected its return
value rather than its role, and which carried a comment thread recording an
unresolved `ValueError: cannot insert level_0, already exists` from repeated
`reset_index` calls. Indexing here is positional via `iloc`, so no index
manipulation is needed and the failure mode cannot recur.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["LesionImageDataset"]


class LesionImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Dataset yielding `(image, label, image_id)` triples.

    Parameters
    ----------
    frame:
        Metadata with an `image_id` column and, for supervised use, a `label`
        column of integer class indices. Only those two columns are retained.
    n_classes:
        Number of classes, determining the width of the one-hot label vector.
    image_dir:
        Directory containing `<image_id>.jpg`.
    transform:
        Torchvision transform converting the PIL image to a tensor. Required:
        a DataLoader cannot collate PIL images, so omitting it is not a usable
        configuration: it defers the failure to collation, where the traceback
        points at the loader rather than the dataset.

    Notes
    -----
    Labels are returned one-hot as float. `nn.CrossEntropyLoss` accepts either
    class indices or a probability distribution over classes, and the latter
    is used here.

    The `image_id` is returned alongside so that predictions can be joined back
    to lesions. Without it, aggregating image-level predictions up to a lesion
    verdict would be impossible.

    Raises
    ------
    KeyError
        If `image_id` is absent.
    FileNotFoundError
        On a missing image file, naming the path. Letting `PIL.Image.open`
        raise from inside a DataLoader worker surfaces as an opaque worker
        crash instead.

    Examples
    --------
    >>> frame = pd.DataFrame({"image_id": ["I1", "I2"], "label": [0, 1]})
    >>> from torchvision import transforms
    >>> to_tensor = transforms.ToTensor()
    >>> dataset = LesionImageDataset(
    ...     frame, n_classes=2, image_dir=Path("."), transform=to_tensor
    ... )
    >>> len(dataset)
    2
    >>> dataset.has_labels
    True

    Without a `label` column the dataset is still usable for inference:

    >>> unlabelled = LesionImageDataset(
    ...     pd.DataFrame({"image_id": ["I1"]}),
    ...     n_classes=2,
    ...     image_dir=Path("."),
    ...     transform=to_tensor,
    ... )
    >>> unlabelled.has_labels
    False
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        n_classes: int,
        image_dir: Path,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        if "image_id" not in frame.columns:
            raise KeyError(
                f"Frame must contain an 'image_id' column. "
                f"Present: {sorted(frame.columns)}."
            )

        self.has_labels = "label" in frame.columns
        columns = ["image_id", "label"] if self.has_labels else ["image_id"]
        # Positional access via iloc throughout, so the caller's index is
        # irrelevant and never needs resetting.
        self._frame = frame[columns].reset_index(drop=True)
        self.n_classes = n_classes
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of rows, which counts images rather than lesions."""
        return len(self._frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Load one image with its one-hot label and identifier."""
        row = self._frame.iloc[index]
        image_id = str(row["image_id"])

        path = self.image_dir / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found: {path} (row {index}, image_id {image_id!r})."
            )

        with Image.open(path) as handle:
            image = handle.convert("RGB")
            tensor = self.transform(image)

        if self.has_labels:
            label = torch.zeros(self.n_classes)
            label[int(row["label"])] = 1.0
        else:
            label = torch.empty(0)

        return tensor, label, image_id
