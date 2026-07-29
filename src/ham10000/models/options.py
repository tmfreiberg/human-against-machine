"""Options describing a model and its training, without depending on either.

Everything here is a value that a configuration file sets: which backbone,
how much of it to train, for how many epochs. These appear in every
`config.yaml` and in an experiment's identity hash, so reading a configuration
must not require PyTorch to be installed. Keeping them apart from the code that
acts on them is what allows `ham10000 configs`, `ham10000 split` and the rest to
run in an environment without the `models` extra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

__all__ = ["Architecture", "FreezeStrategy", "TrainingConfig", "TrainingHistory"]


class Architecture(StrEnum):
    """Supported pretrained backbones."""

    RESNET18 = "resnet18"
    EFFICIENTNET_B0 = "efficientnet_b0"


class FreezeStrategy(StrEnum):
    """How much of the network to train.

    Attributes
    ----------
    ALL:
        Every parameter trainable. Most expressive, most prone to overfitting
        on a dataset this size, and slowest.
    LAST_BLOCK:
        The final convolutional stage plus the head. The usual compromise: the
        early layers hold generic edge and texture filters that transfer from
        ImageNet, while the late layers hold class-specific structure worth
        re-learning.
    HEAD_ONLY:
        Only the replaced classification head, i.e. linear probing on frozen
        features. Fastest, and a genuine baseline: it measures how much of the
        task is solvable from ImageNet features alone.
    """

    ALL = "all"
    LAST_BLOCK = "last_block"
    HEAD_ONLY = "head_only"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters for a fine-tuning run.

    Parameters
    ----------
    epochs:
        Passes over the training set.
    batch_size:
        Images per gradient step.
    learning_rate:
        Adam learning rate. The default of 1e-4 is an order of magnitude below
        Adam's own default, which is appropriate for fine-tuning: a larger rate
        can destroy pretrained features in the first few steps.
    num_workers:
        DataLoader worker processes.
    save_best:
        Save the epoch with the lowest validation loss rather than the last.
        Defaults to `False`; see the module warning.
    seed:
        Seed for torch's RNG, making runs reproducible up to nondeterministic
        GPU kernels.

    Examples
    --------
    >>> TrainingConfig().epochs
    10
    >>> TrainingConfig(epochs=1, batch_size=4).batch_size
    4
    """

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_workers: int = 0
    save_best: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject values that would fail confusingly later."""
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}.")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}."
            )


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch losses from a run.

    Only completed epochs are recorded, so the length of the history is the
    truth about how far a run got. Pre-filling the arrays with a sentinel would
    leave a truncated run writing values indistinguishable from real ones.

    Examples
    --------
    >>> history = TrainingHistory()
    >>> history.record(train=0.9, validation=1.1)
    >>> history.record(train=0.6, validation=0.8)
    >>> history.best_epoch
    1
    >>> len(history)
    2
    """

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of completed epochs."""
        return len(self.train_loss)

    def record(self, *, train: float, validation: float) -> None:
        """Append one epoch's losses."""
        self.train_loss.append(train)
        self.validation_loss.append(validation)

    @property
    def best_epoch(self) -> int:
        """Index of the epoch with the lowest validation loss."""
        if not self.validation_loss:
            raise ValueError("No epochs recorded.")
        return min(
            range(len(self.validation_loss)), key=self.validation_loss.__getitem__
        )

    def to_json(self, path: Path) -> None:
        """Write the history to a JSON file."""
        path.write_text(
            json.dumps(
                {
                    "train_loss": self.train_loss,
                    "validation_loss": self.validation_loss,
                },
                indent=2,
            )
        )
