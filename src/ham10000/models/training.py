"""Fine-tuning loop for lesion classifiers.

Replaces `multiclass_models.cnn.train`, which was a 170-line method on a class
whose `__init__` took twenty-one optional parameters, most defaulting to `None`
and resolved by a cascade of `if x is None` assignments. Configuration is now a
frozen dataclass, and training is a function over explicit arguments.

Changes worth knowing
---------------------
**Freezing** is applied through
:func:`~ham10000.models.architectures.apply_freezing`, and only the trainable
parameters are handed to the optimiser.

**Loaders are built once**, outside the epoch loop.

**Checkpoints are written once, at the end.** When `save_best` is set, the best
weights are loaded back into the model before returning, so the metrics a
caller computes afterwards describe the model that was actually saved rather
than whatever was left in memory.

**Loss history is returned** rather than stored on the object and written out
as a side effect.

Warning
-------
The last epoch's weights are saved unless `save_best` is set. With no early
stopping, a run that begins overfitting at epoch 5 still saves epoch 10.

`save_best` keeps the lowest-validation-loss epoch instead, with two caveats.
Selecting an epoch by validation loss and then reporting validation metrics is
selection on the evaluation set, which makes the reported score optimistic by
an unknown amount. And the selection criterion is not the reported metric, so
the lowest-loss epoch is not necessarily the most accurate one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader

from ham10000.data.labels import LabelScheme
from ham10000.models.architectures import trainable_parameters
from ham10000.models.dataset import LesionImageDataset
from ham10000.models.inference import select_device
from ham10000.models.options import TrainingConfig, TrainingHistory

__all__ = ["train_model"]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, str]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    on_batch: Callable[[int, int, float], None] | None = None,
) -> float:
    """Run one pass, training when an optimiser is supplied. Returns mean loss."""
    training = optimizer is not None
    model.train(training)

    total = 0.0
    batches = 0

    with torch.set_grad_enabled(training):
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            loss = criterion(model(images), labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item()
            batches += 1
            if on_batch is not None:
                on_batch(batches, len(loader), total / batches)

    if batches == 0:
        raise ValueError("DataLoader yielded no batches; the frame is empty.")
    return total / batches


def train_model(
    model: nn.Module,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    scheme: LabelScheme,
    *,
    image_dir: Path,
    transform: Callable[[Image.Image], torch.Tensor],
    config: TrainingConfig | None = None,
    checkpoint_path: Path | None = None,
    device: torch.device | str | None = None,
    on_epoch: Callable[[int, float, float], None] | None = None,
    on_batch: Callable[[int, int, float], None] | None = None,
) -> TrainingHistory:
    """Fine-tune `model` and return its per-epoch loss history.

    Parameters
    ----------
    model:
        A model with its head already sized and its freezing already applied,
        typically from
        :func:`~ham10000.models.architectures.build_classifier`.
    train_frame, validation_frame:
        Metadata with `image_id` and `label` columns.
    scheme:
        Label scheme, fixing the one-hot width.
    image_dir:
        Directory containing `<image_id>.jpg`.
    transform:
        Transform applied to every image. Note that this is used for validation
        too, which makes validation stochastic -- deliberate, and the mechanism
        the expanded validation set relies on.
    config:
        Hyperparameters. Defaults to :class:`TrainingConfig`.
    checkpoint_path:
        Where to write `state_dict`. Nothing is written when omitted.
    device:
        Compute device. Detected when omitted.
    on_batch:
        Called with `(batch, total_batches, running_mean_loss)` after every
        training batch. Without it a long run produces no output until the
        first epoch ends, which on CPU can be ten minutes of silence
        indistinguishable from a hang.
    on_epoch:
        Called with `(epoch, train_loss, validation_loss)` after each epoch.
        Without it the function is silent, which keeps it usable from a
        script, a test, or a notebook without flooding output.

    Returns
    -------
    TrainingHistory

    Raises
    ------
    ValueError
        If no parameter is trainable, or a frame lacks a `label` column.

    Notes
    -----
    Loss is `nn.CrossEntropyLoss` over one-hot targets. That accepts a
    probability distribution as the target. Passing class indices instead
    would give the same gradients by a different numerical path.
    """
    config = config or TrainingConfig()
    resolved = torch.device(device) if device is not None else select_device()

    for name, frame in (("train", train_frame), ("validation", validation_frame)):
        if "label" not in frame.columns:
            raise ValueError(f"{name}_frame must contain a 'label' column.")

    parameters = list(trainable_parameters(model))
    if not parameters:
        raise ValueError(
            "No trainable parameters. Apply a freezing strategy that leaves at "
            "least the head unfrozen."
        )

    torch.manual_seed(config.seed)

    def build(
        frame: pd.DataFrame, *, shuffle: bool
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, str]]:
        dataset = LesionImageDataset(
            frame,
            n_classes=scheme.n_classes,
            image_dir=image_dir,
            transform=transform,
        )
        # A final training batch of size 1 crashes BatchNorm, which cannot
        # compute a variance from one sample. This fires whenever
        # len(train) % batch_size == 1, roughly a 1-in-32 chance per run at the
        # default batch size, and it surfaces mid-run after minutes of
        # training. Dropping the single leftover row avoids it.
        drop_last = shuffle and len(dataset) % config.batch_size == 1
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            drop_last=drop_last,
        )

    train_loader = build(train_frame, shuffle=True)
    validation_loader = build(validation_frame, shuffle=False)

    model.to(resolved)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(parameters, lr=config.learning_rate)

    history = TrainingHistory()
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(config.epochs):
        train_loss = _run_epoch(
            model, train_loader, criterion, resolved, optimizer, on_batch
        )
        validation_loss = _run_epoch(model, validation_loader, criterion, resolved)
        history.record(train=train_loss, validation=validation_loss)

        if config.save_best and validation_loss == min(history.validation_loss):
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if on_epoch is not None:
            on_epoch(epoch, train_loss, validation_loss)

    # Restore the best weights into the model itself, not merely to disk.
    # Without this the caller goes on to score the *final* epoch while the
    # checkpoint on disk holds the best one, so the reported metrics describe a
    # model that was never saved and the saved model was never measured.
    if config.save_best and best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        history.to_json(checkpoint_path.with_suffix(".losses.json"))

    return history
