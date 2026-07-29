"""Classifier construction and layer-freezing strategies.

Transfer learning here means taking an ImageNet-pretrained backbone, replacing
its classification head with one sized to the task, and training some suffix of
the network while holding the rest fixed. This module covers those two steps.

Freezing, and the order it must happen in
-----------------------------------------
Torchvision returns pretrained models with *every* parameter already trainable.
A function that only ever sets ``requires_grad = True`` on some suffix of the
network therefore does nothing at all, and every freezing strategy silently
trains the whole model.

:func:`apply_freezing` freezes everything first and then unfreezes the
requested portion, which is the only order that works. It returns the names it
left trainable, so a caller can assert on the result rather than trust it, and
:func:`trainable_parameters` exists so an optimiser is never handed tensors
that will not receive a gradient.

:func:`apply_freezing` freezes everything first, then unfreezes the requested
suffix, and returns the names it made trainable so a caller can assert on the
result instead of trusting it.

EfficientNet support
--------------------
The repository's `train` raises no error for EfficientNet with a partial
unfreeze -- it prints `"Need to update code for EfficientNet regarding
unfreezing final layers"` and returns *without training*. But the recorded
output of `08_ENB0_balance_models_ta_rnd_crop_colab.ipynb`, the notebook that
produced the best reported result, shows lines like `Unfrozen layer:
features.6.0.block.0.0.weight`. That message exists nowhere in the repository.

The best model was therefore trained by a version of this code that was written
in Colab and never committed. This module reimplements that behaviour --
unfreezing a trailing run of `features` blocks plus the classifier -- so the
configuration is at least expressible again.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum

import torch
from torch import nn
from torchvision import models

__all__ = [
    "Architecture",
    "FreezeStrategy",
    "apply_freezing",
    "build_classifier",
    "replace_head",
    "trainable_parameters",
]


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
        features. Fastest, and a genuine baseline -- it measures how much of
        the task is solvable from ImageNet features alone.
    """

    ALL = "all"
    LAST_BLOCK = "last_block"
    HEAD_ONLY = "head_only"


def replace_head(model: nn.Module, n_classes: int) -> nn.Module:
    """Swap the classification head for one with `n_classes` outputs.

    Parameters
    ----------
    model:
        A torchvision ResNet or EfficientNet. Modified in place and returned.
    n_classes:
        Number of output classes, i.e. `LabelScheme.n_classes`.

    Returns
    -------
    nn.Module
        The same object, for chaining.

    Raises
    ------
    TypeError
        If the architecture is not recognised. Leaving the head untouched
        would let an unsupported model train against 1000 ImageNet classes and
        produce a probability table of the wrong width, with nothing to signal
        it.
    ValueError
        If `n_classes` is less than 2.

    Examples
    --------
    >>> model = replace_head(models.resnet18(), n_classes=5)
    >>> model.fc.out_features
    5

    >>> model = replace_head(models.efficientnet_b0(), n_classes=2)
    >>> model.classifier[1].out_features
    2
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}.")

    if isinstance(model, models.ResNet):
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif isinstance(model, models.EfficientNet):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    else:
        raise TypeError(
            f"Cannot replace the head of {type(model).__name__}. "
            "Supported: torchvision ResNet and EfficientNet."
        )
    # isinstance-narrowing against torchvision's classes widens to Any, so
    # re-annotate explicitly rather than silencing the checker.
    narrowed: nn.Module = model
    return narrowed


def _head_parameters(model: nn.Module) -> list[nn.Module]:
    """Return the modules constituting the classification head."""
    if isinstance(model, models.ResNet):
        return [model.fc]
    if isinstance(model, models.EfficientNet):
        return [model.classifier]
    raise TypeError(f"Unsupported architecture: {type(model).__name__}.")


def _last_block_modules(model: nn.Module, n_blocks: int) -> list[nn.Module]:
    """Return the trailing feature blocks to train, plus the head."""
    if isinstance(model, models.ResNet):
        # ResNet exposes four sequential stages; layer4 is the final one.
        stages = [model.layer1, model.layer2, model.layer3, model.layer4]
        return stages[-n_blocks:] + _head_parameters(model)
    if isinstance(model, models.EfficientNet):
        # EfficientNet-B0 has nine entries in `features`. The Colab run that
        # produced the best result unfroze from `features.6` onward, which is
        # the last three of them.
        blocks: list[nn.Module] = list(model.features)
        return blocks[-(n_blocks + 1) :] + _head_parameters(model)
    raise TypeError(f"Unsupported architecture: {type(model).__name__}.")


def apply_freezing(
    model: nn.Module,
    strategy: FreezeStrategy = FreezeStrategy.LAST_BLOCK,
    *,
    n_blocks: int = 1,
) -> frozenset[str]:
    """Freeze the network, then unfreeze the portion the strategy selects.

    Parameters
    ----------
    model:
        Model to modify in place.
    strategy:
        Which portion to train.
    n_blocks:
        For :attr:`FreezeStrategy.LAST_BLOCK`, how many trailing feature stages
        to train in addition to the head.

    Returns
    -------
    frozenset of str
        Names of the parameters left trainable. Returned so a caller can assert
        on the outcome rather than assume it.

    Notes
    -----
    The freeze-then-unfreeze order is the correction. Setting `requires_grad`
    to `True` on a suffix of an already-trainable network changes nothing,
    which would leave every parameter trainable.

    Examples
    --------
    Training everything leaves every parameter trainable:

    >>> model = replace_head(models.resnet18(), 5)
    >>> trainable = apply_freezing(model, FreezeStrategy.ALL)
    >>> len(trainable) == len(list(model.parameters()))
    True

    Head-only training leaves exactly the head:

    >>> model = replace_head(models.resnet18(), 5)
    >>> sorted(apply_freezing(model, FreezeStrategy.HEAD_ONLY))
    ['fc.bias', 'fc.weight']

    The last-block strategy genuinely freezes the early layers -- the property
    a partial-unfreeze strategy must have:

    >>> model = replace_head(models.resnet18(), 5)
    >>> trainable = apply_freezing(model, FreezeStrategy.LAST_BLOCK)
    >>> any(name.startswith("layer1") for name in trainable)
    False
    >>> any(name.startswith("layer4") for name in trainable)
    True
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    if strategy is FreezeStrategy.ALL:
        selected: list[nn.Module] = [model]
    elif strategy is FreezeStrategy.HEAD_ONLY:
        selected = _head_parameters(model)
    else:
        selected = _last_block_modules(model, n_blocks)

    for module in selected:
        for parameter in module.parameters():
            parameter.requires_grad = True

    return frozenset(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def trainable_parameters(model: nn.Module) -> Iterator[torch.nn.Parameter]:
    """Yield only the parameters an optimiser should update.

    Handing frozen parameters to an optimiser allocates state for tensors that
    never receive a gradient, and under weight decay can perturb weights that
    were meant to stay fixed.

    Examples
    --------
    >>> model = replace_head(models.resnet18(), 5)
    >>> _ = apply_freezing(model, FreezeStrategy.HEAD_ONLY)
    >>> len(list(trainable_parameters(model)))
    2
    """
    return (p for p in model.parameters() if p.requires_grad)


def build_classifier(
    architecture: Architecture | str = Architecture.RESNET18,
    *,
    n_classes: int,
    strategy: FreezeStrategy | str = FreezeStrategy.LAST_BLOCK,
    n_blocks: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """Construct a fine-tuning-ready classifier in one call.

    Parameters
    ----------
    architecture:
        Backbone to use.
    n_classes:
        Number of output classes.
    strategy:
        Freezing strategy.
    n_blocks:
        Trailing feature stages to train under
        :attr:`FreezeStrategy.LAST_BLOCK`.
    pretrained:
        Load ImageNet weights. `False` is useful in tests and for measuring how
        much of the performance is due to transfer rather than architecture.

    Returns
    -------
    nn.Module

    Examples
    --------
    >>> model = build_classifier(
    ...     "resnet18", n_classes=2, strategy="head_only", pretrained=False
    ... )
    >>> model.fc.out_features
    2
    >>> sum(p.requires_grad for p in model.parameters())
    2
    """
    architecture = Architecture(architecture)
    strategy = FreezeStrategy(strategy)

    if architecture is Architecture.RESNET18:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model: nn.Module = models.resnet18(weights=weights)
    else:
        weights_b0 = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights_b0)

    replace_head(model, n_classes)
    apply_freezing(model, strategy, n_blocks=n_blocks)
    return model
