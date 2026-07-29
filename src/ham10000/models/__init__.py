"""Model construction, training, and inference.

Symbols are resolved lazily. Importing this package, or any module inside it,
must not require PyTorch: `aggregation` and `thresholds` work on probability
tables with pandas alone, and are useful without a model. Eagerly re-exporting
from `architectures` here would pull torch into every one of those imports,
because importing a submodule runs its package's ``__init__`` first.

The lazy lookup keeps the convenient spelling::

    from ham10000.models import build_classifier

while deferring the torch import to the moment that name is actually used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Re-exported for type checkers and editors only; `__all__` is computed from
# `_EXPORTS` below, which ruff cannot see, hence the suppressions.
if TYPE_CHECKING:  # pragma: no cover
    # ruff: noqa: F401
    from ham10000.models.aggregation import (
        aggregate_predictions,
        aggregate_probabilities,
        majority_vote,
        predicted_label,
    )
    from ham10000.models.architectures import (
        apply_freezing,
        build_classifier,
        replace_head,
        trainable_parameters,
    )
    from ham10000.models.inference import predict_probabilities, select_device
    from ham10000.models.options import (
        Architecture,
        FreezeStrategy,
        TrainingConfig,
        TrainingHistory,
    )
    from ham10000.models.thresholds import (
        apply_cost_sensitive_weights,
        apply_priority_thresholds,
    )
    from ham10000.models.training import train_model

#: Name to the module that defines it. Modules needing torch are marked, so the
#: cost of each name is visible here rather than discovered at import time.
_EXPORTS: dict[str, str] = {
    # pandas and numpy only
    "aggregate_predictions": "aggregation",
    "aggregate_probabilities": "aggregation",
    "majority_vote": "aggregation",
    "predicted_label": "aggregation",
    "Architecture": "options",
    "FreezeStrategy": "options",
    "apply_cost_sensitive_weights": "thresholds",
    "apply_priority_thresholds": "thresholds",
    # requires torch
    "apply_freezing": "architectures",
    "build_classifier": "architectures",
    "replace_head": "architectures",
    "trainable_parameters": "architectures",
    "predict_probabilities": "inference",
    "select_device": "inference",
    "TrainingConfig": "options",
    "TrainingHistory": "options",
    "train_model": "training",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Resolve an exported name on first use.

    Examples
    --------
    >>> from ham10000.models import majority_vote
    >>> callable(majority_vote)
    True
    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    """List the exported names, so tab completion still works."""
    return __all__
