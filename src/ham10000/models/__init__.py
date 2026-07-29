"""Model construction, training, and inference."""

from __future__ import annotations

from ham10000.models.aggregation import (
    aggregate_predictions,
    aggregate_probabilities,
    majority_vote,
    predicted_label,
)
from ham10000.models.architectures import (
    Architecture,
    FreezeStrategy,
    apply_freezing,
    build_classifier,
    replace_head,
    trainable_parameters,
)
from ham10000.models.inference import predict_probabilities, select_device
from ham10000.models.thresholds import (
    apply_cost_sensitive_weights,
    apply_priority_thresholds,
)
from ham10000.models.training import (
    TrainingConfig,
    TrainingHistory,
    train_model,
)

__all__ = [
    "Architecture",
    "FreezeStrategy",
    "TrainingConfig",
    "TrainingHistory",
    "aggregate_predictions",
    "aggregate_probabilities",
    "apply_cost_sensitive_weights",
    "apply_freezing",
    "apply_priority_thresholds",
    "build_classifier",
    "majority_vote",
    "predict_probabilities",
    "predicted_label",
    "replace_head",
    "select_device",
    "train_model",
    "trainable_parameters",
]
