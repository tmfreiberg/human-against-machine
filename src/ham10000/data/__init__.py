"""Data preparation: metadata loading, label schemes, and splitting.

The pipeline reads left to right::

    load_metadata -> restrict/exclude -> LabelScheme -> assign_splits

Each stage is a pure function or an immutable value object, so any stage can be
exercised in a test without constructing the ones before it. Performing all of
it inside a single constructor would mean testing the split required a CSV on
disk and a full pipeline run.
"""

from __future__ import annotations

from ham10000.data.balancing import balance, expand_validation, resample_class
from ham10000.data.labels import OTHER, LabelScheme
from ham10000.data.metadata import (
    REQUIRED_COLUMNS,
    exclude,
    load_metadata,
    restrict,
)
from ham10000.data.splitting import (
    SPLIT_VALUES,
    SplitAssignment,
    SplitConfig,
    assign_splits,
    lesion_overlap,
)

__all__ = [
    "OTHER",
    "REQUIRED_COLUMNS",
    "SPLIT_VALUES",
    "LabelScheme",
    "SplitAssignment",
    "SplitConfig",
    "assign_splits",
    "balance",
    "exclude",
    "expand_validation",
    "lesion_overlap",
    "load_metadata",
    "resample_class",
    "restrict",
]
