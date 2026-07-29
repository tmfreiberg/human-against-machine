"""Test collection rules.

The package is designed to be usable without PyTorch: metadata handling,
splitting, balancing, aggregation, thresholds and evaluation all work on tables
alone, and CI has a job that installs without the `models` extra to prove it.

That job must still be able to *collect* the suite. Two things would otherwise
stop it. `--doctest-modules` imports every module under `src/`, including those
that import torch at module scope, and a few test modules exercise the model
layer directly. Both are skipped when torch is absent, so a torch-free run
reports the tests it did run rather than an error.
"""

from __future__ import annotations

import importlib.util

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

#: Source modules that import torch at module scope, so cannot be collected
#: for doctests without it.
_NEEDS_TORCH = [
    "src/ham10000/models/architectures.py",
    "src/ham10000/models/dataset.py",
    "src/ham10000/models/inference.py",
    "src/ham10000/models/training.py",
    "tests/test_architectures.py",
    "tests/test_experiment.py",
    "tests/test_training.py",
]

collect_ignore = [] if TORCH_AVAILABLE else list(_NEEDS_TORCH)


#: Individual doctests that build a torchvision object, in modules that are
#: otherwise importable without it.
_DOCTESTS_NEEDING_TORCH = ("build_transform",)


def pytest_collection_modifyitems(items: list) -> None:
    """Skip the few doctests that need torch, when it is absent."""
    if TORCH_AVAILABLE:
        return

    import pytest

    skip = pytest.mark.skip(reason="needs the models extra")
    for item in items:
        if any(name in item.nodeid for name in _DOCTESTS_NEEDING_TORCH):
            item.add_marker(skip)
