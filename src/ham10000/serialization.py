"""Safe loading of PyTorch checkpoints.

Two things matter here.

**Security.** ``torch.load`` without ``weights_only=True`` unpickles arbitrary
Python objects, so loading a checkpoint from an untrusted source is equivalent
to executing it. Note that no dependency upgrade fixes this: a call site that
opts out of the safe path stays unsafe at any version. This project's
checkpoints are written with ``torch.save(model.state_dict(), ...)``, which is
plain tensors, so ``weights_only=True`` is compatible with every existing
``.pth`` file.

**A failed load must be fatal.** Printing the error and carrying on leaves
inference running with randomly initialised weights, which produces a
plausible-looking probability table made of noise.

The pure-dictionary helpers here (:func:`strip_data_parallel_prefix`,
:func:`unwrap_checkpoint`) are deliberately free of any torch import, so the
tricky logic is testable in an environment without PyTorch installed.

Examples
--------
>>> unwrap_checkpoint({"state_dict": {"fc.weight": 1}})
{'fc.weight': 1}
>>> strip_data_parallel_prefix({"module.fc.weight": 1})
{'fc.weight': 1}
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    pass

__all__ = [
    "CheckpointError",
    "describe_checkpoint",
    "load_state_dict",
    "strip_data_parallel_prefix",
    "unwrap_checkpoint",
]

#: Keys under which training scripts commonly nest the actual weights.
_NESTED_KEYS = ("state_dict", "model_state_dict", "model")

#: Prefix that ``torch.nn.DataParallel`` prepends to every parameter name.
_DATA_PARALLEL_PREFIX = "module."


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, unreadable, or not a state dict."""


def unwrap_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the parameter mapping from a possibly-nested checkpoint.

    A checkpoint may be a bare ``state_dict``, or a training-resume bundle that
    nests the weights alongside optimiser state and an epoch counter. This
    normalises both to a flat parameter mapping.

    Parameters
    ----------
    payload:
        The object read back from disk.

    Returns
    -------
    dict
        The parameter mapping.

    Examples
    --------
    A bare state dict passes through unchanged:

    >>> unwrap_checkpoint({"fc.weight": 0, "fc.bias": 1})
    {'fc.weight': 0, 'fc.bias': 1}

    A resume bundle is unwrapped:

    >>> unwrap_checkpoint({"epoch": 10, "state_dict": {"fc.weight": 0}})
    {'fc.weight': 0}

    Only a *sole* nesting key is treated as a bundle; a model whose own
    parameters happen to include one of these names is left alone:

    >>> unwrap_checkpoint({"model.weight": 0, "state_dict.bias": 1})
    {'model.weight': 0, 'state_dict.bias': 1}
    """
    for key in _NESTED_KEYS:
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            return dict(nested)
    return dict(payload)


def strip_data_parallel_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the ``module.`` prefix added by :class:`torch.nn.DataParallel`.

    Weights saved from a model wrapped in ``DataParallel``, which happens
    whenever training runs with more than one GPU visible, carry a ``module.``
    prefix on every key. Loading them into an unwrapped model fails
    with a wall of "unexpected key" errors. This is a common and confusing
    artefact of Colab multi-GPU runtimes.

    Stripping is all-or-nothing: it applies only when *every* key carries the
    prefix, so a genuine submodule named ``module`` is never mangled.

    Parameters
    ----------
    state_dict:
        Parameter mapping, possibly prefixed.

    Returns
    -------
    dict
        Mapping with the prefix removed, or a copy of the input if the prefix
        was not uniformly present.

    Examples
    --------
    >>> strip_data_parallel_prefix({"module.fc.weight": 0, "module.fc.bias": 1})
    {'fc.weight': 0, 'fc.bias': 1}

    Mixed keys are left untouched, because stripping would be a guess:

    >>> strip_data_parallel_prefix({"module.fc.weight": 0, "bn.weight": 1})
    {'module.fc.weight': 0, 'bn.weight': 1}

    An empty mapping is returned unchanged rather than treated as "all
    prefixed", which ``all()`` would otherwise report vacuously:

    >>> strip_data_parallel_prefix({})
    {}
    """
    if not state_dict:
        return {}
    if all(key.startswith(_DATA_PARALLEL_PREFIX) for key in state_dict):
        offset = len(_DATA_PARALLEL_PREFIX)
        return {key[offset:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_state_dict(
    path: Path | str,
    *,
    map_location: str = "cpu",
    strip_prefix: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint safely, or raise.

    Unlike the code this replaces, every failure mode is fatal. Silently
    continuing with uninitialised weights produces confident nonsense, which is
    considerably worse than a traceback.

    Parameters
    ----------
    path:
        Path to a ``.pth`` file.
    map_location:
        Device to map storages onto. Defaults to ``"cpu"`` so a checkpoint
        trained on a Colab GPU loads on a laptop without a CUDA runtime.
    strip_prefix:
        Whether to apply :func:`strip_data_parallel_prefix`.

    Returns
    -------
    dict
        Parameter mapping ready for ``model.load_state_dict``.

    Raises
    ------
    CheckpointError
        If the file is missing, cannot be deserialised under
        ``weights_only=True``, or does not contain a mapping.

    Notes
    -----
    ``weights_only=True`` restricts deserialisation to tensors and a small set
    of primitives. Every checkpoint in this project was written as
    ``torch.save(model.state_dict(), ...)``, so all of them satisfy that
    restriction. A checkpoint from elsewhere that fails under this flag should
    be treated as untrusted, not force-loaded.
    """
    import torch  # Imported lazily: the helpers above stay usable without it.

    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(
            f"Could not load {path} with weights_only=True. The file may be "
            "corrupt, or it may contain pickled Python objects rather than "
            "plain tensors. Do not disable weights_only to work around this "
            "unless you produced the file yourself."
        ) from exc

    if not isinstance(payload, Mapping):
        raise CheckpointError(
            f"{path} deserialised to {type(payload).__name__}, not a state dict."
        )

    state_dict = unwrap_checkpoint(payload)
    return strip_data_parallel_prefix(state_dict) if strip_prefix else state_dict


def describe_checkpoint(path: Path | str) -> dict[str, tuple[int, ...]]:
    """Report parameter names and shapes without instantiating a model.

    Intended for auditing recovered weights: it answers "is this file intact,
    and which architecture and class count does it correspond to?" without
    needing to guess the architecture first.

    Parameters
    ----------
    path:
        Path to a ``.pth`` file.

    Returns
    -------
    dict
        Mapping from parameter name to shape.

    Examples
    --------
    The final layer reveals the head width, and therefore the number of
    classes the checkpoint was trained for::

        >>> shapes = describe_checkpoint("rn18_ta_bal.pth")  # doctest: +SKIP
        >>> shapes["fc.weight"]                              # doctest: +SKIP
        (5, 512)
    """
    state_dict = load_state_dict(path)
    return {
        name: tuple(value.shape)
        for name, value in state_dict.items()
        if hasattr(value, "shape")
    }
