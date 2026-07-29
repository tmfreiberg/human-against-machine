"""Unit tests for :mod:`ham10000.serialization`.

These cover the pure-dictionary helpers and the failure paths, and require no
PyTorch installation. Tests that genuinely need torch are marked and skipped
when it is absent, so the suite is meaningful in a lightweight CI job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ham10000.serialization import (
    CheckpointError,
    load_state_dict,
    strip_data_parallel_prefix,
    unwrap_checkpoint,
)

try:
    import torch
except ImportError:  # pragma: no cover - depends on the installed environment
    torch = None  # type: ignore[assignment]

#: Applied only to tests that genuinely deserialise a file. The dictionary
#: helpers below are pure Python and must run everywhere.
requires_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


class TestUnwrapCheckpoint:
    def test_bare_state_dict_passes_through(self) -> None:
        payload = {"fc.weight": 0, "fc.bias": 1}

        assert unwrap_checkpoint(payload) == payload

    @pytest.mark.parametrize("key", ["state_dict", "model_state_dict", "model"])
    def test_nested_bundle_is_unwrapped(self, key: str) -> None:
        assert unwrap_checkpoint({"epoch": 7, key: {"fc.weight": 0}}) == {
            "fc.weight": 0
        }

    def test_parameter_named_like_a_bundle_key_is_not_unwrapped(self) -> None:
        """A tensor at ``model`` must not be mistaken for a nested mapping."""
        payload = {"model.weight": 0, "state_dict.bias": 1}

        assert unwrap_checkpoint(payload) == payload

    def test_result_is_a_copy(self) -> None:
        payload = {"fc.weight": 0}

        unwrap_checkpoint(payload)["fc.weight"] = 99

        assert payload["fc.weight"] == 0


class TestStripDataParallelPrefix:
    def test_uniform_prefix_is_stripped(self) -> None:
        prefixed = {"module.fc.weight": 0, "module.fc.bias": 1}

        assert strip_data_parallel_prefix(prefixed) == {"fc.weight": 0, "fc.bias": 1}

    def test_mixed_keys_are_left_alone(self) -> None:
        """Partial prefixing means stripping would be a guess, so we do not."""
        mixed = {"module.fc.weight": 0, "bn.weight": 1}

        assert strip_data_parallel_prefix(mixed) == mixed

    def test_empty_mapping_is_not_treated_as_uniformly_prefixed(self) -> None:
        """``all()`` is vacuously true on an empty iterable; guard against it."""
        assert strip_data_parallel_prefix({}) == {}

    def test_only_the_leading_occurrence_is_removed(self) -> None:
        prefixed = {"module.module.fc.weight": 0}

        assert strip_data_parallel_prefix(prefixed) == {"module.fc.weight": 0}


class TestLoadStateDict:
    @requires_torch
    def test_missing_file_raises_rather_than_printing(self, tmp_path: Path) -> None:
        """A naive implementation printed and continued with uninitialised weights."""
        with pytest.raises(CheckpointError, match="not found"):
            load_state_dict(tmp_path / "absent.pth")

    @requires_torch
    def test_corrupt_file_raises_checkpoint_error(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.pth"
        corrupt.write_bytes(b"not a torch archive")

        with pytest.raises(CheckpointError, match="weights_only=True"):
            load_state_dict(corrupt)

    @requires_torch
    def test_round_trip_of_a_plain_state_dict(self, tmp_path: Path) -> None:
        """Every checkpoint in this project was saved this way."""
        path = tmp_path / "model.pth"
        torch.save({"fc.weight": torch.zeros(5, 512)}, path)

        loaded = load_state_dict(path)

        assert tuple(loaded["fc.weight"].shape) == (5, 512)

    @requires_torch
    def test_data_parallel_checkpoint_loads_into_unwrapped_model(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "dataparallel.pth"
        torch.save({"module.fc.weight": torch.zeros(5, 512)}, path)

        assert "fc.weight" in load_state_dict(path)

    @requires_torch
    def test_non_mapping_payload_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "tensor.pth"
        torch.save(torch.zeros(3), path)

        with pytest.raises(CheckpointError, match="not a state dict"):
            load_state_dict(path)
