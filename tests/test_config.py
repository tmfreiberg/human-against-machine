"""Unit tests for :mod:`ham10000.config`.

The regression these guard is the one that made the original package
unusable: import-time failure when the environment variable is unset.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ham10000.config import ENV_VAR, ProjectRootNotFoundError, Settings


def test_package_imports_with_no_environment_variable() -> None:
    """The regression: ``Path(os.getenv(...))`` at module scope.

    Run in a subprocess with a scrubbed environment, because the parent
    process may legitimately have the variable set.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import ham10000; print(ham10000.__version__)"],
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        },
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


class TestResolutionOrder:
    """Each strategy is tried in the documented order."""

    def test_explicit_root_wins_over_environment(self, tmp_path: Path) -> None:
        explicit = tmp_path / "explicit"
        explicit.mkdir()
        other = tmp_path / "other"
        other.mkdir()

        settings = Settings.resolve(explicit, env={ENV_VAR: str(other)})

        assert settings.root == explicit.resolve()

    def test_environment_variable_is_used(self, tmp_path: Path) -> None:
        assert Settings.resolve(env={ENV_VAR: str(tmp_path)}).root == tmp_path.resolve()

    def test_upward_search_finds_marker(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        nested = tmp_path / "notebooks" / "task_01"
        nested.mkdir(parents=True)

        assert Settings.resolve(start=nested, env={}).root == tmp_path.resolve()


class TestFailureModes:
    """Failures are explicit, actionable, and never interactive."""

    def test_unresolvable_root_raises_with_instructions(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectRootNotFoundError) as excinfo:
            Settings.resolve(start=tmp_path, env={})

        message = str(excinfo.value)
        assert ENV_VAR in message
        assert "To fix this" in message

    def test_root_pointing_at_a_file_is_rejected(self, tmp_path: Path) -> None:
        not_a_directory = tmp_path / "metadata.csv"
        not_a_directory.write_text("")

        with pytest.raises(ProjectRootNotFoundError, match="not a directory"):
            Settings.resolve(not_a_directory)

    def test_resolution_never_blocks_on_input(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """``path_setup.confirm`` called ``input()`` and could hang in CI."""

        def explode(*args: object, **kwargs: object) -> str:
            raise AssertionError("resolution must not prompt for input")

        monkeypatch.setattr("builtins.input", explode)
        Settings.resolve(tmp_path)


class TestLayout:
    """Directory properties are pure path arithmetic."""

    def test_paths_are_derived_without_touching_disk(self) -> None:
        settings = Settings(Path("/nonexistent/root"))

        assert settings.images == Path("/nonexistent/root/images")
        assert settings.metadata_csv == Path("/nonexistent/root/images/metadata.csv")

    def test_settings_are_frozen(self) -> None:
        settings = Settings(Path("/a"))

        with pytest.raises(AttributeError):
            settings.root = Path("/b")  # type: ignore[misc]

    def test_require_reports_all_missing_directories_at_once(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError) as excinfo:
            Settings(tmp_path).require("images", "models")

        message = str(excinfo.value)
        assert "images" in message
        assert "models" in message

    def test_require_returns_self_for_chaining(self, tmp_path: Path) -> None:
        (tmp_path / "images").mkdir()
        settings = Settings(tmp_path)

        assert settings.require("images") is settings

    def test_require_rejects_unknown_directory_names(self, tmp_path: Path) -> None:
        with pytest.raises(AttributeError, match="not a known project directory"):
            Settings(tmp_path).require("scratch")


def test_every_known_subdirectory_has_a_property() -> None:
    """Guards the inconsistency this suite caught on first run.

    ``KNOWN_SUBDIRECTORIES`` listed ``expository`` before a matching property
    existed. :meth:`Settings.require` resolves each name via ``getattr``, so a
    declared-but-unimplemented entry raises ``AttributeError`` at use.
    """
    from ham10000.config import KNOWN_SUBDIRECTORIES

    settings = Settings(Path("/nonexistent"))
    missing = [name for name in KNOWN_SUBDIRECTORIES if not hasattr(settings, name)]

    assert not missing, f"declared but not implemented: {missing}"
