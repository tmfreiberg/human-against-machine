"""Unit tests for :mod:`ham10000.display`."""

from __future__ import annotations

import pytest

from ham10000.display import display, in_notebook, print_header


class TestInNotebook:
    def test_returns_false_under_pytest(self) -> None:
        """A test run is not a Jupyter kernel, whether or not IPython exists."""
        assert in_notebook() is False

    def test_result_is_cached(self) -> None:
        in_notebook.cache_clear()
        in_notebook()
        assert in_notebook.cache_info().hits >= 0
        in_notebook()
        assert in_notebook.cache_info().hits >= 1


class TestDisplay:
    def test_prints_each_object_on_its_own_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        display("alpha", "beta")

        assert capsys.readouterr().out == "alpha\nbeta\n"

    def test_no_arguments_produces_no_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        display()

        assert capsys.readouterr().out == ""

    def test_non_string_objects_are_rendered(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        display({"mel": 1113})

        assert "mel" in capsys.readouterr().out


class TestPrintHeader:
    def test_rules_match_header_width(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_header("balanced accuracy")

        lines = capsys.readouterr().out.split("\n")

        assert lines[1] == "=" * len("balanced accuracy")
        assert lines[2] == "BALANCED ACCURACY"
        assert lines[3] == lines[1]

    def test_argument_is_not_mutated(self) -> None:
        header = "class distribution"
        print_header(header)

        assert header == "class distribution"

    def test_empty_header_does_not_crash(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        print_header("")

        assert capsys.readouterr().out == "\n\n\n\n\n"
