"""Unit tests for :mod:`ham10000.data.metadata`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ham10000.data.metadata import exclude, load_metadata, restrict

CSV = "lesion_id,image_id,dx\nL1,I1,mel\nL1,I2,mel\nL2,I3,nv\n"


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.csv"
    path.write_text(CSV)
    return path


class TestLoadMetadata:
    def test_num_images_counts_images_per_lesion(self, csv_path: Path) -> None:
        assert load_metadata(csv_path)["num_images"].tolist() == [2, 2, 1]

    def test_num_images_sits_beside_lesion_id(self, csv_path: Path) -> None:
        """Column order is preserved from  so notebook output matches."""
        assert list(load_metadata(csv_path).columns)[:2] == ["lesion_id", "num_images"]

    def test_annotation_can_be_disabled(self, csv_path: Path) -> None:
        assert "num_images" not in load_metadata(csv_path, add_num_images=False).columns

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A naive implementation printed and left the frame undefined."""
        with pytest.raises(FileNotFoundError):
            load_metadata(tmp_path / "absent.csv")

    def test_missing_required_column_names_itself(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("lesion_id,image_id\nL1,I1\n")

        with pytest.raises(ValueError, match="dx"):
            load_metadata(path)

    def test_duplicate_image_ids_are_rejected(self, tmp_path: Path) -> None:
        """A repeated image_id would silently corrupt the split."""
        path = tmp_path / "dupes.csv"
        path.write_text("lesion_id,image_id,dx\nL1,I1,mel\nL2,I1,nv\n")

        with pytest.raises(ValueError, match="duplicate image_id"):
            load_metadata(path)


class TestRestrict:
    def test_keeps_only_listed_values(self) -> None:
        frame = pd.DataFrame({"dx": ["mel", "nv", "bkl"]})

        assert restrict(frame, {"dx": ["mel", "nv"]})["dx"].tolist() == ["mel", "nv"]

    def test_criteria_are_conjunctive(self) -> None:
        frame = pd.DataFrame({"dx": ["mel", "nv"], "sex": ["m", "m"]})

        assert restrict(frame, {"dx": ["mel"], "sex": ["m"]})["dx"].tolist() == ["mel"]

    def test_unknown_columns_are_ignored(self) -> None:
        frame = pd.DataFrame({"dx": ["mel", "nv"]})

        assert len(restrict(frame, {"nonexistent": ["x"]})) == 2

    def test_returns_an_independent_copy(self) -> None:
        """A naive implementation returned a query view and then wrote into it."""
        frame = pd.DataFrame({"dx": ["mel", "nv"]})

        result = restrict(frame, {"dx": ["mel"]})
        result.loc[result.index[0], "dx"] = "CHANGED"

        assert frame["dx"].tolist() == ["mel", "nv"]


class TestExclude:
    def test_drops_listed_values(self) -> None:
        frame = pd.DataFrame({"dx": ["mel", "nv", "bkl"]})

        assert exclude(frame, {"dx": ["mel"]})["dx"].tolist() == ["nv", "bkl"]

    def test_criteria_are_disjunctive(self) -> None:
        """Regression: the query joined negations with `|`, inverting this.

        With two criteria the original kept a row unless it matched *both*,
        contradicting its own printed message. Correct behaviour drops a row
        matching *either*.
        """
        frame = pd.DataFrame({"dx": ["mel", "nv", "bkl"], "sex": ["m", "f", "m"]})

        assert exclude(frame, {"dx": ["mel"], "sex": ["f"]})["dx"].tolist() == ["bkl"]

    def test_empty_criteria_is_a_no_op(self) -> None:
        frame = pd.DataFrame({"dx": ["mel", "nv"]})

        assert len(exclude(frame, {})) == 2
