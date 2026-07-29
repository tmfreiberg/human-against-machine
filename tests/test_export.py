"""Unit tests for :mod:`ham10000.export`.

The properties that matter are fairness ones: the demo pool must contain only
images the model never trained on, one per lesion, balanced across classes. A
demo that quietly shows training images would flatter the model and there would
be no way to tell from the outside.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ham10000.cli import train_main
from ham10000.export import _select_pool, export_demo

pytest.importorskip("onnx", reason="export extra not installed")
pytest.importorskip("onnxruntime", reason="export extra not installed")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Build a two-class project, plus a third class to exercise `restrict`."""
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.RandomState(0)

    rows = []
    counter = 0
    for lesion in range(16):
        dx = "mel" if lesion % 2 else "nv"
        for _ in range(2 if lesion < 6 else 1):
            image_id = f"I{counter:03d}"
            pixels = np.clip(
                (190 if dx == "mel" else 50) + rng.randint(-25, 25, (120, 160, 3)),
                0,
                255,
            ).astype("uint8")
            Image.fromarray(pixels).save(images / f"{image_id}.jpg")
            rows.append({"lesion_id": f"L{lesion}", "image_id": image_id, "dx": dx})
            counter += 1

    for lesion in range(16, 22):
        image_id = f"I{counter:03d}"
        pixels = np.full((120, 160, 3), 120, dtype="uint8")
        Image.fromarray(pixels).save(images / f"{image_id}.jpg")
        rows.append({"lesion_id": f"L{lesion}", "image_id": image_id, "dx": "bkl"})
        counter += 1

    pd.DataFrame(rows).to_csv(images / "metadata.csv", index=False)
    (tmp_path / "pyproject.toml").write_text("")
    return tmp_path


@pytest.fixture
def demo_config(tmp_path: Path) -> Path:
    path = tmp_path / "demo.yaml"
    path.write_text(
        "name: demo\n"
        "description: export test\n"
        "restrict: {dx: [mel, nv]}\n"
        "classes: [mel, nv]\n"
        "pretrained: false\n"
        "freeze_strategy: head_only\n"
        "training: {epochs: 1, batch_size: 4}\n"
        "transform:\n"
        "  - {name: Resize, size: [32, 32]}\n"
        "  - {name: ToTensor}\n"
    )
    return path


@pytest.fixture
def run(project: Path, demo_config: Path) -> Path:
    train_main([str(demo_config), "--data-root", str(project), "--device", "cpu"])
    return next((project / "models").iterdir())


class TestRestrict:
    def test_excluded_classes_do_not_reach_the_model(self, run: Path) -> None:
        """`restrict` must remove bkl entirely, not merely relabel it."""
        predictions = pd.read_csv(run / "predictions.csv")

        assert set(predictions["dx"]) <= {"mel", "nv"}

    def test_restriction_is_recorded_in_the_run_config(self, run: Path) -> None:
        saved = json.loads(
            json.dumps(__import__("yaml").safe_load((run / "config.yaml").read_text()))
        )

        assert saved["restrict"] == {"dx": ["mel", "nv"]}


class TestSelectPool:
    def test_one_image_per_lesion(self) -> None:
        """Two images of one lesion in the pool would be near-duplicates."""
        frame = pd.DataFrame(
            {
                "lesion_id": ["L1", "L1", "L2", "L3"],
                "image_id": ["A", "B", "C", "D"],
                "dx": ["mel", "mel", "nv", "nv"],
            }
        )

        pool = _select_pool(frame, size=4, seed=0)

        assert not pool["lesion_id"].duplicated().any()

    def test_pool_is_balanced_across_classes(self) -> None:
        """Otherwise the game is won by always answering the majority class."""
        frame = pd.DataFrame(
            {
                "lesion_id": [f"L{i}" for i in range(30)],
                "image_id": [f"I{i}" for i in range(30)],
                "dx": ["nv"] * 27 + ["mel"] * 3,
            }
        )

        pool = _select_pool(frame, size=6, seed=0)

        assert pool["dx"].value_counts().to_dict() == {"nv": 3, "mel": 3}

    def test_scarce_class_shrinks_the_pool_rather_than_unbalancing_it(self) -> None:
        frame = pd.DataFrame(
            {
                "lesion_id": [f"L{i}" for i in range(12)],
                "image_id": [f"I{i}" for i in range(12)],
                "dx": ["nv"] * 10 + ["mel"] * 2,
            }
        )

        pool = _select_pool(frame, size=20, seed=0)

        assert pool["dx"].value_counts()["mel"] == 2

    def test_selection_is_reproducible(self) -> None:
        frame = pd.DataFrame(
            {
                "lesion_id": [f"L{i}" for i in range(20)],
                "image_id": [f"I{i}" for i in range(20)],
                "dx": ["nv"] * 10 + ["mel"] * 10,
            }
        )

        first = _select_pool(frame, size=6, seed=3)["image_id"].tolist()
        second = _select_pool(frame, size=6, seed=3)["image_id"].tolist()

        assert first == second


class TestExportDemo:
    def test_writes_every_asset(self, run: Path, project: Path, tmp_path: Path) -> None:
        result = export_demo(
            run,
            image_dir=project / "images",
            output=tmp_path / "demo",
            pool_size=6,
        )

        for asset in ("model.onnx", "labels.json", "pool.json"):
            assert (result.directory / asset).is_file(), asset
        images = list((result.directory / "images").glob("*.jpg"))
        assert len(images) == result.image_count

    def test_pool_contains_only_unseen_images(
        self, run: Path, project: Path, tmp_path: Path
    ) -> None:
        """The fairness guarantee: no training image may appear in the demo."""
        export_demo(
            run, image_dir=project / "images", output=tmp_path / "demo", pool_size=6
        )
        pool = json.loads((tmp_path / "demo" / "pool.json").read_text())
        predictions = pd.read_csv(run / "predictions.csv")

        pooled = {entry["image_id"] for entry in pool["images"]}
        assert pooled <= set(predictions["image_id"])

    def test_preprocessing_constants_are_shipped(
        self, run: Path, project: Path, tmp_path: Path
    ) -> None:
        """The browser must reproduce training preprocessing exactly."""
        export_demo(
            run, image_dir=project / "images", output=tmp_path / "demo", pool_size=6
        )
        labels = json.loads((tmp_path / "demo" / "labels.json").read_text())

        assert labels["classes"] == ["mel", "nv"]
        assert labels["input_size"] == [224, 224]
        assert len(labels["mean"]) == 3

    def test_quantisation_shrinks_the_model(
        self, run: Path, project: Path, tmp_path: Path
    ) -> None:
        quantised = export_demo(
            run, image_dir=project / "images", output=tmp_path / "q", pool_size=2
        )
        full = export_demo(
            run,
            image_dir=project / "images",
            output=tmp_path / "f",
            pool_size=2,
            quantise=False,
        )

        assert quantised.model_bytes < full.model_bytes / 2

    def test_exported_model_runs_and_agrees_with_pytorch(
        self, run: Path, project: Path, tmp_path: Path
    ) -> None:
        """A model that exports but predicts differently is worse than useless."""
        import onnxruntime as ort
        import torch

        from ham10000.experiment import load_config
        from ham10000.models.architectures import build_classifier
        from ham10000.serialization import load_state_dict

        export_demo(
            run,
            image_dir=project / "images",
            output=tmp_path / "demo",
            pool_size=2,
            quantise=False,
        )

        config = load_config(run / "config.yaml")
        model = build_classifier(
            config.architecture, n_classes=2, strategy="head_only", pretrained=False
        )
        model.load_state_dict(load_state_dict(run / "model.pth"))
        model.eval()

        sample = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            expected = model(sample).numpy()

        session = ort.InferenceSession(
            str(tmp_path / "demo" / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        actual = session.run(None, {"image": sample.numpy()})[0]

        np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-4)

    def test_multiclass_run_is_rejected(self, tmp_path: Path) -> None:
        """A two-answer game cannot present a seven-class model."""
        fake = tmp_path / "run"
        fake.mkdir()
        (fake / "config.yaml").write_text("name: seven\nclasses: [mel, nv, bkl]\n")
        (fake / "model.pth").write_text("")
        pd.DataFrame({"dx": ["mel", "nv", "bkl"]}).to_csv(
            fake / "predictions.csv", index=False
        )

        with pytest.raises(ValueError, match="two answers"):
            export_demo(fake, image_dir=tmp_path, output=tmp_path / "out")


class TestSourceDimensions:
    def test_original_size_is_recorded(
        self, run: Path, project: Path, tmp_path: Path
    ) -> None:
        """The browser needs it to reproduce the training crop at the right scale.

        Training crops a fixed pixel window from the full-size original. The
        web images are smaller, so the same pixel window would take a different
        fraction of the lesion and present the network with a scale it never
        saw during training.
        """
        import json as json_module

        export_demo(
            run, image_dir=project / "images", output=tmp_path / "demo", pool_size=4
        )
        labels = json_module.loads((tmp_path / "demo" / "labels.json").read_text())

        assert labels["source_size"] == [160, 120]

    def test_web_images_are_smaller_than_the_source(
        self, run: Path, project: Path, tmp_path: Path
    ) -> None:
        from PIL import Image

        result = export_demo(
            run, image_dir=project / "images", output=tmp_path / "demo", pool_size=4
        )
        web = next((result.directory / "images").glob("*.jpg"))
        original = project / "images" / web.name

        with Image.open(web) as a, Image.open(original) as b:
            assert a.size[0] <= b.size[0]


class TestErrorMessages:
    def test_absent_directory_says_so(self, tmp_path: Path) -> None:
        """Naming a missing config.yaml inside a nonexistent path misleads."""
        with pytest.raises(FileNotFoundError, match="No such run directory"):
            export_demo(tmp_path / "nope", image_dir=tmp_path, output=tmp_path / "o")

    def test_incomplete_directory_names_what_is_missing(self, tmp_path: Path) -> None:
        partial = tmp_path / "run"
        partial.mkdir()
        (partial / "config.yaml").write_text("name: x\nclasses: [mel, nv]\n")

        with pytest.raises(FileNotFoundError, match=r"missing model\.pth"):
            export_demo(partial, image_dir=tmp_path, output=tmp_path / "o")
