"""Unit tests for :mod:`ham10000.experiment`.

The property under test is that a run identifier identifies a run. The
naming scheme did not: it was built from a few flags plus a hand-written
suffix, with collisions resolved by an auto-incrementing counter that records
execution order rather than configuration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ham10000.experiment import ExperimentConfig, build_transform, load_config
from ham10000.models.architectures import Architecture, FreezeStrategy

CONFIG_DIR = Path(__file__).parents[1] / "configs"


class TestRunIdentity:
    def test_identical_configurations_share_an_identifier(self) -> None:
        first = ExperimentConfig(name="a", classes=["mel", "nv"])
        second = ExperimentConfig(name="a", classes=["mel", "nv"])

        assert first.run_id == second.run_id

    def test_labels_do_not_affect_identity(self) -> None:
        """Renaming an experiment must not invent a new identity for it."""
        original = ExperimentConfig(name="a", classes=["mel"], description="x")
        renamed = ExperimentConfig(name="b", classes=["mel"], description="y")

        assert original.run_id == renamed.run_id

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("architecture", Architecture.EFFICIENTNET_B0),
            ("freeze_strategy", FreezeStrategy.HEAD_ONLY),
            ("pretrained", False),
            ("train_one_image_per_lesion", False),
            ("validation_expansion", 3),
            ("balance", {"mel": 2000}),
            ("n_blocks", 2),
        ],
    )
    def test_every_substantive_field_changes_identity(
        self, field: str, value: object
    ) -> None:
        base = ExperimentConfig(name="a", classes=["mel", "nv"])
        changed = ExperimentConfig(name="a", classes=["mel", "nv"], **{field: value})

        assert base.run_id != changed.run_id

    def test_class_specification_changes_identity(self) -> None:
        base = ExperimentConfig(name="a", classes=["mel", "nv"])
        changed = ExperimentConfig(name="a", classes=["mel", "nv", "bkl"])

        assert base.run_id != changed.run_id

    def test_transform_changes_identity(self) -> None:
        """The failure: crop-only and crop+jitter shared a name family."""
        crop = ExperimentConfig(
            name="a",
            classes=["mel"],
            transform=[{"name": "RandomCrop", "size": [300, 300]}],
        )
        crop_and_jitter = ExperimentConfig(
            name="a",
            classes=["mel"],
            transform=[
                {"name": "RandomCrop", "size": [300, 300]},
                {"name": "ColorJitter", "brightness": 0.4},
            ],
        )

        assert crop.run_id != crop_and_jitter.run_id

    def test_identifier_is_stable_across_processes(self) -> None:
        """A hash of a dict would vary run to run; this must not."""
        assert (
            ExperimentConfig(name="a", classes=["mel", "nv"]).run_id == "5d6e44f25579"
        )

    def test_adding_a_defaulted_field_does_not_change_identity(self) -> None:
        """The identifier must survive the schema growing.

        Only settings differing from their defaults enter the fingerprint. If
        every field entered it, adding an optional field would change the hash
        of every existing configuration and orphan every run directory on disk.
        """
        from dataclasses import replace

        base = ExperimentConfig(name="a", classes=["mel", "nv"])
        explicit_default = replace(base, restrict=None, n_blocks=1)

        assert base.run_id == explicit_default.run_id

    def test_only_non_default_settings_are_fingerprinted(self) -> None:
        config = ExperimentConfig(name="a", classes=["mel", "nv"], n_blocks=3)

        assert set(config.fingerprint()) == {"classes", "n_blocks"}

    def test_worker_count_does_not_change_identity(self) -> None:
        """`num_workers` is a machine-specific performance knob, not a setting.

        Including it meant the same experiment run with `--num-workers 4`
        landed in a different directory from the default, so `models/` showed
        two entries for one experiment.
        """
        from dataclasses import replace

        base = ExperimentConfig(name="a", classes=["mel", "nv"])
        parallel = replace(base, training=replace(base.training, num_workers=8))

        assert base.run_id == parallel.run_id

    def test_slug_contains_a_readable_label_and_the_hash(self) -> None:
        config = ExperimentConfig(name="Balanced TA run", classes=["mel"])

        assert config.slug.startswith("balanced-ta-run-")
        assert config.slug.endswith(config.run_id)


class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "absent.yaml")

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        """A silently ignored typo answers a different question than intended."""
        path = tmp_path / "e.yaml"
        path.write_text("name: x\nclasses: [mel]\nlernin_rate: 0.1\n")

        with pytest.raises(ValueError, match="Unknown key"):
            load_config(path)

    def test_nested_sections_become_typed_objects(self, tmp_path: Path) -> None:
        path = tmp_path / "e.yaml"
        path.write_text(
            "name: x\nclasses: [mel]\nsplit: {seed: 7}\ntraining: {epochs: 3}\n"
        )

        config = load_config(path)

        assert config.split.seed == 7
        assert config.training.epochs == 3

    def test_invalid_nested_values_are_rejected(self, tmp_path: Path) -> None:
        """Validation in SplitConfig/TrainingConfig applies to YAML too."""
        path = tmp_path / "e.yaml"
        path.write_text("name: x\nclasses: [mel]\ntraining: {epochs: 0}\n")

        with pytest.raises(ValueError, match="epochs must be at least 1"):
            load_config(path)

    def test_enum_fields_accept_strings(self, tmp_path: Path) -> None:
        path = tmp_path / "e.yaml"
        path.write_text(
            "name: x\nclasses: [mel]\narchitecture: efficientnet_b0\n"
            "freeze_strategy: head_only\n"
        )

        config = load_config(path)

        assert config.architecture is Architecture.EFFICIENTNET_B0
        assert config.freeze_strategy is FreezeStrategy.HEAD_ONLY


class TestShippedConfigs:
    def test_configs_directory_is_present(self) -> None:
        assert CONFIG_DIR.is_dir()

    @pytest.mark.parametrize(
        "path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name
    )
    def test_every_shipped_config_loads_and_builds(self, path: Path) -> None:
        config = load_config(path)

        assert config.name
        assert config.description.strip()
        build_transform(config.transform)

    def test_shipped_configs_have_distinct_identifiers(self) -> None:
        ids = [load_config(p).run_id for p in sorted(CONFIG_DIR.glob("*.yaml"))]

        assert len(set(ids)) == len(ids)


class TestSave:
    def test_config_is_written_beside_the_artefacts(self, tmp_path: Path) -> None:
        """So a reader of models/ can establish what a run actually was."""
        config = ExperimentConfig(name="demo", classes=["mel", "nv"])

        path = config.save(tmp_path / config.slug)

        written = yaml.safe_load(path.read_text())
        assert written["name"] == "demo"
        assert written["classes"] == ["mel", "nv"]

    def test_saved_config_round_trips_to_the_same_identifier(
        self, tmp_path: Path
    ) -> None:
        config = ExperimentConfig(
            name="demo",
            classes=["mel", "nv"],
            transform=[{"name": "ToTensor"}],
        )
        path = config.save(tmp_path / "run")

        assert load_config(path).run_id == config.run_id


class TestBuildTransform:
    def test_steps_are_built_in_order(self) -> None:
        pipeline = build_transform(
            [{"name": "Resize", "size": [224, 224]}, {"name": "ToTensor"}]
        )

        assert len(pipeline.transforms) == 2
        assert type(pipeline.transforms[0]).__name__ == "Resize"

    def test_empty_specification_is_allowed(self) -> None:
        assert build_transform([]).transforms == []

    def test_unknown_transform_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown transform"):
            build_transform([{"name": "Teleport"}])

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValueError, match="missing 'name'"):
            build_transform([{"size": [1, 1]}])

    def test_arguments_are_passed_through(self) -> None:
        pipeline = build_transform([{"name": "RandomRotation", "degrees": 180}])

        assert pipeline.transforms[0].degrees == [-180.0, 180.0]


class TestRunExperiment:
    """End-to-end: config in, run directory out."""

    @pytest.fixture
    def dataset(self, tmp_path: Path) -> tuple[Path, Path]:
        """Generate a small two-class dataset on disk."""
        import numpy as np
        import pandas as pd
        from PIL import Image

        image_dir = tmp_path / "images"
        image_dir.mkdir()
        rng = np.random.RandomState(0)

        rows = []
        counter = 0
        for lesion in range(10):
            dx = "mel" if lesion % 2 else "nv"
            for _ in range(2 if lesion < 4 else 1):
                image_id = f"I{counter:03d}"
                base = 200 if dx == "mel" else 40
                pixels = np.clip(
                    base + rng.randint(-20, 20, size=(48, 48, 3)), 0, 255
                ).astype("uint8")
                Image.fromarray(pixels).save(image_dir / f"{image_id}.jpg")
                rows.append({"lesion_id": f"L{lesion}", "image_id": image_id, "dx": dx})
                counter += 1

        metadata = tmp_path / "metadata.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)
        return metadata, image_dir

    @pytest.fixture
    def config(self) -> ExperimentConfig:
        from ham10000.models.training import TrainingConfig

        return ExperimentConfig(
            name="smoke test",
            classes=["mel", "nv"],
            pretrained=False,
            freeze_strategy=FreezeStrategy.HEAD_ONLY,
            training=TrainingConfig(epochs=1, batch_size=4),
            transform=[
                {"name": "Resize", "size": [32, 32]},
                {"name": "ToTensor"},
            ],
        )

    def test_run_writes_a_complete_run_directory(
        self, dataset: tuple[Path, Path], config: ExperimentConfig, tmp_path: Path
    ) -> None:
        from ham10000.experiment import run_experiment

        metadata, image_dir = dataset

        result = run_experiment(
            config,
            metadata_path=metadata,
            image_dir=image_dir,
            output_dir=tmp_path / "runs",
            device="cpu",
        )

        assert result.directory.name == config.slug
        for artefact in (
            "config.yaml",
            "model.pth",
            "model.losses.json",
            "predictions.csv",
            "metrics.json",
        ):
            assert (result.directory / artefact).is_file(), artefact

    def test_saved_config_identifies_the_run(
        self, dataset: tuple[Path, Path], config: ExperimentConfig, tmp_path: Path
    ) -> None:
        """The property the filename scheme lacked."""
        from ham10000.experiment import run_experiment

        metadata, image_dir = dataset

        result = run_experiment(
            config,
            metadata_path=metadata,
            image_dir=image_dir,
            output_dir=tmp_path / "runs",
            device="cpu",
        )

        recovered = load_config(result.directory / "config.yaml")
        assert recovered.run_id == config.run_id

    def test_metrics_are_reported_per_lesion(
        self, dataset: tuple[Path, Path], config: ExperimentConfig, tmp_path: Path
    ) -> None:
        """A lesion photographed twice must not count twice toward the score."""
        import pandas as pd

        from ham10000.experiment import run_experiment

        metadata, image_dir = dataset

        result = run_experiment(
            config,
            metadata_path=metadata,
            image_dir=image_dir,
            output_dir=tmp_path / "runs",
            device="cpu",
        )

        predictions = pd.read_csv(result.directory / "predictions.csv")
        assert result.report.n_samples == predictions["lesion_id"].nunique()

    def test_balanced_run_completes_and_scores_only_validation_lesions(
        self, dataset: tuple[Path, Path], tmp_path: Path
    ) -> None:
        import pandas as pd

        from ham10000.experiment import run_experiment
        from ham10000.models.training import TrainingConfig

        metadata, image_dir = dataset
        config = ExperimentConfig(
            name="balanced smoke",
            classes=["mel", "nv"],
            pretrained=False,
            freeze_strategy=FreezeStrategy.HEAD_ONLY,
            training=TrainingConfig(epochs=1, batch_size=4),
            balance={"mel": 6, "nv": 6},
            transform=[{"name": "Resize", "size": [32, 32]}, {"name": "ToTensor"}],
        )

        result = run_experiment(
            config,
            metadata_path=metadata,
            image_dir=image_dir,
            output_dir=tmp_path / "runs",
            device="cpu",
        )

        predictions = pd.read_csv(result.directory / "predictions.csv")
        metadata_frame = pd.read_csv(metadata)

        # Every scored lesion exists in the source data, and the scored set is
        # a strict subset of it -- the training lesions are absent.
        scored = set(predictions["lesion_id"])
        assert scored < set(metadata_frame["lesion_id"])
        assert result.report.n_samples == len(scored)
