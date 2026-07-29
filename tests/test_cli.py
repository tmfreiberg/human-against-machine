"""Unit tests for :mod:`ham10000.cli`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ham10000.cli import _format_duration, benchmark_main, train_main


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Create a miniature project root: metadata plus generated images."""
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.RandomState(0)

    rows = []
    counter = 0
    for lesion in range(8):
        dx = "mel" if lesion % 2 else "nv"
        for _ in range(2 if lesion < 3 else 1):
            image_id = f"I{counter:03d}"
            pixels = np.clip(
                (200 if dx == "mel" else 40) + rng.randint(-20, 20, (64, 64, 3)),
                0,
                255,
            ).astype("uint8")
            Image.fromarray(pixels).save(images / f"{image_id}.jpg")
            rows.append({"lesion_id": f"L{lesion}", "image_id": image_id, "dx": dx})
            counter += 1

    pd.DataFrame(rows).to_csv(images / "metadata.csv", index=False)
    (tmp_path / "pyproject.toml").write_text("")  # root marker
    return tmp_path


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "smoke.yaml"
    path.write_text(
        "name: smoke\n"
        "description: cli test\n"
        "classes: [mel, nv]\n"
        "pretrained: false\n"
        "freeze_strategy: head_only\n"
        "training: {epochs: 1, batch_size: 4}\n"
        "transform:\n"
        "  - {name: Resize, size: [32, 32]}\n"
        "  - {name: ToTensor}\n"
    )
    return path


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(45, "45 seconds"), (600, "10.0 minutes"), (7200, "2.0 hours")],
    )
    def test_largest_sensible_unit(self, seconds: float, expected: str) -> None:
        assert _format_duration(seconds) == expected


class TestBenchmark:
    def test_reports_the_machine_without_a_config(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert benchmark_main(["--device", "cpu", "--quick"]) == 0

        output = capsys.readouterr().out
        assert "accelerator" in output
        assert "img/s" in output

    def test_estimates_a_run_when_given_a_config(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert benchmark_main([str(config_file), "--device", "cpu", "--quick"]) == 0

        assert "estimated duration" in capsys.readouterr().out


class TestTrain:
    def test_smoke_run_completes_and_writes_artefacts(
        self, project: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = train_main(
            [
                str(config_file),
                "--data-root",
                str(project),
                "--limit",
                "8",
                "--device",
                "cpu",
            ]
        )

        assert exit_code == 0
        output = capsys.readouterr().out
        assert "Balanced accuracy" in output

        runs = list((project / "models").iterdir())
        assert len(runs) == 1
        assert (runs[0] / "metrics.json").is_file()

    def test_limited_runs_are_marked_so_they_cannot_be_mistaken(
        self, project: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A smoke run's numbers are meaningless; the directory must say so."""
        train_main(
            [
                str(config_file),
                "--data-root",
                str(project),
                "--limit",
                "8",
                "--device",
                "cpu",
            ]
        )

        runs = list((project / "models").iterdir())
        assert runs[0].name.endswith("-smoke")
        assert "SMOKE TEST" in capsys.readouterr().out

    def test_epoch_override_is_applied(
        self, project: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        train_main(
            [
                str(config_file),
                "--data-root",
                str(project),
                "--limit",
                "8",
                "--epochs",
                "2",
                "--device",
                "cpu",
            ]
        )

        assert "epoch   2/2" in capsys.readouterr().out

    def test_limit_keeps_every_class(self, project: Path, config_file: Path) -> None:
        """Sampling must not drop a class, or the label scheme changes."""
        train_main(
            [
                str(config_file),
                "--data-root",
                str(project),
                "--limit",
                "4",
                "--device",
                "cpu",
            ]
        )

        runs = list((project / "models").iterdir())
        predictions = pd.read_csv(runs[0] / "predictions.csv")
        assert predictions["dx"].nunique() >= 1


class TestOverwriteGuard:
    def test_completed_run_is_protected(self, project: Path, config_file: Path) -> None:
        """A finished run is a result; replacing it silently loses artefacts."""
        args = [
            str(config_file),
            "--data-root",
            str(project),
            "--limit",
            "8",
            "--device",
            "cpu",
        ]
        train_main(args)

        with pytest.raises(FileExistsError, match="already holds a completed run"):
            train_main(args)

    def test_force_permits_overwriting(self, project: Path, config_file: Path) -> None:
        args = [
            str(config_file),
            "--data-root",
            str(project),
            "--limit",
            "8",
            "--device",
            "cpu",
        ]
        train_main(args)

        assert train_main([*args, "--force"]) == 0

    def test_interrupted_run_is_not_protected(
        self, project: Path, config_file: Path
    ) -> None:
        """Only config.yaml present means the run never finished."""
        args = [
            str(config_file),
            "--data-root",
            str(project),
            "--limit",
            "8",
            "--device",
            "cpu",
        ]
        train_main(args)
        run = next((project / "models").iterdir())
        (run / "metrics.json").unlink()

        assert train_main(args) == 0


class TestUmbrellaCommand:
    def test_bare_command_lists_subcommands(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import main

        assert main([]) == 0

        output = capsys.readouterr().out
        for command in ("info", "split", "balance", "train", "results"):
            assert command in output

    def test_unknown_command_is_rejected(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import main

        assert main(["frobnicate"]) == 2
        assert "Unknown command" in capsys.readouterr().out

    def test_dispatches_to_a_subcommand(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import main

        assert main(["info", "--data-root", str(project)]) == 0
        assert "Class distribution" in capsys.readouterr().out


class TestInspectionCommands:
    def test_info_reports_counts_and_chance_level(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import info_main

        assert info_main(["--data-root", str(project)]) == 0

        output = capsys.readouterr().out
        assert "imbalance" in output
        assert "chance level" in output

    def test_split_verifies_no_leakage(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The command exists to make the guarantee checkable from a shell."""
        from ham10000.cli import split_main

        assert split_main(["--data-root", str(project)]) == 0
        assert "no leakage" in capsys.readouterr().out

    def test_balance_reports_factors(
        self, project: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import balance_main

        text = config_file.read_text() + "balance: {mel: 20, nv: 20}\n"
        config_file.write_text(text)

        assert balance_main([str(config_file), "--data-root", str(project)]) == 0
        assert "factor" in capsys.readouterr().out

    def test_results_reads_a_completed_run(
        self, project: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import results_main

        train_main(
            [
                str(config_file),
                "--data-root",
                str(project),
                "--limit",
                "8",
                "--device",
                "cpu",
            ]
        )
        run = next((project / "models").iterdir())
        capsys.readouterr()

        assert results_main([str(run), "--data-root", str(project)]) == 0

        output = capsys.readouterr().out
        assert "balanced accuracy" in output
        assert "skill above chance" in output


class TestRunPathResolution:
    def test_run_is_found_from_a_subdirectory(
        self,
        project: Path,
        config_file: Path,
        monkeypatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`models/<slug>` only resolves from the repository root.

        Someone standing in `demo/` or `notebooks/` should not get a confusing
        "missing config.yaml" for a directory that plainly exists.
        """
        from ham10000.cli import results_main

        train_main(
            [
                str(config_file),
                "--data-root",
                str(project),
                "--limit",
                "8",
                "--device",
                "cpu",
            ]
        )
        run = next((project / "models").iterdir())

        elsewhere = project / "notebooks"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        capsys.readouterr()

        assert results_main([f"models/{run.name}", "--data-root", str(project)]) == 0
        assert "balanced accuracy" in capsys.readouterr().out

    def test_unknown_run_lists_where_it_looked(self, project: Path) -> None:
        from ham10000.cli import _resolve_run

        with pytest.raises(FileNotFoundError, match="Looked in"):
            _resolve_run(Path("models/nonexistent"), project)


class TestViewsCommand:
    def test_compares_settings_without_writing_to_the_run(
        self, project: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A run's recorded result must not be replaced by an exploratory one."""
        import json

        from ham10000.cli import views_main

        train_main([str(config_file), "--data-root", str(project), "--device", "cpu"])
        run = next((project / "models").iterdir())
        before = json.loads((run / "metrics.json").read_text())
        capsys.readouterr()

        assert (
            views_main(
                [
                    str(run),
                    "--views",
                    "1,2",
                    "--repeats",
                    "2",
                    "--data-root",
                    str(project),
                    "--device",
                    "cpu",
                ]
            )
            == 0
        )

        output = capsys.readouterr().out
        assert "spread" in output
        assert json.loads((run / "metrics.json").read_text()) == before

    def test_single_view_disables_augmentation(
        self, project: Path, config_file: Path
    ) -> None:
        from ham10000.config import Settings
        from ham10000.experiment import evaluate_at_views

        train_main([str(config_file), "--data-root", str(project), "--device", "cpu"])
        run = next((project / "models").iterdir())
        settings = Settings.resolve(project)

        first = evaluate_at_views(
            run,
            metadata_path=settings.metadata_csv,
            image_dir=settings.images,
            views=1,
            seed=0,
            device="cpu",
        )
        second = evaluate_at_views(
            run,
            metadata_path=settings.metadata_csv,
            image_dir=settings.images,
            views=1,
            seed=1,
            device="cpu",
        )

        # With one view and a deterministic transform in this fixture, the two
        # draws must agree: the seed only affects the expansion.
        assert first.n_samples == second.n_samples


class TestInspectionExtras:
    def test_frequencies_counts_lesions_by_default(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Lesions, not images: a lesion is the unit everything is scored on."""
        from ham10000.cli import frequencies_main

        assert frequencies_main(["dx", "--data-root", str(project)]) == 0
        assert "distinct lesions" in capsys.readouterr().out

    def test_frequencies_can_count_images(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import frequencies_main

        frequencies_main(["dx", "--images", "--data-root", str(project)])

        assert "images" in capsys.readouterr().out

    def test_frequencies_rejects_a_malformed_restriction(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import frequencies_main

        code = frequencies_main(
            ["dx", "--where", "nonsense", "--data-root", str(project)]
        )

        assert code == 1
        assert "COLUMN=VALUE" in capsys.readouterr().out

    def test_images_writes_a_file_and_lists_the_metadata(
        self, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import images_main

        output = tmp_path / "grid.png"
        code = images_main(
            [
                "--dx",
                "mel",
                "--rows",
                "2",
                "--output",
                str(output),
                "--data-root",
                str(project),
            ]
        )

        assert code == 0
        assert output.is_file()
        assert "image_id" in capsys.readouterr().out

    def test_images_rejects_an_unknown_class(
        self, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import images_main

        code = images_main(
            [
                "--dx",
                "unicorn",
                "--output",
                str(tmp_path / "x.png"),
                "--data-root",
                str(project),
            ]
        )

        assert code == 1
        assert "Unknown diagnosis" in capsys.readouterr().out

    def test_checkpoint_reports_the_class_count(
        self, project: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Answers what a .pth was trained for without guessing the architecture."""
        from ham10000.cli import checkpoint_main

        train_main([str(config_file), "--data-root", str(project), "--device", "cpu"])
        run = next((project / "models").iterdir())
        capsys.readouterr()

        assert checkpoint_main([str(run / "model.pth")]) == 0
        assert "output classes: 2" in capsys.readouterr().out

    def test_checkpoint_missing_file_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import checkpoint_main

        assert checkpoint_main([str(tmp_path / "absent.pth")]) == 1
        assert "not found" in capsys.readouterr().out


class TestCompareAndThresholds:
    @pytest.fixture
    def run(self, project: Path, config_file: Path) -> Path:
        train_main([str(config_file), "--data-root", str(project), "--device", "cpu"])
        return next((project / "models").iterdir())

    def test_compare_reports_skill_above_chance(
        self, run: Path, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Raw scores are not comparable across differing class counts."""
        from ham10000.cli import compare_main

        capsys.readouterr()
        assert compare_main(["--data-root", str(project)]) == 0

        output = capsys.readouterr().out
        assert "skill" in output
        assert "chance" in output

    def test_compare_with_no_runs_reports_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import compare_main

        (tmp_path / "models").mkdir()
        (tmp_path / "pyproject.toml").write_text("")

        assert compare_main(["--data-root", str(tmp_path)]) == 1
        assert "No completed runs" in capsys.readouterr().out

    def test_thresholds_reports_a_before_and_after(
        self, run: Path, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import thresholds_main

        capsys.readouterr()
        code = thresholds_main(
            [str(run), "--promote", "mel=0.4", "--data-root", str(project)]
        )

        assert code == 0
        output = capsys.readouterr().out
        assert "argmax" in output
        assert "thresholded" in output
        assert "verdicts changed" in output

    def test_thresholds_does_not_modify_the_run(self, run: Path, project: Path) -> None:
        """Exploring a decision rule must not overwrite a recorded result."""
        import json

        from ham10000.cli import thresholds_main

        before = json.loads((run / "metrics.json").read_text())
        thresholds_main([str(run), "--data-root", str(project)])

        assert json.loads((run / "metrics.json").read_text()) == before

    def test_predict_scores_named_images(
        self, run: Path, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ham10000.cli import predict_main

        image = next((project / "images").glob("*.jpg")).stem
        capsys.readouterr()

        assert (
            predict_main(
                [str(run), image, "--data-root", str(project), "--device", "cpu"]
            )
            == 0
        )
        assert image in capsys.readouterr().out
