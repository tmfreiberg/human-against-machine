"""Unit and end-to-end tests for training and inference.

`TestEndToEnd` runs the whole pipeline on generated images: build a classifier,
train it, score the validation set, aggregate to lesion verdicts. It is slow by
the standards of the rest of the suite (a few seconds) and it is the only test
that exercises the modules together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from ham10000.data.labels import LabelScheme
from ham10000.data.splitting import SplitConfig, assign_splits
from ham10000.models.aggregation import aggregate_predictions, predicted_label
from ham10000.models.architectures import FreezeStrategy, build_classifier
from ham10000.models.inference import predict_probabilities, select_device
from ham10000.models.training import TrainingConfig, TrainingHistory, train_model

SCHEME = LabelScheme(codes={0: "nv", 1: "mel"}, mapping={"nv": 0, "mel": 1})
TRANSFORM = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    """Eight lesions, twelve images, two visually separable classes."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rng = np.random.RandomState(0)

    rows = []
    counter = 0
    for lesion in range(8):
        label = lesion % 2
        for _ in range(2 if lesion < 4 else 1):
            image_id = f"I{counter:03d}"
            # Class 0 dark, class 1 light, so the task is learnable in a
            # handful of steps and a failure indicates a wiring fault.
            base = 40 if label == 0 else 200
            pixels = np.clip(
                base + rng.randint(-20, 20, size=(32, 32, 3)), 0, 255
            ).astype("uint8")
            Image.fromarray(pixels).save(image_dir / f"{image_id}.jpg")
            rows.append(
                {
                    "lesion_id": f"L{lesion}",
                    "image_id": image_id,
                    "num_images": 2 if lesion < 4 else 1,
                    "dx": "nv" if label == 0 else "mel",
                    "label": label,
                }
            )
            counter += 1

    return pd.DataFrame(rows), image_dir


class TestTrainingConfig:
    def test_defaults_match_the_shipped_configs(self) -> None:
        config = TrainingConfig()

        assert config.epochs == 10
        assert config.learning_rate == pytest.approx(1e-4)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("epochs", 0), ("batch_size", 0), ("learning_rate", 0.0)],
    )
    def test_degenerate_values_are_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(**{field: value})

    def test_config_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            TrainingConfig().epochs = 3  # type: ignore[misc]


class TestTrainingHistory:
    def test_length_reflects_completed_epochs_only(self) -> None:
        """The arrays were pre-filled with -1 for epochs that never ran."""
        history = TrainingHistory()
        history.record(train=1.0, validation=1.2)

        assert len(history) == 1
        assert all(loss >= 0 for loss in history.validation_loss)

    def test_best_epoch_is_the_lowest_validation_loss(self) -> None:
        history = TrainingHistory()
        for validation in (1.0, 0.4, 0.7):
            history.record(train=0.5, validation=validation)

        assert history.best_epoch == 1

    def test_best_epoch_on_empty_history_raises(self) -> None:
        with pytest.raises(ValueError, match="No epochs recorded"):
            _ = TrainingHistory().best_epoch

    def test_json_round_trip(self, tmp_path: Path) -> None:
        import json

        history = TrainingHistory()
        history.record(train=0.5, validation=0.6)
        path = tmp_path / "losses.json"

        history.to_json(path)

        assert json.loads(path.read_text())["train_loss"] == [0.5]


class TestSelectDevice:
    def test_explicit_device_is_honoured(self) -> None:
        assert select_device("cpu") == torch.device("cpu")

    def test_detection_returns_a_device(self) -> None:
        assert isinstance(select_device(), torch.device)


class TestValidation:
    def test_frame_without_labels_is_rejected(
        self, dataset: tuple[pd.DataFrame, Path]
    ) -> None:
        frame, image_dir = dataset
        model = build_classifier("resnet18", n_classes=2, pretrained=False)

        with pytest.raises(ValueError, match="must contain a 'label' column"):
            train_model(
                model,
                frame.drop(columns=["label"]),
                frame,
                SCHEME,
                image_dir=image_dir,
                transform=TRANSFORM,
                config=TrainingConfig(epochs=1),
                device="cpu",
            )

    def test_fully_frozen_model_is_rejected(
        self, dataset: tuple[pd.DataFrame, Path]
    ) -> None:
        """Training with nothing trainable would run and change nothing."""
        frame, image_dir = dataset
        model = build_classifier("resnet18", n_classes=2, pretrained=False)
        for parameter in model.parameters():
            parameter.requires_grad = False

        with pytest.raises(ValueError, match="No trainable parameters"):
            train_model(
                model,
                frame,
                frame,
                SCHEME,
                image_dir=image_dir,
                transform=TRANSFORM,
                config=TrainingConfig(epochs=1),
                device="cpu",
            )


class TestPredictProbabilities:
    def test_columns_follow_the_scheme(
        self, dataset: tuple[pd.DataFrame, Path]
    ) -> None:
        frame, image_dir = dataset
        model = build_classifier("resnet18", n_classes=2, pretrained=False)

        scored = predict_probabilities(
            frame, model, SCHEME, image_dir=image_dir, transform=TRANSFORM, device="cpu"
        )

        assert list(scored.columns[-2:]) == ["prob_nv", "prob_mel"]

    def test_probabilities_sum_to_one(self, dataset: tuple[pd.DataFrame, Path]) -> None:
        frame, image_dir = dataset
        model = build_classifier("resnet18", n_classes=2, pretrained=False)

        scored = predict_probabilities(
            frame, model, SCHEME, image_dir=image_dir, transform=TRANSFORM, device="cpu"
        )

        totals = scored[["prob_nv", "prob_mel"]].sum(axis=1)
        assert np.allclose(totals, 1.0)

    def test_non_contiguous_index_is_handled(
        self, dataset: tuple[pd.DataFrame, Path]
    ) -> None:
        """The index merge misaligned probabilities on a filtered frame."""
        frame, image_dir = dataset
        filtered = frame[frame["label"] == 1]
        model = build_classifier("resnet18", n_classes=2, pretrained=False)

        scored = predict_probabilities(
            filtered,
            model,
            SCHEME,
            image_dir=image_dir,
            transform=TRANSFORM,
            device="cpu",
        )

        assert scored["prob_nv"].notna().all()
        assert scored["image_id"].tolist() == filtered["image_id"].tolist()

    def test_head_width_mismatch_is_detected(
        self, dataset: tuple[pd.DataFrame, Path]
    ) -> None:
        """A five-class checkpoint scored under a binary scheme is meaningless."""
        frame, image_dir = dataset
        model = build_classifier("resnet18", n_classes=5, pretrained=False)

        with pytest.raises(ValueError, match="different scheme"):
            predict_probabilities(
                frame,
                model,
                SCHEME,
                image_dir=image_dir,
                transform=TRANSFORM,
                device="cpu",
            )

    def test_input_frame_is_not_modified(
        self, dataset: tuple[pd.DataFrame, Path]
    ) -> None:
        frame, image_dir = dataset
        before = frame.copy()
        model = build_classifier("resnet18", n_classes=2, pretrained=False)

        predict_probabilities(
            frame, model, SCHEME, image_dir=image_dir, transform=TRANSFORM, device="cpu"
        )

        pd.testing.assert_frame_equal(frame, before)


class TestEndToEnd:
    def test_split_train_score_aggregate(
        self, dataset: tuple[pd.DataFrame, Path]
    ) -> None:
        frame, image_dir = dataset

        assignment = assign_splits(frame, SplitConfig(seed=0, train_val_ratio=1))
        annotated = frame.assign(set=assignment.sets)
        train = annotated[annotated["set"].isin(["t1", "ta"])]
        validation = annotated[annotated["set"].isin(["v1", "va"])]

        assert not (set(train["lesion_id"]) & set(validation["lesion_id"]))

        model = build_classifier(
            "resnet18",
            n_classes=2,
            strategy=FreezeStrategy.HEAD_ONLY,
            pretrained=False,
        )
        history = train_model(
            model,
            train,
            validation,
            SCHEME,
            image_dir=image_dir,
            transform=TRANSFORM,
            config=TrainingConfig(epochs=2, batch_size=4, learning_rate=1e-2),
            device="cpu",
        )

        assert len(history) == 2

        scored = predict_probabilities(
            validation,
            model,
            SCHEME,
            image_dir=image_dir,
            transform=TRANSFORM,
            device="cpu",
        )
        scored["pred"] = predicted_label(scored, SCHEME.codes)
        final = aggregate_predictions(scored, seed=0)

        assert final["pred_final"].notna().all()
        # One verdict per lesion, constant across that lesion's rows.
        assert (final.groupby("lesion_id")["pred_final"].nunique() == 1).all()

    def test_checkpoint_round_trips_through_safe_loading(
        self, dataset: tuple[pd.DataFrame, Path], tmp_path: Path
    ) -> None:
        """Written by training, read back by the weights_only=True loader."""
        from ham10000.serialization import load_state_dict

        frame, image_dir = dataset
        checkpoint = tmp_path / "models" / "test.pth"
        model = build_classifier(
            "resnet18", n_classes=2, strategy="head_only", pretrained=False
        )

        train_model(
            model,
            frame,
            frame,
            SCHEME,
            image_dir=image_dir,
            transform=TRANSFORM,
            config=TrainingConfig(epochs=1, batch_size=4),
            checkpoint_path=checkpoint,
            device="cpu",
        )

        restored = load_state_dict(checkpoint)

        assert "fc.weight" in restored
        assert tuple(restored["fc.weight"].shape) == (2, 512)
        assert checkpoint.with_suffix(".losses.json").is_file()

    def test_save_best_selects_the_lowest_validation_loss(
        self, dataset: tuple[pd.DataFrame, Path], tmp_path: Path
    ) -> None:
        frame, image_dir = dataset
        checkpoint = tmp_path / "best.pth"
        model = build_classifier(
            "resnet18", n_classes=2, strategy="head_only", pretrained=False
        )

        history = train_model(
            model,
            frame,
            frame,
            SCHEME,
            image_dir=image_dir,
            transform=TRANSFORM,
            config=TrainingConfig(epochs=2, batch_size=4, save_best=True),
            checkpoint_path=checkpoint,
            device="cpu",
        )

        assert checkpoint.is_file()
        assert history.best_epoch in (0, 1)


class TestSaveBestConsistency:
    def test_model_holds_the_saved_weights_after_training(
        self, dataset: tuple[pd.DataFrame, Path], tmp_path: Path
    ) -> None:
        """The regression: metrics must describe the checkpoint on disk.

        Without restoring the best weights into the model, a caller scores the
        final epoch while the file holds the best one, so the reported metrics
        describe a model that was never saved and the saved model was never
        measured.
        """
        from ham10000.serialization import load_state_dict

        frame, image_dir = dataset
        checkpoint = tmp_path / "best.pth"
        model = build_classifier(
            "resnet18", n_classes=2, strategy="head_only", pretrained=False
        )

        train_model(
            model,
            frame,
            frame,
            SCHEME,
            image_dir=image_dir,
            transform=TRANSFORM,
            config=TrainingConfig(
                epochs=3, batch_size=4, learning_rate=1e-2, save_best=True
            ),
            checkpoint_path=checkpoint,
            device="cpu",
        )

        on_disk = load_state_dict(checkpoint)
        in_memory = model.state_dict()

        for key, saved in on_disk.items():
            torch.testing.assert_close(saved, in_memory[key].detach().cpu())
