"""Unit tests for :mod:`ham10000.models.architectures`.

`TestFreezingActuallyFreezes` is the important class here. Torchvision returns
pretrained models with every parameter already trainable, so an implementation
that only ever assigns ``requires_grad = True`` silently trains the whole
network whatever strategy is asked for. These tests fail against that mistake
rather than letting it pass unnoticed.
"""

from __future__ import annotations

import pytest
from torch import nn
from torchvision import models

from ham10000.models.architectures import (
    Architecture,
    FreezeStrategy,
    apply_freezing,
    build_classifier,
    replace_head,
    trainable_parameters,
)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestReplaceHead:
    def test_resnet_head_is_resized(self) -> None:
        assert replace_head(models.resnet18(weights=None), 5).fc.out_features == 5

    def test_efficientnet_head_is_resized(self) -> None:
        model = replace_head(models.efficientnet_b0(weights=None), 3)

        assert model.classifier[1].out_features == 3

    def test_backbone_width_is_preserved(self) -> None:
        """The new head must consume the backbone's feature width unchanged."""
        model = models.resnet18(weights=None)
        width = model.fc.in_features

        assert replace_head(model, 7).fc.in_features == width

    def test_unsupported_architecture_raises(self) -> None:
        """A naive implementation silently left the head at 1000 ImageNet outputs."""
        with pytest.raises(TypeError, match="Cannot replace the head"):
            replace_head(nn.Linear(10, 10), 5)

    @pytest.mark.parametrize("n_classes", [0, 1, -3])
    def test_degenerate_class_counts_are_rejected(self, n_classes: int) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            replace_head(models.resnet18(weights=None), n_classes)


class TestFreezingActuallyFreezes:
    def test_last_block_freezes_early_layers(self) -> None:
        """The regression. In  this returned every parameter."""
        model = replace_head(models.resnet18(weights=None), 5)

        trainable = apply_freezing(model, FreezeStrategy.LAST_BLOCK)

        assert not any(name.startswith("layer1") for name in trainable)
        assert not any(name.startswith("conv1") for name in trainable)
        assert any(name.startswith("layer4") for name in trainable)
        assert any(name.startswith("fc") for name in trainable)

    def test_last_block_trains_strictly_fewer_parameters_than_all(self) -> None:
        """In  these two configurations were numerically identical."""
        partial = replace_head(models.resnet18(weights=None), 5)
        apply_freezing(partial, FreezeStrategy.LAST_BLOCK)

        everything = replace_head(models.resnet18(weights=None), 5)
        apply_freezing(everything, FreezeStrategy.ALL)

        assert count_trainable(partial) < count_trainable(everything)

    def test_head_only_trains_just_the_head(self) -> None:
        model = replace_head(models.resnet18(weights=None), 5)

        trainable = apply_freezing(model, FreezeStrategy.HEAD_ONLY)

        assert trainable == {"fc.weight", "fc.bias"}

    def test_all_leaves_everything_trainable(self) -> None:
        model = replace_head(models.resnet18(weights=None), 5)

        trainable = apply_freezing(model, FreezeStrategy.ALL)

        assert len(trainable) == len(list(model.parameters()))

    def test_freezing_is_idempotent(self) -> None:
        """Re-applying must not accumulate; it re-freezes from scratch."""
        model = replace_head(models.resnet18(weights=None), 5)

        once = apply_freezing(model, FreezeStrategy.HEAD_ONLY)
        twice = apply_freezing(model, FreezeStrategy.HEAD_ONLY)

        assert once == twice

    def test_strategies_can_be_switched_in_either_direction(self) -> None:
        """Narrowing after widening must actually narrow."""
        model = replace_head(models.resnet18(weights=None), 5)

        apply_freezing(model, FreezeStrategy.ALL)
        apply_freezing(model, FreezeStrategy.HEAD_ONLY)

        assert count_trainable(model) == 2565

    def test_efficientnet_partial_unfreeze_is_supported(self) -> None:
        """The repository version printed a message and returned without training."""
        model = replace_head(models.efficientnet_b0(weights=None), 5)

        trainable = apply_freezing(model, FreezeStrategy.LAST_BLOCK, n_blocks=2)

        assert trainable
        assert count_trainable(model) < sum(p.numel() for p in model.parameters())
        assert any(name.startswith("classifier") for name in trainable)
        assert not any(name.startswith("features.0") for name in trainable)

    def test_more_blocks_means_more_trainable_parameters(self) -> None:
        counts = []
        for n_blocks in (1, 2, 3):
            model = replace_head(models.resnet18(weights=None), 5)
            apply_freezing(model, FreezeStrategy.LAST_BLOCK, n_blocks=n_blocks)
            counts.append(count_trainable(model))

        assert counts == sorted(counts)
        assert len(set(counts)) == 3


class TestTrainableParameters:
    def test_yields_only_unfrozen_parameters(self) -> None:
        model = replace_head(models.resnet18(weights=None), 5)
        apply_freezing(model, FreezeStrategy.HEAD_ONLY)

        assert len(list(trainable_parameters(model))) == 2

    def test_optimiser_receives_no_frozen_tensors(self) -> None:
        """Frozen tensors in an optimiser allocate state that never updates."""
        model = replace_head(models.resnet18(weights=None), 5)
        apply_freezing(model, FreezeStrategy.LAST_BLOCK)

        assert all(p.requires_grad for p in trainable_parameters(model))


class TestBuildClassifier:
    @pytest.mark.parametrize("architecture", list(Architecture))
    def test_every_architecture_builds(self, architecture: Architecture) -> None:
        model = build_classifier(architecture, n_classes=4, pretrained=False)

        assert count_trainable(model) > 0

    def test_accepts_plain_strings(self) -> None:
        model = build_classifier(
            "resnet18", n_classes=2, strategy="head_only", pretrained=False
        )

        assert count_trainable(model) == 1026

    def test_unknown_architecture_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid Architecture"):
            build_classifier("vgg16", n_classes=2, pretrained=False)
