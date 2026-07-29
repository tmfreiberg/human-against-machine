"""Declarative experiment configuration.

An experiment is a YAML file, and one runner reads it. Adding an experiment
means adding a file, and the difference between two experiments is a diff
between two configs rather than a diff between two near-duplicate notebooks.

Why the run identifier is a hash
--------------------------------
A name built from a few flags plus a hand-written suffix is not an identifier,
because it does not depend on most of the configuration. Two runs differing in
augmentation, or in the balancing target, can land on the same name and be
distinguished only by a trailing counter that records the order they happened
to execute in. Anything that later selects a model by name then has no way to
know which configuration produced it.

:meth:`ExperimentConfig.run_id` derives from a hash of the entire resolved
configuration, and :meth:`ExperimentConfig.save` writes the config alongside
the artefacts. A run directory therefore identifies exactly one configuration,
and the claim can be checked rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from ham10000.data.splitting import SplitConfig
from ham10000.models.options import (
    Architecture,
    FreezeStrategy,
    TrainingConfig,
)

__all__ = ["ExperimentConfig", "build_transform", "load_config"]


def _as_nested_dict(value: object) -> dict[str, Any] | None:
    """Return a dataclass *instance* as a dict, or None if it is not one.

    `dataclasses.is_dataclass` is true for both instances and classes, while
    `asdict` accepts only instances, so the narrowing has to be explicit.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return None


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """A complete, reproducible description of one experiment.

    Every field that affects the outcome lives here, which is what makes
    :meth:`run_id` a genuine identifier.

    Parameters
    ----------
    name:
        Human-readable label, used for display only. Deliberately excluded from
        the hash: renaming an experiment must not invent a new identity for it.
    description:
        Prose note on what the experiment tests. Also excluded from the hash.
    classes:
        Either a list of diagnoses to model individually, or a mapping from
        class name to the diagnoses it groups.
    split:
        Train/validation split parameters.
    architecture, freeze_strategy, n_blocks, pretrained:
        Model construction.
    training:
        Hyperparameters.
    transform:
        Ordered torchvision transform specifications; see
        :func:`build_transform`.
    restrict:
        Column to permitted values, applied to the whole dataset before
        anything else. `{"dx": ["mel", "nv"]}` narrows the problem to those two
        diagnoses, which is how a demo model is trained to answer a two-way
        question without a third class it could fall back on.
    balance:
        Per-class target counts for training-set resampling. Omitted means no
        balancing.
    train_one_image_per_lesion:
        Use one designated image per training lesion rather than all of them.
    validation_expansion:
        Repeat each validation lesion this many times for test-time
        augmentation. Only meaningful with a stochastic transform.

    Examples
    --------
    >>> config = ExperimentConfig(name="demo", classes=["mel", "nv"])
    >>> config.run_id[:4].isalnum()
    True

    The identifier depends on the configuration, not on the label:

    >>> renamed = ExperimentConfig(name="different", classes=["mel", "nv"])
    >>> config.run_id == renamed.run_id
    True

    But any substantive change gives a new identity:

    >>> changed = ExperimentConfig(name="demo", classes=["mel", "nv", "bkl"])
    >>> config.run_id == changed.run_id
    False
    """

    name: str
    classes: Sequence[str] | Mapping[str, Sequence[str]]
    description: str = ""
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    architecture: Architecture = Architecture.RESNET18
    freeze_strategy: FreezeStrategy = FreezeStrategy.LAST_BLOCK
    n_blocks: int = 1
    pretrained: bool = True
    transform: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    restrict: Mapping[str, Sequence[str]] | None = None
    balance: Mapping[str, int] | None = None
    train_one_image_per_lesion: bool = True
    validation_expansion: int | None = None

    @staticmethod
    def _plain(value: Any) -> Any:
        """Coerce enums and tuples to YAML- and JSON-representable values.

        `StrEnum` members are `str` subclasses, so JSON already serialises them
        by value and the hash is unaffected. PyYAML, however, refuses to
        represent them, which would make `save` fail on any configuration that
        names an architecture.
        """
        if isinstance(value, StrEnum):
            return str(value)
        if isinstance(value, Mapping):
            return {k: ExperimentConfig._plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ExperimentConfig._plain(v) for v in value]
        return value

    def fingerprint(self) -> dict[str, Any]:
        """Return the outcome-determining configuration as plain data.

        Only settings that differ from their defaults are included. This is
        what makes the identifier stable as the schema grows: adding a new
        optional field with a default would otherwise change the hash of every
        existing configuration and orphan every run directory already on disk.

        Also excluded regardless of value: `name` and `description`, which are
        labels rather than settings, and `training.num_workers`, which is a
        machine-specific performance knob. Including the last one meant running
        the same experiment with `--num-workers 4` produced a different
        identifier from running it with the default, so `models/` showed two
        entries for one experiment.

        Examples
        --------
        >>> "name" in ExperimentConfig(name="x", classes=["mel"]).fingerprint()
        False
        """
        defaults = {
            field.name: (
                field.default_factory()
                if field.default_factory is not MISSING
                else field.default
            )
            for field in fields(self)
        }

        data = asdict(self)
        data.pop("name")
        data.pop("description")
        data["training"].pop("num_workers", None)

        significant: dict[str, Any] = {}
        for key, value in data.items():
            default = defaults.get(key)
            nested_default = _as_nested_dict(default)
            if nested_default is not None and isinstance(value, dict):
                default_fields = dict(nested_default)
                default_fields.pop("num_workers", None)
                nested = {
                    inner: inner_value
                    for inner, inner_value in value.items()
                    if default_fields.get(inner) != inner_value
                }
                if nested:
                    significant[key] = nested
            elif value != (nested_default if nested_default is not None else default):
                significant[key] = value

        return {key: self._plain(value) for key, value in significant.items()}

    @property
    def run_id(self) -> str:
        """Short stable hash of the configuration.

        Twelve hex characters, which is ample for the tens of experiments this
        project will ever have and short enough to type.

        Examples
        --------
        >>> len(ExperimentConfig(name="x", classes=["mel"]).run_id)
        12
        """
        payload = json.dumps(self.fingerprint(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def slug(self) -> str:
        """Directory name: readable label plus the identifying hash.

        The label aids a human scanning a directory listing; the hash is what
        makes the name unique.

        Examples
        --------
        >>> ExperimentConfig(name="Balanced TA", classes=["mel"]).slug[:11]
        'balanced-ta'
        """
        label = "".join(
            character.lower() if character.isalnum() else "-" for character in self.name
        ).strip("-")
        while "--" in label:
            label = label.replace("--", "-")
        return f"{label}-{self.run_id}"

    def save(self, directory: Path) -> Path:
        """Write the configuration beside the artefacts it produced.

        This is what lets a reader of `models/` establish what any given run
        actually was, without having to trust the directory name.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                {"name": self.name, "description": self.description}
                | self.fingerprint(),
                sort_keys=False,
                default_flow_style=False,
            )
        )
        return path


def load_config(path: Path | str) -> ExperimentConfig:
    r"""Read an experiment configuration from YAML.

    Parameters
    ----------
    path:
        Path to a `.yaml` file.

    Returns
    -------
    ExperimentConfig

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    ValueError
        On an unknown key. A silently ignored typo in a config file is a
        particularly unpleasant failure: the run completes, reports plausible
        numbers, and answers a different question than intended.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "e.yaml"
    ...     _ = path.write_text("name: demo\nclasses: [mel, nv]\n")
    ...     load_config(path).name
    'demo'

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "e.yaml"
    ...     _ = path.write_text("name: demo\nclasses: [mel]\nlernin_rate: 3\n")
    ...     load_config(path)
    Traceback (most recent call last):
        ...
    ValueError: Unknown key(s) in ...
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    known = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {path}: {sorted(unknown)}. Known keys: {sorted(known)}."
        )

    if "split" in raw:
        raw["split"] = SplitConfig(**raw["split"])
    if "training" in raw:
        raw["training"] = TrainingConfig(**raw["training"])
    if "architecture" in raw:
        raw["architecture"] = Architecture(raw["architecture"])
    if "freeze_strategy" in raw:
        raw["freeze_strategy"] = FreezeStrategy(raw["freeze_strategy"])

    return ExperimentConfig(**raw)


def build_transform(specification: Sequence[Mapping[str, Any]]) -> Any:
    """Construct a torchvision transform pipeline from declarative specs.

    Parameters
    ----------
    specification:
        Ordered list of mappings, each with a `name` naming a
        `torchvision.transforms` class and the remaining keys passed as
        keyword arguments.

    Returns
    -------
    torchvision.transforms.Compose

    Raises
    ------
    ValueError
        On an unknown transform name, or one that is not a torchvision
        transform. The check matters: `getattr` on a module will happily return
        something that is not a transform at all.

    Examples
    --------
    >>> pipeline = build_transform(
    ...     [
    ...         {"name": "Resize", "size": [224, 224]},
    ...         {"name": "ToTensor"},
    ...     ]
    ... )
    >>> len(pipeline.transforms)
    2

    >>> build_transform([{"name": "Teleport"}])
    Traceback (most recent call last):
        ...
    ValueError: Unknown transform 'Teleport'...
    """
    from torchvision import transforms

    steps = []
    for spec in specification:
        parameters = dict(spec)
        name = parameters.pop("name", None)
        if name is None:
            raise ValueError(f"Transform specification is missing 'name': {spec}.")
        factory = getattr(transforms, str(name), None)
        if factory is None or not callable(factory):
            raise ValueError(
                f"Unknown transform {name!r}. Must be a class in "
                "torchvision.transforms."
            )
        steps.append(factory(**parameters))

    return transforms.Compose(steps)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Outcome of a run: where the artefacts are and how the model scored.

    Parameters
    ----------
    config:
        The configuration that produced this result.
    directory:
        Run directory holding the checkpoint, the resolved config, and the
        metrics.
    report:
        Metrics on the validation set, aggregated to one verdict per lesion.
    history:
        Per-epoch losses.
    """

    config: ExperimentConfig
    directory: Path
    report: Any
    history: Any


def rescore_run(
    run: Path,
    *,
    image_dir: Path,
    device: str | None = None,
) -> ExperimentResult:
    """Recompute a run's predictions and metrics from its saved checkpoint.

    For repairing a run whose metrics do not describe its checkpoint. That
    happened for runs trained with `save_best` before the fix: the best weights
    were written to `model.pth` while the reported metrics were computed from
    the final epoch still held in memory.

    Re-uses the validation rows recorded in the run's existing
    `predictions.csv` rather than reconstructing the split, so the rescored
    metrics cover exactly the same lesions as the previous ones and the two
    are directly comparable.

    Parameters
    ----------
    run:
        A completed run directory.
    image_dir:
        Directory containing the images.
    device:
        Compute device. Detected when omitted.

    Returns
    -------
    ExperimentResult
        With `predictions.csv` and `metrics.json` rewritten in place.
    """
    import pandas as pd

    from ham10000.data.labels import LabelScheme
    from ham10000.evaluation.metrics import evaluate
    from ham10000.models.aggregation import aggregate_predictions, predicted_label
    from ham10000.models.architectures import build_classifier
    from ham10000.models.inference import predict_probabilities
    from ham10000.serialization import load_state_dict

    run = Path(run)
    config = load_config(run / "config.yaml")
    previous = pd.read_csv(run / "predictions.csv")

    # Keep the metadata columns; drop everything the model produced.
    generated = [
        column
        for column in previous.columns
        if column.startswith("prob_") or column in {"pred", "pred_final"}
    ]
    validation_frame = previous.drop(columns=generated)

    scheme = LabelScheme.build(config.classes, set(validation_frame["dx"].unique()))
    model = build_classifier(
        config.architecture,
        n_classes=scheme.n_classes,
        strategy=config.freeze_strategy,
        n_blocks=config.n_blocks,
        pretrained=False,
    )
    model.load_state_dict(load_state_dict(run / "model.pth"))

    scored = predict_probabilities(
        validation_frame,
        model,
        scheme,
        image_dir=Path(image_dir),
        transform=build_transform(config.transform),
        device=device,
    )
    scored["pred"] = predicted_label(scored, scheme.codes)
    final = aggregate_predictions(scored, seed=config.split.seed)
    final.to_csv(run / "predictions.csv", index=False)

    per_lesion = final.drop_duplicates(subset="lesion_id")
    report = evaluate(
        per_lesion["label"].to_numpy(), per_lesion["pred_final"].to_numpy()
    )
    (run / "metrics.json").write_text(
        json.dumps(report.to_series().to_dict(), indent=2)
    )

    history = json.loads((run / "model.losses.json").read_text())
    return ExperimentResult(
        config=config, directory=run, report=report, history=history
    )


def evaluate_at_views(
    run: Path,
    *,
    metadata_path: Path,
    image_dir: Path,
    views: int,
    seed: int = 0,
    device: str | None = None,
) -> Any:
    """Score a saved checkpoint with a chosen number of augmented views.

    Test-time augmentation: each validation lesion is scored several times
    under the stochastic evaluation transform and the predictions are combined
    into one verdict. `views=1` disables it.

    This is deliberately **non-destructive**. It rebuilds the validation set
    from the metadata rather than reusing the run's `predictions.csv`, and
    writes nothing back, so a run's recorded result is never silently replaced
    by one obtained under different evaluation settings.

    Because the transform is random, repeated calls with the same `views` give
    slightly different answers. That spread is the thing to compare any
    apparent gain against: see :func:`ham10000.cli.views_main`, which reports
    it.

    Parameters
    ----------
    run:
        A completed run directory.
    metadata_path, image_dir:
        Dataset location.
    views:
        Predictions per lesion.
    seed:
        Seed for the expansion draw. Vary it across repeats to measure spread.
    device:
        Compute device.

    Returns
    -------
    ClassificationReport
    """
    from ham10000.data.balancing import expand_validation
    from ham10000.data.labels import LabelScheme
    from ham10000.data.metadata import load_metadata
    from ham10000.data.metadata import restrict as restrict_rows
    from ham10000.data.splitting import assign_splits
    from ham10000.evaluation.metrics import evaluate
    from ham10000.models.aggregation import aggregate_predictions, predicted_label
    from ham10000.models.architectures import build_classifier
    from ham10000.models.inference import predict_probabilities
    from ham10000.serialization import load_state_dict

    run = Path(run)
    config = load_config(run / "config.yaml")

    frame = load_metadata(metadata_path)
    if config.restrict:
        frame = restrict_rows(frame, config.restrict)
        frame["num_images"] = frame["lesion_id"].map(frame["lesion_id"].value_counts())

    scheme = LabelScheme.build(config.classes, set(frame["dx"].unique()))
    frame["label"] = frame["dx"].map(scheme.mapping)
    frame["set"] = assign_splits(frame, config.split).sets

    keep = ["v1"] if config.train_one_image_per_lesion else ["v1", "va"]
    validation = frame[frame["set"].isin(keep)].copy()
    if views > 1:
        validation = expand_validation(
            validation,
            views,
            seed=seed,
            one_image_per_lesion=config.train_one_image_per_lesion,
        )

    model = build_classifier(
        config.architecture,
        n_classes=scheme.n_classes,
        strategy=config.freeze_strategy,
        n_blocks=config.n_blocks,
        pretrained=False,
    )
    model.load_state_dict(load_state_dict(run / "model.pth"))

    scored = predict_probabilities(
        validation,
        model,
        scheme,
        image_dir=Path(image_dir),
        transform=build_transform(config.transform),
        device=device,
    )
    scored["pred"] = predicted_label(scored, scheme.codes)
    final = aggregate_predictions(scored, seed=config.split.seed)
    per_lesion = final.drop_duplicates(subset="lesion_id")
    return evaluate(per_lesion["label"].to_numpy(), per_lesion["pred_final"].to_numpy())


def run_experiment(
    config: ExperimentConfig,
    *,
    metadata_path: Path,
    image_dir: Path,
    output_dir: Path,
    device: str | None = None,
    on_epoch: Any = None,
    on_batch: Any = None,
    limit_lesions: int | None = None,
    overwrite: bool = False,
) -> ExperimentResult:
    """Execute one experiment end to end and write its artefacts.

    The whole pipeline in one call: load metadata, build the label scheme,
    split at the lesion level, balance, expand the validation set, build the
    model, train, score, aggregate to lesion verdicts, evaluate, and write
    everything to a run directory named by the configuration hash.

    Parameters
    ----------
    config:
        Experiment definition.
    metadata_path:
        Path to `metadata.csv`.
    image_dir:
        Directory containing `<image_id>.jpg`.
    output_dir:
        Parent directory for run directories.
    device:
        Compute device. Detected when omitted.
    on_epoch:
        Optional per-epoch callback, forwarded to
        :func:`~ham10000.models.training.train_model`.
    overwrite:
        Permit writing into a run directory that already holds a completed
        run. Refused by default: a completed directory is a result, and
        silently replacing it loses the artefacts without a trace.
    limit_lesions:
        Cap the dataset at this many lesions, sampled per class so that every
        class survives. For smoke tests: it verifies the pipeline end to end in
        minutes rather than hours. Results from a limited run are meaningless
        and the run directory is suffixed `-smoke` so they cannot be mistaken
        for real ones.

    Returns
    -------
    ExperimentResult

    Notes
    -----
    The run directory contains `config.yaml`, `model.pth`,
    `model.losses.json`, `predictions.csv` and `metrics.json`. Writing the
    config alongside the artefacts is the point: a directory then records what
    it is, rather than encoding a fraction of the settings in its name.

    Balancing is applied to the training split only, and
    :func:`~ham10000.data.balancing.balance` refuses a frame containing
    validation rows, so the guarantee is enforced rather than assumed.
    """
    from ham10000.data.balancing import balance, expand_validation
    from ham10000.data.labels import LabelScheme
    from ham10000.data.metadata import load_metadata
    from ham10000.data.splitting import assign_splits
    from ham10000.evaluation.metrics import evaluate
    from ham10000.models.aggregation import aggregate_predictions, predicted_label
    from ham10000.models.architectures import build_classifier
    from ham10000.models.inference import predict_probabilities
    from ham10000.models.training import train_model

    frame = load_metadata(metadata_path)

    if config.restrict:
        from ham10000.data.metadata import restrict as restrict_rows

        frame = restrict_rows(frame, config.restrict)
        if frame.empty:
            raise ValueError(f"Restriction {dict(config.restrict)} left no rows.")
        # num_images counts images of a lesion in the *restricted* frame, and
        # every image of a lesion shares its dx, so the counts are unaffected
        # by a dx restriction. Recomputed anyway, since a restriction on
        # another column could drop some of a lesion's images.
        frame["num_images"] = frame["lesion_id"].map(frame["lesion_id"].value_counts())

    if limit_lesions is not None:
        # Sample lesions, not rows, so a lesion's images stay together and the
        # split remains lesion-disjoint. Stratified so no class disappears.
        lesions = frame.drop_duplicates("lesion_id")
        per_class = max(2, limit_lesions // lesions["dx"].nunique())
        keep = (
            lesions.groupby("dx", group_keys=False)["lesion_id"]
            .apply(lambda s: s.head(per_class))
            .tolist()
        )
        frame = frame[frame["lesion_id"].isin(keep)].copy()

    scheme = LabelScheme.build(config.classes, set(frame["dx"].unique()))
    frame["label"] = frame["dx"].map(scheme.mapping)

    assignment = assign_splits(frame, config.split)
    frame["set"] = assignment.sets

    train_frame = frame[
        frame["set"].isin(["t1"] if config.train_one_image_per_lesion else ["t1", "ta"])
    ].copy()
    validation_frame = frame[
        frame["set"].isin(["v1"] if config.train_one_image_per_lesion else ["v1", "va"])
    ].copy()

    balance_targets = config.balance
    if balance_targets and limit_lesions is not None:
        # Scale targets down; balancing to 2000 from a handful of lesions
        # would produce an absurd number of repeats.
        smallest = train_frame["dx"].value_counts().min()
        balance_targets = {k: int(smallest) for k in balance_targets}

    if balance_targets:
        train_frame = balance(
            train_frame,
            balance_targets,
            seed=config.split.seed,
            one_image_per_lesion=config.train_one_image_per_lesion,
        )
    if config.validation_expansion:
        validation_frame = expand_validation(
            validation_frame,
            config.validation_expansion,
            seed=config.split.seed,
            one_image_per_lesion=config.train_one_image_per_lesion,
        )

    transform = build_transform(config.transform)
    model = build_classifier(
        config.architecture,
        n_classes=scheme.n_classes,
        strategy=config.freeze_strategy,
        n_blocks=config.n_blocks,
        pretrained=config.pretrained,
    )

    suffix = "-smoke" if limit_lesions is not None else ""
    directory = Path(output_dir) / f"{config.slug}{suffix}"

    # A directory holding metrics.json is a finished run. An interrupted run
    # leaves only config.yaml and is safe to write over.
    if (directory / "metrics.json").is_file() and not overwrite:
        raise FileExistsError(
            f"{directory} already holds a completed run. Delete it, or pass "
            "overwrite=True (--force on the command line).\n"
            "Note that editing the configuration to avoid the collision would "
            "change the experiment's identity, which is not what you want."
        )
    config.save(directory)

    history = train_model(
        model,
        train_frame,
        validation_frame,
        scheme,
        image_dir=Path(image_dir),
        transform=transform,
        config=config.training,
        checkpoint_path=directory / "model.pth",
        device=device,
        on_epoch=on_epoch,
        on_batch=on_batch,
    )

    scored = predict_probabilities(
        validation_frame,
        model,
        scheme,
        image_dir=Path(image_dir),
        transform=transform,
        device=device,
    )
    scored["pred"] = predicted_label(scored, scheme.codes)
    final = aggregate_predictions(scored, seed=config.split.seed)
    final.to_csv(directory / "predictions.csv", index=False)

    # One row per lesion: a lesion photographed five times must not count five
    # times toward the score.
    per_lesion = final.drop_duplicates(subset="lesion_id")
    report = evaluate(
        per_lesion["label"].to_numpy(), per_lesion["pred_final"].to_numpy()
    )
    (directory / "metrics.json").write_text(
        json.dumps(report.to_series().to_dict(), indent=2)
    )

    return ExperimentResult(
        config=config, directory=directory, report=report, history=history
    )
