"""Command-line entry points.

Two commands:

``ham10000-bench``
    Reports what the machine is and measures how fast it trains, then
    estimates how long a given experiment will take. Run this before
    committing to a long run.

``ham10000-train``
    Executes an experiment from a YAML config and writes a run directory.

Both exist so that running an experiment is a command rather than a code
snippet pasted into a notebook. Without an entry point an experiment is
launched by executing cells in the right order, which is how a notebook comes
to train one model and evaluate another.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from ham10000.config import Settings

__all__ = ["benchmark_main", "train_main"]


def _describe_machine() -> dict[str, str]:
    """Collect the hardware facts that determine training time."""
    import os

    import torch

    facts = {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "logical cores": str(os.cpu_count() or "unknown"),
        "torch threads": str(torch.get_num_threads()),
    }

    if torch.cuda.is_available():
        facts["accelerator"] = f"CUDA: {torch.cuda.get_device_name(0)}"
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        facts["accelerator memory"] = f"{total:.1f} GiB"
    elif torch.backends.mps.is_available():
        facts["accelerator"] = "Apple MPS"
    else:
        facts["accelerator"] = "none (CPU only)"

    return facts


def _measure_throughput(
    device: str,
    strategy: str,
    batch_size: int = 16,
    steps: int = 6,
    repeats: int = 3,
) -> tuple[float, float]:
    """Return (training, inference) images per second for ResNet-18 at 224px.

    Uses random tensors rather than real images: this measures the network,
    which is the bottleneck. Decoding and augmenting HAM10000 JPEGs runs at
    roughly 200 images per second per core, an order of magnitude faster than
    training, so the data pipeline does not constrain a CPU run.

    Reports the **median** of several repeats. A single short measurement is
    unreliable on a laptop or desktop: turbo clocks ramp over the first few
    seconds, and background load shifts the result. An early draft used one
    four-step burst and produced figures that varied by 40% between runs on the
    same machine.

    Anything else competing for the CPU -- including a training run started in
    another terminal -- will depress these numbers and make the estimate
    optimistic in the wrong direction.
    """
    import torch
    from torch import nn, optim

    from ham10000.models.architectures import build_classifier, trainable_parameters

    model = build_classifier(
        "resnet18", n_classes=7, strategy=strategy, pretrained=False
    ).to(device)
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.zeros(batch_size, 7, device=device)
    targets[:, 0] = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_parameters(model), lr=1e-4)

    def synchronise() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    model.train()
    for _ in range(2):  # warm up kernels, caches, and clock boost
        criterion(model(images), targets).backward()
        optimizer.zero_grad()
    synchronise()

    train_rates = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
        synchronise()
        train_rates.append(batch_size * steps / (time.perf_counter() - start))

    model.eval()
    infer_rates = []
    with torch.no_grad():
        model(images)
        synchronise()
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(steps):
                model(images)
            synchronise()
            infer_rates.append(batch_size * steps / (time.perf_counter() - start))

    return median(train_rates), median(infer_rates)


def _format_duration(seconds: float) -> str:
    """Render a duration in the largest sensible unit.

    Examples
    --------
    >>> _format_duration(45)
    '45 seconds'
    >>> _format_duration(600)
    '10.0 minutes'
    >>> _format_duration(7200)
    '2.0 hours'
    """
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.1f} minutes"
    return f"{seconds / 3600:.1f} hours"


def benchmark_main(argv: list[str] | None = None) -> int:
    """Report the machine's capability and estimate a run's duration."""
    parser = argparse.ArgumentParser(
        prog="ham10000-bench",
        description=(
            "Describe this machine and estimate how long an experiment will take on it."
        ),
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="Experiment YAML to estimate. Omit to benchmark only.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force a device (cpu, cuda, mps). Detected when omitted.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Fewer measurement repeats. Faster, noticeably less reliable; "
            "intended for tests and for confirming the command runs at all."
        ),
    )
    arguments = parser.parse_args(argv)

    from ham10000.models.inference import select_device

    device = str(select_device(arguments.device))

    print("Machine")
    print("-------")
    for key, value in _describe_machine().items():
        print(f"  {key:20s} {value}")

    print(f"\nThroughput on {device}, ResNet-18 at 224px")
    print("-" * 44)
    print("  (median of 3 repeats; close other CPU-heavy work first)")
    rates: dict[str, tuple[float, float]] = {}
    sizing = {"steps": 2, "repeats": 1} if arguments.quick else {}
    for strategy in ("all", "last_block", "head_only"):
        train_rate, infer_rate = _measure_throughput(device, strategy, **sizing)
        rates[strategy] = (train_rate, infer_rate)
        print(f"  {strategy:12s} train {train_rate:7.1f} img/s")
    print(f"  {'inference':12s}       {rates['all'][1]:7.1f} img/s")

    if arguments.config is None:
        print("\nPass a config file to estimate a full run.")
        return 0

    from ham10000.experiment import load_config

    config = load_config(arguments.config)
    train_rate, infer_rate = rates[str(config.freeze_strategy)]

    # Image counts depend on the dataset, so this is an estimate from the
    # configured balance targets rather than a measurement of the real split.
    per_epoch_train = sum(config.balance.values()) if config.balance else 7500
    # expand_validation repeats each validation *lesion*, not each image, so
    # the multiplier applies to the lesion count (~1,870 at the default 3:1
    # split) rather than to the image count.
    per_epoch_val = 1870 * (config.validation_expansion or 1)
    epochs = config.training.epochs

    seconds = epochs * (per_epoch_train / train_rate + per_epoch_val / infer_rate)

    # With num_workers=0 the main process decodes and augments each image
    # before the forward pass, serialised with compute. Measured at roughly
    # 200 images/second/core on HAM10000-sized JPEGs.
    decode_rate = 200.0
    if config.training.num_workers == 0:
        seconds += epochs * (per_epoch_train + per_epoch_val) / decode_rate

    print(f"\nEstimate for {arguments.config.name}")
    print("-" * 44)
    print(f"  freeze strategy      {config.freeze_strategy}")
    print(f"  epochs               {epochs}")
    print(f"  train images/epoch   ~{per_epoch_train:,}")
    print(f"  val images/epoch     ~{per_epoch_val:,}")
    print(f"  data loader workers  {config.training.num_workers}")
    print(f"  estimated duration   {_format_duration(seconds)}")
    if config.training.num_workers == 0:
        overlap = epochs * (per_epoch_train + per_epoch_val) / decode_rate
        print(
            f"    of which decoding    ~{_format_duration(overlap)} "
            "(avoidable with --num-workers 4)"
        )
    if seconds > 3 * 3600:
        print(
            "\n  Over three hours. Consider --limit for a smoke test first, a "
            "cheaper\n  freeze strategy, or a free GPU (Kaggle hosts HAM10000 "
            "already)."
        )
    return 0


def train_main(argv: list[str] | None = None) -> int:
    """Run an experiment from a YAML configuration."""
    parser = argparse.ArgumentParser(
        prog="ham10000-train",
        description="Train a model from an experiment configuration.",
    )
    parser.add_argument("config", type=Path, help="Experiment YAML.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Project root. Defaults to $HAM10000_ROOT or an upward search.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for run directories. Defaults to <root>/models.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Use at most N lesions. For smoke tests: verifies the pipeline "
            "end to end in minutes rather than hours."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the configured epochs."
    )
    parser.add_argument(
        "--device", default=None, help="Force a device (cpu, cuda, mps)."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "DataLoader worker processes. Defaults to the config value (0), "
            "which decodes images in the main process, serialised with "
            "training. 2-4 overlaps decoding with compute and typically saves "
            "10-20%% of wall time on CPU."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite a completed run directory. Without this, an existing "
            "result is protected. Do not edit the config to dodge a collision: "
            "the config determines the run's identity."
        ),
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "Start from random weights. Useful for smoke tests offline or "
            "behind a proxy, since it skips the torchvision weight download. "
            "Not useful for a real run: transfer learning is most of the "
            "performance here."
        ),
    )
    arguments = parser.parse_args(argv)

    from dataclasses import replace

    from ham10000.config import Settings
    from ham10000.experiment import load_config, run_experiment

    settings = Settings.resolve(arguments.data_root)
    config = load_config(arguments.config)

    if arguments.epochs is not None:
        config = replace(
            config, training=replace(config.training, epochs=arguments.epochs)
        )
    if arguments.num_workers is not None:
        config = replace(
            config,
            training=replace(config.training, num_workers=arguments.num_workers),
        )
    if arguments.no_pretrained:
        config = replace(config, pretrained=False)

    output = arguments.output or (settings.root / "models")

    # Progress state shared between the two callbacks. A long CPU run
    # otherwise produces no output for ten minutes, which is indistinguishable
    # from a hang.
    epoch_started = time.perf_counter()

    def progress(batch: int, total: int, running_loss: float) -> None:
        nonlocal epoch_started
        if batch == 1:
            epoch_started = time.perf_counter()
        if batch % 10 and batch != total:
            return
        elapsed = time.perf_counter() - epoch_started
        rate = batch / elapsed if elapsed else 0.0
        remaining = (total - batch) / rate if rate else 0.0
        # Carriage return keeps the line in place rather than scrolling
        # hundreds of lines per epoch.
        print(
            f"\r    batch {batch:>4}/{total}  loss {running_loss:.4f}  "
            f"eta {_format_duration(remaining):>12}",
            end="",
            flush=True,
        )

    def report(epoch: int, train_loss: float, validation_loss: float) -> None:
        print(
            f"\r  epoch {epoch + 1:>3}/{config.training.epochs}  "
            f"train {train_loss:.4f}  val {validation_loss:.4f}"
            f"{' ' * 20}",
            flush=True,
        )

    suffix = "-smoke" if arguments.limit is not None else ""
    print(f"Experiment : {config.name}")
    print(f"Run id     : {config.run_id}")
    print(f"Output     : {output / (config.slug + suffix)}")
    if arguments.limit is not None:
        print(
            "\nSMOKE TEST: results from a limited run are meaningless. The run "
            "directory\nis suffixed -smoke so it cannot be mistaken for a real "
            "one."
        )
    print()

    started = time.perf_counter()
    result = run_experiment(
        config,
        metadata_path=settings.metadata_csv,
        image_dir=settings.images,
        output_dir=output,
        device=arguments.device,
        on_epoch=report,
        on_batch=progress,
        limit_lesions=arguments.limit,
        overwrite=arguments.force,
    )
    elapsed = time.perf_counter() - started

    print(f"\nCompleted in {_format_duration(elapsed)}")
    print(f"Balanced accuracy: {result.report.balanced_accuracy:.4f}")
    print(f"Artefacts: {result.directory}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(train_main())


# ---------------------------------------------------------------------------
# Inspection subcommands
#
# These expose from the shell the analysis the notebooks perform. A notebook is
# the right medium for a narrative with figures; it is the wrong medium for
# "what is in this dataset again?" during development, where launching a kernel
# to run four cells is friction.
# ---------------------------------------------------------------------------


def _resolve_run(run: Path, root: Path) -> Path:
    """Locate a run directory given as a path relative to almost anywhere.

    A run is normally referred to as `models/<slug>`, but that only resolves
    from the repository root. Someone standing in `demo/` or `notebooks/` gets
    a confusing "missing config.yaml" for a directory that plainly exists. Try
    the obvious alternatives before giving up.
    """
    candidates = [run, root / run, root / "models" / run.name]
    for candidate in candidates:
        if (candidate / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        f"No run directory found for {run}. Looked in:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


def _load_frame(root: Path | None) -> tuple[pd.DataFrame, Settings]:
    """Resolve the project and load annotated metadata."""
    from ham10000.config import Settings
    from ham10000.data import load_metadata

    settings = Settings.resolve(root)
    return load_metadata(settings.metadata_csv), settings


def info_main(argv: list[str] | None = None) -> int:
    """Summarise the dataset: size, classes, multiplicity, missing values."""
    parser = argparse.ArgumentParser(
        prog="ham10000 info", description="Summarise the dataset."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    frame, _ = _load_frame(arguments.data_root)
    lesions = frame.drop_duplicates("lesion_id")

    print(f"images   {len(frame):,}")
    print(f"lesions  {lesions['lesion_id'].nunique():,}")

    print("\nClass distribution (lesions)")
    counts = lesions["dx"].value_counts()
    share = (counts / counts.sum() * 100).round(1)
    for dx in counts.index:
        print(f"  {dx:6s} {counts[dx]:>6,}  {share[dx]:>5.1f}%")
    print(f"\n  imbalance {counts.max() / counts.min():.1f}:1")
    print(f"  balanced-accuracy chance level {1 / len(counts):.3f}")

    print("\nImages per lesion")
    for images, n in lesions["num_images"].value_counts().sort_index().items():
        print(f"  {images} image(s): {n:>6,} lesions")
    repeated = int((lesions["num_images"] > 1).sum())
    print(
        f"\n  {repeated:,} lesions ({repeated / len(lesions):.1%}) have more than "
        "one image,\n  which is why the split is at the lesion level."
    )

    missing = frame.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\nMissing values")
        for column, count in missing.items():
            print(f"  {column:14s} {count:,}")
    return 0


def multiplicity_main(argv: list[str] | None = None) -> int:
    """Show that how often a lesion was photographed encodes clinical concern."""
    parser = argparse.ArgumentParser(
        prog="ham10000 multiplicity",
        description=(
            "Relate image multiplicity to diagnostic certainty. Multiplicity "
            "records whether a clinician was worried enough to biopsy, which "
            "is why num_images must never be used as a feature."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--dx", default="nv", help="Class to break down.")
    arguments = parser.parse_args(argv)

    import pandas as pd

    frame, _ = _load_frame(arguments.data_root)
    lesions = frame.drop_duplicates("lesion_id")

    subset = lesions[lesions["dx"] == arguments.dx]
    if subset.empty:
        print(f"No lesions with dx={arguments.dx!r}.")
        return 1

    table = (
        pd.crosstab(subset["num_images"], subset["dx_type"], normalize="index")
        .mul(100)
        .round(1)
    )
    print(f"Diagnosis method by image count, for dx={arguments.dx!r} (%)\n")
    print(table.to_string())

    print("\nMean images per lesion, by class")
    means = lesions.groupby("dx")["num_images"].mean().round(2).sort_values()
    for dx, mean in means.items():
        print(f"  {dx:6s} {mean:.2f}")
    return 0


def split_main(argv: list[str] | None = None) -> int:
    """Report the lesion-level split and verify it does not leak."""
    parser = argparse.ArgumentParser(
        prog="ham10000 split", description="Summarise the train/validation split."
    )
    parser.add_argument("config", nargs="?", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    from ham10000.data import LabelScheme, SplitConfig, assign_splits, lesion_overlap
    from ham10000.experiment import load_config

    frame, _ = _load_frame(arguments.data_root)

    if arguments.config is not None:
        config = load_config(arguments.config)
        classes, split = config.classes, config.split
    else:
        classes = sorted(frame["dx"].unique())
        split = SplitConfig()

    scheme = LabelScheme.build(classes, set(frame["dx"].unique()))
    frame["label"] = frame["dx"].map(scheme.mapping)
    assignment = assign_splits(frame, split)
    annotated = frame.assign(set=assignment.sets)

    print(
        f"ratio {split.train_val_ratio}:1, seed {split.seed}, "
        f"stratified={split.stratified}\n"
    )
    print(f"train lesions {len(assignment.train_lesions):,}")
    print(f"val lesions   {len(assignment.val_lesions):,}")
    print("\nImages by split")
    for value, count in annotated["set"].value_counts().sort_index().items():
        print(f"  {value}  {count:>6,}")

    overlap = lesion_overlap(annotated)
    print(f"\nlesions on both sides: {len(overlap)}")
    if overlap:
        print("  LEAKAGE. This should be impossible; please report it.")
        return 1
    print("  no leakage: every image of a lesion is on one side only")
    return 0


def balance_main(argv: list[str] | None = None) -> int:
    """Show what balancing does to each class, including repetition factors."""
    parser = argparse.ArgumentParser(
        prog="ham10000 balance",
        description="Report resampling factors for a configuration.",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    import pandas as pd

    from ham10000.data import LabelScheme, assign_splits
    from ham10000.experiment import load_config

    config = load_config(arguments.config)
    if not config.balance:
        print(f"{config.name} does not use balancing.")
        return 0

    frame, _ = _load_frame(arguments.data_root)
    scheme = LabelScheme.build(config.classes, set(frame["dx"].unique()))
    frame["label"] = frame["dx"].map(scheme.mapping)
    frame["set"] = assign_splits(frame, config.split).sets

    sets = ["t1"] if config.train_one_image_per_lesion else ["t1", "ta"]
    train = frame[frame["set"].isin(sets)]

    available = train["dx"].value_counts()
    table = pd.DataFrame({"available": available})
    table["target"] = [config.balance.get(dx, available[dx]) for dx in table.index]
    table["factor"] = (table["target"] / table["available"]).round(1)
    table["lesions"] = train.groupby("dx")["lesion_id"].nunique()
    table = table.sort_values("factor")

    print(
        f"{config.name}  (one image per lesion: {config.train_one_image_per_lesion})\n"
    )
    print(table.to_string())
    print(f"\ntraining images per epoch: {int(table['target'].sum()):,}")

    worst = table.iloc[-1]
    if worst["factor"] > 10:
        print(
            f"\n  {worst.name} is repeated {worst['factor']:.0f}x over "
            f"{int(worst['lesions'])} distinct lesions.\n  High repetition risks "
            "memorising those lesions rather than learning the class."
        )
    return 0


def configs_main(argv: list[str] | None = None) -> int:
    """List available experiments and whether each has been run."""
    parser = argparse.ArgumentParser(
        prog="ham10000 configs", description="List experiment configurations."
    )
    parser.add_argument("--directory", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    from ham10000.config import Settings
    from ham10000.experiment import load_config

    settings = Settings.resolve(arguments.data_root)
    models = settings.root / "models"

    paths = sorted(arguments.directory.glob("*.yaml"))
    if not paths:
        print(f"No configurations found in {arguments.directory}.")
        return 1

    for path in paths:
        config = load_config(path)
        directory = models / config.slug
        status = "run" if (directory / "metrics.json").is_file() else "-"
        print(f"{status:>4}  {config.run_id}  {path.name:36s} {config.name}")
    print("\n'run' means a completed run directory exists under models/.")
    return 0


def results_main(argv: list[str] | None = None) -> int:
    """Report metrics for a completed run."""
    parser = argparse.ArgumentParser(
        prog="ham10000 results", description="Report metrics for a run."
    )
    parser.add_argument(
        "run",
        nargs="?",
        type=Path,
        default=None,
        help="Run directory. Defaults to the most recent completed non-smoke run.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    import json as json_module

    import pandas as pd

    from ham10000.config import Settings
    from ham10000.data.labels import LabelScheme
    from ham10000.evaluation.metrics import evaluate, per_class_recall
    from ham10000.experiment import load_config
    from ham10000.reporting import confusion_with_recall

    run = arguments.run
    if run is None:
        settings = Settings.resolve(arguments.data_root)
        candidates = [
            path
            for path in (settings.root / "models").glob("*")
            if (path / "metrics.json").is_file() and not path.name.endswith("-smoke")
        ]
        if not candidates:
            print("No completed runs found under models/.")
            return 1
        run = max(candidates, key=lambda path: path.stat().st_mtime)

    run = _resolve_run(run, Settings.resolve(arguments.data_root).root)
    config = load_config(run / "config.yaml")
    predictions = pd.read_csv(run / "predictions.csv")
    losses = json_module.loads((run / "model.losses.json").read_text())

    scheme = LabelScheme.build(config.classes, set(predictions["dx"].unique()))
    per_lesion = predictions.drop_duplicates(subset="lesion_id")
    report = evaluate(
        per_lesion["label"].to_numpy(), per_lesion["pred_final"].to_numpy()
    )

    chance = 1 / scheme.n_classes
    print(f"run        {run.name}")
    print(f"experiment {config.name}\n")
    print(f"lesions evaluated  {report.n_samples:,}")
    print(f"balanced accuracy  {report.balanced_accuracy:.4f}")
    print(f"chance level       {chance:.4f}")
    print(
        f"skill above chance {(report.balanced_accuracy - chance) / (1 - chance):.4f}"
    )
    print(f"plain accuracy     {report.accuracy:.4f}  (misleading alone)")

    validation = losses["validation_loss"]
    best = min(range(len(validation)), key=validation.__getitem__)
    print(f"\nlowest validation loss  epoch {best + 1} ({validation[best]:.4f})")
    print(f"saved checkpoint        epoch {len(validation)} ({validation[-1]:.4f})")
    if config.training.save_best:
        print(f"  save_best is set, so the checkpoint holds epoch {best + 1}.")
        print(
            "  Note that the selection is by validation *loss*, which is not "
            "the\n  metric reported above. The lowest-loss epoch is not "
            "necessarily the\n  most accurate one."
        )
    elif best + 1 != len(validation):
        print(
            "  The saved model is not this run's best. Training keeps the final\n"
            "  epoch by default; set save_best in the config to keep the best."
        )

    print("\nPer-class (rows true, columns predicted)")
    table = confusion_with_recall(
        per_lesion["label"].to_numpy(),
        per_lesion["pred_final"].to_numpy(),
        scheme.codes,
    )
    print(table.to_string())

    # Read melanoma sensitivity from the typed helper rather than indexing the
    # confusion table, which mixes counts and floats in one frame.
    recalls = per_class_recall(
        per_lesion["label"].to_numpy(),
        per_lesion["pred_final"].to_numpy(),
        scheme.codes,
    )
    if "mel" in recalls.index:
        sensitivity = float(recalls["mel"])
        support = int((per_lesion["label"] == scheme.mapping["mel"]).sum())
        missed = round(support * (1 - sensitivity))
        print(
            f"\nmelanoma sensitivity {sensitivity:.3f}  ({missed} of {support} missed)"
        )
    return 0


def rescore_main(argv: list[str] | None = None) -> int:
    """Recompute a run's metrics from its saved checkpoint."""
    parser = argparse.ArgumentParser(
        prog="ham10000 rescore",
        description=(
            "Recompute predictions and metrics from a run's saved checkpoint. "
            "Repairs runs whose reported metrics describe a different model "
            "from the one on disk."
        ),
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args(argv)

    import json as json_module

    from ham10000.config import Settings
    from ham10000.experiment import rescore_run

    settings = Settings.resolve(arguments.data_root)
    run = _resolve_run(arguments.run, settings.root)
    before = json_module.loads((run / "metrics.json").read_text())

    result = rescore_run(run, image_dir=settings.images, device=arguments.device)

    print(
        f"balanced accuracy  {before['BACC']:.4f} -> "
        f"{result.report.balanced_accuracy:.4f}"
    )
    print(f"lesions            {result.report.n_samples:,}")
    print("\npredictions.csv and metrics.json now describe model.pth.")
    return 0


def views_main(argv: list[str] | None = None) -> int:
    """Measure whether test-time augmentation is helping, and by how much."""
    parser = argparse.ArgumentParser(
        prog="ham10000 views",
        description=(
            "Re-score a checkpoint at several test-time augmentation settings. "
            "Because the evaluation transform is random, each setting is "
            "measured several times so that any apparent gain can be compared "
            "against the spread. Nothing is written back to the run."
        ),
    )
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--views",
        default="1,3",
        help="Comma-separated view counts to compare. 1 disables augmentation.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Evaluations per setting, each with a different draw.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args(argv)

    from statistics import mean, pstdev

    from ham10000.config import Settings
    from ham10000.experiment import evaluate_at_views

    settings = Settings.resolve(arguments.data_root)
    counts = [int(v) for v in arguments.views.split(",")]

    print(f"{'views':>6} {'mean':>8} {'spread':>8} {'min':>8} {'max':>8}")
    summary: dict[int, list[float]] = {}
    for views in counts:
        scores = []
        for repeat in range(arguments.repeats):
            report = evaluate_at_views(
                arguments.run,
                metadata_path=settings.metadata_csv,
                image_dir=settings.images,
                views=views,
                seed=repeat,
                device=arguments.device,
            )
            scores.append(report.balanced_accuracy)
        summary[views] = scores
        spread = pstdev(scores) if len(scores) > 1 else 0.0
        print(
            f"{views:>6} {mean(scores):>8.4f} {spread:>8.4f} "
            f"{min(scores):>8.4f} {max(scores):>8.4f}"
        )

    if len(counts) > 1 and arguments.repeats > 1:
        baseline, best = counts[0], counts[-1]
        gain = mean(summary[best]) - mean(summary[baseline])
        noise = max(pstdev(summary[baseline]), pstdev(summary[best]))
        print(
            f"\n{best} views vs {baseline}: {gain:+.4f}, "
            f"against a run-to-run spread of {noise:.4f}."
        )
        if noise and abs(gain) < 2 * noise:
            print(
                "  The difference is smaller than twice the spread, so this "
                "does not\n  distinguish the two settings."
            )
    return 0


def export_main(argv: list[str] | None = None) -> int:
    """Export a trained run to browser-servable demo assets."""
    parser = argparse.ArgumentParser(
        prog="ham10000 export",
        description="Export a two-class run to ONNX plus an image pool.",
    )
    parser.add_argument("run", type=Path, help="Run directory to export.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination. Defaults to <root>/demo/public/model.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=300,
        help="Images in the pool, split evenly across classes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-quantise",
        action="store_true",
        help="Keep float32 weights. Roughly four times larger.",
    )
    arguments = parser.parse_args(argv)

    from ham10000.config import Settings
    from ham10000.export import export_demo

    settings = Settings.resolve(arguments.data_root)
    output = arguments.output or (settings.root / "demo" / "public" / "model")

    result = export_demo(
        _resolve_run(arguments.run, settings.root),
        image_dir=settings.images,
        output=output,
        pool_size=arguments.pool_size,
        seed=arguments.seed,
        quantise=not arguments.no_quantise,
    )

    print(f"model     {result.model_mb:.1f} MB  ({', '.join(result.classes)})")
    print(f"images    {result.image_count}")
    print(f"output    {result.directory}")
    print(
        "\nPool drawn from the validation split, one image per lesion, so the "
        "model\nhas never seen any of them."
    )
    return 0


def images_main(argv: list[str] | None = None) -> int:
    """Render a grid of lesion images to a file, with their metadata."""
    parser = argparse.ArgumentParser(
        prog="ham10000 images",
        description=(
            "Render lesion images to an image file. Pick images one of three "
            "ways: by diagnosis, by lesion, or by naming image ids directly."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("lesions.png"), help="File to write."
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--dx", help="Comma-separated diagnoses, one per column, sampled."
    )
    selector.add_argument(
        "--lesion", help="Comma-separated lesion ids, one per row, all images."
    )
    selector.add_argument(
        "--image", help="Comma-separated image ids, shown in the order given."
    )
    parser.add_argument("--rows", type=int, default=4, help="Rows to render.")
    parser.add_argument("--cols", type=int, default=4, help="Columns, for --image.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--captions",
        action="store_true",
        help=(
            "Print full metadata beneath each image. Off by default: eight "
            "lines under each thumbnail obscure a visual point."
        ),
    )
    parser.add_argument("--size", type=float, default=2.4, help="Inches per cell.")
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")  # write a file; never try to open a window

    from ham10000.exploration import diagnosis_grid, image_grid, lesion_grid

    frame, settings = _load_frame(arguments.data_root)
    cell = (arguments.size, arguments.size)

    if arguments.dx:
        classes = [d.strip() for d in arguments.dx.split(",")]
        unknown = set(classes) - set(frame["dx"].unique())
        if unknown:
            print(f"Unknown diagnosis: {sorted(unknown)}")
            return 1
        figure = diagnosis_grid(
            classes,
            frame,
            settings.images,
            n_rows=arguments.rows,
            seed=arguments.seed,
            cell_size=cell,
            captions=arguments.captions,
        )
        shown = frame[frame["dx"].isin(classes)]
    elif arguments.lesion:
        lesions = [i.strip() for i in arguments.lesion.split(",")]
        unknown = set(lesions) - set(frame["lesion_id"])
        if unknown:
            print(f"Unknown lesion: {sorted(unknown)}")
            return 1
        figure = lesion_grid(
            lesions,
            frame,
            settings.images,
            max_rows=max(arguments.rows, len(lesions)),
            seed=arguments.seed,
            cell_size=cell,
            captions=arguments.captions,
        )
        shown = frame[frame["lesion_id"].isin(lesions)]
    else:
        ids = [i.strip() for i in arguments.image.split(",")]
        unknown = set(ids) - set(frame["image_id"])
        if unknown:
            print(f"Unknown image: {sorted(unknown)}")
            return 1
        figure = image_grid(
            ids,
            frame,
            settings.images,
            n_cols=arguments.cols,
            cell_size=cell,
            captions=arguments.captions,
        )
        shown = frame[frame["image_id"].isin(ids)]

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=110, bbox_inches="tight")

    columns = [
        c
        for c in (
            "lesion_id",
            "image_id",
            "num_images",
            "dx",
            "dx_type",
            "age",
            "sex",
            "localization",
        )
        if c in shown.columns
    ]
    print(shown[columns].head(40).to_string(index=False))
    print(f"\nwrote {arguments.output}")
    return 0


def frequencies_main(argv: list[str] | None = None) -> int:
    """Tabulate the values of a metadata column."""
    parser = argparse.ArgumentParser(
        prog="ham10000 frequencies",
        description="Absolute and relative frequencies of a metadata column.",
    )
    parser.add_argument("column", help="Column to count, e.g. dx or localization.")
    parser.add_argument(
        "--where",
        help="Restriction as COLUMN=VALUE, e.g. --where dx=mel.",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help=(
            "Count images rather than lesions. Lesions are the default because "
            "they are the unit everything is scored on."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    from ham10000.exploration import frequencies

    frame, _ = _load_frame(arguments.data_root)
    if not arguments.images:
        frame = frame.drop_duplicates("lesion_id")

    where = None
    if arguments.where:
        if "=" not in arguments.where:
            print("--where must look like COLUMN=VALUE")
            return 1
        column, value = arguments.where.split("=", 1)
        where = (column.strip(), value.strip())

    unit = "images" if arguments.images else "distinct lesions"
    print(f"{len(frame):,} {unit}\n")
    print(frequencies(frame, arguments.column, where=where).to_string())
    return 0


def checkpoint_main(argv: list[str] | None = None) -> int:
    """Inspect a saved checkpoint without instantiating a model."""
    parser = argparse.ArgumentParser(
        prog="ham10000 checkpoint",
        description=(
            "Report the parameter names and shapes in a .pth file. Answers "
            "'is this file intact, and what was it trained for?' without "
            "needing to guess the architecture first."
        ),
    )
    parser.add_argument("path", type=Path, help="Checkpoint to inspect.")
    parser.add_argument(
        "--all", action="store_true", help="List every parameter, not a summary."
    )
    arguments = parser.parse_args(argv)

    from ham10000.serialization import CheckpointError, describe_checkpoint

    try:
        shapes = describe_checkpoint(arguments.path)
    except CheckpointError as error:
        print(error)
        return 1

    total = sum(int(np.prod(shape)) for shape in shapes.values())
    print(f"{len(shapes)} tensors, {total:,} parameters")

    # The final layer's first dimension is the number of classes the
    # checkpoint was trained for.
    head = [n for n in shapes if n.endswith(("fc.weight", "classifier.1.weight"))]
    if head:
        print(f"output classes: {shapes[head[0]][0]}  (from {head[0]})")

    if arguments.all:
        print()
        for name, shape in shapes.items():
            print(f"  {name:52s} {tuple(shape)}")
    else:
        print("\nfirst and last five tensors:")
        names = list(shapes)
        for name in names[:5] + (["..."] if len(names) > 10 else []) + names[-5:]:
            if name == "...":
                print("  ...")
            else:
                print(f"  {name:52s} {tuple(shapes[name])}")
    return 0


def compare_main(argv: list[str] | None = None) -> int:
    """Compare metrics across completed runs."""
    parser = argparse.ArgumentParser(
        prog="ham10000 compare",
        description=(
            "Stack the metrics of several completed runs into one table, best "
            "first. Sorted by balanced accuracy, not plain accuracy, which "
            "would rank an always-majority model competitively."
        ),
    )
    parser.add_argument(
        "runs", type=Path, nargs="*", help="Run directories. Defaults to all."
    )
    parser.add_argument("--sort-by", default="BACC", help="Metric to sort on.")
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    import json as json_module

    import pandas as pd

    from ham10000.config import Settings
    from ham10000.experiment import load_config

    settings = Settings.resolve(arguments.data_root)
    runs = arguments.runs or sorted(
        path
        for path in (settings.root / "models").glob("*")
        if (path / "metrics.json").is_file() and not path.name.endswith("-smoke")
    )
    if not runs:
        print("No completed runs found.")
        return 1

    rows = {}
    for run in runs:
        metrics = json_module.loads((run / "metrics.json").read_text())
        config = load_config(run / "config.yaml")
        # Chance level differs between runs with different class counts, so a
        # bare score is not comparable across them.
        chance = 1 / len(config.classes)
        metrics["chance"] = round(chance, 4)
        metrics["skill"] = round((metrics["BACC"] - chance) / (1 - chance), 4)
        rows[config.name] = metrics

    table = pd.DataFrame(rows).T
    if arguments.sort_by not in table.columns:
        print(f"Unknown metric {arguments.sort_by!r}. Have: {sorted(table.columns)}")
        return 1
    table = table.sort_values(arguments.sort_by, ascending=False)
    print(table.to_string())
    print(
        "\nBACC is not comparable across rows with different class counts; "
        "`skill`\nis (BACC - chance) / (1 - chance), which is."
    )
    return 0


def thresholds_main(argv: list[str] | None = None) -> int:
    """Show what a sensitivity-biased decision rule does to a run's metrics."""
    parser = argparse.ArgumentParser(
        prog="ham10000 thresholds",
        description=(
            "Re-decide a run's predictions under a threshold rule and report "
            "the effect. Plain argmax is the wrong rule for screening: a "
            "lesion at 45% melanoma against 50% nevus is called a nevus."
        ),
    )
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--promote",
        default="mel=0.4",
        help=(
            "Ordered CLASS=THRESHOLD pairs, comma separated. Order is the "
            "clinical priority: the first class clearing its bar wins."
        ),
    )
    parser.add_argument(
        "--demote", default="", help="CLASS=THRESHOLD pairs to push below argmax."
    )
    parser.add_argument(
        "--rule",
        choices=["priority", "cost-sensitive"],
        default="priority",
        help=(
            "priority walks the list and stops at the first class clearing its "
            "bar; cost-sensitive divides every probability by its threshold."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    import pandas as pd

    from ham10000.data.labels import LabelScheme
    from ham10000.evaluation.metrics import evaluate, per_class_recall
    from ham10000.experiment import load_config
    from ham10000.models.aggregation import aggregate_predictions, predicted_label
    from ham10000.models.thresholds import (
        apply_cost_sensitive_weights,
        apply_priority_thresholds,
    )

    def parse(spec: str) -> list[tuple[str, float]]:
        pairs = []
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, value = item.partition("=")
            pairs.append((name.strip(), float(value)))
        return pairs

    config = load_config(arguments.run / "config.yaml")
    predictions = pd.read_csv(arguments.run / "predictions.csv")
    scheme = LabelScheme.build(config.classes, set(predictions["dx"].unique()))

    promote, demote = parse(arguments.promote), parse(arguments.demote)
    if arguments.rule == "priority":
        adjusted = apply_priority_thresholds(
            predictions, promote=promote, demote=demote
        )
    else:
        adjusted = apply_cost_sensitive_weights(predictions, thresholds=promote)

    adjusted["pred"] = predicted_label(adjusted, scheme.codes)
    final = aggregate_predictions(adjusted, seed=config.split.seed)

    baseline = predictions.drop_duplicates("lesion_id")
    after = final.drop_duplicates("lesion_id")
    target = baseline["label"].to_numpy()

    before_report = evaluate(target, baseline["pred_final"].to_numpy())
    after_report = evaluate(target, after["pred_final"].to_numpy())

    print(f"rule            {arguments.rule}")
    print(f"promote         {promote or 'none'}")
    print(f"demote          {demote or 'none'}\n")
    print(f"{'':18s} {'argmax':>10s} {'thresholded':>12s}")
    print(
        f"{'balanced accuracy':18s} {before_report.balanced_accuracy:>10.4f} "
        f"{after_report.balanced_accuracy:>12.4f}"
    )
    print(
        f"{'plain accuracy':18s} {before_report.accuracy:>10.4f} "
        f"{after_report.accuracy:>12.4f}"
    )

    before_recall = per_class_recall(
        target, baseline["pred_final"].to_numpy(), scheme.codes
    )
    after_recall = per_class_recall(
        target, after["pred_final"].to_numpy(), scheme.codes
    )
    print("\nper-class recall")
    comparison = pd.DataFrame(
        {"argmax": before_recall.round(3), "thresholded": after_recall.round(3)}
    )
    comparison["change"] = (comparison["thresholded"] - comparison["argmax"]).round(3)
    print(comparison.to_string())
    changed = int(
        (baseline["pred_final"].to_numpy() != after["pred_final"].to_numpy()).sum()
    )
    print(f"\n{changed:,} of {len(after):,} lesion verdicts changed")
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    """Score arbitrary images with a trained model."""
    parser = argparse.ArgumentParser(
        prog="ham10000 predict",
        description="Run a trained checkpoint over one or more images.",
    )
    parser.add_argument("run", type=Path, help="Run directory holding model.pth.")
    parser.add_argument("images", nargs="+", help="Image ids, or paths to .jpg files.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    import pandas as pd

    from ham10000.config import Settings
    from ham10000.data.labels import LabelScheme
    from ham10000.experiment import build_transform, load_config
    from ham10000.models.architectures import build_classifier
    from ham10000.models.inference import predict_probabilities
    from ham10000.serialization import load_state_dict

    settings = Settings.resolve(arguments.data_root)
    config = load_config(arguments.run / "config.yaml")

    ids, directories = [], set()
    for item in arguments.images:
        path = Path(item)
        if path.suffix:
            ids.append(path.stem)
            directories.add(path.parent)
        else:
            ids.append(item)
            directories.add(settings.images)
    if len(directories) > 1:
        print("All images must come from one directory.")
        return 1

    frame = pd.DataFrame({"image_id": ids})
    scheme = LabelScheme.build(config.classes, set(config.classes))
    model = build_classifier(
        config.architecture,
        n_classes=scheme.n_classes,
        strategy=config.freeze_strategy,
        pretrained=False,
    )
    model.load_state_dict(load_state_dict(arguments.run / "model.pth"))

    scored = predict_probabilities(
        frame,
        model,
        scheme,
        image_dir=directories.pop(),
        transform=build_transform(config.transform),
        device=arguments.device,
    )
    columns = scheme.probability_columns
    scored[columns] = scored[columns].round(4)
    scored["predicted"] = scored[columns].idxmax(axis=1).str.removeprefix("prob_")
    print(scored[["image_id", *columns, "predicted"]].to_string(index=False))
    print(
        "\nNote that the evaluation transform is stochastic, so re-running "
        "gives\nslightly different probabilities for the same image."
    )
    return 0


_SUBCOMMANDS = {
    "info": (info_main, "Summarise the dataset"),
    "frequencies": (frequencies_main, "Tabulate a metadata column"),
    "images": (images_main, "Render lesion images to a file"),
    "multiplicity": (multiplicity_main, "Relate image count to diagnostic certainty"),
    "split": (split_main, "Summarise the split and check for leakage"),
    "balance": (balance_main, "Report class resampling factors"),
    "configs": (configs_main, "List experiments and their run status"),
    "bench": (benchmark_main, "Measure this machine and estimate a run"),
    "train": (train_main, "Train a model from a configuration"),
    "results": (results_main, "Report metrics for a completed run"),
    "compare": (compare_main, "Compare metrics across runs"),
    "thresholds": (thresholds_main, "Apply a sensitivity-biased decision rule"),
    "predict": (predict_main, "Score images with a trained model"),
    "checkpoint": (checkpoint_main, "Inspect a .pth file"),
    "rescore": (rescore_main, "Recompute a run's metrics from its checkpoint"),
    "views": (views_main, "Compare test-time augmentation settings"),
    "export": (export_main, "Export a run to browser demo assets"),
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a subcommand.

    Deliberately hand-rolled rather than using argparse subparsers, so each
    subcommand keeps its own parser and remains independently callable and
    independently testable.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)

    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print("usage: ham10000 <command> [options]\n")
        print("commands:")
        for name, (_, description) in _SUBCOMMANDS.items():
            print(f"  {name:14s} {description}")
        print("\nRun 'ham10000 <command> --help' for a command's options.")
        return 0

    name, *rest = arguments
    entry = _SUBCOMMANDS.get(name)
    if entry is None:
        print(f"Unknown command {name!r}. Known: {', '.join(_SUBCOMMANDS)}.")
        return 2
    return entry[0](rest)
