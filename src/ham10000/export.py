"""Export a trained run to assets the browser demo can serve.

The demo runs the real model in the visitor's browser: no server, no
precomputed answers. That is what makes it honest — a precomputed lookup table
cannot be distinguished from a hand-written one.

What is produced
----------------
``model.onnx``
    The trained network, dynamically quantised to int8. Roughly 10 MB against
    43 MB for float32, at negligible cost to accuracy for a classifier of this
    size, and small enough to serve from GitHub Pages.
``labels.json``
    Class order and preprocessing constants. The browser must reproduce the
    training preprocessing exactly; shipping the constants alongside the model
    stops the two drifting apart.
``pool.json``
    The image pool, with each image's true diagnosis and the run it came from.
``images/``
    The images themselves, resized for the web.

Fairness
--------
The pool is drawn from the **validation** split only, so every image is one the
model never trained on. One image per lesion, so the pool contains no
near-duplicates. Both constraints are enforced here rather than trusted, and
the run id is recorded in ``pool.json`` so the claim is checkable.
"""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = ["DemoExport", "export_demo"]

#: Longest edge for web-served images. Large enough that a human can see the
#: lesion properly, small enough that a few hundred fit in a Pages site.
WEB_IMAGE_SIZE = (400, 300)

#: JPEG quality for the web images.
WEB_IMAGE_QUALITY = 82


@dataclass(frozen=True, slots=True)
class DemoExport:
    """Where the exported assets went, and what is in them."""

    directory: Path
    model_bytes: int
    image_count: int
    classes: list[str]

    @property
    def model_mb(self) -> float:
        """Model size in mebibytes."""
        return self.model_bytes / 1024**2


def _select_pool(predictions: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Choose a balanced pool of one image per validation lesion.

    Balanced across classes so the game is not trivially won by always
    answering with the majority class -- which, given nevi outnumber melanoma
    roughly nine to one, it otherwise would be.

    Parameters
    ----------
    predictions:
        A run's `predictions.csv`, which contains validation rows only.
    size:
        Target pool size. Split evenly across classes; if a class has fewer
        lesions available, the pool is smaller rather than unbalanced.
    seed:
        Sampling seed.
    """
    one_per_lesion = predictions.drop_duplicates(subset="lesion_id")
    classes = sorted(one_per_lesion["dx"].unique())
    per_class = size // len(classes)

    chosen = [
        group.sample(min(per_class, len(group)), random_state=seed)
        for _, group in one_per_lesion.groupby("dx")
    ]
    pool = pd.concat(chosen).sample(frac=1, random_state=seed)
    return pool.reset_index(drop=True)


def export_demo(
    run: Path,
    *,
    image_dir: Path,
    output: Path,
    pool_size: int = 300,
    seed: int = 0,
    quantise: bool = True,
) -> DemoExport:
    """Export a completed run to browser-servable demo assets.

    Parameters
    ----------
    run:
        A run directory containing `config.yaml`, `model.pth` and
        `predictions.csv`.
    image_dir:
        Source directory of full-size dermatoscopic images.
    output:
        Destination for the demo assets.
    pool_size:
        Number of images to include, split evenly across classes.
    seed:
        Sampling seed for pool selection.
    quantise:
        Apply dynamic int8 quantisation. Roughly a quarter of the size.

    Returns
    -------
    DemoExport

    Raises
    ------
    FileNotFoundError
        If the run is incomplete or a pool image is missing.
    ValueError
        If the run has more than two classes, which the two-answer game cannot
        present.
    """
    import torch

    from ham10000.data.labels import LabelScheme
    from ham10000.experiment import load_config
    from ham10000.models.architectures import build_classifier
    from ham10000.serialization import load_state_dict

    run = Path(run)
    if not run.is_dir():
        # Distinguish "no such directory" from "directory is incomplete".
        # Reporting a missing config.yaml inside a path that does not exist
        # sends the reader looking for the wrong problem -- usually they are
        # simply in the wrong working directory.
        raise FileNotFoundError(
            f"No such run directory: {run.resolve()}\n"
            f"(interpreted relative to {Path.cwd()})"
        )
    for required in ("config.yaml", "model.pth", "predictions.csv"):
        if not (run / required).is_file():
            raise FileNotFoundError(
                f"{run} exists but is missing {required}. An interrupted run "
                "leaves only config.yaml; retrain, or pick a completed run."
            )

    config = load_config(run / "config.yaml")
    predictions = pd.read_csv(run / "predictions.csv")
    scheme = LabelScheme.build(config.classes, set(predictions["dx"].unique()))

    if scheme.n_classes != 2:
        raise ValueError(
            f"The demo presents two answers, but this run has "
            f"{scheme.n_classes} classes ({sorted(scheme.codes.values())}). "
            "Train a two-class model, e.g. configs/08_demo_melanoma_vs_nevus.yaml."
        )

    output = Path(output)
    (output / "images").mkdir(parents=True, exist_ok=True)

    # --- model ------------------------------------------------------------
    model = build_classifier(
        config.architecture,
        n_classes=scheme.n_classes,
        strategy=config.freeze_strategy,
        pretrained=False,
    )
    model.load_state_dict(load_state_dict(run / "model.pth"))
    model.eval()

    onnx_path = output / "model.onnx"
    with warnings.catch_warnings():
        # `dynamo=False` selects the legacy TorchScript exporter, which PyTorch
        # 2.9 deprecated in favour of the torch.export-based one. It is chosen
        # deliberately: the new exporter writes weights to a separate .data
        # file, and a browser demo wants one self-contained artefact it can
        # fetch and quantise. Revisit when the new exporter embeds weights by
        # default.
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*legacy TorchScript-based ONNX export.*",
        )
        torch.onnx.export(
            model,
            (torch.randn(1, 3, 224, 224),),
            str(onnx_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
        )

    if quantise:
        with warnings.catch_warnings():
            # onnxruntime warns that dynamic quantisation will be removed in a
            # future release. Suppressed here rather than globally so that any
            # other deprecation in this module still surfaces.
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from onnxruntime.quantization import QuantType, quantize_dynamic

            raw = output / "model.fp32.onnx"
            onnx_path.rename(raw)
            quantize_dynamic(str(raw), str(onnx_path), weight_type=QuantType.QUInt8)
            raw.unlink()

    # --- image pool -------------------------------------------------------
    pool = _select_pool(predictions, pool_size, seed)
    entries = []
    source_size: list[int] | None = None
    for row in pool.itertuples():
        source = Path(image_dir) / f"{row.image_id}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"Pool image not found: {source}")
        if source_size is None:
            source_size = list(_dimensions(source))
        _write_web_image(source, output / "images" / f"{row.image_id}.jpg")
        entries.append({"image_id": row.image_id, "dx": row.dx})

    # --- preprocessing constants -----------------------------------------
    # The browser must reproduce training preprocessing exactly, or the
    # deployed model is not the model that was measured.
    #
    # `source_size` is what makes the crop reproducible. Training cropped a
    # fixed 300x300 window from the 600x450 original -- half its width. The
    # web images are resized down, so cropping 300x300 from *those* would take
    # a quite different fraction of the lesion and present the network with a
    # scale it never saw. Recording the original dimensions lets the browser
    # scale the crop box to match.
    normalise = next(
        (step for step in config.transform if step.get("name") == "Normalize"),
        {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    )
    crop = next(
        (step for step in config.transform if step.get("name") == "RandomCrop"), None
    )
    (output / "labels.json").write_text(
        json.dumps(
            {
                "classes": [scheme.codes[i] for i in sorted(scheme.codes)],
                "input_size": [224, 224],
                "crop_size": crop["size"] if crop else None,
                "source_size": source_size,
                "mean": normalise["mean"],
                "std": normalise["std"],
                "views": config.validation_expansion or 1,
            },
            indent=2,
        )
    )

    (output / "pool.json").write_text(
        json.dumps(
            {
                "run": run.name,
                "source": "validation split, one image per lesion",
                "classes": [scheme.codes[i] for i in sorted(scheme.codes)],
                "images": entries,
            },
            indent=2,
        )
    )

    return DemoExport(
        directory=output,
        model_bytes=onnx_path.stat().st_size,
        image_count=len(entries),
        classes=[scheme.codes[i] for i in sorted(scheme.codes)],
    )


def _dimensions(path: Path) -> tuple[int, int]:
    """Return an image's (width, height) without decoding its pixels."""
    from PIL import Image

    with Image.open(path) as handle:
        return handle.size


def _write_web_image(source: Path, destination: Path) -> None:
    """Resize and re-encode an image for web delivery."""
    from PIL import Image

    with Image.open(source) as handle:
        image = handle.convert("RGB")
        image.thumbnail(WEB_IMAGE_SIZE, Image.Resampling.LANCZOS)
        image.save(destination, "JPEG", quality=WEB_IMAGE_QUALITY, optimize=True)


def archive_streamlit(source: Path, archive: Path) -> None:
    """Move the retired Streamlit app into the archive.

    The Streamlit demo is superseded by the browser version, which runs the
    real model rather than reading precomputed answers from a CSV. It is
    archived rather than deleted because it is part of the project's history,
    and because its scoring bug, a new image drawn before the previous answer
    was scored, is a clean example of what Streamlit's top-to-bottom rerun
    model does to code that holds state in module-level variables.
    """
    archive.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.move(str(source), str(archive / source.name))
