# Human Against Machine: skin lesion classification

**[Play the demo](https://tmfreiberg.github.io/human-against-machine/book/challenge.html)**: one
dermatoscopic image at a time, melanoma or an ordinary mole? A neural network
answers the same question in your browser, on the same image, at the same
moment.

Fine-tuning image classifiers to distinguish seven kinds of skin lesion, on the
HAM10000 dermatoscopic image dataset. The name is the dataset's own: HAM10000
stands for *Human Against Machine with 10000 training images*, and the
comparison it invites is the point of the demo.

**This is a learning project, not a diagnostic tool.** The model is small, the
dataset is public, and the results are honest rather than impressive. If a
lesion concerns you, see a dermatologist.

## What is here

| | |
|---|---|
| `notebooks/00_exploration.ipynb` | What the dataset contains and what is awkward about it |
| `notebooks/01_pipeline.ipynb` | Every stage from metadata to a reported score, with the reasoning |
| `notebooks/02_results.ipynb` | What the trained model does, and what it fails at |
| `src/ham10000/` | The library |
| `configs/` | One YAML file per experiment |
| `demo/` | The game, running the real model in the browser |
| [`docs/cli.md`](docs/cli.md) | Command-line reference |
| [`SECURITY.md`](SECURITY.md) | Dependency policy and checkpoint loading |

## Results

A ResNet-18 fine-tuned on all seven classes scores **0.69 balanced accuracy**
(95% CI 0.685 to 0.703) on 1,870 held-back lesions. Chance is 0.143, since
balanced accuracy has a floor of `1/n_classes` for any rule that ignores the
image.

The figure that matters more is per class. The model finds **93 of 154
melanomas**, and of the 61 it misses, **30 are called ordinary moles**. For a
screening task those 30 are the real failures: a melanoma called basal cell
carcinoma still gets a referral, one called a nevus gets sent home.

Plain accuracy is 0.82, which sounds better and means less. Answering "nevus"
unconditionally scores 0.72 on this dataset while being useless.

### Limitations

Every figure comes from the validation split, which was also used to choose
thresholds and aggregation rules, so these are model-selection scores rather
than held-out estimates and are optimistic by an unknown amount. A held-out
test set exists for this data, the ISIC 2018 Task 3 set, and this project does
not use it.

The confidence interval above covers variation from random cropping at
evaluation time. It says nothing about variation between training runs, which
is likely larger and has not been measured.

HAM10000 records no ethnicity or Fitzpatrick skin type, and classifiers trained
on one population routinely underperform on another. Nothing here addresses
that.

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tmfreiberg/human-against-machine.git
cd human-against-machine
uv sync --group dev --all-extras
```

`uv sync` reads `pyproject.toml` and the committed `uv.lock`, so the
environment matches exactly what was tested. Dependency groups keep the install
proportionate:

| Group | Contents | Needed for |
|---|---|---|
| base | numpy, pandas, pillow | metadata handling |
| `--extra analysis` | scikit-learn, matplotlib | metrics and figures |
| `--extra models` | torch, torchvision | training and inference |
| `--extra export` | onnx, onnxruntime | building the demo assets |
| `--group dev` | pytest, ruff, mypy, jupyterlab | development |

### The dataset

The images are not in this repository: 10,015 dermatoscopic JPEGs are too large
for git, and `.gitignore` excludes them. Download HAM10000 from the
[Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) and place the images and
`metadata.csv` under `images/`.

Then tell the tools where the project is:

```bash
export HAM10000_ROOT=/path/to/human-against-machine
```

Nothing needs this at import time, so the package still installs and tests
without it.

## Quick start

```bash
ham10000 info                                        # what is in the dataset
ham10000 split                                       # verify the split does not leak
ham10000 bench configs/04_balanced_random_crop.yaml  # how long will training take
```

Then a smoke test, and the real run:

```bash
ham10000 train configs/04_balanced_random_crop.yaml --limit 200 --epochs 1
ham10000 train configs/04_balanced_random_crop.yaml --num-workers 4
ham10000 results
```

Training takes about 40 minutes on six CPU cores and roughly 10 minutes on any
CUDA GPU. `ham10000 bench` will tell you which applies to your machine.

See [`docs/cli.md`](docs/cli.md) for the full command reference.

## Running the demo locally

The [published version](https://<user>.github.io/human-against-machine/) needs
no setup. To build and serve it yourself:

```bash
ham10000 train configs/08_demo_melanoma_vs_nevus.yaml --num-workers 4
ham10000 export models/demo-melanoma-vs-nevus-<id>
cd demo && python -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/`. The page downloads a 10 MB quantised model
and runs it in your browser on a randomly drawn lesion. Nothing is uploaded.

The images come from the validation split, one per lesion, balanced across the
two classes, so the model has never seen them and the game cannot be won by
always answering "mole".

## Development

```bash
uv run pytest        # unit, integration and end-to-end
uv run ruff check .
uv run mypy src/
uv run pre-commit install
```

`pre-commit` runs `nbstripout` on commit, which keeps notebook outputs out of
the repository.

Experiments are configuration, not code. Each file in `configs/` is one
experiment, and its run identifier is a hash of the whole configuration, so a
run directory names exactly one set of settings and `config.yaml` is written
beside the artefacts it produced.

## Acknowledgements

This project began as a capstone for the Erdős Institute Deep Learning
Bootcamp, Spring 2024 cohort, with
[Bailey Forster](https://github.com/BaileyMForster) and
[Henri Antikainen](https://github.com/hpants). Everything since has been a
rewrite, but the original problem framing, the choice of dataset, and the
human-against-machine idea came out of that work.

## Data and citation

Tschandl, P., Rosendahl, C. & Kittler, H. (2018). The HAM10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin
lesions. *Scientific Data* 5, 180161.

## Licence

MIT. The HAM10000 dataset is distributed under CC BY-NC 4.0 and is not included
here.