# Command-line reference

Everything the project does is reachable from one command:

```
ham10000 <subcommand> [options]
```

Run `ham10000` with no arguments to list subcommands, or
`ham10000 <subcommand> --help` for a subcommand's options.

Two aliases exist for the commands used most often, so they can be run without
the umbrella: `ham10000-train` and `ham10000-bench`.

## Finding the project

Most subcommands need to locate the dataset. They look, in order, for an
explicit `--data-root`, then the `HAM10000_ROOT` environment variable, then
walk upward from the working directory looking for `pyproject.toml` or `.git`.

If none of those succeed the command fails with instructions for setting the
variable on your platform, rather than a `TypeError` from somewhere inside the
call stack.

```powershell
$env:HAM10000_ROOT = "C:\path\to\HAM10000-skin-lesion-classification"
```

---

## Looking at the data

### `ham10000 info`

Class distribution over lesions, the imbalance ratio, the balanced-accuracy
chance level, the distribution of images per lesion, and missing values.

```
ham10000 info [--data-root PATH]
```

The chance level is worth noting before reading any score: it is `1/n_classes`
for balanced accuracy, and no rule that ignores the image can beat it.

### `ham10000 frequencies`

Absolute and relative frequencies of any metadata column.

```
ham10000 frequencies COLUMN [--where COLUMN=VALUE] [--images]
```

Counts distinct lesions by default, since that is the unit everything is scored
on; `--images` counts rows instead. `--where` restricts before counting, so
`--where dx=mel` answers "among melanomas, how are body sites distributed?".

### `ham10000 images`

Renders lesion images to a file and prints their metadata.

```
ham10000 images --output FILE (--dx LIST | --lesion LIST | --image LIST) [options]
```

Three ways to choose what to show:

| Selector | Layout | Use |
|---|---|---|
| `--dx mel,nv` | one class per column, sampled | what a class looks like in general |
| `--lesion HAM_0002730` | one lesion per row, all its images | why a lesion was photographed repeatedly |
| `--image ISIC_0024468,...` | in the order given | a figure making a specific point |

Prefer `--image` for anything going into a document. A sampled figure depends
on a seed and changes silently if the sampling code is ever touched; naming the
images pins them.

`--captions` prints full metadata beneath each image. It is off by default
because eight lines under each thumbnail obscure a visual point. `--size` sets
inches per cell, `--rows` and `--cols` the grid.

### `ham10000 multiplicity`

Relates how often a lesion was photographed to how its diagnosis was
confirmed, and reports mean images per lesion by class.

```
ham10000 multiplicity [--dx CLASS] [--data-root PATH]
```

`--dx` selects the class to break down, defaulting to `nv`. The output shows
why `num_images` must never be used as a feature: multiplicity records whether
a clinician was concerned enough to biopsy, which is downstream of the label.

---

## Preparing the data

### `ham10000 split`

Split sizes and an explicit leakage check.

```
ham10000 split [CONFIG] [--data-root PATH]
```

With a config, uses that experiment's split settings; without one, the
defaults. The last line reports how many lesions appear on both sides of the
split, which should always be zero. That guarantee is what the lesion-level
split exists to provide, and this is how to verify it rather than assume it.

### `ham10000 balance`

Per-class resampling factors for a configuration.

```
ham10000 balance CONFIG [--data-root PATH]
```

Shows how many images of each class are available, the target, the resulting
repetition factor, and the number of distinct lesions behind it. A factor above
10 gets a warning: repeating 54 lesions twelve times risks memorising those 54
rather than learning the class.

---

## Running experiments

### `ham10000 configs`

Lists every experiment, its run identifier, and whether a completed run exists.

```
ham10000 configs [--directory DIR] [--data-root PATH]
```

The identifier is a hash of the whole configuration, so it names exactly one
experiment. Changing a setting changes the identifier; changing a comment does
not.

### `ham10000 bench`

Reports what the machine is and measures how fast it trains, then estimates how
long a given experiment will take on it.

```
ham10000 bench [CONFIG] [--device cpu|cuda|mps] [--quick]
```

Measurements are the median of several repeats, because a single short
measurement is unreliable while clock speeds ramp. Close other CPU-heavy work
first, including any training run in another terminal, or the estimate will be
optimistic in the wrong direction.

### `ham10000 train`

Trains a model and writes a run directory.

```
ham10000 train CONFIG [options]
```

| Option | Effect |
|---|---|
| `--limit N` | Use at most N lesions, sampled per class. For smoke tests. |
| `--epochs N` | Override the configured epoch count. |
| `--num-workers N` | DataLoader workers. 2 to 4 overlaps image decoding with compute and typically saves 10 to 20% of wall time on CPU. |
| `--device` | Force `cpu`, `cuda` or `mps`. |
| `--force` | Overwrite a completed run directory. |
| `--no-pretrained` | Random initialisation. Skips the weight download; only useful for smoke tests. |
| `--output PATH` | Where run directories go. Defaults to `<root>/models`. |

Always smoke test before committing to a long run:

```
ham10000 train configs/04_balanced_random_crop.yaml --limit 200 --epochs 1
```

Limited runs land in a directory suffixed `-smoke` and print a warning. Their
numbers are meaningless; the point is that every stage executes.

A completed run directory is protected. Re-running the same configuration
raises rather than overwriting. Do not edit a config to dodge the collision:
the config determines the run's identity, so editing it produces a different
experiment.

---

## Reading results

A run directory contains `config.yaml`, `model.pth`, `model.losses.json`,
`predictions.csv` and `metrics.json`. The config sits beside the artefacts, so
the directory records what it is rather than asking you to trust its name.

### `ham10000 results`

```
ham10000 results [RUN] [--data-root PATH]
```

Without an argument, reports on the most recent completed run that is not a
smoke test. Prints balanced accuracy with its chance level and skill above
chance, plain accuracy, the epoch with the lowest validation loss against the
epoch actually saved, a confusion matrix with per-class recall and support, and
melanoma sensitivity.

Read recall against support. A recall of 1.00 over eighteen lesions is not
evidence of much.

### `ham10000 compare`

Stacks the metrics of several completed runs into one table, best first.

```
ham10000 compare [RUN ...] [--sort-by METRIC]
```

Defaults to every completed run. Sorted by balanced accuracy rather than plain
accuracy, which would rank an always-majority model competitively. Adds a
`chance` and a `skill` column, because a raw balanced accuracy is not
comparable between a two-class and a seven-class run: a binary model at 0.78
has less skill than a seven-class model at 0.68.

### `ham10000 thresholds`

Re-decides a run's predictions under a sensitivity-biased rule and reports the
effect on each class.

```
ham10000 thresholds RUN [--promote mel=0.4,bcc=0.4] [--demote nv=0.6]
                        [--rule priority|cost-sensitive]
```

Plain argmax is the wrong rule for screening: a lesion at 45% melanoma against
50% nevus is called a nevus. Two rules are available and they are not
equivalent.

`priority` walks the promotion list in order and stops at the first class
clearing its bar, so the ordering carries the clinical priority. It is a gate:
below the bar it does nothing.

`cost-sensitive` divides every probability by its threshold and takes the
argmax. It acts unconditionally, so it shifts decisions even where nothing
crosses a bar, and it calls the promoted class noticeably more often.

Writes nothing back to the run. Read the per-class change column rather than
the headline: a rule that lifts melanoma recall while lowering everything else
may still be the one you want.

### `ham10000 predict`

Scores images with a trained checkpoint.

```
ham10000 predict RUN IMAGE [IMAGE ...]
```

Images may be given as ids from the dataset or as paths to `.jpg` files, but
all must come from one directory. The evaluation transform is stochastic, so
re-running gives slightly different probabilities for the same image.

### `ham10000 checkpoint`

Reports the parameter names and shapes in a `.pth` file.

```
ham10000 checkpoint PATH [--all]
```

Answers "is this file intact, and what was it trained for?" without needing to
guess the architecture first. The final layer's width is the number of classes,
which it reports directly.

### `ham10000 rescore`

```
ham10000 rescore RUN [--device ...] [--data-root PATH]
```

Recomputes predictions and metrics from the checkpoint on disk, rewriting them
in place. For repairing a run whose reported metrics describe a different model
from the one saved. Reuses the validation rows already in `predictions.csv`, so
the before and after figures cover the same lesions.

### `ham10000 views`

```
ham10000 views RUN [--views 1,3,5] [--repeats N]
```

Re-scores a checkpoint at several test-time augmentation settings, measuring
each several times so that any apparent gain can be compared against the
run-to-run spread. Writes nothing back to the run.

The evaluation transform is random, so a single measurement at one setting says
little. Three repeats is enough to see the spread exists; ten is enough to
quote it.

---

## The browser demo

### `ham10000 export`

```
ham10000 export RUN [--pool-size N] [--seed N] [--no-quantise] [--output PATH]
```

Exports a two-class run to browser-servable assets: an int8-quantised ONNX
model of about 10 MB, the class order and preprocessing constants, an image
pool, and the images themselves resized for the web.

Three properties are enforced rather than trusted. The pool is drawn from the
validation split only, so the model has never seen any of the images. One image
per lesion, so there are no near-duplicates. Balanced across classes, so the
game cannot be won by always answering with the majority class.

The run must have exactly two classes, since the demo presents two answers.
Train a restricted model for it, as `configs/08_demo_melanoma_vs_nevus.yaml`
does.

To view the result:

```powershell
cd demo
python -m http.server 8000 --bind 127.0.0.1
```

then open `http://127.0.0.1:8000/`. It needs a server rather than opening the
file directly, because `fetch` does not work from a `file://` origin.