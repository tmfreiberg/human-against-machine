# Development notes

How to set up, check and build this project. The overview is in the
[README](../README.md); the command-line tool is described in
[cli.md](cli.md).

## Environment

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. The lock
file `uv.lock` is committed, and `uv sync` installs exactly what it records:

```bash
uv sync --all-extras
uv run pre-commit install
```

PyTorch comes with the `models` extra. Without it the package still imports,
and the tests that need PyTorch are skipped (see `conftest.py` at the root).
CI runs the tests both ways.

Run tools through `uv run` (`uv run ruff`, `uv run mypy`, `uv run pytest`)
rather than any copy installed elsewhere, so that their versions are the ones
in the lock file.

## Checks

These are the commands CI runs. Run all four before committing:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest
```

The pre-commit hook runs Ruff only. The hook's version (`rev:` in
`.pre-commit-config.yaml`) should equal the Ruff version in `uv.lock`
(`uv run ruff --version`); otherwise the hook and CI can disagree about
formatting.

## Data and trained runs

`images/metadata.csv` is tracked. The 10,015 HAM10000 images are not: they go
in `images/` beside it (`images/ISIC_0024306.jpg` and so on). Each run
directory under `models/` is tracked except for its weights, `model.pth`.

## The book

The book is a [Quarto](https://quarto.org/) project: `_quarto.yml` and
`index.qmd` at the root, the three notebooks in `notebooks/`, and the other
chapters in `book/`.

The notebooks are committed with their outputs, and CI renders the book
without executing anything. So:

- To change prose, edit it and check the result with
  `quarto render --no-execute`.
- To change an output, run the notebook again (it needs the images and the
  trained runs), then commit the notebook with its new outputs.
- Do not strip notebook outputs, by hand or with a tool such as nbstripout.
  The published book would show empty cells.

A plain `quarto render` executes the notebooks.

The book is published to GitHub Pages by `.github/workflows/publish.yml`
when changes to its sources reach `main`.

## The demo

`demo/index.html` runs the exported melanoma-versus-nevus model in the
browser with onnxruntime-web, loading the files in `demo/public/model/`. To
try it locally, serve the folder and open <http://127.0.0.1:8000/>:

```bash
uv run python -m http.server 8000 --bind 127.0.0.1 --directory demo
```

Opening `index.html` straight from disk does not work: the browser will not
load the model into a `file://` page.

`uv run ham10000 export --help` describes how the demo's files are built from
a trained two-class run. The repository ignores `*.jpg`, so a new set of
sample images has to be added with `git add -f`.

## Dependencies

Dependabot proposes updates weekly for Python packages (minor and patch
updates grouped) and monthly for GitHub Actions and the pre-commit hook. A
pull request is merged only when CI passes.

The audit job runs pip-audit over every locked package, transitive ones
included, and fails on any known advisory. When an advisory concerns a
package this project only pulls in indirectly, the fix is a lower bound in the
`dev` dependency group of `pyproject.toml`, with a comment naming the
advisory, followed by `uv lock`.
