# Security

## Scope

This is a research and teaching repository. It is not a deployed service, it
handles no user data, and it must not be used for clinical decisions. Reports
here concern the code and its dependency graph.

## Reporting

Open a private security advisory through the repository's Security tab.

## Maintenance posture

Dependencies are declared with explicit version constraints in
`pyproject.toml` and resolved into a committed `uv.lock`. The constraints
matter: Dependabot will not propose a lockfile upgrade for a dependency
declared without one.

Dependabot proposes updates weekly. CI runs `pip-audit` on every push, every
pull request, and on a weekly schedule, so an advisory published after the last
commit still surfaces.

## Notes on the model artefacts

Model weights are loaded through `ham10000.serialization.load_state_dict`,
which sets `weights_only=True`. Without it, `torch.load` unpickles arbitrary
Python objects, so loading a checkpoint from an untrusted source is equivalent
to executing it. No dependency upgrade fixes this: a call site that opts out of
the safe path is unsafe at any version.

A failed load raises rather than printing and continuing. The alternative
leaves inference running on randomly initialised weights, which produces a
plausible-looking probability table made of noise.

## Notes on the browser demo

The demo under `demo/` is static files plus an ONNX model executed in the
visitor's browser. Nothing is uploaded, and no image or answer leaves the
machine. The exported model is served from the same origin as the page.
