"""Project-root discovery and canonical directory layout.

Three properties make this usable on a machine that has never seen the
project:

1. **Nothing runs at import time.** Resolving the root is an explicit call, so
   importing the package cannot fail because an environment variable is unset.
   Without that property the package could not be tested, packaged, or run in
   CI at all.
2. **Nothing is interactive.** Resolution never prompts, so it cannot block in
   CI, a script, or a background job.
3. **Nothing is bound at definition time.** The root is resolved per call
   rather than captured in a default argument, which would freeze whatever
   value happened to exist when the module was first imported.

Directories are exposed as typed properties rather than a dictionary, so mypy
can check them and an editor can autocomplete them.

When the root cannot be found, the exception explains how to fix it, with
instructions matched to the caller's operating system. Putting that guidance in
the exception message rather than an interactive prompt makes it equally useful
to a human at a REPL and to someone reading a CI log.

Resolution order
----------------
:meth:`Settings.resolve` tries the following in order and stops at the first
hit:

1. an explicit ``root`` argument;
2. the ``HAM10000_ROOT`` environment variable;
3. an upward walk from ``start`` (default: the current working directory)
   looking for a marker file that identifies the repository root.

If all four fail, :class:`ProjectRootNotFoundError` is raised with actionable
instructions. Nothing in this module raises at import time.

Examples
--------
Construction is pure path arithmetic and touches no filesystem, which keeps it
trivially testable::

    >>> from pathlib import Path
    >>> settings = Settings(root=Path("/data/ham10000"))
    >>> print(settings.images.as_posix())
    /data/ham10000/images
    >>> print(settings.models.as_posix())
    /data/ham10000/models

Validation is opt-in and explicit, so a caller decides when to pay for I/O::

    >>> settings.images.name
    'images'
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "ENV_VAR",
    "ProjectRootNotFoundError",
    "Settings",
]

#: Environment variable naming the repository root.
ENV_VAR: Final = "HAM10000_ROOT"

#: Filenames that mark a directory as the repository root during the upward
#: walk. ``pyproject.toml`` is checked first because a git worktree or a
#: vendored checkout may contain ``.git`` without being the project root.
ROOT_MARKERS: Final = ("pyproject.toml", ".git")

#: Directories the project expects beneath the root. ``models`` is listed even
#: though it is absent from a fresh clone: ``.gitignore`` excludes ``*.pth``,
#: so trained weights live there untracked.
KNOWN_SUBDIRECTORIES: Final = (
    "expository",
    "images",
    "literature",
    "models",
    "notebooks",
    "presentation",
    "streamlit",
)


class ProjectRootNotFoundError(RuntimeError):
    """Raised when the repository root cannot be determined.

    The message includes platform-appropriate instructions for setting
    :data:`ENV_VAR`, because the overwhelmingly common cause is a fresh clone
    on a machine where the variable was never set.
    """


def _setup_instructions(*, env_var: str = ENV_VAR, osname: str | None = None) -> str:
    """Return copy-pasteable instructions for setting the root variable.

    Parameters
    ----------
    env_var:
        Name of the environment variable to describe.
    osname:
        Value of :data:`os.name` to target. Defaults to the running platform.
        Passing it explicitly makes the function testable on any host.

    Returns
    -------
    str
        A newline-separated, numbered list of steps.

    Examples
    --------
    >>> print(_setup_instructions(env_var="HAM10000_ROOT", osname="posix"))
    To fix this, set HAM10000_ROOT to the repository root:
      1. Open a terminal.
      2. Run:  export HAM10000_ROOT=/path/to/HAM10000-skin-lesion-classification
      3. Verify:  echo $HAM10000_ROOT
      4. Add the export line to your shell profile to make it permanent.

    >>> "setx" in _setup_instructions(osname="nt")
    True
    """
    if osname is None:  # pragma: no cover - trivial platform passthrough
        osname = os.name

    if osname == "nt":
        steps = (
            "1. Open PowerShell.",
            f"2. Run:  setx {env_var} "
            "C:\\path\\to\\HAM10000-skin-lesion-classification",
            f"3. Open a *new* terminal, then verify:  echo $env:{env_var}",
            "4. Restart your editor so it picks up the new environment.",
        )
    else:
        steps = (
            "1. Open a terminal.",
            f"2. Run:  export {env_var}=/path/to/HAM10000-skin-lesion-classification",
            f"3. Verify:  echo ${env_var}",
            "4. Add the export line to your shell profile to make it permanent.",
        )

    header = f"To fix this, set {env_var} to the repository root:"
    return "\n".join([header, *(f"  {step}" for step in steps)])


def _search_upward(start: Path, markers: tuple[str, ...] = ROOT_MARKERS) -> Path | None:
    """Walk upward from ``start`` looking for a directory containing a marker.

    Parameters
    ----------
    start:
        Directory to begin from. If a file is given, its parent is used.
    markers:
        Filenames identifying the root. Checked in order at each level, so an
        earlier marker in a *nearer* directory wins over a later marker in a
        more distant one.

    Returns
    -------
    Path or None
        The first matching directory, or ``None`` if the filesystem root is
        reached without a hit.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     root = Path(tmp)
    ...     _ = (root / "pyproject.toml").write_text("")
    ...     nested = root / "notebooks" / "deep"
    ...     nested.mkdir(parents=True)
    ...     _search_upward(nested) == root.resolve()
    True

    An unmarked tree yields ``None`` rather than raising, so the caller can
    fall through to the next resolution strategy:

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     _search_upward(Path(tmp)) is None
    True
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent

    # ``Path.parents`` excludes the path itself, so chain it explicitly.
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable view of the project's directory layout.

    The constructor performs no I/O and no validation: it is pure path
    arithmetic, so it can be built in a test with a path that does not exist.
    Use :meth:`resolve` for discovery and :meth:`require` when a directory must
    actually be present.

    Parameters
    ----------
    root:
        Repository root. Stored as given; call :meth:`resolve` if you want it
        normalised.

    Examples
    --------
    >>> from pathlib import Path
    >>> settings = Settings(root=Path("/srv/ham10000"))
    >>> print(settings.metadata_csv.as_posix())
    /srv/ham10000/images/metadata.csv

    Instances are frozen and hashable, so they are safe to cache or use as a
    dictionary key:

    >>> len({Settings(Path("/a")), Settings(Path("/a")), Settings(Path("/b"))})
    2
    """

    root: Path

    # -- Canonical subdirectories -------------------------------------------
    #
    # Named properties rather than a dynamic directory scan.
    # Attribute access is checkable by mypy and discoverable by an IDE, and it
    # does not silently change behaviour when a stray directory appears.

    @property
    def images(self) -> Path:
        """Directory holding ``metadata.csv`` and the dermatoscopic images."""
        return self.root / "images"

    @property
    def models(self) -> Path:
        """Directory holding trained weights and cached probability tables.

        Untracked: ``.gitignore`` excludes ``*.pth``, so this is absent from a
        fresh clone until weights are restored or retrained.
        """
        return self.root / "models"

    @property
    def notebooks(self) -> Path:
        """Directory holding the Jupyter notebooks."""
        return self.root / "notebooks"

    @property
    def literature(self) -> Path:
        """Directory holding reference papers."""
        return self.root / "literature"

    @property
    def expository(self) -> Path:
        """Directory holding expository write-ups."""
        return self.root / "expository"

    @property
    def presentation(self) -> Path:
        """Directory holding the bootcamp presentation."""
        return self.root / "presentation"

    @property
    def streamlit(self) -> Path:
        """Directory holding the human-versus-machine demo application."""
        return self.root / "streamlit"

    @property
    def metadata_csv(self) -> Path:
        """Full path to the HAM10000 metadata table."""
        return self.images / "metadata.csv"

    # -- Construction --------------------------------------------------------

    @classmethod
    def resolve(
        cls,
        root: Path | str | None = None,
        *,
        start: Path | str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Settings:
        """Locate the repository root and return a validated :class:`Settings`.

        See the module docstring for the full resolution order.

        Parameters
        ----------
        root:
            Explicit root. Bypasses all discovery when given.
        start:
            Directory the upward walk begins from. Defaults to the current
            working directory.
        env:
            Environment mapping to consult. Defaults to :data:`os.environ`.
            Injecting it keeps tests free of ``monkeypatch`` gymnastics.

        Returns
        -------
        Settings
            With ``root`` resolved to an absolute, symlink-free path.

        Raises
        ------
        ProjectRootNotFoundError
            If no strategy succeeds, or if the located path is not a directory.

        Examples
        --------
        An explicit root short-circuits discovery:

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     Settings.resolve(tmp).root == Path(tmp).resolve()
        True

        The environment is consulted next:

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     settings = Settings.resolve(env={ENV_VAR: tmp})
        ...     settings.root == Path(tmp).resolve()
        True

        Failure is explicit and instructive rather than a ``TypeError`` at
        import time:

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     Settings.resolve(start=tmp, env={})
        Traceback (most recent call last):
            ...
        ham10000.config.ProjectRootNotFoundError: Could not locate the...
        """
        environ: Mapping[str, str] = os.environ if env is None else env

        candidate: Path | None = None

        if root is not None:
            candidate = Path(root)
        elif environ.get(ENV_VAR):
            candidate = Path(environ[ENV_VAR])
        else:
            search_start = Path.cwd() if start is None else Path(start)
            candidate = _search_upward(search_start)

        if candidate is None:
            raise ProjectRootNotFoundError(
                "Could not locate the project root. Tried, in order: an "
                f"explicit argument, ${ENV_VAR}, and an upward search for "
                f"{' or '.join(ROOT_MARKERS)}.\n\n" + _setup_instructions()
            )

        candidate = candidate.expanduser().resolve()
        if not candidate.is_dir():
            raise ProjectRootNotFoundError(
                f"Resolved project root {candidate} is not a directory.\n\n"
                + _setup_instructions()
            )

        return cls(root=candidate)

    # -- Validation ----------------------------------------------------------

    def require(self, *names: str) -> Settings:
        """Assert that the named subdirectories exist, then return ``self``.

        Directories are checked eagerly and reported *together*, so a caller
        missing three of them learns that in one run instead of three.

        Parameters
        ----------
        *names:
            Attribute names of directory properties, e.g. ``"images"``.

        Returns
        -------
        Settings
            ``self``, to permit chaining.

        Raises
        ------
        AttributeError
            If a name is not a known directory property.
        FileNotFoundError
            If any named directory is absent.

        Examples
        --------
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     (Path(tmp) / "images").mkdir()
        ...     settings = Settings(Path(tmp))
        ...     settings.require("images") is settings
        True

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     Settings(Path(tmp)).require("images", "models")
        Traceback (most recent call last):
            ...
        FileNotFoundError: Missing required director...
        """
        missing: list[Path] = []
        for name in names:
            if name not in KNOWN_SUBDIRECTORIES:
                raise AttributeError(
                    f"{name!r} is not a known project directory. "
                    f"Expected one of: {', '.join(KNOWN_SUBDIRECTORIES)}."
                )
            path = getattr(self, name)
            if not path.is_dir():
                missing.append(path)

        if missing:
            listed = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Missing required directories:\n{listed}")
        return self
