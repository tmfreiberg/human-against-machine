"""Output helpers that adapt to notebook and terminal contexts.

Two functions, both concerned with making the same code readable in a notebook
and in a terminal without the caller having to branch.
"""

from __future__ import annotations

import functools

__all__ = ["display", "in_notebook", "print_header"]


@functools.cache
def in_notebook() -> bool:
    """Report whether the current process is an interactive IPython kernel.

    Testing whether `IPython` can be *imported* is a different question, and
    the wrong one: IPython is present in most scientific environments
    regardless of how the code is being run, so a plain script would take the
    notebook branch. Testing for a live kernel gives the intended behaviour.

    Returns
    -------
    bool
        True inside a Jupyter kernel or qtconsole; False in a terminal REPL, a
        script, or a test run.

    Notes
    -----
    Cached: the answer cannot change within a process, and this is called from
    inside display loops.

    Examples
    --------
    >>> isinstance(in_notebook(), bool)
    True
    >>> in_notebook()  # under pytest
    False
    """
    try:
        # Imported from the defining module: `IPython.get_ipython` is a
        # re-export without an explicit __all__ entry, which mypy rejects.
        from IPython.core.getipython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()  # type: ignore[no-untyped-call]
    if shell is None:
        return False
    # ZMQInteractiveShell is the Jupyter kernel; TerminalInteractiveShell is
    # `ipython` at a prompt, which has no rich display.
    return type(shell).__name__ == "ZMQInteractiveShell"


def display(*objects: object) -> None:
    """Render objects richly in a notebook, or print them otherwise.

    Lets the same module produce readable DataFrame tables in Jupyter and
    readable text in a terminal or CI log, without the caller branching.

    Parameters
    ----------
    *objects:
        Anything renderable. DataFrames render as HTML tables in a notebook.

    Examples
    --------
    Outside a notebook this is equivalent to `print`, one object per line:

    >>> display("first", "second")
    first
    second

    Called with nothing, it does nothing:

    >>> display()
    """
    if in_notebook():
        from IPython.display import display as ipython_display

        for obj in objects:
            ipython_display(obj)  # type: ignore[no-untyped-call]
    else:
        for obj in objects:
            print(obj)


def print_header(header: str) -> None:
    """Print an upper-cased heading between rules of matching width.

    Used to separate sections of notebook output.

    Parameters
    ----------
    header:
        Heading text. Upper-cased for display; the argument is not modified.

    Examples
    --------
    >>> print_header("class distribution")
    <BLANKLINE>
    ==================
    CLASS DISTRIBUTION
    ==================
    <BLANKLINE>

    The rules match the length of the *original* string, so a heading whose
    case-folding changes its length still aligns:

    >>> print_header("abc")
    <BLANKLINE>
    ===
    ABC
    ===
    <BLANKLINE>
    """
    rule = "=" * len(header)
    print("\n".join(["", rule, header.upper(), rule, ""]))
