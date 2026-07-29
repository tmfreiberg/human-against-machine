"""HAM10000 skin-lesion classification.

Importing this package has no side effects and requires no environment
variable. Project-root discovery is explicit and deferred to
:meth:`ham10000.config.Settings.resolve`.
"""

from __future__ import annotations

from ham10000.config import ProjectRootNotFoundError, Settings
from ham10000.display import display, print_header
from ham10000.serialization import CheckpointError, load_state_dict

__all__ = [
    "CheckpointError",
    "ProjectRootNotFoundError",
    "Settings",
    "__version__",
    "display",
    "load_state_dict",
    "print_header",
]

__version__ = "0.2.0.dev0"
