"""SignaPy: exploratory feature discovery for labeled datasets.

SignaPy is in early development. ``signapy.discover`` currently supports one
slice: a binary target against categorical features. See
``docs/architecture.md`` for the design contract and roadmap, and
``docs/metrics.md`` for how to interpret the evidence it returns.
"""

from signapy.discovery import discover

__version__ = "0.1.0.dev0"

__all__ = ["__version__", "discover"]
