"""SignaPy: exploratory feature discovery for labeled datasets.

SignaPy is pre-1.0, so the API may still change between minor releases.
``signapy.discover`` currently supports a binary target
against mixed categorical, continuous, and ordinal features. See
``docs/architecture.md`` for the design contract and roadmap, and
``docs/metrics.md`` for how to interpret the evidence it returns.
"""

from signapy.discovery import discover

__version__ = "0.1.0"

__all__ = ["__version__", "discover"]
