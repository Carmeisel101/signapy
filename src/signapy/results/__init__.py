"""Structured, immutable result models returned by SignaPy.

SignaPy returns typed result objects rather than loose dictionaries or
printed output, so evidence can be inspected, compared, and exported.
"""

from signapy.results.models import DiscoveryReport, FeatureResult, ValueResult

__all__ = ["DiscoveryReport", "FeatureResult", "ValueResult"]
