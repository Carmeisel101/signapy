"""Semantic feature and target types.

These describe what a column *means* statistically, not how it is stored.
A pandas dtype alone cannot determine semantic type: an integer column
holding ``0, 1, 2`` may be continuous, ordinal, or a set of category
identifiers. Type inference (planned) will propose a type from the data, and
users will be able to override it.
"""

from enum import Enum


class FeatureType(str, Enum):
    """Semantic type of a candidate feature."""

    CONTINUOUS = "continuous"
    """Numeric values where differences and ordering are meaningful."""

    ORDINAL = "ordinal"
    """Ordered categories where spacing between levels is not meaningful."""

    CATEGORICAL = "categorical"
    """Unordered categories (nominal)."""

    BOOLEAN = "boolean"
    """Exactly two levels, e.g. ``True``/``False`` or a 0/1 flag."""


class TargetType(str, Enum):
    """Semantic type of the prediction target, i.e. the learning task."""

    BINARY = "binary"
    """Binary classification (the primary target type for v0.1)."""

    MULTICLASS = "multiclass"
    """Classification with more than two classes."""

    CONTINUOUS = "continuous"
    """Regression on a continuous target."""
