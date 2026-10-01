"""Value-level localization for categorical and ordinal features.

Pure computation over a feature/target pair. This module does not infer
types, choose a method, or know anything about :mod:`signapy.results`.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class CategoricalValueMetrics:
    """Value-level evidence for one category or ordinal level.

    Attributes:
        value: The feature category being described.
        support: Number of rows with this category.
        positive_count: Number of those rows where the target equals the
            configured positive class.
        positive_rate: ``positive_count / support``.
        baseline_rate: Positive rate across the whole input (not just this
            category).
        lift: ``positive_rate / baseline_rate``. Above 1 means the positive
            class is over-represented for this category relative to the
            overall dataset.
    """

    value: Hashable
    support: int
    positive_count: int
    positive_rate: float
    baseline_rate: float
    lift: float

    def __post_init__(self) -> None:
        if self.support < 0:
            raise ValueError(f"support must be non-negative, got {self.support}")
        if not 0 <= self.positive_count <= self.support:
            raise ValueError(
                f"positive_count ({self.positive_count}) must be between 0 "
                f"and support ({self.support})"
            )
        if not 0.0 <= self.positive_rate <= 1.0:
            raise ValueError(
                f"positive_rate must be in [0, 1], got {self.positive_rate}"
            )
        if not 0.0 <= self.baseline_rate <= 1.0:
            raise ValueError(
                f"baseline_rate must be in [0, 1], got {self.baseline_rate}"
            )


def categorical_lift(
    feature: npt.ArrayLike | pd.Series,
    target: npt.ArrayLike | pd.Series,
    *,
    positive_class: Hashable,
) -> tuple[CategoricalValueMetrics, ...]:
    """Per-category support, positive rate, baseline rate, and lift.

    Args:
        feature: Categorical values, one per row. Accepts a
            :class:`pandas.Series`, :class:`numpy.ndarray`, or plain
            sequence.
        target: Target labels, one per row, aligned by **position** with
            ``feature`` (not by pandas index — if both are
            :class:`~pandas.Series` with mismatched indexes, they are still
            paired up by position, not label). Must have exactly two
            distinct values.
        positive_class: The target value treated as the positive class. Its
            rate defines both ``baseline_rate`` and each category's
            ``positive_rate``.

    Returns:
        One :class:`CategoricalValueMetrics` per distinct feature value, in
        order of first appearance in ``feature``.

    Raises:
        ValueError: If ``feature`` and ``target`` have different lengths,
            are empty, contain missing values, if ``target`` does not have
            exactly two distinct values, if ``positive_class`` is not one of
            those two observed values (treated as an invalid argument, not a
            statistical edge case — this is almost always a typo), or, as a
            defensive fallback that should be unreachable given the checks
            above, if the baseline positive rate is otherwise zero.

    Missing values are rejected rather than silently dropped or treated as
    their own category: this is a low-level metrics function, and that
    policy decision belongs to the discovery layer that calls it.
    """
    # Convert to bare numpy arrays, not pandas Series: two equal-length
    # Series can carry different (e.g. one reset, one not) pandas indexes,
    # and boolean-indexing one Series by a mask derived from the other then
    # aligns by label, not position, which raises IndexingError or silently
    # misaligns feature/target pairs. Positional alignment is the documented
    # contract, so working in plain arrays avoids the ambiguity entirely.
    feature = pd.Series(feature).to_numpy()
    target = pd.Series(target).to_numpy()

    if len(feature) != len(target):
        raise ValueError(
            f"feature and target must have the same length, got "
            f"{len(feature)} and {len(target)}"
        )
    if len(feature) == 0:
        raise ValueError("feature and target must not be empty")
    if pd.isna(feature).any() or pd.isna(target).any():
        raise ValueError(
            "categorical_lift does not accept missing values; clean or "
            "impute feature/target before calling"
        )

    unique_targets = pd.unique(target)
    if len(unique_targets) != 2:
        labels = sorted(str(v) for v in unique_targets)
        raise ValueError(
            f"categorical_lift requires a binary target, found "
            f"{len(unique_targets)} distinct classes: {labels}"
        )
    if positive_class not in unique_targets:
        labels = sorted(str(v) for v in unique_targets)
        raise ValueError(
            f"positive_class {positive_class!r} is not one of the observed "
            f"target classes {labels}; check for a typo or the wrong value"
        )

    is_positive = target == positive_class
    baseline_rate = float(np.mean(is_positive))
    if baseline_rate == 0.0:
        # Unreachable in practice: positive_class is confirmed above to be
        # one of the two observed classes in a non-empty target, so its rate
        # must be > 0. Kept as a guard against future changes to the checks
        # above rather than surfacing a division by zero.
        raise ValueError("baseline positive rate is zero; lift is undefined")

    results = []
    for value in pd.unique(feature):
        value_mask = feature == value
        support = int(np.sum(value_mask))
        positive_count = int(np.sum(is_positive[value_mask]))
        positive_rate = positive_count / support
        results.append(
            CategoricalValueMetrics(
                value=value,
                support=support,
                positive_count=positive_count,
                positive_rate=positive_rate,
                baseline_rate=baseline_rate,
                lift=positive_rate / baseline_rate,
            )
        )
    return tuple(results)
