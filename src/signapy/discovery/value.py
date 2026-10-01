"""Value-level discovery: adapt low-level lift metrics to ``ValueResult``.

This module contains no statistics of its own. It calls
:func:`signapy.metrics.lift.categorical_lift` and renames its fields to the
:class:`~signapy.results.ValueResult` vocabulary used across SignaPy's
discovery results.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy.typing as npt
import pandas as pd

from signapy.metrics.lift import categorical_lift
from signapy.results import ValueResult


def build_value_results(
    feature: npt.ArrayLike | pd.Series,
    target: npt.ArrayLike | pd.Series,
    *,
    positive_class: Hashable,
) -> tuple[ValueResult, ...]:
    """Value-level evidence for a categorical or ordinal feature.

    Args:
        feature: Categorical values or ordinal level labels, one per row.
            Callers are responsible for applying SignaPy's missing-data
            policy before calling this function; ``categorical_lift`` itself
            rejects missing values outright.
        target: Target labels, one per row, positionally aligned with
            ``feature`` (see :func:`~signapy.metrics.lift.categorical_lift`
            for what "positionally aligned" means for pandas ``Series``).
            Must have exactly two distinct values.
        positive_class: The target value treated as the positive class.

    Returns:
        One :class:`~signapy.results.ValueResult` per distinct feature
        value, in order of first appearance in ``feature``. ``target_count``
        and ``target_rate`` are ``categorical_lift``'s ``positive_count`` and
        ``positive_rate`` under SignaPy's results naming.

    Raises:
        ValueError: Anything :func:`~signapy.metrics.lift.categorical_lift`
            raises (empty input, mismatched lengths, missing values, a
            non-binary target, or an unobserved ``positive_class``).
    """
    lift_results = categorical_lift(feature, target, positive_class=positive_class)
    return tuple(
        ValueResult(
            value=result.value,
            support=result.support,
            target_count=result.positive_count,
            target_rate=result.positive_rate,
            baseline_rate=result.baseline_rate,
            lift=result.lift,
        )
        for result in lift_results
    )
