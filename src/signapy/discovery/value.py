"""Value-level discovery: adapt low-level lift metrics to ``ValueResult``.

This module contains no statistics of its own. It calls
:func:`signapy.metrics.lift.categorical_lift` and renames its fields to the
:class:`~signapy.results.ValueResult` vocabulary used across SignaPy's
discovery results. Continuous features are localized by first cutting them
into quantile bins (:func:`build_binned_value_results`) and then treating
each bin as a value.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable

import numpy as np
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


# Continuous localization cuts a feature into at most this many quantile bins,
# and never into bins averaging fewer than _MIN_ROWS_PER_BIN rows.
MAX_BINS = 5
_MIN_ROWS_PER_BIN = 5


def _marks(numbers: npt.NDArray[np.float64]) -> list[str]:
    """Distinct text for each of the (distinct, sorted) ``numbers``."""
    # Widen the precision until every mark is distinct, so two bins can
    # never collapse into one value when numbers are close together.
    for digits in (3, 4, 6, 17):
        marks = [f"{number:.{digits}g}" for number in numbers]
        if len(set(marks)) == len(marks):
            break
    return marks


def _bin_labels(edges: npt.NDArray[np.float64]) -> list[str]:
    """Interval labels for consecutive ``edges``; the first bin is closed."""
    if len(edges) < 2:
        return []
    marks = _marks(edges)
    labels = [f"({low}, {high}]" for low, high in itertools.pairwise(marks)]
    labels[0] = "[" + labels[0][1:]
    return labels


def _quantile_bins(
    values: pd.Series, n_bins: int
) -> tuple[npt.NDArray[np.intp], list[str]]:
    """Cut ``values`` into up to ``n_bins`` quantile bins.

    When ties collapse duplicate edges, retries with fewer bins so a full set
    of ``k`` distinct bins is preferred; if no ``k`` survives intact, the
    attempt that kept the most bins is used.
    """
    best: tuple[npt.NDArray[np.intp], list[str]] = (np.empty(0, dtype=np.intp), [])
    for k in range(n_bins, 1, -1):
        codes, edges = pd.qcut(values, k, labels=False, retbins=True, duplicates="drop")
        labels = _bin_labels(np.asarray(edges, dtype=float))
        if len(labels) == k:
            return codes.to_numpy(), labels
        if len(labels) > len(best[1]):
            best = (codes.to_numpy(), labels)
    return best


def build_binned_value_results(
    feature: npt.ArrayLike | pd.Series,
    target: npt.ArrayLike | pd.Series,
    *,
    positive_class: Hashable,
) -> tuple[ValueResult, ...] | None:
    """Value-level evidence for a continuous feature, via quantile bins.

    The feature is cut into up to :data:`MAX_BINS` quantile bins of roughly
    equal size (ties can make them unequal, which each bin's ``support``
    reports). When ties collapse bins, the cut is retried with fewer bins; a
    feature with at most :data:`MAX_BINS` distinct values gets one bin per
    value, labelled by the value itself (``"0"``, ``"1"``). Each bin is
    described like any other value: support, target rate, baseline rate, and
    lift. Otherwise bin labels are interval strings such as ``"(38.4, 44.1]"``
    (the first bin is closed on the left: ``"[10, 20]"``), and results are
    returned in ascending bin order.

    Args:
        feature: Finite numeric values, one per row, with no missing values
            (callers apply SignaPy's missing-data policy first).
        target: Target labels, positionally aligned with ``feature``.
        positive_class: The target value treated as the positive class.

    Returns:
        One :class:`~signapy.results.ValueResult` per bin, or ``None`` when
        the feature cannot be split into at least two bins (too few rows, or
        a constant feature). ``None`` means "not performed", never an
        empty result.
    """
    values = pd.Series(feature).astype(float).reset_index(drop=True)
    labels_target = pd.Series(target).reset_index(drop=True)

    n_bins = min(MAX_BINS, len(values) // _MIN_ROWS_PER_BIN)
    if n_bins < 2:
        return None
    uniques, inverse = np.unique(values.to_numpy(), return_inverse=True)
    if len(uniques) < 2:
        return None
    if len(uniques) <= MAX_BINS:
        # Few distinct values: each one is its own bin, so ties can't hide it.
        codes, labels = inverse.reshape(-1), _marks(uniques)
    else:
        codes, labels = _quantile_bins(values, n_bins)
        if len(labels) < 2:
            return None

    bin_labels = np.asarray(labels, dtype=object)[codes]
    by_label = {
        result.value: result
        for result in build_value_results(
            bin_labels, labels_target, positive_class=positive_class
        )
    }
    return tuple(by_label[label] for label in labels if label in by_label)
