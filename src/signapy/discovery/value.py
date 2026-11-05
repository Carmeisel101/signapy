"""Value-level discovery: adapt low-level lift metrics to ``ValueResult``.

This module contains no statistics of its own. It calls
:func:`signapy.metrics.lift.categorical_lift` and renames its fields to the
:class:`~signapy.results.ValueResult` vocabulary used across SignaPy's
discovery results. Continuous features are localized by first cutting them
into adaptive bins (:func:`build_binned_value_results`) and then treating
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


# Continuous localization cuts a feature into at most this many bins,
# and never into bins averaging fewer than _MIN_ROWS_PER_BIN rows.
MAX_BINS = 5
_MIN_ROWS_PER_BIN = 5


def _marks(numbers: npt.NDArray[np.float64], precision: int = 3) -> list[str]:
    """Distinct text for each of the (distinct, sorted) ``numbers``."""
    # Widen the precision until every mark is distinct, so two bins can
    # never collapse into one value when numbers are close together. The last
    # step is the shortest text that round-trips the float, always distinct.
    for digits in (d for d in (3, 4, 6) if d >= precision):
        marks = [f"{number:.{digits}g}" for number in numbers]
        if len(set(marks)) == len(marks):
            return marks
    return [repr(float(number)) for number in numbers]


def _bin_labels(edges: npt.NDArray[np.float64], precision: int = 3) -> list[str]:
    """Interval labels for consecutive ``edges``; the first bin is closed."""
    if len(edges) < 2:
        return []
    marks = _marks(edges, precision)
    labels = [f"({low}, {high}]" for low, high in itertools.pairwise(marks)]
    labels[0] = "[" + labels[0][1:]
    return labels


def _quantile_bins(
    values: pd.Series, n_bins: int, precision: int = 3
) -> tuple[npt.NDArray[np.intp], list[str]]:
    """Cut ``values`` into up to ``n_bins`` quantile bins.

    When ties collapse duplicate edges, retries with fewer bins so a full set
    of ``k`` distinct bins is preferred; if no ``k`` survives intact, the
    attempt that kept the most bins is used.
    """
    best: tuple[npt.NDArray[np.intp], list[str]] = (np.empty(0, dtype=np.intp), [])
    for k in range(n_bins, 1, -1):
        codes, edges = pd.qcut(values, k, labels=False, retbins=True, duplicates="drop")
        labels = _bin_labels(np.asarray(edges, dtype=float), precision)
        if len(labels) == k:
            return codes.to_numpy(), labels
        if len(labels) > len(best[1]):
            best = (codes.to_numpy(), labels)
    return best


def _segment_bins(
    segment: npt.NDArray[np.float64],
    n_bins: int,
    max_distinct: int,
    precision: int = 3,
) -> tuple[npt.NDArray[np.object_], list[str]]:
    """Label every row of ``segment`` with its bin; also the labels in order.

    A segment with at most ``max_distinct`` distinct values gets one bin per
    value; otherwise it is cut into up to ``n_bins`` quantile bins, falling
    back to a single bin when ties leave no usable cut.
    """
    uniques, inverse = np.unique(segment, return_inverse=True)
    if len(uniques) <= max_distinct:
        labels = _marks(uniques, precision)
        return np.asarray(labels, dtype=object)[inverse.reshape(-1)], labels
    if n_bins >= 2:
        codes, labels = _quantile_bins(pd.Series(segment), n_bins, precision)
        if len(labels) >= 2:
            return np.asarray(labels, dtype=object)[codes], labels
    low, high = _marks(np.array([uniques[0], uniques[-1]]), precision)
    label = f"[{low}, {high}]"
    return np.full(len(segment), label, dtype=object), [label]


def _spike_bins(
    values: npt.NDArray[np.float64], n_bins: int
) -> tuple[npt.NDArray[np.object_], list[str]] | None:
    """Bins around a single dominant value, or ``None`` when there isn't one.

    A value holding more rows than one quantile bin would (such as the zeros
    in a zero-inflated column) cannot be split by quantiles. It becomes its
    own bin, labelled by the value, and the rows below and above it are
    binned separately so no interval overlaps the spike.
    """
    uniques, counts = np.unique(values, return_counts=True)
    top = int(np.argmax(counts))
    if counts[top] * n_bins <= len(values):
        return None
    # The spike and each side are labelled independently, so nearby but
    # distinct numbers can print alike; widen precision until every label
    # across the whole partition is distinct.
    for precision in (3, 17):  # 17: exact text
        result = _partition_around(values, n_bins, uniques[top], precision)
        if len(set(result[1])) == len(result[1]):
            break
    return result


def _partition_around(
    values: npt.NDArray[np.float64], n_bins: int, spike: float, precision: int
) -> tuple[npt.NDArray[np.object_], list[str]]:
    """The spike-and-sides partition of :func:`_spike_bins`, at one precision."""
    sides = [values < spike, values > spike]
    sides = [side for side in sides if side.any()]
    rest = int(sum(side.sum() for side in sides))

    budget = max(n_bins - 1, 1)
    if len(sides) == 2 and budget >= 2:
        low_share = sides[0].sum() / rest
        low = min(max(round(budget * low_share), 1), budget - 1)
        shares = [low, budget - low]
    else:
        shares = [budget] * len(sides)

    out = np.empty(len(values), dtype=object)
    below_order: list[str] = []
    above_order: list[str] = []
    for side, k in zip(sides, shares, strict=True):
        k = max(1, min(k, int(side.sum()) // _MIN_ROWS_PER_BIN))
        side_labels, order = _segment_bins(values[side], k, k, precision)
        out[side] = side_labels
        if values[side][0] < spike:
            below_order = order
        else:
            above_order = order
    spike_label = _marks(np.array([spike]), precision)[0]
    out[values == spike] = spike_label
    return out, [*below_order, spike_label, *above_order]


def build_binned_value_results(
    feature: npt.ArrayLike | pd.Series,
    target: npt.ArrayLike | pd.Series,
    *,
    positive_class: Hashable,
) -> tuple[ValueResult, ...] | None:
    """Value-level evidence for a continuous feature, via adaptive binning.

    The feature is cut into up to :data:`MAX_BINS` bins, using quantiles of
    roughly equal size for ordinary continuous data (ties can make them
    unequal, which each bin's ``support`` reports). When ties collapse bins,
    the cut is retried with fewer bins; a
    feature with at most :data:`MAX_BINS` distinct values gets one bin per
    value, labelled by the value itself (``"0"``, ``"1"``). A single value
    that holds more rows than one bin would (the zeros in a zero-inflated
    column) becomes its own bin, with the rows below and above it binned
    separately, so intervals never overlap it. Each bin is
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
    array = values.to_numpy()
    if len(np.unique(array)) < 2:
        return None
    spiked = _spike_bins(array, n_bins) if len(np.unique(array)) > MAX_BINS else None
    # A dominant value is isolated whether or not quantile edges collapsed:
    # an interval that contains it would overstate its neighbours' support.
    bin_labels, labels = spiked or _segment_bins(array, n_bins, MAX_BINS)
    if len(labels) < 2:
        return None

    by_label = {
        result.value: result
        for result in build_value_results(
            bin_labels, labels_target, positive_class=positive_class
        )
    }
    return tuple(by_label[label] for label in labels if label in by_label)
