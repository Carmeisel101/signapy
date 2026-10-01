"""Feature-level association metrics for categorical, continuous, and ordinal data.

Pure computation over a contingency table (:func:`cramers_v`) or a
feature/target pair (:func:`point_biserial`, :func:`spearman_rho`). This
module does not infer types, choose a method, or know anything about
:mod:`signapy.results`.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from decimal import Decimal
from numbers import Rational, Real

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats


def _validate_contingency_table(contingency_table: npt.ArrayLike) -> np.ndarray:
    """Convert and validate a 2D contingency table of non-negative counts.

    Raises:
        ValueError: If the table is not 2D, is empty, has fewer than two
            rows or columns, contains negative, non-finite, or fractional
            values, or has a row/column that sums to zero (making expected
            frequencies undefined).

    Values must be integer-valued counts. This is not just a formality: the
    bias-corrected Cramér's V (see :func:`cramers_v`) treats the table sum
    as a literal observation count ``n``, and its correction terms are only
    meaningful for actual counts — passing e.g. weights or proportions would
    silently produce a meaningless result rather than a clear error.
    """
    table = np.asarray(contingency_table, dtype=float)
    if table.size == 0:
        raise ValueError("contingency_table must not be empty")
    if table.ndim != 2:
        raise ValueError(
            f"contingency_table must be 2-dimensional, got shape {table.shape}"
        )
    if 0 in table.shape:
        raise ValueError("contingency_table must not be empty")
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(
            "contingency_table needs at least 2 categories and 2 target "
            f"classes to compute an association, got shape {table.shape}"
        )
    if not np.all(np.isfinite(table)):
        raise ValueError("contingency_table must contain only finite values")
    if np.any(table < 0):
        raise ValueError("contingency_table must not contain negative values")
    if not np.all(np.equal(table, np.round(table))):
        raise ValueError(
            "contingency_table must contain integer-valued counts, got "
            "fractional values (weighted/proportional tables are not "
            "supported)"
        )
    if np.any(table.sum(axis=1) == 0) or np.any(table.sum(axis=0) == 0):
        raise ValueError(
            "contingency_table has a row or column that sums to zero; "
            "expected frequencies are undefined"
        )
    return table


def cramers_v(
    contingency_table: npt.ArrayLike, *, bias_correction: bool = True
) -> float:
    """Cramér's V effect size for association between two categorical variables.

    Measures the strength of association in a contingency table, on a scale
    from 0 (no association) to 1 (perfect association). Unlike the
    chi-square statistic, it is not directly a function of sample size,
    which makes it comparable across features with different ``n``.

    Args:
        contingency_table: A 2D array-like of non-negative observed counts
            (rows = feature categories, columns = target classes). Accepts
            a :class:`numpy.ndarray`, nested sequence, or
            :class:`pandas.DataFrame`.
        bias_correction: If ``True`` (the default), apply the Bergsma (2013)
            bias correction, which reduces the small-sample upward bias of
            the standard estimator. If ``False``, compute the standard
            (Cramér, 1946) formula.

    Returns:
        Cramér's V, in ``[0, 1]``.

    Raises:
        ValueError: If ``contingency_table`` is invalid (see
            :func:`_validate_contingency_table`), or if, with
            ``bias_correction=True``, the corrected table dimensions
            collapse to one or fewer categories (the sample is too small
            relative to the table shape for the correction to be defined).

    The chi-square statistic underlying V is computed without Yates'
    continuity correction (``correction=False``), consistent with
    :func:`signapy.metrics.significance.chi_square`.
    """
    table = _validate_contingency_table(contingency_table)
    n = table.sum()
    chi2, _p_value, _dof, _expected = scipy.stats.chi2_contingency(
        table, correction=False
    )
    rows, cols = table.shape
    phi2 = chi2 / n

    if not bias_correction:
        denominator = min(rows - 1, cols - 1)
        return float(np.sqrt(phi2 / denominator))

    phi2_corrected = max(0.0, phi2 - (rows - 1) * (cols - 1) / (n - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (n - 1)
    cols_corrected = cols - (cols - 1) ** 2 / (n - 1)
    denominator = min(rows_corrected - 1, cols_corrected - 1)
    if denominator <= 0:
        raise ValueError(
            "bias-corrected Cramér's V is undefined for this table: the "
            "corrected dimensions collapse to one or fewer categories "
            f"(n={n:g} is too small relative to shape {table.shape}); "
            "retry with bias_correction=False"
        )
    return float(np.sqrt(phi2_corrected / denominator))


# Point-biserial correlation needs degrees of freedom n - 2 >= 1 for a
# well-defined p-value; below this, scipy's pointbiserialr can return a NaN
# p-value or emit its own warning rather than a clear error.
_MIN_POINT_BISERIAL_OBSERVATIONS = 3


def _validate_real_numeric_feature(array: np.ndarray, *, caller_name: str) -> None:
    """Require every feature value to be a real, non-boolean number.

    Shared by :func:`point_biserial` and :func:`spearman_rho` — both need a
    feature of plain real-valued measurements/ranks, not categorical
    booleans, calendar dates/durations, or complex numbers. ``caller_name``
    (e.g. ``"point_biserial"``) is folded into the error messages so they
    name whichever function the caller actually called.
    """
    object_values = array if array.dtype == object else ()

    if pd.api.types.is_bool_dtype(array.dtype) or any(
        isinstance(value, bool | np.bool_) for value in object_values
    ):
        raise ValueError(
            "feature must not be boolean; a boolean feature is categorical, "
            f"not continuous, and unsupported by {caller_name} in this "
            "release (see signapy.metrics.lift.categorical_lift instead)"
        )

    if (
        pd.api.types.is_datetime64_any_dtype(array.dtype)
        or pd.api.types.is_timedelta64_dtype(array.dtype)
        or any(
            isinstance(
                value, np.datetime64 | np.timedelta64 | pd.Timestamp | pd.Timedelta
            )
            for value in object_values
        )
    ):
        raise ValueError(
            f"feature must not be datetime- or timedelta-valued (got dtype "
            f"{array.dtype}); {caller_name} treats values as continuous "
            "measurements, not calendar dates or durations — convert to a "
            "numeric measurement (e.g. days since a reference date) before "
            "calling if that's what you intend"
        )

    if np.issubdtype(array.dtype, np.complexfloating) or any(
        isinstance(value, complex | np.complexfloating) for value in object_values
    ):
        raise ValueError(
            f"feature must be real-valued (got dtype {array.dtype}); "
            f"{caller_name} does not support complex numbers, which would "
            "otherwise be silently truncated to their real component"
        )

    if array.dtype == object:
        is_real_numeric = all(
            isinstance(value, Real | Decimal) for value in object_values
        )
    else:
        is_real_numeric = pd.api.types.is_numeric_dtype(array.dtype)
    if not is_real_numeric:
        raise ValueError(
            f"feature must be numeric; only real numeric values are accepted, "
            f"got dtype {array.dtype}"
        )


def _is_finite_real(value: Real | Decimal) -> bool:
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, Rational):
        return True
    return bool(np.isfinite(value))


@dataclass(frozen=True)
class PointBiserialResult:
    """Point-biserial correlation between a continuous feature and a binary target.

    Attributes:
        coefficient: The point-biserial correlation coefficient, in
            ``[-1, 1]``. Positive means larger feature values are associated
            with the positive class; negative means the opposite.
        p_value: p-value for the null hypothesis that the coefficient is
            zero (equivalent to a two-sided Pearson correlation test).
        positive_n: Number of rows where the target equals the configured
            positive class.
        negative_n: Number of rows where it does not.
        positive_mean: Mean feature value among positive-class rows.
        negative_mean: Mean feature value among negative-class rows.
        mean_difference: ``positive_mean - negative_mean``, in the
            feature's original units — unlike ``coefficient``, this carries
            scale, which is useful context the normalized coefficient
            doesn't give you.
    """

    coefficient: float
    p_value: float
    positive_n: int
    negative_n: int
    positive_mean: float
    negative_mean: float
    mean_difference: float

    def __post_init__(self) -> None:
        if not -1.0 - 1e-9 <= self.coefficient <= 1.0 + 1e-9:
            raise ValueError(f"coefficient must be in [-1, 1], got {self.coefficient}")
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")
        if self.positive_n < 1:
            raise ValueError(f"positive_n must be positive, got {self.positive_n}")
        if self.negative_n < 1:
            raise ValueError(f"negative_n must be positive, got {self.negative_n}")
        expected_difference = self.positive_mean - self.negative_mean
        if not np.isclose(self.mean_difference, expected_difference):
            raise ValueError(
                f"mean_difference ({self.mean_difference}) must equal "
                f"positive_mean - negative_mean ({expected_difference})"
            )


def point_biserial(
    feature: npt.ArrayLike, target: npt.ArrayLike, *, positive_class: Hashable
) -> PointBiserialResult:
    """Point-biserial correlation between a continuous feature and a binary target.

    Point-biserial correlation is the Pearson correlation between a
    continuous variable and a binary variable (the target, encoded 1 for
    ``positive_class`` and 0 otherwise). It answers a different question
    from :func:`cramers_v`: not "how strong is the association" in an
    undirected sense, but "do larger feature values tend to go with the
    positive or the negative class" — it has a sign.

    Args:
        feature: A 1D array-like of continuous, real-valued (numeric)
            measurements, one per row. Accepts Python numeric sequences,
            :class:`numpy.ndarray`, :class:`pandas.Series` (including
            pandas nullable numeric dtypes, once missing values are
            resolved), integer- or float-valued measurements. Rejected as
            not being a continuous measurement: boolean values (categorical,
            not continuous — including boolean values stored in an
            ``object``-dtype array/Series, not just true ``bool``/nullable
            ``boolean`` dtype), datetime and timedelta values (calendar
            dates and durations, not measurements), and complex values
            (not real-valued), and numeric strings (representations that
            would require coercion rather than numeric measurements).
        target: A 1D array-like of target labels, one per row, positionally
            aligned with ``feature`` (not by pandas index, if both are
            ``Series`` — see :func:`signapy.metrics.lift.categorical_lift`
            for what that distinction means in practice). Must have exactly
            two distinct values.
        positive_class: The target value encoded as 1 (the "positive" side
            of the correlation's sign). Reversing this reverses the sign of
            ``coefficient`` (and of ``mean_difference``) but not its
            magnitude.

    Returns:
        A :class:`PointBiserialResult` with the coefficient, its p-value,
        and both groups' counts and means.

    Raises:
        ValueError: If ``feature`` or ``target`` is not one-dimensional, if
            ``feature`` is boolean (including boolean values stored with
            ``object`` dtype), datetime-, timedelta-, or complex-valued, or
            otherwise not real-valued numeric, if they have different
            lengths, if either is empty, if either contains missing values,
            if ``feature`` contains a non-finite value (``inf`` or
            ``-inf``), if ``target`` does not have exactly two distinct
            values, if ``positive_class`` is not one of them, if fewer than
            :data:`_MIN_POINT_BISERIAL_OBSERVATIONS` total observations
            remain, or if ``feature`` is constant (zero variance, making
            the correlation undefined).

    Uses :func:`scipy.stats.pointbiserialr`, which computes the coefficient
    and its p-value in a single call rather than two.
    """
    feature_array = np.asarray(feature)
    target_array = np.asarray(target)

    if feature_array.ndim != 1:
        raise ValueError(
            f"feature must be one-dimensional, got shape {feature_array.shape}"
        )
    if target_array.ndim != 1:
        raise ValueError(
            f"target must be one-dimensional, got shape {target_array.shape}"
        )

    if len(feature_array) != len(target_array):
        raise ValueError(
            f"feature and target must have the same length, got "
            f"{len(feature_array)} and {len(target_array)}"
        )
    if len(feature_array) == 0:
        raise ValueError("feature and target must not be empty")

    if pd.isna(feature_array).any():
        raise ValueError(
            "point_biserial does not accept missing feature values; clean "
            "or impute before calling"
        )
    if pd.isna(target_array).any():
        raise ValueError(
            "point_biserial does not accept missing target values; clean "
            "or impute before calling"
        )

    _validate_real_numeric_feature(feature_array, caller_name="point_biserial")

    try:
        feature_values = pd.to_numeric(
            pd.Series(feature_array), errors="raise"
        ).to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"feature must be numeric, got dtype {feature_array.dtype}: {error}"
        ) from error

    if not np.all(np.isfinite(feature_values)):
        raise ValueError(
            "feature must not contain infinite values (inf or -inf); "
            "resolve or drop these rows before calling"
        )

    unique_targets = pd.unique(target_array)
    if len(unique_targets) != 2:
        labels = sorted(str(value) for value in unique_targets)
        raise ValueError(
            f"point_biserial requires a binary target, found "
            f"{len(unique_targets)} distinct classes: {labels}"
        )
    if positive_class not in unique_targets:
        labels = sorted(str(value) for value in unique_targets)
        raise ValueError(
            f"positive_class {positive_class!r} is not one of the observed "
            f"target classes {labels}; check for a typo or the wrong value"
        )

    n = len(feature_values)
    if n < _MIN_POINT_BISERIAL_OBSERVATIONS:
        raise ValueError(
            f"point_biserial needs at least "
            f"{_MIN_POINT_BISERIAL_OBSERVATIONS} observations to compute a "
            f"well-defined statistic, got {n}"
        )

    if np.all(feature_values == feature_values[0]):
        raise ValueError(
            "feature has zero variance (a constant value); point-biserial "
            "correlation is undefined"
        )

    is_positive = target_array == positive_class
    result = scipy.stats.pointbiserialr(is_positive, feature_values)

    positive_mean = float(feature_values[is_positive].mean())
    negative_mean = float(feature_values[~is_positive].mean())

    return PointBiserialResult(
        coefficient=float(result.statistic),
        p_value=float(result.pvalue),
        positive_n=int(is_positive.sum()),
        negative_n=int((~is_positive).sum()),
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        mean_difference=positive_mean - negative_mean,
    )


# Mirrors _MIN_POINT_BISERIAL_OBSERVATIONS: Spearman's p-value also needs
# degrees of freedom n - 2 >= 1 to be well-defined. Below this, scipy's
# spearmanr can return a NaN p-value without raising or warning at all.
_MIN_SPEARMAN_OBSERVATIONS = 3


@dataclass(frozen=True)
class SpearmanResult:
    """Spearman rank correlation between an ordinal feature and a binary target.

    Attributes:
        coefficient: Spearman's rho, in ``[-1, 1]``. Positive means higher
            feature ranks are associated with the positive class; negative
            means the opposite; near zero means no *monotonic* trend (see
            :func:`spearman_rho`'s docstring for what that specifically
            does and doesn't rule out).
        p_value: p-value for the null hypothesis that the true rank
            correlation is zero.
        n: Number of (feature, target) pairs used.
        n_levels: Number of distinct feature values (ranks/levels) observed
            among those pairs.
    """

    coefficient: float
    p_value: float
    n: int
    n_levels: int

    def __post_init__(self) -> None:
        if not -1.0 - 1e-9 <= self.coefficient <= 1.0 + 1e-9:
            raise ValueError(f"coefficient must be in [-1, 1], got {self.coefficient}")
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")
        if self.n < 1:
            raise ValueError(f"n must be positive, got {self.n}")
        if self.n_levels < 1:
            raise ValueError(f"n_levels must be positive, got {self.n_levels}")
        if self.n_levels > self.n:
            raise ValueError(f"n_levels ({self.n_levels}) cannot exceed n ({self.n})")


def spearman_rho(
    feature: npt.ArrayLike, target: npt.ArrayLike, *, positive_class: Hashable
) -> SpearmanResult:
    """Spearman rank correlation between an ordinal feature and a binary target.

    Spearman's rho is the Pearson correlation between the *ranks* of
    ``feature`` and the target (encoded 1 for ``positive_class`` and 0
    otherwise), rather than between their raw values. That distinction is
    exactly what makes it the right tool for ordinal data: it only uses the
    *order* of feature values, never their spacing.

    **Do not substitute a direct call to** :func:`point_biserial` **on
    integer level codes as a shortcut.** Pearson/point-biserial correlation
    on raw codes implicitly assumes those codes are equally spaced (that
    the gap between level 0 and 1 equals the gap between 1 and 2) — an
    assumption ordinal data explicitly does not license. Spearman's
    internal rank transform instead weights each tied level by how many
    observations share it, which is a materially different, and more
    appropriate, computation for unevenly-sized groups. Feed this function
    the level codes directly (e.g. ``.cat.codes`` from an ordered pandas
    ``Categorical``); it performs its own rank transform via
    :func:`scipy.stats.spearmanr`, including the tie-averaging that a
    naive Pearson-on-codes calculation would skip.

    Args:
        feature: A 1D array-like of ordinal level codes (or any real-valued
            ranks), one per row. Only the relative *order* of values
            matters, not their magnitude or spacing. Same type
            restrictions as :func:`point_biserial`'s ``feature`` (no
            boolean, datetime, timedelta, complex, or other non-real
            values — see that function's docstring for the full list).
        target: A 1D array-like of target labels, one per row, positionally
            aligned with ``feature`` (not by pandas index, if both are
            ``Series`` — see :func:`signapy.metrics.lift.categorical_lift`
            for what that distinction means in practice). Must have exactly
            two distinct values.
        positive_class: The target value encoded as 1 (the "positive" side
            of the correlation's sign). Reversing this reverses the sign of
            ``coefficient`` exactly, not its magnitude.

    Returns:
        A :class:`SpearmanResult` with the coefficient, its p-value, and
        how many observations/distinct levels went into it.

    Raises:
        ValueError: If ``feature`` or ``target`` is not one-dimensional, if
            ``feature`` is boolean (including boolean values stored with
            ``object`` dtype), datetime-, timedelta-, or complex-valued, or
            otherwise not real-valued numeric, if they have different
            lengths, if either is empty, if either contains missing values,
            if ``feature`` contains a non-finite value (``inf`` or
            ``-inf``), if ``target`` does not have exactly two distinct
            values, if ``positive_class`` is not one of them, if fewer than
            :data:`_MIN_SPEARMAN_OBSERVATIONS` total observations remain,
            or if ``feature`` has fewer than two distinct levels (a
            constant feature, for which rank correlation is undefined).

    One statistical caveat worth knowing: Spearman only detects *monotonic*
    trends. A feature whose middle level has the highest (or lowest)
    target rate — "medium" converting better than both "low" and "high" —
    can show a weak Spearman coefficient even though the feature clearly
    relates to the target; the per-level evidence (support, target rate,
    lift — see :func:`signapy.metrics.lift.categorical_lift`) is what
    surfaces that kind of non-monotonic pattern.

    Uses :func:`scipy.stats.spearmanr`, which handles tied ranks — which
    every feature value sharing an ordinal level will produce — by
    averaging, consistent with the standard definition of Spearman's rho.
    """
    feature_array = np.asarray(feature)
    target_array = np.asarray(target)

    if feature_array.ndim != 1:
        raise ValueError(
            f"feature must be one-dimensional, got shape {feature_array.shape}"
        )
    if target_array.ndim != 1:
        raise ValueError(
            f"target must be one-dimensional, got shape {target_array.shape}"
        )

    if len(feature_array) != len(target_array):
        raise ValueError(
            f"feature and target must have the same length, got "
            f"{len(feature_array)} and {len(target_array)}"
        )
    if len(feature_array) == 0:
        raise ValueError("feature and target must not be empty")

    if pd.isna(feature_array).any():
        raise ValueError(
            "spearman_rho does not accept missing feature values; clean "
            "or impute before calling"
        )
    if pd.isna(target_array).any():
        raise ValueError(
            "spearman_rho does not accept missing target values; clean "
            "or impute before calling"
        )

    _validate_real_numeric_feature(feature_array, caller_name="spearman_rho")

    feature_values = feature_array
    if not all(_is_finite_real(value) for value in feature_values):
        raise ValueError(
            "feature must not contain infinite values (inf or -inf); "
            "resolve or drop these rows before calling"
        )

    unique_targets = pd.unique(target_array)
    if len(unique_targets) != 2:
        labels = sorted(str(value) for value in unique_targets)
        raise ValueError(
            f"spearman_rho requires a binary target, found "
            f"{len(unique_targets)} distinct classes: {labels}"
        )
    if positive_class not in unique_targets:
        labels = sorted(str(value) for value in unique_targets)
        raise ValueError(
            f"positive_class {positive_class!r} is not one of the observed "
            f"target classes {labels}; check for a typo or the wrong value"
        )

    n = len(feature_values)
    if n < _MIN_SPEARMAN_OBSERVATIONS:
        raise ValueError(
            f"spearman_rho needs at least {_MIN_SPEARMAN_OBSERVATIONS} "
            f"observations to compute a well-defined statistic, got {n}"
        )

    unique_levels = np.unique(feature_values)
    if len(unique_levels) < 2:
        raise ValueError(
            "feature has fewer than 2 distinct levels (a constant feature); "
            "Spearman rank correlation is undefined"
        )

    is_positive = target_array == positive_class
    result = scipy.stats.spearmanr(feature_values, is_positive)

    return SpearmanResult(
        coefficient=float(result.statistic),
        p_value=float(result.pvalue),
        n=n,
        n_levels=len(unique_levels),
    )
