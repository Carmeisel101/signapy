"""Feature-level association: Cramér's V.

Pure computation over a contingency table. This module does not infer
types, choose a method, or know anything about :mod:`signapy.results`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
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
