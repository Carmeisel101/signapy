"""Feature-level significance testing: the chi-square test of independence.

Pure computation over a contingency table. This module does not infer
types, choose a method, or know anything about :mod:`signapy.results`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.stats


def _validate_contingency_table(contingency_table: npt.ArrayLike) -> np.ndarray:
    """Convert and validate a 2D contingency table of non-negative counts.

    Duplicated from :mod:`signapy.metrics.association` rather than shared,
    so each metrics module has no internal dependencies on the others.

    Raises:
        ValueError: If the table is not 2D, is empty, has fewer than two
            rows or columns, contains negative, non-finite, or fractional
            values, or has a row/column that sums to zero (making expected
            frequencies undefined).

    Values must be integer-valued counts, for consistency with
    :func:`signapy.metrics.association.cramers_v`, whose bias correction
    treats the table sum as a literal observation count.
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
            f"classes to compute a test statistic, got shape {table.shape}"
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


@dataclass(frozen=True)
class ChiSquareResult:
    """Result of a chi-square test of independence.

    Attributes:
        statistic: The chi-square test statistic.
        p_value: The p-value for the test.
        dof: Degrees of freedom, ``(rows - 1) * (cols - 1)``.
    """

    statistic: float
    p_value: float
    dof: int


def chi_square(contingency_table: npt.ArrayLike) -> ChiSquareResult:
    """Chi-square test of independence for a contingency table.

    Thin wrapper over :func:`scipy.stats.chi2_contingency`, called with
    ``correction=False`` (no Yates' continuity correction), consistent
    with :func:`signapy.metrics.association.cramers_v`.

    Args:
        contingency_table: A 2D array-like of non-negative observed counts
            (rows = feature categories, columns = target classes). Accepts
            a :class:`numpy.ndarray`, nested sequence, or
            :class:`pandas.DataFrame`.

    Returns:
        A :class:`ChiSquareResult` with the statistic, p-value, and degrees
        of freedom.

    Raises:
        ValueError: If ``contingency_table`` is invalid (see
            :func:`_validate_contingency_table`).
    """
    table = _validate_contingency_table(contingency_table)
    statistic, p_value, dof, _expected = scipy.stats.chi2_contingency(
        table, correction=False
    )
    return ChiSquareResult(
        statistic=float(statistic), p_value=float(p_value), dof=int(dof)
    )
