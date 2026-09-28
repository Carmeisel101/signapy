"""Discovery workflows: the user-facing layer of SignaPy.

This package answers discovery *questions* rather than exposing individual
statistical tests. It chooses an appropriate method from the feature and
target types, calls into :mod:`signapy.metrics`, and packages the evidence
into :mod:`signapy.results` models.

Modules:

- ``feature``: feature-level discovery for one categorical feature
  (:func:`~signapy.discovery.feature.analyze_categorical_feature`)
- ``value``: value-level discovery, adapting
  :func:`~signapy.metrics.lift.categorical_lift` to
  :class:`~signapy.results.ValueResult`
  (:func:`~signapy.discovery.value.build_value_results`)

This module's :func:`discover` is the top-level entry point, exported as
``signapy.discover``.
"""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from signapy.discovery.feature import analyze_categorical_feature
from signapy.results import DiscoveryReport

__all__ = ["discover"]


def _is_supported_categorical_dtype(series: pd.Series) -> bool:
    """Whether ``series`` has a dtype this discovery slice treats as categorical.

    Supported: ``object``, pandas ``category``, and pandas ``string``
    dtypes. Numeric dtypes (including integer-coded categories) and
    ``bool`` are deliberately unsupported: this slice does not infer
    semantic type from values, so a numeric or boolean column is never
    silently reinterpreted as categorical (see ``docs/architecture.md``).
    """
    dtype = series.dtype
    return pd.api.types.is_object_dtype(dtype) or isinstance(
        dtype, pd.CategoricalDtype | pd.StringDtype
    )


def discover(
    df: pd.DataFrame,
    target: str,
    *,
    positive_class: Hashable,
) -> DiscoveryReport:
    """Discover categorical feature/value evidence against a binary target.

    For every column in ``df`` other than ``target``, with a supported
    categorical dtype, this computes feature-level evidence (bias-corrected
    Cramér's V and a chi-square test) and value-level evidence (per-category
    support, target rate, baseline rate, and lift), and returns them as a
    :class:`~signapy.results.DiscoveryReport`.

    This is SignaPy's first discovery slice: binary targets and categorical
    features only. ``df`` is never mutated.

    Args:
        df: The labeled dataset.
        target: Name of the binary target column in ``df``.
        positive_class: The target value treated as the positive class.

    Returns:
        A :class:`~signapy.results.DiscoveryReport` with one
        :class:`~signapy.results.FeatureResult` per supported feature
        column, in the same order as ``df.columns`` (excluding ``target``).

    Raises:
        ValueError: If ``target`` is not a column of ``df``; if ``df`` is
            empty; if ``target``'s non-missing values are not exactly two
            distinct classes; if ``positive_class`` is not one of them; if
            any other column has an unsupported dtype (see
            :func:`_is_supported_categorical_dtype`); or if a feature has
            fewer than two observed categories once its own missing values
            are dropped (see
            :func:`signapy.discovery.feature.analyze_categorical_feature`).

    Missing-data policy (see ``docs/architecture.md`` for the full
    rationale):

    1. Rows with a missing ``target`` are excluded from all analysis.
    2. For each feature, rows with a missing value for *that* feature are
       additionally excluded from *that feature's* analysis only; other
       features are unaffected.
    3. ``FeatureResult.n`` is the number of rows with both a non-missing
       target and a non-missing value for that feature.
    4. ``FeatureResult.missing_rate`` is the fraction of target-valid rows
       where that feature is missing.
    5. Value-level baseline rate and lift are computed from the same
       feature-valid rows used for that feature's Cramér's V and
       chi-square.

    Categorical feature policy: only ``object``, pandas ``string``, and
    pandas ``category`` dtype columns are treated as categorical features.
    Numeric columns (including integer-coded categories) and boolean
    columns are rejected with an explicit error rather than silently
    reinterpreted — inferring semantic type from values is deliberately
    out of scope for this slice.
    """
    if target not in df.columns:
        raise ValueError(
            f"target column {target!r} not found in DataFrame columns: "
            f"{list(df.columns)}"
        )
    if df.empty:
        raise ValueError("df must not be empty")

    target_valid_mask = df[target].notna()
    target_valid_df = df.loc[target_valid_mask]

    unique_targets = pd.unique(target_valid_df[target])
    if len(unique_targets) != 2:
        labels = sorted(str(value) for value in unique_targets)
        raise ValueError(
            f"target column {target!r} must be binary, found "
            f"{len(unique_targets)} distinct non-missing classes: {labels}"
        )
    if positive_class not in unique_targets:
        labels = sorted(str(value) for value in unique_targets)
        raise ValueError(
            f"positive_class {positive_class!r} is not one of the observed "
            f"target classes {labels}; check for a typo or the wrong value"
        )

    feature_columns = [column for column in df.columns if column != target]
    results = []
    for name in feature_columns:
        series = target_valid_df[name]
        if not _is_supported_categorical_dtype(series):
            raise ValueError(
                f"feature {name!r} has unsupported dtype {series.dtype}; "
                "signapy.discover only supports categorical columns "
                '(object, pandas "string", or pandas "category" dtype) in '
                "this release. Continuous, ordinal, boolean, and "
                "integer-coded-category features are not yet supported — "
                "see docs/architecture.md."
            )
        results.append(
            analyze_categorical_feature(
                name,
                series,
                target_valid_df[target],
                positive_class=positive_class,
            )
        )

    return DiscoveryReport(target=target, features=tuple(results))
