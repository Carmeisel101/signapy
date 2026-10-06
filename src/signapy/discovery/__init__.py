"""Discovery workflows: the user-facing layer of SignaPy.

This package answers discovery *questions* rather than exposing individual
statistical tests. It chooses an appropriate method from the feature and
target types, calls into :mod:`signapy.metrics`, and packages the evidence
into :mod:`signapy.results` models.

Modules:

- ``feature``: feature-level discovery for one categorical feature
  (:func:`~signapy.discovery.feature.analyze_categorical_feature`), one
  continuous feature
  (:func:`~signapy.discovery.feature.analyze_continuous_feature`), or one
  ordinal feature
  (:func:`~signapy.discovery.feature.analyze_ordinal_feature`)
- ``value``: value-level discovery, adapting
  :func:`~signapy.metrics.lift.categorical_lift` to
  :class:`~signapy.results.ValueResult`
  (:func:`~signapy.discovery.value.build_value_results`). Reused by both
  the categorical and ordinal analyzers (ordinal reorders the result to
  its declared category order); no continuous equivalent yet — continuous
  localization (binning) is deferred (see ``docs/architecture.md``).

This module's :func:`discover` is the top-level entry point, exported as
``signapy.discover``. It supports categorical, continuous, and ordinal
features against a binary target; :data:`_DISPATCH` is where support for
additional ``(FeatureType, TargetType)`` combinations will be added.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING

import pandas as pd

from signapy.discovery._frame import (
    is_polars_dataframe,
    is_polars_lazyframe,
    polars_to_pandas,
)
from signapy.discovery.feature import (
    analyze_categorical_feature,
    analyze_continuous_feature,
    analyze_ordinal_feature,
)
from signapy.profiling import FeatureType, TargetType
from signapy.results import DiscoveryReport, FeatureResult

__all__ = ["discover"]

if TYPE_CHECKING:
    import polars as pl

_FeatureAnalyzer = Callable[..., FeatureResult]

# Method dispatch: which analyzer handles a given (declared feature type,
# target type) combination. Adding a new supported combination (e.g. a
# continuous target, or boolean features) means adding an entry here and to
# _SUPPORTED_FEATURE_TYPES/_resolve_feature_type, not a longer if/elif chain.
_DISPATCH: dict[tuple[FeatureType, TargetType], _FeatureAnalyzer] = {
    (FeatureType.CATEGORICAL, TargetType.BINARY): analyze_categorical_feature,
    (FeatureType.CONTINUOUS, TargetType.BINARY): analyze_continuous_feature,
    (FeatureType.ORDINAL, TargetType.BINARY): analyze_ordinal_feature,
}

# Feature types accepted in the *public* feature_types argument. A stricter
# (and currently identical) set than _DISPATCH's keys would allow, kept
# separate so "not a supported FeatureType value here" and "no analyzer for
# this combination" can be reported as the distinct failures they are.
_SUPPORTED_FEATURE_TYPES = (
    FeatureType.CATEGORICAL,
    FeatureType.CONTINUOUS,
    FeatureType.ORDINAL,
)


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


def _check_ordinal_dtype(name: str, series: pd.Series) -> None:
    """Require an ordered pandas ``Categorical`` for a feature declared ORDINAL.

    SignaPy never infers an order (not from the values, and never
    alphabetically) — the order must be explicit:
    ``pd.Categorical(df[name], categories=[...], ordered=True)``.

    Raises two distinct, actionable errors rather than one generic one,
    since they're different mistakes with different fixes:

    Raises:
        ValueError: If ``series`` isn't a pandas ``category`` dtype at all
            (the column needs to be built as a ``Categorical`` first), or
            if it is one but ``ordered=False`` (the order declaration is
            missing, not the categorical-ness itself).
    """
    dtype = series.dtype
    if not isinstance(dtype, pd.CategoricalDtype):
        raise ValueError(
            f"feature {name!r} has unsupported dtype {dtype}; ordinal "
            "features must be an ordered pandas Categorical — construct it "
            f"with e.g. df[{name!r}] = pd.Categorical(df[{name!r}], "
            "categories=[...], ordered=True) before calling discover(). "
            "signapy never infers a category order from the values, and "
            "never assumes alphabetical order — see docs/architecture.md."
        )
    if not dtype.ordered:
        raise ValueError(
            f"feature {name!r} is an unordered categorical "
            f"(pd.Categorical(..., ordered=False)); ordinal features need "
            f"an explicit declared order — construct it with "
            f"pd.Categorical(df[{name!r}], categories=[...], ordered=True) "
            "rather than relying on signapy to infer one — see "
            "docs/architecture.md."
        )


def _resolve_feature_type(name: str, declared: FeatureType | str) -> FeatureType:
    """Validate and normalize one ``feature_types`` entry.

    Raises:
        ValueError: If ``declared`` is not a valid :class:`FeatureType`
            value, or is a valid one that this discovery slice does not yet
            support (anything other than
            ``CATEGORICAL``/``CONTINUOUS``/``ORDINAL``).
    """
    if isinstance(declared, FeatureType):
        feature_type = declared
    else:
        try:
            feature_type = FeatureType(declared)
        except ValueError as error:
            valid = [member.value for member in FeatureType]
            raise ValueError(
                f"feature_types[{name!r}] = {declared!r} is not a valid "
                f"FeatureType; valid values are {valid}"
            ) from error
    if feature_type not in _SUPPORTED_FEATURE_TYPES:
        supported = [member.value for member in _SUPPORTED_FEATURE_TYPES]
        raise ValueError(
            f"feature_types[{name!r}] = {feature_type.value!r} is not yet "
            f"supported by signapy.discover; supported feature types are "
            f"{supported}"
        )
    return feature_type


def discover(
    df: pd.DataFrame | pl.DataFrame,
    target: str,
    *,
    positive_class: Hashable,
    feature_types: Mapping[str, FeatureType | str] | None = None,
) -> DiscoveryReport:
    """Discover feature/value evidence against a binary target.

    Computes feature-level evidence for each selected feature — and, for
    categorical and ordinal features, value-level evidence too — and returns
    them as a :class:`~signapy.results.DiscoveryReport`.

    - **Categorical** features get bias-corrected Cramér's V, a chi-square
      test, and per-category support/target rate/baseline rate/lift, in
      order of first appearance (``FeatureResult.values`` populated).
    - **Continuous** features get a point-biserial correlation and its
      p-value, plus group counts and means in ``details``
      (``FeatureResult.values`` is always ``None`` — continuous
      localization is deferred, see ``docs/architecture.md``).
    - **Ordinal** features get a Spearman rank correlation and its p-value,
      plus the number of distinct levels in ``details``, and per-level
      support/target rate/baseline rate/lift like categorical — but in the
      feature's *declared* category order, not order of first appearance.
      Ordinal features must be an ordered pandas ``Categorical``
      (``pd.Categorical(..., ordered=True)``); SignaPy never infers an
      order, from the values or alphabetically.

    This is SignaPy's discovery layer for binary targets only. ``df`` is
    never mutated. Polars input gives the same results as the equivalent
    pandas input; a Polars ``Enum`` column maps to an ordered ``Categorical``
    (usable as ORDINAL), and Polars ``NaN`` is treated as missing like null.

    Args:
        df: The labeled dataset: a pandas or Polars ``DataFrame``. Polars
            input is read without mutation and without requiring PyArrow;
            see :mod:`signapy.discovery._frame` for the dtype mapping.
        target: Name of the binary target column in ``df``.
        positive_class: The target value treated as the positive class.
        feature_types: Optional mapping from column name to
            :class:`~signapy.profiling.FeatureType` (or its string value,
            e.g. ``"categorical"``), selecting exactly which columns to
            analyze and how. When given:

            - Its keys are the *only* features analyzed; other columns of
              ``df`` are ignored.
            - ``target`` must not be one of the keys.
            - Every key must be a column of ``df``.
            - Only ``FeatureType.CATEGORICAL``, ``FeatureType.CONTINUOUS``,
              and ``FeatureType.ORDINAL`` are supported in this release.

            When omitted (``None``, the default), every non-target column
            of ``df`` is analyzed as categorical — this preserves the
            behavior from before ``feature_types`` existed, including
            raising if a column's dtype isn't a supported categorical dtype.
            Numeric columns are never silently treated as continuous, and
            categorical columns are never silently treated as ordinal:
            declaring them via ``feature_types`` is required, because e.g.
            an integer column might be a measured quantity, an ordinal
            level, or a category identifier, and SignaPy does not guess
            which.

    Returns:
        A :class:`~signapy.results.DiscoveryReport` with one
        :class:`~signapy.results.FeatureResult` per selected feature, in
        the same order as ``df.columns`` (restricted to the selected
        features when ``feature_types`` is given).

    Raises:
        ValueError: If ``target`` is not a column of ``df``; if ``df`` is
            empty; if ``target``'s non-missing values are not exactly two
            distinct classes; if ``positive_class`` is not one of them; if
            ``feature_types`` is an empty mapping, includes ``target``, or
            names a column not in ``df``; if a declared feature type is
            invalid or not yet supported (see :func:`_resolve_feature_type`);
            if a categorical column's dtype is unsupported (see
            :func:`_is_supported_categorical_dtype`); if an ordinal column
            isn't an ordered ``Categorical`` (see :func:`_check_ordinal_dtype`);
            or if a feature's own data is invalid for its declared type once
            its missing values are dropped (see
            :func:`signapy.discovery.feature.analyze_categorical_feature`,
            :func:`signapy.discovery.feature.analyze_continuous_feature`,
            and :func:`signapy.discovery.feature.analyze_ordinal_feature`).

    Missing-data policy (see ``docs/architecture.md`` for the full
    rationale), the same for every feature type:

    1. Rows with a missing ``target`` are excluded from all analysis.
    2. For each feature, rows with a missing value for *that* feature are
       additionally excluded from *that feature's* analysis only; other
       features are unaffected.
    3. ``FeatureResult.n`` is the number of rows with both a non-missing
       target and a non-missing value for that feature.
    4. ``FeatureResult.missing_rate`` is the fraction of target-valid rows
       where that feature is missing.
    5. For categorical and ordinal features, value-level baseline rate and
       lift are computed from the same feature-valid rows used for that
       feature's feature-level statistic.
    6. Infinite values are invalid, not missing — a continuous feature
       containing ``inf``/``-inf`` raises rather than having those rows
       silently dropped.
    """
    if is_polars_lazyframe(df):
        raise TypeError("df is a Polars LazyFrame; call .collect() first")
    if is_polars_dataframe(df):
        columns = list(df.columns)
        if target in columns:
            wanted = columns if feature_types is None else {target, *feature_types}
            df = polars_to_pandas(df, wanted)
    elif not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas or Polars DataFrame, got {type(df).__name__}"
        )
    else:
        columns = list(df.columns)

    if target not in columns:
        raise ValueError(
            f"target column {target!r} not found in DataFrame columns: {columns}"
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

    if feature_types is None:
        # Backward-compatible default: every non-target column, analyzed as
        # categorical. Equivalent to declaring every column CATEGORICAL
        # explicitly, so it goes through the same resolved_types/dispatch
        # path below rather than a separate code path to keep in sync.
        resolved_types: dict[str, FeatureType] = {
            column: FeatureType.CATEGORICAL for column in df.columns if column != target
        }
    else:
        if len(feature_types) == 0:
            raise ValueError("feature_types must not be empty")
        if target in feature_types:
            raise ValueError(
                f"feature_types must not include the target column {target!r}"
            )
        for name in feature_types:
            if name not in columns:
                raise ValueError(
                    f"feature_types names {name!r}, which is not a column "
                    f"of df: {columns}"
                )
        resolved_types = {
            name: _resolve_feature_type(name, declared)
            for name, declared in feature_types.items()
        }

    feature_columns = [column for column in df.columns if column in resolved_types]
    results = []
    for name in feature_columns:
        feature_type = resolved_types[name]
        series = target_valid_df[name]

        analyzer = _DISPATCH.get((feature_type, TargetType.BINARY))
        if analyzer is None:
            # Currently unreachable: _resolve_feature_type only allows
            # values in _SUPPORTED_FEATURE_TYPES, which are exactly
            # _DISPATCH's keys for TargetType.BINARY (the only target type
            # this layer supports). Kept as an explicit, named failure
            # rather than a silent KeyError so it stays correct if either
            # set changes independently in the future.
            supported = sorted(f"{ft.value}+{tt.value}" for ft, tt in _DISPATCH)
            raise ValueError(
                f"feature {name!r} declared as {feature_type.value!r} "
                f"against a {TargetType.BINARY.value!r} target has no "
                f"discovery method; supported combinations are {supported}"
            )

        if feature_type is FeatureType.CATEGORICAL and not (
            _is_supported_categorical_dtype(series)
        ):
            raise ValueError(
                f"feature {name!r} has unsupported dtype {series.dtype}; "
                "signapy.discover only supports categorical columns "
                '(object, pandas "string", or pandas "category" dtype) in '
                "this release. Continuous and ordinal features must be "
                "declared via feature_types; boolean and integer-coded-"
                "category features are not yet supported — see "
                "docs/architecture.md."
            )
        if feature_type is FeatureType.ORDINAL:
            _check_ordinal_dtype(name, series)

        results.append(
            analyzer(
                name,
                series,
                target_valid_df[target],
                positive_class=positive_class,
            )
        )

    return DiscoveryReport(target=target, features=tuple(results))
