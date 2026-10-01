"""Feature-level discovery for one categorical, continuous, or ordinal
feature against a binary target.

This module owns SignaPy's per-feature missing-data policy (dropping rows
where *this* feature is missing, and reporting the resulting rate) and
feature-specific minimum-observation checks. The statistics themselves come
from :mod:`signapy.metrics`; this module only prepares the feature-valid
data, calls them, and assembles a :class:`~signapy.results.FeatureResult`.
"""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from signapy.discovery.value import build_value_results
from signapy.metrics.association import cramers_v, point_biserial, spearman_rho
from signapy.metrics.significance import chi_square
from signapy.profiling import FeatureType, TargetType
from signapy.results import FeatureResult


def analyze_categorical_feature(
    name: str,
    feature: pd.Series,
    target: pd.Series,
    *,
    positive_class: Hashable,
) -> FeatureResult:
    """Feature- and value-level discovery for one categorical feature.

    Applies SignaPy's per-feature missing-data policy: rows where ``feature``
    is missing are excluded from this feature's analysis (they are not
    dropped from other features). ``target`` must already be restricted to
    rows with a non-missing target value — that policy is target-wide and is
    the caller's responsibility (see :func:`signapy.discovery.discover`).

    Args:
        name: Column name, used as ``FeatureResult.feature`` and in error
            messages.
        feature: The feature column, restricted to rows with a non-missing
            target but not yet to non-missing feature values.
        target: The binary target column, positionally aligned with
            ``feature`` and restricted to the same non-missing-target rows,
            with exactly two distinct values.
        positive_class: The target value treated as the positive class.

    Returns:
        A :class:`~signapy.results.FeatureResult` with bias-corrected
        Cramér's V (``effect_size_method="cramers_v"``), a chi-square test
        (``test_method="chi_square"``, ``details={"statistic": ...,
        "dof": ...}``), and per-category
        :class:`~signapy.results.ValueResult` objects — all computed from
        the same feature-valid rows.

    Raises:
        ValueError: If fewer than two feature categories remain after
            dropping this feature's missing values, or if the resulting
            feature/target contingency table is otherwise invalid (e.g. one
            target class happens to disappear once rows with a missing
            feature value are dropped).
    """
    # Positional arrays throughout: feature/target may carry pandas indexes
    # that (after upstream filtering) happen to already agree, but working
    # positionally avoids relying on that rather than risking a silent
    # label-based misalignment (see signapy.metrics.lift.categorical_lift).
    feature_values = pd.Series(feature).to_numpy()
    target_values = pd.Series(target).to_numpy()

    n_target_valid = len(feature_values)
    missing_mask = pd.isna(feature_values)
    missing_rate = float(missing_mask.mean()) if n_target_valid else 0.0

    valid_mask = ~missing_mask
    feature_valid = feature_values[valid_mask]
    target_valid = target_values[valid_mask]
    n = int(valid_mask.sum())

    unique_categories = pd.unique(feature_valid)
    if len(unique_categories) < 2:
        raise ValueError(
            f"feature {name!r} has fewer than 2 observed categories after "
            f"dropping missing values (found {len(unique_categories)}); "
            "Cramér's V and chi-square require at least 2 categories"
        )

    table = pd.crosstab(feature_valid, target_valid)
    try:
        effect_size = cramers_v(table)
        chi_result = chi_square(table)
    except ValueError as error:
        raise ValueError(
            f"could not compute association for feature {name!r}: {error}"
        ) from error

    # categorical_lift (via build_value_results) re-validates missing values
    # and target arity on feature_valid/target_valid; both are already
    # guaranteed clean by the filtering above, so this is just consistent
    # reuse of the same low-level function, not redundant policy.
    values = build_value_results(
        feature_valid, target_valid, positive_class=positive_class
    )

    return FeatureResult(
        feature=name,
        feature_type=FeatureType.CATEGORICAL,
        target_type=TargetType.BINARY,
        effect_size_method="cramers_v",
        effect_size=effect_size,
        n=n,
        missing_rate=missing_rate,
        test_method="chi_square",
        p_value=chi_result.p_value,
        values=values,
        details={"statistic": chi_result.statistic, "dof": chi_result.dof},
    )


def analyze_continuous_feature(
    name: str,
    feature: pd.Series,
    target: pd.Series,
    *,
    positive_class: Hashable,
) -> FeatureResult:
    """Feature-level discovery for one continuous feature.

    Applies the same per-feature missing-data policy as
    :func:`analyze_categorical_feature`: rows where ``feature`` is missing
    are excluded from this feature's analysis only. ``target`` must already
    be restricted to rows with a non-missing target value.

    Unlike the categorical case, this produces no value-level
    ``ValueResult``s — continuous localization (binning) is deferred to a
    future feature (see ``docs/architecture.md``) — so
    ``FeatureResult.values`` is always ``None`` here.

    Args:
        name: Column name, used as ``FeatureResult.feature`` and in error
            messages.
        feature: The feature column, restricted to rows with a non-missing
            target but not yet to non-missing feature values.
        target: The binary target column, positionally aligned with
            ``feature`` and restricted to the same non-missing-target rows,
            with exactly two distinct values.
        positive_class: The target value treated as the positive class.

    Returns:
        A :class:`~signapy.results.FeatureResult` with the point-biserial
        coefficient (``effect_size_method="point_biserial"``) and its
        p-value (``test_method="point_biserial"``), and group counts/means
        in ``details`` (``positive_n``, ``negative_n``, ``positive_mean``,
        ``negative_mean``, ``mean_difference``).

    Raises:
        ValueError: Anything :func:`~signapy.metrics.association.point_biserial`
            raises for the feature-valid data, with the feature name added
            for context. See that function's docstring for the authoritative
            validation contract.
    """
    feature_values = pd.Series(feature).to_numpy()
    target_values = pd.Series(target).to_numpy()

    n_target_valid = len(feature_values)
    missing_mask = pd.isna(feature_values)
    missing_rate = float(missing_mask.mean()) if n_target_valid else 0.0

    valid_mask = ~missing_mask
    feature_valid = feature_values[valid_mask]
    target_valid = target_values[valid_mask]
    n = int(valid_mask.sum())

    try:
        result = point_biserial(
            feature_valid, target_valid, positive_class=positive_class
        )
    except ValueError as error:
        raise ValueError(
            f"could not compute association for feature {name!r}: {error}"
        ) from error

    return FeatureResult(
        feature=name,
        feature_type=FeatureType.CONTINUOUS,
        target_type=TargetType.BINARY,
        effect_size_method="point_biserial",
        effect_size=result.coefficient,
        n=n,
        missing_rate=missing_rate,
        test_method="point_biserial",
        p_value=result.p_value,
        values=None,
        details={
            "positive_n": result.positive_n,
            "negative_n": result.negative_n,
            "positive_mean": result.positive_mean,
            "negative_mean": result.negative_mean,
            "mean_difference": result.mean_difference,
        },
    )


def analyze_ordinal_feature(
    name: str,
    feature: pd.Series,
    target: pd.Series,
    *,
    positive_class: Hashable,
) -> FeatureResult:
    """Feature- and value-level discovery for one ordinal feature.

    Applies the same per-feature missing-data policy as the categorical and
    continuous cases: rows where ``feature`` is missing are excluded from
    this feature's analysis only. ``target`` must already be restricted to
    rows with a non-missing target value.

    Unlike :func:`analyze_categorical_feature`/:func:`analyze_continuous_feature`,
    this requires ``feature`` to already be an ordered pandas ``Categorical``
    Series (``pd.CategoricalDtype(ordered=True)``) — the caller (see
    :func:`signapy.discovery.discover`) is responsible for rejecting
    anything else before calling this function, since validating that is a
    DataFrame-level, not per-feature, concern.

    Args:
        name: Column name, used as ``FeatureResult.feature`` and in error
            messages.
        feature: The feature column, an ordered ``Categorical`` Series,
            restricted to rows with a non-missing target but not yet to
            non-missing feature values.
        target: The binary target column, positionally aligned with
            ``feature`` and restricted to the same non-missing-target rows,
            with exactly two distinct values.
        positive_class: The target value treated as the positive class.

    Returns:
        A :class:`~signapy.results.FeatureResult` with a Spearman rank
        correlation (``effect_size_method="spearman_rho"``,
        ``test_method="spearman"``, ``details={"n_levels": ...}``), and
        per-level :class:`~signapy.results.ValueResult` objects — in the
        feature's *declared* category order (``feature.cat.categories``),
        not order of first appearance. A declared category with zero
        observations in the feature-valid data is simply absent from
        ``values``, not included with zero-filled fields.

    Raises:
        ValueError: Anything :func:`~signapy.metrics.association.spearman_rho`
            raises for the feature-valid rank codes (fewer than two
            distinct levels, too few observations, a target that is no
            longer binary once this feature's missing values are dropped,
            etc.), with the feature name added for context.
    """
    # Unlike the categorical/continuous analyzers, this keeps `feature` as
    # a pandas Series rather than converting to a bare array right away:
    # the declared category order and the integer rank codes both come
    # from its .cat accessor, which a plain numpy array doesn't carry.
    feature_series = pd.Series(feature)
    target_values = pd.Series(target).to_numpy()

    category_order = list(feature_series.cat.categories)

    n_target_valid = len(feature_series)
    missing_mask = feature_series.isna().to_numpy()
    missing_rate = float(missing_mask.mean()) if n_target_valid else 0.0

    valid_mask = ~missing_mask
    feature_valid_series = feature_series[valid_mask]
    target_valid = target_values[valid_mask]
    n = int(valid_mask.sum())

    codes = feature_valid_series.cat.codes.to_numpy()

    try:
        result = spearman_rho(codes, target_valid, positive_class=positive_class)
    except ValueError as error:
        raise ValueError(
            f"could not compute association for feature {name!r}: {error}"
        ) from error

    # build_value_results (via categorical_lift) returns one ValueResult
    # per observed level, in order of first appearance; reorder to the
    # feature's declared category order instead. A declared category with
    # no observations in this feature-valid subset has no ValueResult to
    # reorder — categorical_lift never invents a zero-support entry — so it
    # is simply absent from `values`, not zero-filled.
    labels = feature_valid_series.to_numpy()
    unordered_values = build_value_results(
        labels, target_valid, positive_class=positive_class
    )
    values_by_label = {
        value_result.value: value_result for value_result in unordered_values
    }
    values = tuple(
        values_by_label[category]
        for category in category_order
        if category in values_by_label
    )

    return FeatureResult(
        feature=name,
        feature_type=FeatureType.ORDINAL,
        target_type=TargetType.BINARY,
        effect_size_method="spearman_rho",
        effect_size=result.coefficient,
        n=n,
        missing_rate=missing_rate,
        test_method="spearman",
        p_value=result.p_value,
        values=values,
        details={"n_levels": result.n_levels},
    )
