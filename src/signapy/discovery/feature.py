"""Feature-level discovery for one categorical feature against a binary target.

This module owns SignaPy's per-feature missing-data policy (dropping rows
where *this* feature is missing, and reporting the resulting rate) and the
minimum-categories check. The statistics themselves come from
:mod:`signapy.metrics`; this module only builds the contingency table,
calls them, and assembles a :class:`~signapy.results.FeatureResult`.
"""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from signapy.discovery.value import build_value_results
from signapy.metrics.association import cramers_v
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
