"""Run SignaPy's mixed categorical/continuous discovery slice.

Creates a small, domain-neutral DataFrame with one categorical feature
(``category_context``), one continuous feature
(``continuous_measurement``), and a binary target (``outcome``), declares
their semantic types explicitly via ``feature_types``, runs
``signapy.discover``, and prints both features' evidence.

Run with:

    python examples/mixed_categorical_continuous_discovery.py
"""

from __future__ import annotations

import pandas as pd

import signapy
from signapy.profiling import FeatureType
from signapy.results import FeatureResult


def build_dataset() -> pd.DataFrame:
    """A small synthetic dataset with one categorical and one continuous feature."""
    category_context = ["low"] * 10 + ["medium"] * 10 + ["high"] * 10

    # "low" group: centered around 10, one missing value
    low_measurements = [9.5, 10.2, 8.8, 10.9, 9.1, 10.4, 9.8, None, 11.0, 9.3]
    # "medium" group: centered around 20
    medium_measurements = [19.6, 20.5, 18.9, 21.1, 19.4, 20.8, 19.9, 20.2, 18.6, 21.4]
    # "high" group: centered around 30
    high_measurements = [29.4, 30.6, 28.8, 31.2, 29.9, 30.1, 28.5, 31.5, 29.7, 30.3]
    continuous_measurement = [
        *low_measurements,
        *medium_measurements,
        *high_measurements,
    ]

    # "low": mostly negative, "medium": roughly even, "high": mostly positive
    outcome = [
        *([False] * 8 + [True] * 2),
        *([False] * 5 + [True] * 5),
        *([False] * 2 + [True] * 8),
    ]
    return pd.DataFrame(
        {
            "category_context": category_context,
            "continuous_measurement": continuous_measurement,
            "outcome": outcome,
        }
    )


def print_categorical(result: FeatureResult) -> None:
    print(f"\n=== {result.feature} (categorical) ===")
    print(
        f"cramers_v={result.effect_size:.4f}  "
        f"chi_square(statistic={result.details['statistic']:.4f}, "
        f"dof={result.details['dof']})  "
        f"p_value={result.p_value:.4g}"
    )
    print(f"n={result.n}  missing_rate={result.missing_rate:.4f}")
    # Categorical features get value-level localization: where within the
    # feature does the association live?
    print(
        f"{'value':<10}{'support':>8}{'target_count':>14}{'target_rate':>13}"
        f"{'baseline_rate':>15}{'lift':>8}"
    )
    for value_result in result.values:
        print(
            f"{value_result.value!s:<10}{value_result.support:>8}"
            f"{value_result.target_count:>14}{value_result.target_rate:>13.4f}"
            f"{value_result.baseline_rate:>15.4f}{value_result.lift:>8.4f}"
        )


def print_continuous(result: FeatureResult) -> None:
    print(f"\n=== {result.feature} (continuous) ===")
    print(
        f"point_biserial coefficient={result.effect_size:.4f}  "
        f"p_value={result.p_value:.4g}"
    )
    print(f"n={result.n}  missing_rate={result.missing_rate:.4f}")
    details = result.details
    print(
        f"positive: n={details['positive_n']}, mean={details['positive_mean']:.4f}  "
        f"negative: n={details['negative_n']}, mean={details['negative_mean']:.4f}  "
        f"mean_difference={details['mean_difference']:.4f}"
    )
    # Unlike the categorical feature above, result.values is None: continuous
    # localization (which *range* of values carries the signal) requires
    # binning, and SignaPy deliberately doesn't do that automatically yet
    # (see docs/architecture.md, "Continuous localization"). The
    # point-biserial coefficient and group means are feature-level evidence
    # only; there is currently no continuous equivalent of categorical_lift.
    print(f"values={result.values!r}  (continuous localization is not yet implemented)")


def main() -> None:
    df = build_dataset()
    print("Input DataFrame (first 5 rows of 30):")
    print(df.head())

    report = signapy.discover(
        df,
        target="outcome",
        positive_class=True,
        feature_types={
            "category_context": FeatureType.CATEGORICAL,
            "continuous_measurement": FeatureType.CONTINUOUS,
        },
    )

    print_categorical(report.feature("category_context"))
    print_continuous(report.feature("continuous_measurement"))


if __name__ == "__main__":
    main()
