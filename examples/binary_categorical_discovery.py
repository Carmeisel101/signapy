"""Run SignaPy's first discovery slice on a small synthetic dataset.

Creates a labeled DataFrame with two categorical features
(``acquisition_channel``, ``region``) and a binary target (``converted``),
runs ``signapy.discover``, and prints feature- and value-level evidence for
each feature.

Run with:

    python examples/binary_categorical_discovery.py
"""

from __future__ import annotations

import pandas as pd

import signapy
from signapy.results import FeatureResult


def build_dataset() -> pd.DataFrame:
    """A small synthetic acquisition dataset with two categorical features."""
    acquisition_channel = (
        ["referral"] * 20 + ["organic"] * 20 + ["paid"] * 20 + [None] * 2
    )
    region = ["west"] * 30 + ["east"] * 32
    converted = (
        # referral: 8/20 converted
        [True] * 8
        + [False] * 12
        # organic: 5/20 converted
        + [True] * 5
        + [False] * 15
        # paid: 2/20 converted
        + [True] * 2
        + [False] * 18
        # the 2 rows with a missing acquisition_channel
        + [True, False]
    )
    return pd.DataFrame(
        {
            "acquisition_channel": acquisition_channel,
            "region": region,
            "converted": converted,
        }
    )


def print_feature(result: FeatureResult) -> None:
    print(f"\n=== {result.feature} ===")
    print(
        f"cramers_v={result.effect_size:.4f}  "
        f"chi_square(statistic={result.details['statistic']:.4f}, "
        f"dof={result.details['dof']})  "
        f"p_value={result.p_value:.4g}"
    )
    print(f"n={result.n}  missing_rate={result.missing_rate:.4f}")
    print(
        f"{'value':<12}{'support':>8}{'target_count':>14}{'target_rate':>13}"
        f"{'baseline_rate':>15}{'lift':>8}"
    )
    for value_result in result.values:
        print(
            f"{value_result.value!s:<12}{value_result.support:>8}"
            f"{value_result.target_count:>14}{value_result.target_rate:>13.4f}"
            f"{value_result.baseline_rate:>15.4f}{value_result.lift:>8.4f}"
        )


def main() -> None:
    df = build_dataset()
    print("Input DataFrame (first 5 rows of 62):")
    print(df.head())

    report = signapy.discover(df, target="converted", positive_class=True)

    for feature_result in report.features:
        print_feature(feature_result)


if __name__ == "__main__":
    main()
