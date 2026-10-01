"""Run SignaPy's ordinal discovery slice.

Creates a small, domain-neutral DataFrame with one ordinal feature
(``severity``, declared as an *ordered* pandas ``Categorical`` — SignaPy
never infers an order), and a binary target (``outcome``), runs
``signapy.discover``, and prints the feature- and value-level evidence.

Run with:

    python examples/ordinal_binary_discovery.py
"""

from __future__ import annotations

import pandas as pd

import signapy
from signapy.profiling import FeatureType
from signapy.results import FeatureResult


def build_dataset() -> pd.DataFrame:
    """A small synthetic dataset with one ordinal feature, increasing with target."""
    # low: 2/10 positive, medium: 5/10, high: 8/10 -> a real monotonic trend.
    severity = pd.Categorical(
        ["low"] * 10 + ["medium"] * 10 + ["high"] * 10,
        categories=["low", "medium", "high"],
        ordered=True,
    )
    outcome = [
        *([False] * 8 + [True] * 2),
        *([False] * 5 + [True] * 5),
        *([False] * 2 + [True] * 8),
    ]
    return pd.DataFrame({"severity": severity, "outcome": outcome})


def print_ordinal(result: FeatureResult) -> None:
    print(f"\n=== {result.feature} (ordinal) ===")
    print(
        f"spearman_rho coefficient={result.effect_size:.4f}  "
        f"p_value={result.p_value:.4g}  n_levels={result.details['n_levels']}"
    )
    print(f"n={result.n}  missing_rate={result.missing_rate:.4f}")
    # Unlike categorical's "order of first appearance", ordinal values come
    # back in the feature's *declared* category order (here: low, medium,
    # high) regardless of which order the rows happened to appear in.
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
    # A caveat worth knowing (see docs/metrics.md): Spearman only detects
    # *monotonic* trends. If the middle level had the highest rate while
    # "low" and "high" were similar, spearman_rho could read as near zero
    # even though the per-level values below would still show a clear
    # relationship with the target -- that's exactly why the value-level
    # table is printed here too, not just the single coefficient.


def main() -> None:
    df = build_dataset()
    print("Input DataFrame (first 5 rows of 30):")
    print(df.head())

    report = signapy.discover(
        df,
        target="outcome",
        positive_class=True,
        feature_types={"severity": FeatureType.ORDINAL},
    )

    print_ordinal(report.feature("severity"))


if __name__ == "__main__":
    main()
