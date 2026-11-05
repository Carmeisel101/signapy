import numpy as np
import pandas as pd
import pytest

import signapy
from signapy.discovery.value import MAX_BINS, build_binned_value_results
from signapy.profiling import FeatureType


def rising_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.random(n) < 1 / (1 + np.exp(-2 * x))
    return pd.DataFrame({"x": x, "y": y})


def discover_x(df: pd.DataFrame):
    return signapy.discover(
        df,
        target="y",
        positive_class=True,
        feature_types={"x": FeatureType.CONTINUOUS},
    ).feature("x")


class TestBinnedValueResults:
    def test_returns_equal_count_bins_in_ascending_order(self):
        df = rising_df()
        values = build_binned_value_results(df["x"], df["y"], positive_class=True)
        assert len(values) == MAX_BINS
        assert sum(v.support for v in values) == len(df)
        assert {v.support for v in values} == {len(df) // MAX_BINS}
        lifts = [v.lift for v in values]
        assert lifts == sorted(lifts)  # relationship is monotone by construction

    def test_labels_are_intervals_first_bin_closed(self):
        df = rising_df()
        values = build_binned_value_results(df["x"], df["y"], positive_class=True)
        assert values[0].value.startswith("[")
        assert all(v.value.startswith("(") for v in values[1:])
        assert len({v.value for v in values}) == len(values)

    def test_lift_matches_manual_calculation(self):
        df = rising_df()
        values = build_binned_value_results(df["x"], df["y"], positive_class=True)
        top = values[-1]
        cutoff = df["x"].quantile(0.8)
        in_top = df[df["x"] > cutoff]
        assert top.support == len(in_top)
        assert top.target_rate == pytest.approx(in_top["y"].mean())
        assert top.baseline_rate == pytest.approx(df["y"].mean())

    def test_few_distinct_values_get_one_bin_each(self):
        x = [0] * 60 + [1] * 20 + [2] * 20
        y = [False, True] * 50
        values = build_binned_value_results(x, y, positive_class=True)
        assert [v.value for v in values] == ["0", "1", "2"]
        assert [v.support for v in values] == [60, 20, 20]

    def test_two_distinct_values_give_two_bins(self):
        x = [0] * 50 + [1] * 50
        y = [False, True] * 50
        values = build_binned_value_results(x, y, positive_class=True)
        assert [v.support for v in values] == [50, 50]

    def test_heavy_ties_retry_with_fewer_bins(self):
        # 6 distinct values, so no per-value bins; 5 quantile bins collapse.
        x = [0.0] * 40 + [1.0] * 20 + [2.0] * 20 + [3.0] * 10 + [4.0] * 5 + [5.0] * 5
        y = [False, True] * 50
        values = build_binned_value_results(x, y, positive_class=True)
        assert 2 <= len(values) < MAX_BINS
        assert sum(v.support for v in values) == 100

    @staticmethod
    def _bins(x):
        y = [False, True] * (len(x) // 2)
        values = build_binned_value_results(x, y, positive_class=True)
        return None if values is None else [(v.value, v.support) for v in values]

    def test_dominant_value_gets_its_own_bin(self):
        bins = self._bins([0.0] * 90 + [float(i) for i in range(1, 11)])
        assert bins[0] == ("0", 90)
        assert [support for _, support in bins[1:]] == [5, 5]

    def test_dominant_value_in_the_middle_splits_the_range_around_it(self):
        x = [0.0] * 50 + list(range(-20, 0)) + list(range(1, 31))
        bins = self._bins(x)
        labels = [label for label, _ in bins]
        assert labels[2] == "0"
        assert sum(support for _, support in bins) == len(x)
        assert all(label.startswith(("[", "(")) for label in labels if label != "0")
        # Intervals below the spike end at it, and those above start after it.
        assert all("-" in label for label in labels[:2])
        assert not any("-" in label for label in labels[3:])

    def test_dominant_value_at_the_top(self):
        bins = self._bins(list(range(1, 21)) + [100.0] * 60)
        assert bins[-1] == ("100", 60)
        assert sum(support for _, support in bins) == 80

    def test_tiny_remainder_is_one_bin_not_many(self):
        bins = self._bins([0.0] * 95 + [1.0, 2, 3, 4, 5, 6, 7])
        assert bins == [("0", 95), ("[1, 7]", 7)]

    def test_spike_not_used_when_quantiles_already_give_full_bins(self):
        x = list(range(100))
        assert len(self._bins(x)) == MAX_BINS

    def test_too_few_rows_returns_none(self):
        x = list(range(9))
        y = [False] * 5 + [True] * 4
        assert build_binned_value_results(x, y, positive_class=True) is None

    def test_too_few_distinct_values_returns_none(self):
        x = [1.0] * 50
        y = [False, True] * 25
        assert build_binned_value_results(x, y, positive_class=True) is None

    def test_close_edges_get_distinct_labels(self):
        x = np.linspace(1.0, 1.0001, 100)
        y = [False, True] * 50
        values = build_binned_value_results(x, y, positive_class=True)
        assert len({v.value for v in values}) == len(values)


class TestDiscoverIntegration:
    def test_values_populated_and_consistent_with_feature_n(self):
        result = discover_x(rising_df())
        assert result.values is not None
        assert sum(v.support for v in result.values) == result.n

    def test_missing_values_excluded_from_bins(self):
        df = rising_df()
        df.loc[df.index[:40], "x"] = np.nan
        result = discover_x(df)
        assert result.n == 160
        assert sum(v.support for v in result.values) == 160

    def test_report_round_trips_through_to_dict_and_plot(self):
        pytest.importorskip("matplotlib")
        report = signapy.discover(
            rising_df(),
            target="y",
            positive_class=True,
            feature_types={"x": FeatureType.CONTINUOUS},
        )
        data = report.to_dict()
        assert len(data["features"][0]["values"]) == MAX_BINS
        texts = [t.get_text() for t in report.plot().axes[0].texts]
        assert not any("no value-level breakdown" in t for t in texts)
