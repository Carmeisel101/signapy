import pandas as pd
import pytest

import signapy
from signapy.metrics.association import cramers_v
from signapy.metrics.lift import categorical_lift
from signapy.metrics.significance import chi_square
from signapy.profiling import FeatureType, TargetType
from signapy.results import DiscoveryReport, FeatureResult, ValueResult

# Same distribution as tests/test_association.py, tests/test_significance.py,
# and tests/test_lift.py's two-category table:
#              negative  positive
# referral          70        30
# paid              90        10
CHANNEL = ["referral"] * 100 + ["paid"] * 100
CONVERTED = [True] * 30 + [False] * 70 + [True] * 10 + [False] * 90


def basic_df() -> pd.DataFrame:
    return pd.DataFrame({"acquisition_channel": CHANNEL, "converted": CONVERTED})


def values_by_value(feature_result: FeatureResult) -> dict:
    return {v.value: v for v in feature_result.values}


class TestDiscoverKnownValues:
    def test_matches_known_two_category_metrics(self):
        report = signapy.discover(basic_df(), target="converted", positive_class=True)
        result = report.feature("acquisition_channel")

        assert result.effect_size_method == "cramers_v"
        assert result.effect_size == pytest.approx(0.24034381938205007)
        assert result.test_method == "chi_square"
        assert result.p_value == pytest.approx(0.00040695201744495935)
        assert result.details["statistic"] == pytest.approx(12.5)
        assert result.details["dof"] == 1
        assert result.n == 200
        assert result.missing_rate == pytest.approx(0.0)
        assert result.feature_type is FeatureType.CATEGORICAL
        assert result.target_type is TargetType.BINARY

    def test_value_results(self):
        report = signapy.discover(basic_df(), target="converted", positive_class=True)
        by_value = values_by_value(report.feature("acquisition_channel"))

        assert by_value["referral"] == ValueResult(
            value="referral",
            support=100,
            target_count=30,
            target_rate=0.3,
            baseline_rate=0.2,
            lift=pytest.approx(1.5),
        )
        assert by_value["paid"] == ValueResult(
            value="paid",
            support=100,
            target_count=10,
            target_rate=0.1,
            baseline_rate=0.2,
            lift=pytest.approx(0.5),
        )

    def test_report_and_feature_result_types(self):
        report = signapy.discover(basic_df(), target="converted", positive_class=True)
        assert isinstance(report, DiscoveryReport)
        assert report.target == "converted"
        assert isinstance(report.feature("acquisition_channel"), FeatureResult)


class TestDiscoverMultipleFeaturesAndOrdering:
    def test_column_order_is_preserved(self):
        df = pd.DataFrame(
            {
                "region": ["west"] * 80 + ["east"] * 120,
                "converted": CONVERTED[:80] + CONVERTED[80:],
                "acquisition_channel": CHANNEL,
            }
        )
        report = signapy.discover(df, target="converted", positive_class=True)
        assert [r.feature for r in report.features] == [
            "region",
            "acquisition_channel",
        ]

    def test_feature_lookup(self):
        df = pd.DataFrame(
            {
                "acquisition_channel": CHANNEL,
                "region": ["west"] * 80 + ["east"] * 120,
                "converted": CONVERTED,
            }
        )
        report = signapy.discover(df, target="converted", positive_class=True)
        assert report.feature("region").feature == "region"
        assert report.feature("acquisition_channel").feature == "acquisition_channel"

    def test_matches_low_level_metrics_directly(self):
        # Cross-check discover()'s wiring against the same low-level
        # functions called directly on a manually built contingency table.
        df = pd.DataFrame(
            {
                "region": ["west"] * 80 + ["east"] * 120,
                "converted": [True] * 30 + [False] * 50 + [True] * 10 + [False] * 110,
            }
        )
        report = signapy.discover(df, target="converted", positive_class=True)
        result = report.feature("region")

        table = pd.crosstab(df["region"], df["converted"])
        expected_v = cramers_v(table)
        expected_chi = chi_square(table)
        expected_lift = categorical_lift(
            df["region"], df["converted"], positive_class=True
        )

        assert result.effect_size == pytest.approx(expected_v)
        assert result.details["statistic"] == pytest.approx(expected_chi.statistic)
        assert result.p_value == pytest.approx(expected_chi.p_value)
        assert result.details["dof"] == expected_chi.dof

        by_value = values_by_value(result)
        for lift_result in expected_lift:
            value_result = by_value[lift_result.value]
            assert value_result.support == lift_result.support
            assert value_result.target_count == lift_result.positive_count
            assert value_result.target_rate == pytest.approx(lift_result.positive_rate)
            assert value_result.baseline_rate == pytest.approx(
                lift_result.baseline_rate
            )
            assert value_result.lift == pytest.approx(lift_result.lift)


class TestDiscoverMissingData:
    def test_missing_target_rows_excluded_from_all_analysis(self):
        df = pd.DataFrame(
            {
                "feature": ["a", "a", "b", "b", "a"],
                "target": [True, False, True, None, True],
            }
        )
        report = signapy.discover(df, target="target", positive_class=True)
        result = report.feature("feature")

        # Row index 3 (target missing) is dropped entirely: n=4, not 5.
        assert result.n == 4
        assert result.missing_rate == pytest.approx(0.0)
        assert result.effect_size == pytest.approx(0.0)
        assert result.details["statistic"] == pytest.approx(0.4444444444444444)
        assert result.p_value == pytest.approx(0.5049850750938457)
        assert result.details["dof"] == 1

        by_value = values_by_value(result)
        assert by_value["a"].support == 3
        assert by_value["a"].target_count == 2
        assert by_value["a"].target_rate == pytest.approx(2 / 3)
        assert by_value["a"].baseline_rate == pytest.approx(0.75)
        assert by_value["b"].support == 1
        assert by_value["b"].target_count == 1

    def test_per_feature_missing_values_and_missing_rate(self):
        df = pd.DataFrame(
            {
                "with_gap": ["a", "a", "b", "b", None],
                "no_gap": ["x", "x", "y", "y", "x"],
                "target": [True, False, True, False, True],
            }
        )
        report = signapy.discover(df, target="target", positive_class=True)

        with_gap = report.feature("with_gap")
        no_gap = report.feature("no_gap")

        # Both features share the same 5 target-valid rows, but only
        # "with_gap" has a missing value, and only its own analysis drops it.
        assert with_gap.n == 4
        assert with_gap.missing_rate == pytest.approx(0.2)
        assert no_gap.n == 5
        assert no_gap.missing_rate == pytest.approx(0.0)

        with_gap_values = values_by_value(with_gap)
        assert with_gap_values["a"].support == 2
        assert with_gap_values["a"].target_count == 1
        assert with_gap_values["a"].baseline_rate == pytest.approx(0.5)
        assert with_gap_values["b"].support == 2

    def test_does_not_mutate_caller_dataframe(self):
        df = basic_df()
        original = df.copy(deep=True)
        signapy.discover(df, target="converted", positive_class=True)
        pd.testing.assert_frame_equal(df, original)


class TestDiscoverPositiveClass:
    def test_flipping_positive_class_flips_value_level_rates_only(self):
        df = basic_df()
        report_true = signapy.discover(df, target="converted", positive_class=True)
        report_false = signapy.discover(df, target="converted", positive_class=False)

        result_true = report_true.feature("acquisition_channel")
        result_false = report_false.feature("acquisition_channel")

        # Cramér's V and the chi-square test are symmetric in which class is
        # "positive" -- only the value-level rates/lifts should change.
        assert result_true.effect_size == pytest.approx(result_false.effect_size)
        assert result_true.p_value == pytest.approx(result_false.p_value)

        by_value_true = values_by_value(result_true)
        by_value_false = values_by_value(result_false)
        assert by_value_true["referral"].target_rate == pytest.approx(0.3)
        assert by_value_false["referral"].target_rate == pytest.approx(0.7)
        assert by_value_true["referral"].baseline_rate == pytest.approx(0.2)
        assert by_value_false["referral"].baseline_rate == pytest.approx(0.8)


class TestDiscoverDtypePolicy:
    def test_accepts_object_dtype(self):
        report = signapy.discover(basic_df(), target="converted", positive_class=True)
        assert report.feature("acquisition_channel").feature_type is (
            FeatureType.CATEGORICAL
        )

    def test_accepts_pandas_string_dtype(self):
        df = basic_df()
        df["acquisition_channel"] = df["acquisition_channel"].astype("string")
        report = signapy.discover(df, target="converted", positive_class=True)
        assert report.feature("acquisition_channel").effect_size == pytest.approx(
            0.24034381938205007
        )

    def test_accepts_pandas_category_dtype(self):
        df = basic_df()
        df["acquisition_channel"] = df["acquisition_channel"].astype("category")
        report = signapy.discover(df, target="converted", positive_class=True)
        assert report.feature("acquisition_channel").effect_size == pytest.approx(
            0.24034381938205007
        )


class TestDiscoverRejections:
    def test_rejects_unknown_target_column(self):
        with pytest.raises(ValueError, match="not found"):
            signapy.discover(basic_df(), target="nope", positive_class=True)

    def test_rejects_empty_dataframe(self):
        df = pd.DataFrame(columns=["acquisition_channel", "converted"])
        with pytest.raises(ValueError, match="empty"):
            signapy.discover(df, target="converted", positive_class=True)

    def test_rejects_single_class_target(self):
        df = pd.DataFrame(
            {"feature": ["a", "b", "a", "b"], "target": [True, True, True, True]}
        )
        with pytest.raises(ValueError, match="must be binary"):
            signapy.discover(df, target="target", positive_class=True)

    def test_rejects_more_than_two_target_classes(self):
        df = pd.DataFrame({"feature": ["a", "b", "c"], "target": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="must be binary"):
            signapy.discover(df, target="target", positive_class="x")

    def test_rejects_unobserved_positive_class(self):
        with pytest.raises(ValueError, match="not one of the observed"):
            signapy.discover(basic_df(), target="converted", positive_class="nope")

    def test_rejects_numeric_feature_columns(self):
        df = basic_df()
        df["numeric_feature"] = list(range(len(df)))
        with pytest.raises(ValueError, match="unsupported dtype"):
            signapy.discover(df, target="converted", positive_class=True)

    def test_rejects_boolean_feature_columns(self):
        df = basic_df()
        df["boolean_feature"] = [True, False] * (len(df) // 2)
        with pytest.raises(ValueError, match="unsupported dtype"):
            signapy.discover(df, target="converted", positive_class=True)

    def test_rejects_single_category_features(self):
        df = pd.DataFrame(
            {
                "feature": ["only_one"] * 10,
                "target": [True, False] * 5,
            }
        )
        with pytest.raises(ValueError, match="fewer than 2 observed categories"):
            signapy.discover(df, target="target", positive_class=True)


def test_no_warnings_raised(recwarn):
    signapy.discover(basic_df(), target="converted", positive_class=True)
    assert len(recwarn) == 0


def test_public_import():
    assert callable(signapy.discover)
