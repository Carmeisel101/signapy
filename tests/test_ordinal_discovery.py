import pandas as pd
import pytest

import signapy
from signapy.metrics.association import spearman_rho
from signapy.profiling import FeatureType, TargetType
from signapy.results import FeatureResult

# Increasing severity -> increasing conversion rate: low 2/10, medium 5/10,
# high 8/10. Same shape as tests/test_spearman.py's INCREASING_* fixtures.
SEVERITY_LEVELS = ["low"] * 10 + ["medium"] * 10 + ["high"] * 10
INCREASING_OUTCOME = (
    [False] * 8 + [True] * 2 + [False] * 5 + [True] * 5 + [False] * 2 + [True] * 8
)


def ordinal_series(
    levels: list, categories: list | None = None, ordered: bool = True
) -> pd.Categorical:
    return pd.Categorical(
        levels, categories=categories or ["low", "medium", "high"], ordered=ordered
    )


def severity_df(
    levels=None, outcome=None, categories=None, ordered=True
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "severity": ordinal_series(
                levels if levels is not None else SEVERITY_LEVELS,
                categories=categories,
                ordered=ordered,
            ),
            "converted": outcome if outcome is not None else INCREASING_OUTCOME,
        }
    )


def values_by_value(feature_result: FeatureResult) -> dict:
    return {v.value: v for v in feature_result.values}


class TestOrdinalDiscoveryKnownValues:
    def test_matches_known_spearman_values(self):
        report = signapy.discover(
            severity_df(),
            target="converted",
            positive_class=True,
            feature_types={"severity": FeatureType.ORDINAL},
        )
        result = report.feature("severity")

        assert result.effect_size_method == "spearman_rho"
        assert result.effect_size == pytest.approx(0.4898979485566356)
        assert result.test_method == "spearman"
        assert result.p_value == pytest.approx(0.0059963724801474)
        assert result.n == 30
        assert result.missing_rate == pytest.approx(0.0)
        assert result.feature_type is FeatureType.ORDINAL
        assert result.target_type is TargetType.BINARY
        assert result.details == {"n_levels": 3}

    def test_matches_low_level_metric_directly(self):
        df = severity_df()
        report = signapy.discover(
            df,
            target="converted",
            positive_class=True,
            feature_types={"severity": FeatureType.ORDINAL},
        )
        result = report.feature("severity")

        codes = df["severity"].cat.codes.to_numpy()
        expected = spearman_rho(codes, df["converted"], positive_class=True)
        assert result.effect_size == pytest.approx(expected.coefficient)
        assert result.p_value == pytest.approx(expected.p_value)
        assert result.details["n_levels"] == expected.n_levels


class TestOrdinalValueOrdering:
    def test_values_follow_declared_category_order_not_first_appearance(self):
        # Reverse the row order so "first appearance" would be high, medium,
        # low if that's what the implementation used.
        reversed_levels = list(reversed(SEVERITY_LEVELS))
        reversed_outcome = list(reversed(INCREASING_OUTCOME))
        df = severity_df(levels=reversed_levels, outcome=reversed_outcome)

        report = signapy.discover(
            df,
            target="converted",
            positive_class=True,
            feature_types={"severity": FeatureType.ORDINAL},
        )
        assert [v.value for v in report.feature("severity").values] == [
            "low",
            "medium",
            "high",
        ]

    def test_value_level_rates_and_lift(self):
        report = signapy.discover(
            severity_df(),
            target="converted",
            positive_class=True,
            feature_types={"severity": FeatureType.ORDINAL},
        )
        by_value = values_by_value(report.feature("severity"))

        assert by_value["low"].support == 10
        assert by_value["low"].target_rate == pytest.approx(0.2)
        assert by_value["low"].baseline_rate == pytest.approx(0.5)
        assert by_value["low"].lift == pytest.approx(0.4)

        assert by_value["medium"].target_rate == pytest.approx(0.5)
        assert by_value["medium"].lift == pytest.approx(1.0)

        assert by_value["high"].target_rate == pytest.approx(0.8)
        assert by_value["high"].lift == pytest.approx(1.6)

    def test_declared_category_with_zero_support_is_absent_not_zero_filled(self):
        # "critical" is declared but never observed in the data.
        df = severity_df(categories=["low", "medium", "high", "critical"])
        report = signapy.discover(
            df,
            target="converted",
            positive_class=True,
            feature_types={"severity": FeatureType.ORDINAL},
        )
        values = report.feature("severity").values
        assert [v.value for v in values] == ["low", "medium", "high"]
        assert "critical" not in {v.value for v in values}


class TestOrdinalDiscoveryPositiveClass:
    def test_reversing_positive_class_flips_sign_and_rates(self):
        report_true = signapy.discover(
            severity_df(),
            target="converted",
            positive_class=True,
            feature_types={"severity": FeatureType.ORDINAL},
        )
        report_false = signapy.discover(
            severity_df(),
            target="converted",
            positive_class=False,
            feature_types={"severity": FeatureType.ORDINAL},
        )
        result_true = report_true.feature("severity")
        result_false = report_false.feature("severity")

        assert result_true.effect_size == pytest.approx(-result_false.effect_size)
        assert result_true.p_value == pytest.approx(result_false.p_value)

        by_value_true = values_by_value(result_true)
        by_value_false = values_by_value(result_false)
        assert by_value_true["low"].target_rate == pytest.approx(0.2)
        assert by_value_false["low"].target_rate == pytest.approx(0.8)


class TestOrdinalDiscoveryMissingData:
    def test_feature_specific_missing_value_removal(self):
        levels = list(SEVERITY_LEVELS)
        levels[0] = None
        df = severity_df(levels=levels)
        report = signapy.discover(
            df,
            target="converted",
            positive_class=True,
            feature_types={"severity": FeatureType.ORDINAL},
        )
        result = report.feature("severity")
        assert result.n == 29
        assert result.missing_rate == pytest.approx(1 / 30)
        assert values_by_value(result)["low"].support == 9

    def test_missing_target_rows_excluded_from_all_analysis(self):
        outcome = list(INCREASING_OUTCOME)
        outcome[0] = None
        df = severity_df(outcome=outcome)
        report = signapy.discover(
            df,
            target="converted",
            positive_class=True,
            feature_types={"severity": FeatureType.ORDINAL},
        )
        assert report.feature("severity").n == 29


class TestOrdinalDiscoveryRejections:
    def test_rejects_non_categorical_dtype(self):
        df = severity_df()
        df["severity"] = df["severity"].astype(str)
        with pytest.raises(ValueError, match="ordered pandas Categorical"):
            signapy.discover(
                df,
                target="converted",
                positive_class=True,
                feature_types={"severity": FeatureType.ORDINAL},
            )

    def test_rejects_unordered_categorical_with_distinct_message(self):
        df = severity_df(ordered=False)
        with pytest.raises(ValueError, match="unordered categorical"):
            signapy.discover(
                df,
                target="converted",
                positive_class=True,
                feature_types={"severity": FeatureType.ORDINAL},
            )

    def test_two_dtype_errors_are_distinct(self):
        df_object = severity_df()
        df_object["severity"] = df_object["severity"].astype(str)
        df_unordered = severity_df(ordered=False)

        with pytest.raises(ValueError) as object_error:
            signapy.discover(
                df_object,
                target="converted",
                positive_class=True,
                feature_types={"severity": FeatureType.ORDINAL},
            )
        with pytest.raises(ValueError) as unordered_error:
            signapy.discover(
                df_unordered,
                target="converted",
                positive_class=True,
                feature_types={"severity": FeatureType.ORDINAL},
            )
        assert str(object_error.value) != str(unordered_error.value)

    def test_rejects_constant_feature_after_missing_value_filtering(self):
        # Only "low" survives once rows with a missing feature value are
        # dropped -> fewer than 2 distinct levels remain.
        levels = ["low"] * 5 + [None] * 25
        outcome = [True, False, True, False, True] + [True, False] * 12 + [True]
        df = severity_df(levels=levels, outcome=outcome)
        with pytest.raises(ValueError, match="could not compute association"):
            signapy.discover(
                df,
                target="converted",
                positive_class=True,
                feature_types={"severity": FeatureType.ORDINAL},
            )

    def test_rejects_too_few_observations(self):
        df = severity_df(
            levels=["low", "high"],
            outcome=[True, False],
            categories=["low", "medium", "high"],
        )
        with pytest.raises(ValueError, match="could not compute association"):
            signapy.discover(
                df,
                target="converted",
                positive_class=True,
                feature_types={"severity": FeatureType.ORDINAL},
            )


class TestOrdinalInMixedDiscovery:
    def test_categorical_continuous_and_ordinal_together(self):
        df = severity_df()
        df["channel"] = ["ref", "org"] * 15
        df["measurement"] = list(range(30))

        report = signapy.discover(
            df,
            target="converted",
            positive_class=True,
            feature_types={
                "severity": FeatureType.ORDINAL,
                "channel": FeatureType.CATEGORICAL,
                "measurement": FeatureType.CONTINUOUS,
            },
        )
        names = {r.feature for r in report.features}
        assert names == {"severity", "channel", "measurement"}
        assert report.feature("severity").feature_type is FeatureType.ORDINAL
        assert report.feature("severity").values is not None
        assert report.feature("measurement").values is None

    def test_column_order_preserved_with_ordinal(self):
        df = severity_df()
        df["channel"] = ["ref", "org"] * 15
        # Reorder columns: channel, converted, severity.
        df = df[["channel", "converted", "severity"]]

        report = signapy.discover(
            df,
            target="converted",
            positive_class=True,
            feature_types={
                "severity": FeatureType.ORDINAL,
                "channel": FeatureType.CATEGORICAL,
            },
        )
        assert [r.feature for r in report.features] == ["channel", "severity"]


def test_no_warnings_raised(recwarn):
    signapy.discover(
        severity_df(),
        target="converted",
        positive_class=True,
        feature_types={"severity": FeatureType.ORDINAL},
    )
    assert len(recwarn) == 0
