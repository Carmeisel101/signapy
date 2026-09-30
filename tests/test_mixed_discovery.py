import pandas as pd
import pytest

import signapy
from signapy.metrics.association import cramers_v, point_biserial
from signapy.metrics.significance import chi_square
from signapy.profiling import FeatureType, TargetType
from signapy.results import FeatureResult

# Same known point-biserial fixture as tests/test_point_biserial.py.
MEASUREMENT = [10, 12, 14, 20, 22, 24, 30, 32, 34]
OUTCOME = ["no"] * 6 + ["yes"] * 3


def continuous_df() -> pd.DataFrame:
    return pd.DataFrame({"measurement": MEASUREMENT, "outcome": OUTCOME})


def mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category_context": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
            "continuous_measurement": [
                10.0,
                11.0,
                9.0,
                12.0,
                8.0,
                10.5,
                9.5,
                11.5,
                8.5,
                10.0,
                20.0,
                22.0,
                18.0,
                21.0,
                19.0,
                20.5,
                19.5,
                21.5,
                18.5,
                20.0,
                30.0,
                32.0,
                28.0,
                31.0,
                29.0,
                30.5,
                29.5,
                31.5,
                28.5,
                30.0,
            ],
            "outcome": (
                [False] * 7
                + [True] * 3
                + [False] * 4
                + [True] * 6
                + [True] * 9
                + [False] * 1
            ),
        }
    )


class TestContinuousDiscoveryKnownValues:
    def test_matches_known_point_biserial_values(self):
        report = signapy.discover(
            continuous_df(),
            target="outcome",
            positive_class="yes",
            feature_types={"measurement": FeatureType.CONTINUOUS},
        )
        result = report.feature("measurement")

        assert result.effect_size_method == "point_biserial"
        assert result.effect_size == pytest.approx(0.8492077756084466)
        assert result.test_method == "point_biserial"
        assert result.p_value == pytest.approx(0.0037706443519795758)
        assert result.n == 9
        assert result.missing_rate == pytest.approx(0.0)
        assert result.feature_type is FeatureType.CONTINUOUS
        assert result.target_type is TargetType.BINARY

    def test_correct_group_summaries_in_details(self):
        report = signapy.discover(
            continuous_df(),
            target="outcome",
            positive_class="yes",
            feature_types={"measurement": FeatureType.CONTINUOUS},
        )
        details = report.feature("measurement").details
        assert details["positive_n"] == 3
        assert details["negative_n"] == 6
        assert details["positive_mean"] == pytest.approx(32.0)
        assert details["negative_mean"] == pytest.approx(17.0)
        assert details["mean_difference"] == pytest.approx(15.0)

    def test_values_is_none(self):
        report = signapy.discover(
            continuous_df(),
            target="outcome",
            positive_class="yes",
            feature_types={"measurement": FeatureType.CONTINUOUS},
        )
        assert report.feature("measurement").values is None

    def test_result_is_feature_result(self):
        report = signapy.discover(
            continuous_df(),
            target="outcome",
            positive_class="yes",
            feature_types={"measurement": FeatureType.CONTINUOUS},
        )
        assert isinstance(report.feature("measurement"), FeatureResult)


class TestContinuousDiscoveryInputTypes:
    def test_integer_measurement_declared_continuous(self):
        df = continuous_df()
        df["measurement"] = df["measurement"].astype(int)
        report = signapy.discover(
            df,
            target="outcome",
            positive_class="yes",
            feature_types={"measurement": FeatureType.CONTINUOUS},
        )
        assert report.feature("measurement").effect_size == pytest.approx(
            0.8492077756084466
        )

    def test_float_measurement_declared_continuous(self):
        df = continuous_df()
        df["measurement"] = df["measurement"].astype(float)
        report = signapy.discover(
            df,
            target="outcome",
            positive_class="yes",
            feature_types={"measurement": FeatureType.CONTINUOUS},
        )
        assert report.feature("measurement").effect_size == pytest.approx(
            0.8492077756084466
        )

    def test_pandas_nullable_numeric_dtype(self):
        df = continuous_df()
        df["measurement"] = df["measurement"].astype("Int64")
        report = signapy.discover(
            df,
            target="outcome",
            positive_class="yes",
            feature_types={"measurement": FeatureType.CONTINUOUS},
        )
        assert report.feature("measurement").effect_size == pytest.approx(
            0.8492077756084466
        )

    def test_feature_type_string_form_accepted(self):
        report = signapy.discover(
            continuous_df(),
            target="outcome",
            positive_class="yes",
            feature_types={"measurement": "continuous"},
        )
        assert report.feature("measurement").feature_type is FeatureType.CONTINUOUS


class TestContinuousDiscoveryMissingData:
    def test_feature_specific_missing_value_removal(self):
        df = continuous_df()
        df.loc[0, "measurement"] = None  # a "no" row
        report = signapy.discover(
            df,
            target="outcome",
            positive_class="yes",
            feature_types={"measurement": FeatureType.CONTINUOUS},
        )
        result = report.feature("measurement")
        assert result.n == 8
        assert result.missing_rate == pytest.approx(1 / 9)

    def test_infinite_values_raise_rather_than_being_removed(self):
        df = continuous_df()
        df["measurement"] = df["measurement"].astype(float)
        df.loc[0, "measurement"] = float("inf")
        with pytest.raises(ValueError, match="infinite"):
            signapy.discover(
                df,
                target="outcome",
                positive_class="yes",
                feature_types={"measurement": FeatureType.CONTINUOUS},
            )

    def test_both_target_classes_disappearing_after_feature_filtering(self):
        # Drop the feature value for every "yes" row -> after this
        # feature's own missing-value filtering, only "no" remains.
        df = continuous_df()
        df.loc[6:8, "measurement"] = None
        with pytest.raises(ValueError, match="could not compute association"):
            signapy.discover(
                df,
                target="outcome",
                positive_class="yes",
                feature_types={"measurement": FeatureType.CONTINUOUS},
            )


class TestMixedDiscovery:
    def test_one_categorical_and_one_continuous_feature(self):
        df = mixed_df()
        report = signapy.discover(
            df,
            target="outcome",
            positive_class=True,
            feature_types={
                "category_context": FeatureType.CATEGORICAL,
                "continuous_measurement": FeatureType.CONTINUOUS,
            },
        )

        categorical_result = report.feature("category_context")
        continuous_result = report.feature("continuous_measurement")

        table = pd.crosstab(df["category_context"], df["outcome"])
        expected_v = cramers_v(table)
        expected_chi = chi_square(table)
        assert categorical_result.effect_size == pytest.approx(expected_v)
        assert categorical_result.p_value == pytest.approx(expected_chi.p_value)
        assert categorical_result.values is not None

        expected_pb = point_biserial(
            df["continuous_measurement"], df["outcome"], positive_class=True
        )
        assert continuous_result.effect_size == pytest.approx(expected_pb.coefficient)
        assert continuous_result.p_value == pytest.approx(expected_pb.p_value)
        assert continuous_result.values is None

    def test_categorical_values_populated_continuous_values_none(self):
        report = signapy.discover(
            mixed_df(),
            target="outcome",
            positive_class=True,
            feature_types={
                "category_context": FeatureType.CATEGORICAL,
                "continuous_measurement": FeatureType.CONTINUOUS,
            },
        )
        assert report.feature("category_context").values is not None
        assert len(report.feature("category_context").values) == 3
        assert report.feature("continuous_measurement").values is None

    def test_multiple_features_of_each_type(self):
        df = mixed_df()
        df["category_context_2"] = df["category_context"]
        df["continuous_measurement_2"] = df["continuous_measurement"]
        report = signapy.discover(
            df,
            target="outcome",
            positive_class=True,
            feature_types={
                "category_context": FeatureType.CATEGORICAL,
                "category_context_2": FeatureType.CATEGORICAL,
                "continuous_measurement": FeatureType.CONTINUOUS,
                "continuous_measurement_2": FeatureType.CONTINUOUS,
            },
        )
        names = {result.feature for result in report.features}
        assert names == {
            "category_context",
            "category_context_2",
            "continuous_measurement",
            "continuous_measurement_2",
        }
        assert report.feature("category_context_2").effect_size == pytest.approx(
            report.feature("category_context").effect_size
        )
        assert report.feature("continuous_measurement_2").effect_size == pytest.approx(
            report.feature("continuous_measurement").effect_size
        )

    def test_results_follow_dataframe_column_order(self):
        df = mixed_df()
        # feature_types given in the opposite order to df's columns.
        report = signapy.discover(
            df,
            target="outcome",
            positive_class=True,
            feature_types={
                "continuous_measurement": FeatureType.CONTINUOUS,
                "category_context": FeatureType.CATEGORICAL,
            },
        )
        assert [result.feature for result in report.features] == [
            "category_context",
            "continuous_measurement",
        ]

    def test_extra_dataframe_columns_ignored_when_not_selected(self):
        df = mixed_df()
        df["untouched"] = range(len(df))
        report = signapy.discover(
            df,
            target="outcome",
            positive_class=True,
            feature_types={"category_context": FeatureType.CATEGORICAL},
        )
        assert [result.feature for result in report.features] == ["category_context"]

    def test_input_dataframe_is_not_mutated(self):
        df = mixed_df()
        original = df.copy(deep=True)
        signapy.discover(
            df,
            target="outcome",
            positive_class=True,
            feature_types={
                "category_context": FeatureType.CATEGORICAL,
                "continuous_measurement": FeatureType.CONTINUOUS,
            },
        )
        pd.testing.assert_frame_equal(df, original)

    def test_backward_compatible_with_categorical_only_calls(self):
        # feature_types=None (the old default) must match declaring the
        # same columns CATEGORICAL explicitly.
        df = mixed_df().drop(columns=["continuous_measurement"])
        implicit = signapy.discover(df, target="outcome", positive_class=True)
        explicit = signapy.discover(
            df,
            target="outcome",
            positive_class=True,
            feature_types={"category_context": FeatureType.CATEGORICAL},
        )
        assert implicit.feature("category_context").effect_size == pytest.approx(
            explicit.feature("category_context").effect_size
        )
        assert implicit.feature("category_context").p_value == pytest.approx(
            explicit.feature("category_context").p_value
        )


class TestMixedDiscoveryRejections:
    def test_rejects_unknown_selected_column(self):
        with pytest.raises(ValueError, match="not a column of df"):
            signapy.discover(
                mixed_df(),
                target="outcome",
                positive_class=True,
                feature_types={"nope": FeatureType.CATEGORICAL},
            )

    def test_rejects_target_in_feature_types(self):
        with pytest.raises(ValueError, match="must not include the target"):
            signapy.discover(
                mixed_df(),
                target="outcome",
                positive_class=True,
                feature_types={"outcome": FeatureType.CATEGORICAL},
            )

    def test_rejects_empty_feature_types(self):
        with pytest.raises(ValueError, match="must not be empty"):
            signapy.discover(
                mixed_df(), target="outcome", positive_class=True, feature_types={}
            )

    def test_rejects_unsupported_declared_feature_type(self):
        with pytest.raises(ValueError, match="not yet supported"):
            signapy.discover(
                mixed_df(),
                target="outcome",
                positive_class=True,
                feature_types={"category_context": FeatureType.BOOLEAN},
            )

    def test_rejects_ordinal_declared_feature_type(self):
        with pytest.raises(ValueError, match="not yet supported"):
            signapy.discover(
                mixed_df(),
                target="outcome",
                positive_class=True,
                feature_types={"category_context": FeatureType.ORDINAL},
            )

    def test_rejects_invalid_type_string(self):
        with pytest.raises(ValueError, match="not a valid FeatureType"):
            signapy.discover(
                mixed_df(),
                target="outcome",
                positive_class=True,
                feature_types={"category_context": "categoricaal"},
            )

    def test_rejects_non_numeric_column_declared_continuous(self):
        with pytest.raises(ValueError, match="could not compute association"):
            signapy.discover(
                mixed_df(),
                target="outcome",
                positive_class=True,
                feature_types={"category_context": FeatureType.CONTINUOUS},
            )

    def test_rejects_unsupported_dtype_categorical_column(self):
        df = mixed_df()
        with pytest.raises(ValueError, match="unsupported dtype"):
            signapy.discover(
                df,
                target="outcome",
                positive_class=True,
                feature_types={"continuous_measurement": FeatureType.CATEGORICAL},
            )


def test_no_warnings_raised(recwarn):
    signapy.discover(
        mixed_df(),
        target="outcome",
        positive_class=True,
        feature_types={
            "category_context": FeatureType.CATEGORICAL,
            "continuous_measurement": FeatureType.CONTINUOUS,
        },
    )
    assert len(recwarn) == 0
