import dataclasses

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from signapy.metrics.association import PointBiserialResult, point_biserial

# Known positive relationship: "yes" rows have clearly higher feature values.
POSITIVE_FEATURE = [10, 12, 14, 20, 22, 24, 30, 32, 34]
POSITIVE_TARGET = ["no"] * 6 + ["yes"] * 3

# Same values, but a target with the classes interleaved so the split is
# roughly balanced and unrelated to the feature -> near-zero relationship.
NEAR_ZERO_FEATURE = [1, 2, 3, 4, 5, 6, 7, 8]
NEAR_ZERO_TARGET = ["a", "b", "a", "b", "a", "b", "a", "b"]


class TestPointBiserialKnownValues:
    def test_known_positive_relationship(self):
        result = point_biserial(POSITIVE_FEATURE, POSITIVE_TARGET, positive_class="yes")
        assert result.coefficient == pytest.approx(0.8492077756084466)
        assert result.p_value == pytest.approx(0.0037706443519795758)
        assert result.positive_n == 3
        assert result.negative_n == 6
        assert result.positive_mean == pytest.approx(32.0)
        assert result.negative_mean == pytest.approx(17.0)
        assert result.mean_difference == pytest.approx(15.0)

    def test_known_negative_relationship_via_positive_class_reversal(self):
        # Same data, opposite positive_class -> same magnitude, flipped sign.
        result = point_biserial(POSITIVE_FEATURE, POSITIVE_TARGET, positive_class="no")
        assert result.coefficient == pytest.approx(-0.8492077756084467)
        assert result.p_value == pytest.approx(0.0037706443519795758)
        assert result.positive_n == 6
        assert result.negative_n == 3
        assert result.positive_mean == pytest.approx(17.0)
        assert result.negative_mean == pytest.approx(32.0)
        assert result.mean_difference == pytest.approx(-15.0)

    def test_near_zero_relationship(self):
        result = point_biserial(NEAR_ZERO_FEATURE, NEAR_ZERO_TARGET, positive_class="b")
        assert result.coefficient == pytest.approx(0.21821789023599225)
        assert result.p_value == pytest.approx(0.6036450565101367)

    def test_matches_scipy_pointbiserialr_directly(self):
        is_positive = np.array([t == "yes" for t in POSITIVE_TARGET])
        expected = scipy.stats.pointbiserialr(
            is_positive, np.array(POSITIVE_FEATURE, dtype=float)
        )
        result = point_biserial(POSITIVE_FEATURE, POSITIVE_TARGET, positive_class="yes")
        assert result.coefficient == pytest.approx(expected.statistic)
        assert result.p_value == pytest.approx(expected.pvalue)


class TestPointBiserialInvariants:
    def test_row_order_permutation_leaves_result_unchanged(self):
        rng = np.random.default_rng(0)
        permutation = rng.permutation(len(POSITIVE_FEATURE))
        permuted_feature = [POSITIVE_FEATURE[i] for i in permutation]
        permuted_target = [POSITIVE_TARGET[i] for i in permutation]

        baseline = point_biserial(
            POSITIVE_FEATURE, POSITIVE_TARGET, positive_class="yes"
        )
        permuted = point_biserial(
            permuted_feature, permuted_target, positive_class="yes"
        )

        assert permuted.coefficient == pytest.approx(baseline.coefficient)
        assert permuted.p_value == pytest.approx(baseline.p_value)
        assert permuted.positive_mean == pytest.approx(baseline.positive_mean)
        assert permuted.negative_mean == pytest.approx(baseline.negative_mean)

    def test_positive_linear_rescaling_preserves_coefficient(self):
        baseline = point_biserial(
            POSITIVE_FEATURE, POSITIVE_TARGET, positive_class="yes"
        )
        rescaled_feature = [value * 2.0 + 5.0 for value in POSITIVE_FEATURE]
        rescaled = point_biserial(
            rescaled_feature, POSITIVE_TARGET, positive_class="yes"
        )
        assert rescaled.coefficient == pytest.approx(baseline.coefficient)

    def test_negative_linear_rescaling_reverses_coefficient(self):
        baseline = point_biserial(
            POSITIVE_FEATURE, POSITIVE_TARGET, positive_class="yes"
        )
        rescaled_feature = [value * -3.0 + 1.0 for value in POSITIVE_FEATURE]
        rescaled = point_biserial(
            rescaled_feature, POSITIVE_TARGET, positive_class="yes"
        )
        assert rescaled.coefficient == pytest.approx(-baseline.coefficient)

    def test_coefficient_bounded(self):
        # Perfect separation: coefficient should sit at the boundary, not
        # exceed it.
        result = point_biserial(
            [1, 2, 3, 100, 101, 102], ["no"] * 3 + ["yes"] * 3, positive_class="yes"
        )
        assert -1.0 <= result.coefficient <= 1.0


class TestPointBiserialAcceptedInputTypes:
    def test_accepts_python_lists(self):
        result = point_biserial(POSITIVE_FEATURE, POSITIVE_TARGET, positive_class="yes")
        assert result.coefficient == pytest.approx(0.8492077756084466)

    def test_accepts_numpy_arrays(self):
        result = point_biserial(
            np.array(POSITIVE_FEATURE, dtype=float),
            np.array(POSITIVE_TARGET),
            positive_class="yes",
        )
        assert result.coefficient == pytest.approx(0.8492077756084466)

    def test_accepts_pandas_series(self):
        result = point_biserial(
            pd.Series(POSITIVE_FEATURE),
            pd.Series(POSITIVE_TARGET),
            positive_class="yes",
        )
        assert result.coefficient == pytest.approx(0.8492077756084466)

    def test_accepts_integer_valued_measurements(self):
        result = point_biserial(
            [1, 2, 3, 10, 11, 12], ["a"] * 3 + ["b"] * 3, positive_class="b"
        )
        assert result.positive_mean == pytest.approx(11.0)

    def test_accepts_pandas_nullable_int64(self):
        feature = pd.array([1, 2, 3, 4, 5, 6], dtype="Int64")
        result = point_biserial(
            feature, ["a", "a", "a", "b", "b", "b"], positive_class="b"
        )
        assert result.positive_mean == pytest.approx(5.0)
        assert result.negative_mean == pytest.approx(2.0)

    def test_accepts_pandas_nullable_float64(self):
        feature = pd.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="Float64")
        result = point_biserial(
            feature, ["a", "a", "a", "b", "b", "b"], positive_class="b"
        )
        assert result.positive_mean == pytest.approx(5.0)


class TestPointBiserialInvalidInputs:
    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            point_biserial([], [], positive_class="yes")

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            point_biserial([1, 2, 3], [1, 2], positive_class=1)

    def test_rejects_non_one_dimensional_feature(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            point_biserial([[1, 2], [3, 4]], ["a", "b"], positive_class="a")

    def test_rejects_non_one_dimensional_target(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            point_biserial([1, 2], [["a"], ["b"]], positive_class="a")

    def test_rejects_missing_feature_values(self):
        with pytest.raises(ValueError, match="missing feature values"):
            point_biserial([1, 2, None], ["a", "b", "a"], positive_class="a")

    def test_rejects_missing_target_values(self):
        with pytest.raises(ValueError, match="missing target values"):
            point_biserial([1, 2, 3], ["a", None, "a"], positive_class="a")

    def test_rejects_non_numeric_feature(self):
        with pytest.raises(ValueError, match="must be numeric"):
            point_biserial(["x", "y", "z"], ["a", "b", "a"], positive_class="a")

    def test_rejects_boolean_feature(self):
        with pytest.raises(ValueError, match="must not be boolean"):
            point_biserial([True, False, True], ["a", "b", "a"], positive_class="a")

    def test_rejects_boolean_feature_as_numpy_array(self):
        with pytest.raises(ValueError, match="must not be boolean"):
            point_biserial(
                np.array([True, False, True]), ["a", "b", "a"], positive_class="a"
            )

    def test_rejects_boolean_feature_as_nullable_dtype(self):
        feature = pd.array([True, False, True], dtype="boolean")
        with pytest.raises(ValueError, match="must not be boolean"):
            point_biserial(feature, ["a", "b", "a"], positive_class="a")

    def test_rejects_boolean_feature_stored_as_object_dtype(self):
        # Regression test: True/False stored in an object-dtype array used
        # to bypass the boolean check entirely and coerce through
        # pandas.to_numeric to 1.0/0.0 as if it were an ordinary continuous
        # measurement.
        feature = np.array([True, False, True], dtype=object)
        with pytest.raises(ValueError, match="must not be boolean"):
            point_biserial(feature, ["a", "b", "a"], positive_class="a")

    def test_rejects_datetime_feature(self):
        # Regression test: a datetime64 feature used to be silently
        # coerced to nanosecond-epoch timestamp floats by pandas.to_numeric.
        feature = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
        )
        with pytest.raises(ValueError, match="datetime"):
            point_biserial(feature, ["a", "b", "a", "b"], positive_class="a")

    def test_rejects_timedelta_feature(self):
        feature = pd.to_timedelta([1, 2, 3, 4], unit="D")
        with pytest.raises(ValueError, match="timedelta"):
            point_biserial(feature, ["a", "b", "a", "b"], positive_class="a")

    def test_rejects_complex_feature(self):
        # Regression test: complex values used to be silently truncated to
        # their real component by pandas.to_numeric/.to_numpy(dtype=float),
        # which also emits a ComplexWarning (an error under this project's
        # pytest config) rather than a clear ValueError.
        feature = np.array([1 + 2j, 2 + 1j, 3 + 0j, 10 + 5j])
        with pytest.raises(ValueError, match="real-valued"):
            point_biserial(feature, ["a", "b", "a", "b"], positive_class="a")

    def test_rejects_infinite_feature_values(self):
        with pytest.raises(ValueError, match="infinite"):
            point_biserial([1, 2, float("inf")], ["a", "b", "a"], positive_class="a")

    def test_rejects_negative_infinite_feature_values(self):
        with pytest.raises(ValueError, match="infinite"):
            point_biserial([1, 2, float("-inf")], ["a", "b", "a"], positive_class="a")

    def test_rejects_single_target_class(self):
        with pytest.raises(ValueError, match="binary target"):
            point_biserial([1, 2, 3], ["a", "a", "a"], positive_class="a")

    def test_rejects_three_target_classes(self):
        with pytest.raises(ValueError, match="binary target"):
            point_biserial([1, 2, 3], ["a", "b", "c"], positive_class="a")

    def test_rejects_unobserved_positive_class(self):
        with pytest.raises(ValueError, match="not one of the observed"):
            point_biserial([1, 2, 3], ["a", "b", "a"], positive_class="z")

    def test_rejects_constant_feature(self):
        with pytest.raises(ValueError, match="zero variance"):
            point_biserial([5, 5, 5, 5], ["a", "b", "a", "b"], positive_class="a")

    def test_rejects_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 3 observations"):
            point_biserial([1, 2], ["a", "b"], positive_class="a")


class TestPointBiserialResult:
    def make(self, **overrides):
        fields = dict(
            coefficient=0.5,
            p_value=0.01,
            positive_n=3,
            negative_n=6,
            positive_mean=32.0,
            negative_mean=17.0,
            mean_difference=15.0,
        )
        return PointBiserialResult(**{**fields, **overrides})

    def test_is_frozen(self):
        result = self.make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.coefficient = 0.9

    def test_rejects_coefficient_outside_bounds(self):
        with pytest.raises(ValueError, match="coefficient"):
            self.make(coefficient=1.5)

    def test_rejects_p_value_outside_bounds(self):
        with pytest.raises(ValueError, match="p_value"):
            self.make(p_value=-0.1)

    def test_rejects_non_positive_group_counts(self):
        with pytest.raises(ValueError, match="positive_n"):
            self.make(positive_n=0)
        with pytest.raises(ValueError, match="negative_n"):
            self.make(negative_n=0)

    def test_rejects_inconsistent_mean_difference(self):
        with pytest.raises(ValueError, match="mean_difference"):
            self.make(mean_difference=999.0)


def test_no_warnings_raised(recwarn):
    point_biserial(POSITIVE_FEATURE, POSITIVE_TARGET, positive_class="yes")
    point_biserial(NEAR_ZERO_FEATURE, NEAR_ZERO_TARGET, positive_class="b")
    assert len(recwarn) == 0
