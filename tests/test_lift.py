import dataclasses

import numpy as np
import pandas as pd
import pytest

from signapy.metrics.lift import CategoricalValueMetrics, categorical_lift

# Same distribution as the two-category contingency table used in
# test_association.py / test_significance.py:
#              negative  positive
# referral          70        30
# paid              90        10
FEATURE = ["referral"] * 100 + ["paid"] * 100
TARGET = ["positive"] * 30 + ["negative"] * 70 + ["positive"] * 10 + ["negative"] * 90


def results_by_value(results):
    return {r.value: r for r in results}


class TestCategoricalLiftKnownValues:
    def test_per_category_metrics(self):
        results = categorical_lift(FEATURE, TARGET, positive_class="positive")
        by_value = results_by_value(results)

        assert by_value["referral"].support == 100
        assert by_value["referral"].positive_count == 30
        assert by_value["referral"].positive_rate == pytest.approx(0.3)
        assert by_value["referral"].baseline_rate == pytest.approx(0.2)
        assert by_value["referral"].lift == pytest.approx(1.5)

        assert by_value["paid"].support == 100
        assert by_value["paid"].positive_count == 10
        assert by_value["paid"].positive_rate == pytest.approx(0.1)
        assert by_value["paid"].baseline_rate == pytest.approx(0.2)
        assert by_value["paid"].lift == pytest.approx(0.5)

    def test_preserves_order_of_first_appearance(self):
        results = categorical_lift(FEATURE, TARGET, positive_class="positive")
        assert [r.value for r in results] == ["referral", "paid"]

    def test_three_category_lift(self):
        feature = ["referral"] * 200 + ["organic"] * 200 + ["paid"] * 200
        target = (
            ["positive"] * 60
            + ["negative"] * 140
            + ["positive"] * 40
            + ["negative"] * 160
            + ["positive"] * 20
            + ["negative"] * 180
        )
        results = categorical_lift(feature, target, positive_class="positive")
        by_value = results_by_value(results)
        baseline = 120 / 600
        assert by_value["referral"].lift == pytest.approx((60 / 200) / baseline)
        assert by_value["organic"].lift == pytest.approx((40 / 200) / baseline)
        assert by_value["paid"].lift == pytest.approx((20 / 200) / baseline)


class TestCategoricalLiftAcceptedInputTypes:
    @pytest.mark.parametrize(
        "as_type",
        [lambda x: x, lambda x: np.asarray(x), pd.Series],
        ids=["list", "numpy_array", "pandas_series"],
    )
    def test_accepts_array_like_inputs(self, as_type):
        results = categorical_lift(
            as_type(FEATURE), as_type(TARGET), positive_class="positive"
        )
        assert results_by_value(results)["referral"].lift == pytest.approx(1.5)

    def test_series_with_different_indexes_are_aligned_positionally(self):
        # Regression test: feature/target as pandas Series with mismatched
        # indexes must still be paired up by position, not by label.
        feature = pd.Series(FEATURE, index=range(len(FEATURE)))
        target = pd.Series(TARGET, index=range(1000, 1000 + len(TARGET)))
        results = categorical_lift(feature, target, positive_class="positive")
        by_value = results_by_value(results)
        assert by_value["referral"].positive_count == 30
        assert by_value["referral"].lift == pytest.approx(1.5)
        assert by_value["paid"].positive_count == 10
        assert by_value["paid"].lift == pytest.approx(0.5)

    def test_series_with_non_default_and_unsorted_index(self):
        feature = pd.Series(["a", "b", "a", "b"], index=[3, 1, 2, 0])
        target = pd.Series(["x", "y", "y", "x"], index=[9, 8, 7, 6])
        # Positionally: (a,x), (b,y), (a,y), (b,x) -> a: 1/2 positive,
        # b: 1/2 positive.
        results = categorical_lift(feature, target, positive_class="x")
        by_value = results_by_value(results)
        assert by_value["a"].support == 2
        assert by_value["a"].positive_count == 1
        assert by_value["b"].support == 2
        assert by_value["b"].positive_count == 1


class TestCategoricalValueMetrics:
    def make(self, **overrides):
        fields = dict(
            value="referral",
            support=100,
            positive_count=30,
            positive_rate=0.3,
            baseline_rate=0.2,
            lift=1.5,
        )
        return CategoricalValueMetrics(**{**fields, **overrides})

    def test_is_frozen(self):
        result = self.make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.lift = 2.0

    def test_rejects_positive_count_above_support(self):
        with pytest.raises(ValueError, match="positive_count"):
            self.make(support=10, positive_count=11)

    def test_rejects_negative_support(self):
        with pytest.raises(ValueError, match="support"):
            self.make(support=-1, positive_count=0)


class TestCategoricalLiftDegenerateInputs:
    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            categorical_lift(["a", "b"], ["x"], positive_class="x")

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            categorical_lift([], [], positive_class="x")

    def test_rejects_missing_feature_values(self):
        with pytest.raises(ValueError, match="missing values"):
            categorical_lift(["a", None], ["x", "y"], positive_class="x")

    def test_rejects_missing_target_values(self):
        with pytest.raises(ValueError, match="missing values"):
            categorical_lift(["a", "b"], ["x", None], positive_class="x")

    def test_rejects_single_target_class(self):
        with pytest.raises(ValueError, match="binary target"):
            categorical_lift(["a", "b"], ["x", "x"], positive_class="x")

    def test_rejects_three_target_classes(self):
        with pytest.raises(ValueError, match="binary target"):
            categorical_lift(["a", "b", "c"], ["x", "y", "z"], positive_class="x")

    def test_rejects_unobserved_positive_class(self):
        # target is binary ("x"/"y"), but positive_class "z" never occurs.
        # This is treated as an invalid argument (likely a typo), not the
        # statistical "zero baseline rate" edge case.
        with pytest.raises(ValueError, match="not one of the observed target classes"):
            categorical_lift(
                ["a", "a", "b", "b"], ["x", "y", "x", "y"], positive_class="z"
            )


def test_no_warnings_raised(recwarn):
    categorical_lift(FEATURE, TARGET, positive_class="positive")
    assert len(recwarn) == 0
