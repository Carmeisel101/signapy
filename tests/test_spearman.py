import dataclasses
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from signapy.metrics.association import SpearmanResult, spearman_rho

# Three ordinal levels (low=0, medium=1, high=2), unequal group sizes
# (10/10/10 here, but deliberately unequal elsewhere below), increasing
# positive rate per level: low 2/10, medium 5/10, high 8/10.
INCREASING_FEATURE = [0] * 10 + [1] * 10 + [2] * 10
INCREASING_TARGET = (
    ["neg"] * 8 + ["pos"] * 2 + ["neg"] * 5 + ["pos"] * 5 + ["neg"] * 2 + ["pos"] * 8
)

# Same feature, decreasing positive rate per level.
DECREASING_TARGET = (
    ["pos"] * 8 + ["neg"] * 2 + ["pos"] * 5 + ["neg"] * 5 + ["pos"] * 2 + ["neg"] * 8
)

# No relationship: target alternates regardless of level.
NULL_FEATURE = [0, 1, 2] * 10
NULL_TARGET = (["a", "b"] * 15)[:30]

# Non-monotonic: "medium" has the highest positive rate, "low" and "high"
# are both low and symmetric -> Spearman should read as ~0 despite a real
# per-level relationship (the documented caveat).
NON_MONOTONIC_FEATURE = [0] * 10 + [1] * 10 + [2] * 10
NON_MONOTONIC_TARGET = (
    ["neg"] * 8 + ["pos"] * 2 + ["neg"] * 2 + ["pos"] * 8 + ["neg"] * 8 + ["pos"] * 2
)


class TestSpearmanKnownValues:
    def test_increasing_relationship(self):
        result = spearman_rho(
            INCREASING_FEATURE, INCREASING_TARGET, positive_class="pos"
        )
        assert result.coefficient == pytest.approx(0.4898979485566356)
        assert result.p_value == pytest.approx(0.0059963724801474)
        assert result.n == 30
        assert result.n_levels == 3

    def test_decreasing_relationship(self):
        result = spearman_rho(
            INCREASING_FEATURE, DECREASING_TARGET, positive_class="pos"
        )
        assert result.coefficient == pytest.approx(-0.4898979485566356)
        assert result.p_value == pytest.approx(0.0059963724801474)

    def test_null_relationship(self):
        result = spearman_rho(NULL_FEATURE, NULL_TARGET, positive_class="b")
        assert result.coefficient == pytest.approx(0.0, abs=1e-12)
        assert result.p_value == pytest.approx(1.0)

    def test_non_monotonic_relationship_reads_as_near_zero(self):
        # Documents the caveat in spearman_rho's docstring: a real
        # per-level relationship (medium peaks) can still read as ~0 here,
        # because Spearman only detects monotonic trends.
        result = spearman_rho(
            NON_MONOTONIC_FEATURE, NON_MONOTONIC_TARGET, positive_class="pos"
        )
        assert result.coefficient == pytest.approx(0.0, abs=1e-12)

    def test_tied_levels(self):
        result = spearman_rho(
            [1, 1, 2, 2, 3, 3], ["a", "b", "a", "b", "b", "b"], positive_class="b"
        )
        assert result.coefficient == pytest.approx(0.4330127018922194)
        assert result.p_value == pytest.approx(0.3910758879640664)

    def test_matches_scipy_spearmanr_directly(self):
        is_positive = np.array([t == "pos" for t in INCREASING_TARGET])
        expected = scipy.stats.spearmanr(
            np.array(INCREASING_FEATURE, dtype=float), is_positive
        )
        result = spearman_rho(
            INCREASING_FEATURE, INCREASING_TARGET, positive_class="pos"
        )
        assert result.coefficient == pytest.approx(expected.statistic)
        assert result.p_value == pytest.approx(expected.pvalue)

    def test_diverges_from_naive_pearson_on_codes_with_unequal_groups(self):
        # Regression/documentation test for the "don't shortcut via
        # point_biserial on integer codes" warning in spearman_rho's
        # docstring: with unevenly-sized groups, true tie-aware Spearman
        # and naive Pearson-on-codes give different numbers.
        feature = [0] * 10 + [1] * 10 + [2] * 180
        target = (
            ["neg"] * 8
            + ["pos"] * 2
            + ["neg"] * 5
            + ["pos"] * 5
            + ["neg"] * 30
            + ["pos"] * 150
        )
        is_positive = np.array([t == "pos" for t in target], dtype=float)
        naive_pearson = np.corrcoef(np.array(feature, dtype=float), is_positive)[0, 1]
        result = spearman_rho(feature, target, positive_class="pos")
        assert result.coefficient != pytest.approx(naive_pearson, abs=1e-6)


class TestSpearmanInvariants:
    def test_row_order_permutation_leaves_result_unchanged(self):
        rng = np.random.default_rng(0)
        permutation = rng.permutation(len(INCREASING_FEATURE))
        permuted_feature = [INCREASING_FEATURE[i] for i in permutation]
        permuted_target = [INCREASING_TARGET[i] for i in permutation]

        baseline = spearman_rho(
            INCREASING_FEATURE, INCREASING_TARGET, positive_class="pos"
        )
        permuted = spearman_rho(permuted_feature, permuted_target, positive_class="pos")

        assert permuted.coefficient == pytest.approx(baseline.coefficient)
        assert permuted.p_value == pytest.approx(baseline.p_value)

    def test_positive_class_reversal_flips_sign_exactly(self):
        pos = spearman_rho(INCREASING_FEATURE, INCREASING_TARGET, positive_class="pos")
        neg = spearman_rho(INCREASING_FEATURE, INCREASING_TARGET, positive_class="neg")
        assert neg.coefficient == pytest.approx(-pos.coefficient)
        assert neg.p_value == pytest.approx(pos.p_value)

    def test_monotonic_rescaling_of_codes_preserves_coefficient(self):
        # Spearman only uses rank order, so any strictly increasing
        # relabeling of the levels (not just 0,1,2) gives the same result.
        baseline = spearman_rho(
            INCREASING_FEATURE, INCREASING_TARGET, positive_class="pos"
        )
        level_map = {0: 10, 1: 250, 2: 9000}
        rescaled_feature = [level_map[level] for level in INCREASING_FEATURE]
        rescaled = spearman_rho(
            rescaled_feature, INCREASING_TARGET, positive_class="pos"
        )
        assert rescaled.coefficient == pytest.approx(baseline.coefficient)

    def test_coefficient_bounded(self):
        result = spearman_rho(
            [0, 1, 2, 3, 4, 5], ["a", "a", "a", "b", "b", "b"], positive_class="b"
        )
        assert -1.0 <= result.coefficient <= 1.0


class TestSpearmanAcceptedInputTypes:
    def test_accepts_python_lists(self):
        result = spearman_rho(
            INCREASING_FEATURE, INCREASING_TARGET, positive_class="pos"
        )
        assert result.coefficient == pytest.approx(0.4898979485566356)

    def test_accepts_numpy_arrays(self):
        result = spearman_rho(
            np.array(INCREASING_FEATURE, dtype=float),
            np.array(INCREASING_TARGET),
            positive_class="pos",
        )
        assert result.coefficient == pytest.approx(0.4898979485566356)

    def test_accepts_pandas_series(self):
        result = spearman_rho(
            pd.Series(INCREASING_FEATURE),
            pd.Series(INCREASING_TARGET),
            positive_class="pos",
        )
        assert result.coefficient == pytest.approx(0.4898979485566356)

    def test_accepts_ordered_categorical_codes(self):
        # The realistic path: .cat.codes from an ordered pandas Categorical.
        categorical = pd.Categorical(
            ["low"] * 10 + ["medium"] * 10 + ["high"] * 10,
            categories=["low", "medium", "high"],
            ordered=True,
        )
        result = spearman_rho(
            categorical.codes, INCREASING_TARGET, positive_class="pos"
        )
        assert result.coefficient == pytest.approx(0.4898979485566356)

    @pytest.mark.parametrize(
        "feature",
        [
            [2**53, 2**53 + 1, 2**53 + 2],
            [
                Decimal("1.0000000000000000001"),
                Decimal("1.0000000000000000002"),
                Decimal("1.0000000000000000003"),
            ],
        ],
    )
    def test_preserves_distinct_high_precision_levels(self, feature):
        result = spearman_rho(feature, [0, 0, 1], positive_class=1)

        assert result.coefficient == pytest.approx(np.sqrt(3) / 2)
        assert result.n_levels == 3


class TestSpearmanInvalidInputs:
    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            spearman_rho([], [], positive_class="a")

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            spearman_rho([1, 2, 3], [1, 2], positive_class=1)

    def test_rejects_non_one_dimensional_feature(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            spearman_rho([[1, 2], [3, 4]], ["a", "b"], positive_class="a")

    def test_rejects_missing_feature_values(self):
        with pytest.raises(ValueError, match="missing feature values"):
            spearman_rho([0, 1, None], ["a", "b", "a"], positive_class="a")

    def test_rejects_missing_target_values(self):
        with pytest.raises(ValueError, match="missing target values"):
            spearman_rho([0, 1, 2], ["a", None, "a"], positive_class="a")

    def test_rejects_non_numeric_feature(self):
        with pytest.raises(ValueError, match="must be numeric"):
            spearman_rho(["x", "y", "z"], ["a", "b", "a"], positive_class="a")

    def test_rejects_boolean_feature(self):
        with pytest.raises(ValueError, match="must not be boolean"):
            spearman_rho([True, False, True], ["a", "b", "a"], positive_class="a")

    def test_rejects_boolean_feature_stored_as_object_dtype(self):
        feature = np.array([True, False, True], dtype=object)
        with pytest.raises(ValueError, match="must not be boolean"):
            spearman_rho(feature, ["a", "b", "a"], positive_class="a")

    def test_rejects_datetime_feature(self):
        feature = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
        )
        with pytest.raises(ValueError, match="datetime"):
            spearman_rho(feature, ["a", "b", "a", "b"], positive_class="a")

    def test_rejects_complex_feature(self):
        feature = np.array([1 + 2j, 2 + 1j, 3 + 0j, 10 + 5j])
        with pytest.raises(ValueError, match="real-valued"):
            spearman_rho(feature, ["a", "b", "a", "b"], positive_class="a")

    def test_rejects_infinite_feature_values(self):
        with pytest.raises(ValueError, match="infinite"):
            spearman_rho([0, 1, float("inf")], ["a", "b", "a"], positive_class="a")

    def test_rejects_single_target_class(self):
        with pytest.raises(ValueError, match="binary target"):
            spearman_rho([0, 1, 2], ["a", "a", "a"], positive_class="a")

    def test_rejects_three_target_classes(self):
        with pytest.raises(ValueError, match="binary target"):
            spearman_rho([0, 1, 2], ["a", "b", "c"], positive_class="a")

    def test_rejects_unobserved_positive_class(self):
        with pytest.raises(ValueError, match="not one of the observed"):
            spearman_rho([0, 1, 2], ["a", "b", "a"], positive_class="z")

    def test_rejects_constant_feature(self):
        # Regression test for the scipy ConstantInputWarning this guard
        # exists to pre-empt (otherwise an error under this project's
        # pytest filterwarnings=error config), and matches the "fewer than
        # two observed levels" rejection requirement.
        with pytest.raises(ValueError, match="fewer than 2 distinct levels"):
            spearman_rho([5, 5, 5, 5], ["a", "b", "a", "b"], positive_class="a")

    def test_rejects_too_few_observations(self):
        # Regression test: scipy.stats.spearmanr(n=2) returns a NaN
        # p-value silently, with no warning and no error, rather than
        # raising. This guard exists so spearman_rho never does that.
        with pytest.raises(ValueError, match="at least 3 observations"):
            spearman_rho([0, 1], ["a", "b"], positive_class="a")


class TestSpearmanResult:
    def make(self, **overrides):
        fields = dict(coefficient=0.5, p_value=0.01, n=30, n_levels=3)
        return SpearmanResult(**{**fields, **overrides})

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

    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError, match="n must be positive"):
            self.make(n=0)

    def test_rejects_non_positive_n_levels(self):
        with pytest.raises(ValueError, match="n_levels must be positive"):
            self.make(n_levels=0)

    def test_rejects_n_levels_exceeding_n(self):
        with pytest.raises(ValueError, match="n_levels"):
            self.make(n=3, n_levels=5)


def test_no_warnings_raised(recwarn):
    spearman_rho(INCREASING_FEATURE, INCREASING_TARGET, positive_class="pos")
    spearman_rho(NULL_FEATURE, NULL_TARGET, positive_class="b")
    assert len(recwarn) == 0
