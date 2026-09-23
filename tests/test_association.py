import numpy as np
import pandas as pd
import pytest

from signapy.metrics.association import cramers_v

# Referral vs. paid acquisition channel, converted vs. not.
#              negative  positive
# referral          70        30
# paid              90        10
TWO_CATEGORY_TABLE = [[70, 30], [90, 10]]

# Three categories with non-uniform outcomes.
THREE_CATEGORY_TABLE = [[50, 150], [120, 80], [200, 50]]

# Rows are proportional to each other -> no association.
INDEPENDENT_TABLE = [[50, 50], [50, 50]]

# No overlap between categories and outcomes -> perfect association.
PERFECT_TABLE = [[100, 0], [0, 100]]


class TestCramersVKnownValues:
    def test_two_category_table_bias_corrected(self):
        assert cramers_v(TWO_CATEGORY_TABLE) == pytest.approx(0.24034381938205007)

    def test_two_category_table_standard(self):
        assert cramers_v(TWO_CATEGORY_TABLE, bias_correction=False) == pytest.approx(
            0.25
        )

    def test_three_category_table_bias_corrected(self):
        assert cramers_v(THREE_CATEGORY_TABLE) == pytest.approx(0.4580809823608554)

    def test_three_category_table_standard(self):
        assert cramers_v(THREE_CATEGORY_TABLE, bias_correction=False) == pytest.approx(
            0.46108190714505926
        )

    def test_independent_table_is_zero(self):
        assert cramers_v(INDEPENDENT_TABLE) == pytest.approx(0.0, abs=1e-12)
        assert cramers_v(INDEPENDENT_TABLE, bias_correction=False) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_perfect_association_table_is_one(self):
        assert cramers_v(PERFECT_TABLE) == pytest.approx(1.0)
        assert cramers_v(PERFECT_TABLE, bias_correction=False) == pytest.approx(1.0)

    def test_bias_correction_reduces_v_for_small_samples(self):
        corrected = cramers_v(TWO_CATEGORY_TABLE, bias_correction=True)
        standard = cramers_v(TWO_CATEGORY_TABLE, bias_correction=False)
        assert corrected < standard


class TestCramersVAcceptedInputTypes:
    @pytest.mark.parametrize(
        "as_type",
        [
            lambda t: t,
            lambda t: np.asarray(t),
            lambda t: pd.DataFrame(t),
        ],
        ids=["nested_list", "numpy_array", "dataframe"],
    )
    def test_accepts_array_like_inputs(self, as_type):
        table = as_type(TWO_CATEGORY_TABLE)
        assert cramers_v(table) == pytest.approx(0.24034381938205007)


class TestCramersVDegenerateInputs:
    def test_rejects_empty_table(self):
        with pytest.raises(ValueError, match="empty"):
            cramers_v([])

    def test_rejects_single_feature_category(self):
        with pytest.raises(ValueError, match="at least 2 categories"):
            cramers_v([[70, 30]])

    def test_rejects_single_target_class(self):
        with pytest.raises(ValueError, match="at least 2 categories"):
            cramers_v([[70], [90]])

    def test_rejects_negative_values(self):
        with pytest.raises(ValueError, match="negative"):
            cramers_v([[70, -30], [90, 10]])

    def test_rejects_non_finite_values(self):
        with pytest.raises(ValueError, match="finite"):
            cramers_v([[70, float("nan")], [90, 10]])

    def test_rejects_zero_row_sum(self):
        with pytest.raises(ValueError, match="sums to zero"):
            cramers_v([[0, 0], [90, 10]])

    def test_rejects_zero_column_sum(self):
        with pytest.raises(ValueError, match="sums to zero"):
            cramers_v([[70, 0], [90, 0]])

    def test_rejects_fractional_values(self):
        # Regression test: fractional entries corrupt the bias correction,
        # which treats the table sum as a literal observation count. For
        # this independent table, an unguarded implementation returns
        # ~0.79 instead of 0.0.
        with pytest.raises(ValueError, match="integer-valued counts"):
            cramers_v([[0.1, 0.1], [0.1, 0.1]])

    def test_rejects_fractional_values_standard(self):
        with pytest.raises(ValueError, match="integer-valued counts"):
            cramers_v([[0.1, 0.1], [0.1, 0.1]], bias_correction=False)

    def test_bias_corrected_v_undefined_for_sparse_large_table(self):
        # A 5x5 table with only one observation per cell collapses the
        # bias-corrected dimensions to <= 1 category.
        table = np.eye(5)
        with pytest.raises(ValueError, match="undefined"):
            cramers_v(table, bias_correction=True)
        # The standard estimator remains defined for the same table.
        cramers_v(table, bias_correction=False)


def test_no_warnings_raised(recwarn):
    cramers_v(TWO_CATEGORY_TABLE)
    cramers_v(THREE_CATEGORY_TABLE)
    cramers_v(PERFECT_TABLE)
    assert len(recwarn) == 0
