import dataclasses

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from signapy.metrics.significance import ChiSquareResult, chi_square

TWO_CATEGORY_TABLE = [[70, 30], [90, 10]]
THREE_CATEGORY_TABLE = [[50, 150], [120, 80], [200, 50]]
INDEPENDENT_TABLE = [[50, 50], [50, 50]]
PERFECT_TABLE = [[100, 0], [0, 100]]


class TestChiSquareMatchesScipy:
    @pytest.mark.parametrize(
        "table",
        [TWO_CATEGORY_TABLE, THREE_CATEGORY_TABLE, INDEPENDENT_TABLE, PERFECT_TABLE],
        ids=["two_category", "three_category", "independent", "perfect"],
    )
    def test_matches_scipy_chi2_contingency(self, table):
        result = chi_square(table)
        expected_statistic, expected_p, expected_dof, _ = scipy.stats.chi2_contingency(
            table, correction=False
        )
        assert result.statistic == pytest.approx(expected_statistic)
        assert result.p_value == pytest.approx(expected_p)
        assert result.dof == expected_dof

    def test_two_category_known_values(self):
        result = chi_square(TWO_CATEGORY_TABLE)
        assert result.statistic == pytest.approx(12.5)
        assert result.p_value == pytest.approx(0.00040695201744495935)
        assert result.dof == 1

    def test_independent_table_statistic_is_zero(self):
        result = chi_square(INDEPENDENT_TABLE)
        assert result.statistic == pytest.approx(0.0, abs=1e-12)
        assert result.p_value == pytest.approx(1.0)


class TestChiSquareResult:
    def test_is_frozen(self):
        result = chi_square(TWO_CATEGORY_TABLE)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.statistic = 0.0

    def test_fields(self):
        result = ChiSquareResult(statistic=1.0, p_value=0.5, dof=1)
        assert result.statistic == 1.0
        assert result.p_value == 0.5
        assert result.dof == 1


class TestChiSquareAcceptedInputTypes:
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
        result = chi_square(as_type(TWO_CATEGORY_TABLE))
        assert result.statistic == pytest.approx(12.5)


class TestChiSquareDegenerateInputs:
    def test_rejects_empty_table(self):
        with pytest.raises(ValueError, match="empty"):
            chi_square([])

    def test_rejects_single_feature_category(self):
        with pytest.raises(ValueError, match="at least 2 categories"):
            chi_square([[70, 30]])

    def test_rejects_single_target_class(self):
        with pytest.raises(ValueError, match="at least 2 categories"):
            chi_square([[70], [90]])

    def test_rejects_negative_values(self):
        with pytest.raises(ValueError, match="negative"):
            chi_square([[70, -30], [90, 10]])

    def test_rejects_non_finite_values(self):
        with pytest.raises(ValueError, match="finite"):
            chi_square([[70, float("inf")], [90, 10]])

    def test_rejects_zero_row_sum(self):
        with pytest.raises(ValueError, match="sums to zero"):
            chi_square([[0, 0], [90, 10]])

    def test_rejects_zero_column_sum(self):
        with pytest.raises(ValueError, match="sums to zero"):
            chi_square([[70, 0], [90, 0]])

    def test_rejects_fractional_values(self):
        with pytest.raises(ValueError, match="integer-valued counts"):
            chi_square([[0.1, 0.1], [0.1, 0.1]])


def test_no_warnings_raised(recwarn):
    chi_square(TWO_CATEGORY_TABLE)
    chi_square(THREE_CATEGORY_TABLE)
    chi_square(PERFECT_TABLE)
    assert len(recwarn) == 0
