import dataclasses

import pytest

from signapy.profiling import FeatureType, TargetType
from signapy.results import DiscoveryReport, FeatureResult, ValueResult


def make_value(**overrides) -> ValueResult:
    fields = dict(
        value="referral",
        support=100,
        target_count=40,
        target_rate=0.4,
        baseline_rate=0.2,
        lift=2.0,
    )
    return ValueResult(**{**fields, **overrides})


def make_feature(**overrides) -> FeatureResult:
    fields = dict(
        feature="channel",
        feature_type=FeatureType.CATEGORICAL,
        target_type=TargetType.BINARY,
        effect_size_method="cramers_v",
        effect_size=0.31,
        n=500,
        missing_rate=0.0,
        test_method="chi_square",
        p_value=0.0004,
    )
    return FeatureResult(**{**fields, **overrides})


class TestValueResult:
    def test_is_immutable(self):
        result = make_value()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.lift = 3.0

    def test_rejects_target_count_above_support(self):
        with pytest.raises(ValueError, match="cannot exceed support"):
            make_value(support=10, target_count=11)

    @pytest.mark.parametrize("field_name", ["support", "target_count"])
    def test_rejects_negative_counts(self, field_name):
        with pytest.raises(ValueError, match=field_name):
            make_value(**{field_name: -1})

    @pytest.mark.parametrize("field_name", ["target_rate", "baseline_rate"])
    def test_rejects_rates_outside_unit_interval(self, field_name):
        with pytest.raises(ValueError, match=field_name):
            make_value(**{field_name: 1.5})


class TestFeatureResult:
    def test_values_default_to_not_performed(self):
        assert make_feature().values is None

    def test_holds_value_results(self):
        values = (make_value(value="referral"), make_value(value="paid", lift=0.7))
        result = make_feature(values=values)
        assert [v.value for v in result.values] == ["referral", "paid"]

    def test_test_fields_are_optional(self):
        result = make_feature(test_method=None, p_value=None)
        assert result.p_value is None

    def test_rejects_p_value_without_test_method(self):
        with pytest.raises(ValueError, match="test_method"):
            make_feature(test_method=None, p_value=0.01)

    @pytest.mark.parametrize(
        ("field_name", "bad"), [("p_value", 1.1), ("missing_rate", -0.1)]
    )
    def test_rejects_invalid_probabilities(self, field_name, bad):
        with pytest.raises(ValueError, match=field_name):
            make_feature(**{field_name: bad})

    def test_rejects_negative_n(self):
        with pytest.raises(ValueError, match="n must be"):
            make_feature(n=-1)


class TestDiscoveryReport:
    def test_feature_lookup_by_name(self):
        channel = make_feature(feature="channel")
        region = make_feature(feature="region")
        report = DiscoveryReport(target="label", features=(channel, region))
        assert report.feature("region") is region

    def test_feature_lookup_missing_raises_key_error(self):
        report = DiscoveryReport(target="label", features=(make_feature(),))
        with pytest.raises(KeyError, match="nope"):
            report.feature("nope")

    def test_rejects_duplicate_features(self):
        with pytest.raises(ValueError, match="duplicate"):
            DiscoveryReport(target="label", features=(make_feature(), make_feature()))

    def test_rejects_target_as_feature(self):
        with pytest.raises(ValueError, match="target"):
            DiscoveryReport(target="channel", features=(make_feature(),))
