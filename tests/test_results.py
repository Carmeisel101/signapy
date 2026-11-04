import dataclasses
import datetime
import decimal
import enum
import json
import uuid

import numpy as np
import pandas as pd
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


def _demo_report():
    from signapy import discover

    df = pd.DataFrame({"c": list("aabbab" * 5), "y": [1, 0, 1, 1, 0, 0] * 5})
    return discover(df, "y", positive_class=1)


def test_report_to_dict_is_json_serializable():
    data = _demo_report().to_dict()
    json.dumps(data)
    assert data["target"] == "y"
    assert data["features"][0]["feature_type"] == "categorical"


def test_report_str_and_html():
    report = _demo_report()
    assert "feature" in str(report) and "c" in str(report)
    assert str(report).endswith("report.feature(name).values")
    assert "to_dict()" in str(report)
    assert "<table>" in report._repr_html_()


def test_report_html_embeds_mosaic_when_matplotlib_available():
    pytest.importorskip("matplotlib")
    html_out = _demo_report()._repr_html_()
    assert 'src="data:image/png;base64,' in html_out
    assert html_out.index("<img") < html_out.index("<table>")


def test_report_html_falls_back_to_table_without_matplotlib(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.split(".")[0] == "matplotlib":
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    html_out = _demo_report()._repr_html_()
    assert "<img" not in html_out
    assert "<table>" in html_out


def test_report_plot_renders_mosaic():
    import io

    pytest.importorskip("matplotlib")

    fig = _demo_report().plot()
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("Mosaic" in t for t in texts)
    assert "c" in texts
    assert any(t.endswith("×") for t in texts)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    assert buffer.getvalue().startswith(b"\x89PNG")


def test_report_plot_handles_featureless_rows_and_empty_report():
    pytest.importorskip("matplotlib")

    continuous = make_feature(
        feature="age",
        feature_type=FeatureType.CONTINUOUS,
        effect_size_method="point_biserial",
        values=None,
    )
    texts = [
        t.get_text() for t in DiscoveryReport("y", (continuous,)).plot().axes[0].texts
    ]
    assert any("no value-level breakdown" in t for t in texts)
    with pytest.raises(ValueError, match="no features"):
        DiscoveryReport("y").plot()


class _Color(enum.Enum):
    RED = "red"


class _ArgItem:
    """Hashable label whose unrelated ``item`` method needs an argument."""

    def item(self, key):
        return key

    def __str__(self):
        return "arg-item"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("plain", "plain"),
        (3, 3),
        (2.5, 2.5),
        (True, True),
        (np.int64(7), 7),
        (np.bool_(True), True),
        (np.float32(0.5), 0.5),
        (pd.Timestamp("2024-01-02 03:04:05"), "2024-01-02T03:04:05"),
        (datetime.date(2024, 1, 2), "2024-01-02"),
        (
            np.datetime64("2024-01-02T03:04:05.123456789", "ns"),
            "2024-01-02T03:04:05.123456789",
        ),
        (np.datetime64("2024-01-02", "D"), "2024-01-02T00:00:00"),
        (pd.Timedelta(hours=1, minutes=30), "P0DT1H30M0S"),
        (np.timedelta64(90, "s"), "P0DT0H1M30S"),
        (decimal.Decimal("1.50"), "1.50"),
        (uuid.UUID(int=1), "00000000-0000-0000-0000-000000000001"),
        ((1, np.int64(2)), [1, 2]),
        (_Color.RED, "red"),
        (_ArgItem(), "arg-item"),
    ],
)
def test_to_dict_encodes_category_labels_as_json(label, expected):
    feature = make_feature(values=(make_value(value=label),))
    data = DiscoveryReport("y", (feature,)).to_dict()
    encoded = data["features"][0]["values"][0]["value"]
    assert encoded == expected
    assert json.loads(json.dumps(data))["features"][0]["values"][0]["value"] == expected


@pytest.mark.parametrize("dtype", ["longdouble", "clongdouble"])
def test_to_dict_handles_extended_precision_numpy_scalars(dtype):
    label = getattr(np, dtype)(1.5)
    feature = make_feature(values=(make_value(value=label),))
    json.dumps(DiscoveryReport("y", (feature,)).to_dict())


def _strict_json(data):
    return json.loads(json.dumps(data, allow_nan=False))


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        (np.float64("inf"), "inf"),
        (np.float32("-inf"), "-inf"),
        (float("nan"), "nan"),
        (np.timedelta64(1, "M"), "1 months"),
        (np.timedelta64(2, "Y"), "2 years"),
        ((float("inf"), 1), ["inf", 1]),
        (complex(1, 2), "(1+2j)"),
    ],
)
def test_to_dict_labels_are_strict_json_for_edge_values(label, expected):
    feature = make_feature(values=(make_value(value=label),))
    data = DiscoveryReport("y", (feature,)).to_dict()
    assert _strict_json(data)["features"][0]["values"][0]["value"] == expected


def test_to_dict_numeric_fields_use_null_for_non_finite_floats():
    value = make_value(lift=float("inf"))
    feature = make_feature(
        effect_size=float("nan"),
        values=(value,),
        details={"statistic": float("inf"), "dof": 2},
    )
    data = _strict_json(DiscoveryReport("y", (feature,)).to_dict())
    entry = data["features"][0]
    assert entry["effect_size"] is None
    assert entry["values"][0]["lift"] is None
    assert entry["details"] == {"statistic": None, "dof": 2}


def test_plot_shows_long_feature_names_in_full():
    pytest.importorskip("matplotlib")
    names = ["customer_segment_alpha", "customer_segment_beta"]
    features = tuple(
        make_feature(feature=name, values=(make_value(),)) for name in names
    )
    texts = [
        t.get_text().replace("\n", "")
        for t in DiscoveryReport("y", features).plot().axes[0].texts
    ]
    for name in names:
        assert name in texts
    assert not any("…" in t for t in texts if t.startswith("customer"))


def _mosaic(names):
    features = tuple(
        make_feature(feature=name, values=(make_value(),)) for name in names
    )
    return DiscoveryReport("y", features).plot()


ODD_NAMES = [
    "segment ",
    " segment",
    "a\nb\nc\nd",
    "tab\tname",
    "x" * 120,
    "a_very_long_column_name_that_wraps_over_many_separate_display_lines_here",
    "",
    "   ",
]


@pytest.mark.parametrize("name", ODD_NAMES)
def test_display_and_wrap_never_drop_characters(name):
    from signapy.results._plot import _display, _wrap_name

    shown = _display(name)
    assert "\n" not in shown
    assert "".join(_wrap_name(shown)) == shown


def test_display_makes_invisible_differences_visible():
    from signapy.results._plot import _display

    assert _display("segment ") == "segment␣"
    assert _display(" segment") == "␣segment"
    assert _display("a\nb") == "a\\nb"
    assert _display("") == "(empty)"
    assert len({_display(n) for n in ["segment", "segment ", " segment"]}) == 3


@pytest.mark.parametrize("name", ODD_NAMES)
def test_plot_renders_odd_feature_names(name):
    import io

    pytest.importorskip("matplotlib")
    fig = _mosaic([name, "other"])
    fig.savefig(io.BytesIO(), format="png")


def test_plot_rows_grow_so_long_names_never_overlap_following_lines():
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = _mosaic(["x" * 100, "short"])
    renderer = FigureCanvasAgg(fig).get_renderer()
    texts = fig.axes[0].texts
    name_text = next(t for t in texts if t.get_text().startswith("xxxx"))
    effect_text = next(t for t in texts if t.get_text().startswith("V = "))
    assert name_text.get_window_extent(renderer).y0 >= (
        effect_text.get_window_extent(renderer).y1
    )
