import dataclasses
import datetime
import decimal
import enum
import itertools
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


class _RaisingStr:
    def __str__(self):
        raise RuntimeError("boom")


class _LyingStr(str):
    def __str__(self):
        return "lie"


class _LyingTimestamp(datetime.datetime):
    def isoformat(self, *args, **kwargs):
        return "lie"


def test_to_dict_never_raises_on_hostile_labels():
    labels = [
        _RaisingStr(),
        _LyingStr("truth"),
        _LyingTimestamp(2024, 1, 2, 3, 4, 5),
    ]
    feature = make_feature(values=tuple(make_value(value=label) for label in labels))
    values = _strict_json(DiscoveryReport("y", (feature,)).to_dict())["features"][0][
        "values"
    ]
    assert values[0]["value"].endswith("._RaisingStr>")
    assert values[1]["value"] == "truth"
    assert values[2]["value"] == "2024-01-02T03:04:05"


def test_to_dict_normalizes_numpy_counts():
    value = make_value(support=np.int64(100), target_count=np.int64(40))
    feature = make_feature(n=np.int64(500), values=(value,))
    entry = _strict_json(DiscoveryReport("y", (feature,)).to_dict())["features"][0]
    assert entry["n"] == 500
    assert entry["values"][0]["support"] == 100
    assert entry["values"][0]["target_count"] == 40


def test_plot_rare_tiles_stay_inside_their_allocation():
    pytest.importorskip("matplotlib")
    from matplotlib.patches import FancyBboxPatch

    from signapy.results._plot import _HEADER_H, _LABEL_W, _MARGIN, _WIDTH

    supports = [9970] + [1] * 30
    values = tuple(
        make_value(value=f"v{i}", support=s, target_count=0, target_rate=0.0, lift=0.0)
        for i, s in enumerate(supports)
    )
    fig = DiscoveryReport("y", (make_feature(values=values),)).plot()
    tile_x0 = _MARGIN + _LABEL_W + 0.15
    tiles = sorted(
        (p.get_x(), p.get_x() + p.get_width())
        for p in fig.axes[0].patches
        if isinstance(p, FancyBboxPatch)
        and abs(p.get_y() - _HEADER_H) < 1e-9
        and p.get_x() >= tile_x0 - 1e-9
    )
    assert len(tiles) == len(supports)
    assert tiles[-1][1] <= _WIDTH - _MARGIN + 1e-9
    for (_, right), (left, _) in itertools.pairwise(tiles):
        assert right <= left + 1e-9


def test_plot_handles_zero_total_support():
    pytest.importorskip("matplotlib")
    zero = make_value(support=0, target_count=0, target_rate=0.0, lift=0.0)
    texts = [
        t.get_text()
        for t in DiscoveryReport("y", (make_feature(values=(zero,)),))
        .plot()
        .axes[0]
        .texts
    ]
    assert any("no value-level breakdown" in t for t in texts)


def test_plot_max_features_keeps_strongest_and_says_so():
    pytest.importorskip("matplotlib")
    features = tuple(
        make_feature(feature=f"f{i}", effect_size=i / 10, values=(make_value(),))
        for i in range(1, 6)
    )
    fig = DiscoveryReport("y", features).plot(max_features=2)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "f5" in texts and "f4" in texts and "f1" not in texts
    assert any("top 2 of 5" in t for t in texts)
    with pytest.raises(ValueError, match="max_features"):
        DiscoveryReport("y", features).plot(max_features=0)


def test_html_repr_caps_features_and_survives_drawing_errors(monkeypatch):
    pytest.importorskip("matplotlib")
    features = tuple(
        make_feature(feature=f"f{i}", effect_size=0.5, values=(make_value(),))
        for i in range(60)
    )
    report = DiscoveryReport("y", features)
    assert "<img" in report._repr_html_()

    def boom(self, max_features=None):
        raise ValueError("cannot draw")

    monkeypatch.setattr(DiscoveryReport, "plot", boom)
    html_out = report._repr_html_()
    assert "<img" not in html_out and "<table>" in html_out


def test_str_escapes_control_characters_in_feature_names():
    feature = make_feature(feature="a\nb", values=(make_value(),))
    text = str(DiscoveryReport("y", (feature,)))
    assert "a\\nb" in text
    assert text.count("\n") == 3


def test_plot_groups_rows_by_measure_then_strength():
    pytest.importorskip("matplotlib")
    specs = [
        ("cat_strong", "cramers_v", 0.3),
        ("num_strong", "point_biserial", 0.5),
        ("cat_weak", "cramers_v", 0.2),
        ("num_weak", "point_biserial", 0.1),
    ]
    features = tuple(
        make_feature(
            feature=name,
            effect_size_method=method,
            effect_size=effect,
            values=(make_value(),),
        )
        for name, method, effect in specs
    )
    texts = [t.get_text() for t in DiscoveryReport("y", features).plot().axes[0].texts]
    order = [t for t in texts if t in {name for name, _, _ in specs}]
    assert order == ["num_strong", "num_weak", "cat_strong", "cat_weak"]


def test_plot_cuts_long_value_labels_but_to_dict_keeps_them():
    pytest.importorskip("matplotlib")
    from signapy.results._plot import _MAX_LABEL_CHARS

    label = "a_really_long_category_label_" * 3
    feature = make_feature(values=(make_value(value=label),))
    report = DiscoveryReport("y", (feature,))
    drawn = [
        t.get_text() for t in report.plot().axes[0].texts if "really" in t.get_text()
    ]
    assert drawn
    assert all(len(t) <= _MAX_LABEL_CHARS and t.endswith("…") for t in drawn)
    assert report.to_dict()["features"][0]["values"][0]["value"] == label


@pytest.mark.parametrize("names", [["short"], ["short", "x" * 100], ["a", "b", "c"]])
def test_plot_colorbar_sits_below_the_last_row(names):
    pytest.importorskip("matplotlib")
    from matplotlib.patches import FancyBboxPatch

    features = tuple(
        make_feature(feature=n, effect_size=0.5 - i / 10, values=(make_value(),))
        for i, n in enumerate(names)
    )
    fig = DiscoveryReport("y", features).plot()
    height = fig.get_figheight()
    rows_bottom = max(
        p.get_y() + p.get_height()
        for p in fig.axes[0].patches
        if isinstance(p, FancyBboxPatch) and p.get_height() < height - 1
    )
    colorbar_top = (1 - fig.axes[1].get_position().y1) * height
    assert rows_bottom <= colorbar_top


class _HostileMeta(type):
    def __getattribute__(cls, name):
        if name in {"__module__", "__qualname__"}:
            raise RuntimeError("no metadata for you")
        return super().__getattribute__(name)


class _HostileLabel(metaclass=_HostileMeta):
    def __str__(self):
        raise RuntimeError("boom")

    def __hash__(self):
        return 1


def test_to_dict_survives_labels_with_hostile_metaclass():
    feature = make_feature(values=(make_value(value=_HostileLabel()),))
    data = _strict_json(DiscoveryReport("y", (feature,)).to_dict())
    assert data["features"][0]["values"][0]["value"] == "<unserializable>"


def test_plot_draws_dollar_signs_literally():
    pytest.importorskip("matplotlib")
    names = ["$revenue$", "cost $"]
    features = tuple(
        make_feature(feature=n, effect_size=0.5 - i / 10, values=(make_value(),))
        for i, n in enumerate(names)
    )
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = DiscoveryReport("y", features).plot()
    renderer = FigureCanvasAgg(fig).get_renderer()
    shown = [
        t.get_text()
        for t in fig.axes[0].texts
        if "revenue" in t.get_text() or "cost" in t.get_text()
    ]
    assert shown == [r"\$revenue\$", r"cost \$"]
    for text in fig.axes[0].texts:
        text.get_window_extent(renderer)  # would raise on a bad math parse


@pytest.mark.parametrize("name", ["W" * 13, "M" * 40, "WWWW_MMMM_WWWW_MMMM"])
def test_plot_wide_glyph_names_stay_inside_label_panel(name):
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from signapy.results._plot import _LABEL_W, _MARGIN

    fig = _mosaic([name, "other"])
    renderer = FigureCanvasAgg(fig).get_renderer()
    ax = fig.axes[0]
    panel_right_px = ax.transData.transform((_MARGIN + _LABEL_W, 0))[0]
    name_text = next(
        t for t in ax.texts if t.get_text().replace("\n", "").startswith(name[:2])
    )
    assert name_text.get_window_extent(renderer).x1 <= panel_right_px


def test_wrap_name_with_custom_width_predicate_never_drops_text():
    from signapy.results._plot import _wrap_name

    name = "wide_WWWWWWWWWWWWWWWW_tail"
    lines = _wrap_name(name, lambda s: sum(3 if c == "W" else 1 for c in s) <= 10)
    assert "".join(lines) == name


def _texts(fig):
    return [t.get_text() for t in fig.axes[0].texts]


def test_plot_lists_unlabelled_narrow_tiles_in_a_strip():
    pytest.importorskip("matplotlib")
    values = (
        make_value(value="common", support=9900, lift=0.9),
        make_value(value="rare_hot", support=60, target_count=40, lift=3.4),
        make_value(value="rare_mild", support=40, target_count=20, lift=1.1),
    )
    fig = DiscoveryReport("y", (make_feature(values=values),)).plot()
    texts = _texts(fig)
    assert any("Narrow tiles" in t for t in texts)
    strip = next(t for t in texts if "rare_hot" in t)
    assert "3.40×" in strip and "0.6%" in strip
    assert strip.index("rare_hot") < strip.index("rare_mild")


def test_plot_has_no_strip_when_every_tile_is_labelled():
    pytest.importorskip("matplotlib")
    values = (
        make_value(value="a", support=50, target_count=20),
        make_value(value="b", support=50, target_count=20),
    )
    fig = DiscoveryReport("y", (make_feature(values=values),)).plot()
    assert not any("Narrow tiles" in t for t in _texts(fig))


def test_plot_strip_is_capped_and_says_how_many_were_left_out():
    pytest.importorskip("matplotlib")
    from signapy.results._plot import _STRIP_MAX_ENTRIES

    values = (
        make_value(value="big", support=10000),
        *(
            make_value(value=f"r{i}", support=1, target_count=0, lift=1.5)
            for i in range(30)
        ),
    )
    fig = DiscoveryReport("y", (make_feature(values=values),)).plot()
    texts = _texts(fig)
    shown = sum(f"r{i}:" in t or f": r{i} " in t for t in texts for i in range(30))
    assert shown <= _STRIP_MAX_ENTRIES
    assert any(f"+{30 - _STRIP_MAX_ENTRIES} more" in t for t in texts)


def test_plot_mid_width_tiles_show_lift_and_support():
    pytest.importorskip("matplotlib")
    values = (
        make_value(value="a", support=92, target_count=30, lift=0.8),
        make_value(value="b", support=8, target_count=6, lift=2.0),
    )
    fig = DiscoveryReport("y", (make_feature(values=values),)).plot()
    texts = _texts(fig)
    assert "2.00×" in texts and "8.0%" in texts


def test_plot_strip_keeps_colorbar_below_everything():
    pytest.importorskip("matplotlib")
    values = (
        make_value(value="big", support=10000),
        *(make_value(value=f"r{i}", support=1, target_count=0) for i in range(5)),
    )
    fig = DiscoveryReport("y", (make_feature(values=values),)).plot()
    height = fig.get_figheight()
    colorbar_top = (1 - fig.axes[1].get_position().y1) * height
    strip_bottom = max(
        t.get_position()[1] for t in fig.axes[0].texts if "r0" in t.get_text()
    )
    assert strip_bottom < colorbar_top
