"""The mosaic: SignaPy's signature visual for a :class:`DiscoveryReport`.

One row per feature, ordered by strength of association. Within a row, one
rounded tile per value: tile *width* is the value's support (its share of the
feature's rows) and tile *colour* is its lift against the baseline target
rate. A feature's signal therefore reads at a glance as saturated tiles, and
thin, saturated tiles flag strong lift resting on little data.

Drawing happens in a coordinate system where one data unit is one inch, so
rounded corners stay round regardless of how the figure is scaled.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from signapy.results.models import DiscoveryReport, FeatureResult

# Short symbol for each effect-size method, shown beside the feature name.
_SYMBOLS = {"cramers_v": "V", "point_biserial": "r", "spearman_rho": "ρ"}

_INK = "#1a1f2b"
_MUTED = "#5b6475"
_PANEL = "#f3f5f8"
_NEUTRAL = "#eceff3"
_COLORS = [
    "#0b2a6b",
    "#2b7bba",
    "#a6d4f2",
    "#f6f7f9",
    "#fcb7a0",
    "#e8553f",
    "#8f0a1c",
]

_WIDTH = 11.0
_MARGIN = 0.3
_LABEL_W = 1.75
_ROW_H = 0.95
_ROW_GAP = 0.14
_TILE_GAP = 0.08
_HEADER_H = 1.3
_FOOTER_H = 1.5


def _wrap_name(name: str, width: int = 17) -> list[str]:
    """Split ``name`` into lines of at most ``width`` characters, never dropping text.

    Lines break after underscores and spaces where possible, and inside a
    token only when that token alone is longer than ``width``, so distinct
    names always stay distinguishable.
    """
    tokens = re.findall(r"[^_ ]+[_ ]?|[_ ]", name)
    lines: list[str] = []
    current = ""
    for token in tokens:
        while len(token) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(token[:width])
            token = token[width:]
        if len(current) + len(token) > width:
            lines.append(current)
            current = ""
        current += token
    if current:
        lines.append(current)
    return [line.rstrip() for line in lines] or [name]


def _symbol(feature: FeatureResult) -> str:
    return _SYMBOLS.get(feature.effect_size_method, "effect")


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 1, 1)] + "…"


def _baseline_text(features: list[FeatureResult]) -> str | None:
    rates = [v.baseline_rate for f in features for v in f.values or ()]
    if not rates:
        return None
    low, high = min(rates), max(rates)
    # Baselines differ slightly per feature when missing values drop
    # different rows; show the range rather than hide it.
    if high - low < 5e-4:
        return f"{(low + high) / 2:.3f}"
    return f"{low:.3f}–{high:.3f}"


def plot_mosaic(report: DiscoveryReport) -> Figure:
    """Render ``report`` as a mosaic and return the matplotlib ``Figure``."""
    try:
        import numpy as np
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        from matplotlib.figure import Figure
        from matplotlib.patches import FancyBboxPatch
    except ImportError as error:
        raise ImportError(
            "DiscoveryReport.plot() requires matplotlib; "
            "install it with `pip install signapy[plot]`"
        ) from error

    if not report.features:
        raise ValueError("report has no features to plot")

    features = sorted(report.features, key=lambda f: -abs(f.effect_size))
    lifts = [v.lift for f in features for v in f.values or () if math.isfinite(v.lift)]
    norm = TwoSlopeNorm(
        vcenter=1.0,
        vmin=min([0.4, *lifts]),
        vmax=max([1.6, *lifts]),
    )
    cmap = LinearSegmentedColormap.from_list("signapy_lift", _COLORS)

    height = _HEADER_H + len(features) * (_ROW_H + _ROW_GAP) + _FOOTER_H
    fig = Figure(figsize=(_WIDTH, height), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, _WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")

    def box(x, y, w, h, color, radius=0.1, edge="none", lw=0.0, z=1):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=color,
                edgecolor=edge,
                linewidth=lw,
                zorder=z,
            )
        )

    box(0.08, 0.08, _WIDTH - 0.16, height - 0.16, "white", 0.2, "#e3e6ec", 1.0, 0)

    # Header: title, subtitle, baseline badge.
    ax.text(
        _MARGIN + 0.1,
        0.5,
        f"Mosaic: Feature vs. {report.target} Lift",
        fontsize=18,
        fontweight="bold",
        color=_INK,
        va="center",
    )
    ax.text(
        _MARGIN + 0.1,
        0.92,
        "Each tile shows a value's lift (target rate ÷ baseline). "
        "Tile width = support (relative frequency).",
        fontsize=10,
        color=_MUTED,
        va="center",
    )
    baseline = _baseline_text(features)
    if baseline is not None:
        label = f"Baseline (target rate): {baseline}"
        badge_w = 0.12 * len(label) * 0.62 + 0.5
        box(_WIDTH - _MARGIN - badge_w, 0.3, badge_w, 0.5, "#eef1f6", 0.18)
        ax.text(
            _WIDTH - _MARGIN - badge_w / 2,
            0.55,
            label,
            fontsize=10,
            color=_INK,
            ha="center",
            va="center",
        )

    tile_x0 = _MARGIN + _LABEL_W + 0.15
    tile_w = _WIDTH - _MARGIN - tile_x0

    for i, feature in enumerate(features):
        y = _HEADER_H + i * (_ROW_H + _ROW_GAP)
        mid = y + _ROW_H / 2

        box(_MARGIN, y, _LABEL_W, _ROW_H, _PANEL, 0.12)
        name_lines = _wrap_name(str(feature.feature))
        if len(name_lines) == 1 and len(name_lines[0]) <= 13:
            name_size, name_y, name_va = 12, mid - 0.27, "center"
            effect_y, n_y = mid, mid + 0.25
        elif len(name_lines) <= 2:
            name_size, name_y, name_va = 9, y + 0.22, "center"
            effect_y, n_y = y + 0.57, y + 0.79
        else:
            name_size, name_y, name_va = 7.5, y + 0.07, "top"
            effect_y, n_y = y + 0.7, y + 0.87
        ax.text(
            _MARGIN + 0.18,
            name_y,
            "\n".join(name_lines),
            fontsize=name_size,
            fontweight="bold",
            color=_INK,
            va=name_va,
            linespacing=1.15,
        )
        ax.text(
            _MARGIN + 0.18,
            effect_y,
            f"{_symbol(feature)} = {feature.effect_size:.2f}",
            fontsize=10,
            color=_MUTED,
            va="center",
        )
        ax.text(
            _MARGIN + 0.18,
            n_y,
            f"n = {feature.n:,}",
            fontsize=10,
            color=_MUTED,
            va="center",
        )

        if not feature.values:
            box(tile_x0, y, tile_w, _ROW_H, _NEUTRAL, 0.12)
            ax.text(
                tile_x0 + tile_w / 2,
                mid,
                f"{feature.feature_type.value}: no value-level breakdown",
                fontsize=10,
                color=_MUTED,
                style="italic",
                ha="center",
                va="center",
            )
            continue

        total = sum(v.support for v in feature.values)
        x = tile_x0
        for value in feature.values:
            width = value.support / total * tile_w
            tile = max(width - _TILE_GAP, 0.02)
            rgba = cmap(norm(min(max(value.lift, norm.vmin), norm.vmax)))
            box(x, y, tile, _ROW_H, rgba, 0.1)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ink = _INK if luminance > 0.55 else "white"
            cx = x + tile / 2
            name = _short(str(value.value), max(int(tile / 0.085), 2))
            if tile >= 1.1:
                ax.text(
                    cx,
                    mid - 0.28,
                    name,
                    fontsize=11 if len(name) <= 10 else 9.5,
                    fontweight="bold",
                    color=ink,
                    ha="center",
                    va="center",
                )
                ax.text(
                    cx,
                    mid,
                    f"{value.lift:.2f}×",
                    fontsize=12.5,
                    fontweight="bold",
                    color=ink,
                    ha="center",
                    va="center",
                )
                ax.text(
                    cx,
                    mid + 0.27,
                    f"({value.support / total:.1%})",
                    fontsize=9.5,
                    color=ink,
                    ha="center",
                    va="center",
                )
            elif tile >= 0.6:
                ax.text(
                    cx,
                    mid - 0.12,
                    name,
                    fontsize=8.5,
                    fontweight="bold",
                    color=ink,
                    ha="center",
                    va="center",
                )
                ax.text(
                    cx,
                    mid + 0.14,
                    f"{value.lift:.2f}×",
                    fontsize=8.5,
                    color=ink,
                    ha="center",
                    va="center",
                )
            x += width

    # Colour bar.
    bar_w = 6.5
    bar_top = height - _FOOTER_H + 0.35
    cax = fig.add_axes(
        (
            (_WIDTH - bar_w) / 2 / _WIDTH,
            1 - (bar_top + 0.25) / height,
            bar_w / _WIDTH,
            0.25 / height,
        )
    )
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
    )
    if norm.vmax - norm.vmin <= 2.4:
        cbar.set_ticks(
            np.round(np.arange(math.ceil(norm.vmin * 5) / 5, norm.vmax + 1e-9, 0.2), 1)
        )
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=10, colors=_INK)
    cbar.set_label("Lift (target rate ÷ baseline)", fontsize=11, color=_INK)
    return fig
