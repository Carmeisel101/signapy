"""Result models for feature-level and value-level discovery.

Design notes (see ``docs/architecture.md`` for the full rationale):

- Feature-level and value-level evidence are separate models. A
  :class:`FeatureResult` answers "does this feature contain signal?"; each
  :class:`ValueResult` answers "how does this particular value behave?".
  A feature result *contains* its value results, but the two are never
  merged into one abstraction.
- Fields shared by every discovery method are explicit attributes.
  Method-specific numbers (e.g. the chi-square statistic or degrees of
  freedom) go in ``details`` so the generic schema stays small.
- Models are frozen dataclasses. Constructors check structural invariants
  (counts, probabilities) so inconsistent evidence fails loudly at creation.
"""

from __future__ import annotations

import base64
import datetime
import enum
import html
import io
import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from signapy.profiling.types import FeatureType, TargetType
from signapy.results._plot import _display

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _check_count(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


# The notebook repr draws at most this many features; use ``plot()`` for more.
_HTML_MAX_FEATURES = 30


def _type_name(value: Any) -> str:
    """The label's type name; a constant if even that cannot be read.

    A hostile metaclass can make ``__module__`` or ``__qualname__`` raise, so
    this is the one place that must never depend on user-controlled code.
    """
    try:
        return f"<{type(value).__module__}.{type(value).__qualname__}>"
    except Exception:
        return "<unserializable>"


def _text(value: Any) -> str:
    """``str(value)``, or the type's name if the object's own ``__str__`` fails."""
    try:
        return str(value)
    except Exception:
        return _type_name(value)


def _plain(value: Any) -> Any:
    """Encode a category label as plain, strict-JSON-safe data. Never raises.

    The rule is closed and type-based, and uses the *base-class* conversions
    (``str.__str__``, ``datetime.datetime.isoformat``, ...) so a subclass
    cannot override how it is encoded:

    - ``None``, ``bool``, ``int``, ``str`` and *finite* ``float`` pass
      through (numpy and builtin subclasses are normalised); non-finite
      floats (``inf``, ``-inf``, ``nan``), which strict JSON cannot
      represent, become the strings ``"inf"``, ``"-inf"`` and ``"nan"``;
    - enums become their (encoded) ``value``;
    - dates, times, datetimes, ``pd.Timestamp`` and ``np.datetime64`` become
      ISO-8601 strings, keeping sub-second and nanosecond precision;
    - timedeltas (``datetime``, ``pd.Timedelta``, ``np.timedelta64``) become
      ISO-8601 durations;
    - temporal values that cannot be converted exactly (ambiguous
      ``np.timedelta64`` month/year units, out-of-range dates) fall back to
      ``str(value)``;
    - other numpy scalars become the matching builtin (``np.longdouble``
      becomes ``float``);
    - tuples and lists become lists of encoded items;
    - anything else (``Decimal``, ``UUID``, ...) becomes ``str(value)``;
    - if encoding or ``str()`` raises for any label, the result is the
      label's type name, e.g. ``"<mymodule.MyLabel>"``, rather than an error.
    """
    try:
        return _encode(value)
    except Exception:
        return _type_name(value)


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float | np.floating):
        number = float.__float__(value) if isinstance(value, float) else float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, pd.Timestamp):
        return pd.Timestamp.isoformat(value)
    if isinstance(value, datetime.datetime):
        return datetime.datetime.isoformat(value)
    if isinstance(value, datetime.date):
        return datetime.date.isoformat(value)
    if isinstance(value, datetime.time):
        return datetime.time.isoformat(value)
    if isinstance(value, np.datetime64):
        try:
            return pd.Timestamp(value).isoformat()
        except (ValueError, OverflowError):
            return _text(value)
    if isinstance(value, datetime.timedelta | np.timedelta64):
        try:
            return pd.Timedelta(value).isoformat()
        except (ValueError, OverflowError):
            return _text(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return _text(value)


def _number(value: float | int | None) -> float | int | None:
    """A result number as strict JSON: non-finite floats become ``None``."""
    if value is None:
        return None
    number = _plain(value)
    return number if isinstance(number, int | float) else None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class ValueResult:
    """Value-level evidence for one category, ordinal level, or numeric bin.

    Scoped to binary targets for v0.1: rates refer to the positive class.

    Attributes:
        value: The feature category or ordinal level being described.
        support: Number of rows with this value (non-missing target).
        target_count: Number of those rows where the target is positive.
        target_rate: ``target_count / support``.
        baseline_rate: Positive rate across all rows used for the feature.
        lift: ``target_rate / baseline_rate``. Values above 1 mean the
            positive class is over-represented for this value.
    """

    value: Hashable
    support: int
    target_count: int
    target_rate: float
    baseline_rate: float
    lift: float

    def __post_init__(self) -> None:
        _check_count("support", self.support)
        _check_count("target_count", self.target_count)
        if self.target_count > self.support:
            raise ValueError(
                f"target_count ({self.target_count}) cannot exceed "
                f"support ({self.support})"
            )
        _check_probability("target_rate", self.target_rate)
        _check_probability("baseline_rate", self.baseline_rate)


@dataclass(frozen=True)
class FeatureResult:
    """Feature-level evidence of association between one feature and the target.

    Attributes:
        feature: Column name of the feature.
        feature_type: Semantic type used for the analysis.
        target_type: Semantic type of the target.
        effect_size_method: Name of the effect-size measure, e.g.
            ``"cramers_v"``.
        effect_size: Strength of association; interpretation depends on
            ``effect_size_method``.
        n: Number of rows used in the analysis (after excluding missing
            values).
        missing_rate: Fraction of rows where the feature was missing.
        test_method: Name of the significance test, e.g. ``"chi_square"``,
            or ``None`` if no test was run.
        p_value: p-value from ``test_method``, or ``None`` if no test was run.
        values: Value-level results, or ``None`` if value-level discovery was
            not performed for this feature. An empty tuple is not used to
            mean "not performed".
        details: Method-specific numbers, e.g. ``{"statistic": 12.3, "dof": 2}``
            for a categorical feature, or ``{"positive_n": 150, "positive_mean":
            42.1, ...}`` for a continuous one. Both integer counts and
            floating-point measurements are legitimate here, represented
            honestly rather than coerced to one type.
    """

    feature: str
    feature_type: FeatureType
    target_type: TargetType
    effect_size_method: str
    effect_size: float
    n: int
    missing_rate: float
    test_method: str | None = None
    p_value: float | None = None
    values: tuple[ValueResult, ...] | None = None
    details: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_count("n", self.n)
        _check_probability("missing_rate", self.missing_rate)
        if self.p_value is not None:
            if self.test_method is None:
                raise ValueError("p_value requires test_method to be set")
            _check_probability("p_value", self.p_value)


@dataclass(frozen=True)
class DiscoveryReport:
    """Collection of discovery results for one target.

    Attributes:
        target: Column name of the target.
        features: Feature-level results, in analysis order.
    """

    target: str
    features: tuple[FeatureResult, ...] = ()

    def __post_init__(self) -> None:
        names = [result.feature for result in self.features]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate feature results: {duplicates}")
        if self.target in names:
            raise ValueError(f"target {self.target!r} cannot also be a feature")

    def feature(self, name: str) -> FeatureResult:
        """Return the result for feature ``name``.

        Raises:
            KeyError: If no result exists for ``name``.
        """
        for result in self.features:
            if result.feature == name:
                return result
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as plain, JSON-friendly Python data.

        Enums become their string values and numpy scalars become Python
        numbers. Category labels are always encoded to JSON-safe data:
        timestamps and dates become ISO-8601 strings, timedeltas ISO-8601
        durations, tuples lists, and other objects (``Decimal``, ``UUID``,
        ...) their ``str()``. Non-finite floats, which strict JSON cannot
        represent, become ``None`` in numeric result fields and the strings
        ``"inf"``/``"-inf"``/``"nan"`` as labels, so the output always
        passes ``json.dumps(..., allow_nan=False)``. See ``_plain`` for the
        full rule.
        """
        return {
            "target": _plain(self.target),
            "features": [
                {
                    "feature": _plain(r.feature),
                    "feature_type": r.feature_type.value,
                    "target_type": r.target_type.value,
                    "effect_size_method": r.effect_size_method,
                    "effect_size": _number(r.effect_size),
                    "n": _number(r.n),
                    "missing_rate": _number(r.missing_rate),
                    "test_method": r.test_method,
                    "p_value": _number(r.p_value),
                    "values": None
                    if r.values is None
                    else [
                        {
                            "value": _plain(v.value),
                            "support": _number(v.support),
                            "target_count": _number(v.target_count),
                            "target_rate": _number(v.target_rate),
                            "baseline_rate": _number(v.baseline_rate),
                            "lift": _number(v.lift),
                        }
                        for v in r.values
                    ],
                    "details": {k: _number(d) for k, d in r.details.items()},
                }
                for r in self.features
            ],
        }

    _HEADER = ("feature", "type", "method", "effect", "p", "n", "missing")

    def _rows(self) -> list[tuple[str, ...]]:
        return [
            (
                _display(str(r.feature)),
                r.feature_type.value,
                r.effect_size_method,
                _fmt(r.effect_size),
                _fmt(r.p_value),
                str(r.n),
                f"{r.missing_rate:.1%}",
            )
            for r in self.features
        ]

    def __str__(self) -> str:
        rows = [self._HEADER, *self._rows()]
        widths = [max(len(row[i]) for row in rows) for i in range(len(self._HEADER))]
        title = (
            f"DiscoveryReport(target={self.target!r}, {len(self.features)} features)"
        )
        lines = [title]
        for row in rows:
            lines.append(
                "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
            )
        lines.append("Full detail: report.to_dict() or report.feature(name).values")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Mosaic plus summary table for Jupyter and Databricks notebooks.

        The mosaic is embedded as an inline PNG, so it renders anywhere HTML
        does. Without matplotlib (the optional ``plot`` extra) only the table
        is shown.
        """
        head = "".join(f"<th>{html.escape(h)}</th>" for h in self._HEADER)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
            for row in self._rows()
        )
        return (
            f"<div><b>DiscoveryReport</b> &mdash; target: "
            f"<code>{html.escape(self.target)}</code></div>"
            f"{self._mosaic_html()}"
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        )

    def _mosaic_html(self) -> str:
        """Inline ``<img>`` of the mosaic, or ``""`` if it can't be drawn."""
        if not self.features:
            return ""
        # A notebook repr must never fail: any drawing problem falls back to
        # the summary table alone. Large reports are capped so one cell
        # display stays small.
        try:
            fig = self.plot(max_features=_HTML_MAX_FEATURES)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        except Exception:
            return ""
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return (
            f'<img alt="Mosaic of feature lift against {html.escape(self.target)}" '
            f'style="max-width:100%;height:auto" '
            f'src="data:image/png;base64,{encoded}">'
        )

    def plot(self, max_features: int | None = None) -> Figure:
        """Render the report as a mosaic and return the matplotlib ``Figure``.

        One row per feature, ordered by absolute effect size (the measures
        differ by feature type, so read the order as a rough guide, not a
        strict ranking). Within a row,
        each value is a tile whose width is its support and whose colour is
        its lift against the baseline target rate. Continuous features appear as
        quantile bins; a feature with no value-level results (too few rows or
        distinct values to bin) appears as a single neutral tile.

        Args:
            max_features: Plot only the ``max_features`` features with the
                largest absolute effect size, and say so in the figure.
                ``None`` (the default) plots every feature.

        The figure is not registered with ``pyplot``: in a notebook, evaluate
        the returned figure to display it; in a script, call
        ``fig.savefig(...)``.

        Requires matplotlib (``pip install signapy[plot]``).
        """
        from signapy.results._plot import plot_mosaic

        return plot_mosaic(self, max_features=max_features)
