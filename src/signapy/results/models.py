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

import html
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from signapy.profiling.types import FeatureType, TargetType

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _check_count(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _plain(value: Any) -> Any:
    """Convert numpy scalars to Python builtins so output is serializable."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class ValueResult:
    """Value-level evidence for one category or ordinal level.

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
        numbers. Value labels are kept as-is, so non-JSON-native labels
        (e.g. timestamps) pass through unchanged.
        """
        return {
            "target": self.target,
            "features": [
                {
                    "feature": r.feature,
                    "feature_type": r.feature_type.value,
                    "target_type": r.target_type.value,
                    "effect_size_method": r.effect_size_method,
                    "effect_size": r.effect_size,
                    "n": r.n,
                    "missing_rate": r.missing_rate,
                    "test_method": r.test_method,
                    "p_value": r.p_value,
                    "values": None
                    if r.values is None
                    else [
                        {
                            "value": _plain(v.value),
                            "support": v.support,
                            "target_count": v.target_count,
                            "target_rate": v.target_rate,
                            "baseline_rate": v.baseline_rate,
                            "lift": v.lift,
                        }
                        for v in r.values
                    ],
                    "details": {k: _plain(d) for k, d in r.details.items()},
                }
                for r in self.features
            ],
        }

    _HEADER = ("feature", "type", "method", "effect", "p", "n", "missing")

    def _rows(self) -> list[tuple[str, ...]]:
        return [
            (
                r.feature,
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
        """HTML table for Jupyter and Databricks notebooks."""
        head = "".join(f"<th>{html.escape(h)}</th>" for h in self._HEADER)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
            for row in self._rows()
        )
        return (
            f"<div><b>DiscoveryReport</b> &mdash; target: "
            f"<code>{html.escape(self.target)}</code></div>"
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        )

    def plot(self) -> Figure:
        """Render the report as a mosaic and return the matplotlib ``Figure``.

        One row per feature, ordered by strength of association. Within a row,
        each value is a tile whose width is its support and whose colour is
        its lift against the baseline target rate. Continuous features appear as
        quantile bins; a feature with no value-level results (too few rows or
        distinct values to bin) appears as a single neutral tile.

        The figure is not registered with ``pyplot``: in a notebook, evaluate
        the returned figure to display it; in a script, call
        ``fig.savefig(...)``.

        Requires matplotlib (``pip install signapy[plot]``).
        """
        from signapy.results._plot import plot_mosaic

        return plot_mosaic(self)
