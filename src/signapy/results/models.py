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

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field

from signapy.profiling.types import FeatureType, TargetType


def _check_count(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class ValueResult:
    """Value-level evidence for one value of a categorical feature.

    Scoped to binary targets for v0.1: rates refer to the positive class.

    Attributes:
        value: The feature value (category) being described.
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
        details: Method-specific numbers, e.g. ``{"statistic": 12.3,
            "dof": 2}``.
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
    details: Mapping[str, float] = field(default_factory=dict)

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
