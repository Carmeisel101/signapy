# SignaPy Architecture

This document is the design contract for SignaPy. Coding agents and
contributors should treat it as the source of truth. If the implementation
needs to depart from it, update this document in the same change.

## 1. Goals

SignaPy is an exploratory feature-discovery library for labeled datasets. Its
long-term mission is to **find, quantify, localize, and validate predictive
signal**.

- **Task-aware.** Infer target type and feature type, choose an appropriate
  discovery strategy, and allow explicit overrides when inference is ambiguous.
- **Concept-first.** Users think in discovery questions (profiling, feature,
  value, interaction, stability), not in statistical tests. Tests are
  implementation details that provide evidence.
- **Structured evidence.** Return typed, immutable result objects, never
  loose dicts or printed output.
- **Lightweight.** Runtime dependencies are limited to `numpy`, `pandas`, and
  `scipy`.

## 2. Non-goals

SignaPy is **not**:

- an AutoML framework
- a feature-selection oracle
- a modeling framework
- a replacement for domain expertise
- a tool that automatically drops features based on p-values

> SignaPy discovers evidence of predictive signal.
> It does not decide what your model should use.

The library surfaces evidence. The data scientist makes the modeling decision.
SignaPy must not add heavy ML dependencies (scikit-learn, XGBoost, PyTorch,
TensorFlow) to do statistical feature discovery.

## 3. Terminology

| Term | Meaning |
| --- | --- |
| **Feature** | A candidate input column being evaluated. |
| **Target** | The labeled outcome column. |
| **Value** | A distinct level of a (categorical/boolean/ordinal) feature. Continuous features may later be localized via bins/ranges. |
| **Signal** | A non-trivial statistical relationship between a feature (or value) and the target. |
| **Evidence** | Numbers supporting or weakening a claim of signal: effect size, uncertainty, support, prevalence, localization, stability. |
| **Feature type** | Semantic type of a feature: `continuous`, `ordinal`, `categorical`, `boolean`. |
| **Target type** | Semantic type of the target (the learning task): `binary`, `multiclass`, `continuous`. |
| **Support** | Number of rows backing a value-level claim. |
| **Baseline rate** | Positive-class rate across all rows used for a feature. |
| **Lift** | `target_rate / baseline_rate` for a value. |

## 4. Discovery hierarchy

1. **Profiling:** What data am I working with?
2. **Feature-level discovery:** Does an individual feature contain signal
   relative to the target?
3. **Value-level discovery:** Where within a feature does that signal live?
4. **Interaction discovery** *(future):* Does additional signal emerge from
   combinations of features or values?
5. **Stability analysis** *(future):* Does the discovered signal persist across
   samples, folds, time periods, or other meaningful partitions?

## 5. Feature-level vs value-level analysis

This distinction is fundamental. The two levels must stay separate in both
the code and the result models.

### Feature-level: "Does this feature contain signal?"

These methods describe the relationship between the **feature as a whole**
and the target.

| Feature → Target | Effect size | Test |
| --- | --- | --- |
| Categorical → Binary | Cramér's V | chi-square |
| Boolean → Binary | Phi coefficient | chi-square |
| Continuous/Ordinal → Binary | Spearman correlation, point-biserial correlation | (associated test) |

### Value-level: "Where within this feature does the signal live?"

For categorical features with a binary target, candidate metrics:

- support (observation count)
- target count
- target rate
- baseline target rate
- categorical lift
- risk difference *(later)*
- odds ratio *(later)*

Example: feature-level discovery shows `acquisition_channel` is associated
with the target. Value-level discovery shows where:

```text
referral -> lift 1.91
organic  -> lift 1.28
paid     -> lift 0.73
```

A feature can show strong feature-level association that is concentrated in
a single rare value. A feature can also show weak overall association while
one well-supported value has a large lift. Both views are needed.

## 6. Planned types

Types are **semantic**, not storage dtypes. A pandas dtype alone cannot
determine semantic type: `0, 1, 2` may be continuous, ordinal, or category
identifiers. Inference proposes a type, and the user can override it.

Defined in `signapy.profiling.types`:

- `FeatureType`: `CONTINUOUS`, `ORDINAL`, `CATEGORICAL`, `BOOLEAN`
- `TargetType`: `BINARY`, `MULTICLASS`, `CONTINUOUS`

For v0.1, `BINARY` is the only supported target type, and `CATEGORICAL` is the
first supported feature type. The other members exist so the result schema
and method selection can be designed with them in mind. They do not imply
support.

## 7. Results philosophy

SignaPy returns structured, immutable results (`signapy.results.models`):

- **`FeatureResult`**: feature-level evidence. Its generic fields apply to
  every method: `feature`, `feature_type`, `target_type`,
  `effect_size_method`, `effect_size`, `n`, `missing_rate`, `test_method`,
  `p_value`.
  - Effect size and significance test are named **separately**
    (`effect_size_method="cramers_v"`, `test_method="chi_square"`) because
    they are different kinds of evidence and are not always paired.
  - Method-specific numbers (chi-square statistic, degrees of freedom, …) go
    in `details: Mapping[str, float]` so the generic schema doesn't grow with
    each method. If a detail becomes universal, promote it to a field.
  - `values` holds value-level results. `None` means "value-level discovery
    not performed" (e.g. a continuous feature). It is not an empty tuple.
- **`ValueResult`**: value-level evidence for one categorical value against a
  binary target: `value`, `support`, `target_count`, `target_rate`,
  `baseline_rate`, `lift`. It is intentionally scoped to binary targets for
  v0.1. Multiclass or regression targets will need different value-level
  fields (per-class rates, target means). Add those as separate models rather
  than making this one generic.
- **`DiscoveryReport`**: a thin container: `target`, `features`, and
  `feature(name)` lookup. Keep it thin. Presentation such as tables,
  `to_frame()`, and ranking views belongs in methods or helpers added only
  when needed.

Models are frozen dataclasses. Constructors validate structural invariants:
counts are non-negative, `target_count <= support`, probabilities are in
`[0, 1]`, a `p_value` requires a `test_method`, and a report has no duplicate
features and does not list the target as a feature. Metric code is
responsible for handling degenerate inputs *before* building a result. For
example, it decides what to report when a contingency table has a zero
expected count.

Results report **evidence only**. They carry no `selected`, `keep`, or
`significant` flags.

## 8. Statistical philosophy

Feature discovery must not reduce to `p < 0.05`. Statistical significance
and predictive usefulness are different things:

- With large `n`, negligible effects become "significant".
- With small `n`, meaningful effects may not reach significance.
- A strong effect backed by 12 rows is weak evidence.

SignaPy's evidence should combine:

- **effect size:** how strong is the relationship?
- **statistical uncertainty:** p-values, and later confidence intervals
- **sample support:** how many rows back the claim?
- **target prevalence:** baseline rate context for rates and lifts
- **signal localization:** which values carry the signal?
- **stability** *(future):* does it persist across partitions?

Default orderings or summaries should favor effect size with support and
uncertainty context over raw p-values.

## 9. Package and module responsibilities

```text
src/signapy/
├── __init__.py        # public API surface; currently only __version__
├── py.typed           # PEP 561 marker: the package ships type hints
├── profiling/         # "what data is this?"
│   ├── types.py       # FeatureType, TargetType (implemented)
│   ├── (inference)    # planned: propose semantic types from data
│   └── (summary)      # planned: missingness, cardinality, prevalence
├── metrics/           # pure statistics: arrays/tables in, numbers out
│   ├── association.py  # implemented: cramers_v (categorical, Phi/Spearman/
│   │                    #   point-biserial for other feature types planned)
│   ├── significance.py # implemented: chi_square, ChiSquareResult
│   └── lift.py          # implemented: categorical_lift, CategoricalValueMetrics
├── discovery/         # user-facing workflows: type → method → results
│   ├── (feature)      # planned: feature-level discovery
│   └── (value)        # planned: value-level discovery
└── results/
    └── models.py      # FeatureResult, ValueResult, DiscoveryReport (implemented)
```

Dependency direction: `discovery` → `metrics`, `profiling`, `results`;
`results` → `profiling.types`. `metrics` depends on nothing inside SignaPy.

Rules:

- **`metrics` is pure.** No type inference, no method selection, no result
  objects. Functions take `pd.Series`/`np.ndarray`/contingency tables and
  return numbers or small tuples. This keeps them easy to test against known
  values.
- **`discovery` owns decisions.** It maps `(feature_type, target_type)` to
  methods, handles missing values, and builds result models. A top-level
  `signapy.discover(df, target=...)` will be the main entry point. It is
  exported only once it works.
- **`profiling` owns types.** Type inference and user overrides live here.
- **No `utils` package until there is real shared code.** Avoid catch-all
  modules.

Modules marked "planned" are **not created as empty files**. Add each one
together with its implementation and tests.

## 10. First vertical slice (next milestone)

```text
binary target + categorical feature
        ↓
feature-level discovery: Cramér's V + chi-square
        ↓
value-level discovery: support, target count, target rate,
                       baseline target rate, categorical lift
```

Suggested implementation order:

1. `metrics/association.py`: `cramers_v(contingency_table)` — **implemented**
2. `metrics/significance.py`: chi-square test via `scipy.stats` — **implemented**
3. `metrics/lift.py`: per-value support, target count, target rate, lift —
   **implemented**
4. `discovery/feature.py` and `discovery/value.py`: build `FeatureResult`
   and `ValueResult` — not yet started
5. `signapy.discover()`: minimal entry point, binary target and categorical
   features only, and an explicit error for anything unsupported — not yet
   started
6. An example in `examples/` — not yet started

Steps 1-3 (the pure metrics layer) are implemented, on branch
`feature/categorical-metrics`. They are deliberately **not wired into
`discovery` or `results` yet**: `signapy.discover()` still does not exist,
and `metrics` functions return their own small result types
(`ChiSquareResult`, `CategoricalValueMetrics`), not `FeatureResult` /
`ValueResult`. That wiring is the next task. For how to choose between and
interpret these three metrics as a user, see
[`docs/metrics.md`](metrics.md); this section stays focused on
implementation decisions.

### Settled decisions (metrics layer)

- **Positive class:** `categorical_lift` takes an explicit, required
  `positive_class` keyword argument. The metrics layer does not guess which
  class is "positive" — that inference (or a user override) belongs to the
  discovery layer that will call it.
- **Missing values:** rejected, not silently dropped or treated as their own
  category. All three metrics functions raise `ValueError` if given missing
  values (NaN in a contingency table, or `NaN`/`None` in `feature`/`target`).
  This keeps `metrics` policy-free; the discovery layer decides how to
  handle missingness (e.g. as its own category) before calling down.
- **Cramér's V bias correction:** `cramers_v(..., bias_correction=True)` by
  default, implementing the Bergsma (2013) small-sample correction.
  `bias_correction=False` gives the standard (Cramér, 1946) formula. The
  corrected estimator can become undefined for small, sparse tables (the
  corrected dimensions collapse to ≤1 category); this raises a clear
  `ValueError` rather than returning `NaN` or a nonsensical value, and the
  error message suggests retrying with `bias_correction=False`.
- **Chi-square continuity correction:** `chi_square()` and the chi-square
  statistic inside `cramers_v()` both call
  `scipy.stats.chi2_contingency(table, correction=False)` — no Yates'
  continuity correction — so the two are consistent with each other and with
  direct `scipy` cross-checks.
- **Degenerate tables:** a contingency table must be 2D, non-empty, at least
  2×2, contain only finite, non-negative, **integer-valued** counts, and
  have no row or column summing to zero (which would make an expected
  frequency zero/undefined). Any violation raises `ValueError` before scipy
  is called. Integer-valued counts are required (not just non-negative
  values) because the bias-corrected Cramér's V treats the table sum as a
  literal observation count `n`; fractional/weighted tables silently
  corrupt that correction rather than raising, e.g. `cramers_v([[0.1, 0.1],
  [0.1, 0.1]])` (a table proportional to independence) returned ~0.79
  instead of 0.0 before this check was added. Weighted/proportional tables
  are not supported by this metrics layer.
- **Unobserved positive class:** `categorical_lift` requires a binary target
  (exactly two distinct observed classes), and separately requires that
  `positive_class` be one of those two observed values. A `positive_class`
  that was not observed is treated primarily as an **invalid argument** (in
  practice almost always a typo) and raises its own `ValueError` naming
  `positive_class` and the classes that were actually observed — not framed
  as the statistical "zero baseline rate" edge case, since those are
  different failure modes even though, given a binary target, they are the
  same underlying condition. A `baseline_rate == 0.0` check remains as a
  defensive fallback (technically unreachable given the checks above) rather
  than letting a future change to those checks silently produce a division
  by zero.

Tests must be deterministic and use small synthetic datasets whose expected
statistics are known (hand-computed, or checked against `scipy.stats`).

## 11. Future: interaction discovery

Question: *Does signal emerge when features or values are combined?*

Candidates: categorical interaction lift, pairwise interaction analysis,
conditional associations. This will likely become a `discovery/interaction.py`
module with its own result model (e.g. `InteractionResult`) that references
the participating features or values. It will not be forced into
`FeatureResult`. Not in v0.1.

## 12. Future: stability analysis

Question: *Can this discovered signal be trusted across different samples?*

Candidates: fold stability, bootstrap stability, temporal stability, subgroup
stability. Stability evaluates the **signal itself** (e.g. the spread of
Cramér's V or a value's lift across resamples), not model performance. It
will likely wrap existing feature- and value-level discovery and attach
stability evidence to results. Not in v0.1.

## 13. Packaging, versioning, and tooling

- src layout, `pyproject.toml`, `hatchling` build backend.
- Distribution name `signapy`, import name `signapy`.
- Python `>=3.10`, tested on 3.10 to 3.14.
- Versions follow PEP 440. The single source is `__version__` in
  `src/signapy/__init__.py`, which hatchling reads. Development builds use
  `.devN`, and releases will follow semantic versioning.
- Dev tooling: `pytest` (warnings are errors) and `ruff` (lint and format).
- CI (`.github/workflows/ci.yml`) runs lint, format check, import check, and
  tests. PyPI publishing is intentionally not configured and will be a
  separate task.
