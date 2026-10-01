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
├── __init__.py        # public API surface: __version__, discover (implemented)
├── py.typed           # PEP 561 marker: the package ships type hints
├── profiling/         # "what data is this?"
│   ├── types.py       # FeatureType, TargetType (implemented)
│   ├── (inference)    # planned: propose semantic types from data
│   └── (summary)      # planned: missingness, cardinality, prevalence
├── metrics/           # pure statistics: arrays/tables in, numbers out
│   ├── association.py  # implemented: cramers_v, point_biserial,
│   │                    #   spearman_rho (Phi for boolean features planned)
│   ├── significance.py # implemented: chi_square, ChiSquareResult
│   └── lift.py          # implemented: categorical_lift, CategoricalValueMetrics
│                        #   (reused for ordinal, reordered; no continuous
│                        #   equivalent yet, see §10a)
├── discovery/         # user-facing workflows: type → method → results
│   ├── __init__.py     # implemented: discover(df, target, *, positive_class,
│   │                    #   feature_types=None), method dispatch table
│   ├── feature.py       # implemented: analyze_categorical_feature,
│   │                    #   analyze_continuous_feature, analyze_ordinal_feature
│   └── value.py          # implemented: build_value_results (categorical and
│                        #   ordinal; discovery/feature.py reorders for ordinal)
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
  methods, handles missing values, and builds result models. The top-level
  `signapy.discover(df, target=..., *, positive_class=..., feature_types=None)`
  is the main entry point, exported from `signapy.discovery` and re-exported
  as `signapy.discover`. The `(feature_type, target_type) → analyzer`
  mapping is a literal dispatch dict in `discovery/__init__.py`
  (`_DISPATCH`), not an `if`/`elif` chain — adding a new supported
  combination means adding an entry there (and to
  `_SUPPORTED_FEATURE_TYPES`), not extending a growing conditional.
- **`profiling` owns types.** Type inference and user overrides live here.
- **No `utils` package until there is real shared code.** Avoid catch-all
  modules.

Modules marked "planned" are **not created as empty files**. Add each one
together with its implementation and tests.

## 10. First vertical slice — **implemented**

```text
binary target + categorical feature
        ↓
feature-level discovery: Cramér's V + chi-square
        ↓
value-level discovery: support, target count, target rate,
                       baseline target rate, categorical lift
```

Implementation order (all complete, on branches `feature/categorical-metrics`
and `feature/binary-categorical-discovery`):

1. `metrics/association.py`: `cramers_v(contingency_table)` — **implemented**
2. `metrics/significance.py`: chi-square test via `scipy.stats` — **implemented**
3. `metrics/lift.py`: per-value support, target count, target rate, lift —
   **implemented**
4. `discovery/feature.py` and `discovery/value.py`: build `FeatureResult`
   and `ValueResult` — **implemented**
5. `signapy.discover()`: minimal entry point, binary target and categorical
   features only, and an explicit error for anything unsupported —
   **implemented**
6. An example in `examples/` — **implemented**
   (`examples/binary_categorical_discovery.py`)

The pure metrics layer (steps 1-3) is now wired into `discovery` and
`results` (steps 4-5): `signapy.discover(df, target=..., *,
positive_class=...)` is a real, exported entry point that returns a
`DiscoveryReport` of `FeatureResult`/`ValueResult` objects. `metrics`
functions still return their own small result types (`ChiSquareResult`,
`CategoricalValueMetrics`) — `discovery/feature.py` and `discovery/value.py`
are exactly the adapter layer that converts those into
`FeatureResult`/`ValueResult`, so `metrics` itself stays unaware of
`results`. For how to choose between and interpret the three underlying
metrics as a user, see [`docs/metrics.md`](metrics.md); this section stays
focused on implementation decisions.

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
- **Point-biserial correlation (`point_biserial`):** the continuous-feature
  counterpart to `cramers_v`, added for the categorical+continuous mixed
  discovery slice (see
  [§10a](#10a-second-vertical-slice-mixed-categorical-and-continuous-discovery-implemented)).
  Settled decisions:
  - Uses `scipy.stats.pointbiserialr` directly (one call computes both the
    coefficient and its p-value — never call it twice to get them
    separately).
  - `positive_class` controls the sign: the target is encoded as `target ==
    positive_class` (1/0), so `coefficient > 0` means larger feature values
    go with the positive class. Reversing `positive_class` reverses the
    sign of both `coefficient` and `mean_difference`, exactly (verified by
    a property test), not just approximately.
  - `mean_difference` is defined as `positive_mean - negative_mean`, in the
    feature's original units. `PointBiserialResult.__post_init__` enforces
    this as an invariant (within floating-point tolerance) rather than
    trusting callers to keep the two consistent.
  - The feature is validated value by value before numeric conversion, so
    only genuine real numeric measurements are accepted. In particular,
    boolean, datetime, timedelta, complex, and string representations are
    rejected even when hidden in an `object`-dtype array. This keeps pandas
    coercion from silently defining the metric's input semantics. The
    function docstring owns the exhaustive user-facing validation contract.
  - Non-finite feature values (`inf`, `-inf`) are rejected outright, not
    treated as missing and dropped — a caller who wants to exclude them
    needs to do that explicitly, since silently dropping them would be a
    policy decision the metrics layer doesn't make.
  - A constant feature (zero variance) raises `ValueError` rather than
    returning `NaN` or `0.0`: Pearson correlation is undefined, not zero,
    when one variable has no variance.
  - A minimum of 3 total observations is required
    (`_MIN_POINT_BISERIAL_OBSERVATIONS`), so degrees of freedom
    (`n - 2 >= 1`) stay positive and the p-value stays well-defined, rather
    than relying on `scipy` to warn or return `NaN` for smaller samples.
  - Pandas nullable numeric dtypes (`Int64`, `Float64`, ...) are accepted
    once free of missing values: `numpy.asarray` on a fully-populated
    nullable array already resolves to a plain numeric dtype, so no special
    conversion path was needed.

### Settled decisions (discovery layer)

`signapy.discover(df, target, *, positive_class)` is a thin orchestration
layer: DataFrame-level validation, missing-data policy, and dtype policy
live here; the actual statistics stay in `metrics`.

- **Missing-data policy.** Two independent passes, in this order:
  1. Rows with a missing `target` are excluded from **all** analysis (their
     outcome is unknown, so they can't support any feature's evidence).
  2. For each feature independently, rows with a missing value for *that*
     feature are excluded from *that feature's* analysis only — a missing
     value in one feature does not affect another feature's results.

  Consequences of this policy, reflected directly in `FeatureResult` fields:
  - `n` is the number of rows with both a non-missing `target` and a
    non-missing value for that particular feature.
  - `missing_rate` is the fraction of *target-valid* rows (not all of
    `df`) where that feature is missing.
  - Value-level `baseline_rate` and `lift` are computed from the exact same
    feature-valid rows used for that feature's Cramér's V and chi-square —
    there is one contingency table per feature, and both the feature-level
    and value-level evidence for that feature come from it.

  Missing values are never treated as their own category and never
  imputed. This is a discovery-layer decision, not a metrics-layer one: the
  underlying `metrics` functions continue to reject missing values outright
  (see the metrics-layer decisions above); `discovery/feature.py` is what
  drops them before calling down.
- **Categorical feature dtype policy.** Only pandas `object`, `string`, and
  `category` dtype columns are treated as categorical features. Numeric
  columns — including integer-coded categories like `0, 1, 2` — and `bool`
  columns are rejected with an explicit `ValueError` naming the column and
  its dtype, not silently reinterpreted as categorical. This is deliberate:
  SignaPy does not yet infer semantic type from values (see [§6, Planned
  types](#6-planned-types)), so guessing that an integer or boolean column
  is "really" categorical would be exactly the kind of silent policy
  decision this project avoids. A future profiling/type-override feature is
  the right place to let a user say "treat this numeric/boolean column as
  categorical" explicitly.
- **DataFrame-level validation, checked before any per-feature work:** an
  unknown `target` column, an empty `df`, a `target` whose non-missing
  values are not exactly two distinct classes, and a `positive_class` not
  among those two classes are all rejected with a `ValueError` up front.
  Per-feature problems (unsupported dtype, fewer than two observed
  categories once that feature's missing values are dropped, or an
  otherwise-invalid contingency table) are still surfaced with a clear
  error, but only when that feature is reached — column order in `df` is
  preserved for `DiscoveryReport.features`, so which feature's error
  surfaces first follows `df`'s own column order.
- **`df` is never mutated.** `discover()` only reads from `df` (boolean
  masking and column selection, which return copies/views, not in-place
  operations).

Tests must be deterministic and use small synthetic datasets whose expected
statistics are known (hand-computed, or checked against `scipy.stats`).

## 10a. Second vertical slice: mixed categorical and continuous discovery (implemented)

```text
categorical feature → binary target        continuous feature → binary target
        ↓                                          ↓
Cramér's V + chi-square                     point-biserial correlation
        ↓                                          ↓
categorical_lift (support, rate,            group counts, means, and
baseline rate, lift)                        mean difference — no
                                             value-level localization yet
```

Extends `signapy.discover()` to also handle continuous features, selected
and declared explicitly via a new `feature_types` argument, on branches
`feature/continuous-binary-metrics` (the pure metric,
`signapy.metrics.association.point_biserial`) and
`feature/mixed-feature-discovery` (the wiring below).

### Public API addition

```python
def discover(
    df: pd.DataFrame,
    target: str,
    *,
    positive_class: Hashable,
    feature_types: Mapping[str, FeatureType | str] | None = None,
) -> DiscoveryReport: ...
```

- `feature_types=None` (the default) is **exactly** the original behavior:
  every non-target column, analyzed as categorical, with the same dtype
  check and error message as before. Implemented as the same code path as
  the explicit case (defaulting to `{col: FeatureType.CATEGORICAL for col
  in df.columns if col != target}`), not a separate branch, so the two
  can't silently drift apart.
- With `feature_types` given: its keys are the *only* columns analyzed, in
  `df`'s own column order (not the mapping's insertion order); `target`
  must not be a key; every key must be a column of `df`; every value must
  resolve to a supported `FeatureType` (`CATEGORICAL` or `CONTINUOUS` — the
  only two so far; `ORDINAL`/`BOOLEAN`/an invalid string all raise); an
  empty mapping raises.

### Settled decisions

- **Why `point_biserial` and not something else:** point-biserial
  correlation is exactly Pearson correlation between a continuous variable
  and a binary one — well-established, directly comparable in spirit to
  `cramers_v` (an effect size) plus `chi_square` (its significance test) in
  one call, and directly available via `scipy.stats.pointbiserialr`. See
  `docs/metrics.md` §4 for the user-facing interpretation.
- **Numeric columns require an explicit `feature_types` declaration.** An
  integer column might be a measured quantity, an ordinal level, or a
  compact category identifier — SignaPy does not infer this from the
  values (consistent with [§6, Planned types](#6-planned-types)). There is
  deliberately no "auto-detect continuous" path; the default
  (`feature_types=None`) only ever treats columns as categorical.
- **Positive-class directionality.** `positive_class` controls the *sign*
  of `point_biserial`'s `coefficient` and `mean_difference`: the target is
  encoded as `target == positive_class` (1/0), so `coefficient > 0` means
  larger feature values go with the positive class. Reversing
  `positive_class` reverses both signs exactly, not just approximately —
  verified by a property test (`tests/test_point_biserial.py`).
  `cramers_v`/`chi_square` have no such directionality (categorical
  variables have no ordering), which is one of the two questions
  `docs/metrics.md` explicitly distinguishes between feature types.
- **`mean_difference` is `positive_mean - negative_mean`,** in the
  feature's original units — not normalized like `coefficient`. This is
  deliberate: it gives interpretable, actionable scale
  (`docs/metrics.md`'s guiding principle that effect size and p-value
  aren't the whole story) that a `[-1, 1]` correlation coefficient alone
  doesn't communicate.
- **Missing and infinite values.** Same per-feature missing-data policy as
  categorical features (§10's settled decisions, item by item): a
  continuous feature's own missing values are dropped from *that feature's*
  analysis only, contributing to its own `missing_rate`; a non-missing but
  infinite value (`inf`/`-inf`) is **not** treated as missing and is not
  silently dropped — `point_biserial` raises, and
  `analyze_continuous_feature` re-raises with the feature name added, the
  same wrapping pattern `analyze_categorical_feature` already uses around
  `cramers_v`/`chi_square`.
- **Why continuous localization (binning) was deferred.** Categorical
  `categorical_lift` works because "which value" is already a discrete,
  finite question. For a continuous feature, "which range of values" isn't
  answerable without first choosing bin boundaries — equal-width, quantile,
  or domain-specific — and that choice is a modeling decision this project
  is not making on a user's behalf in this iteration (consistent with
  SignaPy's non-goals: it surfaces evidence, it doesn't make modeling
  choices for you). `FeatureResult.values` is therefore always `None` for
  a continuous feature, never an empty tuple (preserving the existing
  `values=None` meaning "not performed," from the results model's original
  design in §7) and never something invented to fill the gap. A future
  feature can add `values` for continuous features once binning is
  designed, without changing `FeatureResult`'s shape.
- **`FeatureResult.details` typing widened** from `Mapping[str, float]` to
  `Mapping[str, float | int]`: continuous `details` legitimately mixes
  integer group counts (`positive_n`, `negative_n`) with floating-point
  measurements (`positive_mean`, `negative_mean`, `mean_difference`), and
  the old annotation would have required dishonestly casting the counts to
  `float`.
- **Dispatch, not a growing conditional.** `discovery/__init__.py` holds a
  `_DISPATCH: dict[tuple[FeatureType, TargetType], analyzer]` mapping.
  Both entries today target `TargetType.BINARY`; a future continuous or
  multiclass target adds new dispatch entries rather than restructuring
  existing ones. An unmatched `(feature_type, target_type)` pair raises
  naming the feature, its declared type, the target type, and the
  currently supported combinations — currently unreachable in practice
  (every value `_resolve_feature_type` accepts has a dispatch entry for
  `TargetType.BINARY`, the only target type this layer supports), kept as
  an explicit named failure rather than a `KeyError` so it stays correct if
  either set changes independently later.

### Out of scope (this slice)

Confirmed non-goals, unchanged from the spec: classifier training,
multivariate/interaction analysis, automatic feature selection, automatic
continuous binning, per-bin lift, confidence intervals, multiple-testing
correction, cluster-aware tests, multiclass or continuous targets, and
stability analysis. See §12 and the "Statistical limitations" section of
`docs/metrics.md` for the general caveats (causation, sample-size
sensitivity, clustering, no out-of-sample validation, missing-data
population shift, multiple comparisons) that apply across every metric in
this project, not just the new one. (Ordinal discovery, listed as out of
scope here originally, was implemented next — see §10b.)

## 10b. Third vertical slice: ordinal discovery (implemented)

```text
ordinal feature (ordered Categorical) → binary target
        ↓
feature-level discovery: Spearman rank correlation (spearman_rho)
        ↓
value-level discovery: categorical_lift, reordered to the
                       feature's declared category order
```

Extends `signapy.discover()`'s `feature_types` to accept
`FeatureType.ORDINAL`, on branch `feature/ordinal-binary-discovery`. Sits
between categorical and continuous: order is meaningful, but the distance
between levels isn't assumed to be — the central design constraint this
whole slice is built around.

### Settled decisions

- **Explicit order, never inferred.** `FeatureType.ORDINAL` requires the
  column to already be an ordered pandas `Categorical`
  (`pd.Categorical(df[col], categories=[...], ordered=True)`). SignaPy
  never infers that `"low"`, `"medium"`, `"high"` are ordered, and never
  falls back to alphabetical order — consistent with [§6, Planned
  types](#6-planned-types)'s "never guess semantic type from values"
  principle, and with the numeric-column precedent set in §10a. Two
  distinct errors cover the two distinct mistakes: declaring a column that
  isn't a `category` dtype at all, versus declaring one that is but has
  `ordered=False` (see `_check_ordinal_dtype` in `discovery/__init__.py`).
- **Spearman, not Pearson/point-biserial on integer level codes.** This was
  the single most important call to get right. Point-biserial on raw codes
  (`low=0, medium=1, high=2`) implicitly assumes equally-spaced levels —
  exactly what "ordinal" means *not* assuming. Verified numerically during
  design: for an unevenly-sized three-level feature (10/10/180), naive
  Pearson-on-codes gave `0.3713` while true tie-aware Spearman gave
  `0.3585` — different numbers, because Spearman's internal rank transform
  weights each tied level by how many observations share it, while raw
  codes don't. `spearman_rho()` (`signapy/metrics/association.py`) takes
  rank/level codes and calls `scipy.stats.spearmanr` directly, which does
  this tie-averaging internally — never take the shortcut of calling
  `point_biserial()` on integer codes instead.
- **Degenerate-input guards, added from concrete `scipy` behavior, not
  assumption.** Verified directly before implementing:
  - A constant feature (one distinct level) makes `scipy.stats.spearmanr`
    return `nan`/`nan` **and** emit a `ConstantInputWarning` — which, like
    the complex-number case in `point_biserial`'s dtype-validation fix,
    would fail this project's `pytest` `filterwarnings=error` config.
    `spearman_rho` checks for fewer than 2 distinct levels and raises
    `ValueError` before ever calling `scipy`.
  - `n=2` returns a defined coefficient but `pvalue=nan`, silently, with no
    warning at all. `spearman_rho` has its own
    `_MIN_SPEARMAN_OBSERVATIONS = 3` guard (mirroring
    `_MIN_POINT_BISERIAL_OBSERVATIONS`, same `n - 2 >= 1` degrees-of-freedom
    reasoning), independent of the "fewer than 2 levels" check — a feature
    can have exactly 2 levels and still fail this guard with too few total
    rows.
- **Validation reuse, not duplication.** `spearman_rho` shares
  `point_biserial`'s `_validate_real_numeric_feature` helper (rejecting
  boolean, datetime/timedelta, complex, and other non-real values) — now
  parameterized with a `caller_name` so error messages correctly say
  `"spearman_rho"` rather than hardcoding `"point_biserial"`. Both
  functions need the same "real numeric input" guarantee; writing it twice
  in the same module would have been pure duplication, not the
  module-independence duplication pattern `_validate_contingency_table`
  uses *across* `association.py`/`significance.py`.
- **Value-level evidence reuses `categorical_lift`/`build_value_results`
  unchanged, reordered afterward.** `analyze_ordinal_feature`
  (`discovery/feature.py`) calls `build_value_results` exactly as the
  categorical path does — getting results in order of first appearance —
  then reorders that tuple to the feature's `.cat.categories` order. A
  declared category with zero observations in the feature-valid subset is
  simply absent from the reordered tuple, not zero-filled:
  `categorical_lift` never invents a zero-support entry, and ordinal
  doesn't either.
- **Spearman's "monotonic trends only" limitation is treated as a
  first-class caveat, not a footnote.** A feature whose middle level peaks
  (e.g. "medium" converts far better than both "low" and "high") can
  produce `spearman_rho`'s `coefficient = 0.0` *exactly*, despite a real,
  strong per-level relationship — verified with a constructed example
  (`docs/metrics.md` §10): symmetric low/high rates around a peaked middle
  level give `coefficient=0.0, p_value=1.0` while the middle level's lift
  is `2.0`. This is why `docs/metrics.md` §7 recommends always reading the
  reordered `values` for an ordinal feature, not just `effect_size` —
  mirrored by a dedicated regression test in `tests/test_spearman.py`.
- **`details` for ordinal is `{"n_levels": ...}`,** not empty and not
  carrying a redundant "statistic" (unlike `chi_square`, Spearman's
  statistic *is* the coefficient — there's no separate raw test statistic
  to report). `n_levels` — the count of distinct observed levels — gives
  useful context `FeatureResult.n` alone doesn't (e.g. distinguishing "3
  levels declared, 3 observed" from "4 declared, only 2 observed after
  filtering").

Tests (`tests/test_spearman.py`, `tests/test_ordinal_discovery.py`) cover
increasing, decreasing, null, tied, and non-monotonic relationships; every
documented invalid-input and degenerate case; declared-order value
reordering (including a zero-support declared category); positive-class
sign reversal (exact, not approximate, matching `point_biserial`'s
existing invariant); and mixed discovery with categorical and continuous
features in the same report.

### Out of scope (this slice)

Confirmed non-goals: ordinal value-level confidence intervals, ordinal
interaction effects, boolean-feature discovery (`FeatureType.BOOLEAN`
remains unsupported), semantic type inference (an ordered `Categorical` is
still a deliberate, explicit declaration, not something SignaPy proposes),
and everything else §10a already excludes (multiclass/continuous targets,
stability analysis, multiple-testing correction, and so on).

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
