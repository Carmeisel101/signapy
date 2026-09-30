# Metrics Guide

This page is for choosing which SignaPy metric answers your question, and for
reading the numbers correctly once you have them. It doesn't repeat the
function-level details (exact arguments, validation, error messages) that live
in each function's docstring, and it doesn't cover package structure or design
rationale — see [`docs/architecture.md`](architecture.md) for that. See the
[README](../README.md) for installation and the project's overall scope.

> **Status:** the functions described here — `cramers_v`, `chi_square`,
> `point_biserial`, and `categorical_lift` — are implemented in
> `signapy.metrics`, and all except `point_biserial` are also wired into
> `signapy.discover()` for categorical features; `point_biserial` is wired
> in for continuous features via `discover(..., feature_types={...})`. You
> can also call any of them directly on a contingency table or on
> `feature`/`target` arrays, as this page mostly does, without going through
> `discover()`.

## 1. Overview

SignaPy's metrics split into two levels that answer different questions, and
the metric that applies depends on whether the feature is categorical or
continuous:

| Level | Feature type | Metric | Question |
| --- | --- | --- | --- |
| Feature-level | Categorical | `cramers_v` | How strong is the overall association? |
| Feature-level | Categorical | `chi_square` | Is the observed association unlikely under independence? |
| Feature-level | Continuous | `point_biserial` | Do larger feature values go with the positive or negative class, and how strongly? |
| Value-level | Categorical | `categorical_lift` | Which individual categories are above or below baseline? |
| Value-level | Continuous | *(not yet implemented)* | Which *ranges* of values are above or below baseline? |

Feature-level metrics summarize a feature *as a whole* against a target.
Value-level metrics break that summary down by category (or, eventually,
by range, for continuous features — see [§5a](#5a-no-continuous-lift-yet)).
Neither level replaces the other: a feature can show a strong overall
association that turns out to be driven by one category, or a weak overall
association that still hides one category worth investigating. Read
feature-level and value-level evidence together, not as substitutes.

`signapy.discover()` characterizes these univariate statistical
relationships. It does not train a classifier, select features
automatically, or tell you how features perform *together* — see
[`docs/architecture.md`](architecture.md) for that distinction.

## 2. Cramér's V

**Primary question: how strong is the overall categorical association?**

`cramers_v(contingency_table)` returns a single number in `[0, 1]`:

- **0** means no association between the feature and the target in this
  sample.
- **1** means perfect association — knowing the feature category tells you
  the target exactly.
- Values in between indicate partial association; larger values indicate a
  stronger relationship.

There is no universally correct cutoff for "strong" or "weak." What counts
as a meaningful Cramér's V depends on the domain, the number of categories
and target classes, the sample size, and what decision the number is
informing. Treat published rules of thumb as starting points for discussion,
not thresholds to apply mechanically.

**When it's appropriate:** two categorical variables (the feature and the
target), when you want a single summary of association strength that is
roughly comparable across features and datasets. Unlike the chi-square
statistic (below), Cramér's V is not a direct function of sample size, so a
value of 0.3 means roughly the same thing whether `n` is 100 or 100,000 —
which is what makes it useful for comparing features against each other.

**Limitations:**

- It says nothing about *direction* — categorical variables don't have a
  sign, so "association" here isn't "higher X causes higher Y."
- It says nothing about *causation*.
- It carries no statistical significance information on its own — a
  moderate V computed from 20 rows and the same V computed from 20,000 rows
  deserve very different confidence. Pair it with `chi_square` (or, later,
  a sample-support check) rather than reading it in isolation.
- It doesn't tell you *which* categories drive the relationship — for that,
  see [`categorical_lift`](#5-categorical-lift).

`cramers_v` defaults to the Bergsma (2013) bias-corrected estimator
(`bias_correction=True`), which reduces the small-sample upward bias of the
classic (Cramér, 1946) formula; pass `bias_correction=False` for the
uncorrected version. See the function's docstring for the exact formulas and
edge-case behavior (e.g. when the corrected estimator is undefined for very
sparse tables).

## 3. Chi-square test

**Primary question: is the observed association unlikely under
independence?**

`chi_square(contingency_table)` returns a `ChiSquareResult` with three
fields:

- **`statistic`** — the chi-square statistic: a measure of how far the
  observed counts are from what you'd expect if the feature and target were
  independent. Larger values mean a bigger gap from independence, but the
  statistic's scale depends on the table's size and shape, so it isn't
  comparable across different tables the way Cramér's V is.
- **`p_value`** — the probability of seeing a gap from independence at least
  this large, if the feature and target were actually independent, given
  this sample size.
- **`dof`** — degrees of freedom, `(rows - 1) * (cols - 1)`.

**A low p-value is evidence against independence — nothing more.** It does
not mean the effect is large, useful, or worth acting on. A p-value can be
very small while the underlying association (as measured by Cramér's V) is
practically negligible, especially with a large sample. See the
[worked example](#7-worked-example-categorical-feature) below for exactly
this case.

**Appropriate use and assumptions:** the chi-square test of independence
assumes the observations are independent of each other (e.g. not repeated
measurements of the same entity), and that expected cell frequencies are
large enough for the chi-square approximation to hold — a common informal
guideline is expected counts of at least 5 in most cells. `chi_square()`
does not check this for you. With sparse tables (many categories, some with
very few observations), the p-value can be unreliable; consider merging rare
categories, collecting more data, or using an exact test instead.

**Limitations:**

- Like any significance test, its p-value shrinks as `n` grows for a fixed
  effect size — with enough data, even a trivial association becomes
  "significant."
- It doesn't communicate effect magnitude. Use `cramers_v` for that.
- It doesn't establish causality.

`signapy`'s `chi_square()` computes the statistic without Yates' continuity
correction (`correction=False`), matching the chi-square statistic used
internally by `cramers_v`. See the function docstring for details.

## 4. Point-biserial correlation

**Primary question: do continuous values tend to be higher or lower for the
configured positive class?**

`point_biserial(feature, target, *, positive_class)` is the continuous
counterpart to `cramers_v`/`chi_square`: it applies when the *feature* is
continuous and the target is binary (`cramers_v`/`chi_square` apply when
*both* are categorical). It returns a `PointBiserialResult`:

- **`coefficient`** — the point-biserial correlation, in `[-1, 1]`. This is
  literally the Pearson correlation between the feature and the target
  encoded as 1 for `positive_class` and 0 otherwise:

  | Coefficient | Meaning |
  | --- | --- |
  | `> 0` | Larger feature values are associated with the positive class. |
  | `< 0` | Larger feature values are associated with the negative class. |
  | `≈ 0` | Little *linear* association is present. |

  **The sign depends on which class you configured as `positive_class`.**
  Reversing it reverses the sign of `coefficient` (and of
  `mean_difference`, below) without changing its magnitude — this is a
  choice you make when calling the function, not a fact intrinsic to the
  data.
- **`p_value`** — same interpretation as `chi_square`'s: evidence against
  the null hypothesis that the true correlation is zero, nothing more. A
  small `p_value` here does not mean `coefficient` is large or that the
  relationship is useful.
- **`positive_n` / `negative_n`** — how many rows fall in each target
  class.
- **`positive_mean` / `negative_mean`** — the feature's mean within each
  group, in the feature's *original units*.
- **`mean_difference`** — `positive_mean - negative_mean`. Unlike the
  normalized `coefficient`, this carries scale: "the positive class
  averages 7.8 units higher" is often more directly actionable than "the
  correlation is 0.48."

**When it's appropriate:** a continuous feature and a binary target, when
you want to know both the direction/strength of a linear relationship
(`coefficient`) and its size in real units (`mean_difference`).

**Limitations:**

- It measures **linear** association only. A weak `coefficient` does not
  rule out a real, strong *nonlinear* relationship (e.g. a feature that's
  elevated for the positive class only in a middle range) — this metric
  would report something close to zero for exactly that pattern.
- Like `chi_square`'s p-value, `p_value` is sensitive to sample size: with
  enough rows, a negligible `coefficient` can still be "significant."
- `mean_difference` and `coefficient` describe association, not causation.
- There's currently no continuous equivalent of `categorical_lift` — see
  [§5a](#5a-no-continuous-lift-yet).

`signapy`'s `point_biserial()` uses `scipy.stats.pointbiserialr` directly
(a single call computes both the coefficient and its p-value). See the
function docstring for exact validation rules (missing values, non-finite
values, boolean features, minimum sample size, and so on).

## 5. Categorical lift

**Primary question: which individual categories have higher or lower
positive rates than the baseline?**

`categorical_lift(feature, target, *, positive_class)` returns one
`CategoricalValueMetrics` per distinct feature value, each with:

- `support` — how many rows have this category.
- `positive_count` — how many of those rows have `target == positive_class`.
- `positive_rate` — `positive_count / support`.
- `baseline_rate` — the positive rate across the *entire* input (the same
  value on every result).
- `lift` — `positive_rate / baseline_rate`.

**Interpreting lift:**

| Lift | Meaning |
| --- | --- |
| `1.0` | This category's positive rate matches the overall baseline. |
| `> 1.0` | This category has an above-baseline positive rate — e.g. `1.5` means 50% higher than baseline. |
| `< 1.0` | This category has a below-baseline positive rate — e.g. `0.5` means half the baseline. |

Lift is **value-level** evidence: it doesn't summarize the feature as a
whole, and a feature can have a near-zero Cramér's V while still containing
one category with a notably high or low lift (or vice versa — a feature-wide
association driven by several categories, none individually extreme). Use
lift to find *where* signal lives once feature-level evidence suggests
there's signal to look for, per the
[categorical worked example](#7-worked-example-categorical-feature).

**Watch support, not just lift.** A `lift` of `3.0` computed from 4 rows is
much weaker evidence than a `lift` of `1.2` computed from 40,000 rows. High
lift with low `support` is exactly the kind of thing that looks dramatic and
turns out to be noise; treat it as a hypothesis to check against more data,
not a conclusion.

**Limitations:** `categorical_lift` does not run a significance test (no
p-value, no confidence interval) and does not imply causation — a category
with high lift is *associated* with the positive class in this sample, not
necessarily a cause of it.

### 5a. No continuous lift yet

There is currently no continuous equivalent of `categorical_lift`. For a
continuous feature, `discover()` always returns `values=None` — the
feature-level `point_biserial` evidence (§4) is all you get in this
release. Localizing signal within a continuous range (e.g. "risk climbs
sharply above 40 units") requires binning the feature, and SignaPy
deliberately doesn't do that automatically: bin boundaries are a modeling
choice (equal-width? quantile? domain-specific cutoffs?) that this project
isn't making on your behalf. It's planned future work — see
[`docs/architecture.md`](architecture.md) — not an oversight.

## 6. How to use the metrics together

For a **categorical** feature:

1. **`chi_square`** — is there evidence against independence at all?
2. **`cramers_v`** — if so, how strong is that association in practical
   terms?
3. **`categorical_lift`** — which specific categories are pulling the
   positive rate up or down?

For a **continuous** feature, the analogous workflow collapses to one step,
since `point_biserial` combines a directional effect size and a p-value
(and there's no lift-equivalent yet, per [§5a](#5a-no-continuous-lift-yet)):

1. **`point_biserial`** — is there a linear association, how strong is it,
   which direction does it point, and what's the size of the gap between
   groups in real units (`mean_difference`)?

Common combinations and how to read them:

| Pattern | Reading |
| --- | --- |
| Low `p_value` + low `cramers_v`/`coefficient` | Statistically detectable, practically weak. With enough data, real but tiny effects become significant; don't treat this as an important feature on its own. |
| Meaningful `cramers_v` + informative lifts | Stronger evidence overall, with `categorical_lift` telling you which categories to look at or act on. |
| Meaningful `\|coefficient\|` + a large `mean_difference` | Stronger evidence overall for a continuous feature: not just "detectable," but a sizeable, interpretable gap between the two groups. |
| Large lift with small `support` | Hypothesis-generating, not strong evidence. Worth a closer look (more data, a stability check across folds/time — [planned](architecture.md#12-future-stability-analysis)), not an immediate conclusion. |
| Low `\|coefficient\|` alone | Weak *linear* evidence only — doesn't rule out a nonlinear relationship a correlation coefficient can't see (see §4's limitations). |

None of these numbers, alone or combined, tell you whether to keep a feature
in a model — see the [statistical philosophy](architecture.md#8-statistical-philosophy)
in the architecture doc. SignaPy surfaces evidence; you make the call.

### General limitations, across every metric on this page

- **Association is not causation**, for any metric here.
- **A p-value's sensitivity to sample size** (mentioned per-metric above)
  applies everywhere a p-value appears.
- **Missing-value exclusion changes the analyzed population.** Every metric
  and `discover()` itself drop rows with a missing target, and — separately,
  per feature — rows missing that particular feature's value. If missingness
  isn't random (e.g. a sensor that fails more often under specific
  conditions), the rows that remain are a biased sample of the whole, and
  the evidence describes that biased subset, not the full population.
- **Repeated or clustered observations inflate apparent significance.** All
  of these tests assume independent observations. If your rows include
  repeated measurements of the same entity, or naturally cluster by group,
  subject, time, or event, ordinary p-values will tend to be more
  "significant" than they should be — SignaPy does not detect or correct
  for this.
- **No out-of-sample validation.** Every number on this page describes the
  data you passed in. None of it estimates how these relationships would
  hold up on new data — that's a different, and currently unimplemented,
  question (see ["stability analysis"](architecture.md#12-future-stability-analysis)).
- **Univariate evidence isn't combined performance.** Each `FeatureResult`
  describes one feature against the target, in isolation. It says nothing
  about how features interact, or how well a model using several of them
  together would perform — see the
  [interaction discovery](architecture.md#11-future-interaction-discovery) and
  [statistical philosophy](architecture.md#8-statistical-philosophy) sections
  of the architecture doc.
- **Testing many features raises a multiple-comparisons concern.** Running
  `discover()` across many columns and looking for the smallest p-values
  invites the same false-positive inflation as any multiple-testing
  scenario. SignaPy does not currently apply a correction (e.g.
  Bonferroni, Benjamini-Hochberg) — that judgment, for now, is yours.

## 7. Worked example: categorical feature

A feature `acquisition_channel` (`referral` / `organic` / `paid`) against a
binary target `converted` (`yes` / `no`), 20,000 rows per channel, with
conversion rates of 5.2%, 4.8%, and 4.4% respectively — small, realistic
differences.

```python
import pandas as pd
from signapy.metrics.association import cramers_v
from signapy.metrics.significance import chi_square
from signapy.metrics.lift import categorical_lift

# df has columns "acquisition_channel" and "converted" ("yes"/"no"), 60,000 rows
table = pd.crosstab(df["acquisition_channel"], df["converted"])
print(table)
```

```text
converted               no   yes
acquisition_channel
organic              19040   960
paid                 19120   880
referral             18960  1040
```

```python
cramers_v(table)
```
```text
0.01414553463493725
```

```python
chi_square(table)
```
```text
ChiSquareResult(statistic=14.005602240896359, p_value=0.0009093312484352182, dof=2)
```

```python
categorical_lift(df["acquisition_channel"], df["converted"], positive_class="yes")
```
```text
(CategoricalValueMetrics(value='referral', support=20000, positive_count=1040,
                          positive_rate=0.052, baseline_rate=0.048, lift=1.0833333333333333),
 CategoricalValueMetrics(value='organic', support=20000, positive_count=960,
                          positive_rate=0.048, baseline_rate=0.048, lift=1.0),
 CategoricalValueMetrics(value='paid', support=20000, positive_count=880,
                          positive_rate=0.044, baseline_rate=0.048, lift=0.9166666666666666))
```

| channel | support | positive_rate | baseline_rate | lift |
| --- | --- | --- | --- | --- |
| referral | 20,000 | 0.052 | 0.048 | 1.083 |
| organic | 20,000 | 0.048 | 0.048 | 1.000 |
| paid | 20,000 | 0.044 | 0.048 | 0.917 |

**Interpretation.** `p_value = 0.0009` is well below any conventional
threshold — with 60,000 rows, this data is strong evidence against
`acquisition_channel` and `converted` being independent. But
`cramers_v = 0.014` says the association is, in practical terms, very weak:
this is a textbook case of a statistically significant result with a small
effect size, driven almost entirely by sample size rather than a strong
underlying relationship. The lifts confirm this reading at the category
level: `referral` converts about 8% above baseline and `paid` about 8% below
it — real, well-supported differences (20,000 rows per category), but modest
ones, not a channel that dominates conversion. Whether an 8% relative
difference in conversion rate is worth acting on is a business question
Cramér's V and chi-square can't answer; that judgment belongs to you, with
this evidence as input.

## 8. Worked example: mixed categorical and continuous features

A categorical feature `category_context` (`low` / `medium` / `high`) and a
continuous feature `continuous_measurement`, both against a binary target
`outcome`, 30 rows, with one missing `continuous_measurement` value. This is
the same dataset as
[`examples/mixed_categorical_continuous_discovery.py`](../examples/mixed_categorical_continuous_discovery.py),
which builds and prints it in full — see the code there for the raw values.

```python
import signapy
from signapy.profiling import FeatureType

report = signapy.discover(
    df,
    target="outcome",
    positive_class=True,
    feature_types={
        "category_context": FeatureType.CATEGORICAL,
        "continuous_measurement": FeatureType.CONTINUOUS,
    },
)

report.feature("category_context")
report.feature("continuous_measurement")
```

```text
FeatureResult(feature='category_context', feature_type=FeatureType.CATEGORICAL,
    target_type=TargetType.BINARY, effect_size_method='cramers_v',
    effect_size=0.42..., n=30, missing_rate=0.0, test_method='chi_square',
    p_value=0.0273..., values=(...3 ValueResults...),
    details={'statistic': 7.2, 'dof': 2})

FeatureResult(feature='continuous_measurement', feature_type=FeatureType.CONTINUOUS,
    target_type=TargetType.BINARY, effect_size_method='point_biserial',
    effect_size=0.4756, n=29, missing_rate=0.0333..., test_method='point_biserial',
    p_value=0.009125..., values=None,
    details={'positive_n': 15, 'negative_n': 14, 'positive_mean': 24.08,
             'negative_mean': 16.3, 'mean_difference': 7.78})
```

**Interpretation.** `category_context` shows a moderate, statistically
detectable association (`cramers_v ≈ 0.42`, `p_value ≈ 0.027`) — its three
`values` (not shown in full above) tell you `low` sits well below baseline
and `high` well above it, per §5's interpretation. `continuous_measurement`
shows a comparable-strength, positive linear association
(`coefficient ≈ 0.48`, `p_value ≈ 0.009`): rows in the positive class
average about 7.8 units higher (`mean_difference`) than rows in the
negative class — one missing value (`missing_rate ≈ 0.033`) was excluded
from just this feature's analysis, per SignaPy's per-feature missing-data
policy (see [`docs/architecture.md`](architecture.md)).

Note what's missing from the continuous result: no per-range breakdown.
`category_context.values` has three entries you can inspect directly;
`continuous_measurement.values` is `None`. The point-biserial coefficient
and group means tell you the feature *as a whole* leans toward the positive
class at higher values, but not, for example, whether the relationship is
roughly linear across the whole range or concentrated above some threshold
— that's exactly the continuous-localization gap described in
[§5a](#5a-no-continuous-lift-yet).
