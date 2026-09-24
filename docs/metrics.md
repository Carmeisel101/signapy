# Metrics Guide

This page is for choosing which SignaPy metric answers your question, and for
reading the numbers correctly once you have them. It doesn't repeat the
function-level details (exact arguments, validation, error messages) that live
in each function's docstring, and it doesn't cover package structure or design
rationale — see [`docs/architecture.md`](architecture.md) for that. See the
[README](../README.md) for installation and the project's overall scope.

> **Status:** the functions described here — `cramers_v`, `chi_square`, and
> `categorical_lift` — are implemented in `signapy.metrics` today. They are
> low-level, pure functions: you call them directly on a contingency table or
> on `feature`/`target` arrays. They are not yet wired into a higher-level
> `signapy.discover()` workflow.

## 1. Overview

SignaPy's metrics split into two levels that answer different questions:

| Level | Metric | Question |
| --- | --- | --- |
| Feature-level | `cramers_v` | How strong is the overall association? |
| Feature-level | `chi_square` | Is the observed association unlikely under independence? |
| Value-level | `categorical_lift` | Which individual categories are above or below baseline? |

Feature-level metrics summarize a categorical feature *as a whole* against a
target. Value-level metrics break that summary down by category. Neither
level replaces the other: a feature can show a strong overall association
that turns out to be driven by one category, or a weak overall association
that still hides one category worth investigating. Read feature-level and
value-level evidence together, not as substitutes.

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
  see [`categorical_lift`](#4-categorical-lift).

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
[worked example](#6-worked-example) below for exactly this case.

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

## 4. Categorical lift

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
there's signal to look for, per the [worked example](#6-worked-example).

**Watch support, not just lift.** A `lift` of `3.0` computed from 4 rows is
much weaker evidence than a `lift` of `1.2` computed from 40,000 rows. High
lift with low `support` is exactly the kind of thing that looks dramatic and
turns out to be noise; treat it as a hypothesis to check against more data,
not a conclusion.

**Limitations:** `categorical_lift` does not run a significance test (no
p-value, no confidence interval) and does not imply causation — a category
with high lift is *associated* with the positive class in this sample, not
necessarily a cause of it.

## 5. How to use the metrics together

A compact workflow:

1. **`chi_square`** — is there evidence against independence at all?
2. **`cramers_v`** — if so, how strong is that association in practical
   terms?
3. **`categorical_lift`** — which specific categories are pulling the
   positive rate up or down?

Common combinations and how to read them:

| Pattern | Reading |
| --- | --- |
| Low `p_value` + low `cramers_v` | Statistically detectable, practically weak. With enough data, real but tiny effects become significant; don't treat this as an important feature on its own. |
| Meaningful `cramers_v` + informative lifts | Stronger evidence overall, with `categorical_lift` telling you which categories to look at or act on. |
| Large lift with small `support` | Hypothesis-generating, not strong evidence. Worth a closer look (more data, a stability check across folds/time — [planned](architecture.md#12-future-stability-analysis)), not an immediate conclusion. |

None of these numbers, alone or combined, tell you whether to keep a feature
in a model — see the [statistical philosophy](architecture.md#8-statistical-philosophy)
in the architecture doc. SignaPy surfaces evidence; you make the call.

## 6. Worked example

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
