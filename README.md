<h1 align="center">
  <img src="https://raw.githubusercontent.com/Carmeisel101/signapy/main/docs/assets/signapy-logo.png" alt="SignaPy" width="480">
</h1>

<p align="center">
  <a href="https://pypi.org/project/signapy/"><img src="https://img.shields.io/pypi/v/signapy.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/signapy/"><img src="https://img.shields.io/pypi/pyversions/signapy.svg" alt="Python versions"></a>
  <a href="https://pypi.org/project/signapy/"><img src="https://img.shields.io/pypi/l/signapy.svg" alt="License"></a>
</p>

**SignaPy is an exploratory feature-discovery library for labeled datasets.**

Long-term mission: *find, quantify, localize, and validate predictive signal.*

> **Status: alpha (`0.1.0a2`).** `signapy.discover()` works against
> **binary targets** with **categorical, continuous, and ordinal features**
> today. Everything under [Planned next](#planned-next) — other
> feature/target types, interactions, stability, and more — is not
> implemented yet.

## The problem

Before modeling, data scientists want to know which features carry signal about
the target, and where within those features that signal lives. Today this means
picking the right test for each feature/target combination (Cramér's V? Phi?
Spearman? point-biserial?), running it by hand, and stitching the output together.

SignaPy aims to make that a single, task-aware workflow. It uses the feature
types you declare (or the categorical-only default), chooses an appropriate
method, and returns structured evidence. You shouldn't need to know ahead of
time which statistical test applies.

SignaPy discovers evidence of predictive signal.
It does not decide what your model should use.

It is **not** an AutoML framework, a feature-selection oracle, or a modeling
library, and it does not drop features based on p-values.

## Discovery hierarchy

| Level | Question |
| --- | --- |
| Profiling | What data am I working with? |
| Feature-level discovery | Does this feature contain signal about the target? |
| Value-level discovery | Where within this feature does the signal live? |
| Interaction discovery *(future)* | Does signal emerge from combinations of features or values? |
| Stability analysis *(future)* | Does the signal hold across samples, folds, time, or subgroups? |

For example, feature-level discovery might find that `acquisition_channel` is
associated with conversion. Value-level discovery then shows *where*:

```text
referral -> lift 1.91
organic  -> lift 1.28
paid     -> lift 0.73
```

## Installation

SignaPy is available from PyPI and supports Python 3.10 through 3.14.

```bash
pip install signapy
```

The current release is an alpha (`0.1.0a2`). To install the exact current
version, which is useful when reproducibility is important:

```bash
pip install signapy==0.1.0a2
```

See the [SignaPy PyPI project page](https://pypi.org/project/signapy/) for
available releases and package metadata.

### Development installation

These instructions are for contributors, or for anyone working from the
source repository rather than the PyPI release:

```bash
git clone https://github.com/Carmeisel101/signapy.git
cd signapy
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

Runtime dependencies are `numpy`, `pandas`, and `scipy`. SignaPy doesn't
depend on any ML framework. Polars is optional (`pip install "signapy[polars]"`,
or just install `polars` yourself) and PyArrow is never required.

## Usage

Categorical-only (every non-target column analyzed automatically):

```python
import signapy

report = signapy.discover(df, target="converted", positive_class=True)

report.features  # feature-level results
report.feature("acquisition_channel")  # one feature's evidence
report.feature("acquisition_channel").values  # value-level results
```

Mixed categorical and continuous — pass `feature_types` to select exactly
which columns to analyze and declare each one's semantic type:

```python
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

report.feature("continuous_measurement").effect_size  # point-biserial coefficient
report.feature("continuous_measurement").p_value  # its p-value
report.feature("continuous_measurement").values  # per-bin lift
```

An **ordinal** feature needs an explicit order — SignaPy never infers one,
not from the values and never alphabetically — declared via an ordered
pandas `Categorical`:

```python
df["severity"] = pd.Categorical(
    df["severity"],
    categories=["low", "medium", "high"],
    ordered=True,
)

report = signapy.discover(
    df,
    target="converted",
    positive_class=True,
    feature_types={"severity": FeatureType.ORDINAL},
)

report.feature("severity").effect_size  # Spearman's rho
report.feature(
    "severity"
).values  # support/rate/baseline/lift, in low->medium->high order
```

Pandas converts any value omitted from `categories=[...]` to a missing value;
under SignaPy's missing-data policy, that row is then excluded from this
feature's analysis. Make sure the declaration includes every intended level.

### Polars input

`discover()` accepts a Polars `DataFrame` directly — no `.to_pandas()` (and so
no PyArrow) needed. `target`, `positive_class`, and `feature_types` behave
exactly as for pandas, and results match the equivalent pandas input. The
frame is never mutated.

```python
import polars as pl

attempts = attempts.with_columns(
    pl.col("roof").cast(pl.Categorical),
    pl.col("severity").cast(pl.Enum(["low", "medium", "high"])),
)
report = signapy.discover(
    attempts,
    target="is_made",
    positive_class=1,
    feature_types={
        "kick_distance": "continuous",
        "roof": "categorical",
        "severity": "ordinal",
    },
)
```

Dtype mapping to know about:

| Polars dtype | Treated as |
| --- | --- |
| integer, float | numeric (continuous when declared); null and `NaN` are both missing |
| `String`, `Categorical` | categorical |
| `Enum` | categorical, or ordinal in the Enum's declared order |
| `Boolean` | not supported as a categorical or continuous feature (same as pandas `bool`); usable as a target |
| `Date`, `Datetime`, `Duration` | rejected as continuous/categorical (same as pandas) |

An ordinal feature must be an `Enum` (an ordered declaration); a plain
`Categorical` has no declared order and is rejected as ordinal.

### Reading the report

`discover()` returns a `DiscoveryReport` of numbers without drawing during
the analysis. The same report has several explicit and notebook-native views:

| You want | Use |
| --- | --- |
| Python access | `report.features`, `report.feature("name")`, `.values` |
| Plain data for scripts, JSON, or agents | `report.to_dict()` |
| A quick look in a terminal | `print(report)` |
| The mosaic and a summary table in Jupyter or Databricks | `report` as the last line of a cell (table only without matplotlib) |
| The mosaic (needs `pip install signapy[plot]`) | `report.plot()` |

`print(report)` shows one summary row per feature; the per-value and per-bin
lift lives in `to_dict()` and `.values`.

### Reading the mosaic

`report.plot()` draws SignaPy's signature view: one row per feature, one
tile per value, so you see *where* in each feature the signal lives, not just
how strong it is overall.

- **Tile width** is the value's support, its share of that feature's rows.
  A thin tile means little data behind it.
- **Tile color** is the value's lift against the baseline target rate, on a
  scale centered at 1 (white). Red is over-represented in the positive class,
  blue is under-represented. Wide tiles print the value, lift, and support
  percentage; mid-width tiles print the same three in a smaller size.
- **Continuous features** appear as up to five quantile bins of roughly equal
  size (heavy ties can make them unequal, so check each tile's width),
  labelled like `(38.4, 44.1]` (a feature with five or fewer distinct values
  gets one tile per value instead; a dominant value such as a column's many
  zeros gets its own tile, labelled `0`, with the rest binned around it), read left to right: steadily rising lift is a
  roughly linear relationship, a mid-range peak is non-monotonic.
- **Label panel** (left): the feature name, its effect size (`V` for Cramér's
  V, `ρ` for Spearman, `r` for point-biserial), and its `n`. Long names wrap
  onto extra lines and the row grows; nothing is cut off. Spaces and control
  characters that would make two names look alike are made visible (`␣`,
  `\n`).
- **Row order:** effect sizes from different measures are not strictly
  comparable, so rows are grouped by measure (the group with the strongest
  feature first) and sorted by absolute effect size within each group. Read
  rankings within a group, not across groups.
- **Long value labels** are cut to 20 characters with an ellipsis. The full
  labels are always in `report.to_dict()`.
- **Narrow tiles** (under about 0.6 inches) are too thin for text, so they
  are listed in a strip below the rows as `feature: value lift × · support`,
  strongest departure from baseline first. The strip shows at most 12; the
  rest are in `report.to_dict()`.
- **Large reports:** `report.plot(max_features=N)` draws only the `N` features
  with the largest absolute effect size and says so in the figure. The
  notebook view does this automatically for the top 30.

The baseline in the header badge is a range when features have different
missing rows, because each feature's baseline uses only its own valid rows.

### Using SignaPy from an agent

Nothing in SignaPy runs `to_dict()` for you, so an agent has to ask for it.
The reliable pattern is to run `discover()` and serialize the whole report:

```python
import json
import signapy

report = signapy.discover(df, target="converted", positive_class=True)
print(json.dumps(report.to_dict(), indent=2))
```

`to_dict()` returns plain, JSON-safe data: for each feature the
`feature_type`, `effect_size_method` and `effect_size`, `p_value`, `n`,
`missing_rate`, `details`, and `values` (per-category, per-level, or
per-continuous-bin `support`, `target_rate`, `baseline_rate`, and `lift`; `None`
when a feature has no value-level results). Category labels are always
encoded to JSON-safe data: timestamps and dates become ISO-8601 strings,
timedeltas ISO-8601 durations, tuples lists, and other objects (`Decimal`,
`UUID`, ...) their `str()`. Read effect sizes within their
method (`cramers_v`, `point_biserial`, `spearman_rho`), since they aren't
strictly comparable across methods, and treat everything as exploratory
association, not causation. See [`docs/metrics.md`](docs/metrics.md) for
interpretation.

`discover()` supports **categorical, continuous, and ordinal features
against a binary target** (see [v0.1 scope](#v01-scope) below):

- Without `feature_types` (the default), every non-target column of `df`
  with an `object`, pandas `string`, or pandas `category` dtype is analyzed
  as categorical; other dtypes raise a clear error.
- With `feature_types`, only the named columns are analyzed, each as its
  declared type (`FeatureType.CATEGORICAL`, `FeatureType.CONTINUOUS`, or
  `FeatureType.ORDINAL` — the only three supported so far). **Numeric
  columns are never silently treated as continuous, and categorical columns
  are never silently treated as ordinal** — an integer column might be a
  measured quantity, an ordinal level, or a category identifier, and
  SignaPy doesn't guess which; declaring it via `feature_types` is
  required, and an ordinal declaration additionally requires the column to
  already be an ordered `Categorical`.
- Categorical features get Cramér's V, a chi-square test, and per-category
  `values` (support, target rate, baseline rate, lift), in order of first
  appearance. Continuous features get a point-biserial correlation, its
  p-value, group counts/means in `details`, and `values` over up to five
  adaptive bins (usually quantile intervals like `(38.4, 44.1]`; see
  [Reading the mosaic](#reading-the-mosaic)), showing *where within the range*
  the signal lives; `values` is `None` only when the
  feature has too few rows or is constant (see
  [`docs/architecture.md`](docs/architecture.md)).
  Ordinal features get a Spearman rank correlation, its p-value, and the
  same per-level `values` as categorical — but in the feature's *declared*
  category order, not order of first appearance.

See [`docs/metrics.md`](docs/metrics.md) for how to choose between and
interpret `effect_size`/`p_value` across all three feature types
(including why Spearman, not a naive Pearson correlation on level codes, is
the right tool for ordinal data), and
[`examples/binary_categorical_discovery.py`](examples/binary_categorical_discovery.py) /
[`examples/mixed_categorical_continuous_discovery.py`](examples/mixed_categorical_continuous_discovery.py) /
[`examples/ordinal_binary_discovery.py`](examples/ordinal_binary_discovery.py)
for runnable, printed examples.

Also available: the low-level, pure statistical building blocks in
`signapy.metrics` (`association.cramers_v`, `association.point_biserial`,
`association.spearman_rho`, `significance.chi_square`,
`lift.categorical_lift`) that `discover()` is built on, the result models
(`signapy.results.FeatureResult`, `ValueResult`, `DiscoveryReport`), and the
semantic type enums (`signapy.profiling.FeatureType`, `TargetType`).

`signapy.discover()` characterizes univariate statistical relationships. It
does not train a classifier, select features automatically, or evaluate how
features perform in combination — that judgment, and any modeling built on
this evidence, stays yours.

## v0.1 scope

Implemented so far:

- **Target:** binary classification
- **Categorical features:** Cramér's V (effect size), chi-square test
  (p-value), and per-category support/target rate/baseline rate/lift
- **Continuous features:** point-biserial correlation (effect size and
  p-value), group counts/means, and per-bin support/target rate/baseline
  rate/lift over up to five adaptive bins (quantiles for ordinary continuous
  data, with tied low-cardinality and dominant values handled explicitly)
- **Ordinal features:** Spearman rank correlation (effect size and
  p-value), and per-level support/target rate/baseline rate/lift in
  declared category order; requires an explicit ordered `Categorical`
- **Feature selection:** explicit, via the `feature_types` argument;
  numeric/boolean columns are never inferred as a semantic type, and an
  order is never inferred for ordinal columns

See [`docs/architecture.md`](docs/architecture.md) for the full design,
including the missing-data and dtype policies this slice follows.

## Planned next

Not yet implemented: boolean features, semantic type overrides (e.g.
integer-coded categories), multiclass or continuous targets,
configurable continuous binning (bin count and strategy), additional value-level metrics
(risk difference, odds ratio, confidence intervals), interaction discovery,
and stability analysis. See the
[discovery hierarchy](#discovery-hierarchy) above and
[`docs/architecture.md`](docs/architecture.md) for how these fit the overall
design.

## Versioning

SignaPy uses [PEP 440](https://peps.python.org/pep-0440/) versions and will
follow [semantic versioning](https://semver.org/) for releases. Pre-release
development builds use `.devN` suffixes. The current alpha release is
`0.1.0a2`. The version is defined once, in `src/signapy/__init__.py`.

## License

MIT. See [LICENSE](LICENSE).
