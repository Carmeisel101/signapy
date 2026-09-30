# SignaPy

**SignaPy is an exploratory feature-discovery library for labeled datasets.**

Long-term mission: *find, quantify, localize, and validate predictive signal.*

> **Status: pre-alpha (`0.1.0.dev0`).** `signapy.discover()` works against
> **binary targets** with **categorical and continuous features** today.
> Everything under [Planned next](#planned-next) — other feature/target
> types, interactions, stability, and more — is not implemented yet.

## The problem

Before modeling, data scientists want to know which features carry signal about
the target, and where within those features that signal lives. Today this means
picking the right test for each feature/target combination (Cramér's V? Phi?
Spearman? point-biserial?), running it by hand, and stitching the output together.

SignaPy aims to make that a single, task-aware workflow. It infers what kind of
feature and target it is looking at, chooses an appropriate method, and returns
structured evidence. You shouldn't need to know ahead of time which statistical
test applies.

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

## Installation (development)

SignaPy supports Python 3.10 to 3.14. It isn't on PyPI yet.

```bash
git clone <repo-url> signapy
cd signapy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Runtime dependencies are `numpy`, `pandas`, and `scipy`. SignaPy doesn't
depend on any ML framework.

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
report.feature("continuous_measurement").values  # always None (see below)
```

`discover()` supports **categorical and continuous features against a
binary target** (see [v0.1 scope](#v01-scope) below):

- Without `feature_types` (the default), every non-target column of `df`
  with an `object`, pandas `string`, or pandas `category` dtype is analyzed
  as categorical; other dtypes raise a clear error.
- With `feature_types`, only the named columns are analyzed, each as its
  declared type (`FeatureType.CATEGORICAL` or `FeatureType.CONTINUOUS` —
  the only two supported so far). **Numeric columns are never silently
  treated as continuous** — an integer column might be a measured quantity,
  an ordinal level, or a category identifier, and SignaPy doesn't guess
  which; declaring it via `feature_types` is required.
- Categorical features get Cramér's V, a chi-square test, and per-category
  `values` (support, target rate, baseline rate, lift). Continuous features
  get a point-biserial correlation, its p-value, and group counts/means in
  `details` — but **`values` is always `None`** for continuous features:
  localizing *where within a continuous range* the signal lives needs
  binning, which SignaPy doesn't do automatically (see
  [`docs/architecture.md`](docs/architecture.md)).

See [`docs/metrics.md`](docs/metrics.md) for how to choose between and
interpret `effect_size`/`p_value` for both feature types, and
[`examples/binary_categorical_discovery.py`](examples/binary_categorical_discovery.py) /
[`examples/mixed_categorical_continuous_discovery.py`](examples/mixed_categorical_continuous_discovery.py)
for runnable, printed examples.

Also available: the low-level, pure statistical building blocks in
`signapy.metrics` (`association.cramers_v`, `association.point_biserial`,
`significance.chi_square`, `lift.categorical_lift`) that `discover()` is
built on, the result models (`signapy.results.FeatureResult`, `ValueResult`,
`DiscoveryReport`), and the semantic type enums
(`signapy.profiling.FeatureType`, `TargetType`).

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
  p-value) and group counts/means; no value-level localization yet
- **Feature selection:** explicit, via the `feature_types` argument;
  numeric/boolean columns are never inferred as a semantic type

See [`docs/architecture.md`](docs/architecture.md) for the full design,
including the missing-data and dtype policies this slice follows.

## Planned next

Not yet implemented: boolean features, semantic type overrides (e.g.
integer-coded categories), ordinal features, multiclass or continuous
targets, continuous-feature localization (binning), additional value-level
metrics (risk difference, odds ratio, confidence intervals), interaction
discovery, and stability analysis. See the
[discovery hierarchy](#discovery-hierarchy) above and
[`docs/architecture.md`](docs/architecture.md) for how these fit the overall
design.

## Versioning

SignaPy uses [PEP 440](https://peps.python.org/pep-0440/) versions and will
follow [semantic versioning](https://semver.org/) for releases. Pre-release
development builds use `.devN` suffixes (currently `0.1.0.dev0`). The version
is defined once, in `src/signapy/__init__.py`.

## License

MIT. See [LICENSE](LICENSE).
