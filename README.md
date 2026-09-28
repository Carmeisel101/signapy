# SignaPy

**SignaPy is an exploratory feature-discovery library for labeled datasets.**

Long-term mission: *find, quantify, localize, and validate predictive signal.*

> **Status: pre-alpha (`0.1.0.dev0`).** `signapy.discover()` works for one
> slice today: **a binary target against categorical features.** Everything
> under [Planned next](#planned-next) — other feature/target types,
> interactions, stability, and more — is not implemented yet.

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

```python
import signapy

report = signapy.discover(df, target="converted", positive_class=True)

report.features  # feature-level results
report.feature("acquisition_channel")  # one feature's evidence
report.feature("acquisition_channel").values  # value-level results
```

`discover()` currently supports **categorical features against a binary
target** (see [v0.1 scope](#v01-scope) below): every non-target column of
`df` with an `object`, pandas `string`, or pandas `category` dtype is
analyzed; other dtypes (numeric, boolean) raise a clear error rather than
being silently reinterpreted. See [`docs/metrics.md`](docs/metrics.md) for
how to choose between and interpret `effect_size` (Cramér's V), `p_value`
(chi-square), and the value-level `lift`, and
[`examples/binary_categorical_discovery.py`](examples/binary_categorical_discovery.py)
for a runnable, printed example.

Also available: the low-level, pure statistical building blocks in
`signapy.metrics` (`association.cramers_v`, `significance.chi_square`,
`lift.categorical_lift`) that `discover()` is built on, the result models
(`signapy.results.FeatureResult`, `ValueResult`, `DiscoveryReport`), and the
semantic type enums (`signapy.profiling.FeatureType`, `TargetType`).

## v0.1 scope

The first capability is one end-to-end vertical slice, implemented:

- **Target:** binary classification
- **Feature:** categorical
- **Feature-level:** Cramér's V (effect size) and chi-square test (p-value)
- **Value-level:** support, target count, target rate, baseline target rate,
  and categorical lift

See [`docs/architecture.md`](docs/architecture.md) for the full design,
including the missing-data and categorical-dtype policies this slice
follows.

## Planned next

Not yet implemented: boolean features, semantic type overrides (e.g.
integer-coded categories), continuous and ordinal features, multiclass
targets, additional value-level metrics (risk difference, odds ratio,
confidence intervals), interaction discovery, and stability analysis. See
the [discovery hierarchy](#discovery-hierarchy) above and
[`docs/architecture.md`](docs/architecture.md) for how these fit the overall
design.

## Versioning

SignaPy uses [PEP 440](https://peps.python.org/pep-0440/) versions and will
follow [semantic versioning](https://semver.org/) for releases. Pre-release
development builds use `.devN` suffixes (currently `0.1.0.dev0`). The version
is defined once, in `src/signapy/__init__.py`.

## License

MIT. See [LICENSE](LICENSE).
