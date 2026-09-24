# SignaPy

**SignaPy is an exploratory feature-discovery library for labeled datasets.**

Long-term mission: *find, quantify, localize, and validate predictive signal.*

> **Status: pre-alpha (`0.1.0.dev0`).** The package structure, type vocabulary,
> and result models exist. **No discovery functionality is implemented yet.**
> Everything under [Planned usage](#planned-usage) describes intended behavior,
> not current behavior.

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

## Planned usage

> **Not implemented yet.** This is the target API.

```python
import signapy

report = signapy.discover(df, target="converted")

report.features  # feature-level results
report.feature("acquisition_channel")  # one feature's evidence
report.feature("acquisition_channel").values  # value-level results
```

Available now: the result models that this API will return
(`signapy.results.FeatureResult`, `ValueResult`, `DiscoveryReport`), the
semantic type enums (`signapy.profiling.FeatureType`, `TargetType`), and the
low-level, pure statistical building blocks in `signapy.metrics` —
`association.cramers_v`, `significance.chi_square`, and
`lift.categorical_lift` — which are not yet wired into `discover()`. See
[`docs/metrics.md`](docs/metrics.md) for how to choose between and interpret
these metrics.

## Planned v0.1 scope

The first capability is one end-to-end vertical slice:

- **Target:** binary classification
- **Feature:** categorical
- **Feature-level:** Cramér's V (effect size) and chi-square test (p-value)
- **Value-level:** support, target count, target rate, baseline target rate,
  and categorical lift

More feature types, target types, and methods follow once this slice has
proven the architecture. See [`docs/architecture.md`](docs/architecture.md) for
the full design.

## Versioning

SignaPy uses [PEP 440](https://peps.python.org/pep-0440/) versions and will
follow [semantic versioning](https://semver.org/) for releases. Pre-release
development builds use `.devN` suffixes (currently `0.1.0.dev0`). The version
is defined once, in `src/signapy/__init__.py`.

## License

MIT. See [LICENSE](LICENSE).
