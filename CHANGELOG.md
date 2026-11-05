# Changelog

## Unreleased

- `DiscoveryReport` gains `to_dict()`, a terminal `str()` table, a notebook
  HTML repr, and `plot()`.
- `plot()` renders SignaPy's signature mosaic (tile width = support, colour =
  lift) and returns a matplotlib `Figure`. Requires the optional `plot` extra.
- In Jupyter and Databricks, evaluating a report now shows the mosaic (inline
  PNG) above the summary table; without matplotlib it shows the table alone.
- `print(report)` ends with a pointer to `to_dict()`, and the README has new
  "Reading the report" and "Using SignaPy from an agent" sections.
- Mosaic conventions: rows are grouped by effect-size measure (V, ρ, r) and
  sorted by absolute effect size within each group; value labels are cut to
  20 characters; long feature names wrap onto extra lines and the row grows;
  `plot(max_features=N)` draws only the strongest `N` features, and the
  notebook view caps at 30. See the README's "Reading the mosaic".
- Every tile now carries its evidence: mid-width tiles show lift and support
  percentage, and tiles too narrow for text are listed in a strip under the
  rows (`feature: value lift × · support`, strongest lift first, at most 12).
- Continuous features now get value-level evidence: `discover()` cuts each into
  up to five quantile bins of roughly equal size (ties in the data can make
  them unequal, and scarce rows mean fewer bins) and reports support, target
  rate, baseline rate, and lift per bin, labelled as intervals like
  `(38.4, 44.1]`. When ties collapse bins, the cut is retried with fewer bins,
  and a feature with five or fewer distinct values gets one bin per value
  (labelled `0`, `1`, ...). `FeatureResult.values` is `None` only when a
  feature has too few rows or is constant.

## 0.1.0a2 — Polars input support

- `signapy.discover` accepts a Polars `DataFrame` (optional `polars` extra; no
  PyArrow required). Results match equivalent pandas input.

## 0.1.0a1 — first public alpha

Initial release. The API is experimental and may change.

- `signapy.discover` for a binary target against categorical, continuous, and
  ordinal features.
- Evidence to find, quantify, localize, and validate predictive signal; see
  `docs/metrics.md` for interpretation and `docs/architecture.md` for design.
