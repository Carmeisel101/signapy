# Changelog

## Unreleased

- `DiscoveryReport` gains `to_dict()`, a terminal `str()` table, a notebook
  HTML repr, and `plot()`.
- `plot()` renders SignaPy's signature mosaic (tile width = support, colour =
  lift) and returns a matplotlib `Figure`. Requires the optional `plot` extra.
- Continuous features now get value-level evidence: `discover()` cuts each into
  up to five equal-count quantile bins (fewer when rows or distinct values are
  scarce) and reports support, target rate, baseline rate, and lift per bin,
  labelled as intervals like `(38.4, 44.1]`. `FeatureResult.values` is `None`
  only when a feature can't be split into at least two bins.

## 0.1.0a2 — Polars input support

- `signapy.discover` accepts a Polars `DataFrame` (optional `polars` extra; no
  PyArrow required). Results match equivalent pandas input.

## 0.1.0a1 — first public alpha

Initial release. The API is experimental and may change.

- `signapy.discover` for a binary target against categorical, continuous, and
  ordinal features.
- Evidence to find, quantify, localize, and validate predictive signal; see
  `docs/metrics.md` for interpretation and `docs/architecture.md` for design.
