# Changelog

## Unreleased

- `DiscoveryReport` gains `to_dict()`, a terminal `str()` table, a notebook
  HTML repr, and `plot()`.
- `plot()` renders SignaPy's signature mosaic (tile width = support, colour =
  lift) and returns a matplotlib `Figure`. Requires the optional `plot` extra.

## 0.1.0a2 — Polars input support

- `signapy.discover` accepts a Polars `DataFrame` (optional `polars` extra; no
  PyArrow required). Results match equivalent pandas input.

## 0.1.0a1 — first public alpha

Initial release. The API is experimental and may change.

- `signapy.discover` for a binary target against categorical, continuous, and
  ordinal features.
- Evidence to find, quantify, localize, and validate predictive signal; see
  `docs/metrics.md` for interpretation and `docs/architecture.md` for design.
