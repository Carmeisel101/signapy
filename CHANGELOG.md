# Changelog

## Unreleased

- `signapy.discover` accepts a Polars `DataFrame` (optional `polars` extra; no
  PyArrow required). Results match equivalent pandas input.

## 0.1.0a1 — first public alpha

Initial release. The API is experimental and may change.

- `signapy.discover` for a binary target against categorical, continuous, and
  ordinal features.
- Evidence to find, quantify, localize, and validate predictive signal; see
  `docs/metrics.md` for interpretation and `docs/architecture.md` for design.
