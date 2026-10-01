"""Statistical metrics: pure computations with no discovery logic.

Functions here take arrays/Series (or contingency tables) and return numbers
or small, explicit result types. They do not infer semantic types, choose
methods, build :mod:`signapy.results` objects, or select features; that is
the job of :mod:`signapy.discovery`.

Modules:

- ``association``: feature-level effect size, for a categorical feature
  (:func:`~signapy.metrics.association.cramers_v`), a continuous one
  (:func:`~signapy.metrics.association.point_biserial`), or an ordinal one
  (:func:`~signapy.metrics.association.spearman_rho`)
- ``significance``: hypothesis testing
  (:func:`~signapy.metrics.significance.chi_square`)
- ``lift``: value-level localization for categorical and ordinal features
  (:func:`~signapy.metrics.lift.categorical_lift`); no continuous equivalent
  yet (see ``docs/architecture.md``)

Each module validates its own inputs independently and raises ``ValueError``
on invalid or degenerate input (e.g. empty data, a single category, a zero
expected frequency) rather than guessing a policy. Missing values are
rejected, not silently dropped or imputed.
"""
