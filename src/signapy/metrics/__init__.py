"""Statistical metrics: pure computations with no discovery logic.

Functions here take arrays/Series (or contingency tables) and return numbers
or small, explicit result types. They do not infer semantic types, choose
methods, build :mod:`signapy.results` objects, or select features; that is
the job of :mod:`signapy.discovery`.

Modules:

- ``association``: feature-level effect size
  (:func:`~signapy.metrics.association.cramers_v`)
- ``significance``: hypothesis testing
  (:func:`~signapy.metrics.significance.chi_square`)
- ``lift``: value-level localization
  (:func:`~signapy.metrics.lift.categorical_lift`)

Each module validates its own inputs independently and raises ``ValueError``
on invalid or degenerate input (e.g. empty data, a single category, a zero
expected frequency) rather than guessing a policy. Missing values are
rejected, not silently dropped or imputed.
"""
