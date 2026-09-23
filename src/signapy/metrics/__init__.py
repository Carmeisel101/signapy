"""Statistical metrics: pure computations with no discovery logic.

Functions here take arrays/Series (or contingency tables) and return numbers.
They do not infer types, choose methods, or build result objects; that is the
job of :mod:`signapy.discovery`.

Planned modules (not yet implemented):

- ``association``: feature-level effect sizes (e.g. Cramér's V, Phi)
- ``significance``: hypothesis tests (e.g. chi-square)
- ``lift``: value-level localization (support, target rate, lift)
"""
