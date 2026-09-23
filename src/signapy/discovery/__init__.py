"""Discovery workflows: the user-facing layer of SignaPy.

This package answers discovery *questions* rather than exposing individual
statistical tests. It chooses an appropriate method from the feature and
target types, calls into :mod:`signapy.metrics`, and packages the evidence
into :mod:`signapy.results` models.

Planned modules (not yet implemented):

- ``feature``: feature-level discovery ("Does this feature contain signal?")
- ``value``: value-level discovery ("Where within this feature does the
  signal live?")
"""
