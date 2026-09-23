from importlib.metadata import version

from packaging.version import Version

import signapy


def test_version_is_pep440_prerelease():
    parsed = Version(signapy.__version__)
    assert parsed.is_prerelease


def test_version_matches_installed_metadata():
    assert version("signapy") == signapy.__version__
