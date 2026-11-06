from importlib.metadata import version

from packaging.version import Version

import signapy


def test_version_is_a_pep440_release():
    parsed = Version(signapy.__version__)
    assert not parsed.is_prerelease
    assert not parsed.is_devrelease


def test_version_matches_installed_metadata():
    assert version("signapy") == signapy.__version__
