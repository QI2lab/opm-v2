"""Test integration with the packaged Windows Picard shutter driver."""

from __future__ import annotations

import sys

import pytest

from opm_v2.hardware.PicardShutter import PiUsbNet


@pytest.mark.skipif(sys.platform != "win32", reason="Picard driver is Windows-only")
def test_packaged_picard_driver_loads() -> None:
    """Load the vendored .NET assembly from the package directory."""
    assert PiUsbNet is not None
