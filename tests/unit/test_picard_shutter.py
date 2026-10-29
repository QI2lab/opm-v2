"""Keep simulated shutter control independent of the vendor .NET runtime."""

import sys
from unittest.mock import patch

import pytest

from opm_v2.hardware import PicardShutter as shutter_module


def test_simulated_shutter_never_loads_vendor_runtime():
    """Operate the simulated shutter even when the physical loader is unavailable."""
    with patch.object(
        shutter_module, "_load_piusb", side_effect=AssertionError("Vendor SDK loaded")
    ):
        shutter = shutter_module.PicardShutter(simulate=True)
        shutter.openShutter()
        assert shutter.state == "Open"
        shutter.closeShutter()
        assert shutter.state == "Closed"
        assert shutter.is_connected


def test_physical_shutter_reports_missing_runtime(monkeypatch):
    """Retain a useful driver error when physical mode cannot import pythonnet."""
    monkeypatch.setattr(shutter_module, "PiUsbNet", None)
    monkeypatch.setattr(shutter_module, "_piusb_import_error", None)
    monkeypatch.setitem(sys.modules, "clr", None)
    with pytest.raises(RuntimeError, match="PiUsbNet could not be loaded") as error:
        shutter_module.PicardShutter(shutter_id=1)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
