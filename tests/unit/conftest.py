"""Keep unit tests isolated from MMCore, TensorStore, and GUI fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.APump import APump
from opm_v2.hardware.ElveFlow import OB1Controller
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.PicardShutter import PicardShutter

_SINGLETON_HARDWARE = (
    AOMirror,
    APump,
    OB1Controller,
    OPMNIDAQ,
    PicardShutter,
)


@pytest.fixture(autouse=True)
def reset_acquisition_singletons() -> Iterator[None]:
    """Reset only the in-process hardware fakes used by unit tests."""
    for hardware_class in _SINGLETON_HARDWARE:
        hardware_class.reset_instance()
    yield
    for hardware_class in _SINGLETON_HARDWARE:
        hardware_class.reset_instance()
