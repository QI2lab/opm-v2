"""Cooperative cancellation at AO software boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class AOCancelled(Exception):
    """A user requested that AO stop before its next operation."""


def check_ao_stop(stop_requested: Callable[[], bool] | None) -> None:
    """Raise AOCancelled when the supplied stop predicate is set.

    Raises
    ------
    AOCancelled
        When cancellation has been requested.
    """
    if stop_requested is not None and stop_requested():
        raise AOCancelled("AO stopped by user")
