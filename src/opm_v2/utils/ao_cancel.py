"""Cooperative cancellation at AO software boundaries."""


class AOCancelled(Exception):
    """A user requested that AO stop before its next operation."""


def check_ao_stop(stop_requested):
    """Raise AOCancelled when the supplied stop predicate is set.

    Raises
    ------
    AOCancelled
        When cancellation has been requested.
    """
    if stop_requested is not None and stop_requested():
        raise AOCancelled("AO stopped by user")
