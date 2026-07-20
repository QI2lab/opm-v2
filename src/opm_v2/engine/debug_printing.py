"""Provide standard logging helpers shared by the OPM modules."""

from __future__ import annotations

import logging

LOGGER = logging.getLogger("opm_v2")


def _message(header: str, lines: tuple[object, ...]) -> str:
    """Build one structured log message.

    Parameters
    ----------
    header : str
        Block heading.
    *lines : object
        Values to print beneath the heading.

    Returns
    -------
    str
        Header and detail values joined into one log message.
    """
    details = " | ".join(str(line) for line in lines)
    return f"{header}: {details}" if details else header


def debug(header: str, *lines: object, enabled: bool = True) -> None:
    """Log a debug message when enabled.

    Parameters
    ----------
    header : str
        Block heading.
    *lines : object
        Values to print beneath the heading.
    enabled : bool
        Whether to emit the block.
    """
    if enabled:
        LOGGER.debug(_message(header, lines))


def info(header: str, *lines: object) -> None:
    """Log a status message.

    Parameters
    ----------
    header : str
        Block heading.
    *lines : object
        Values to print beneath the heading.
    """
    LOGGER.info(_message(header, lines))


def warning(header: str, *lines: object) -> None:
    """Log a warning message.

    Parameters
    ----------
    header : str
        Block heading.
    *lines : object
        Values to print beneath the heading.
    """
    LOGGER.warning(_message(header, lines))
