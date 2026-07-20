"""Provide console-printing helpers shared by the OPM modules."""

DEBUG_SEPARATOR = "-" * 72


def print_block(header: str, *lines: object) -> None:
    """Print a visually separated console message block.

    Parameters
    ----------
    header : str
        Block heading.
    *lines : object
        Values to print beneath the heading.
    """
    print(f"\n{DEBUG_SEPARATOR}")
    print(f"----- {header} -----")
    for line in lines:
        print(line)
    print(DEBUG_SEPARATOR)


def debug(header: str, *lines: object, enabled: bool = True) -> None:
    """Print a debug block when enabled.

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
        print_block(f"DEBUGGING: {header}", *lines)


def info(header: str, *lines: object) -> None:
    """Print a status block.

    Parameters
    ----------
    header : str
        Block heading.
    *lines : object
        Values to print beneath the heading.
    """
    print_block(header, *lines)


def warning(header: str, *lines: object) -> None:
    """Print a warning block.

    Parameters
    ----------
    header : str
        Block heading.
    *lines : object
        Values to print beneath the heading.
    """
    print_block(f"WARNING: {header}", *lines)
