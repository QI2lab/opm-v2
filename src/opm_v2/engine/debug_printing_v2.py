"""Small console-printing helpers shared by the v2 OPM modules."""

DEBUG_SEPARATOR = "-" * 72


def print_block(header: str, *lines: object) -> None:
    """Print a visually separated console message block."""
    print(f"\n{DEBUG_SEPARATOR}")
    print(f"----- {header} -----")
    for line in lines:
        print(line)
    print(DEBUG_SEPARATOR)


def debug(header: str, *lines: object, enabled: bool = True) -> None:
    """Print a debug block when ``enabled`` is true."""
    if enabled:
        print_block(f"DEBUGGING: {header}", *lines)


def info(header: str, *lines: object) -> None:
    """Print a non-debug status block that should always be visible."""
    print_block(header, *lines)


def warning(header: str, *lines: object) -> None:
    """Print a warning block that should always be visible."""
    print_block(f"WARNING: {header}", *lines)
