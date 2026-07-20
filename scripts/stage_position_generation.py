"""Backward-compatible entry point for stage-position generation."""

from __future__ import annotations

if __package__:
    from .stage_positions import build_parser, main
else:
    from stage_positions import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
