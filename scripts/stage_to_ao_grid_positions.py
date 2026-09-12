"""Convert acquisition stage positions into a reusable AO grid."""

from __future__ import annotations

if __package__:
    from .ao_positions import build_parser, main
else:
    from ao_positions import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
