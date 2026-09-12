"""Categorize the OPM test suite without loading category-specific fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

TEST_CATEGORIES = {"unit", "gui", "integration"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark each test using its immediate category directory.

    Parameters
    ----------
    items : list[pytest.Item]
        Tests collected beneath the shared ``tests`` root.
    """
    tests_root = Path(__file__).parent.resolve()
    for item in items:
        try:
            relative_path = Path(str(item.path)).resolve().relative_to(tests_root)
        except ValueError:
            continue
        category = relative_path.parts[0]
        if category in TEST_CATEGORIES:
            item.add_marker(category)
