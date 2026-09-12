# Test suite

The suite is separated by execution boundary:

- `unit/`: isolated logic tested with pure values, mocks, or in-memory hardware.
- `gui/`: Qt widgets, application bootstrap, and Stage Explorer behavior.
- `integration/`: MMCore demo devices, complete event builders, storage, packaged
  drivers, and command-line workflows.

Run a category without importing the others:

```powershell
uv run pytest tests/unit
uv run pytest tests/gui
uv run pytest tests/integration
```

The equivalent markers are `unit`, `gui`, and `integration`. The full suite is:

```powershell
uv run pytest
```
