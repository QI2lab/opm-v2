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

The equivalent markers are `unit`, `gui`, and `integration`. Windows CI runs all
three categories in separate processes and accumulates their coverage. This
keeps native MMCore and Qt initialization isolated from other test collections.

The full suite can also be collected in one process:

```powershell
uv run pytest
```
