"""Stateful NI-DAQ test double for simulated OPM acquisitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ


class MockDAQError(RuntimeError):
    """Base exception raised by the stateful mock DAQ."""


class MockDAQRoutingError(MockDAQError):
    """Match NI-DAQmx error -89130 while a device reset is incomplete."""

    error_code = -89130


class MockDAQTaskError(MockDAQError):
    """Report use of a cleared, invalid, or incorrectly ordered mock task."""


@dataclass
class MockDAQTask:
    """Represent one NI-DAQmx task handle and its lifecycle."""

    name: str
    generation: int
    routes: tuple[str, ...] = ()
    valid: bool = True
    running: bool = False

    def start(self, current_generation: int) -> None:
        """Start a valid task created for the active device generation."""
        self._validate(current_generation)
        self.running = True

    def stop(self, current_generation: int) -> None:
        """Stop a valid task created for the active device generation."""
        self._validate(current_generation)
        self.running = False

    def clear(self) -> None:
        """Invalidate the task handle and release its routes."""
        self.running = False
        self.valid = False

    def _validate(self, current_generation: int) -> None:
        """Reject cleared handles and handles invalidated by a device reset.

        Raises
        ------
        MockDAQTaskError
            If the task is cleared or belongs to an earlier device generation.
        """
        if not self.valid or self.generation != current_generation:
            raise MockDAQTaskError(f"{self.name} is an invalid or cleared task")


class MockOPMNIDAQ(OPMNIDAQ):
    """Model the complete OPM DAQ lifecycle without loading NI hardware.

    Waveform and acquisition-parameter calculations are inherited from
    :class:`OPMNIDAQ`.  This class replaces the original boolean-only simulated
    task behavior with explicit task handles, routes, reset generations, operation
    ordering, and deterministic failure injection.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize waveform state and an idle mock NI device."""
        kwargs["simulate"] = True
        super().__init__(*args, **kwargs)
        self.operation_log: list[str] = []
        self._device_generation = 0
        self._reset_in_progress = False
        self._hold_next_reset = False
        self._injected_failures: dict[str, list[BaseException]] = {}

    @property
    def reset_in_progress(self) -> bool:
        """Whether the mock device is unavailable for task routing."""
        return self._reset_in_progress

    @property
    def tasks(self) -> tuple[MockDAQTask, ...]:
        """All currently allocated mock task handles."""
        return tuple(
            task
            for task in (self._task_di, self._task_do, self._task_ao)
            if isinstance(task, MockDAQTask)
        )

    @property
    def reserved_routes(self) -> tuple[str, ...]:
        """Routes owned by valid current-generation tasks."""
        return tuple(
            route
            for task in self.tasks
            if task.valid and task.generation == self._device_generation
            for route in task.routes
        )

    def hold_next_reset(self) -> None:
        """Leave the next reset incomplete until :meth:`complete_reset` is called."""
        self._hold_next_reset = True

    def complete_reset(self) -> None:
        """Make a held mock device reset ready for new task routing."""
        if self._reset_in_progress:
            self.operation_log.append("reset:complete")
        self._reset_in_progress = False

    def inject_failure(self, operation: str, exception: BaseException) -> None:
        """Raise ``exception`` the next time the named operation is attempted."""
        self._injected_failures.setdefault(str(operation), []).append(exception)

    def reset(self) -> None:
        """Invalidate every task and reset analog and digital outputs."""
        self.operation_log.append("reset:begin")
        self._maybe_fail("reset")
        self.clear_tasks()
        self._device_generation += 1
        self._reset_in_progress = True
        self._ao_waveform = np.asarray(
            [self._ao_neutral_positions], dtype=float
        )
        self._do_waveform = np.zeros(
            (1, self._num_do_channels), dtype=np.uint8
        )
        if self._hold_next_reset:
            self._hold_next_reset = False
            return
        self.complete_reset()

    def reset_ao_channels(self) -> None:
        """Clear tasks and set both simulated analog outputs to neutral."""
        self.operation_log.append("reset:ao")
        self._maybe_fail("reset_ao")
        self.clear_tasks()
        self._ao_waveform = np.asarray(
            [self._ao_neutral_positions], dtype=float
        )

    def reset_do_channels(self) -> None:
        """Clear tasks and set every simulated digital output low."""
        self.operation_log.append("reset:do")
        self._maybe_fail("reset_do")
        self.clear_tasks()
        self._do_waveform = np.zeros(
            (1, self._num_do_channels), dtype=np.uint8
        )

    def program_daq_waveforms(self) -> None:
        """Allocate routed DI, DO, and AO tasks for generated waveforms.

        Raises
        ------
        MockDAQRoutingError
            If task routing is attempted before a held reset completes.
        MockDAQTaskError
            If analog or digital waveforms have not been generated.
        """
        self.operation_log.append("program:attempt")
        self._maybe_fail("program")
        if self._reset_in_progress:
            raise MockDAQRoutingError(
                "Device not available for routing while reset is in progress "
                "(NI-DAQmx status -89130)"
            )
        if np.asarray(self._do_waveform).ndim != 2:
            raise MockDAQTaskError(
                "Digital waveform must be generated before DAQ programming"
            )
        if np.asarray(self._ao_waveform).ndim != 2:
            raise MockDAQTaskError(
                "Analog waveform must be generated before DAQ programming"
            )

        self.clear_tasks()
        generation = self._device_generation
        self._task_di = MockDAQTask(
            "TaskDI",
            generation,
            routes=(
                self._channel_di_change_trigger,
                self._channel_di_start_trigger,
            ),
        )
        self._task_do = MockDAQTask(
            "TaskDO",
            generation,
            routes=(self._channel_di_change_trigger,),
        )
        self._task_ao = MockDAQTask(
            "TaskAO",
            generation,
            routes=(
                self._channel_di_trigger_from_camera,
                self._channel_ao_start_trigger,
            ),
        )
        self._programmed = True
        self._programmed_signature = self._acquisition_signature()
        self.operation_log.append("program:complete")

    def start_waveform_playback(self) -> None:
        """Start every programmed task in dependency-safe order.

        Raises
        ------
        MockDAQRoutingError
            If playback is requested before a held reset completes.
        MockDAQTaskError
            If the tasks are absent, stale, or not programmed.
        """
        self.operation_log.append("start:attempt")
        self._maybe_fail("start")
        if self._reset_in_progress:
            raise MockDAQRoutingError(
                "Device not available while reset is in progress "
                "(NI-DAQmx status -89130)"
            )
        if not self.programmed() or len(self.tasks) != 3:
            raise MockDAQTaskError("DAQ waveforms must be programmed before start")
        for task in (self._task_ao, self._task_do, self._task_di):
            task.start(self._device_generation)
        self._running = True
        self.operation_log.append("start:complete")

    def stop_waveform_playback(self) -> None:
        """Stop every running task without clearing its handle."""
        self.operation_log.append("stop:attempt")
        self._maybe_fail("stop")
        if not self._running:
            self.operation_log.append("stop:idle")
            return
        for task in (self._task_di, self._task_do, self._task_ao):
            if isinstance(task, MockDAQTask):
                task.stop(self._device_generation)
        self._running = False
        self.operation_log.append("stop:complete")

    def clear_tasks(self) -> None:
        """Stop, invalidate, and release every allocated mock task."""
        self.operation_log.append("clear:attempt")
        self._maybe_fail("clear")
        if self._running:
            self.stop_waveform_playback()
        for task_name in ("_task_di", "_task_do", "_task_ao"):
            task = getattr(self, task_name, None)
            if isinstance(task, MockDAQTask):
                task.clear()
            setattr(self, task_name, None)
        self._running = False
        self._programmed = False
        self._programmed_signature = None
        self.operation_log.append("clear:complete")

    def _maybe_fail(self, operation: str) -> None:
        """Raise and consume the next deterministic failure for an operation."""
        failures = self._injected_failures.get(operation)
        if not failures:
            return
        exception = failures.pop(0)
        if not failures:
            self._injected_failures.pop(operation, None)
        raise exception
