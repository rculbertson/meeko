"""Tests for meeko.leds.LedController.

The controller drives an internal worker thread that issues blocking
USB control transfers. Tests substitute a fake device that records
``ctrl_transfer`` calls; we then poll until the queue drains so we can
assert against a stable view of what the worker did.
"""

from __future__ import annotations

import struct
import threading
import time

import pytest

from meeko import leds
from meeko.leds import (
    _RESID_GPO,
    EFFECT_BREATH,
    EFFECT_OFF,
    EFFECT_SOLID,
    PALETTE,
    LedController,
    LedState,
    XvfLedDevice,
)

# Writes the worker issues once at startup before its action loop:
# brightness, speed, gammify. Brightness and speed only affect the
# breath effect (ignored by others) so they're set once and left alone.
_STARTUP_WRITES = 3


class FakeUsbDevice:
    """Stand-in for a pyusb Device. Records every control transfer."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int, bytes]] = []
        self._lock = threading.Lock()

    def ctrl_transfer(
        self,
        bm_request_type: int,
        b_request: int,
        w_value: int,
        w_index: int,
        data: bytes,
        timeout: int,
    ) -> None:
        with self._lock:
            self.calls.append(
                (bm_request_type, b_request, w_value, w_index, bytes(data))
            )

    def snapshot(self) -> list[tuple[int, int, int, int, bytes]]:
        with self._lock:
            return list(self.calls)


def _wait_for_calls(fake: FakeUsbDevice, n: int, timeout: float = 1.0) -> None:
    """Block until at least n ctrl_transfers have been recorded."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if len(fake.snapshot()) >= n:
            return
        time.sleep(0.01)
    raise AssertionError(
        f"Expected at least {n} ctrl_transfer calls, got {len(fake.snapshot())}"
    )


def _make_controller(fake: FakeUsbDevice) -> LedController:
    return LedController(device_factory=lambda: XvfLedDevice(fake))


def _start_and_wait(fake: FakeUsbDevice, controller: LedController) -> None:
    """Start the controller and wait until the worker's startup writes
    are flushed. Tests then assert against ``snapshot()[_STARTUP_WRITES:]``
    for a clean view of post-startup behavior."""
    controller.start()
    _wait_for_calls(fake, _STARTUP_WRITES)


# --- enable / disable paths ----------------------------------------------


def test_disabled_when_device_missing() -> None:
    """If the device factory returns None, controller stays disabled and
    every method is a no-op."""
    controller = LedController(device_factory=lambda: None)
    controller.start()
    assert not controller.enabled
    # Should not raise:
    controller.set_state(LedState.LISTENING)
    controller.error()
    controller.close()


def test_disabled_kwarg_skips_device_factory() -> None:
    """disabled=True short-circuits before the device factory runs."""
    factory_called = False

    def factory() -> XvfLedDevice | None:
        nonlocal factory_called
        factory_called = True
        return None

    controller = LedController(device_factory=factory, disabled=True)
    controller.start()
    assert not controller.enabled
    assert not factory_called


# --- worker startup ------------------------------------------------------


def _vendor_out_call(cmdid: int, payload: bytes) -> tuple[int, int, int, int, bytes]:
    return (0x40, 0, cmdid, _RESID_GPO, payload)


def test_worker_issues_startup_writes() -> None:
    """Brightness, speed, and gammify are applied once at worker
    startup so per-state code doesn't have to repeat them."""
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    try:
        _wait_for_calls(fake, _STARTUP_WRITES)
        startup = fake.snapshot()[:_STARTUP_WRITES]
        assert startup[0] == _vendor_out_call(13, bytes([PALETTE.breath_brightness]))
        assert startup[1] == _vendor_out_call(15, bytes([PALETTE.breath_speed]))
        assert startup[2] == _vendor_out_call(14, bytes([1]))
    finally:
        controller.close()


# --- per-state command sequences -----------------------------------------


def test_set_state_idle_writes_effect_off() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)
    try:
        controller.set_state(LedState.IDLE)
        _wait_for_calls(fake, _STARTUP_WRITES + 1)
        assert fake.snapshot()[_STARTUP_WRITES:] == [
            _vendor_out_call(12, bytes([EFFECT_OFF]))
        ]
    finally:
        controller.close()


def test_set_state_listening_configures_solid() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)
    try:
        controller.set_state(LedState.LISTENING)
        _wait_for_calls(fake, _STARTUP_WRITES + 2)
        calls = fake.snapshot()[_STARTUP_WRITES : _STARTUP_WRITES + 2]
        assert calls[0] == _vendor_out_call(16, struct.pack("<I", PALETTE.listening))
        assert calls[1] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
    finally:
        controller.close()


def test_set_state_listening_active_configures_solid() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)
    try:
        controller.set_state(LedState.LISTENING_ACTIVE)
        _wait_for_calls(fake, _STARTUP_WRITES + 2)
        calls = fake.snapshot()[_STARTUP_WRITES : _STARTUP_WRITES + 2]
        assert calls[0] == _vendor_out_call(
            16, struct.pack("<I", PALETTE.listening_active)
        )
        assert calls[1] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
    finally:
        controller.close()


def test_set_state_processing_configures_breath() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)
    try:
        controller.set_state(LedState.PROCESSING)
        _wait_for_calls(fake, _STARTUP_WRITES + 2)
        calls = fake.snapshot()[_STARTUP_WRITES : _STARTUP_WRITES + 2]
        assert calls[0] == _vendor_out_call(16, struct.pack("<I", PALETTE.processing))
        assert calls[1] == _vendor_out_call(12, bytes([EFFECT_BREATH]))
    finally:
        controller.close()


def test_set_state_speaking_configures_solid() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)
    try:
        controller.set_state(LedState.SPEAKING)
        _wait_for_calls(fake, _STARTUP_WRITES + 2)
        calls = fake.snapshot()[_STARTUP_WRITES : _STARTUP_WRITES + 2]
        assert calls[0] == _vendor_out_call(16, struct.pack("<I", PALETTE.speaking))
        assert calls[1] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
    finally:
        controller.close()


# --- animations ----------------------------------------------------------


def test_error_plays_red_breath_then_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(leds, "_ERROR_FLASH_S", 0.05)
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)
    try:
        controller.set_state(LedState.LISTENING)
        _wait_for_calls(fake, _STARTUP_WRITES + 2)
        baseline = len(fake.snapshot())

        controller.error()
        _wait_for_calls(fake, baseline + 6, timeout=2.0)
        new_calls = fake.snapshot()[baseline:]

        assert new_calls[0] == _vendor_out_call(16, struct.pack("<I", PALETTE.error))
        assert new_calls[1] == _vendor_out_call(13, bytes([PALETTE.breath_brightness]))
        assert new_calls[2] == _vendor_out_call(15, bytes([PALETTE.breath_speed]))
        assert new_calls[3] == _vendor_out_call(12, bytes([EFFECT_BREATH]))
        # State restoration after the flash (LISTENING solid):
        assert new_calls[4] == _vendor_out_call(
            16, struct.pack("<I", PALETTE.listening)
        )
        assert new_calls[5] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
    finally:
        controller.close()


def test_error_plays_full_duration_when_state_set_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator's error path sets state BEFORE calling error()
    so the worker drains the state action first, leaving the queue
    empty when _sleep_or_interrupt enters its wait. Reversing that
    order would leave the state action queued while the breath starts,
    aborting the sleep immediately and squashing the 3s breath into a
    momentary flash."""
    # Use a real (short) flash duration so we can observe whether the
    # sleep actually waits.
    monkeypatch.setattr(leds, "_ERROR_FLASH_S", 0.1)
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)
    try:
        # Mimic what main.py's exception path does: set new state, then
        # request the error animation.
        controller.set_state(LedState.LISTENING)
        controller.error()

        start = time.monotonic()
        # Expected writes after startup: LISTENING (2) + error setup (4)
        # + LISTENING restore (2) = 8.
        _wait_for_calls(fake, _STARTUP_WRITES + 8, timeout=2.0)
        elapsed = time.monotonic() - start
        # The breath must have actually slept; otherwise this would
        # finish in a few ms.
        assert elapsed >= 0.08, (
            f"Error breath was aborted; elapsed={elapsed:.3f}s — "
            f"the sleep in _apply_error_flash should have waited "
            f"~_ERROR_FLASH_S=0.1s"
        )
        calls = fake.snapshot()
        # Restore picks up _current_state = LISTENING (set before
        # error()), not whatever was there before.
        assert calls[-2] == _vendor_out_call(16, struct.pack("<I", PALETTE.listening))
        assert calls[-1] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
    finally:
        controller.close()


def test_animation_preempted_by_new_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A state change arriving mid-flash should abort the sleep so the
    user's barge-in isn't stuck behind a 3s red breath, and the worker
    must not re-apply the pre-flash state on the way out — that would
    cause a one-frame flicker before the new state takes hold."""
    # Long enough that the test would hang if preemption didn't work.
    monkeypatch.setattr(leds, "_ERROR_FLASH_S", 5.0)
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)
    try:
        controller.set_state(LedState.LISTENING)
        _wait_for_calls(fake, _STARTUP_WRITES + 2)
        controller.error()
        # Wait until the error flash has issued its 4 setup writes:
        _wait_for_calls(fake, _STARTUP_WRITES + 6, timeout=1.0)
        # Now interrupt with a new state. The flash should abort early
        # and the new state should be applied directly with no
        # restoration of the pre-flash LISTENING state in between.
        controller.set_state(LedState.PROCESSING)
        # Expected writes after startup: LISTENING (2) + error setup (4)
        # + PROCESSING (2) = 8. If the worker restored LISTENING after
        # the interrupted flash there would be 10.
        expected = _STARTUP_WRITES + 8
        _wait_for_calls(fake, expected, timeout=2.0)
        calls = fake.snapshot()
        assert calls[-1] == _vendor_out_call(12, bytes([EFFECT_BREATH]))
        assert len(calls) == expected, (
            f"Expected exactly {expected} writes (no LISTENING restore between "
            f"flash and PROCESSING), got {len(calls)}: {calls}"
        )
        # The write right after the error setup must be the PROCESSING
        # color, not a LISTENING restore.
        assert calls[_STARTUP_WRITES + 6] == _vendor_out_call(
            16, struct.pack("<I", PALETTE.processing)
        )
    finally:
        controller.close()


# --- close ---------------------------------------------------------------


def test_close_turns_leds_off() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)
    controller.set_state(LedState.LISTENING)
    _wait_for_calls(fake, _STARTUP_WRITES + 2)
    controller.close()
    # Last call should be EFFECT_OFF from the close action.
    last = fake.snapshot()[-1]
    assert last == _vendor_out_call(12, bytes([EFFECT_OFF]))


# --- worker resilience ----------------------------------------------------


class FlakyUsbDevice(FakeUsbDevice):
    """Fake device that can be told to fail the next N transfers.

    Stands in for a transient USB error — the reason every action the
    worker runs is wrapped rather than allowed to kill the thread.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fail_next = 0

    def ctrl_transfer(self, *args, **kwargs) -> None:  # type: ignore[override]
        if self.fail_next > 0:
            self.fail_next -= 1
            raise OSError("usb write failed")
        super().ctrl_transfer(*args, **kwargs)


def test_worker_survives_a_failing_action() -> None:
    """One bad transfer must not cost the session its LEDs: the worker
    logs it and keeps serving later actions."""
    fake = FlakyUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)
    try:
        fake.fail_next = 1
        controller.set_state(LedState.LISTENING)  # this one blows up
        controller.set_state(LedState.SPEAKING)
        _wait_for_calls(fake, _STARTUP_WRITES + 2, timeout=2.0)

        # SPEAKING was applied despite the failure before it.
        calls = fake.snapshot()[_STARTUP_WRITES:]
        assert calls[-1] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
        assert calls[-2] == _vendor_out_call(16, struct.pack("<I", PALETTE.speaking))
        assert controller._thread is not None and controller._thread.is_alive()
    finally:
        controller.close()


def test_close_completes_when_every_transfer_is_failing() -> None:
    """Turning the ring off is best-effort: `_apply_off` swallows USB
    errors, so a dying device can't block shutdown."""
    fake = FlakyUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)

    fake.fail_next = 100  # every remaining transfer raises, including OFF
    controller.close()

    assert controller._thread is not None
    assert not controller._thread.is_alive()
    assert controller.enabled is False


def test_close_stops_the_worker_even_if_applying_off_raises() -> None:
    """The stop signal must not be mistaken for a failed command.

    `close` is handled before the per-action `try`, so a raise here ends
    the thread instead of being logged as a command failure and leaving
    the loop running until `close()`'s join times out. Stubbing
    `_apply_off` is what makes the two guards separable: it swallows USB
    errors itself, so a failing device alone never reaches this path.
    """
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    _start_and_wait(fake, controller)

    def boom() -> None:
        raise OSError("usb gone")

    controller._apply_off = boom  # type: ignore[method-assign]
    controller.close()

    assert controller._thread is not None
    assert not controller._thread.is_alive()


def test_startup_failure_leaves_the_worker_serving_actions() -> None:
    """A USB error during the one-time configuration is logged, not
    fatal — later state changes still reach the ring.

    One failure is enough to abort configuration: all three writes share
    a single `try`, so the first one that raises skips the rest.
    """
    fake = FlakyUsbDevice()
    fake.fail_next = 1
    controller = _make_controller(fake)
    controller.start()
    try:
        controller.set_state(LedState.SPEAKING)
        _wait_for_calls(fake, 2, timeout=2.0)
        calls = fake.snapshot()
        assert calls[-1] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
    finally:
        controller.close()
