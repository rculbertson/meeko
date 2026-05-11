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
    EFFECT_DOA,
    EFFECT_OFF,
    EFFECT_SOLID,
    PALETTE,
    LedController,
    LedState,
    XvfLedDevice,
)


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


# --- per-state command sequences -----------------------------------------


def _vendor_out_call(cmdid: int, payload: bytes) -> tuple[int, int, int, int, bytes]:
    return (0x40, 0, cmdid, _RESID_GPO, payload)


def test_set_state_idle_writes_effect_off() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    try:
        controller.set_state(LedState.IDLE)
        _wait_for_calls(fake, 1)
        assert fake.snapshot() == [_vendor_out_call(12, bytes([EFFECT_OFF]))]
    finally:
        controller.close()


def test_set_state_listening_configures_solid() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    try:
        controller.set_state(LedState.LISTENING)
        _wait_for_calls(fake, 2)
        calls = fake.snapshot()[:2]
        assert calls[0] == _vendor_out_call(16, struct.pack("<I", PALETTE.listening))
        assert calls[1] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
    finally:
        controller.close()


def test_set_state_listening_active_configures_doa() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    try:
        controller.set_state(LedState.LISTENING_ACTIVE)
        _wait_for_calls(fake, 2)
        calls = fake.snapshot()[:2]
        assert calls[0] == _vendor_out_call(
            17,
            struct.pack(
                "<II",
                PALETTE.listening_active_base,
                PALETTE.listening_active_indicator,
            ),
        )
        assert calls[1] == _vendor_out_call(12, bytes([EFFECT_DOA]))
    finally:
        controller.close()


def test_set_state_processing_configures_breath() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    try:
        controller.set_state(LedState.PROCESSING)
        _wait_for_calls(fake, 4)
        calls = fake.snapshot()[:4]
        assert calls[0] == _vendor_out_call(16, struct.pack("<I", PALETTE.processing))
        assert calls[1] == _vendor_out_call(13, bytes([PALETTE.breath_brightness]))
        assert calls[2] == _vendor_out_call(15, bytes([PALETTE.breath_speed]))
        assert calls[3] == _vendor_out_call(12, bytes([EFFECT_BREATH]))
    finally:
        controller.close()


def test_set_state_speaking_configures_solid() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    try:
        controller.set_state(LedState.SPEAKING)
        _wait_for_calls(fake, 2)
        calls = fake.snapshot()[:2]
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
    controller.start()
    try:
        controller.set_state(LedState.LISTENING)
        _wait_for_calls(fake, 2)
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
    controller.start()
    try:
        # Mimic what main.py's exception path does: set new state, then
        # request the error animation.
        controller.set_state(LedState.LISTENING)
        controller.error()

        start = time.monotonic()
        # Expected writes: LISTENING (2) + error setup (4) + LISTENING
        # restore (2) = 8.
        _wait_for_calls(fake, 8, timeout=2.0)
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
    controller.start()
    try:
        controller.set_state(LedState.LISTENING)
        _wait_for_calls(fake, 2)
        controller.error()
        # Wait until the error flash has issued its 4 setup writes:
        _wait_for_calls(fake, 6, timeout=1.0)
        # Now interrupt with a new state. The flash should abort early
        # and the new state should be applied directly with no
        # restoration of the pre-flash LISTENING state in between.
        controller.set_state(LedState.PROCESSING)
        # Expected writes: LISTENING (2) + error setup (4) + PROCESSING
        # (4) = 10. If the worker restored LISTENING after the
        # interrupted flash there would be 12 writes.
        _wait_for_calls(fake, 10, timeout=2.0)
        calls = fake.snapshot()
        assert calls[-1] == _vendor_out_call(12, bytes([EFFECT_BREATH]))
        assert len(calls) == 10, (
            f"Expected exactly 10 writes (no LISTENING restore between flash "
            f"and PROCESSING), got {len(calls)}: {calls}"
        )
        # The writes after the error setup must be the PROCESSING block,
        # not a LISTENING restore.
        assert calls[6] == _vendor_out_call(16, struct.pack("<I", PALETTE.processing))
    finally:
        controller.close()


# --- close ---------------------------------------------------------------


def test_close_turns_leds_off() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    controller.set_state(LedState.LISTENING)
    _wait_for_calls(fake, 2)
    controller.close()
    # Last call should be EFFECT_OFF from the close action.
    last = fake.snapshot()[-1]
    assert last == _vendor_out_call(12, bytes([EFFECT_OFF]))
