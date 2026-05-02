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
    EFFECT_RAINBOW,
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
    controller.session_transition()
    controller.error()
    controller.close()


def test_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """MEEKO_LED_DISABLED short-circuits before the device factory runs."""
    monkeypatch.setenv("MEEKO_LED_DISABLED", "1")
    factory_called = False

    def factory() -> XvfLedDevice | None:
        nonlocal factory_called
        factory_called = True
        return None

    controller = LedController(device_factory=factory)
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
        _wait_for_calls(fake, 3)
        calls = fake.snapshot()[:3]
        assert calls[0] == _vendor_out_call(16, struct.pack("<I", PALETTE.listening))
        assert calls[1] == _vendor_out_call(13, bytes([PALETTE.brightness]))
        assert calls[2] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
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
        assert calls[1] == _vendor_out_call(13, bytes([PALETTE.brightness]))
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
        _wait_for_calls(fake, 3)
        calls = fake.snapshot()[:3]
        assert calls[0] == _vendor_out_call(16, struct.pack("<I", PALETTE.speaking))
        assert calls[1] == _vendor_out_call(13, bytes([PALETTE.brightness]))
        assert calls[2] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
    finally:
        controller.close()


# --- animations ----------------------------------------------------------


def test_session_transition_plays_rainbow_then_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rainbow flash should run then re-apply the most recent state."""
    monkeypatch.setattr(leds, "_SESSION_FLASH_S", 0.05)
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    try:
        controller.set_state(LedState.LISTENING)
        _wait_for_calls(fake, 3)
        baseline = len(fake.snapshot())

        controller.session_transition()
        # Expect: brightness, speed, rainbow effect, [sleep], then state re-apply
        # (LISTENING solid = color + brightness + effect = 3 more calls)
        _wait_for_calls(fake, baseline + 6, timeout=2.0)
        new_calls = fake.snapshot()[baseline:]

        assert new_calls[0] == _vendor_out_call(13, bytes([PALETTE.brightness]))
        assert new_calls[1] == _vendor_out_call(15, bytes([8]))
        assert new_calls[2] == _vendor_out_call(12, bytes([EFFECT_RAINBOW]))
        # After flash, listening state is reapplied (solid color + brightness + effect):
        assert new_calls[3] == _vendor_out_call(
            16, struct.pack("<I", PALETTE.listening)
        )
        assert new_calls[4] == _vendor_out_call(13, bytes([PALETTE.brightness]))
        assert new_calls[5] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
    finally:
        controller.close()


def test_error_plays_red_breath_then_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(leds, "_ERROR_FLASH_S", 0.05)
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    try:
        controller.set_state(LedState.LISTENING)
        _wait_for_calls(fake, 3)
        baseline = len(fake.snapshot())

        controller.error()
        _wait_for_calls(fake, baseline + 7, timeout=2.0)
        new_calls = fake.snapshot()[baseline:]

        assert new_calls[0] == _vendor_out_call(16, struct.pack("<I", PALETTE.error))
        assert new_calls[1] == _vendor_out_call(13, bytes([PALETTE.brightness]))
        assert new_calls[2] == _vendor_out_call(15, bytes([2]))
        assert new_calls[3] == _vendor_out_call(12, bytes([EFFECT_BREATH]))
        # State restoration after the flash (LISTENING solid):
        assert new_calls[4] == _vendor_out_call(
            16, struct.pack("<I", PALETTE.listening)
        )
        assert new_calls[5] == _vendor_out_call(13, bytes([PALETTE.brightness]))
        assert new_calls[6] == _vendor_out_call(12, bytes([EFFECT_SOLID]))
    finally:
        controller.close()


def test_animation_preempted_by_new_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A state change arriving mid-flash should abort the sleep so the
    user's barge-in isn't stuck behind a 3s red breath."""
    # Long enough that the test would hang if preemption didn't work.
    monkeypatch.setattr(leds, "_ERROR_FLASH_S", 5.0)
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    try:
        controller.set_state(LedState.LISTENING)
        _wait_for_calls(fake, 3)
        controller.error()
        # Wait until the error flash has issued its 4 setup writes:
        _wait_for_calls(fake, 7, timeout=1.0)
        # Now interrupt with a new state. The flash should abort early
        # and the new state should be applied.
        controller.set_state(LedState.PROCESSING)
        # PROCESSING adds 4 writes, plus the auto-restore from the
        # interrupted error (LISTENING solid = 3 writes) gets queued
        # before PROCESSING. Eventually the last write is the PROCESSING
        # breath effect.
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            calls = fake.snapshot()
            if len(calls) >= 10 and calls[-1] == _vendor_out_call(
                12, bytes([EFFECT_BREATH])
            ):
                break
            time.sleep(0.02)
        else:
            raise AssertionError(
                f"PROCESSING state was not applied; calls={fake.snapshot()}"
            )
    finally:
        controller.close()


# --- close ---------------------------------------------------------------


def test_close_turns_leds_off() -> None:
    fake = FakeUsbDevice()
    controller = _make_controller(fake)
    controller.start()
    controller.set_state(LedState.LISTENING)
    _wait_for_calls(fake, 3)
    controller.close()
    # Last call should be EFFECT_OFF from the close action.
    last = fake.snapshot()[-1]
    assert last == _vendor_out_call(12, bytes([EFFECT_OFF]))
