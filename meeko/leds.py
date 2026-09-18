"""LED indicators for the ReSpeaker XVF3800 mic ring.

Drives the WS2812 ring on the XVF3800 to reflect Meeko's state machine
(IDLE / LISTENING / PROCESSING / SPEAKING) plus session-transition and
error cues.

We talk to the XVF3800 directly via pyusb vendor control transfers
rather than vendoring ReSpeaker's xvf_host.py — only five commands are
needed (LED_EFFECT, LED_BRIGHTNESS, LED_GAMMIFY, LED_SPEED, LED_COLOR),
all on the GPO servicer resid (20). Wire format:

    bmRequestType = OUT|VENDOR|DEVICE   (write)
    bRequest      = 0
    wValue        = cmdid
    wIndex        = 20  (GPO_SERVICER_RESID)
    data          = little-endian payload (uint8 or uint32 per command)

LED actions are issued from a dedicated worker thread so blocking USB
control transfers never stall the asyncio loop. The public API is
synchronous and non-blocking — callers post intents to a queue and
return immediately. Useful because some Meeko callsites
(``request_barge_in``, Speaker enter/exit callbacks) are sync and can't
await.

If the XVF3800 isn't found, or pyusb/libusb isn't available, or LED
control is disabled via config (``[system] led_disabled = true`` or
``MEEKO_LED_DISABLED=1``), every public call is a no-op.
"""

from __future__ import annotations

import logging
import queue
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger("meeko")

XVF_VENDOR_ID = 0x2886
XVF_PRODUCT_ID = 0x001A

# GPO_SERVICER_RESID: the resource id all LED commands live under.
_RESID_GPO = 20

_CMD_LED_EFFECT = 12
_CMD_LED_BRIGHTNESS = 13
_CMD_LED_GAMMIFY = 14
_CMD_LED_SPEED = 15
_CMD_LED_COLOR = 16

# LED_EFFECT modes per XVF3800 firmware.
EFFECT_OFF = 0
EFFECT_BREATH = 1
EFFECT_SOLID = 3

_USB_TIMEOUT_MS = 1000

# Error red-breath duration.
_ERROR_FLASH_S = 3.0


class LedState(StrEnum):
    """Externally-visible LED state names.

    Decoupled from ``meeko.orchestrator.state.State``, and necessarily so:
    ``StateManager`` lives in ``meeko.orchestrator.state``, which imports
    this module to drive the ring. Naming the states separately keeps that
    dependency pointing one way.
    ``StateManager`` maps ``State`` → ``LedState``.

    ``LISTENING`` and ``LISTENING_ACTIVE`` are both sub-states of the
    orchestrator's ``LISTENING``: the former is a steady cyan
    (post-wake-word, confirming Meeko is ready), the latter is a
    brighter cyan (engaged on Deepgram ``StartOfTurn``, confirming
    Meeko is hearing speech). Both are solid effects; the brightness
    bump is the only visual difference.
    """

    IDLE = "idle"
    LISTENING = "listening"
    LISTENING_ACTIVE = "listening_active"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class _Palette:
    """Defaults — tweak in one place."""

    listening: int = 0x00C8C8  # cyan — "I heard the wake word"
    listening_active: int = 0x00FFFF  # brighter cyan — "I'm hearing you speak"
    processing: int = 0x0055FF  # blue (breath)
    speaking: int = 0x00A020  # soft green (solid)
    error: int = 0xFF0000  # red
    breath_brightness: int = 255
    breath_speed: int = 2


PALETTE = _Palette()


# --- USB transport -------------------------------------------------------


class XvfLedDevice:
    """Thin wrapper over a pyusb device with just the LED commands.

    Tests substitute a fake object that exposes ``ctrl_transfer`` and
    records calls.
    """

    def __init__(self, dev: object) -> None:
        self._dev = dev

    def _write(self, cmdid: int, payload: bytes) -> None:
        # bmRequestType = 0x40: OUT | VENDOR | DEVICE
        self._dev.ctrl_transfer(  # type: ignore[attr-defined]
            0x40, 0, cmdid, _RESID_GPO, payload, _USB_TIMEOUT_MS
        )

    def set_effect(self, mode: int) -> None:
        self._write(_CMD_LED_EFFECT, bytes([mode & 0xFF]))

    def set_brightness(self, value: int) -> None:
        self._write(_CMD_LED_BRIGHTNESS, bytes([value & 0xFF]))

    def set_gammify(self, enabled: bool) -> None:
        self._write(_CMD_LED_GAMMIFY, bytes([1 if enabled else 0]))

    def set_speed(self, value: int) -> None:
        self._write(_CMD_LED_SPEED, bytes([value & 0xFF]))

    def set_color(self, rgb: int) -> None:
        self._write(_CMD_LED_COLOR, struct.pack("<I", rgb & 0xFFFFFFFF))

    def close(self) -> None:
        try:
            import usb.util  # type: ignore[import-not-found]

            usb.util.dispose_resources(self._dev)
        except Exception:
            pass


def _find_xvf_device() -> XvfLedDevice | None:
    """Locate the XVF3800 via pyusb. Return None if unavailable."""
    try:
        import usb.core  # type: ignore[import-not-found]
    except Exception as e:
        logger.info("LED: pyusb not available (%s); LED control disabled", e)
        return None

    try:
        dev = usb.core.find(idVendor=XVF_VENDOR_ID, idProduct=XVF_PRODUCT_ID)
    except Exception as e:
        logger.info("LED: usb.core.find failed (%s); LED control disabled", e)
        return None
    if dev is None:
        logger.info("LED: XVF3800 not found; LED control disabled")
        return None
    return XvfLedDevice(dev)


# --- Controller ----------------------------------------------------------


@dataclass
class _Action:
    kind: str  # "state" | "error" | "close"
    state: LedState | None = None


class LedController:
    """Reflect Meeko's state on the XVF3800 LED ring.

    Public methods are synchronous and non-blocking. They enqueue work
    for an internal daemon thread that issues the actual USB control
    transfers. If the device is unavailable, every method is a no-op.

    The ``error`` animation plays a short red breath then re-applies
    the most recent state. It can be preempted by a newer item on the
    queue, so a barge-in immediately after an error isn't stuck behind
    the red flash.
    """

    def __init__(
        self,
        device_factory: Callable[[], XvfLedDevice | None] = _find_xvf_device,
        *,
        disabled: bool = False,
    ) -> None:
        self._device_factory = device_factory
        self._disabled = disabled
        self._device: XvfLedDevice | None = None
        self._enabled = False
        self._queue: queue.Queue[_Action] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._current_state: LedState | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        """Open the device and start the worker thread.

        Safe to call when no device is present — controller stays
        disabled and all methods become no-ops.
        """
        if self._disabled:
            logger.info("LED: disabled via config; LED control off")
            return
        self._device = self._device_factory()
        if self._device is None:
            return
        self._enabled = True
        self._thread = threading.Thread(
            target=self._worker, name="meeko-leds", daemon=True
        )
        self._thread.start()
        logger.info("LED: controller started")

    def set_state(self, state: LedState) -> None:
        if not self._enabled:
            return
        self._queue.put(_Action(kind="state", state=state))

    def error(self) -> None:
        if not self._enabled:
            return
        self._queue.put(_Action(kind="error"))

    def close(self) -> None:
        if not self._enabled:
            return
        self._queue.put(_Action(kind="close"))
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._enabled = False

    # --- worker --------------------------------------------------------

    def _worker(self) -> None:
        assert self._device is not None
        try:
            # Some settings can be applied once at startup, so
            # we don't have to set them on each state change.
            # Brightness and speed apply only to breath effect, and
            # are ignored by other effects, so we can leave them set.
            # Wrapped in try/except like each action below: a
            # transient USB error here shouldn't kill the worker and
            # silently disable LEDs for the rest of the session.
            try:
                self._device.set_brightness(PALETTE.breath_brightness)
                self._device.set_speed(PALETTE.breath_speed)
                # We want gammify always on.
                self._device.set_gammify(True)
            except Exception:
                logger.exception("LED: startup configuration failed")
            while True:
                action = self._queue.get()
                try:
                    if action.kind == "close":
                        self._apply_off()
                        return
                    if action.kind == "state":
                        assert action.state is not None
                        self._apply_state(action.state)
                        self._current_state = action.state
                    elif action.kind == "error":
                        interrupted = self._apply_error_flash()
                        if not interrupted and self._current_state is not None:
                            self._apply_state(self._current_state)
                except Exception:
                    logger.exception("LED: command failed")
        finally:
            try:
                self._device.close()
            except Exception:
                pass

    def _apply_state(self, state: LedState) -> None:
        assert self._device is not None
        d = self._device
        if state == LedState.IDLE:
            d.set_effect(EFFECT_OFF)
        elif state == LedState.LISTENING:
            d.set_color(PALETTE.listening)
            d.set_effect(EFFECT_SOLID)
        elif state == LedState.LISTENING_ACTIVE:
            d.set_color(PALETTE.listening_active)
            d.set_effect(EFFECT_SOLID)
        elif state == LedState.PROCESSING:
            d.set_color(PALETTE.processing)
            d.set_effect(EFFECT_BREATH)
        elif state == LedState.SPEAKING:
            d.set_color(PALETTE.speaking)
            d.set_effect(EFFECT_SOLID)

    def _apply_off(self) -> None:
        assert self._device is not None
        try:
            self._device.set_effect(EFFECT_OFF)
        except Exception:
            pass

    def _apply_error_flash(self) -> bool:
        """Returns True if interrupted by a queued action, False if the
        full flash duration elapsed."""
        assert self._device is not None
        d = self._device
        d.set_color(PALETTE.error)
        d.set_brightness(PALETTE.breath_brightness)
        d.set_speed(PALETTE.breath_speed)
        d.set_effect(EFFECT_BREATH)
        return self._sleep_or_interrupt(_ERROR_FLASH_S)

    def _sleep_or_interrupt(self, total_s: float) -> bool:
        """Sleep up to ``total_s``; abort early if new work arrives.

        Returns True if interrupted, False if the full duration elapsed.
        """
        end = time.monotonic() + total_s
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return False
            if not self._queue.empty():
                return True
            time.sleep(min(0.05, remaining))
