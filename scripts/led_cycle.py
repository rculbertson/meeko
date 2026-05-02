"""Cycle through Meeko's LED states on the XVF3800 with arrow keys.

Usage:
    uv run python -m scripts.led_cycle

Right arrow → next state.  Left arrow → previous state.  q → quit.

Bypasses LedController (no worker thread, no queue) and talks to
``XvfLedDevice`` directly, so what you see on the ring is exactly what
the per-state command sequence in ``meeko/leds.py`` issues.
"""

from __future__ import annotations

import sys
import termios
import tty
from dataclasses import dataclass

from meeko.leds import (
    EFFECT_BREATH,
    EFFECT_DOA,
    EFFECT_OFF,
    EFFECT_RAINBOW,
    EFFECT_SOLID,
    PALETTE,
    XvfLedDevice,
    _find_xvf_device,
)


@dataclass
class Step:
    meeko_state: str
    led_state: str
    color_name: str
    hex_code: str  # for printing; multi-color steps describe both

    def apply(self, dev: XvfLedDevice) -> None:  # noqa: ARG002
        raise NotImplementedError


@dataclass
class _Off(Step):
    def apply(self, dev: XvfLedDevice) -> None:
        dev.set_effect(EFFECT_OFF)


@dataclass
class _Solid(Step):
    color: int = 0

    def apply(self, dev: XvfLedDevice) -> None:
        dev.set_color(self.color)
        dev.set_brightness(PALETTE.brightness)
        dev.set_effect(EFFECT_SOLID)


@dataclass
class _Doa(Step):
    base: int = 0
    indicator: int = 0

    def apply(self, dev: XvfLedDevice) -> None:
        dev.set_doa_color(self.base, self.indicator)
        dev.set_effect(EFFECT_DOA)


@dataclass
class _Breath(Step):
    color: int = 0
    speed: int = 1

    def apply(self, dev: XvfLedDevice) -> None:
        dev.set_color(self.color)
        dev.set_brightness(PALETTE.brightness)
        dev.set_speed(self.speed)
        dev.set_effect(EFFECT_BREATH)


@dataclass
class _Rainbow(Step):
    speed: int = 8

    def apply(self, dev: XvfLedDevice) -> None:
        dev.set_brightness(PALETTE.brightness)
        dev.set_speed(self.speed)
        dev.set_effect(EFFECT_RAINBOW)


STEPS: list[Step] = [
    _Off(
        meeko_state="IDLE",
        led_state="IDLE",
        color_name="off",
        hex_code="—",
    ),
    _Solid(
        meeko_state="LISTENING",
        led_state="LISTENING",
        color_name="teal (solid)",
        hex_code=f"0x{PALETTE.listening:06X}",
        color=PALETTE.listening,
    ),
    _Doa(
        meeko_state="LISTENING (user speaking)",
        led_state="LISTENING_ACTIVE",
        color_name="teal base + cyan DoA indicator",
        hex_code=(
            f"base 0x{PALETTE.listening_active_base:06X}, "
            f"indicator 0x{PALETTE.listening_active_indicator:06X}"
        ),
        base=PALETTE.listening_active_base,
        indicator=PALETTE.listening_active_indicator,
    ),
    _Breath(
        meeko_state="PROCESSING",
        led_state="PROCESSING",
        color_name="soft blue (breath)",
        hex_code=f"0x{PALETTE.processing:06X}",
        color=PALETTE.processing,
        speed=PALETTE.breath_speed,
    ),
    _Solid(
        meeko_state="SPEAKING",
        led_state="SPEAKING",
        color_name="soft blue (solid)",
        hex_code=f"0x{PALETTE.speaking:06X}",
        color=PALETTE.speaking,
    ),
    _Rainbow(
        meeko_state="(any → any)",
        led_state="SESSION_TRANSITION",
        color_name="rainbow sweep",
        hex_code="—",
    ),
    _Breath(
        meeko_state="(any)",
        led_state="ERROR",
        color_name="red (breath)",
        hex_code=f"0x{PALETTE.error:06X}",
        color=PALETTE.error,
        speed=2,
    ),
]


def _read_key() -> str:
    """Read one keypress. Returns 'right', 'left', 'q', or '' for other."""
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        # Escape sequence — read the next two bytes for arrow keys.
        seq = sys.stdin.read(2)
        if seq == "[C":
            return "right"
        if seq == "[D":
            return "left"
        return ""
    if ch in ("q", "Q", "\x03"):  # q or Ctrl-C
        return "q"
    return ""


def _print_step(idx: int, step: Step) -> None:
    print(
        f"\n[{idx + 1}/{len(STEPS)}] meeko={step.meeko_state:30s} "
        f"led={step.led_state:20s} "
        f"color={step.color_name:35s} {step.hex_code}"
    )


def main() -> None:
    raw_dev = _find_xvf_device()
    if raw_dev is None:
        print("XVF3800 not found. Plug it in and check the udev rule.")
        sys.exit(1)
    # _find_xvf_device returns an XvfLedDevice already.
    dev: XvfLedDevice = raw_dev

    idx = 0
    print("Arrow keys: ← previous  → next   q: quit")
    _print_step(idx, STEPS[idx])
    STEPS[idx].apply(dev)

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key()
            if key == "q":
                break
            if key == "right":
                idx = (idx + 1) % len(STEPS)
            elif key == "left":
                idx = (idx - 1) % len(STEPS)
            else:
                continue
            _print_step(idx, STEPS[idx])
            STEPS[idx].apply(dev)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        # Always leave the ring dark on exit.
        try:
            dev.set_effect(EFFECT_OFF)
        finally:
            dev.close()
        print("\nbye")


if __name__ == "__main__":
    main()
