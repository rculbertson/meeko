"""ESC-key listener for macOS barge-in.

Reads stdin in cbreak mode from a background thread and sets an
asyncio.Event when the user presses ESC. Used as a stopgap barge-in
trigger on Macs without hardware AEC — the production path on the Pi
uses Deepgram SpeechStarted instead.
"""

import asyncio
import logging
import os
import select
import sys
import termios
import tty
from contextlib import contextmanager
from typing import BinaryIO

ESC = b"\x1b"

logger = logging.getLogger("meeko")


@contextmanager
def _cbreak(fd: int):
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_loop(
    stream: BinaryIO,
    loop: asyncio.AbstractEventLoop,
    event: asyncio.Event,
    stop_event: asyncio.Event,
) -> None:
    fd = stream.fileno()
    while not stop_event.is_set():
        try:
            ready, _, _ = select.select([fd], [], [], 0.1)
        except OSError, ValueError:
            return
        if not ready:
            continue
        try:
            b = os.read(fd, 1)
        except OSError, ValueError:
            return
        if not b:
            return
        if b == ESC:
            loop.call_soon_threadsafe(event.set)


async def esc_listener(
    barge_in_event: asyncio.Event,
    stop_event: asyncio.Event,
    stream: BinaryIO | None = None,
) -> None:
    """Watch stdin for ESC presses; set barge_in_event on each.

    No-op when stdin is not a TTY (tests, piped input). Returns when
    stop_event fires or stdin closes."""
    stream = stream or sys.stdin.buffer
    try:
        is_tty = stream.isatty()
    except AttributeError, ValueError:
        is_tty = False
    if not is_tty:
        await stop_event.wait()
        return

    loop = asyncio.get_running_loop()
    fd = stream.fileno()
    with _cbreak(fd):
        await asyncio.to_thread(_read_loop, stream, loop, barge_in_event, stop_event)
