"""Tests for meeko.keyboard ESC listener."""

import asyncio
import io

import pytest

from meeko.keyboard import esc_listener


class _FakeStream:
    """Minimal stdin stand-in exposing isatty()."""

    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty

    def fileno(self) -> int:
        raise OSError("no real fd")


async def test_non_tty_stream_is_noop_until_stop():
    """When stdin isn't a TTY, the listener just waits on stop_event."""
    barge_in_event = asyncio.Event()
    stop_event = asyncio.Event()
    task = asyncio.create_task(
        esc_listener(barge_in_event, stop_event, stream=_FakeStream(is_tty=False))
    )
    await asyncio.sleep(0.01)
    assert not task.done()
    assert not barge_in_event.is_set()
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


async def test_non_tty_no_buffer_attribute_falls_back_gracefully():
    """A stream without isatty() (e.g., BytesIO) is treated as non-TTY."""
    barge_in_event = asyncio.Event()
    stop_event = asyncio.Event()
    # io.BytesIO has no isatty() returning True, so it's non-TTY.
    task = asyncio.create_task(
        esc_listener(barge_in_event, stop_event, stream=io.BytesIO())
    )
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)
    assert not barge_in_event.is_set()


def test_read_loop_sets_event_on_esc(monkeypatch):
    """_read_loop pushes event.set onto the loop when it reads ESC."""
    from meeko import keyboard as kb

    reads = iter([b"a", b"\x1b", b""])
    monkeypatch.setattr(kb.os, "read", lambda fd, n: next(reads))
    monkeypatch.setattr(kb.select, "select", lambda r, w, x, t: (r, [], []))

    calls: list = []

    class _FakeLoop:
        def call_soon_threadsafe(self, fn):
            calls.append(fn)

    class _FakeEvent:
        def __init__(self):
            self._set = False

        def set(self):
            self._set = True

        def is_set(self):
            return self._set

    stop = _FakeEvent()
    event = _FakeEvent()

    class _FakeStreamWithFd:
        def fileno(self):
            return 0

    kb._read_loop(_FakeStreamWithFd(), _FakeLoop(), event, stop)

    # One scheduled callback for the ESC byte; none for the 'a' byte.
    assert len(calls) == 1
    calls[0]()
    assert event.is_set()


def test_read_loop_exits_on_stop_event(monkeypatch):
    from meeko import keyboard as kb

    # select always times out so the loop only exits via stop_event.
    monkeypatch.setattr(kb.select, "select", lambda r, w, x, t: ([], [], []))

    class _FakeEvent:
        def __init__(self, initial=False):
            self._set = initial

        def set(self):
            self._set = True

        def is_set(self):
            return self._set

    stop = _FakeEvent(initial=True)
    event = _FakeEvent()

    class _FakeLoop:
        def call_soon_threadsafe(self, fn):
            pytest.fail("should not signal when no bytes read")

    class _FakeStreamWithFd:
        def fileno(self):
            return 0

    kb._read_loop(_FakeStreamWithFd(), _FakeLoop(), event, stop)
