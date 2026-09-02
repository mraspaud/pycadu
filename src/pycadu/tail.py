"""Reading a file while it is still being written.

A station writes a pass to disk over the fifteen minutes it takes to receive it.
An ordinary reader pointed at that file stops at whatever had been written when
it got there, and reports a short pass rather than an error. To follow the
reception instead, a reader has to wait — first for more bytes, and then for the
writer to be finished with the file.

The second part is the hard one, because nothing in the bytes says the pass is
over. The best answer is to be told: the operating system knows the moment the
writer let go of the file, and :func:`closed_by_writer` reports it. Failing that
the caller supplies whatever signal it has — a marker file, a rename, a message
from the scheduler — and failing that too, going quiet for long enough is what
is left, and it is the default.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

__all__ = ["TailedFile", "WriterClosed", "closed_by_writer", "tail_file"]

#: How long a source may go quiet before it is taken to have ended. Long enough
#: to sit out a reception gap — the meos passes drop up to 24 s of frames —
#: without leaving a finished pass waiting around for minutes.
DEFAULT_IDLE_TIMEOUT = 60.0

#: How often a quiet source is asked again. Frames are seconds apart, so this
#: costs nothing and bounds how late the end of a pass is noticed.
DEFAULT_POLL_INTERVAL = 0.5


_log = logging.getLogger(__name__)


class TailedFile:
    """An open file that waits for more instead of reporting the end.

    Reads exactly like the file object it wraps — :meth:`readinto` fills the
    buffer and returns how many bytes it got — except that it returns 0 only
    once the pass is genuinely over. Everything else is the caller's: what a
    completed pass looks like, and how long to wait for one that says nothing.
    """

    def __init__(
        self,
        path: Path,
        *,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        is_complete: Callable[[], bool] | None = None,
    ) -> None:
        """Follow *path* until it is complete, or until it goes quiet.

        Parameters
        ----------
        idle_timeout:
            Seconds without a new byte before the pass is taken to have ended.
            This has to outlast the longest gap in reception, or a quiet stretch
            mid-pass truncates the product.
        poll_interval:
            Seconds between asking a quiet source again.
        is_complete:
            Called to ask whether the writer has finished — a marker file
            appearing, a rename, whatever the station offers. When it is not
            given, the idle timeout is the only signal, which costs one timeout
            at the end of every pass.

        """
        self._path = Path(path)
        self._idle_timeout = idle_timeout
        self._poll_interval = poll_interval
        self._is_complete = is_complete
        self._handle = open(self._path, "rb")
        self._quiet_since: float | None = None

    def readinto(self, buffer) -> int:
        """Fill *buffer*, waiting for bytes that have not arrived yet.

        Returns 0 only when the pass is over: either the writer has said so and
        the file has been read to its end, or nothing has arrived for the idle
        timeout.
        """
        while True:
            filled = self._handle.readinto(buffer)
            if filled:
                self._quiet_since = None
                return filled
            if self._ended():
                return 0
            time.sleep(self._poll_interval)

    def _ended(self) -> bool:
        """Whether nothing further is coming, asked at the end of the file.

        Deliberately asked *after* a read came back empty, never before one:
        between the writer finishing and this noticing sit the last bytes of the
        pass, and checking first would publish a product without them.
        """
        if self._is_complete is not None and self._is_complete():
            return True
        now = time.monotonic()
        if self._quiet_since is None:
            self._quiet_since = now
            return False
        return now - self._quiet_since >= self._idle_timeout

    def close(self) -> None:
        """Release the underlying file."""
        self._handle.close()

    def __enter__(self) -> TailedFile:
        """Return the source itself, for use in a ``with`` block."""
        return self

    def __exit__(self, *exc_info) -> None:
        """Release the underlying file however the block ended."""
        self.close()


class _WriterClosedHandler:
    """Watches a directory for the one event that means a writer is finished.

    Subclasses watchdog's handler lazily, at construction, so that this module
    imports without watchdog installed — the tailing reader falls back to an
    idle timeout in that case.
    """

    def __new__(cls, path: Path, mark_closed) -> object:
        """Build a watchdog handler bound to *path*."""
        from watchdog.events import EVENT_TYPE_CLOSED, FileSystemEventHandler

        watched = str(path.resolve())

        class _Closed(FileSystemEventHandler):
            def on_any_event(self, event) -> None:
                """Record a close-after-write of the file being followed."""
                # Only a *write* close: the tailing reader holds the same file
                # open, and a reader letting go says nothing about the pass.
                if event.event_type != EVENT_TYPE_CLOSED or event.is_directory:
                    return
                if str(Path(_as_text(event.src_path)).resolve()) == watched:
                    mark_closed()

        return _Closed()


class WriterClosed:
    """Reports when the process writing *path* has closed it.

    The exact signal, where the platform offers one: on Linux the kernel raises
    a close-after-write event and this catches it, so a pass ends the moment its
    writer is finished rather than an idle timeout later. Call it to ask.

    The watch has to be in place **before** the writer closes, so build this
    before opening the file to read.

    :attr:`watching` says whether anything is actually being watched. It is
    False when the ``watch`` extra is not installed, and there are two cases
    where it is True but the event will never come: a writer on **another host**
    writing over NFS, which the local kernel cannot see, and a file whose writer
    had already finished. Both fall back to the idle timeout, so the pass still
    ends — later than it needed to.
    """

    def __init__(self, path: Path) -> None:
        """Start watching *path* for its writer letting go of it."""
        self._path = Path(path)
        self._closed = False
        self._observer = None
        self._start()

    def _start(self) -> None:
        """Begin watching, or record that nothing is watching."""
        try:
            from watchdog.observers import Observer
        except ImportError:
            _log.info(
                "watchdog is not installed, so the end of %s will be inferred from a quiet "
                "period rather than from its writer closing it; install pycadu[watch] for "
                "the exact signal",
                self._path.name,
            )
            return

        observer = Observer()
        observer.schedule(_WriterClosedHandler(self._path, self._mark_closed), str(self._path.parent))
        observer.start()
        self._observer = observer

    def _mark_closed(self) -> None:
        """Record that the writer has finished."""
        self._closed = True

    @property
    def watching(self) -> bool:
        """Whether a close will actually be reported."""
        return self._observer is not None

    def __call__(self) -> bool:
        """Whether the writer has closed the file."""
        return self._closed

    def close(self) -> None:
        """Stop watching."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

    def __enter__(self) -> WriterClosed:
        """Return the watcher itself, for use in a ``with`` block."""
        return self

    def __exit__(self, *exc_info) -> None:
        """Stop watching however the block ended."""
        self.close()


def _as_text(src_path) -> str:
    """Return an event path as text, whichever way watchdog reported it."""
    return src_path.decode() if isinstance(src_path, bytes) else src_path


def closed_by_writer(path: Path) -> WriterClosed:
    """Watch *path* for its writer closing it, as a completion signal.

    Hand the result to :func:`tail_file` as ``is_complete``. Needs the ``watch``
    extra (``pip install pycadu[watch]``); without it nothing is watched and the
    idle timeout is what ends the pass.
    """
    return WriterClosed(path)


def tail_file(
    path: Path,
    *,
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    is_complete: Callable[[], bool] | None = None,
) -> TailedFile:
    """Open *path* as a source that follows it while it is being written.

    Hand the result to :func:`pycadu.reader.iter_frames_from` to read frames off
    a pass as it is received.
    """
    return TailedFile(
        path, idle_timeout=idle_timeout, poll_interval=poll_interval, is_complete=is_complete
    )
