"""Reading a file that is still being written.

A station writes a pass to disk over the fifteen minutes it takes to receive it.
Pointed at that file, an ordinary reader stops at whatever had been written when
it got there and calls it a pass — silently short. What it has to do instead is
wait: for more bytes, and then for the writer to be finished.

Knowing the writer is finished is the whole problem. There is no signal in the
bytes themselves, so the caller supplies one, and the fallback — nothing new for
long enough — is the only one that needs no cooperation from the writer.
"""

import threading
import time
from pathlib import Path


def _write_slowly(path: Path, data: bytes, *, chunk: int, pause: float, done: threading.Event | None = None):
    """Write *data* to *path* a chunk at a time, as a station would."""

    def writer():
        with open(path, "wb") as fh:
            for start in range(0, len(data), chunk):
                fh.write(data[start : start + chunk])
                fh.flush()
                time.sleep(pause)
        if done is not None:
            done.set()

    thread = threading.Thread(target=writer, name="slow-writer", daemon=True)
    thread.start()
    return thread


def test_it_reads_the_bytes_that_arrive_after_it_started(tmp_path: Path):
    """The reader outlives the end of the file as it was when it opened it."""
    from pycadu.tail import tail_file

    path = tmp_path / "growing.dat"
    path.write_bytes(b"")
    writer = _write_slowly(path, bytes(range(256)) * 40, chunk=1024, pause=0.01)

    read = bytearray()
    with tail_file(path, idle_timeout=1.0, poll_interval=0.01) as source:
        view = memoryview(bytearray(512))
        while filled := source.readinto(view):
            read.extend(view[:filled])
    writer.join(timeout=5)

    assert bytes(read) == bytes(range(256)) * 40


def test_it_gives_up_only_once_nothing_further_has_arrived(tmp_path: Path):
    """A pause in the middle of a pass is not the end of it.

    Reception is bursty; a source that took the first quiet moment for the end
    of the pass would truncate almost every pass it read.
    """
    from pycadu.tail import tail_file

    path = tmp_path / "bursty.dat"
    path.write_bytes(b"first")

    def late_writer():
        time.sleep(0.3)
        with open(path, "ab") as fh:
            fh.write(b"second")

    threading.Thread(target=late_writer, daemon=True).start()

    read = bytearray()
    with tail_file(path, idle_timeout=1.0, poll_interval=0.01) as source:
        view = memoryview(bytearray(64))
        while filled := source.readinto(view):
            read.extend(view[:filled])

    assert bytes(read) == b"firstsecond"


def test_it_ends_when_the_writer_says_it_is_finished(tmp_path: Path):
    """Given a signal, the reader does not have to sit out its idle timeout.

    Waiting for silence costs the timeout on every pass, at the end, which is
    exactly where end-of-pass latency is being counted. A writer that can say it
    is done — a marker file, a rename — spares that.
    """
    from pycadu.tail import tail_file

    path = tmp_path / "signalled.dat"
    path.write_bytes(b"a pass")
    marker = tmp_path / "signalled.dat.done"

    started = time.monotonic()
    read = bytearray()
    with tail_file(path, idle_timeout=30.0, poll_interval=0.01, is_complete=marker.exists) as source:
        view = memoryview(bytearray(64))
        marker.touch()
        while filled := source.readinto(view):
            read.extend(view[:filled])
    elapsed = time.monotonic() - started

    assert bytes(read) == b"a pass"
    assert elapsed < 5, f"waited {elapsed:.1f}s for a source that had said it was finished"


def test_what_arrived_after_the_writer_finished_is_still_read(tmp_path: Path):
    """The completion signal must not race the last bytes out of the product.

    The marker appears after the write, but the reader may only notice it later
    — and between those two moments sit the last bytes of the pass. Checking the
    signal before the final read, rather than after, loses them.
    """
    from pycadu.tail import tail_file

    path = tmp_path / "raced.dat"
    path.write_bytes(b"")
    marker = tmp_path / "raced.dat.done"

    with open(path, "wb") as fh:
        fh.write(b"the whole pass")
    marker.touch()

    read = bytearray()
    with tail_file(path, idle_timeout=30.0, poll_interval=0.01, is_complete=marker.exists) as source:
        view = memoryview(bytearray(4))
        while filled := source.readinto(view):
            read.extend(view[:filled])

    assert bytes(read) == b"the whole pass"


def test_a_growing_pass_yields_the_frames_the_finished_one_does(tmp_path: Path):
    """The point of all of it: same file, same frames, arriving or arrived."""
    from pycadu.reader import iter_frames_from, iter_frames_read
    from pycadu.tail import tail_file

    from .test_reader import (
        CADU_STRIDE_1024,
        FRAME_SIZE,
        PATTERN,
        _two_frames,
        _wrap_frames_in_cadu_blocks,
    )

    file_data = _wrap_frames_in_cadu_blocks(_two_frames(), vcid=3) * 3
    finished = tmp_path / "finished.dat"
    finished.write_bytes(file_data)
    expected = list(iter_frames_read(finished, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE))
    assert len(expected) == 6, f"fixture gave {len(expected)} frames"

    growing = tmp_path / "growing.dat"
    growing.write_bytes(b"")
    written = threading.Event()
    _write_slowly(growing, file_data, chunk=len(file_data) // 20, pause=0.01, done=written)

    with tail_file(growing, idle_timeout=1.0, poll_interval=0.01) as source:
        arrived = list(
            iter_frames_from(source, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE, blocks_per_chunk=64)
        )

    assert written.is_set(), "the reader finished before the writer did"
    assert arrived == expected, f"{len(arrived)} frames arrived, {len(expected)} were written"


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll *predicate* until it holds or the timeout runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _watching(finished) -> None:
    """Skip the test where the platform cannot report a writer's close."""
    if not finished.watching:
        import pytest

        pytest.skip("no close-watching available on this platform")


def test_the_writer_closing_the_file_is_what_says_the_pass_is_over(tmp_path: Path):
    """The kernel knows when the writer let go; nothing else has to be guessed.

    Waiting for silence costs an idle timeout at the end of every pass, and has
    to be set longer than the longest gap in reception to be safe. Being told
    the writer closed the file is both immediate and exact.
    """
    from pycadu.tail import closed_by_writer

    path = tmp_path / "pass.dat"
    path.write_bytes(b"")

    with closed_by_writer(path) as finished:
        _watching(finished)
        handle = open(path, "wb")
        handle.write(b"still coming")
        handle.flush()
        assert not finished(), "called the pass over while the writer still had it open"

        handle.close()
        assert _wait_for(finished), "the writer closed the file and nothing noticed"


def test_a_reader_letting_go_does_not_end_the_pass(tmp_path: Path):
    """Only a writer's close counts — and the pass is being read as it arrives.

    The tailing reader has the same file open the whole time, and other things
    may look at it too. If any close ended the pass, the reader would end it
    itself the moment anything glanced at the file.
    """
    from pycadu.tail import closed_by_writer

    path = tmp_path / "pass.dat"
    path.write_bytes(b"a pass in progress")

    with closed_by_writer(path) as finished:
        _watching(finished)
        with open(path, "rb") as reader:
            reader.read()
        assert not _wait_for(finished, timeout=0.3), "a reader closing the file ended the pass"


def test_a_pass_ends_as_soon_as_its_writer_does(tmp_path: Path):
    """End to end: the frames stop arriving when the writer lets go, not later."""
    from pycadu.reader import iter_frames_from
    from pycadu.tail import closed_by_writer, tail_file

    from .test_reader import (
        CADU_STRIDE_1024,
        FRAME_SIZE,
        PATTERN,
        _two_frames,
        _wrap_frames_in_cadu_blocks,
    )

    file_data = _wrap_frames_in_cadu_blocks(_two_frames(), vcid=3) * 3
    path = tmp_path / "growing.dat"
    path.write_bytes(b"")

    with closed_by_writer(path) as finished:
        _watching(finished)
        _write_slowly(path, file_data, chunk=len(file_data) // 20, pause=0.01)
        started = time.monotonic()
        # An idle timeout long enough that only the close can end this.
        with tail_file(path, idle_timeout=600.0, poll_interval=0.01, is_complete=finished) as source:
            frames = list(
                iter_frames_from(source, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE, blocks_per_chunk=64)
            )
        elapsed = time.monotonic() - started

    assert len(frames) == 6, f"{len(frames)} frames came off a pass of 6"
    assert elapsed < 30, f"took {elapsed:.1f}s to notice a pass that had ended"
