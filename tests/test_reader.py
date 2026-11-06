import io
import math
import tempfile
from pathlib import Path

import pytest

from pycadu.constants import (
    CADU_STRIDE_1024,
    CCSDS_HEADER_LENGTH,
    CCSDS_SYNC_MARKER,
    RS_PARITY_LENGTH,
    VCDU_PAYLOAD_LENGTH,
)
from pycadu.reader import cadu_blocks, vcdu_frames, vcdu_payloads

PATTERN = bytes([0xAA, 0x55] * 6)  # stand-in for any instrument sync pattern
FRAME_SIZE = 0x38E4F4


def _wrap_frames_in_cadu_blocks(frame_data: bytes, vcid: int = 3) -> bytes:
    n_blocks = math.ceil(len(frame_data) / VCDU_PAYLOAD_LENGTH)
    out = bytearray()
    for i in range(n_blocks):
        payload = frame_data[i * VCDU_PAYLOAD_LENGTH : (i + 1) * VCDU_PAYLOAD_LENGTH]
        payload = payload.ljust(VCDU_PAYLOAD_LENGTH, b"\x00")
        header = (
            CCSDS_SYNC_MARKER + bytes([0x4D, vcid & 0x3F]) + b"\x00" * (CCSDS_HEADER_LENGTH - 6)
        )
        out += header + payload + b"\x00" * RS_PARITY_LENGTH
    return bytes(out)


def _two_frames() -> bytes:
    frame = bytearray(FRAME_SIZE)
    frame[: len(PATTERN)] = PATTERN
    return bytes(frame) * 2


def test_cadu_blocks_yields_one_block_per_stride():
    n = 5
    with tempfile.TemporaryDirectory() as tmp:
        dat = Path(tmp) / "test.dat"
        dat.write_bytes(b"\x00" * (CADU_STRIDE_1024 * n))
        blocks = list(cadu_blocks(dat, CADU_STRIDE_1024))
    assert len(blocks) == n
    assert all(len(b) == CADU_STRIDE_1024 for b in blocks)


def test_vcdu_payloads_yields_only_matching_vcid():
    vcid_3_block = (
        CCSDS_SYNC_MARKER
        + bytes([0x4D, 3])
        + b"\x00" * (CCSDS_HEADER_LENGTH - 6)
        + b"\xab" * VCDU_PAYLOAD_LENGTH
        + b"\x00" * RS_PARITY_LENGTH
    )
    vcid_6_block = (
        CCSDS_SYNC_MARKER
        + bytes([0x4D, 6])
        + b"\x00" * (CCSDS_HEADER_LENGTH - 6)
        + b"\xcd" * VCDU_PAYLOAD_LENGTH
        + b"\x00" * RS_PARITY_LENGTH
    )
    blocks = iter([vcid_3_block, vcid_6_block, vcid_3_block])
    payloads = list(vcdu_payloads(blocks, CADU_STRIDE_1024, target_vcid=3))
    assert len(payloads) == 2
    assert all(p == b"\xab" * VCDU_PAYLOAD_LENGTH for p in payloads)


def test_vcdu_frames_yields_complete_frames_from_payload_stream():
    two_frames = _two_frames()
    chunk = VCDU_PAYLOAD_LENGTH
    payloads = (two_frames[i : i + chunk] for i in range(0, len(two_frames), chunk))
    result = list(vcdu_frames(payloads, PATTERN, FRAME_SIZE))
    assert len(result) == 2
    assert all(len(f) == FRAME_SIZE for f in result)


def test_vcdu_frames_handles_bit_shifted_payload_stream():
    import numpy as np

    frame = bytearray(FRAME_SIZE)
    frame[: len(PATTERN)] = PATTERN
    raw = bytes(frame) + bytes(2)

    arr = np.frombuffer(raw, dtype=np.uint8)
    right_shifted = np.zeros(len(arr), dtype=np.uint8)
    right_shifted[0] = arr[0] >> 2
    for i in range(1, len(arr)):
        right_shifted[i] = ((arr[i - 1] << 6) | (arr[i] >> 2)) & 0xFF
    payload = bytes(right_shifted[:-1])

    payloads = (
        payload[i : i + VCDU_PAYLOAD_LENGTH] for i in range(0, len(payload), VCDU_PAYLOAD_LENGTH)
    )
    result = list(vcdu_frames(payloads, PATTERN, FRAME_SIZE))
    assert len(result) == 1
    assert result[0][: len(PATTERN)] == PATTERN


def test_read_frames_batch_matches_streaming():
    from pycadu.reader import read_frames_batch

    file_data = _wrap_frames_in_cadu_blocks(_two_frames(), vcid=3) * 3
    with tempfile.TemporaryDirectory() as tmp:
        dat = Path(tmp) / "test.dat"
        dat.write_bytes(file_data)
        stream = vcdu_frames(
            vcdu_payloads(cadu_blocks(dat, CADU_STRIDE_1024), CADU_STRIDE_1024, 3),
            PATTERN,
            FRAME_SIZE,
        )
        expected = list(stream)
        actual = read_frames_batch(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE)
    assert actual == expected


def test_read_frames_batch_finds_shifted_sync_after_syncless_first_chunk():
    """Sync search must survive a first chunk (10k blocks) with no sync marker.

    Regression test: a failed first search used to clobber bit_shift from None
    to 0, permanently disabling bit-level sync detection, so a bit-shifted
    stream preceded by a long sync-free stretch yielded zero frames.
    """
    import numpy as np

    from pycadu.reader import read_frames_batch

    shift = 4
    raw = _two_frames() + bytes(2)
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    right_shifted = np.empty(len(arr), dtype=np.uint8)
    right_shifted[0] = arr[0] >> shift
    right_shifted[1:] = (((arr[:-1] << (8 - shift)) | (arr[1:] >> shift)) & 0xFF).astype(np.uint8)
    # 11k blocks of sync-free payload: more than one 10k-block processing chunk
    junk = b"\x00" * (11_000 * VCDU_PAYLOAD_LENGTH)
    file_data = _wrap_frames_in_cadu_blocks(junk + bytes(right_shifted[:-1]), vcid=3)
    with tempfile.TemporaryDirectory() as tmp:
        dat = Path(tmp) / "test.dat"
        dat.write_bytes(file_data)
        frames = read_frames_batch(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE)
    assert len(frames) == 2
    assert all(f[: len(PATTERN)] == PATTERN for f in frames)


def test_iter_frames_batch_matches_read_frames_batch():
    """The streaming reader must yield exactly what the batch reader returns.

    ``read_frames_batch`` is fast because it scans blocks with vectorised numpy,
    but it accumulates the whole pass in a list. Consumers that process frames
    as they arrive need that same scanning without the list, so the two must
    agree frame for frame.
    """
    from pycadu.reader import iter_frames_batch, read_frames_batch

    file_data = _wrap_frames_in_cadu_blocks(_two_frames(), vcid=3) * 3
    with tempfile.TemporaryDirectory() as tmp:
        dat = Path(tmp) / "test.dat"
        dat.write_bytes(file_data)
        expected = read_frames_batch(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE)
        actual = list(iter_frames_batch(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE))
    assert expected, "fixture produced no frames, so the comparison proves nothing"
    assert actual == expected


def test_iter_frames_batch_yields_before_the_file_is_exhausted():
    """Frames are handed over as they are found, not gathered first."""
    from pycadu.reader import iter_frames_batch

    file_data = _wrap_frames_in_cadu_blocks(_two_frames(), vcid=3) * 3
    with tempfile.TemporaryDirectory() as tmp:
        dat = Path(tmp) / "test.dat"
        dat.write_bytes(file_data)
        stream = iter_frames_batch(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE)
        first = next(stream)
        assert isinstance(first, bytes)
        assert len(first) == FRAME_SIZE
        list(stream)


def test_iter_frames_batch_can_be_abandoned_early():
    """Closing the generator before the file ends must not error.

    The scan exposes the mmap through a numpy view; if that view outlives the
    mapping, closing it raises BufferError. Streaming consumers stop early all
    the time — a bounded read, or an exception downstream — so this is the
    normal case, not an edge case.
    """
    from pycadu.reader import iter_frames_batch

    file_data = _wrap_frames_in_cadu_blocks(_two_frames(), vcid=3) * 3
    with tempfile.TemporaryDirectory() as tmp:
        dat = Path(tmp) / "test.dat"
        dat.write_bytes(file_data)
        stream = iter_frames_batch(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE)
        next(stream)
        stream.close()  # must not raise


def _fixture_file(tmp: str) -> Path:
    """A small CADU file holding six frames across several blocks."""
    dat = Path(tmp) / "test.dat"
    dat.write_bytes(_wrap_frames_in_cadu_blocks(_two_frames(), vcid=3) * 3)
    return dat


def test_reading_the_file_yields_the_frames_mapping_it_does():
    """The reader that reads must find exactly what the reader that maps finds.

    Mapping the whole file makes every page it touches resident for as long as
    the reader lives, which is fine for a reader that runs to completion up
    front and expensive for one that runs beside the pipeline consuming it. The
    two must be interchangeable, or the choice is not one a caller can make.
    """
    from pycadu.reader import iter_frames_batch, iter_frames_read

    with tempfile.TemporaryDirectory() as tmp:
        dat = _fixture_file(tmp)
        mapped = list(iter_frames_batch(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE))
        read = list(iter_frames_read(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE))
    assert mapped, "fixture produced no frames, so the comparison proves nothing"
    assert read == mapped


def test_reading_finds_a_shifted_sync_after_a_syncless_first_chunk():
    """The sync search spans chunk boundaries here exactly as it does mapped."""
    import numpy as np

    from pycadu.reader import iter_frames_read

    shift = 4
    raw = _two_frames() + bytes(2)
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    right_shifted = np.empty(len(arr), dtype=np.uint8)
    right_shifted[0] = arr[0] >> shift
    right_shifted[1:] = (((arr[:-1] << (8 - shift)) | (arr[1:] >> shift)) & 0xFF).astype(np.uint8)
    junk = b"\x00" * (11_000 * VCDU_PAYLOAD_LENGTH)
    file_data = _wrap_frames_in_cadu_blocks(junk + bytes(right_shifted[:-1]), vcid=3)
    with tempfile.TemporaryDirectory() as tmp:
        dat = Path(tmp) / "test.dat"
        dat.write_bytes(file_data)
        frames = list(iter_frames_read(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE))
    assert len(frames) == 2
    assert all(f[: len(PATTERN)] == PATTERN for f in frames)


def test_reading_takes_in_no_more_than_a_chunk_at_a_time():
    """A frame is handed over having read a bounded prefix, not the whole file.

    This is the property the reader exists for: what it holds is one chunk of
    blocks, whatever the file's size — so a consumer running beside it does not
    have the whole pass resident behind its back.
    """
    from pycadu.reader import iter_frames_from

    blocks_per_chunk = 64
    with tempfile.TemporaryDirectory() as tmp:
        dat = _fixture_file(tmp)
        with dat.open("rb") as source:
            stream = iter_frames_from(
                source, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE, blocks_per_chunk=blocks_per_chunk
            )
            next(stream)
            consumed = source.tell()
            frames = list(stream)
        size = dat.stat().st_size
    assert consumed < size, f"read {consumed} of {size} bytes before the first frame"
    assert frames, "only one frame came out of a six-frame file"


def test_reading_takes_any_binary_stream_not_only_a_path():
    """A live feed is a stream, not a file, and the reader must not care.

    Handed an open source it pulls from, this is the same reader an operational
    receiver can point at a socket or a pipe.
    """
    import io

    from pycadu.reader import iter_frames_from, iter_frames_read

    with tempfile.TemporaryDirectory() as tmp:
        dat = _fixture_file(tmp)
        expected = list(iter_frames_read(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE))
        source = io.BytesIO(dat.read_bytes())
        actual = list(iter_frames_from(source, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE))
    assert expected
    assert actual == expected


def test_a_frame_split_across_chunk_boundaries_is_completed_not_dropped():
    """Frames straddle the reader's chunks, and every seam must be invisible.

    A frame is ~4300 blocks long, so with any realistic chunk size most frames
    begin in one chunk and end in another — dropping the ones on the seams would
    quietly lose most of the pass. Here the chunk is smaller than a frame, so
    *every* frame spans several of them.
    """
    from pycadu.reader import iter_frames_from, iter_frames_read

    blocks_per_frame = -(-FRAME_SIZE // VCDU_PAYLOAD_LENGTH)
    with tempfile.TemporaryDirectory() as tmp:
        dat = _fixture_file(tmp)
        whole = list(iter_frames_read(dat, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE))
        with dat.open("rb") as source:
            in_pieces = list(
                iter_frames_from(
                    source, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE,
                    blocks_per_chunk=blocks_per_frame // 3,
                )
            )
    assert len(whole) == 6, f"fixture gave {len(whole)} frames"
    assert in_pieces == whole, f"{len(in_pieces)} frames survived being cut into thirds, not {len(whole)}"


class _CountingSource:
    """A binary source that records how much of itself has been read."""

    def __init__(self, data: bytes) -> None:
        """Serve *data*, counting the bytes handed out."""
        self._data = data
        self.consumed = 0

    def readinto(self, buffer) -> int:
        """Fill *buffer* from what is left, as a file object would."""
        chunk = self._data[self.consumed : self.consumed + len(buffer)]
        buffer[: len(chunk)] = chunk
        self.consumed += len(chunk)
        return len(chunk)


def test_reading_ahead_keeps_pulling_while_the_consumer_works():
    """A slow source and a busy consumer should wait for each other less.

    One thread doing both the reading and the scanning alternates between
    blocking on the source and using the CPU — fine on a local disk, wasteful on
    anything with latency. Reading ahead overlaps the two.
    """
    import time

    from pycadu.reader import iter_frames_from

    with tempfile.TemporaryDirectory() as tmp:
        dat = _fixture_file(tmp)
        data = dat.read_bytes()

        def consumed_after_first_frame(prefetch: int) -> int:
            source = _CountingSource(data)
            stream = iter_frames_from(
                source, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE,
                blocks_per_chunk=64, prefetch=prefetch,
            )
            next(stream)
            time.sleep(0.3)  # the consumer is busy with the frame it was given
            ahead = source.consumed
            stream.close()
            return ahead

        alone = consumed_after_first_frame(0)
        reading_ahead = consumed_after_first_frame(4)

    assert reading_ahead > alone, (
        f"reading ahead got through {reading_ahead} bytes against {alone} without"
    )
    assert reading_ahead <= len(data), "the source was drained past what was asked for"


def test_reading_ahead_yields_the_same_frames():
    """Overlapping the read with the scan may not change what comes out."""
    from pycadu.reader import iter_frames_from

    with tempfile.TemporaryDirectory() as tmp:
        dat = _fixture_file(tmp)
        data = dat.read_bytes()
        plain = list(
            iter_frames_from(_CountingSource(data), CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE)
        )
        ahead = list(
            iter_frames_from(
                _CountingSource(data), CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE, prefetch=4
            )
        )
    assert plain
    assert ahead == plain


def test_reading_ahead_stays_bounded():
    """The read-ahead is the depth asked for, not the rest of the pass.

    An unbounded prefetch is just a slower way of loading the whole file, which
    is the thing this reader exists not to do. What it may run ahead by is
    measured against the same scan without it: a frame is far longer than a
    chunk here, so the scan itself consumes many chunks before the first frame
    appears, and that is not read-ahead.
    """
    import time

    from pycadu.reader import iter_frames_from

    blocks_per_chunk, prefetch = 64, 2

    def consumed_after_first_frame(depth: int, data: bytes) -> int:
        source = _CountingSource(data)
        stream = iter_frames_from(
            source, CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE,
            blocks_per_chunk=blocks_per_chunk, prefetch=depth,
        )
        next(stream)
        time.sleep(0.3)
        consumed = source.consumed
        stream.close()
        return consumed

    with tempfile.TemporaryDirectory() as tmp:
        data = _fixture_file(tmp).read_bytes() * 4
        alone = consumed_after_first_frame(0, data)
        ahead = consumed_after_first_frame(prefetch, data)

    chunk_bytes = blocks_per_chunk * CADU_STRIDE_1024
    # `prefetch` queued, one held by the reader, one part-filled
    assert ahead - alone <= chunk_bytes * (prefetch + 2), (
        f"read {(ahead - alone) / chunk_bytes:.1f} chunks ahead with a depth of {prefetch}"
    )


class _FailingSource:
    """A source that breaks part way through, as a socket or a disk can."""

    def __init__(self, data: bytes, *, fails_after: int) -> None:
        """Serve *data*, raising once *fails_after* reads have been answered."""
        self._data = data
        self._reads = 0
        self._fails_after = fails_after
        self.consumed = 0

    def readinto(self, buffer) -> int:
        """Fill *buffer*, or raise if this source has served its quota."""
        if self._reads >= self._fails_after:
            msg = "the source went away"
            raise OSError(msg)
        self._reads += 1
        chunk = self._data[self.consumed : self.consumed + len(buffer)]
        buffer[: len(chunk)] = chunk
        self.consumed += len(chunk)
        return len(chunk)


def test_a_slow_consumer_still_reaches_the_end_of_the_stream():
    """The end of the stream has to arrive however long the consumer takes.

    The reader signals the end through the same bounded queue it hands chunks
    over on. If that signal can be dropped when the queue is momentarily full,
    a consumer that pauses — for as long as calibrating a chunk takes — drains
    what is queued and then waits for an end that was thrown away.
    """
    import time

    from pycadu.reader import iter_frames_from

    with tempfile.TemporaryDirectory() as tmp:
        dat = _fixture_file(tmp)
        stream = iter_frames_from(
            _CountingSource(dat.read_bytes()), CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE,
            blocks_per_chunk=64, prefetch=1,
        )
        frames = []
        for frame in stream:
            frames.append(frame)
            if len(frames) == 1:
                time.sleep(1.2)  # longer than any timeout the handover may use

    assert len(frames) == 6, f"{len(frames)} frames came off a six-frame pass"


def test_a_source_that_breaks_is_reported_however_full_the_queue_is():
    """A failure must reach the consumer, not be swallowed with the sentinel.

    Reading ahead moves the failure onto another thread. If it is only carried
    alongside a signal that can be dropped, a broken source becomes a silent
    short pass — which is the worst outcome available.
    """
    import time

    from pycadu.reader import iter_frames_from

    with tempfile.TemporaryDirectory() as tmp:
        dat = _fixture_file(tmp)
        stream = iter_frames_from(
            _FailingSource(dat.read_bytes(), fails_after=200), CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE,
            blocks_per_chunk=64, prefetch=1,
        )
        with pytest.raises(OSError, match="went away"):
            for _frame in stream:
                time.sleep(1.2)  # let the producer finish while the queue is full


def test_a_stream_with_no_sync_in_it_does_not_accumulate():
    """A stream that never syncs must cost the same however long it runs.

    The wrong sync pattern, a corrupted pass, or a long lead-in before the first
    frame all look the same to the reader: nothing to cut a frame on. If it
    keeps every byte while it waits for a sync, the documented one-chunk bound
    fails in exactly the case where the stream is longest. What is measured is
    therefore the *growth*: scanning four times as much sync-free data must not
    cost four times the memory.
    """
    import tracemalloc

    from pycadu.reader import iter_frames_from

    def peak_scanning(blocks: int) -> float:
        """Peak tracked memory, in MB, while scanning *blocks* of sync-free CADU."""
        junk = _wrap_frames_in_cadu_blocks(b"\x00" * (blocks * VCDU_PAYLOAD_LENGTH), vcid=3)
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            frames = list(
                iter_frames_from(_CountingSource(junk), CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE)
            )
            assert frames == [], "sync-free data yielded frames"
            return (tracemalloc.get_traced_memory()[1] - before) / 1e6
        finally:
            tracemalloc.stop()

    short = peak_scanning(10_000)
    long = peak_scanning(40_000)

    assert long < short * 1.5, (
        f"scanning 4x the sync-free data cost {long:.0f} MB against {short:.0f} MB — it is accumulating"
    )


def test_finding_the_sync_does_not_pin_the_chunk_it_was_found_in():
    """What the seek held is not wanted once it has found what it sought.

    The search accumulates until it syncs, and everything before the sync is by
    definition junk. Holding that chunk anyway pins it for the length of the
    pass — a fixed cost rather than a growing one, so the sync-free growth test
    cannot see it, and the bound this reader exists for is one chunk *worse*
    than advertised on every pass that does sync. It is held twice over if the
    caller keeps the copy the seek handed it, too.

    What the reader may legitimately hold mid-pass is the chunk it is reading,
    the part of the previous one it has not cut into frames yet, and the frame
    it has just handed over. Anything beyond that is a copy nobody wants.
    """
    import gc
    import tracemalloc

    from pycadu.reader import BLOCKS_PER_CHUNK, iter_frames_from

    # Two chunks of sync-free lead-in, then the frames: the sync is found in a
    # chunk of its own, which is what would be retained.
    junk = b"\x00" * (2 * BLOCKS_PER_CHUNK * VCDU_PAYLOAD_LENGTH)
    data = _wrap_frames_in_cadu_blocks(junk + _two_frames(), vcid=3)

    tracemalloc.start()
    try:
        gc.collect()
        before = tracemalloc.get_traced_memory()[0]
        stream = iter_frames_from(_CountingSource(data), CADU_STRIDE_1024, 3, PATTERN, FRAME_SIZE)
        first = next(stream)
        gc.collect()
        # While the reader is suspended mid-pass: what it is *holding*, not its peak.
        held = tracemalloc.get_traced_memory()[0] - before
        stream.close()
    finally:
        tracemalloc.stop()

    assert len(first) == FRAME_SIZE
    budget = (
        BLOCKS_PER_CHUNK * CADU_STRIDE_1024  # the chunk being read
        + BLOCKS_PER_CHUNK * VCDU_PAYLOAD_LENGTH  # its payload, not yet cut into frames
        + FRAME_SIZE  # the frame just handed over
    )
    assert held < budget, (
        f"held {held / 1e6:.1f} MB against a budget of {budget / 1e6:.1f} MB — "
        "about a chunk more than the reader has any use for"
    )


BLOCKS_PER_FRAME = 3


def _blocks_numbered(counters: list[int], vcid: int = 3) -> bytes:
    """Blocks carrying *counters*, block *n* holding block ``n % 3`` of frame ``n // 3``.

    A counter left out of *counters* is a block lost on the way down: its payload
    is missing from the stream as well as its number.
    """
    out = bytearray()
    for counter in counters:
        frame, block = divmod(counter, BLOCKS_PER_FRAME)
        payload = bytes([0x10 + frame % 0x40]) * VCDU_PAYLOAD_LENGTH
        if block == 0:
            payload = PATTERN + payload[len(PATTERN) :]
        header = CCSDS_SYNC_MARKER + bytes([0x4D, vcid & 0x3F]) + (counter % 2**24).to_bytes(3, "big")
        out += header.ljust(CCSDS_HEADER_LENGTH, b"\x00") + payload + b"\x00" * RS_PARITY_LENGTH
    return bytes(out)


def _frames_and_warnings(stream: bytes, caplog) -> tuple[list[bytes], list]:
    """Cut *stream* through the public reader, keeping the warnings it logs."""
    from pycadu.reader import FrameSpec, iter_frames

    spec = FrameSpec(CADU_STRIDE_1024, 3, PATTERN, BLOCKS_PER_FRAME * VCDU_PAYLOAD_LENGTH)
    with caplog.at_level("WARNING", logger="pycadu"):
        frames = list(iter_frames(io.BytesIO(stream), spec, blocks_per_chunk=2))
    return frames, [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_frame_that_lost_blocks_says_how_many(caplog):
    """The VCDU counter is the only witness to a block that never arrived."""
    frames, warnings = _frames_and_warnings(_blocks_numbered([0, 1, 3, 4, 5, 6, 7, 8]), caplog)

    assert [len(f) for f in frames] == [2 * VCDU_PAYLOAD_LENGTH] + [3 * VCDU_PAYLOAD_LENGTH] * 2
    assert len(warnings) == 1, [w.getMessage() for w in warnings]
    assert (warnings[0].frame_index, warnings[0].lost_blocks) == (0, 1)
    assert "frame 0" in warnings[0].getMessage()
    assert "1 VCDU block" in warnings[0].getMessage()


def test_the_first_block_of_a_stream_has_lost_nothing(caplog):
    """A pass starts wherever the counter happens to be, with no predecessor to have lost."""
    frames, warnings = _frames_and_warnings(_blocks_numbered(list(range(999, 1008))), caplog)

    assert len(frames) == 3
    assert warnings == [], [w.getMessage() for w in warnings]


@pytest.mark.xfail(strict=True, reason="a counter wrapping to 0 reads as a step back")
def test_a_counter_wrapping_round_has_lost_nothing(caplog):
    """The counter is 24 bits wide, so after 0xFFFFFF comes 0, not a loss."""
    frames, warnings = _frames_and_warnings(_blocks_numbered(list(range(2**24 - 4, 2**24 + 5))), caplog)

    assert len(frames) == 3
    assert warnings == [], [w.getMessage() for w in warnings]


def test_a_repeated_block_has_lost_nothing(caplog):
    """A block sent twice under one counter is a duplicate, not a loss of the whole counter range."""
    counters = [0, 1, 2, 3, 4, 5, 5, 6, 7, 8]  # block 5 arrives twice

    frames, warnings = _frames_and_warnings(_blocks_numbered(counters), caplog)

    assert len(frames) == 3
    assert warnings == [], [w.getMessage() for w in warnings]
