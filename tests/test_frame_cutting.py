"""Both ways of cutting frames must agree about what a stream contains.

Frames are reassembled from two kinds of source — a stream of payload byte
chunks, and chunks of physical blocks read or mapped in bulk — and the two
carried separate implementations of the same reassembly. They are only
interchangeable if they answer the same way on the awkward inputs: a sync
pattern that is bit-shifted, chunk boundaries that fall mid-frame, and a run of
corruption that has to be resynchronised past.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pycadu.constants import CADU_DATA_LENGTH, CCSDS_HEADER_LENGTH, CCSDS_SYNC_MARKER, RS_PARITY_LENGTH
from pycadu.reader import _frames_from_block_chunks, vcdu_frames

PATTERN = bytes([0xAA, 0x55] * 6)
FRAME_SIZE = 512
VCID = 3


def _frames(count: int) -> bytes:
    """*count* distinguishable frames, each opening with the sync pattern."""
    out = bytearray()
    for index in range(count):
        frame = bytearray(FRAME_SIZE)
        frame[: len(PATTERN)] = PATTERN
        frame[len(PATTERN)] = index
        out += frame
    return bytes(out)


def _shifted(data: bytes, bits: int) -> bytes:
    """The same bytes with the whole stream shifted right by *bits*.

    A trailing byte is added first so the shift pushes padding off the end
    rather than the last frame's final bits.
    """
    if bits == 0:
        return data
    padded = data + b"\x00"
    value = int.from_bytes(padded, "big") >> bits
    return value.to_bytes(len(padded), "big")


def _as_blocks(payload_stream: bytes) -> np.ndarray:
    """Wrap a payload stream in CADU blocks, as they arrive off the wire."""
    payload_len = CADU_DATA_LENGTH - CCSDS_HEADER_LENGTH
    blocks = []
    for start in range(0, len(payload_stream), payload_len):
        payload = payload_stream[start : start + payload_len].ljust(payload_len, b"\x00")
        header = CCSDS_SYNC_MARKER + bytes([0x4D, VCID & 0x3F]) + b"\x00" * (CCSDS_HEADER_LENGTH - 6)
        blocks.append(header + payload + b"\x00" * RS_PARITY_LENGTH)
    return np.frombuffer(b"".join(blocks), dtype=np.uint8).reshape(len(blocks), -1)


def _cut_from_bytes(stream: bytes, chunk: int) -> list[bytes]:
    """Frames as the payload-stream reassembler sees them."""
    chunks = (stream[i : i + chunk] for i in range(0, len(stream), chunk))
    return list(vcdu_frames(chunks, PATTERN, FRAME_SIZE))


def _cut_from_blocks(stream: bytes, blocks_per_chunk: int) -> list[bytes]:
    """Frames as the block-chunk reassembler sees them."""
    blocks = _as_blocks(stream)
    chunks = (blocks[i : i + blocks_per_chunk] for i in range(0, len(blocks), blocks_per_chunk))
    return list(_frames_from_block_chunks(chunks, VCID, PATTERN, FRAME_SIZE))


@pytest.mark.parametrize("bits", range(8))
def test_the_two_readers_agree_on_a_bit_shifted_stream(bits):
    """Whatever the sync alignment, both find the same frames."""
    stream = _shifted(b"\x00" * 64 + _frames(4), bits)

    from_bytes = _cut_from_bytes(stream, chunk=97)
    from_blocks = _cut_from_blocks(stream, blocks_per_chunk=2)

    assert len(from_bytes) == 4, f"payload reader found {len(from_bytes)} frames of 4"
    assert from_bytes == from_blocks


def test_sync_lost_mid_stream_is_recovered_by_both_readers():
    """Corruption costs the frames it touches, not the rest of the pass."""
    good = _frames(2)
    corrupt = bytes([0xFF] * (FRAME_SIZE + 7))  # not a whole number of frames
    stream = good + corrupt + _frames(2)

    from_bytes = _cut_from_bytes(stream, chunk=101)
    from_blocks = _cut_from_blocks(stream, blocks_per_chunk=3)

    assert from_bytes == from_blocks
    assert len(from_bytes) >= 3, f"only {len(from_bytes)} frames survived the corruption"


def test_a_sync_pattern_straddling_a_chunk_boundary_is_still_found():
    """A pattern split across two chunks must not be dropped with the buffer.

    Discarding everything on a failed search loses a pattern whose first bytes
    are already in hand, which costs the whole rest of the stream rather than
    the corruption that caused the resynchronisation.
    """
    corrupt = bytes([0xFF] * (FRAME_SIZE * 2 - len(PATTERN) // 2))
    stream = corrupt + _frames(3)

    # A chunk size that lands a boundary inside the first real sync pattern.
    boundary = len(corrupt) + len(PATTERN) // 2
    chunks = [stream[:boundary], stream[boundary:]]

    from_bytes = list(vcdu_frames(iter(chunks), PATTERN, FRAME_SIZE))
    assert len(from_bytes) == 3, f"the straddling pattern cost {3 - len(from_bytes)} frames"


def test_a_frame_after_corruption_is_not_lost_with_the_buffer():
    """A run of corruption must cost its own frames and no more.

    Found by fuzzing the two readers against each other: on this stream the
    payload reader returned two frames where the block reader returned three.
    Discarding the whole buffer on a failed pattern search throws away bytes
    that the next sync pattern has already started in, so the frame after the
    corruption never begins.
    """
    corrupt = (Path(__file__).parent / "data_corrupt_run.bin").read_bytes()
    assert PATTERN not in corrupt, "the fixture must not contain a sync pattern"
    stream = _frames(1) + corrupt + _frames(2)
    chunks = [stream[i : i + 78] for i in range(0, len(stream), 78)]

    assert len(list(vcdu_frames(iter(chunks), PATTERN, FRAME_SIZE))) == 3


def test_a_pattern_begun_at_the_end_of_a_discarded_buffer_survives():
    """Resynchronising must keep the bytes a pattern could still be starting in.

    This is the exact shape where discarding the whole buffer differs from
    keeping a pattern-length tail: the buffer is long enough to be searched, the
    search fails because the pattern is only partly present, and the missing
    part is in the very next chunk. Throwing the partial match away costs every
    frame after it.
    """
    partial = len(PATTERN) - 1
    first = bytes([0xFF] * FRAME_SIZE) + PATTERN[:partial]
    second = PATTERN[partial:] + bytes(FRAME_SIZE - len(PATTERN)) + _frames(2)

    found = list(vcdu_frames(iter([first, second]), PATTERN, FRAME_SIZE))
    assert len(found) == 3, f"the split pattern cost {3 - len(found)} of 3 frames"
