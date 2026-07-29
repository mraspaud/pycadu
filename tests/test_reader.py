import math
import tempfile
from pathlib import Path

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
