import numpy as np

from pycadu.sync import find_sync_pattern

PATTERN = bytes([0x1A, 0xCF, 0xFC, 0x1D])  # CCSDS ASM — a generic test fixture


def test_returns_offset_when_pattern_found_at_start():
    data = PATTERN + bytes(100)
    assert find_sync_pattern(data, PATTERN) == (0, 0)


def test_returns_offset_when_pattern_found_mid_buffer():
    data = bytes(50) + PATTERN + bytes(50)
    assert find_sync_pattern(data, PATTERN) == (50, 0)


def test_returns_minus_one_and_no_shift_when_pattern_absent():
    """The shift must be None, not 0: 0 is a valid shift and callers unpack it directly."""
    data = bytes(100)
    assert find_sync_pattern(data, PATTERN) == (-1, None)


def test_respects_start_offset():
    data = PATTERN + bytes(50) + PATTERN
    assert find_sync_pattern(data, PATTERN, start=len(PATTERN)) == (len(PATTERN) + 50, 0)


def test_finds_bit_shifted_pattern():
    pattern = np.frombuffer(PATTERN, dtype=np.uint8)
    shifted = bytearray(len(PATTERN) + 1)
    for i in range(len(PATTERN)):
        shifted[i] |= pattern[i] >> 2
        shifted[i + 1] |= (pattern[i] << 6) & 0xFF
    data = bytes(10) + bytes(shifted) + bytes(20)
    byte_pos, bit_shift = find_sync_pattern(data, PATTERN)
    assert byte_pos == 10
    assert bit_shift == 2
