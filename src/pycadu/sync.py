"""Bit-level sync pattern search for CADU/VCDU streams."""

import numpy as np


def find_sync_pattern(data: bytes, pattern: bytes, start: int = 0) -> tuple[int, int | None]:
    """Search for *pattern* in *data* at all 8 bit offsets.

    Returns ``(byte_pos, bit_shift)`` of the earliest match, or ``(-1, None)`` if not
    found: the shift is None rather than 0 so that a caller unpacking straight into
    its own "shift not yet known" state keeps that state instead of latching onto a
    bogus byte-aligned 0.  *start* constrains the search to positions ``>= start``.
    """
    pos = data.find(pattern, start)
    if pos != -1:
        return (pos, 0)

    best: tuple[int, int] | None = None
    for shift in range(1, 8):
        shifted = _build_shifted(data, shift)
        pos = shifted.find(pattern, start)
        if pos != -1:
            if best is None or pos < best[0]:
                best = (pos, shift)
            break

    return best if best is not None else (-1, None)


def _build_shifted(data: bytes, shift: int) -> bytes:
    if shift == 0:
        return data
    arr = np.frombuffer(data, dtype=np.uint8)
    lo = arr[:-1] << np.uint8(shift)
    lo |= arr[1:] >> np.uint8(8 - shift)
    return lo.tobytes()
