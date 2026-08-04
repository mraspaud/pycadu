"""Generic CADU/VCDU streaming and batch frame reassembly."""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from pycadu.cadu import extract_cadu_data, extract_vcdu_payload, extract_vcid
from pycadu.constants import CADU_DATA_LENGTH, CCSDS_HEADER_LENGTH
from pycadu.sync import _build_shifted, find_sync_pattern

#: Physical blocks scanned per vectorised pass. The chunk is what a reading
#: source holds at once, so it also sets its memory cost.
BLOCKS_PER_CHUNK = 10_000


class _SyncSeeker:
    """Looks for the first sync pattern in a stream, keeping only what it must.

    The pattern can straddle whatever the stream was cut into, and it can sit at
    any of eight bit alignments, so the search cannot be done chunk by chunk in
    isolation. What it *can* do is forget: once a chunk has been searched and
    found wanting, the only bytes still of interest are the last few, where a
    pattern beginning at the end of one chunk continues into the next.

    Keeping the whole stream instead is the difference between a bounded reader
    and one that holds an entire pass whenever the sync is late, the pattern is
    wrong, or the data is corrupt — the cases where the stream is longest.
    """

    def __init__(self, pattern: bytes) -> None:
        """Seek *pattern*, retaining only enough to catch a straddling one."""
        self._pattern = pattern
        #: A pattern shifted by up to seven bits spans one byte more than it is.
        self._overlap = len(pattern) + 1
        self._seen = bytearray()
        #: The most recent byte fed in, which carries the unshift into the next
        #: chunk. Kept apart from the accumulator so that finding the sync can
        #: throw the accumulator away without losing it.
        self.last_byte = 0

    def feed(self, payload) -> tuple[bytes, int] | None:
        """Search *payload*, with the tail of what came before it.

        Returns ``(from_the_sync_on, bit_shift)`` once the pattern is found —
        the stream from the first frame's first byte, shifted onto byte
        alignment — or None, having kept only the overlap.

        What came *before* the sync is junk by definition, and never leaves
        here: everything held is dropped on the way out. A seeker that kept the
        chunk it succeeded on, or handed it to a caller that did, would pin it
        for the length of the pass — a fixed cost that no growth measurement
        can see and every pass that syncs would pay.
        """
        self._seen.extend(payload)
        if self._seen:
            self.last_byte = self._seen[-1]
        byte_pos, bit_shift = find_sync_pattern(bytes(self._seen), self._pattern)
        if byte_pos == -1:
            if len(self._seen) > self._overlap:
                del self._seen[: len(self._seen) - self._overlap]
            return None
        held = bytes(self._seen)
        self._seen = bytearray()
        aligned = _build_shifted(held, bit_shift) if bit_shift else held
        return aligned[byte_pos:], bit_shift


def cadu_blocks(path: Path, stride: int) -> Iterator[bytes]:
    """Yield one raw physical block at a time without loading the whole file."""
    with open(path, "rb") as fh:
        while block := fh.read(stride):
            if len(block) == stride:
                yield block


def vcdu_payloads(blocks: Iterator[bytes], stride: int, target_vcid: int) -> Iterator[bytes]:
    """Yield VCDU payload bytes for blocks matching *target_vcid*."""
    for block in blocks:
        if extract_vcid(block) == target_vcid:
            yield extract_vcdu_payload(extract_cadu_data(block, stride))


def vcdu_frames(payloads: Iterator[bytes], pattern: bytes, frame_size: int) -> Iterator[bytes]:
    """Reassemble VCDU payload chunks into complete fixed-length frames.

    Detects the bit-shift of the sync pattern (0-7 bits) on first sync find,
    then unshifts each subsequent chunk using a one-byte carry to maintain
    bit alignment across chunk boundaries.
    """
    seeker = _SyncSeeker(pattern)
    buf = bytearray()
    bit_shift: int | None = None
    carry: int = 0

    for chunk in payloads:
        if bit_shift is None:
            found = seeker.feed(chunk)
            if found is None:
                continue
            buf = bytearray(found[0])
            bit_shift = found[1]
            carry = seeker.last_byte
            # Let go of the seek's copy: this frame outlives the whole pass.
            found = None
        else:
            if bit_shift:
                arr = np.frombuffer(bytes([carry]) + bytes(chunk), dtype=np.uint8)
                lo = arr[:-1].astype(np.uint16) << bit_shift
                hi = arr[1:].astype(np.uint16) >> (8 - bit_shift)
                buf.extend(((lo | hi) & 0xFF).astype(np.uint8))
                carry = chunk[-1] if chunk else carry
            else:
                buf.extend(chunk)

        while len(buf) >= frame_size:
            if bytes(buf[: len(pattern)]) != pattern:
                nxt = bytes(buf).find(pattern)
                if nxt == -1:
                    buf.clear()
                    break
                del buf[:nxt]
                continue
            yield bytes(buf[:frame_size])
            del buf[:frame_size]


def iter_vcdu_frames(
    path: Path, pattern: bytes, frame_size: int, vcid: int, stride: int
) -> Iterator[bytes]:
    """Yield complete frames from a CADU file for the given VCID."""
    return vcdu_frames(vcdu_payloads(cadu_blocks(path, stride), stride, vcid), pattern, frame_size)


def _mapped_block_chunks(path: Path, stride: int, blocks_per_chunk: int) -> Iterator[np.ndarray]:
    """Yield the file's blocks a chunk at a time, as views onto one mapping.

    Nothing is copied, which is the fastest way to scan a file that is already
    there. The pages a scan touches would otherwise stay resident for as long as
    the reader lives — the whole file, for a reader that runs beside the pipeline
    consuming it — so each chunk is handed back to the kernel once the consumer
    has moved past it. The scan only ever walks forward, which is what makes
    that safe; a page asked for again is simply faulted back in.

    Use :func:`_read_block_chunks` for a source that cannot be mapped at all.
    """
    import mmap as _mmap

    with open(path, "rb") as fh, _mmap.mmap(fh.fileno(), 0, access=_mmap.ACCESS_READ) as mapping:
        raw = np.frombuffer(mapping, dtype=np.uint8)
        n_blocks = len(raw) // stride
        # ``raw`` is a view onto the mapping, so it must be released before the
        # mmap closes — including when a consumer abandons the generator early,
        # which is why this is a try/finally and not a trailing ``del``.
        try:
            for start in range(0, n_blocks, blocks_per_chunk):
                stop = min(start + blocks_per_chunk, n_blocks)
                yield raw[start * stride : stop * stride].reshape(stop - start, stride)
                _release_pages(mapping, start * stride, stop * stride)
        finally:
            del raw


def _release_pages(mapping, first_byte: int, last_byte: int) -> None:
    """Drop the resident pages wholly inside ``[first_byte, last_byte)``.

    Whole pages only: a partial one at either end still holds bytes the scan has
    not finished with. Advisory in every sense — where the platform does not
    offer it, the mapping simply stays resident and only memory suffers.
    """
    import mmap as _mmap

    page = _mmap.PAGESIZE
    start = -(-first_byte // page) * page
    length = ((last_byte - start) // page) * page
    if length > 0:
        with contextlib.suppress(OSError, ValueError, AttributeError):
            mapping.madvise(_mmap.MADV_DONTNEED, start, length)


def _read_block_chunks(source, stride: int, blocks_per_chunk: int) -> Iterator[np.ndarray]:
    """Yield the source's blocks a chunk at a time, reading each one in.

    What is held is one chunk, whatever the source's size, and the source is
    only ever pulled from — so this reads a socket or a pipe as readily as a
    file, and stops at whatever the source declares to be the end.

    Each chunk gets its own buffer, so one may be handed on while the next is
    being filled. That is what lets a prefetcher sit between this and the scan.
    """
    carried = np.zeros(0, dtype=np.uint8)
    while True:
        buffer = np.empty(blocks_per_chunk * stride, dtype=np.uint8)
        view = memoryview(buffer.data).cast("B")
        held = len(carried)
        if held:
            buffer[:held] = carried
        filled = source.readinto(view[held:])
        if not filled:
            break
        filled += held
        # A read may stop mid-block; carry the remainder into the next chunk
        # rather than losing it, since a source delivers what it has, not what
        # it was asked for.
        whole = filled // stride
        carried = buffer[whole * stride : filled].copy()
        if whole:
            yield buffer[: whole * stride].reshape(whole, stride)


def _prefetched(chunks: Iterator[np.ndarray], depth: int) -> Iterator[np.ndarray]:
    """Yield what *chunks* yields, pulling it on a thread *depth* chunks ahead.

    The reading and the scanning then overlap instead of taking turns, which is
    worth having whenever the source has latency — a network filesystem, a
    socket — and costs one buffer per queued chunk, so the depth is what bounds
    it. The thread stops when the consumer does, including when it walks away
    early.
    """
    import queue as _queue
    import threading

    handover: _queue.Queue = _queue.Queue(maxsize=depth)
    stop = threading.Event()

    def hand_over(item) -> bool:
        """Put *item* on the queue, giving up only if the consumer has left.

        Never on a timeout. The end of the stream travels this way too, and a
        consumer busy for longer than any timeout we could pick would then wait
        forever for an end that had been dropped — which is what a chunk of
        calibration looks like.
        """
        while not stop.is_set():
            try:
                handover.put(item, timeout=0.05)
            except _queue.Full:
                continue
            return True
        return False

    def pull() -> None:
        """Fill the queue until the source ends, the consumer leaves, or it breaks."""
        ending: tuple[str, BaseException | None] = ("done", None)
        try:
            for chunk in chunks:
                if not hand_over(("chunk", chunk)):
                    return
        except BaseException as broke:  # re-raised on the consumer's thread
            ending = ("failed", broke)
        # The outcome rides on the queue rather than beside it, so a consumer
        # that receives the end always receives what the end was.
        hand_over(ending)

    reader = threading.Thread(target=pull, name="cadu-prefetch", daemon=True)
    reader.start()
    try:
        while True:
            kind, payload = handover.get()
            if kind == "chunk":
                yield payload
            elif kind == "failed":
                raise payload
            else:
                return
    finally:
        stop.set()
        reader.join(timeout=5.0)


def _frames_from_block_chunks(
    chunks: Iterator[np.ndarray], vcid: int, pattern: bytes, frame_size: int
) -> Iterator[bytes]:
    """Reassemble frames of *vcid* out of chunks of physical blocks.

    The scanning itself: select the blocks of the wanted virtual channel, strip
    their payloads, unshift them onto the sync pattern's bit alignment, and cut
    frames out of the running buffer. Where the blocks came from — a mapping or
    a stream — is the caller's choice and makes no difference to what comes out.
    """
    pat_len = len(pattern)
    buf = bytearray()
    bit_shift: int | None = None
    raw_carry: int = 0
    seeker = _SyncSeeker(pattern)
    scratch: dict[str, np.ndarray] = {}

    def _unshifted(payloads: np.ndarray, shift: int, carry: int) -> np.ndarray:
        """Shift a chunk's payloads into alignment, reusing the working arrays."""
        n = len(payloads)
        if scratch.get("ext") is None or len(scratch["ext"]) < n + 1:
            scratch["ext"] = np.empty(n + 1, dtype=np.uint8)
            scratch["sft"] = np.empty(n, dtype=np.uint8)
            scratch["rhs"] = np.empty(n, dtype=np.uint8)
        ext, sft, rhs = scratch["ext"], scratch["sft"], scratch["rhs"]
        ext[0] = carry
        ext[1 : n + 1] = payloads
        np.left_shift(ext[:n], shift, out=sft[:n])
        np.right_shift(ext[1 : n + 1], 8 - shift, out=rhs[:n])
        np.bitwise_or(sft[:n], rhs[:n], out=sft[:n])
        return sft[:n]

    for chunk in chunks:
        wanted = chunk[(chunk[:, 5] & np.uint8(0x3F)) == vcid]
        payloads = wanted[:, CCSDS_HEADER_LENGTH:CADU_DATA_LENGTH].copy().ravel()
        # Let go of the chunk before anything is yielded: it may be a window
        # onto a mapping or a buffer the source is about to close or refill, and
        # a copy of the payloads is all that is wanted from it anyway.
        chunk = wanted = None
        if len(payloads) == 0:
            continue

        if bit_shift is None:
            found = seeker.feed(payloads)
            raw_carry = seeker.last_byte
            if found is None:
                del payloads
                continue
            buf.extend(found[0])
            bit_shift = found[1]
            # Let go of the seek's copy: this frame outlives the whole pass.
            found = None
        elif bit_shift:
            buf.extend(_unshifted(payloads, bit_shift, raw_carry))
            raw_carry = int(payloads[-1])
        else:
            buf.extend(payloads)
            raw_carry = int(payloads[-1])
        del payloads

        while len(buf) >= frame_size:
            if buf[:pat_len] == bytearray(pattern):
                yield bytes(buf[:frame_size])
                del buf[:frame_size]
            else:
                nxt = bytes(buf).find(pattern)
                if nxt == -1:
                    del buf[: len(buf) - pat_len]
                    break
                del buf[:nxt]


def iter_frames_batch(
    path: Path,
    stride: int,
    vcid: int,
    pattern: bytes,
    frame_size: int,
) -> Iterator[bytes]:
    """Yield all frames of *vcid* from a CADU file, scanning a mapping of it.

    ~20x faster than the per-block streaming generator, because the blocks are
    selected and stripped with vectorised numpy. Nothing is copied out of the
    file, so the pages the scan touches stay resident as long as the generator
    lives; :func:`iter_frames_read` is the same scan over a source that is read
    instead, for a reader that runs alongside its consumer.
    """
    return _frames_from_block_chunks(
        _mapped_block_chunks(path, stride, BLOCKS_PER_CHUNK), vcid, pattern, frame_size
    )


def iter_frames_from(
    source,
    stride: int,
    vcid: int,
    pattern: bytes,
    frame_size: int,
    *,
    blocks_per_chunk: int = BLOCKS_PER_CHUNK,
    prefetch: int = 0,
) -> Iterator[bytes]:
    """Yield all frames of *vcid* read from an open binary *source*.

    The same scan :func:`iter_frames_batch` does, over anything that can be read
    from — a file, a pipe, a socket, or a
    :class:`~pycadu.tail.TailedFile` following a pass as it is received. It
    pulls: a chunk of blocks is read in, scanned and handed on, so what it holds
    is one chunk however long the source turns out to be, and it ends when the
    source does.

    *prefetch* puts the reading on its own thread, that many chunks ahead of the
    scan, so a source with latency is being read while the previous chunk is
    still being scanned. Zero — the default — keeps everything on the calling
    thread, which is what a local file wants.
    """
    chunks = _read_block_chunks(source, stride, blocks_per_chunk)
    if prefetch:
        chunks = _prefetched(chunks, prefetch)
    return _frames_from_block_chunks(chunks, vcid, pattern, frame_size)


def iter_frames_read(
    path: Path,
    stride: int,
    vcid: int,
    pattern: bytes,
    frame_size: int,
    *,
    blocks_per_chunk: int = BLOCKS_PER_CHUNK,
) -> Iterator[bytes]:
    """Yield all frames of *vcid* from a CADU file, reading it a chunk at a time.

    Frame for frame what :func:`iter_frames_batch` yields. The difference is
    residency: this holds one chunk of blocks rather than however much of the
    file the scan has touched, which is what a reader running beside the
    pipeline consuming it needs.
    """
    with open(path, "rb") as source:
        yield from iter_frames_from(
            source, stride, vcid, pattern, frame_size, blocks_per_chunk=blocks_per_chunk
        )


def read_frames_batch(
    path: Path,
    stride: int,
    vcid: int,
    pattern: bytes,
    frame_size: int,
) -> list[bytes]:
    """Read all frames from a CADU file into a list.

    Thin wrapper over :func:`iter_frames_batch`, which does the scanning; use
    that directly to process frames as they are found instead of holding the
    whole pass in memory.
    """
    return list(iter_frames_batch(path, stride, vcid, pattern, frame_size))
