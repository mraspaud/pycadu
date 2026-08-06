"""Generic CADU/VCDU streaming and batch frame reassembly."""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

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


class _ShiftBuffers:
    """Reusable working arrays for unshifting a chunk onto byte alignment.

    A pass is millions of bytes in chunks of a few hundred thousand; allocating
    three arrays per chunk is allocation the reader does not need to do.
    """

    def __init__(self) -> None:
        """Start with nothing allocated; the first chunk sizes the buffers."""
        self._extended: np.ndarray | None = None
        self._shifted: np.ndarray | None = None
        self._remainder: np.ndarray | None = None

    def unshift(self, payloads: np.ndarray, shift: int, carry: int) -> np.ndarray:
        """Return *payloads* shifted left by *shift* bits, continuing from *carry*."""
        n = len(payloads)
        if self._extended is None or len(self._extended) < n + 1:
            self._extended = np.empty(n + 1, dtype=np.uint8)
            self._shifted = np.empty(n, dtype=np.uint8)
            self._remainder = np.empty(n, dtype=np.uint8)
        extended, shifted, remainder = self._extended, self._shifted, self._remainder
        extended[0] = carry
        extended[1 : n + 1] = payloads
        np.left_shift(extended[:n], shift, out=shifted[:n])
        np.right_shift(extended[1 : n + 1], 8 - shift, out=remainder[:n])
        np.bitwise_or(shifted[:n], remainder[:n], out=shifted[:n])
        return shifted[:n]


class _FrameCutter:
    """Cut fixed-length frames out of a stream of payload bytes.

    The awkward parts of reassembly live here so that both readers share them:
    the sync pattern arrives at any of eight bit alignments and may straddle
    whatever the stream was cut into, so alignment is discovered once and then
    carried across chunks a byte at a time.

    Where the payloads came from — a byte stream, a mapping, a socket — makes no
    difference, which is the point: the two readers differ only in how they get
    bytes, and used to differ in how they cut them too.
    """

    def __init__(self, pattern: bytes, frame_size: int) -> None:
        """Cut frames of *frame_size* bytes, each opening with *pattern*."""
        self._pattern = pattern
        self._frame_size = frame_size
        self._seeker = _SyncSeeker(pattern)
        self._buffer = bytearray()
        self._bit_shift: int | None = None
        self._carry = 0
        self._buffers = _ShiftBuffers()

    def feed(self, payloads) -> Iterator[bytes]:
        """Take one chunk of payload bytes and return the frames it completes.

        Absorbing is a plain call rather than part of the generator so that the
        chunk is released before any frame is yielded. A generator suspended
        mid-pass keeps its locals alive, which would pin a whole chunk for as
        long as the consumer takes over the frame it was handed.
        """
        self._absorb(payloads)
        return self._cut()

    def _absorb(self, payloads) -> None:
        """Add one chunk to the buffer, finding the sync alignment if needed."""
        if len(payloads) == 0:
            return
        if self._bit_shift is None:
            found = self._seeker.feed(payloads)
            self._carry = self._seeker.last_byte
            if found is None:
                return
            self._buffer.extend(found[0])
            self._bit_shift = found[1]
            # Let go of the seek's copy: this frame outlives the whole pass.
            found = None
        elif self._bit_shift:
            self._buffer.extend(self._buffers.unshift(_as_array(payloads), self._bit_shift, self._carry))
            self._carry = int(payloads[-1])
        else:
            self._buffer.extend(payloads)
            self._carry = int(payloads[-1])

    def _cut(self) -> Iterator[bytes]:
        """Yield every whole frame the buffer now holds, resynchronising if needed."""
        pattern, frame_size = self._pattern, self._frame_size
        buffer = self._buffer
        while len(buffer) >= frame_size:
            if buffer[: len(pattern)] == pattern:
                yield bytes(buffer[:frame_size])
                del buffer[:frame_size]
                continue
            next_sync = bytes(buffer).find(pattern)
            if next_sync == -1:
                # Keep what a pattern could already be starting in. Dropping the
                # whole buffer instead costs the first frame after a run of
                # corruption as well as the corruption itself.
                del buffer[: len(buffer) - len(pattern)]
                return
            del buffer[:next_sync]


def _as_array(payloads) -> np.ndarray:
    """View *payloads* as bytes, whether they arrived as bytes or as an array."""
    if isinstance(payloads, np.ndarray):
        return payloads
    return np.frombuffer(bytes(payloads), dtype=np.uint8)


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
    """Reassemble VCDU payload chunks into complete fixed-length frames."""
    cutter = _FrameCutter(pattern, frame_size)
    for chunk in payloads:
        yield from cutter.feed(chunk)


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

    Selects the blocks of the wanted virtual channel and strips their payloads;
    the reassembly itself is :class:`_FrameCutter`'s. Where the blocks came from
    — a mapping or a stream — is the caller's choice and makes no difference to
    what comes out.
    """
    cutter = _FrameCutter(pattern, frame_size)
    for chunk in chunks:
        wanted = chunk[(chunk[:, 5] & np.uint8(0x3F)) == vcid]
        payloads = wanted[:, CCSDS_HEADER_LENGTH:CADU_DATA_LENGTH].copy().ravel()
        # Let go of the chunk before anything is yielded: it may be a window onto
        # a mapping or a buffer the source is about to close or refill, and a
        # copy of the payloads is all that is wanted from it anyway.
        chunk = wanted = None
        frames = cutter.feed(payloads)
        # Release the chunk before yielding: a generator suspended while the
        # consumer works on a frame would otherwise pin it for that whole time.
        payloads = None
        yield from frames


@dataclass(frozen=True)
class FrameSpec:
    """How to find one instrument's frames in a CADU stream.

    These four travel together through every reader and every caller, and mean
    nothing apart: a stride without a pattern does not describe anything. Naming
    them together also stops the two integers being handed over in the wrong
    order, which nothing else would catch.
    """

    stride: int
    """Bytes per physical CADU block, including headers and parity."""
    vcid: int
    """Virtual channel carrying the wanted instrument."""
    pattern: bytes
    """Sync pattern each frame opens with."""
    frame_size: int
    """Bytes per reassembled frame."""


def iter_frames(
    source: Path | str | BinaryIO,
    spec: FrameSpec,
    *,
    mapped: bool = False,
    prefetch: int = 0,
    blocks_per_chunk: int = BLOCKS_PER_CHUNK,
) -> Iterator[bytes]:
    """Yield the frames *spec* describes, from a file path or an open stream.

    Reads a chunk of blocks at a time by default, which is what a reader running
    beside the pipeline that consumes it needs: it holds one chunk rather than
    however much of the file a scan has touched.

    *mapped* scans a mapping of the file instead. Frame for frame the same, and
    worth it for a one-shot pass over a file nothing else is waiting on, but the
    pages the scan touches stay resident for as long as the generator lives.

    *prefetch* runs the reading ahead of the consumer on its own thread, up to
    that many chunks. Worth it when the source is slower than the pipeline —
    a pass still arriving over a network mount — and pointless otherwise.
    """
    if mapped:
        if not isinstance(source, (str, Path)):
            msg = "mapped=True needs a path, not an already-open stream"
            raise TypeError(msg)
        chunks = _mapped_block_chunks(Path(source), spec.stride, blocks_per_chunk)
        yield from _frames_from_block_chunks(chunks, spec.vcid, spec.pattern, spec.frame_size)
        return

    if isinstance(source, (str, Path)):
        with open(source, "rb") as opened:
            yield from _iter_frames_from_stream(opened, spec, prefetch, blocks_per_chunk)
        return
    yield from _iter_frames_from_stream(source, spec, prefetch, blocks_per_chunk)


def _iter_frames_from_stream(
    source: BinaryIO, spec: FrameSpec, prefetch: int, blocks_per_chunk: int
) -> Iterator[bytes]:
    """Cut *spec*'s frames out of an open stream, optionally reading ahead."""
    chunks = _read_block_chunks(source, spec.stride, blocks_per_chunk)
    if prefetch > 0:
        chunks = _prefetched(chunks, prefetch)
    yield from _frames_from_block_chunks(chunks, spec.vcid, spec.pattern, spec.frame_size)


def iter_frames_batch(path: Path, stride: int, vcid: int, pattern: bytes, frame_size: int) -> Iterator[bytes]:
    """Yield all frames of *vcid* from a CADU file, scanning a mapping of it.

    Kept for callers written against the older shape; :func:`iter_frames` with
    ``mapped=True`` is the same thing.
    """
    return iter_frames(path, FrameSpec(stride, vcid, pattern, frame_size), mapped=True)


def iter_frames_from(
    source: BinaryIO,
    stride: int,
    vcid: int,
    pattern: bytes,
    frame_size: int,
    *,
    blocks_per_chunk: int = BLOCKS_PER_CHUNK,
    prefetch: int = 0,
) -> Iterator[bytes]:
    """Yield all frames of *vcid* from an open stream. See :func:`iter_frames`."""
    return iter_frames(
        source,
        FrameSpec(stride, vcid, pattern, frame_size),
        prefetch=prefetch,
        blocks_per_chunk=blocks_per_chunk,
    )


def iter_frames_read(
    path: Path,
    stride: int,
    vcid: int,
    pattern: bytes,
    frame_size: int,
    *,
    blocks_per_chunk: int = BLOCKS_PER_CHUNK,
) -> Iterator[bytes]:
    """Yield all frames of *vcid* from a CADU file, reading it. See :func:`iter_frames`."""
    return iter_frames(
        path, FrameSpec(stride, vcid, pattern, frame_size), blocks_per_chunk=blocks_per_chunk
    )


def read_frames_batch(path: Path, stride: int, vcid: int, pattern: bytes, frame_size: int) -> list[bytes]:
    """Read all frames from a CADU file into a list. See :func:`iter_frames`."""
    return list(iter_frames(path, FrameSpec(stride, vcid, pattern, frame_size), mapped=True))
