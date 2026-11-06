pycadu
======

Satellite-agnostic CCSDS/CADU/VCDU frame handling in Python.

Reads raw downlink files containing CADU (Channel Access Data Unit) blocks,
reassembles VCDU payloads, and yields instrument frames delimited by a
caller-supplied sync pattern: a frame runs to the next sync pattern, or to the
instrument's frame size if that comes first.  Works with any CCSDS-compliant downlink:
VIIRS, MERSI, MODIS, etc.

Installation
------------

.. code-block:: bash

    pip install pycadu

Quick start
-----------

.. code-block:: python

    from pycadu import iter_vcdu_frames

    MERSI_SYNC = bytes([0xAA, 0x55] * 6)
    MERSI_FRAME_SIZE = 0x38E4F4
    MERSI_VCID = 3

    for frame in iter_vcdu_frames(
        "FY3D_20240101.dat",
        pattern=MERSI_SYNC,
        frame_size=MERSI_FRAME_SIZE,
        vcid=MERSI_VCID,
        stride=1024,          # or 896 / 1072 depending on ground station
    ):
        process(frame)

API
---

``pycadu``
~~~~~~~~~~

- ``iter_vcdu_frames(path, pattern, frame_size, vcid, stride)`` — top-level convenience.

``pycadu.cadu``
~~~~~~~~~~~~~~~

- ``detect_cadu_stride(data)`` — infer stride (896 / 1024 / 1072) from a sample.
- ``extract_cadu_data(block, stride)`` — strip RS parity / annotation bytes.
- ``extract_vcdu_payload(cadu)`` — strip CCSDS primary header.
- ``extract_vcid(block)`` — read the 6-bit virtual channel identifier.
- ``CADU_STRIDE_1072`` — 1072-byte stride (1024-byte CADU + 48 annotation bytes).

``pycadu.sync``
~~~~~~~~~~~~~~~

- ``find_sync_pattern(data, pattern, start)`` — search at all 8 bit offsets.
- ``_build_shifted(data, shift)`` — build a bit-shifted copy of *data*.

``pycadu.reader``
~~~~~~~~~~~~~~~~~

- ``cadu_blocks(path, stride)`` — stream physical blocks from a file.
- ``vcdu_payloads(blocks, stride, target_vcid)`` — filter by VCID, strip headers.
- ``vcdu_frames(payloads, pattern, frame_size)`` — reassemble variable-length
  chunks into frames of at most ``frame_size`` bytes, each ending at the next sync
  pattern, handling 0-7 bit shifts automatically.
- ``read_frames_batch(path, stride, vcid, pattern, frame_size)`` — memory-mapped
  bulk reader, ~20x faster than the streaming path on large files.

``pycadu.constants``
~~~~~~~~~~~~~~~~~~~~

``CCSDS_SYNC_MARKER``, ``CADU_STRIDE_896``, ``CADU_STRIDE_1024``,
``RS_PARITY_LENGTH``, ``CADU_DATA_LENGTH``, ``CCSDS_HEADER_LENGTH``,
``VCDU_PAYLOAD_LENGTH``.
