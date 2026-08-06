"""pycadu — satellite-agnostic CCSDS/CADU/VCDU frame handling."""

from pycadu.reader import FrameSpec, iter_frames, iter_vcdu_frames

__all__ = ["FrameSpec", "iter_frames", "iter_vcdu_frames"]
