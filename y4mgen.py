import math
import sys
from typing import Tuple

import cv2
import numpy as np


def _framerate_fraction(fps: float) -> Tuple[int, int]:
    """Return numerator and denominator for the Y4M F field similar to original tool.

    The original code uses a heuristic around 1001/1000 to represent NTSC rates like
    29.97 as 30000:1001.
    """
    fr_int_1001 = int(round(fps * 1001))
    if fr_int_1001 % 1000 == 0:
        return fr_int_1001, 1001
    return int(round(fps * 1000)), 1000


def _pad_even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = 0 if (h % 2 == 0) else 1
    pad_w = 0 if (w % 2 == 0) else 1
    if pad_h == 0 and pad_w == 0:
        return frame
    return cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)


def bgr_to_i420_bytes(frame: np.ndarray) -> bytes:
    """Convert a BGR frame (uint8) to I420 planar bytes (Y + U + V).

    Ensures even width/height prior to conversion.
    """
    frame_even = _pad_even(frame)
    # OpenCV COLOR_BGR2YUV_I420 produces YUV I420 planar (Y, U, V)
    yuv = cv2.cvtColor(frame_even, cv2.COLOR_BGR2YUV_I420)
    return yuv.tobytes()


def gen_y4m(image: np.ndarray, duration_ms: int, fps: float, output: str) -> None:
    """Generate a Y4M stream (file or stdout) repeating the static image.

    If output == '-' write to stdout.buffer.
    """
    h, w = image.shape[:2]
    # ensure even dims for YUV420
    padded = _pad_even(image)
    ph, pw = padded.shape[:2]

    fr_n, fr_d = _framerate_fraction(fps)
    frame_count = int(math.ceil(fps * (duration_ms / 1000.0)))

    header = f"YUV4MPEG2 W{pw} H{ph} F{fr_n}:{fr_d} Ip A1:1 C420\n"

    if output == "-":
        out_stream = sys.stdout.buffer
    else:
        out_stream = open(output, "wb")

    try:
        out_stream.write(header.encode("ascii"))
        yuv_bytes = bgr_to_i420_bytes(padded)
        frame_marker = b"FRAME\n"
        for i in range(frame_count):
            out_stream.write(frame_marker)
            out_stream.write(yuv_bytes)
    finally:
        if output != "-":
            out_stream.close()