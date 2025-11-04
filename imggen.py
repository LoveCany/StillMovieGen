import os
import cv2
import numpy as np
import sys


def gen_img(frame: np.ndarray, output: str) -> None:
    """Generate an image file from the given frame.

    If output == '-' write to stdout.buffer.
    """
    if output == "-":
        # Write to stdout as PNG
        success, enc = cv2.imencode(".png", frame)
        if not success:
            raise RuntimeError("Failed to encode image")
        sys.stdout.buffer.write(enc.tobytes())
    else:
        # Write to file
        success, ext = os.path.splitext(output)
        if ext == "":
            ext = ".png"
            output = output + ext
        success, enc = cv2.imencode(ext, frame)
        if not success:
            raise RuntimeError("Failed to encode image")
        with open(output, "wb") as f:
            f.write(enc.tobytes())