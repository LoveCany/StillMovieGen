import argparse
import sys

import cv2
import numpy as np
from framegen import generate_cover_frame
from y4mgen import gen_y4m
from imggen import gen_img


def main():
    parser = argparse.ArgumentParser(description="A Simple Still Movie Generator (Python port)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("-o", "--output", type=str, help="Path to the output video. Use '-' for stdout.")
    parser.add_argument("-d", "--duration", type=int, help="Duration of the output video in milliseconds")
    parser.add_argument("-f", "--fps", type=float, default=30.0, help="Frame rate for the output video. eg. 24, 23.976, 30, 29.97, 60, 59.94")
    parser.add_argument("-c", "--cover", action="store_true", help="Create album-cover style square frame from the input image")
    parser.add_argument("-s", "--scale", type=float, default=0.9, help="Foreground scale (0-1) for cover frame")
    parser.add_argument("-w", "--width", type=int, required=True, help="Width for cover frame generation")
    parser.add_argument("-t", "--height", type=int, required=True, help="Height for cover frame generation")
    parser.add_argument("--format", type=str, default="img", help="Output format (y4m or img)")


    args = parser.parse_args()

    # Read image with alpha if present
    img = cv2.imdecode(np.fromfile(args.input, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image: {args.input}", file=sys.stderr)
        sys.exit(2)

    # If image has alpha channel and we need BGRA -> convert to BGRA array
    if img.ndim == 2:
        # grayscale -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if args.cover:
        frame = generate_cover_frame(img, width=args.width, height=args.height, scale=args.scale)
    else:
        # if image has alpha channel, composite over black background
        if img.shape[2] == 4:
            alpha = img[:, :, 3].astype(np.float32) / 255.0
            rgb = img[:, :, :3].astype(np.float32)
            bg = np.zeros_like(rgb)
            for c in range(3):
                bg[:, :, c] = rgb[:, :, c] * alpha + bg[:, :, c] * (1.0 - alpha)
            frame = bg.astype(np.uint8)
        else:
            frame = img

    # Write image
    if args.format == "img":
        gen_img(frame, args.output)
    elif args.format == "y4m":
        gen_y4m(frame, args.duration, args.fps, args.output)


if __name__ == "__main__":
    main()
