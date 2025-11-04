import cv2
import numpy as np


def _pad_even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = 0 if (h % 2 == 0) else 1
    pad_w = 0 if (w % 2 == 0) else 1
    if pad_h == 0 and pad_w == 0:
        return frame
    return cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)


def generate_cover_frame(img: np.ndarray, width: int, height: int, scale: float = 0.9) -> np.ndarray:
    """Create a cover-style frame of exactly (width, height).

    Behavior:
    - create a blurred background by resizing the full image to COVER the target
      (preserve aspect, center-crop), then blur
    - scale the whole input image to fit within (scale*width, scale*height) preserving aspect
      and paste it centered on the blurred background
    - if input has alpha, use it to blend the foreground; final output is BGR (3 channels)
    """
    if img is None:
        raise ValueError("Input image is None")

    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid image or target dimensions")

    has_alpha = img.shape[2] == 4
    src_rgb = img[:, :, :3]

    # Background: scale to COVER target so no empty areas, then center-crop to (width,height)
    scale_bg = max(width / iw, height / ih)
    bg_w = max(1, int(round(iw * scale_bg)))
    bg_h = max(1, int(round(ih * scale_bg)))
    bg_resized = cv2.resize(src_rgb, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    ox = (bg_w - width) // 2 if bg_w > width else 0
    oy = (bg_h - height) // 2 if bg_h > height else 0
    bg = bg_resized[oy : oy + height, ox : ox + width].copy()

    sigma = max(1.0, max(width, height) / 16.0)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Foreground: scale whole input to fit within scale * target while preserving aspect
    tgt_w = max(1, int(round(width * scale)))
    tgt_h = max(1, int(round(height * scale)))
    scale_fg = min(tgt_w / iw, tgt_h / ih)
    fg_w = max(1, int(round(iw * scale_fg)))
    fg_h = max(1, int(round(ih * scale_fg)))

    interp = cv2.INTER_AREA if scale_fg < 1.0 else cv2.INTER_LINEAR
    fg_rgb = cv2.resize(src_rgb, (fg_w, fg_h), interpolation=interp)

    # If alpha present, resize alpha channel as well
    if has_alpha:
        alpha_ch = img[:, :, 3]
        fg_alpha = cv2.resize(alpha_ch, (fg_w, fg_h), interpolation=cv2.INTER_LINEAR)
        alpha = (fg_alpha.astype(np.float32) / 255.0)[..., None]
    else:
        alpha = None

    bx = (width - fg_w) // 2
    by = (height - fg_h) // 2

    # Composite
    roi = bg[by : by + fg_h, bx : bx + fg_w].astype(np.float32)
    fg_f = fg_rgb.astype(np.float32)
    if alpha is not None:
        # blend using alpha
        roi = roi * (1.0 - alpha) + fg_f * alpha
        bg[by : by + fg_h, bx : bx + fg_w] = np.clip(roi, 0, 255).astype(np.uint8)
    else:
        bg[by : by + fg_h, bx : bx + fg_w] = fg_rgb

    return bg


def bgr_to_i420_bytes(frame: np.ndarray) -> bytes:
    """Convert a BGR frame (uint8) to I420 planar bytes (Y + U + V).

    Ensures even width/height prior to conversion.
    """
    frame_even = _pad_even(frame)
    # OpenCV COLOR_BGR2YUV_I420 produces YUV I420 planar (Y, U, V)
    yuv = cv2.cvtColor(frame_even, cv2.COLOR_BGR2YUV_I420)
    return yuv.tobytes()