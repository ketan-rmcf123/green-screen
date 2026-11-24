"""
Background Management and Alpha Compositing Engine.
"""

from typing import Tuple, Optional
import os
import cv2
import numpy as np


class BackgroundManager:
    """Manages virtual background images, blur synthesis, and alpha compositing."""

    @staticmethod
    def composite(frame_bgr: np.ndarray, bg_bgr: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
        """
        Alpha blend foreground and background:
        result = alpha * foreground + (1.0 - alpha) * background
        """
        # Ensure 3 channels for alpha
        if len(alpha_mask.shape) == 2:
            alpha_3d = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
        else:
            alpha_3d = alpha_mask

        # Ensure same size
        h, w = frame_bgr.shape[:2]
        if bg_bgr.shape[:2] != (h, w):
            bg_bgr = BackgroundManager.fit_background(bg_bgr, (w, h))

        fg = frame_bgr.astype(np.float32)
        bg = bg_bgr.astype(np.float32)

        blended = alpha_3d * fg + (1.0 - alpha_3d) * bg
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def create_blurred_background(frame_bgr: np.ndarray, blur_ksize: int = 35) -> np.ndarray:
        """Create a bokeh/blurred version of the original frame."""
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        return cv2.GaussianBlur(frame_bgr, (ksize, ksize), 0)

    @staticmethod
    def create_solid_background(resolution: Tuple[int, int], bgr_color: Tuple[int, int, int]) -> np.ndarray:
        """Create a solid color background image."""
        w, h = resolution
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = bgr_color
        return bg

    @staticmethod
    def fit_background(bg_bgr: np.ndarray, target_resolution: Tuple[int, int]) -> np.ndarray:
        """
        Resize and center-crop background to match target resolution (width, height)
        without distorting aspect ratio.
        """
        target_w, target_h = target_resolution
        bg_h, bg_w = bg_bgr.shape[:2]

        scale = max(target_w / bg_w, target_h / bg_h)
        new_w = int(bg_w * scale)
        new_h = int(bg_h * scale)

        resized = cv2.resize(bg_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        cropped = resized[start_y : start_y + target_h, start_x : start_x + target_w]
        return cropped

    @staticmethod
    def load_background_image(file_path: str, target_resolution: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load an image from disk and optionally fit to target resolution."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Background image not found: {file_path}")

        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from {file_path}")

        if target_resolution:
            return BackgroundManager.fit_background(img, target_resolution)
        return img
