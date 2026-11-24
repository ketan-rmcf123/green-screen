"""
High-Precision Chroma Keying Engine with Soft Feathering.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class ChromaKeyer:
    """
    HSV-based chroma keyer supporting custom color targets, tolerance ranges,
    morphological hole filling, and Gaussian feathering for smooth edge blending.
    """

    DEFAULT_GREEN_LOWER = np.array([35, 80, 50], dtype=np.uint8)
    DEFAULT_GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)

    DEFAULT_BLUE_LOWER = np.array([100, 100, 50], dtype=np.uint8)
    DEFAULT_BLUE_UPPER = np.array([135, 255, 255], dtype=np.uint8)

    def __init__(
        self,
        lower_hsv: Optional[np.ndarray] = None,
        upper_hsv: Optional[np.ndarray] = None,
        feather_radius: int = 5,
        morph_iterations: int = 2
    ) -> None:
        self.lower_hsv = lower_hsv if lower_hsv is not None else self.DEFAULT_GREEN_LOWER.copy()
        self.upper_hsv = upper_hsv if upper_hsv is not None else self.DEFAULT_GREEN_UPPER.copy()
        self.feather_radius = feather_radius
        self.morph_iterations = morph_iterations

    def set_target_color_rgb(self, r: int, g: int, b: int, tolerance: int = 30) -> None:
        """Derive HSV range around a specific RGB color value."""
        color_bgr = np.uint8([[[b, g, r]]])
        hsv_color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
        hue = int(hsv_color[0])

        lower_hue = max(0, hue - tolerance)
        upper_hue = min(179, hue + tolerance)

        self.lower_hsv = np.array([lower_hue, 70, 50], dtype=np.uint8)
        self.upper_hsv = np.array([upper_hue, 255, 255], dtype=np.uint8)

    def generate_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Generate a normalized alpha mask [0.0 .. 1.0] where 1.0 is foreground
        and 0.0 is background.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        key_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Foreground is the inverse of key_mask (green is 0, subject is 255)
        fg_mask = cv2.bitwise_not(key_mask)

        # Morphological noise removal & hole filling
        if self.morph_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)

        # Soft edge feathering
        if self.feather_radius > 0:
            ksize = self.feather_radius * 2 + 1
            fg_mask = cv2.GaussianBlur(fg_mask, (ksize, ksize), 0)

        # Normalize to 0.0 - 1.0 float32
        alpha = fg_mask.astype(np.float32) / 255.0
        return alpha
