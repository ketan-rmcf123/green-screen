"""
AI Selfie Segmentation and Edge Refinement Engine.
"""

from typing import Optional
import cv2
import numpy as np


class SegmentationEngine:
    """
    Selfie segmentation engine supporting MediaPipe AI segmentation
    with graceful fallback to adaptive thresholding when MediaPipe is unavailable.
    """

    def __init__(self, threshold: float = 0.5, feather_radius: int = 7) -> None:
        self.threshold = threshold
        self.feather_radius = feather_radius
        self._mp_segmenter = None
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp
            if hasattr(mp.solutions, "selfie_segmentation"):
                self._mp_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except Exception:
            self._mp_segmenter = None

    @property
    def has_ai_segmentation(self) -> bool:
        return self._mp_segmenter is not None

    def segment_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Produce a smoothed alpha mask [0.0 .. 1.0] for the person.
        """
        if self._mp_segmenter is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._mp_segmenter.process(frame_rgb)
            mask = results.segmentation_mask
            alpha = (mask > self.threshold).astype(np.float32)
        else:
            # Adaptive fallback: background subtraction / central subject focus
            h, w = frame_bgr.shape[:2]
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            # Center weighted ellipse mask
            center_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(center_mask, (w // 2, h // 2), (w // 3, h // 2), 0, 0, 360, 1.0, -1)
            # Combine with Otsu threshold
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            alpha = (otsu.astype(np.float32) / 255.0) * 0.5 + center_mask * 0.5

        # Soft edge feathering
        if self.feather_radius > 0:
            ksize = self.feather_radius * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

        return np.clip(alpha, 0.0, 1.0)

    def close(self) -> None:
        if self._mp_segmenter is not None:
            try:
                self._mp_segmenter.close()
            except Exception:
                pass


MaskGenerator = SegmentationEngine
