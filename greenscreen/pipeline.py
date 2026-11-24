"""
Unified Virtual Background Pipeline.
"""

from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np

from .chroma_key import ChromaKeyer
from .segmentation import SegmentationEngine
from .backgrounds import BackgroundManager


class PipelineMode(str, Enum):
    CHROMA_KEY = "chroma_key"
    AI_SEGMENTATION = "ai_segmentation"


class BackgroundType(str, Enum):
    IMAGE = "image"
    BLUR = "blur"
    SOLID = "solid"


class VirtualBackgroundPipeline:
    """
    High-level virtual background compositor orchestrating segmentation,
    background synthesis, and seamless compositing.
    """

    def __init__(
        self,
        mode: PipelineMode = PipelineMode.CHROMA_KEY,
        chroma_keyer: Optional[ChromaKeyer] = None,
        segmenter: Optional[SegmentationEngine] = None
    ) -> None:
        self.mode = mode
        self.chroma_keyer = chroma_keyer or ChromaKeyer()
        self.segmenter = segmenter or SegmentationEngine()
        self.background_type = BackgroundType.IMAGE
        self.custom_background: Optional[np.ndarray] = None
        self.blur_radius: int = 35
        self.solid_color: Tuple[int, int, int] = (255, 255, 255)  # White BGR

    def set_background_image(self, bg_image_bgr: np.ndarray) -> None:
        self.background_type = BackgroundType.IMAGE
        self.custom_background = bg_image_bgr

    def set_background_blur(self, blur_radius: int = 35) -> None:
        self.background_type = BackgroundType.BLUR
        self.blur_radius = blur_radius

    def set_background_solid(self, bgr_color: Tuple[int, int, int]) -> None:
        self.background_type = BackgroundType.SOLID
        self.solid_color = bgr_color

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Process a single frame and return composite result."""
        h, w = frame_bgr.shape[:2]

        # 1. Generate alpha mask
        if self.mode == PipelineMode.AI_SEGMENTATION:
            alpha = self.segmenter.segment_frame(frame_bgr)
        else:
            alpha = self.chroma_keyer.generate_mask(frame_bgr)

        # 2. Synthesize background
        if self.background_type == BackgroundType.BLUR:
            bg = BackgroundManager.create_blurred_background(frame_bgr, self.blur_radius)
        elif self.background_type == BackgroundType.SOLID:
            bg = BackgroundManager.create_solid_background((w, h), self.solid_color)
        else:
            if self.custom_background is not None:
                bg = BackgroundManager.fit_background(self.custom_background, (w, h))
            else:
                bg = BackgroundManager.create_blurred_background(frame_bgr, self.blur_radius)

        # 3. Alpha composite
        return BackgroundManager.composite(frame_bgr, bg, alpha)
