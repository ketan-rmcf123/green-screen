"""
Automated Unit Tests for GreenScreen Studio.
"""

import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from greenscreen.chroma_key import ChromaKeyer
from greenscreen.segmentation import SegmentationEngine
from greenscreen.backgrounds import BackgroundManager
from greenscreen.pipeline import VirtualBackgroundPipeline, PipelineMode, BackgroundType


class TestChromaKeyer(unittest.TestCase):
    def test_green_mask_generation(self):
        # Create pure green image
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:] = (0, 255, 0)  # Pure green BGR

        # Put a non-green square in the center
        frame[30:70, 30:70] = (255, 0, 0)  # Blue square

        keyer = ChromaKeyer(feather_radius=0, morph_iterations=0)
        mask = keyer.generate_mask(frame)

        # Center should be foreground (approx 1.0)
        self.assertGreater(mask[50, 50], 0.8)
        # Corner should be green / background (approx 0.0)
        self.assertLess(mask[10, 10], 0.2)


class TestBackgroundManager(unittest.TestCase):
    def test_fit_background_aspect_ratio(self):
        bg = np.zeros((200, 400, 3), dtype=np.uint8)
        fitted = BackgroundManager.fit_background(bg, (100, 100))
        self.assertEqual(fitted.shape[:2], (100, 100))

    def test_blur_background(self):
        frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        blurred = BackgroundManager.create_blurred_background(frame, blur_ksize=5)
        self.assertEqual(blurred.shape, frame.shape)

    def test_alpha_compositing(self):
        fg = np.full((10, 10, 3), 100, dtype=np.uint8)
        bg = np.full((10, 10, 3), 200, dtype=np.uint8)
        alpha = np.full((10, 10), 0.5, dtype=np.float32)

        composite = BackgroundManager.composite(fg, bg, alpha)
        # Expected value: 0.5 * 100 + 0.5 * 200 = 150
        self.assertTrue(np.allclose(composite, 150, atol=2))


class TestPipeline(unittest.TestCase):
    def test_pipeline_end_to_end(self):
        pipeline = VirtualBackgroundPipeline(mode=PipelineMode.CHROMA_KEY)
        pipeline.set_background_blur(blur_radius=11)

        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[:] = (0, 255, 0)

        out = pipeline.process_frame(frame)
        self.assertEqual(out.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
