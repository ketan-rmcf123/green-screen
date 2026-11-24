"""
GreenScreen: Multi-Engine Virtual Background Studio.
"""

from .chroma_key import ChromaKeyer
from .segmentation import SegmentationEngine, MaskGenerator
from .backgrounds import BackgroundManager
from .pipeline import VirtualBackgroundPipeline

__version__ = "2.0.0"
__all__ = [
    "ChromaKeyer",
    "SegmentationEngine",
    "MaskGenerator",
    "BackgroundManager",
    "VirtualBackgroundPipeline"
]
