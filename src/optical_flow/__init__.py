"""
Optical Flow Module for Motion Detection

This module provides optical flow-based motion extraction methods
for respiratory rate detection, including:
- Dense optical flow (Farneback)
- Sparse optical flow (Lucas-Kanade)
- Motion signal extraction from ROI
"""

from .dense_flow import FarnebackFlow, extract_motion_farneback
from .sparse_flow import LucasKanadeFlow, extract_motion_sparse
from .motion_extraction import MotionExtractor, extract_respiratory_motion

__all__ = [
    'FarnebackFlow',
    'extract_motion_farneback',
    'LucasKanadeFlow',
    'extract_motion_sparse',
    'MotionExtractor',
    'extract_respiratory_motion',
]
