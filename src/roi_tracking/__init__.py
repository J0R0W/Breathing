"""
ROI Detection and Tracking Module

This module provides various methods for detecting and tracking
the region of interest (ROI) for respiratory rate monitoring:
- Manual ROI selection
- Pose-based automatic detection (MediaPipe)
- Motion-based detection
- Object tracking (KCF, CSRT, etc.)
"""

from .roi_detector import ROIDetector, ManualROIDetector, PoseBasedROIDetector
from .tracker import ROITracker

__all__ = [
    'ROIDetector',
    'ManualROIDetector',
    'PoseBasedROIDetector',
    'ROITracker',
]
