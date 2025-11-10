"""
Motion extraction utilities combining dense and sparse optical flow.

This module provides a unified interface for extracting respiratory motion
signals from video using various optical flow methods.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from .dense_flow import FarnebackFlow, extract_motion_farneback
from .sparse_flow import LucasKanadeFlow, extract_motion_sparse


class MotionExtractor:
    """
    Unified motion extractor supporting multiple methods.

    This class provides a simple interface for extracting motion signals
    from video frames using either dense or sparse optical flow.

    Example:
        >>> extractor = MotionExtractor(method='dense')
        >>> cap = cv2.VideoCapture('video.mp4')
        >>> motion_signal = []
        >>>
        >>> ret, prev_frame = cap.read()
        >>> prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        >>>
        >>> while True:
        ...     ret, frame = cap.read()
        ...     if not ret:
        ...         break
        ...     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ...     motion = extractor.extract(prev_gray, gray, roi=(200,150,200,200))
        ...     motion_signal.append(motion)
        ...     prev_gray = gray
    """

    def __init__(
        self,
        method: str = 'dense',
        **kwargs
    ):
        """
        Initialize motion extractor.

        Args:
            method: Extraction method
                - 'dense': Dense optical flow (Farneback)
                - 'sparse': Sparse optical flow (Lucas-Kanade)
            **kwargs: Additional parameters passed to underlying algorithm

        Example:
            >>> # Dense flow with custom parameters
            >>> extractor = MotionExtractor(method='dense', winsize=20)
            >>>
            >>> # Sparse flow with more features
            >>> extractor = MotionExtractor(method='sparse', max_corners=150)
        """
        self.method = method

        if method == 'dense':
            self.flow_computer = FarnebackFlow(**kwargs)
        elif method == 'sparse':
            self.flow_computer = LucasKanadeFlow(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def extract(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Extract motion signal value.

        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            roi: Optional ROI as (x, y, w, h)

        Returns:
            Motion value (float)

        Example:
            >>> extractor = MotionExtractor()
            >>> motion = extractor.extract(prev_frame, curr_frame, roi)
        """
        if self.method == 'dense':
            return extract_motion_farneback(prev_gray, curr_gray, roi, method='mean')
        elif self.method == 'sparse':
            return self.flow_computer.extract_motion(prev_gray, curr_gray, roi)

    def reset(self):
        """Reset internal state (relevant for sparse flow)."""
        if self.method == 'sparse':
            self.flow_computer.reset()


def extract_respiratory_motion(
    video_path: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
    method: str = 'dense',
    max_frames: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Extract motion signal from entire video file.

    Args:
        video_path: Path to video file
        roi: Optional ROI as (x, y, w, h)
        method: 'dense' or 'sparse'
        max_frames: Maximum number of frames to process (None = all)

    Returns:
        Tuple of (motion_signal, fps)
        - motion_signal: Array of motion values
        - fps: Video frame rate

    Example:
        >>> signal, fps = extract_respiratory_motion(
        ...     'breathing_video.mp4',
        ...     roi=(200, 150, 200, 200),
        ...     method='dense'
        ... )
        >>> print(f"Extracted {len(signal)} frames at {fps} fps")
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize extractor
    extractor = MotionExtractor(method=method)

    motion_signal = []

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract motion
        motion = extractor.extract(prev_gray, gray, roi)
        motion_signal.append(motion)

        prev_gray = gray
        frame_count += 1

        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()

    return np.array(motion_signal), fps


def extract_multiregion_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    method: str = 'dense'
) -> List[float]:
    """
    Extract motion from multiple ROIs simultaneously.

    Useful for tracking chest and abdomen separately.

    Args:
        prev_gray: Previous frame (grayscale)
        curr_gray: Current frame (grayscale)
        rois: List of ROIs, each as (x, y, w, h)
        method: 'dense' or 'sparse'

    Returns:
        List of motion values, one per ROI

    Example:
        >>> chest_roi = (200, 150, 200, 150)
        >>> abdomen_roi = (200, 300, 200, 150)
        >>> motions = extract_multiregion_motion(
        ...     prev_frame, curr_frame,
        ...     [chest_roi, abdomen_roi],
        ...     method='dense'
        ... )
        >>> chest_motion, abdomen_motion = motions
    """
    extractor = MotionExtractor(method=method)

    motions = []
    for roi in rois:
        motion = extractor.extract(prev_gray, curr_gray, roi)
        motions.append(motion)

    return motions


def adaptive_roi_motion_extraction(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    roi: Tuple[int, int, int, int],
    method: str = 'dense',
    quality_threshold: float = 0.3
) -> Tuple[float, float]:
    """
    Extract motion with quality estimation.

    Args:
        prev_gray: Previous frame (grayscale)
        curr_gray: Current frame (grayscale)
        roi: ROI as (x, y, w, h)
        method: 'dense' or 'sparse'
        quality_threshold: Minimum quality score (0-1)

    Returns:
        Tuple of (motion_value, quality_score)

    Example:
        >>> motion, quality = adaptive_roi_motion_extraction(
        ...     prev_frame, curr_frame, roi=(200, 150, 200, 200)
        ... )
        >>> if quality > 0.5:
        ...     print(f"Good quality motion: {motion:.2f}")
    """
    extractor = MotionExtractor(method=method)
    motion = extractor.extract(prev_gray, curr_gray, roi)

    # Estimate quality based on ROI characteristics
    x, y, w, h = roi
    roi_frame = curr_gray[y:y+h, x:x+w]

    # Check texture (good features for tracking)
    corners = cv2.goodFeaturesToTrack(
        roi_frame,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=7
    )
    texture_score = len(corners) / 100.0 if corners is not None else 0.0

    # Check contrast
    contrast_score = np.std(roi_frame) / 128.0  # Normalized to [0,1]

    # Combined quality score
    quality = (texture_score + contrast_score) / 2.0

    return motion, np.clip(quality, 0.0, 1.0)


def compare_motion_methods(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    roi: Tuple[int, int, int, int]
) -> dict:
    """
    Compare different motion extraction methods.

    Useful for debugging and method selection.

    Args:
        prev_gray: Previous frame (grayscale)
        curr_gray: Current frame (grayscale)
        roi: ROI as (x, y, w, h)

    Returns:
        Dictionary with results from different methods

    Example:
        >>> results = compare_motion_methods(prev_frame, curr_frame, roi)
        >>> print(f"Dense (mean): {results['dense_mean']:.2f}")
        >>> print(f"Dense (vertical): {results['dense_vertical']:.2f}")
        >>> print(f"Sparse (pca): {results['sparse_pca']:.2f}")
    """
    results = {}

    # Dense methods
    results['dense_mean'] = extract_motion_farneback(
        prev_gray, curr_gray, roi, method='mean'
    )
    results['dense_median'] = extract_motion_farneback(
        prev_gray, curr_gray, roi, method='median'
    )
    results['dense_vertical'] = extract_motion_farneback(
        prev_gray, curr_gray, roi, method='vertical'
    )

    # Sparse methods
    sparse_tracker = LucasKanadeFlow()
    results['sparse_pca'] = sparse_tracker.extract_motion(
        prev_gray, curr_gray, roi, method='pca'
    )

    sparse_tracker.reset()
    results['sparse_mean'] = sparse_tracker.extract_motion(
        prev_gray, curr_gray, roi, method='mean'
    )

    sparse_tracker.reset()
    results['sparse_vertical'] = sparse_tracker.extract_motion(
        prev_gray, curr_gray, roi, method='vertical'
    )

    return results
