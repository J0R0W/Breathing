"""
Object tracking for ROI maintenance across frames.

This module provides tracking functionality to follow the ROI
as the subject moves, maintaining accurate monitoring.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def _create_tracker_kcf():
    """Create KCF tracker with OpenCV version compatibility."""
    try:
        # Try new API (OpenCV 4.5.1+)
        return cv2.legacy.TrackerKCF_create()
    except AttributeError:
        try:
            # Try old API (OpenCV < 4.5.1)
            return cv2.TrackerKCF_create()
        except AttributeError:
            # Try static method API (OpenCV 4.8+)
            return cv2.TrackerKCF.create()


def _create_tracker_csrt():
    """Create CSRT tracker with OpenCV version compatibility."""
    try:
        # Try new API (OpenCV 4.5.1+)
        return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        try:
            # Try old API (OpenCV < 4.5.1)
            return cv2.TrackerCSRT_create()
        except AttributeError:
            # Try static method API (OpenCV 4.8+)
            return cv2.TrackerCSRT.create()


def _create_tracker_mosse():
    """Create MOSSE tracker with OpenCV version compatibility."""
    try:
        # Try legacy API (most common for MOSSE)
        return cv2.legacy.TrackerMOSSE_create()
    except AttributeError:
        try:
            # Try old API
            return cv2.TrackerMOSSE_create()
        except AttributeError:
            # Try static method API
            return cv2.TrackerMOSSE.create()


def _create_tracker_medianflow():
    """Create MedianFlow tracker with OpenCV version compatibility."""
    try:
        # Try legacy API (most common for MedianFlow)
        return cv2.legacy.TrackerMedianFlow_create()
    except AttributeError:
        try:
            # Try old API
            return cv2.TrackerMedianFlow_create()
        except AttributeError:
            # Try static method API
            return cv2.TrackerMedianFlow.create()


class ROITracker:
    """
    ROI tracker using OpenCV's object tracking algorithms.

    Tracks the ROI across frames, handling small movements and maintaining
    monitoring accuracy.

    Supported trackers:
    - KCF: Fast and accurate for most cases
    - CSRT: More accurate but slower
    - MOSSE: Fastest but less accurate
    - MedianFlow: Good for predictable motion

    Example:
        >>> tracker = ROITracker(tracker_type='KCF')
        >>> tracker.init(first_frame, roi)
        >>>
        >>> while True:
        ...     ret, frame = cap.read()
        ...     success, new_roi = tracker.update(frame)
        ...     if success:
        ...         x, y, w, h = new_roi
        ...         cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    """

    TRACKER_TYPES = {
        'KCF': _create_tracker_kcf,
        'CSRT': _create_tracker_csrt,
        'MOSSE': _create_tracker_mosse,
        'MedianFlow': _create_tracker_medianflow,
    }

    def __init__(self, tracker_type: str = 'KCF'):
        """
        Initialize ROI tracker.

        Args:
            tracker_type: Type of tracker ('KCF', 'CSRT', 'MOSSE', 'MedianFlow')

        Raises:
            ValueError: If tracker type is not supported

        Example:
            >>> tracker = ROITracker('KCF')
        """
        if tracker_type not in self.TRACKER_TYPES:
            raise ValueError(
                f"Unknown tracker type: {tracker_type}. "
                f"Available: {list(self.TRACKER_TYPES.keys())}"
            )

        self.tracker_type = tracker_type
        self.tracker = None
        self.initialized = False
        self.failure_count = 0
        self.max_failures = 10

    def init(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> bool:
        """
        Initialize tracker with first frame and ROI.

        Args:
            frame: Initial frame (BGR or grayscale)
            roi: Initial ROI as (x, y, w, h)

        Returns:
            True if initialization succeeded, False otherwise

        Example:
            >>> success = tracker.init(first_frame, (200, 150, 200, 200))
        """
        # Create new tracker
        self.tracker = self.TRACKER_TYPES[self.tracker_type]()

        # Initialize
        success = self.tracker.init(frame, roi)

        if success:
            self.initialized = True
            self.failure_count = 0

        return success

    def update(
        self,
        frame: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Update tracker with new frame.

        Args:
            frame: Current frame

        Returns:
            Tuple of (success, roi)
            - success: True if tracking succeeded
            - roi: Updated ROI as (x, y, w, h) or None if failed

        Example:
            >>> success, roi = tracker.update(frame)
            >>> if success:
            ...     print(f"Tracked to: {roi}")
        """
        if not self.initialized:
            return False, None

        # Update tracker
        success, bbox = self.tracker.update(frame)

        if success:
            self.failure_count = 0
            # Convert to integers
            x, y, w, h = [int(v) for v in bbox]

            # Validate ROI is within frame
            h_frame, w_frame = frame.shape[:2]
            if (x < 0 or y < 0 or x + w > w_frame or y + h > h_frame or
                w <= 0 or h <= 0):
                success = False

        if not success:
            self.failure_count += 1

        # Check if too many failures
        if self.failure_count >= self.max_failures:
            self.initialized = False
            return False, None

        if success:
            return True, (x, y, w, h)

        return False, None

    def reset(self):
        """Reset tracker state."""
        self.tracker = None
        self.initialized = False
        self.failure_count = 0

    def is_initialized(self) -> bool:
        """Check if tracker is initialized."""
        return self.initialized

    def get_failure_count(self) -> int:
        """Get number of consecutive tracking failures."""
        return self.failure_count


class AdaptiveROITracker:
    """
    Adaptive tracker that can re-initialize when tracking fails.

    Combines object tracking with ROI detection for robust operation.

    Example:
        >>> from roi_tracking import PoseBasedROIDetector
        >>> detector = PoseBasedROIDetector()
        >>> tracker = AdaptiveROITracker(detector=detector)
        >>>
        >>> # First frame - detect ROI
        >>> roi = tracker.process(first_frame)
        >>>
        >>> # Subsequent frames - track or re-detect
        >>> while True:
        ...     ret, frame = cap.read()
        ...     roi = tracker.process(frame)
        ...     if roi:
        ...         # Use ROI for motion extraction
        ...         pass
    """

    def __init__(
        self,
        detector=None,
        tracker_type: str = 'KCF',
        redetect_threshold: int = 5
    ):
        """
        Initialize adaptive tracker.

        Args:
            detector: ROI detector instance (e.g., PoseBasedROIDetector)
            tracker_type: Type of object tracker
            redetect_threshold: Number of failures before re-detection

        Example:
            >>> from roi_tracking import PoseBasedROIDetector
            >>> detector = PoseBasedROIDetector()
            >>> tracker = AdaptiveROITracker(detector=detector)
        """
        self.detector = detector
        self.tracker = ROITracker(tracker_type=tracker_type)
        self.redetect_threshold = redetect_threshold
        self.current_roi = None
        self.tracking_mode = False

    def process(
        self,
        frame: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Process frame: track existing ROI or detect new one.

        Args:
            frame: Current frame

        Returns:
            ROI as (x, y, w, h) or None if detection/tracking failed

        Example:
            >>> roi = adaptive_tracker.process(frame)
            >>> if roi:
            ...     x, y, w, h = roi
        """
        # If not tracking, try to detect
        if not self.tracking_mode:
            return self._detect_and_init(frame)

        # Try to track
        success, roi = self.tracker.update(frame)

        if success:
            self.current_roi = roi
            return roi

        # Tracking failed - check if we should re-detect
        if self.tracker.get_failure_count() >= self.redetect_threshold:
            print("Tracking lost, re-detecting ROI...")
            self.tracking_mode = False
            return self._detect_and_init(frame)

        # Return last known ROI
        return self.current_roi

    def _detect_and_init(
        self,
        frame: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect ROI and initialize tracker."""
        if self.detector is None:
            print("No detector available, cannot initialize.")
            return None

        # Detect ROI
        roi = self.detector.detect(frame)

        if roi is None:
            return None

        # Initialize tracker
        success = self.tracker.init(frame, roi)

        if success:
            self.tracking_mode = True
            self.current_roi = roi
            return roi

        return None

    def reset(self):
        """Reset tracker and detector state."""
        self.tracker.reset()
        self.tracking_mode = False
        self.current_roi = None

    def get_current_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current ROI."""
        return self.current_roi


def smooth_roi_sequence(
    rois: list,
    window_size: int = 5
) -> list:
    """
    Smooth ROI positions over time to reduce jitter.

    Args:
        rois: List of ROIs, each as (x, y, w, h) or None
        window_size: Size of smoothing window

    Returns:
        List of smoothed ROIs

    Example:
        >>> # Tracked ROIs over 100 frames
        >>> rois = [tracker.update(frame)[1] for frame in frames]
        >>> smoothed_rois = smooth_roi_sequence(rois, window_size=5)
    """
    if not rois:
        return []

    # Convert None to previous valid value
    filled_rois = []
    last_valid = None

    for roi in rois:
        if roi is not None:
            filled_rois.append(roi)
            last_valid = roi
        else:
            filled_rois.append(last_valid if last_valid else (0, 0, 0, 0))

    # Apply moving average
    smoothed = []
    half_window = window_size // 2

    for i in range(len(filled_rois)):
        start = max(0, i - half_window)
        end = min(len(filled_rois), i + half_window + 1)

        window_rois = filled_rois[start:end]

        # Average each component
        x_avg = int(np.mean([r[0] for r in window_rois]))
        y_avg = int(np.mean([r[1] for r in window_rois]))
        w_avg = int(np.mean([r[2] for r in window_rois]))
        h_avg = int(np.mean([r[3] for r in window_rois]))

        smoothed.append((x_avg, y_avg, w_avg, h_avg))

    return smoothed


def calculate_roi_stability(
    rois: list,
    window_size: int = 30
) -> float:
    """
    Calculate stability score for ROI sequence.

    Higher score means more stable tracking.

    Args:
        rois: List of ROIs
        window_size: Window for stability calculation

    Returns:
        Stability score (0-1, higher = more stable)

    Example:
        >>> stability = calculate_roi_stability(rois)
        >>> print(f"Tracking stability: {stability:.2f}")
    """
    if len(rois) < window_size:
        window_size = len(rois)

    if window_size < 2:
        return 0.0

    # Calculate frame-to-frame displacements
    displacements = []

    for i in range(1, min(len(rois), window_size)):
        if rois[i] is None or rois[i-1] is None:
            continue

        x1, y1, w1, h1 = rois[i-1]
        x2, y2, w2, h2 = rois[i]

        # Center displacement
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2

        displacement = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
        displacements.append(displacement)

    if not displacements:
        return 0.0

    # Stability = inverse of average displacement
    avg_displacement = np.mean(displacements)

    # Normalize (assume frame width ~640, displacement > 50 is unstable)
    stability = 1.0 / (1.0 + avg_displacement / 50.0)

    return np.clip(stability, 0.0, 1.0)
