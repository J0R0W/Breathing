"""
ROI Detection implementations.

This module provides various methods to detect the region of interest
where breathing motion is expected to occur.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class ROIDetector(ABC):
    """
    Abstract base class for ROI detection.

    All ROI detectors should inherit from this class and implement
    the detect() method.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect ROI in frame.

        Args:
            frame: Input frame (BGR or grayscale)

        Returns:
            ROI as (x, y, w, h) or None if detection failed

        Example:
            >>> detector = SomeROIDetector()
            >>> roi = detector.detect(frame)
            >>> if roi:
            ...     x, y, w, h = roi
        """
        pass


class ManualROIDetector(ROIDetector):
    """
    Manual ROI selection using mouse interaction.

    The user draws a bounding box with the mouse to select the ROI.

    Example:
        >>> detector = ManualROIDetector()
        >>> cap = cv2.VideoCapture(0)
        >>> ret, frame = cap.read()
        >>> roi = detector.detect(frame)
        >>> print(f"Selected ROI: {roi}")
    """

    def __init__(self, window_name: str = "Select ROI"):
        """
        Initialize manual ROI detector.

        Args:
            window_name: Name of the selection window
        """
        self.window_name = window_name

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Allow user to manually select ROI.

        Args:
            frame: Input frame

        Returns:
            ROI as (x, y, w, h) or None if canceled

        Example:
            >>> detector = ManualROIDetector()
            >>> roi = detector.detect(first_frame)
        """
        print("Please select the chest/abdomen region for monitoring.")
        print("Press SPACE or ENTER to confirm, ESC to cancel.")

        # Use OpenCV's selectROI
        roi = cv2.selectROI(
            self.window_name,
            frame,
            fromCenter=False,
            showCrosshair=True
        )

        cv2.destroyWindow(self.window_name)

        x, y, w, h = roi

        # Check if selection is valid
        if w > 0 and h > 0:
            return (int(x), int(y), int(w), int(h))

        return None


class PoseBasedROIDetector(ROIDetector):
    """
    Automatic ROI detection using body pose estimation.

    Uses MediaPipe Pose to detect body landmarks and extract
    the chest region automatically.

    Attributes:
        confidence_threshold: Minimum confidence for detection
        use_chest: If True, use chest region; if False, use abdomen

    Example:
        >>> detector = PoseBasedROIDetector()
        >>> roi = detector.detect(frame)
        >>> if roi:
        ...     print(f"Auto-detected ROI: {roi}")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        use_chest: bool = True,
        padding: int = 20
    ):
        """
        Initialize pose-based ROI detector.

        Args:
            confidence_threshold: Minimum detection confidence (0-1)
            use_chest: Use chest region (True) or abdomen (False)
            padding: Padding around detected region in pixels

        Raises:
            ImportError: If mediapipe is not installed
        """
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=confidence_threshold,
                min_tracking_confidence=confidence_threshold
            )
        except ImportError:
            raise ImportError(
                "MediaPipe is required for pose-based ROI detection. "
                "Install it with: pip install mediapipe"
            )

        self.confidence_threshold = confidence_threshold
        self.use_chest = use_chest
        self.padding = padding

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect chest or abdomen ROI using pose landmarks.

        Args:
            frame: Input frame (BGR)

        Returns:
            ROI as (x, y, w, h) or None if detection failed

        Example:
            >>> detector = PoseBasedROIDetector()
            >>> roi = detector.detect(frame)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        if self.use_chest:
            roi = self._extract_chest_roi(landmarks, w, h)
        else:
            roi = self._extract_abdomen_roi(landmarks, w, h)

        return roi

    def _extract_chest_roi(
        self,
        landmarks,
        frame_width: int,
        frame_height: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Extract chest region from landmarks."""
        # Get shoulder and mid-torso landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        # Check visibility
        if (left_shoulder.visibility < self.confidence_threshold or
            right_shoulder.visibility < self.confidence_threshold):
            return None

        # Calculate chest ROI (shoulders to mid-torso)
        x1 = int(min(left_shoulder.x, left_hip.x) * frame_width)
        x2 = int(max(right_shoulder.x, right_hip.x) * frame_width)
        y1 = int(left_shoulder.y * frame_height)
        y2 = int((left_shoulder.y + left_hip.y) / 2 * frame_height)

        # Add padding
        x1 = max(0, x1 - self.padding)
        x2 = min(frame_width, x2 + self.padding)
        y1 = max(0, y1 - self.padding)
        y2 = min(frame_height, y2 + self.padding)

        w = x2 - x1
        h = y2 - y1

        if w > 0 and h > 0:
            return (x1, y1, w, h)

        return None

    def _extract_abdomen_roi(
        self,
        landmarks,
        frame_width: int,
        frame_height: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Extract abdomen region from landmarks."""
        # Get hip landmarks
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        # Check visibility
        if (left_hip.visibility < self.confidence_threshold or
            right_hip.visibility < self.confidence_threshold):
            return None

        # Calculate abdomen ROI (hips region)
        x1 = int(left_hip.x * frame_width)
        x2 = int(right_hip.x * frame_width)
        y1 = int(left_hip.y * frame_height)

        # Extend downward
        h_region = int(0.15 * frame_height)  # 15% of frame height
        y2 = y1 + h_region

        # Add horizontal padding
        width_ext = int(0.3 * (x2 - x1))
        x1 = max(0, x1 - width_ext - self.padding)
        x2 = min(frame_width, x2 + width_ext + self.padding)
        y1 = max(0, y1 - self.padding)
        y2 = min(frame_height, y2 + self.padding)

        w = x2 - x1
        h = y2 - y1

        if w > 0 and h > 0:
            return (x1, y1, w, h)

        return None

    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


class MotionBasedROIDetector(ROIDetector):
    """
    Detect ROI based on periodic motion analysis.

    Analyzes frame differences over time to find regions with
    respiratory frequency motion.

    Example:
        >>> detector = MotionBasedROIDetector()
        >>> # Collect frames
        >>> frames = []
        >>> for i in range(100):
        ...     ret, frame = cap.read()
        ...     frames.append(frame)
        >>> roi = detector.detect_from_sequence(frames)
    """

    def __init__(
        self,
        expected_freq_hz: float = 0.3,
        num_frames: int = 90
    ):
        """
        Initialize motion-based ROI detector.

        Args:
            expected_freq_hz: Expected breathing frequency (default: 0.3 Hz = 18 BPM)
            num_frames: Number of frames to analyze (default: 90 = 3 sec at 30 fps)
        """
        self.expected_freq_hz = expected_freq_hz
        self.num_frames = num_frames
        self.frames_buffer = []

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Accumulate frames for motion analysis.

        This method needs to be called multiple times with consecutive frames.
        Once enough frames are accumulated, call detect_from_sequence().

        Args:
            frame: Input frame

        Returns:
            None (use detect_from_sequence() after accumulating frames)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frames_buffer.append(gray)

        if len(self.frames_buffer) > self.num_frames:
            self.frames_buffer.pop(0)

        return None

    def detect_from_sequence(
        self,
        frames: list = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect ROI from sequence of frames.

        Args:
            frames: List of frames (if None, use internal buffer)

        Returns:
            ROI as (x, y, w, h) or None

        Example:
            >>> detector = MotionBasedROIDetector()
            >>> frames = [read_frame() for _ in range(90)]
            >>> roi = detector.detect_from_sequence(frames)
        """
        if frames is None:
            frames = self.frames_buffer

        if len(frames) < self.num_frames:
            return None

        # Convert to grayscale if needed
        if len(frames[0].shape) == 3:
            frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # Calculate frame differences
        h, w = frames[0].shape
        motion_map = np.zeros((h, w), dtype=np.float32)

        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i-1]).astype(np.float32)
            motion_map += diff

        motion_map /= len(frames)

        # Apply threshold
        threshold = np.mean(motion_map) + np.std(motion_map)
        binary = (motion_map > threshold).astype(np.uint8) * 255

        # Find largest contour
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(w + 2*padding, motion_map.shape[1] - x)
        h = min(h + 2*padding, motion_map.shape[0] - y)

        return (x, y, w, h)


def visualize_roi(
    frame: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Draw ROI on frame for visualization.

    Args:
        frame: Input frame
        roi: ROI as (x, y, w, h)
        color: Rectangle color (BGR)
        thickness: Line thickness
        label: Optional text label

    Returns:
        Frame with ROI drawn

    Example:
        >>> roi = detector.detect(frame)
        >>> vis = visualize_roi(frame, roi, label="Chest")
        >>> cv2.imshow("ROI", vis)
    """
    vis = frame.copy()

    if roi is None:
        return vis

    x, y, w, h = roi

    # Draw rectangle
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

    # Draw label if provided
    if label:
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            vis,
            (x, y - label_size[1] - 10),
            (x + label_size[0] + 10, y),
            color,
            -1
        )
        cv2.putText(
            vis,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return vis
