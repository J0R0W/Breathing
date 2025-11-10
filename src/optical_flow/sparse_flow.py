"""
Sparse optical flow implementation using Lucas-Kanade algorithm.

This module provides sparse optical flow which tracks selected feature
points, offering computational efficiency compared to dense flow.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from sklearn.decomposition import PCA


class LucasKanadeFlow:
    """
    Sparse optical flow using Lucas-Kanade algorithm.

    Tracks good features (corners) across frames, more efficient than
    dense flow but requires good texture in the ROI.

    Attributes:
        max_corners: Maximum number of feature points to track
        quality_level: Quality threshold for corner detection (0-1)
        min_distance: Minimum distance between corners
        lk_params: Parameters for Lucas-Kanade algorithm

    Example:
        >>> tracker = LucasKanadeFlow(max_corners=100)
        >>> prev_frame = cv2.imread('frame1.jpg', 0)
        >>> curr_frame = cv2.imread('frame2.jpg', 0)
        >>> motion = tracker.extract_motion(prev_frame, curr_frame)
    """

    def __init__(
        self,
        max_corners: int = 100,
        quality_level: float = 0.3,
        min_distance: int = 7,
        block_size: int = 7
    ):
        """
        Initialize Lucas-Kanade tracker.

        Args:
            max_corners: Maximum number of corners to track
            quality_level: Quality threshold (0-1)
            min_distance: Minimum distance between corners in pixels
            block_size: Size of averaging block for corner detection
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.prev_points = None

    def detect_features(
        self,
        frame_gray: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Detect good features to track.

        Args:
            frame_gray: Grayscale frame
            roi: Optional ROI as (x, y, w, h)
            mask: Optional mask specifying where to detect features

        Returns:
            Array of feature points shape (N, 1, 2)

        Example:
            >>> tracker = LucasKanadeFlow()
            >>> features = tracker.detect_features(frame, roi=(100, 100, 200, 200))
        """
        if roi is not None:
            x, y, w, h = roi
            roi_frame = frame_gray[y:y+h, x:x+w]

            # Detect features in ROI
            features = cv2.goodFeaturesToTrack(
                roi_frame,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                mask=mask
            )

            if features is not None:
                # Convert coordinates back to full frame
                features[:, 0, 0] += x
                features[:, 0, 1] += y

        else:
            features = cv2.goodFeaturesToTrack(
                frame_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                mask=mask
            )

        return features

    def track_features(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track features from previous frame to current frame.

        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            prev_points: Feature points in previous frame

        Returns:
            Tuple of (curr_points, status, error)
            - curr_points: Tracked points in current frame
            - status: Array indicating successful tracking (1) or failure (0)
            - error: Tracking error for each point

        Example:
            >>> curr_pts, status, err = tracker.track_features(frame1, frame2, prev_pts)
        """
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_points,
            None,
            **self.lk_params
        )

        return curr_points, status, error

    def extract_motion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        method: str = 'pca'
    ) -> float:
        """
        Extract motion signal using sparse optical flow.

        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            roi: Optional ROI as (x, y, w, h)
            method: Motion extraction method
                - 'pca': Use first principal component
                - 'mean': Mean motion magnitude
                - 'median': Median motion magnitude
                - 'vertical': Mean vertical motion

        Returns:
            Motion value (float)

        Example:
            >>> tracker = LucasKanadeFlow()
            >>> motion = tracker.extract_motion(frame1, frame2, roi=(100,100,200,200))
        """
        # Detect or reuse features
        if self.prev_points is None:
            self.prev_points = self.detect_features(prev_gray, roi)

        if self.prev_points is None or len(self.prev_points) == 0:
            return 0.0

        # Track features
        curr_points, status, error = self.track_features(
            prev_gray, curr_gray, self.prev_points
        )

        # Filter out unsuccessful tracks
        good_prev = self.prev_points[status == 1]
        good_curr = curr_points[status == 1]

        if len(good_prev) == 0:
            # Reset and try again
            self.prev_points = None
            return 0.0

        # Calculate motion vectors
        motion_vectors = good_curr - good_prev

        # Extract motion value based on method
        if method == 'pca':
            # Use PCA to find dominant motion direction
            if len(motion_vectors) < 2:
                motion_value = np.mean(np.linalg.norm(motion_vectors, axis=1))
            else:
                pca = PCA(n_components=1)
                transformed = pca.fit_transform(motion_vectors)
                motion_value = np.mean(np.abs(transformed))

        elif method == 'mean':
            magnitudes = np.linalg.norm(motion_vectors, axis=1)
            motion_value = np.mean(magnitudes)

        elif method == 'median':
            magnitudes = np.linalg.norm(motion_vectors, axis=1)
            motion_value = np.median(magnitudes)

        elif method == 'vertical':
            # Focus on vertical motion
            motion_value = np.mean(np.abs(motion_vectors[:, 1]))

        else:
            raise ValueError(f"Unknown method: {method}")

        # Update for next frame
        self.prev_points = good_curr.reshape(-1, 1, 2)

        return float(motion_value)

    def reset(self):
        """Reset tracked features."""
        self.prev_points = None


def extract_motion_sparse(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    max_corners: int = 100
) -> float:
    """
    Convenience function for sparse motion extraction.

    Args:
        prev_gray: Previous frame (grayscale)
        curr_gray: Current frame (grayscale)
        roi: Optional ROI as (x, y, w, h)
        max_corners: Maximum number of features to track

    Returns:
        Motion value

    Example:
        >>> motion = extract_motion_sparse(frame1, frame2, roi=(100, 100, 200, 200))
    """
    tracker = LucasKanadeFlow(max_corners=max_corners)
    motion = tracker.extract_motion(prev_gray, curr_gray, roi)
    return motion


def visualize_sparse_flow(
    frame: np.ndarray,
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    status: np.ndarray
) -> np.ndarray:
    """
    Visualize tracked features and their motion.

    Args:
        frame: Current frame (BGR)
        prev_points: Previous feature locations
        curr_points: Current feature locations
        status: Tracking status array

    Returns:
        Visualization image with drawn features and motion trails

    Example:
        >>> vis = visualize_sparse_flow(frame, prev_pts, curr_pts, status)
        >>> cv2.imshow('Tracking', vis)
    """
    vis = frame.copy()

    # Filter good points
    good_prev = prev_points[status == 1]
    good_curr = curr_points[status == 1]

    # Draw motion trails
    for prev_pt, curr_pt in zip(good_prev, good_curr):
        prev_x, prev_y = prev_pt.ravel().astype(int)
        curr_x, curr_y = curr_pt.ravel().astype(int)

        # Draw line showing motion
        cv2.line(vis, (prev_x, prev_y), (curr_x, curr_y), (0, 255, 0), 1)

        # Draw current point
        cv2.circle(vis, (curr_x, curr_y), 3, (0, 0, 255), -1)

    return vis


def calculate_motion_statistics(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    status: np.ndarray
) -> dict:
    """
    Calculate statistics about tracked motion.

    Args:
        prev_points: Previous feature locations
        curr_points: Current feature locations
        status: Tracking status

    Returns:
        Dictionary with motion statistics:
        - 'num_tracked': Number of successfully tracked points
        - 'mean_magnitude': Mean motion magnitude
        - 'std_magnitude': Standard deviation of magnitudes
        - 'mean_horizontal': Mean horizontal motion
        - 'mean_vertical': Mean vertical motion
        - 'coherence': Motion coherence (0-1)

    Example:
        >>> stats = calculate_motion_statistics(prev_pts, curr_pts, status)
        >>> print(f"Tracked {stats['num_tracked']} points")
    """
    good_prev = prev_points[status == 1]
    good_curr = curr_points[status == 1]

    if len(good_prev) == 0:
        return {
            'num_tracked': 0,
            'mean_magnitude': 0.0,
            'std_magnitude': 0.0,
            'mean_horizontal': 0.0,
            'mean_vertical': 0.0,
            'coherence': 0.0
        }

    # Motion vectors
    motion_vectors = good_curr - good_prev

    # Magnitudes
    magnitudes = np.linalg.norm(motion_vectors, axis=1)

    # Components
    horizontal = motion_vectors[:, 0]
    vertical = motion_vectors[:, 1]

    # Coherence (alignment of motion vectors)
    mean_vector = np.mean(motion_vectors, axis=0)
    mean_magnitude = np.linalg.norm(mean_vector)
    coherence = mean_magnitude / (np.mean(magnitudes) + 1e-10)

    return {
        'num_tracked': len(good_prev),
        'mean_magnitude': float(np.mean(magnitudes)),
        'std_magnitude': float(np.std(magnitudes)),
        'mean_horizontal': float(np.mean(horizontal)),
        'mean_vertical': float(np.mean(vertical)),
        'coherence': float(np.clip(coherence, 0, 1))
    }


def adaptive_feature_redetection(
    tracker: LucasKanadeFlow,
    frame_gray: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    min_features: int = 20
) -> bool:
    """
    Check if features need to be re-detected and do so if necessary.

    Args:
        tracker: LucasKanadeFlow instance
        frame_gray: Current frame (grayscale)
        roi: Optional ROI
        min_features: Minimum number of features required

    Returns:
        True if features were re-detected, False otherwise

    Example:
        >>> tracker = LucasKanadeFlow()
        >>> # ... track for several frames ...
        >>> if adaptive_feature_redetection(tracker, frame, roi):
        ...     print("Features re-detected")
    """
    if tracker.prev_points is None or len(tracker.prev_points) < min_features:
        tracker.prev_points = tracker.detect_features(frame_gray, roi)
        return True

    return False
