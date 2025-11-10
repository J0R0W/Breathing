"""
Dense optical flow implementation using Farneback algorithm.

This module provides dense optical flow computation which tracks motion
of all pixels in the region of interest.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class FarnebackFlow:
    """
    Dense optical flow using Farneback algorithm.

    This class provides a wrapper around OpenCV's Farneback optical flow
    with optimized parameters for respiratory motion detection.

    Attributes:
        pyr_scale: Image pyramid scale factor
        levels: Number of pyramid layers
        winsize: Averaging window size
        iterations: Number of iterations at each pyramid level
        poly_n: Size of pixel neighborhood
        poly_sigma: Standard deviation of Gaussian for polynomial expansion

    Example:
        >>> flow_computer = FarnebackFlow()
        >>> prev_frame = cv2.imread('frame1.jpg', 0)
        >>> curr_frame = cv2.imread('frame2.jpg', 0)
        >>> flow = flow_computer.compute(prev_frame, curr_frame)
        >>> magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2
    ):
        """
        Initialize Farneback optical flow computer.

        Args:
            pyr_scale: Image scale at each pyramid level (< 1.0)
            levels: Number of pyramid layers
            winsize: Averaging window size (larger = smoother)
            iterations: Iterations at each pyramid level
            poly_n: Neighborhood pixel size for polynomial expansion
            poly_sigma: Gaussian sigma for smoothing derivatives
        """
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def compute(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Compute dense optical flow between two frames.

        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            roi: Optional ROI as (x, y, w, h). If provided, only compute
                 flow within this region.

        Returns:
            Flow field of shape (H, W, 2) where:
            - flow[..., 0] = horizontal displacement
            - flow[..., 1] = vertical displacement

        Example:
            >>> flow_computer = FarnebackFlow()
            >>> flow = flow_computer.compute(frame1, frame2, roi=(100, 100, 200, 200))
        """
        # Extract ROI if specified
        if roi is not None:
            x, y, w, h = roi
            prev_roi = prev_gray[y:y+h, x:x+w]
            curr_roi = curr_gray[y:y+h, x:x+w]
        else:
            prev_roi = prev_gray
            curr_roi = curr_gray

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_roi,
            curr_roi,
            None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0
        )

        return flow

    def compute_magnitude(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute flow magnitude and angle.

        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            roi: Optional ROI as (x, y, w, h)

        Returns:
            Tuple of (magnitude, angle) arrays

        Example:
            >>> flow_computer = FarnebackFlow()
            >>> mag, ang = flow_computer.compute_magnitude(frame1, frame2)
        """
        flow = self.compute(prev_gray, curr_gray, roi)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return magnitude, angle


def extract_motion_farneback(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    method: str = 'mean'
) -> float:
    """
    Extract single motion value from optical flow.

    This is a convenience function that computes optical flow
    and reduces it to a single scalar value representing motion intensity.

    Args:
        prev_gray: Previous frame (grayscale)
        curr_gray: Current frame (grayscale)
        roi: Optional ROI as (x, y, w, h)
        method: How to aggregate motion
            - 'mean': Mean magnitude
            - 'median': Median magnitude
            - 'max': Maximum magnitude
            - 'vertical': Mean vertical motion (breathing)
            - 'horizontal': Mean horizontal motion

    Returns:
        Single float representing motion intensity

    Example:
        >>> motion_value = extract_motion_farneback(
        ...     frame1, frame2,
        ...     roi=(200, 150, 200, 200),
        ...     method='mean'
        ... )
    """
    flow_computer = FarnebackFlow()
    flow = flow_computer.compute(prev_gray, curr_gray, roi)

    if method == 'mean':
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return np.mean(magnitude)

    elif method == 'median':
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return np.median(magnitude)

    elif method == 'max':
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return np.max(magnitude)

    elif method == 'vertical':
        # Focus on vertical motion (chest rise/fall)
        return np.mean(np.abs(flow[..., 1]))

    elif method == 'horizontal':
        # Horizontal motion
        return np.mean(np.abs(flow[..., 0]))

    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_flow(
    flow: np.ndarray,
    method: str = 'hsv'
) -> np.ndarray:
    """
    Visualize optical flow field.

    Args:
        flow: Flow field of shape (H, W, 2)
        method: Visualization method
            - 'hsv': Color-coded by direction and magnitude
            - 'arrows': Draw arrows showing motion
            - 'magnitude': Grayscale magnitude image

    Returns:
        Visualization image (BGR format for OpenCV display)

    Example:
        >>> flow = flow_computer.compute(frame1, frame2)
        >>> vis = visualize_flow(flow, method='hsv')
        >>> cv2.imshow('Flow', vis)
    """
    if method == 'hsv':
        # Convert flow to HSV color encoding
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Create HSV image
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
        hsv[..., 1] = 255  # Full saturation
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to BGR for display
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    elif method == 'magnitude':
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mag_uint8 = mag_normalized.astype(np.uint8)
        # Convert to BGR (3 channels)
        return cv2.cvtColor(mag_uint8, cv2.COLOR_GRAY2BGR)

    elif method == 'arrows':
        # Draw arrows on black background
        h, w = flow.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)

        # Subsample for visualization (every 10th pixel)
        step = 10
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                # Only draw if motion is significant
                if fx**2 + fy**2 > 1:
                    cv2.arrowedLine(
                        vis,
                        (x, y),
                        (int(x + fx*2), int(y + fy*2)),
                        (0, 255, 0),
                        1,
                        tipLength=0.3
                    )
        return vis

    else:
        raise ValueError(f"Unknown visualization method: {method}")


def dominant_motion_direction(flow: np.ndarray) -> str:
    """
    Determine dominant motion direction from flow field.

    Args:
        flow: Flow field of shape (H, W, 2)

    Returns:
        Dominant direction: 'up', 'down', 'left', 'right', or 'mixed'

    Example:
        >>> flow = flow_computer.compute(frame1, frame2)
        >>> direction = dominant_motion_direction(flow)
        >>> print(f"Motion is predominantly {direction}")
    """
    # Average flow vectors
    mean_fx = np.mean(flow[..., 0])
    mean_fy = np.mean(flow[..., 1])

    # Calculate magnitude
    magnitude = np.sqrt(mean_fx**2 + mean_fy**2)

    # Threshold for "significant" motion
    if magnitude < 0.5:
        return 'minimal'

    # Determine dominant direction
    angle = np.arctan2(mean_fy, mean_fx) * 180 / np.pi

    if -45 <= angle < 45:
        return 'right'
    elif 45 <= angle < 135:
        return 'down'
    elif angle >= 135 or angle < -135:
        return 'left'
    else:  # -135 <= angle < -45
        return 'up'


def motion_coherence(flow: np.ndarray) -> float:
    """
    Calculate motion coherence (how aligned flow vectors are).

    High coherence indicates organized motion (e.g., breathing).
    Low coherence indicates chaotic/noisy motion.

    Args:
        flow: Flow field of shape (H, W, 2)

    Returns:
        Coherence score (0-1, higher = more coherent)

    Example:
        >>> flow = flow_computer.compute(frame1, frame2)
        >>> coherence = motion_coherence(flow)
        >>> print(f"Motion coherence: {coherence:.2f}")
    """
    # Get flow vectors
    flow_flat = flow.reshape(-1, 2)

    # Calculate magnitudes
    magnitudes = np.linalg.norm(flow_flat, axis=1)

    # Average vector
    mean_vector = np.mean(flow_flat, axis=0)
    mean_magnitude = np.linalg.norm(mean_vector)

    # Coherence = mean magnitude / mean of magnitudes
    # (perfect coherence = 1, random motion = 0)
    if np.mean(magnitudes) < 1e-10:
        return 0.0

    coherence = mean_magnitude / np.mean(magnitudes)
    return np.clip(coherence, 0.0, 1.0)
