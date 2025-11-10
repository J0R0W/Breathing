"""
Complete Respiratory Rate Monitoring System.

This module provides an integrated system that combines ROI detection,
tracking, motion extraction, and respiratory rate estimation.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from collections import deque

# Import our modules
from signal_processing import butter_bandpass_filter, fft_respiratory_rate
from optical_flow import MotionExtractor
from roi_tracking import PoseBasedROIDetector, ManualROIDetector, AdaptiveROITracker


class RespiratoryRateMonitor:
    """
    Complete respiratory rate monitoring system.

    This class integrates all components: ROI detection, tracking,
    motion extraction, and rate estimation.

    Example:
        >>> monitor = RespiratoryRateMonitor(
        ...     detection_method='pose',
        ...     motion_method='dense'
        ... )
        >>>
        >>> # From webcam
        >>> monitor.run_webcam()
        >>>
        >>> # From video file
        >>> rate = monitor.process_video('video.mp4')
        >>> print(f"Average respiratory rate: {rate:.1f} BPM")
    """

    def __init__(
        self,
        detection_method: str = 'pose',
        motion_method: str = 'dense',
        window_size: float = 30.0,
        fps: float = 30.0,
        freq_min: float = 0.2,
        freq_max: float = 0.8
    ):
        """
        Initialize respiratory rate monitor.

        Args:
            detection_method: ROI detection ('manual', 'pose')
            motion_method: Motion extraction ('dense', 'sparse')
            window_size: Analysis window size in seconds
            fps: Expected frame rate
            freq_min: Minimum respiratory frequency (Hz)
            freq_max: Maximum respiratory frequency (Hz)

        Example:
            >>> monitor = RespiratoryRateMonitor(
            ...     detection_method='pose',
            ...     window_size=30.0
            ... )
        """
        self.detection_method = detection_method
        self.motion_method = motion_method
        self.window_size = window_size
        self.fps = fps
        self.freq_min = freq_min
        self.freq_max = freq_max

        # Initialize components
        self._init_detector()
        self.motion_extractor = MotionExtractor(method=motion_method)
        self.tracker = AdaptiveROITracker(
            detector=self.detector,
            tracker_type='KCF'
        )

        # Motion signal buffer
        self.buffer_size = int(window_size * fps)
        self.motion_buffer = deque(maxlen=self.buffer_size)

        # Current state
        self.current_roi = None
        self.current_rate = None
        self.frame_count = 0

    def _init_detector(self):
        """Initialize ROI detector based on method."""
        if self.detection_method == 'manual':
            self.detector = ManualROIDetector()
        elif self.detection_method == 'pose':
            try:
                self.detector = PoseBasedROIDetector()
            except ImportError:
                print("MediaPipe not available, falling back to manual detection")
                self.detector = ManualROIDetector()
        else:
            raise ValueError(f"Unknown detection method: {self.detection_method}")

    def process_frame(
        self,
        frame: np.ndarray,
        prev_gray: Optional[np.ndarray] = None
    ) -> Tuple[Optional[float], Optional[Tuple[int, int, int, int]]]:
        """
        Process single frame.

        Args:
            frame: Current frame (BGR)
            prev_gray: Previous frame (grayscale), if available

        Returns:
            Tuple of (respiratory_rate, roi)
            - respiratory_rate: Current estimate in BPM (or None)
            - roi: Current ROI (or None)

        Example:
            >>> rate, roi = monitor.process_frame(frame, prev_gray)
        """
        # Get ROI (detect or track)
        roi = self.tracker.process(frame)

        if roi is None:
            return None, None

        self.current_roi = roi

        # Extract motion if we have previous frame
        if prev_gray is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = self.motion_extractor.extract(prev_gray, gray, roi)
            self.motion_buffer.append(motion)

        # Estimate rate if buffer is full
        respiratory_rate = None
        if len(self.motion_buffer) >= self.buffer_size:
            respiratory_rate = self._estimate_rate()
            self.current_rate = respiratory_rate

        self.frame_count += 1

        return respiratory_rate, roi

    def _estimate_rate(self) -> Optional[float]:
        """Estimate respiratory rate from motion buffer."""
        if len(self.motion_buffer) < self.buffer_size:
            return None

        # Convert buffer to array
        signal = np.array(list(self.motion_buffer))

        # Apply bandpass filter
        filtered = butter_bandpass_filter(
            signal,
            lowcut=self.freq_min,
            highcut=self.freq_max,
            fs=self.fps,
            order=3
        )

        # Estimate rate using FFT
        try:
            rate = fft_respiratory_rate(
                filtered,
                fs=self.fps,
                freq_min=self.freq_min,
                freq_max=self.freq_max
            )
            return rate
        except Exception as e:
            print(f"Rate estimation error: {e}")
            return None

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> Optional[float]:
        """
        Process entire video file.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)

        Returns:
            Average respiratory rate in BPM

        Example:
            >>> rate = monitor.process_video('breathing.mp4')
            >>> print(f"Rate: {rate:.1f} BPM")
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get actual FPS
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.buffer_size = int(self.window_size * self.fps)
        self.motion_buffer = deque(maxlen=self.buffer_size)

        rates = []
        prev_gray = None

        print(f"Processing video at {self.fps} fps...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rate, roi = self.process_frame(frame, prev_gray)

            if rate is not None:
                rates.append(rate)

            prev_gray = gray

            if max_frames and self.frame_count >= max_frames:
                break

            # Show progress
            if self.frame_count % 30 == 0:
                print(f"Processed {self.frame_count} frames...", end='\r')

        cap.release()
        print(f"\nProcessed {self.frame_count} frames total.")

        if rates:
            avg_rate = np.median(rates)  # Use median for robustness
            return avg_rate

        return None

    def run_webcam(
        self,
        camera_id: int = 0,
        display: bool = True
    ):
        """
        Run real-time monitoring from webcam.

        Args:
            camera_id: Camera device ID (usually 0)
            display: Whether to display video with overlays

        Example:
            >>> monitor = RespiratoryRateMonitor()
            >>> monitor.run_webcam()
            >>> # Press 'q' to quit
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        # Get actual FPS
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30.0  # Default if not available

        self.buffer_size = int(self.window_size * self.fps)
        self.motion_buffer = deque(maxlen=self.buffer_size)

        print("Starting webcam monitoring...")
        print("Press 'q' to quit, 'r' to reset ROI")

        prev_gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Process frame
            rate, roi = self.process_frame(frame, prev_gray)

            if display:
                # Draw ROI
                if roi:
                    x, y, w, h = roi
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display rate
                if rate:
                    text = f"RR: {rate:.1f} BPM"
                    cv2.putText(
                        frame, text, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
                    )

                # Display status
                buffer_fill = len(self.motion_buffer) / self.buffer_size
                status = f"Buffer: {buffer_fill*100:.0f}%"
                cv2.putText(
                    frame, status, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                cv2.imshow('Respiratory Rate Monitor', frame)

            prev_gray = gray

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.tracker.reset()
                self.motion_buffer.clear()
                print("ROI reset")

        cap.release()
        if display:
            cv2.destroyAllWindows()

    def get_current_rate(self) -> Optional[float]:
        """Get most recent respiratory rate estimate."""
        return self.current_rate

    def get_current_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current ROI."""
        return self.current_roi

    def reset(self):
        """Reset monitor state."""
        self.tracker.reset()
        self.motion_buffer.clear()
        self.motion_extractor.reset()
        self.current_roi = None
        self.current_rate = None
        self.frame_count = 0


class BatchVideoProcessor:
    """
    Process multiple videos in batch.

    Example:
        >>> processor = BatchVideoProcessor()
        >>> results = processor.process_directory('videos/', '*.mp4')
        >>> for video, rate in results.items():
        ...     print(f"{video}: {rate:.1f} BPM")
    """

    def __init__(self, **monitor_kwargs):
        """
        Initialize batch processor.

        Args:
            **monitor_kwargs: Arguments passed to RespiratoryRateMonitor
        """
        self.monitor_kwargs = monitor_kwargs

    def process_directory(
        self,
        directory: str,
        pattern: str = '*.mp4'
    ) -> dict:
        """
        Process all videos in directory.

        Args:
            directory: Directory path
            pattern: File pattern (e.g., '*.mp4', '*.avi')

        Returns:
            Dictionary mapping video paths to respiratory rates

        Example:
            >>> processor = BatchVideoProcessor()
            >>> results = processor.process_directory('videos/')
        """
        import glob
        import os

        video_files = glob.glob(os.path.join(directory, pattern))

        results = {}

        for video_path in video_files:
            print(f"\nProcessing: {video_path}")

            monitor = RespiratoryRateMonitor(**self.monitor_kwargs)

            try:
                rate = monitor.process_video(video_path)
                results[video_path] = rate
                print(f"Result: {rate:.1f} BPM" if rate else "Failed")
            except Exception as e:
                print(f"Error: {e}")
                results[video_path] = None

        return results

    def export_results(
        self,
        results: dict,
        output_path: str
    ):
        """
        Export results to CSV file.

        Args:
            results: Results dictionary from process_directory()
            output_path: Output CSV file path

        Example:
            >>> results = processor.process_directory('videos/')
            >>> processor.export_results(results, 'results.csv')
        """
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Video', 'Respiratory Rate (BPM)'])

            for video, rate in results.items():
                rate_str = f"{rate:.2f}" if rate else "N/A"
                writer.writerow([video, rate_str])

        print(f"Results exported to: {output_path}")
