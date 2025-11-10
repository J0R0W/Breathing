#!/usr/bin/env python3
"""
Example 3: Complete Respiratory Rate Monitoring System

This example demonstrates the full-featured monitoring system with:
- Automatic or manual ROI detection
- Adaptive tracking
- Real-time visualization
- Rate history and statistics

Usage:
    python 03_complete_monitor.py --webcam
    python 03_complete_monitor.py --video path/to/video.mp4 --detect manual
    python 03_complete_monitor.py --video path/to/video.mp4 --detect pose
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import numpy as np
import argparse
from collections import deque
import time

from respiratory_monitor import RespiratoryRateMonitor


class VisualMonitor(RespiratoryRateMonitor):
    """
    Extended monitor with enhanced visualization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Rate history
        self.rate_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        self.start_time = time.time()

    def run_webcam_enhanced(self, camera_id=0):
        """Run with enhanced visualization."""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        # Get FPS
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30.0

        self.buffer_size = int(self.window_size * self.fps)
        self.motion_buffer = deque(maxlen=self.buffer_size)

        print("\n" + "=" * 70)
        print("RESPIRATORY RATE MONITOR - ENHANCED")
        print("=" * 70)
        print("\nControls:")
        print("  Q     - Quit")
        print("  R     - Reset ROI detection")
        print("  SPACE - Pause/Resume")
        print("\nWaiting for ROI detection...")

        prev_gray = None
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Process frame
                rate, roi = self.process_frame(frame, prev_gray)

                # Update history
                if rate:
                    current_time = time.time() - self.start_time
                    self.rate_history.append(rate)
                    self.time_history.append(current_time)

                prev_gray = gray

            # Visualization
            vis = self._create_visualization(frame, rate, roi)

            cv2.imshow('Respiratory Rate Monitor', vis)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset()
                self.rate_history.clear()
                self.time_history.clear()
                self.start_time = time.time()
                print("Monitor reset")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")

        cap.release()
        cv2.destroyAllWindows()

        # Print statistics
        self._print_statistics()

    def _create_visualization(self, frame, rate, roi):
        """Create enhanced visualization."""
        h, w = frame.shape[:2]

        # Create larger canvas
        canvas = np.zeros((h + 200, w, 3), dtype=np.uint8)
        canvas[:h, :] = frame

        # Draw ROI
        if roi:
            x, y, w_roi, h_roi = roi
            cv2.rectangle(canvas, (x, y), (x+w_roi, y+h_roi), (0, 255, 0), 2)
            cv2.putText(canvas, "ROI", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Status panel
        panel_y = h + 10

        # Current rate (large)
        if rate:
            rate_text = f"{rate:.1f}"
            cv2.putText(canvas, rate_text, (20, panel_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            cv2.putText(canvas, "BPM", (150, panel_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(canvas, "Detecting...", (20, panel_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

        # Statistics
        if len(self.rate_history) > 0:
            mean_rate = np.mean(list(self.rate_history))
            std_rate = np.std(list(self.rate_history))

            stats_y = panel_y + 100
            cv2.putText(canvas, f"Mean: {mean_rate:.1f} BPM", (20, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, f"Std: {std_rate:.1f}", (20, stats_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Buffer status
        buffer_fill = len(self.motion_buffer) / self.buffer_size
        bar_width = 200
        bar_height = 20
        bar_x = w - bar_width - 20
        bar_y = panel_y + 20

        # Background
        cv2.rectangle(canvas, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)

        # Fill
        fill_width = int(bar_width * buffer_fill)
        color = (0, 255, 0) if buffer_fill >= 0.95 else (0, 255, 255)
        cv2.rectangle(canvas, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height),
                     color, -1)

        # Text
        cv2.putText(canvas, f"Buffer: {buffer_fill*100:.0f}%",
                   (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Rate history plot (mini)
        if len(self.rate_history) > 1:
            plot_x = w - 250
            plot_y = panel_y + 80
            plot_w = 220
            plot_h = 100

            self._draw_rate_plot(canvas, plot_x, plot_y, plot_w, plot_h)

        return canvas

    def _draw_rate_plot(self, canvas, x, y, w, h):
        """Draw mini rate history plot."""
        if len(self.rate_history) < 2:
            return

        rates = list(self.rate_history)

        # Normalize
        min_rate = max(min(rates) - 2, 0)
        max_rate = max(rates) + 2
        rate_range = max_rate - min_rate

        if rate_range == 0:
            return

        # Background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 50), -1)

        # Plot line
        points = []
        for i, rate in enumerate(rates):
            px = x + int((i / len(rates)) * w)
            py = y + h - int(((rate - min_rate) / rate_range) * h)
            points.append((px, py))

        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i+1], (0, 255, 255), 1)

        # Labels
        cv2.putText(canvas, f"{max_rate:.0f}", (x + w + 5, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(canvas, f"{min_rate:.0f}", (x + w + 5, y + h),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    def _print_statistics(self):
        """Print final statistics."""
        if not self.rate_history:
            return

        rates = list(self.rate_history)

        print("\n" + "=" * 70)
        print("SESSION STATISTICS")
        print("=" * 70)
        print(f"Number of estimates: {len(rates)}")
        print(f"Mean rate:           {np.mean(rates):.1f} BPM")
        print(f"Median rate:         {np.median(rates):.1f} BPM")
        print(f"Std deviation:       {np.std(rates):.1f} BPM")
        print(f"Min rate:            {np.min(rates):.1f} BPM")
        print(f"Max rate:            {np.max(rates):.1f} BPM")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Complete respiratory rate monitoring system'
    )
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--detect', type=str, default='pose',
                       choices=['manual', 'pose'],
                       help='ROI detection method')
    parser.add_argument('--motion', type=str, default='dense',
                       choices=['dense', 'sparse'],
                       help='Motion extraction method')
    parser.add_argument('--window', type=float, default=30.0,
                       help='Analysis window (seconds)')
    args = parser.parse_args()

    # Create monitor
    monitor = VisualMonitor(
        detection_method=args.detect,
        motion_method=args.motion,
        window_size=args.window
    )

    try:
        if args.webcam:
            monitor.run_webcam_enhanced(camera_id=0)

        elif args.video:
            print(f"Processing video: {args.video}")
            rate = monitor.process_video(args.video)

            if rate:
                print(f"\nAverage Respiratory Rate: {rate:.1f} BPM")
            else:
                print("Failed to estimate respiratory rate")

        else:
            print("Please specify --video or --webcam")

    except KeyboardInterrupt:
        print("\nStopped by user")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
