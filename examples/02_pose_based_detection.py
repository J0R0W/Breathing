#!/usr/bin/env python3
"""
Example 2: Automatic ROI Detection with MediaPipe Pose

This example demonstrates automatic detection of the chest region
using MediaPipe body pose estimation.

Usage:
    python 02_pose_based_detection.py --webcam
    python 02_pose_based_detection.py --video path/to/video.mp4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import argparse

from respiratory_monitor import RespiratoryRateMonitor


def main():
    parser = argparse.ArgumentParser(description='Pose-based respiratory rate detection')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--method', type=str, default='dense',
                       choices=['dense', 'sparse'],
                       help='Motion extraction method')
    parser.add_argument('--window', type=float, default=30.0,
                       help='Analysis window (seconds)')
    args = parser.parse_args()

    print("=" * 60)
    print("Respiratory Rate Monitor - Pose-Based Detection")
    print("=" * 60)
    print(f"Motion method: {args.method}")
    print(f"Analysis window: {args.window}s")
    print()

    # Create monitor
    monitor = RespiratoryRateMonitor(
        detection_method='pose',
        motion_method=args.method,
        window_size=args.window
    )

    try:
        if args.webcam:
            print("Starting webcam monitoring...")
            print("Instructions:")
            print("  - Position yourself so your upper body is visible")
            print("  - Stay relatively still for best results")
            print("  - Press 'q' to quit")
            print("  - Press 'r' to reset ROI detection")
            print()

            monitor.run_webcam(camera_id=0, display=True)

        elif args.video:
            print(f"Processing video: {args.video}")
            print()

            rate = monitor.process_video(args.video)

            if rate:
                print()
                print("=" * 60)
                print(f"Average Respiratory Rate: {rate:.1f} BPM")
                print("=" * 60)
            else:
                print("Failed to estimate respiratory rate")

        else:
            print("Please specify --video or --webcam")
            return

    except KeyboardInterrupt:
        print("\nStopped by user")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
