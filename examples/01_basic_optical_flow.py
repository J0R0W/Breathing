#!/usr/bin/env python3
"""
Example 1: Basic Optical Flow for Respiratory Rate Detection

This example demonstrates the simplest approach to respiratory rate
estimation using dense optical flow and FFT analysis.

Usage:
    python 01_basic_optical_flow.py --video path/to/video.mp4
    python 01_basic_optical_flow.py --webcam  # Use webcam
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import numpy as np
import argparse
from collections import deque

from signal_processing import butter_bandpass_filter, fft_respiratory_rate
from optical_flow import extract_motion_farneback


def main():
    parser = argparse.ArgumentParser(description='Basic optical flow respiratory rate detection')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--fps', type=float, default=30.0, help='Video frame rate')
    parser.add_argument('--window', type=float, default=30.0, help='Analysis window (seconds)')
    args = parser.parse_args()

    # Open video source
    if args.webcam:
        cap = cv2.VideoCapture(0)
        fps = 30.0  # Assume 30 fps for webcam
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = args.fps
    else:
        print("Please specify --video or --webcam")
        return

    print(f"Video FPS: {fps}")

    # Motion buffer
    buffer_size = int(args.window * fps)
    motion_buffer = deque(maxlen=buffer_size)

    # Manual ROI selection
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return

    print("Select ROI (chest/abdomen region):")
    print("  - Draw rectangle with mouse")
    print("  - Press SPACE or ENTER to confirm")
    print("  - Press ESC to cancel")

    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("No ROI selected")
        return

    print(f"ROI selected: x={x}, y={y}, w={w}, h={h}")

    # Previous frame
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Main loop
    print("\nProcessing video...")
    print("Press 'q' to quit")

    frame_count = 0
    current_rate = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract motion
        motion = extract_motion_farneback(prev_gray, gray, roi, method='mean')
        motion_buffer.append(motion)

        # Estimate rate when buffer is full
        if len(motion_buffer) >= buffer_size:
            # Convert to array
            signal = np.array(list(motion_buffer))

            # Bandpass filter
            filtered = butter_bandpass_filter(signal, 0.2, 0.8, fps, order=3)

            # FFT-based rate estimation
            try:
                current_rate = fft_respiratory_rate(filtered, fps, 0.2, 0.8)
            except Exception as e:
                print(f"Error estimating rate: {e}")

        # Visualization
        vis = frame.copy()

        # Draw ROI
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display rate
        if current_rate:
            text = f"RR: {current_rate:.1f} BPM"
            cv2.putText(vis, text, (30, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Display buffer status
        buffer_fill = len(motion_buffer) / buffer_size
        status = f"Buffer: {buffer_fill*100:.0f}%"
        cv2.putText(vis, status, (30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Respiratory Rate Detection', vis)

        # Update
        prev_gray = gray
        frame_count += 1

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")
    if current_rate:
        print(f"Final respiratory rate: {current_rate:.1f} BPM")


if __name__ == '__main__':
    main()
