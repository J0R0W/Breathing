#!/usr/bin/env python3
"""
Example 4: Batch Processing Multiple Videos

Process multiple video files and export results to CSV.

Usage:
    python 04_batch_processing.py --directory videos/ --output results.csv
    python 04_batch_processing.py --files video1.mp4 video2.mp4 --output results.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import glob
from respiratory_monitor import BatchVideoProcessor


def main():
    parser = argparse.ArgumentParser(description='Batch process multiple videos')
    parser.add_argument('--directory', type=str, help='Directory containing videos')
    parser.add_argument('--pattern', type=str, default='*.mp4',
                       help='File pattern (default: *.mp4)')
    parser.add_argument('--files', nargs='+', help='Specific video files to process')
    parser.add_argument('--output', type=str, default='results.csv',
                       help='Output CSV file')
    parser.add_argument('--detect', type=str, default='pose',
                       choices=['manual', 'pose'],
                       help='ROI detection method')
    parser.add_argument('--motion', type=str, default='dense',
                       choices=['dense', 'sparse'],
                       help='Motion extraction method')
    args = parser.parse_args()

    print("=" * 70)
    print("BATCH VIDEO PROCESSOR")
    print("=" * 70)

    # Create processor
    processor = BatchVideoProcessor(
        detection_method=args.detect,
        motion_method=args.motion
    )

    # Get video files
    if args.directory:
        print(f"Scanning directory: {args.directory}")
        print(f"Pattern: {args.pattern}")

        video_files = glob.glob(os.path.join(args.directory, args.pattern))
        print(f"Found {len(video_files)} videos")

    elif args.files:
        video_files = args.files
        print(f"Processing {len(video_files)} specified files")

    else:
        print("Please specify --directory or --files")
        return

    if not video_files:
        print("No video files found")
        return

    print()

    # Process videos
    results = {}
    success_count = 0
    fail_count = 0

    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")

        from respiratory_monitor import RespiratoryRateMonitor

        monitor = RespiratoryRateMonitor(
            detection_method=args.detect,
            motion_method=args.motion
        )

        try:
            rate = monitor.process_video(video_path)

            if rate:
                results[video_path] = rate
                success_count += 1
                print(f"✓ Result: {rate:.1f} BPM")
            else:
                results[video_path] = None
                fail_count += 1
                print("✗ Failed to estimate rate")

        except Exception as e:
            results[video_path] = None
            fail_count += 1
            print(f"✗ Error: {e}")

    # Export results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total videos:    {len(video_files)}")
    print(f"Successful:      {success_count}")
    print(f"Failed:          {fail_count}")
    print()

    # Save to CSV
    import csv

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Video File', 'Respiratory Rate (BPM)', 'Status'])

        for video_path, rate in results.items():
            if rate:
                status = 'Success'
                rate_str = f"{rate:.2f}"
            else:
                status = 'Failed'
                rate_str = 'N/A'

            writer.writerow([os.path.basename(video_path), rate_str, status])

    print(f"Results exported to: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
