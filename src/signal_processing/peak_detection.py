"""
Peak detection utilities for respiratory rate estimation.

This module provides time-domain methods to detect breathing cycles
by finding peaks in the respiratory signal.
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Optional


def detect_peaks(
    signal_data: np.ndarray,
    fs: float,
    min_distance: Optional[float] = None,
    prominence: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """
    Detect peaks in respiratory signal.

    Args:
        signal_data: Input signal
        fs: Sampling frequency in Hz
        min_distance: Minimum distance between peaks in seconds
                     (default: 0.5s = max 120 BPM)
        prominence: Minimum peak prominence (default: auto from signal std)

    Returns:
        Tuple of (peak_indices, properties)
        - peak_indices: Array of peak locations
        - properties: Dict with 'peak_heights', 'prominences', etc.

    Example:
        >>> # Sine wave with period of 2 seconds (30 BPM)
        >>> t = np.arange(0, 10, 1/30)  # 10 seconds at 30 fps
        >>> signal = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz = 30 BPM
        >>> peaks, props = detect_peaks(signal, fs=30)
        >>> print(f"Found {len(peaks)} peaks")
    """
    # Default parameters
    if min_distance is None:
        min_distance = 0.5  # seconds (max 120 BPM)

    if prominence is None:
        prominence = np.std(signal_data) * 0.3  # 30% of signal std

    # Convert time to samples
    min_distance_samples = int(min_distance * fs)

    # Find peaks
    peak_indices, properties = signal.find_peaks(
        signal_data,
        distance=min_distance_samples,
        prominence=prominence
    )

    return peak_indices, properties


def peak_based_rate(
    signal_data: np.ndarray,
    fs: float,
    method: str = 'median'
) -> float:
    """
    Estimate respiratory rate from peak-to-peak intervals.

    Args:
        signal_data: Input signal
        fs: Sampling frequency in Hz
        method: How to aggregate intervals
            - 'median': Use median interval (robust to outliers)
            - 'mean': Use mean interval
            - 'mode': Use most common interval

    Returns:
        Estimated respiratory rate in BPM

    Example:
        >>> # Create signal with 15 BPM
        >>> t = np.arange(0, 60, 1/30)  # 60 seconds at 30 fps
        >>> signal = np.sin(2 * np.pi * 0.25 * t)  # 0.25 Hz = 15 BPM
        >>> rate = peak_based_rate(signal, fs=30)
        >>> print(f"Rate: {rate:.1f} BPM")  # Should be ~15
    """
    # Detect peaks
    peaks, _ = detect_peaks(signal_data, fs)

    if len(peaks) < 2:
        raise ValueError("Need at least 2 peaks to estimate rate")

    # Calculate intervals (in samples)
    intervals_samples = np.diff(peaks)

    # Convert to seconds
    intervals_seconds = intervals_samples / fs

    # Aggregate intervals
    if method == 'median':
        avg_interval = np.median(intervals_seconds)
    elif method == 'mean':
        avg_interval = np.mean(intervals_seconds)
    elif method == 'mode':
        # Use histogram to find most common interval
        hist, bin_edges = np.histogram(intervals_seconds, bins=20)
        mode_idx = np.argmax(hist)
        avg_interval = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert to BPM
    respiratory_rate = 60.0 / avg_interval

    return respiratory_rate


def detect_breaths(
    signal_data: np.ndarray,
    fs: float
) -> List[Tuple[int, int]]:
    """
    Detect individual breath cycles (inspiration + expiration).

    Args:
        signal_data: Input signal
        fs: Sampling frequency

    Returns:
        List of (start_idx, end_idx) tuples for each breath

    Example:
        >>> signal = np.sin(2*np.pi*0.3*np.arange(900)/30)
        >>> breaths = detect_breaths(signal, fs=30)
        >>> print(f"Detected {len(breaths)} breaths")
    """
    # Find peaks (inhalation maxima)
    peaks, _ = detect_peaks(signal_data, fs)

    # Find troughs (exhalation minima) by inverting signal
    troughs, _ = detect_peaks(-signal_data, fs)

    breaths = []

    # Match peaks with surrounding troughs
    for peak in peaks:
        # Find trough before peak
        troughs_before = troughs[troughs < peak]
        if len(troughs_before) == 0:
            continue
        start = troughs_before[-1]

        # Find trough after peak
        troughs_after = troughs[troughs > peak]
        if len(troughs_after) == 0:
            continue
        end = troughs_after[0]

        breaths.append((start, end))

    return breaths


def calculate_breath_metrics(
    signal_data: np.ndarray,
    fs: float
) -> dict:
    """
    Calculate detailed breath-by-breath metrics.

    Args:
        signal_data: Input signal
        fs: Sampling frequency

    Returns:
        Dictionary with metrics:
        - 'breath_count': Number of breaths detected
        - 'mean_rate': Mean respiratory rate (BPM)
        - 'rate_std': Standard deviation of rate
        - 'mean_amplitude': Mean breath amplitude
        - 'regularity': Breath-to-breath regularity (0-1, higher=more regular)

    Example:
        >>> signal = np.sin(2*np.pi*0.3*np.arange(900)/30)
        >>> metrics = calculate_breath_metrics(signal, fs=30)
        >>> print(f"Mean rate: {metrics['mean_rate']:.1f} BPM")
    """
    breaths = detect_breaths(signal_data, fs)

    if len(breaths) < 2:
        return {
            'breath_count': len(breaths),
            'mean_rate': 0.0,
            'rate_std': 0.0,
            'mean_amplitude': 0.0,
            'regularity': 0.0
        }

    # Calculate durations
    durations = [(end - start) / fs for start, end in breaths]
    rates = [60.0 / d for d in durations]

    # Calculate amplitudes
    amplitudes = []
    for start, end in breaths:
        breath_signal = signal_data[start:end]
        amplitude = np.max(breath_signal) - np.min(breath_signal)
        amplitudes.append(amplitude)

    # Regularity: inverse of coefficient of variation
    cv = np.std(rates) / (np.mean(rates) + 1e-10)
    regularity = 1.0 / (1.0 + cv)

    return {
        'breath_count': len(breaths),
        'mean_rate': np.mean(rates),
        'rate_std': np.std(rates),
        'mean_amplitude': np.mean(amplitudes),
        'regularity': regularity
    }


def adaptive_peak_detection(
    signal_data: np.ndarray,
    fs: float,
    window_size: int = None
) -> np.ndarray:
    """
    Adaptive peak detection with locally-adjusted thresholds.

    Useful for signals with varying amplitude or baseline drift.

    Args:
        signal_data: Input signal
        fs: Sampling frequency
        window_size: Size of adaptive window in samples
                     (default: 3 seconds worth of samples)

    Returns:
        Array of peak indices

    Example:
        >>> # Signal with varying amplitude
        >>> t = np.arange(0, 30, 1/30)
        >>> envelope = 1 + 0.5 * np.sin(2*np.pi*0.1*t)
        >>> signal = envelope * np.sin(2*np.pi*0.3*t)
        >>> peaks = adaptive_peak_detection(signal, fs=30)
    """
    if window_size is None:
        window_size = int(3 * fs)  # 3 seconds

    peaks = []

    # Slide window across signal
    for i in range(0, len(signal_data), window_size // 2):
        window_end = min(i + window_size, len(signal_data))
        window = signal_data[i:window_end]

        # Calculate local threshold
        local_mean = np.mean(window)
        local_std = np.std(window)
        threshold = local_mean + 0.5 * local_std

        # Find peaks in window
        window_peaks, _ = signal.find_peaks(
            window,
            height=threshold,
            distance=int(0.5 * fs)
        )

        # Convert to global indices
        global_peaks = window_peaks + i

        # Add peaks (avoid duplicates near window boundaries)
        for peak in global_peaks:
            if len(peaks) == 0 or peak - peaks[-1] > fs * 0.3:
                peaks.append(peak)

    return np.array(peaks)


def validate_peaks(
    signal_data: np.ndarray,
    peak_indices: np.ndarray,
    fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate detected peaks and remove false positives.

    Args:
        signal_data: Input signal
        peak_indices: Detected peak locations
        fs: Sampling frequency

    Returns:
        Tuple of (valid_peaks, invalid_peaks)

    Example:
        >>> signal = np.sin(2*np.pi*0.3*np.arange(300)/30)
        >>> all_peaks, _ = detect_peaks(signal, fs=30)
        >>> valid, invalid = validate_peaks(signal, all_peaks, fs=30)
    """
    if len(peak_indices) < 2:
        return peak_indices, np.array([])

    # Calculate intervals
    intervals = np.diff(peak_indices) / fs

    # Expected respiratory interval range (0.5 to 5 seconds = 12-120 BPM)
    min_interval = 0.5
    max_interval = 5.0

    # Median interval
    median_interval = np.median(intervals)

    valid = [peak_indices[0]]  # First peak is always valid
    invalid = []

    for i in range(1, len(peak_indices)):
        interval = intervals[i-1]
        peak = peak_indices[i]

        # Check if interval is reasonable
        if min_interval <= interval <= max_interval:
            # Check if interval is not too far from median
            if abs(interval - median_interval) < median_interval * 0.5:
                valid.append(peak)
            else:
                invalid.append(peak)
        else:
            invalid.append(peak)

    return np.array(valid), np.array(invalid)


def zero_crossing_rate(signal_data: np.ndarray, fs: float) -> float:
    """
    Calculate zero-crossing rate of signal.

    Can be used as an alternative rate estimation method.

    Args:
        signal_data: Input signal
        fs: Sampling frequency

    Returns:
        Zero-crossing rate in crossings per minute

    Example:
        >>> signal = np.sin(2*np.pi*0.3*np.arange(300)/30)  # 18 BPM
        >>> zcr = zero_crossing_rate(signal, fs=30)
        >>> # ZCR should be ~2x the frequency (2 crossings per cycle)
        >>> estimated_bpm = zcr / 2
    """
    # Center signal around zero
    signal_centered = signal_data - np.mean(signal_data)

    # Find zero crossings
    crossings = np.where(np.diff(np.sign(signal_centered)))[0]

    # Calculate rate
    duration_seconds = len(signal_data) / fs
    crossings_per_second = len(crossings) / duration_seconds
    crossings_per_minute = crossings_per_second * 60

    return crossings_per_minute
