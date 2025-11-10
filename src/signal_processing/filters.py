"""
Signal filtering utilities for respiratory rate detection.

This module provides various filtering methods to clean and preprocess
respiratory signals extracted from video data.
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 3
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to signal.

    This filter removes frequencies outside the expected respiratory range,
    helping to isolate the breathing signal from noise.

    Args:
        data: Input signal (1D array)
        lowcut: Low cutoff frequency in Hz (e.g., 0.2 Hz = 12 BPM)
        highcut: High cutoff frequency in Hz (e.g., 0.8 Hz = 48 BPM)
        fs: Sampling frequency in Hz (typically video frame rate)
        order: Filter order (higher = sharper cutoff, default=3)

    Returns:
        Filtered signal (same length as input)

    Example:
        >>> signal = np.random.randn(1000)
        >>> filtered = butter_bandpass_filter(signal, 0.2, 0.8, 30, order=3)
    """
    # Calculate normalized frequencies
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply zero-phase filtering (forward and backward)
    # This eliminates phase distortion
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def normalize_signal(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize signal to standard range.

    Args:
        data: Input signal
        method: Normalization method
            - 'zscore': Zero mean, unit variance
            - 'minmax': Scale to [0, 1]
            - 'robust': Use median and IQR (robust to outliers)

    Returns:
        Normalized signal

    Example:
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> normalized = normalize_signal(signal, method='zscore')
    """
    if method == 'zscore':
        # Z-score normalization: (x - mean) / std
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:  # Avoid division by zero
            return data - mean
        return (data - mean) / std

    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val < 1e-10:
            return data - min_val
        return (data - min_val) / (max_val - min_val)

    elif method == 'robust':
        # Robust normalization using median and IQR
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-10:
            return data - median
        return (data - median) / iqr

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denoise_signal(
    data: np.ndarray,
    fs: float,
    cutoff: Optional[float] = None
) -> np.ndarray:
    """
    Apply lowpass filter to denoise signal.

    Args:
        data: Input signal
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz (default: fs/4)

    Returns:
        Denoised signal

    Example:
        >>> noisy_signal = np.sin(2*np.pi*0.3*np.arange(100)/30) + np.random.randn(100)*0.1
        >>> clean_signal = denoise_signal(noisy_signal, fs=30)
    """
    if cutoff is None:
        cutoff = fs / 4  # Default to quarter of sampling rate

    # Design lowpass filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low')

    # Apply filter
    filtered = signal.filtfilt(b, a, data)

    return filtered


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply moving average smoothing.

    Args:
        data: Input signal
        window_size: Size of the moving window

    Returns:
        Smoothed signal

    Example:
        >>> signal = np.random.randn(100)
        >>> smoothed = moving_average(signal, window_size=5)
    """
    if window_size < 1:
        return data

    # Use convolution for efficiency
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(data, window, mode='same')

    return smoothed


def remove_dc_component(data: np.ndarray) -> np.ndarray:
    """
    Remove DC component (mean) from signal.

    Args:
        data: Input signal

    Returns:
        Signal with zero mean

    Example:
        >>> signal = np.array([5, 6, 7, 8, 9])  # mean = 7
        >>> centered = remove_dc_component(signal)  # mean = 0
    """
    return data - np.mean(data)


def detrend_signal(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Remove trend from signal.

    Args:
        data: Input signal
        method: Detrending method
            - 'linear': Remove linear trend
            - 'constant': Remove mean only

    Returns:
        Detrended signal

    Example:
        >>> # Signal with upward trend
        >>> signal = np.arange(100) + np.sin(2*np.pi*0.1*np.arange(100))
        >>> detrended = detrend_signal(signal, method='linear')
    """
    if method == 'linear':
        return signal.detrend(data, type='linear')
    elif method == 'constant':
        return signal.detrend(data, type='constant')
    else:
        raise ValueError(f"Unknown detrend method: {method}")


def apply_window(data: np.ndarray, window_type: str = 'hamming') -> np.ndarray:
    """
    Apply window function to signal.

    Windows reduce spectral leakage in FFT analysis.

    Args:
        data: Input signal
        window_type: Type of window
            - 'hamming': Hamming window (good general purpose)
            - 'hann': Hann window
            - 'blackman': Blackman window (best frequency resolution)
            - 'bartlett': Bartlett window

    Returns:
        Windowed signal

    Example:
        >>> signal = np.random.randn(100)
        >>> windowed = apply_window(signal, window_type='hamming')
    """
    n = len(data)

    if window_type == 'hamming':
        window = np.hamming(n)
    elif window_type == 'hann':
        window = np.hanning(n)
    elif window_type == 'blackman':
        window = np.blackman(n)
    elif window_type == 'bartlett':
        window = np.bartlett(n)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    return data * window


def calculate_snr(signal_data: np.ndarray, noise_data: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR).

    Args:
        signal_data: Clean signal
        noise_data: Noise signal

    Returns:
        SNR in decibels (dB)

    Example:
        >>> signal = np.sin(2*np.pi*0.3*np.arange(100)/30)
        >>> noise = np.random.randn(100) * 0.1
        >>> snr = calculate_snr(signal, noise)
    """
    signal_power = np.mean(signal_data ** 2)
    noise_power = np.mean(noise_data ** 2)

    if noise_power < 1e-10:
        return np.inf

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)

    return snr_db
