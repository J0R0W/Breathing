"""
Frequency analysis utilities for respiratory rate estimation.

This module provides FFT-based methods to extract respiratory rate
from temporal signals.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy.fft import fft, fftfreq


def fft_respiratory_rate(
    signal_data: np.ndarray,
    fs: float,
    freq_min: float = 0.2,
    freq_max: float = 0.8
) -> float:
    """
    Estimate respiratory rate using Fast Fourier Transform (FFT).

    This method converts the signal to frequency domain and finds the
    dominant frequency in the respiratory range.

    Args:
        signal_data: Input temporal signal (motion, color, etc.)
        fs: Sampling frequency in Hz (video frame rate)
        freq_min: Minimum respiratory frequency in Hz (default: 0.2 = 12 BPM)
        freq_max: Maximum respiratory frequency in Hz (default: 0.8 = 48 BPM)

    Returns:
        Estimated respiratory rate in breaths per minute (BPM)

    Example:
        >>> # Synthetic breathing signal at 15 BPM (0.25 Hz)
        >>> t = np.linspace(0, 60, 1800)  # 60 seconds at 30 fps
        >>> signal = np.sin(2 * np.pi * 0.25 * t)
        >>> rate = fft_respiratory_rate(signal, fs=30)
        >>> print(f"Rate: {rate:.1f} BPM")  # Should be ~15 BPM
    """
    # Remove DC component
    signal_data = signal_data - np.mean(signal_data)

    # Apply window to reduce spectral leakage
    window = np.hamming(len(signal_data))
    signal_windowed = signal_data * window

    # Compute FFT
    fft_vals = fft(signal_windowed)
    freqs = fftfreq(len(signal_windowed), 1/fs)

    # Get power spectrum (magnitude squared)
    power = np.abs(fft_vals) ** 2

    # Only consider positive frequencies in respiratory range
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    respiratory_freqs = freqs[mask]
    respiratory_power = power[mask]

    if len(respiratory_power) == 0:
        raise ValueError("No frequencies found in respiratory range")

    # Find frequency with maximum power
    peak_idx = np.argmax(respiratory_power)
    dominant_freq = respiratory_freqs[peak_idx]

    # Convert to breaths per minute
    respiratory_rate = dominant_freq * 60

    return respiratory_rate


def estimate_rate_fft(
    signal_data: np.ndarray,
    fs: float,
    freq_min: float = 0.2,
    freq_max: float = 0.8,
    return_confidence: bool = False
) -> Tuple[float, Optional[float]]:
    """
    Estimate respiratory rate with optional confidence measure.

    Args:
        signal_data: Input signal
        fs: Sampling frequency in Hz
        freq_min: Minimum respiratory frequency in Hz
        freq_max: Maximum respiratory frequency in Hz
        return_confidence: If True, also return confidence score

    Returns:
        If return_confidence is False: respiratory_rate (BPM)
        If return_confidence is True: (respiratory_rate, confidence)

    Example:
        >>> signal = np.sin(2*np.pi*0.3*np.arange(900)/30)  # 18 BPM
        >>> rate, conf = estimate_rate_fft(signal, fs=30, return_confidence=True)
        >>> print(f"Rate: {rate:.1f} BPM, Confidence: {conf:.2f}")
    """
    # Remove DC and apply window
    signal_data = signal_data - np.mean(signal_data)
    window = np.hamming(len(signal_data))
    signal_windowed = signal_data * window

    # FFT
    fft_vals = fft(signal_windowed)
    freqs = fftfreq(len(signal_windowed), 1/fs)
    power = np.abs(fft_vals) ** 2

    # Respiratory range
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    respiratory_freqs = freqs[mask]
    respiratory_power = power[mask]

    if len(respiratory_power) == 0:
        if return_confidence:
            return 0.0, 0.0
        return 0.0

    # Find peak
    peak_idx = np.argmax(respiratory_power)
    dominant_freq = respiratory_freqs[peak_idx]
    respiratory_rate = dominant_freq * 60

    if return_confidence:
        # Confidence based on peak prominence
        peak_power = respiratory_power[peak_idx]
        mean_power = np.mean(respiratory_power)

        # Normalized confidence (0-1)
        if mean_power > 0:
            confidence = min(peak_power / (mean_power * 3), 1.0)
        else:
            confidence = 0.0

        return respiratory_rate, confidence

    return respiratory_rate


def get_frequency_spectrum(
    signal_data: np.ndarray,
    fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get frequency spectrum of signal for visualization.

    Args:
        signal_data: Input signal
        fs: Sampling frequency in Hz

    Returns:
        Tuple of (frequencies, power_spectrum)

    Example:
        >>> signal = np.sin(2*np.pi*0.3*np.arange(300)/30)
        >>> freqs, power = get_frequency_spectrum(signal, fs=30)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(freqs, power)
        >>> plt.xlabel('Frequency (Hz)')
        >>> plt.ylabel('Power')
    """
    # Preprocessing
    signal_data = signal_data - np.mean(signal_data)
    window = np.hamming(len(signal_data))
    signal_windowed = signal_data * window

    # FFT
    fft_vals = fft(signal_windowed)
    freqs = fftfreq(len(signal_windowed), 1/fs)
    power = np.abs(fft_vals) ** 2

    # Only return positive frequencies
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    power = power[positive_mask]

    return freqs, power


def multi_peak_analysis(
    signal_data: np.ndarray,
    fs: float,
    num_peaks: int = 3,
    freq_min: float = 0.2,
    freq_max: float = 0.8
) -> list:
    """
    Find multiple peaks in frequency spectrum.

    Useful for detecting harmonics or identifying multiple respiratory patterns.

    Args:
        signal_data: Input signal
        fs: Sampling frequency
        num_peaks: Number of peaks to find
        freq_min: Minimum frequency to consider
        freq_max: Maximum frequency to consider

    Returns:
        List of (frequency_hz, power, rate_bpm) tuples, sorted by power

    Example:
        >>> signal = np.sin(2*np.pi*0.3*np.arange(900)/30)
        >>> peaks = multi_peak_analysis(signal, fs=30, num_peaks=3)
        >>> for freq, power, rate in peaks:
        ...     print(f"Rate: {rate:.1f} BPM, Power: {power:.2e}")
    """
    from scipy.signal import find_peaks

    # Get spectrum
    signal_data = signal_data - np.mean(signal_data)
    window = np.hamming(len(signal_data))
    signal_windowed = signal_data * window

    fft_vals = fft(signal_windowed)
    freqs = fftfreq(len(signal_windowed), 1/fs)
    power = np.abs(fft_vals) ** 2

    # Respiratory range
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    respiratory_freqs = freqs[mask]
    respiratory_power = power[mask]

    # Find peaks
    peak_indices, properties = find_peaks(
        respiratory_power,
        height=np.max(respiratory_power) * 0.1,  # At least 10% of max
        distance=int(0.05 * fs)  # At least 0.05 Hz apart
    )

    if len(peak_indices) == 0:
        return []

    # Sort by power (descending)
    peak_powers = properties['peak_heights']
    sorted_indices = np.argsort(peak_powers)[::-1]

    # Get top N peaks
    results = []
    for idx in sorted_indices[:num_peaks]:
        peak_idx = peak_indices[idx]
        freq_hz = respiratory_freqs[peak_idx]
        power_val = peak_powers[idx]
        rate_bpm = freq_hz * 60
        results.append((freq_hz, power_val, rate_bpm))

    return results


def spectral_entropy(signal_data: np.ndarray, fs: float) -> float:
    """
    Calculate spectral entropy of signal.

    Low entropy indicates a pure periodic signal (good for rate estimation).
    High entropy indicates noise or irregular breathing.

    Args:
        signal_data: Input signal
        fs: Sampling frequency

    Returns:
        Spectral entropy (0 = pure tone, higher = more chaotic)

    Example:
        >>> # Pure sine wave has low entropy
        >>> pure_signal = np.sin(2*np.pi*0.3*np.arange(300)/30)
        >>> entropy_pure = spectral_entropy(pure_signal, fs=30)
        >>>
        >>> # Noisy signal has high entropy
        >>> noisy_signal = pure_signal + np.random.randn(300) * 0.5
        >>> entropy_noisy = spectral_entropy(noisy_signal, fs=30)
    """
    # Get power spectrum
    freqs, power = get_frequency_spectrum(signal_data, fs)

    # Normalize to probability distribution
    power_normalized = power / np.sum(power)

    # Calculate entropy
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    entropy = -np.sum(power_normalized * np.log2(power_normalized + epsilon))

    return entropy


def autocorrelation_rate(
    signal_data: np.ndarray,
    fs: float,
    freq_min: float = 0.2,
    freq_max: float = 0.8
) -> float:
    """
    Estimate respiratory rate using autocorrelation.

    Alternative to FFT that can be more robust to noise.

    Args:
        signal_data: Input signal
        fs: Sampling frequency
        freq_min: Minimum respiratory frequency
        freq_max: Maximum respiratory frequency

    Returns:
        Estimated respiratory rate in BPM

    Example:
        >>> signal = np.sin(2*np.pi*0.25*np.arange(900)/30)  # 15 BPM
        >>> rate = autocorrelation_rate(signal, fs=30)
    """
    # Remove mean
    signal_data = signal_data - np.mean(signal_data)

    # Compute autocorrelation
    autocorr = np.correlate(signal_data, signal_data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags

    # Expected lag range for respiratory frequencies
    min_lag = int(fs / freq_max)
    max_lag = int(fs / freq_min)

    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1

    # Find peak in valid range (skip first peak at lag=0)
    valid_autocorr = autocorr[min_lag:max_lag]

    if len(valid_autocorr) == 0:
        return 0.0

    peak_lag = min_lag + np.argmax(valid_autocorr)

    # Convert lag to frequency and then to BPM
    period_seconds = peak_lag / fs
    frequency_hz = 1.0 / period_seconds
    respiratory_rate = frequency_hz * 60

    return respiratory_rate


def estimate_confidence_from_spectrum(
    signal_data: np.ndarray,
    fs: float,
    estimated_rate: float
) -> Dict[str, float]:
    """
    Calculate various confidence metrics for estimated rate.

    Args:
        signal_data: Input signal
        fs: Sampling frequency
        estimated_rate: Estimated respiratory rate in BPM

    Returns:
        Dictionary with confidence metrics:
        - 'peak_prominence': How strong the peak is vs background
        - 'spectral_entropy': Signal periodicity (lower is better)
        - 'snr_estimate': Estimated signal-to-noise ratio
        - 'overall_confidence': Combined confidence score (0-1)

    Example:
        >>> signal = np.sin(2*np.pi*0.3*np.arange(900)/30)
        >>> rate = fft_respiratory_rate(signal, fs=30)
        >>> conf = estimate_confidence_from_spectrum(signal, 30, rate)
        >>> print(f"Confidence: {conf['overall_confidence']:.2f}")
    """
    freqs, power = get_frequency_spectrum(signal_data, fs)

    # Find estimated frequency in spectrum
    estimated_freq = estimated_rate / 60
    idx = np.argmin(np.abs(freqs - estimated_freq))

    # Peak prominence
    peak_power = power[idx]
    mean_power = np.mean(power)
    peak_prominence = peak_power / (mean_power + 1e-10)

    # Spectral entropy
    entropy = spectral_entropy(signal_data, fs)

    # SNR estimate (peak vs noise floor)
    # Estimate noise floor as median of power spectrum
    noise_floor = np.median(power)
    snr_estimate = 10 * np.log10((peak_power / (noise_floor + 1e-10)))

    # Overall confidence (0-1 scale)
    # Heuristic: combine normalized metrics
    conf_peak = min(peak_prominence / 5.0, 1.0)  # 5x above mean = full confidence
    conf_entropy = max(0, 1.0 - entropy / 10.0)  # Lower entropy = higher confidence
    conf_snr = min(snr_estimate / 20.0, 1.0) if snr_estimate > 0 else 0.0

    overall_confidence = (conf_peak + conf_entropy + conf_snr) / 3.0

    return {
        'peak_prominence': peak_prominence,
        'spectral_entropy': entropy,
        'snr_estimate': snr_estimate,
        'overall_confidence': overall_confidence
    }
