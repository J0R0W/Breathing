"""
Signal Processing Module for Respiratory Rate Detection

This module provides signal processing utilities including:
- Bandpass filtering (Butterworth)
- FFT-based frequency analysis
- Peak detection
- Signal normalization and preprocessing
"""

from .filters import butter_bandpass_filter, normalize_signal, denoise_signal
from .frequency_analysis import fft_respiratory_rate, estimate_rate_fft
from .peak_detection import detect_peaks, peak_based_rate

__all__ = [
    'butter_bandpass_filter',
    'normalize_signal',
    'denoise_signal',
    'fft_respiratory_rate',
    'estimate_rate_fft',
    'detect_peaks',
    'peak_based_rate',
]
