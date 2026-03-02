"""Signal processing: filtering, STFT, spectrogram, PSD."""

import numpy as np
from scipy import signal


def bandpass_filter(
    data: np.ndarray,
    sample_rate: int,
    low_hz: float = 15_000,
    high_hz: float = 130_000,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter.

    Args:
        data: 1D audio signal.
        sample_rate: Sample rate in Hz.
        low_hz: Low cutoff frequency in Hz.
        high_hz: High cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal.
    """
    nyquist = sample_rate / 2
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)  # Clamp below Nyquist
    sos = signal.butter(order, [low, high], btype="band", output="sos")
    return signal.sosfiltfilt(sos, data)


def compute_spectrogram(
    data: np.ndarray,
    sample_rate: int,
    nperseg: int = 512,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram using STFT.

    Args:
        data: 1D audio signal.
        sample_rate: Sample rate in Hz.
        nperseg: FFT window size in samples (512 at 250 kHz ~ 2 ms).
        noverlap: Overlap in samples. Defaults to 75% of nperseg.

    Returns:
        (frequencies, times, Sxx) where Sxx is power spectral density
        in dB (10 * log10).
    """
    if noverlap is None:
        noverlap = nperseg * 3 // 4
    freqs, times, sxx = signal.spectrogram(
        data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hann",
    )
    # Convert to dB, floor at -120 dB
    sxx_db = 10 * np.log10(sxx + 1e-12)
    return freqs, times, sxx_db


def compute_psd(
    data: np.ndarray,
    sample_rate: int,
    nperseg: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method.

    Returns:
        (frequencies, psd) arrays.
    """
    return signal.welch(data, fs=sample_rate, nperseg=nperseg)
