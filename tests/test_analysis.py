"""Tests for batsound.analysis module."""

import numpy as np

from batsound.analysis import bandpass_filter, compute_spectrogram, compute_psd


def _make_signal(
    freq_hz: float, sample_rate: int = 250_000, duration: float = 0.1
) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t)


def test_bandpass_passes_in_band() -> None:
    sr = 250_000
    sig = _make_signal(45_000, sr)
    filtered = bandpass_filter(sig, sr, low_hz=20_000, high_hz=80_000)
    # Signal should be mostly preserved (high correlation)
    corr = np.corrcoef(sig, filtered)[0, 1]
    assert corr > 0.9


def test_bandpass_attenuates_out_of_band() -> None:
    sr = 250_000
    sig = _make_signal(5_000, sr)  # 5 kHz - below bat range
    filtered = bandpass_filter(sig, sr, low_hz=20_000, high_hz=80_000)
    # Should be strongly attenuated
    assert np.max(np.abs(filtered)) < 0.05


def test_spectrogram_shape() -> None:
    sr = 250_000
    sig = _make_signal(45_000, sr, duration=0.1)
    freqs, times, sxx = compute_spectrogram(sig, sr, nperseg=512)
    assert freqs.shape[0] == sxx.shape[0]
    assert times.shape[0] == sxx.shape[1]
    # Frequency axis should go up to Nyquist
    assert freqs[-1] == sr / 2


def test_spectrogram_peak_at_signal_freq() -> None:
    sr = 250_000
    sig = _make_signal(45_000, sr, duration=0.1)
    freqs, times, sxx = compute_spectrogram(sig, sr, nperseg=512)
    # Average power across time, find peak frequency
    mean_power = sxx.mean(axis=1)
    peak_freq = freqs[np.argmax(mean_power)]
    # Should be within 1 kHz of 45 kHz
    assert abs(peak_freq - 45_000) < 1_000


def test_psd_peak_at_signal_freq() -> None:
    sr = 250_000
    sig = _make_signal(50_000, sr, duration=0.1)
    freqs, psd = compute_psd(sig, sr)
    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - 50_000) < 1_000
