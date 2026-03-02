"""Tests for lepakko.io module."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from lepakko.io import load_wav, wav_info


def _make_wav(path: Path, sample_rate: int = 250_000, duration: float = 0.1) -> None:
    """Write a synthetic WAV file."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 45 kHz sine wave (pipistrelle-like frequency)
    data = np.sin(2 * np.pi * 45_000 * t).astype(np.float64)
    sf.write(str(path), data, sample_rate)


def test_load_wav_preserves_sample_rate(tmp_path: Path) -> None:
    wav_path = tmp_path / "test.wav"
    _make_wav(wav_path, sample_rate=250_000)
    data, sr = load_wav(wav_path)
    assert sr == 250_000
    assert data.ndim == 1
    assert len(data) == 25_000  # 0.1s * 250k


def test_load_wav_stereo_takes_first_channel(tmp_path: Path) -> None:
    wav_path = tmp_path / "stereo.wav"
    sr = 192_000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    left = np.sin(2 * np.pi * 40_000 * t)
    right = np.sin(2 * np.pi * 50_000 * t)
    stereo = np.column_stack([left, right])
    sf.write(str(wav_path), stereo, sr)

    data, sample_rate = load_wav(wav_path)
    assert sample_rate == 192_000
    assert data.ndim == 1
    np.testing.assert_allclose(data, left, atol=1e-4)


def test_wav_info(tmp_path: Path) -> None:
    wav_path = tmp_path / "info.wav"
    _make_wav(wav_path, sample_rate=384_000, duration=0.5)
    meta = wav_info(wav_path)
    assert meta["sample_rate"] == 384_000
    assert meta["channels"] == 1
    assert meta["frames"] == 192_000  # 0.5s * 384k
    assert abs(meta["duration"] - 0.5) < 0.001
