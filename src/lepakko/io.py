"""WAV I/O and metadata helpers."""

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file preserving its native sample rate.

    Returns (data, sample_rate) where data is a 1D float64 array
    (first channel if stereo).
    """
    data, sample_rate = sf.read(path, dtype="float64")
    # If stereo, take first channel
    if data.ndim > 1:
        data = data[:, 0]
    return data, sample_rate


def wav_info(path: Path) -> dict[str, Any]:
    """Return metadata dict for a WAV file without loading audio data."""
    info = sf.info(str(path))
    return {
        "path": str(path),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": info.duration,
        "subtype": info.subtype,
    }
