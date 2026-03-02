"""Spectrogram and signal visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(
    freqs: np.ndarray,
    times: np.ndarray,
    sxx_db: np.ndarray,
    fmin: float = 15_000,
    fmax: float = 130_000,
    title: str = "",
    output: Path | None = None,
) -> None:
    """Plot a spectrogram with frequency axis in kHz.

    Args:
        freqs: Frequency array in Hz.
        times: Time array in seconds.
        sxx_db: Power spectral density in dB.
        fmin: Minimum display frequency in Hz.
        fmax: Maximum display frequency in Hz.
        title: Plot title.
        output: If provided, save figure to this path instead of showing.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.pcolormesh(
        times * 1000,       # ms
        freqs / 1000,       # kHz
        sxx_db,
        shading="gouraud",
        cmap="inferno",
    )
    ax.set_ylim(fmin / 1000, fmax / 1000)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    fig.colorbar(ax.collections[0], ax=ax, label="Power (dB)")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        plt.close(fig)
    else:
        plt.show()
