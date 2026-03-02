"""Streaming spectrogram widget using pyqtgraph."""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg


class SpectrogramWidget(QWidget):
    """Rolling spectrogram display for streaming audio.

    Maintains an overlap buffer between chunks for STFT continuity.
    Uses pyqtgraph ImageItem for efficient updates.

    Args:
        sample_rate: Audio sample rate in Hz.
        fmin: Minimum display frequency in Hz.
        fmax: Maximum display frequency in Hz.
        nperseg: STFT window size in samples.
        n_columns: Number of time columns in the display.
    """

    def __init__(
        self,
        sample_rate: int,
        fmin: float = 15_000,
        fmax: float = 130_000,
        nperseg: int = 512,
        n_columns: int = 500,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.nperseg = nperseg
        self.noverlap = nperseg * 3 // 4
        self.hop = nperseg - self.noverlap
        self.n_columns = n_columns

        # Frequency axis
        self._freqs = np.fft.rfftfreq(nperseg, 1.0 / sample_rate)
        self._fmin_idx = int(np.searchsorted(self._freqs, fmin))
        self._fmax_idx = int(np.searchsorted(self._freqs, fmax))
        self._n_freq_bins = self._fmax_idx - self._fmin_idx
        self._display_freqs = self._freqs[self._fmin_idx : self._fmax_idx]

        # Rolling spectrogram buffer (freq_bins x time_columns)
        self._spec_data = np.full(
            (self._n_freq_bins, n_columns), -120.0, dtype=np.float64
        )

        # Overlap buffer for STFT continuity between chunks
        self._overlap_buf = np.zeros(self.noverlap, dtype=np.float64)
        self._has_overlap = False

        # Hann window
        self._window = np.hanning(nperseg)

        # Setup pyqtgraph
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("left", "Frequency", units="Hz")
        self._plot_widget.setLabel("bottom", "Time")

        # ImageItem for spectrogram
        self._image = pg.ImageItem()
        self._plot_widget.addItem(self._image)

        # Color map: viridis-like
        cmap = pg.colormap.get("viridis")
        self._image.setLookupTable(cmap.getLookupTable())

        # Set axis ranges
        self._plot_widget.setYRange(self.fmin, self.fmax)
        self._plot_widget.setXRange(0, self.n_columns)

        # Color range (dB)
        self._image.setLevels([-120, 0])

        layout.addWidget(self._plot_widget)

    def feed(self, chunk: np.ndarray) -> None:
        """Process a chunk of audio and update the spectrogram.

        Args:
            chunk: 1D float64 array of audio samples.
        """
        # Prepend overlap from previous chunk
        if self._has_overlap:
            data = np.concatenate([self._overlap_buf, chunk])
        else:
            data = chunk
            self._has_overlap = True

        # Compute STFT columns
        new_columns = []
        pos = 0
        while pos + self.nperseg <= len(data):
            segment = data[pos : pos + self.nperseg]
            windowed = segment * self._window
            spectrum = np.fft.rfft(windowed)
            power = np.abs(spectrum) ** 2 / self.nperseg
            power_db = 10 * np.log10(power + 1e-12)
            # Crop to frequency range
            col = power_db[self._fmin_idx : self._fmax_idx]
            new_columns.append(col)
            pos += self.hop

        # Save overlap for next chunk
        overlap_start = max(0, len(data) - self.noverlap)
        self._overlap_buf = data[overlap_start:].copy()
        if len(self._overlap_buf) < self.noverlap:
            padded = np.zeros(self.noverlap, dtype=np.float64)
            padded[: len(self._overlap_buf)] = self._overlap_buf
            self._overlap_buf = padded

        if not new_columns:
            return

        # Roll spectrogram and append new columns
        n_new = len(new_columns)
        new_data = np.column_stack(new_columns)

        if n_new >= self.n_columns:
            self._spec_data = new_data[:, -self.n_columns :]
        else:
            self._spec_data = np.roll(self._spec_data, -n_new, axis=1)
            self._spec_data[:, -n_new:] = new_data

        self._update_image()

    def _update_image(self) -> None:
        """Update the pyqtgraph ImageItem with current spectrogram data."""
        # ImageItem expects (width, height) = (n_columns, n_freq_bins)
        # Transpose: spec_data is (freq, time) -> image needs (time, freq)
        self._image.setImage(
            self._spec_data.T,
            autoLevels=False,
        )
        # Position and scale to match axes
        freq_step = (self.fmax - self.fmin) / self._n_freq_bins if self._n_freq_bins > 0 else 1
        self._image.setRect(0, self.fmin, self.n_columns, self.fmax - self.fmin)
