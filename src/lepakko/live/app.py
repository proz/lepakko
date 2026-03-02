"""LiveApp — main window wiring ChunkReader, SpectrogramWidget, and AudioOutput."""

from pathlib import Path
from typing import Literal

import threading
from collections import deque

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from lepakko.live.stream import ChunkReader, MicReader
from lepakko.live.spectrogram_widget import SpectrogramWidget
from lepakko.live.audio_output import AudioOutput

METHODS = ["time-expansion", "heterodyne", "spectral-shift", "vocoder"]


class LiveApp(QMainWindow):
    """Main window for live bat audio display."""

    def __init__(
        self,
        path: Path | None = None,
        fmin: float = 15_000,
        fmax: float = 130_000,
        division: int = 10,
        method: Literal["time-expansion", "heterodyne", "spectral-shift", "vocoder"] = "spectral-shift",
        tune_hz: float = 45_000,
        highpass_hz: float = 10_000,
        gain: float = 1.0,
        fps: int = 20,
        nperseg: int = 512,
        loop: bool = False,
        mute: bool = False,
        mic: bool = False,
        mic_sr: int = 192_000,
        mic_device: int | str | None = None,
    ) -> None:
        super().__init__()

        self._fmin = fmin
        self._fmax = fmax
        self._division = division
        self._method = method
        self._tune_hz = tune_hz
        self._highpass_hz = highpass_hz
        self._gain = gain
        self._fps = fps
        self._nperseg = nperseg
        self._loop = loop
        self._mute = mute
        self._gain_db = 20 * np.log10(max(gain, 1e-6))
        self._mic_sr = mic_sr
        self._mic_device = mic_device
        self._is_mic = mic

        self._playing = False
        self._reader = None
        self._audio = None
        self._path = path

        self.resize(1200, 600)

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)

        if self._is_mic:
            self._build_ui_mic()
        else:
            self._build_ui(path)

    def _build_ui(self, path: Path) -> None:
        """Build the full UI: left sidebar + right spectrograms."""
        self._is_mic = False
        self._timer.setInterval(1000 // self._effective_fps)

        # Stop playback
        if self._playing:
            self._timer.stop()
            if self._audio:
                self._audio.stop()
            self._playing = False

        # Close previous reader
        if self._reader is not None:
            self._reader.close()

        # Open reader
        reader = ChunkReader(path, chunk_size=1, loop=self._loop)
        sr = reader.sample_rate
        reader.close()
        self._sr = sr
        self._path = path
        self._chunk_size = self._compute_chunk_size()
        self._reader = ChunkReader(path, chunk_size=self._chunk_size, loop=self._loop)

        self.setWindowTitle(f"Lepakko Live \u2014 {path.name}")

        # Central widget: horizontal layout (sidebar | spectrograms)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- Left sidebar ---
        sidebar = QVBoxLayout()
        sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Info
        self._info_label = QLabel(
            f"{path.name}\n{sr / 1000:.0f} kHz | {self._chunk_size} smp"
        )
        sidebar.addWidget(self._info_label)

        # Load / Mic
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._open_file_dialog)
        sidebar.addWidget(load_btn)

        mic_btn = QPushButton("Mic")
        mic_btn.clicked.connect(self._switch_to_mic)
        sidebar.addWidget(mic_btn)

        # Play
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_playback)
        sidebar.addWidget(self._play_btn)

        # Unprocessed sound toggle
        self._original_btn = QPushButton("Unprocessed")
        self._original_btn.setCheckable(True)
        self._original_btn.setChecked(False)
        self._original_btn.clicked.connect(self._toggle_original)
        sidebar.addWidget(self._original_btn)

        # Output toggle
        self._output_btn = QPushButton("Output")
        self._output_btn.setCheckable(True)
        self._output_btn.setChecked(False)
        self._output_btn.clicked.connect(self._toggle_output_spectro)
        sidebar.addWidget(self._output_btn)

        # Loop toggle
        self._loop_btn = QPushButton("Loop")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setChecked(self._loop)
        self._loop_btn.clicked.connect(self._toggle_loop)
        sidebar.addWidget(self._loop_btn)

        # --- Method selector ---
        sidebar.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(METHODS)
        self._method_combo.setCurrentText(self._method)
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        sidebar.addWidget(self._method_combo)

        # Division (all methods)
        sidebar.addWidget(QLabel("Division:"))
        self._division_spin = QSpinBox()
        self._division_spin.setRange(2, 50)
        self._division_spin.setValue(self._division)
        self._division_spin.valueChanged.connect(self._on_param_changed)
        sidebar.addWidget(self._division_spin)

        # Method-specific params container
        self._params_container = QWidget()
        self._params_layout = QVBoxLayout(self._params_container)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        sidebar.addWidget(self._params_container)

        self._build_method_params()

        # Spacer pushes Quit to the bottom
        sidebar.addStretch()

        # Quit
        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(self.close)
        sidebar.addWidget(quit_btn)

        main_layout.addLayout(sidebar)

        # --- Right: spectrograms ---
        self._spectro_layout = QVBoxLayout()

        self._spectrogram = SpectrogramWidget(
            sample_rate=sr, fmin=self._fmin, fmax=self._fmax, nperseg=self._nperseg,
        )
        self._spectro_layout.addWidget(self._spectrogram, stretch=1)

        # Output spectrogram — hidden by default
        self._output_spectrogram = None
        self._output_label = None
        self._build_output_spectro()

        main_layout.addLayout(self._spectro_layout, stretch=1)

        # Audio output
        self._audio = AudioOutput(
            sample_rate=sr,
            division=self._division,
            method=self._method,
            tune_hz=self._tune_hz,
            highpass_hz=self._highpass_hz,
            gain=self._gain,
            mute=self._mute,
        )

        # Original audio stream (plays raw samples at original sr)
        self._orig_stream = None
        self._orig_buffer: deque[np.ndarray] = deque()
        self._orig_lock = threading.Lock()

    @staticmethod
    def _list_input_devices() -> list[tuple[int, str, int]]:
        """Return list of (index, name, max_input_channels) for input devices."""
        import sounddevice as sd
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                result.append((i, d["name"], d["max_input_channels"]))
        return result

    def _build_ui_mic(self) -> None:
        """Build the UI for live microphone input."""
        self._is_mic = True
        self._timer.setInterval(1000 // self._effective_fps)

        # Stop playback
        if self._playing:
            self._timer.stop()
            if self._audio:
                self._audio.stop()
            self._playing = False

        # Close previous reader
        if self._reader is not None:
            self._reader.close()

        # Try to open mic; fall back to device default sample rate on failure
        try:
            self._sr = self._mic_sr
            self._chunk_size = self._compute_chunk_size()
            self._reader = MicReader(
                chunk_size=self._chunk_size,
                sample_rate=self._mic_sr,
                device=self._mic_device,
            )
        except Exception:
            # Retry with device default sample rate
            self._reader = MicReader(
                chunk_size=1,  # temporary, will recompute
                sample_rate=None,
                device=self._mic_device,
            )
            self._mic_sr = self._reader.sample_rate
            self._sr = self._mic_sr
            self._reader.close()
            self._chunk_size = self._compute_chunk_size()
            self._reader = MicReader(
                chunk_size=self._chunk_size,
                sample_rate=self._mic_sr,
                device=self._mic_device,
            )

        sr = self._sr
        self.setWindowTitle(f"Lepakko Live \u2014 Mic ({sr / 1000:.0f} kHz)")

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- Left sidebar ---
        sidebar = QVBoxLayout()
        sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Info
        self._info_label = QLabel(
            f"Mic input\n{sr / 1000:.0f} kHz | {self._chunk_size} smp"
        )
        sidebar.addWidget(self._info_label)

        # Load (switch to file mode)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._open_file_dialog)
        sidebar.addWidget(load_btn)

        # Play
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_playback)
        sidebar.addWidget(self._play_btn)

        # Unprocessed sound toggle
        self._original_btn = QPushButton("Unprocessed")
        self._original_btn.setCheckable(True)
        self._original_btn.setChecked(False)
        self._original_btn.clicked.connect(self._toggle_original)
        sidebar.addWidget(self._original_btn)

        # Output toggle
        self._output_btn = QPushButton("Output")
        self._output_btn.setCheckable(True)
        self._output_btn.setChecked(False)
        self._output_btn.clicked.connect(self._toggle_output_spectro)
        sidebar.addWidget(self._output_btn)

        # --- Device selector ---
        sidebar.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._input_devices = self._list_input_devices()
        current_idx = 0
        for i, (dev_id, name, channels) in enumerate(self._input_devices):
            label = f"{dev_id}: {name} ({channels}ch)"
            self._device_combo.addItem(label, dev_id)
            if self._mic_device is not None and dev_id == self._mic_device:
                current_idx = i
            elif self._mic_device is None and "default" in name.lower():
                current_idx = i
        self._device_combo.setCurrentIndex(current_idx)
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        sidebar.addWidget(self._device_combo)

        # Sample rate
        sidebar.addWidget(QLabel("Sample rate:"))
        self._sr_combo = QComboBox()
        for rate in [44100, 48000, 96000, 192000, 250000, 384000, 500000]:
            self._sr_combo.addItem(f"{rate / 1000:.0f} kHz", rate)
        # Select current
        sr_index = self._sr_combo.findData(self._mic_sr)
        if sr_index >= 0:
            self._sr_combo.setCurrentIndex(sr_index)
        self._sr_combo.currentIndexChanged.connect(self._on_mic_sr_changed)
        sidebar.addWidget(self._sr_combo)

        # --- Method selector ---
        sidebar.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(METHODS)
        self._method_combo.setCurrentText(self._method)
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        sidebar.addWidget(self._method_combo)

        # Division
        sidebar.addWidget(QLabel("Division:"))
        self._division_spin = QSpinBox()
        self._division_spin.setRange(2, 50)
        self._division_spin.setValue(self._division)
        self._division_spin.valueChanged.connect(self._on_param_changed)
        sidebar.addWidget(self._division_spin)

        # Method-specific params container
        self._params_container = QWidget()
        self._params_layout = QVBoxLayout(self._params_container)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        sidebar.addWidget(self._params_container)
        self._build_method_params()

        sidebar.addStretch()

        # Quit
        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(self.close)
        sidebar.addWidget(quit_btn)

        main_layout.addLayout(sidebar)

        # --- Right: spectrograms ---
        self._spectro_layout = QVBoxLayout()
        self._spectrogram = SpectrogramWidget(
            sample_rate=sr, fmin=self._fmin, fmax=self._fmax, nperseg=self._nperseg,
        )
        self._spectro_layout.addWidget(self._spectrogram, stretch=1)

        self._output_spectrogram = None
        self._output_label = None
        self._build_output_spectro()

        main_layout.addLayout(self._spectro_layout, stretch=1)

        # Audio output
        self._audio = AudioOutput(
            sample_rate=sr,
            division=self._division,
            method=self._method,
            tune_hz=self._tune_hz,
            highpass_hz=self._highpass_hz,
            gain=self._gain,
            mute=self._mute,
        )

        # Original audio stream
        self._orig_stream = None
        self._orig_buffer: deque[np.ndarray] = deque()
        self._orig_lock = threading.Lock()

    def _on_device_changed(self, combo_index: int) -> None:
        """Handle device dropdown change — query default SR and rebuild."""
        import sounddevice as sd_mod
        dev_id = self._device_combo.itemData(combo_index)
        self._mic_device = dev_id
        # Use the new device's default sample rate
        try:
            info = sd_mod.query_devices(dev_id, kind="input")
            self._mic_sr = int(info["default_samplerate"])
        except Exception:
            pass
        self._build_ui_mic()

    def _on_mic_sr_changed(self, combo_index: int) -> None:
        """Handle sample rate dropdown change — rebuild mic reader."""
        self._mic_sr = self._sr_combo.itemData(combo_index)
        self._build_ui_mic()

    @property
    def _effective_fps(self) -> int:
        """Tick rate: higher for mic mode to reduce latency."""
        return 100 if self._is_mic else self._fps

    def _compute_chunk_size(self) -> int:
        """Compute chunk size based on method and source.

        For time-expansion, use smaller chunks so audio playback at sr/division
        stays in sync with the tick rate. Mic mode uses smaller chunks (~10ms)
        for lower latency.
        """
        fps = self._effective_fps
        if self._method == "time-expansion":
            return self._sr // (fps * self._division)
        return self._sr // fps

    def _rebuild_reader(self) -> None:
        """Recreate the reader with the correct chunk size."""
        new_chunk_size = self._compute_chunk_size()
        if new_chunk_size == self._chunk_size:
            return

        was_playing = self._playing
        if was_playing:
            self._timer.stop()
            self._audio.stop()
            self._playing = False
            self._play_btn.setText("Play")

        old_reader = self._reader
        old_reader.close()

        self._chunk_size = new_chunk_size
        if self._is_mic:
            self._reader = MicReader(
                chunk_size=self._chunk_size,
                sample_rate=self._sr,
                device=self._mic_device,
            )
            self._info_label.setText(
                f"Mic input\n{self._sr / 1000:.0f} kHz | {self._chunk_size} smp"
            )
        else:
            self._reader = ChunkReader(self._path, chunk_size=self._chunk_size, loop=self._loop)
            self._info_label.setText(
                f"{self._path.name}\n{self._sr / 1000:.0f} kHz | {self._chunk_size} smp"
            )

    def _build_method_params(self) -> None:
        """Populate method-specific parameter widgets."""
        # Clear existing
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._method == "heterodyne":
            self._params_layout.addWidget(QLabel("Tune (Hz):"))
            self._tune_spin = QDoubleSpinBox()
            self._tune_spin.setRange(10_000, 130_000)
            self._tune_spin.setSingleStep(1000)
            self._tune_spin.setDecimals(0)
            self._tune_spin.setValue(self._tune_hz)
            self._tune_spin.valueChanged.connect(self._on_param_changed)
            self._params_layout.addWidget(self._tune_spin)

        elif self._method in ("spectral-shift", "vocoder"):
            self._params_layout.addWidget(QLabel("Highpass (Hz):"))
            self._highpass_spin = QDoubleSpinBox()
            self._highpass_spin.setRange(0, 50_000)
            self._highpass_spin.setSingleStep(1000)
            self._highpass_spin.setDecimals(0)
            self._highpass_spin.setValue(self._highpass_hz)
            self._highpass_spin.valueChanged.connect(self._on_param_changed)
            self._params_layout.addWidget(self._highpass_spin)

            if self._method == "spectral-shift":
                self._gain_label = QLabel(f"Gain: {self._gain_db:+.0f} dB")
                self._params_layout.addWidget(self._gain_label)
                self._gain_slider = QSlider(Qt.Orientation.Horizontal)
                self._gain_slider.setRange(-20, 40)
                self._gain_slider.setValue(int(self._gain_db))
                self._gain_slider.valueChanged.connect(self._on_gain_changed)
                self._params_layout.addWidget(self._gain_slider)

    def _build_output_spectro(self) -> None:
        """Rebuild the output spectrogram widgets for the current method/division.

        For all methods, shows a spectrogram of the processed audio output.
        Uses output_sr for the frequency axis so it reflects what you hear.
        """
        # Remove old widgets if rebuilding
        if self._output_label is not None:
            self._spectro_layout.removeWidget(self._output_label)
            self._output_label.deleteLater()
        if self._output_spectrogram is not None:
            self._spectro_layout.removeWidget(self._output_spectrogram)
            self._output_spectrogram.deleteLater()

        output_sr = self._sr // self._division
        output_fmax = output_sr / 2

        # For spectral-shift/vocoder, output_sr = input_sr but content is
        # below sr/(2*division). Use the content range as fmax.
        if self._method in ("spectral-shift", "vocoder"):
            output_sr = self._sr
            output_fmax = self._sr / (2 * self._division)

        # Scale nperseg for heterodyne (truly decimated — fewer output samples).
        # Time-expansion keeps same sample count, just relabels frequency axis.
        output_nperseg = self._nperseg
        if self._method == "heterodyne":
            output_nperseg = max(64, self._nperseg * output_sr // self._sr)

        self._output_label = QLabel(
            f"Output ({self._method}) | 0\u2013{output_fmax / 1000:.1f} kHz"
        )
        self._output_spectrogram = SpectrogramWidget(
            sample_rate=output_sr,
            fmin=0,
            fmax=output_fmax,
            nperseg=output_nperseg,
        )
        self._spectro_layout.addWidget(self._output_label)
        self._spectro_layout.addWidget(self._output_spectrogram, stretch=1)

        show = self._output_btn.isChecked()
        self._output_label.setVisible(show)
        self._output_spectrogram.setVisible(show)

    def _on_method_changed(self, method: str) -> None:
        """Handle method dropdown change."""
        self._method = method
        self._build_method_params()
        self._build_output_spectro()
        self._rebuild_reader()
        self._rebuild_audio()

    def _on_param_changed(self) -> None:
        """Read current param values and rebuild audio output."""
        old_division = self._division
        self._division = self._division_spin.value()

        if self._method == "heterodyne":
            self._tune_hz = self._tune_spin.value()
        elif self._method in ("spectral-shift", "vocoder"):
            self._highpass_hz = self._highpass_spin.value()

        # Rebuild output spectrogram if division changed (affects frequency range)
        if self._division != old_division:
            self._build_output_spectro()
            self._rebuild_reader()

        self._rebuild_audio()

    def _on_gain_changed(self, value: int) -> None:
        """Handle gain slider change (value in dB)."""
        self._gain_db = value
        self._gain = 10 ** (value / 20)
        self._gain_label.setText(f"Gain: {value:+d} dB")
        self._rebuild_audio()

    def _rebuild_audio(self) -> None:
        """Recreate AudioOutput with current settings (does not reset file position)."""
        if self._playing:
            self._timer.stop()
            self._audio.stop()
            self._playing = False
            self._play_btn.setText("Play")

        self._audio = AudioOutput(
            sample_rate=self._sr,
            division=self._division,
            method=self._method,
            tune_hz=self._tune_hz,
            highpass_hz=self._highpass_hz,
            gain=self._gain,
            mute=self._mute,
        )

    def _open_file_dialog(self) -> None:
        """Open a file dialog to choose a WAV file."""
        start_dir = str(self._path.parent) if self._path else str(Path.home())
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open WAV file", start_dir, "WAV files (*.wav *.WAV)"
        )
        if path_str:
            self._is_mic = False
            self._build_ui(Path(path_str))

    def _switch_to_mic(self) -> None:
        """Switch from file mode to microphone input."""
        self._is_mic = True
        self._build_ui_mic()

    def start(self) -> None:
        """Show the window. Playback starts when the user clicks Play."""

    def _toggle_loop(self) -> None:
        """Toggle loop mode on/off."""
        self._loop = self._loop_btn.isChecked()
        if self._reader is not None:
            self._reader.loop = self._loop

    def _start_orig_stream(self) -> None:
        """Start the original audio output stream."""
        import sounddevice as sd

        self._orig_buffer.clear()
        self._orig_stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._orig_callback,
        )
        self._orig_stream.start()

    def _stop_orig_stream(self) -> None:
        """Stop the original audio output stream."""
        if self._orig_stream is not None:
            self._orig_stream.stop()
            self._orig_stream.close()
            self._orig_stream = None

    def _orig_callback(
        self, outdata: np.ndarray, frames: int, time_info: object, status: object,
    ) -> None:
        """Sounddevice callback for original audio playback."""
        filled = 0
        while filled < frames:
            with self._orig_lock:
                if not self._orig_buffer:
                    break
                block = self._orig_buffer[0]
            needed = frames - filled
            if len(block) <= needed:
                outdata[filled : filled + len(block), 0] = block
                filled += len(block)
                with self._orig_lock:
                    self._orig_buffer.popleft()
            else:
                outdata[filled:frames, 0] = block[:needed]
                filled = frames
                with self._orig_lock:
                    self._orig_buffer[0] = block[needed:]
        if filled < frames:
            outdata[filled:, 0] = 0.0

    def _toggle_original(self) -> None:
        """Switch between original and processed audio mid-playback."""
        if not self._playing:
            return
        if self._original_btn.isChecked():
            self._audio.stop()
            self._start_orig_stream()
        else:
            self._stop_orig_stream()
            self._audio.start()

    def _toggle_playback(self) -> None:
        """Toggle between play and stop."""
        if self._playing:
            self._timer.stop()
            self._audio.stop()
            self._stop_orig_stream()
            if self._is_mic:
                self._reader.stop()
            self._play_btn.setText("Play")
            self._playing = False
        else:
            if self._is_mic:
                self._reader.start()
            elif self._reader.finished:
                # Restart from beginning if file has finished
                self._reader.seek(0)
                self._rebuild_audio()
                self._info_label.setText(
                    f"{self._path.name}\n{self._sr / 1000:.0f} kHz | {self._chunk_size} smp"
                )
            if self._original_btn.isChecked():
                self._start_orig_stream()
            else:
                self._audio.start()
            self._timer.start()
            self._play_btn.setText("Stop")
            self._playing = True

    def _toggle_output_spectro(self) -> None:
        """Show or hide the output spectrogram."""
        visible = self._output_btn.isChecked()
        if self._output_label:
            self._output_label.setVisible(visible)
        if self._output_spectrogram:
            self._output_spectrogram.setVisible(visible)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Feed a single chunk to spectrograms and audio output."""
        self._spectrogram.feed(chunk)
        output = self._audio.feed(chunk)
        if self._output_spectrogram is not None and output is not None and len(output) > 0:
            self._output_spectrogram.feed(output)

        # Feed raw chunk to original audio stream
        if self._orig_stream is not None:
            with self._orig_lock:
                self._orig_buffer.append(chunk.astype(np.float32))

    def _tick(self) -> None:
        """Read chunks and feed to spectrogram and audio."""
        if self._is_mic:
            # Drain all available chunks to stay in sync with real-time
            chunk = self._reader.read_chunk()
            while chunk is not None:
                self._process_chunk(chunk)
                chunk = self._reader.read_chunk()
            return

        chunk = self._reader.read_chunk()
        if chunk is None:
            self._timer.stop()
            self._audio.stop()
            self._stop_orig_stream()
            self._play_btn.setText("Play")
            self._playing = False
            self._info_label.setText(self._info_label.text() + "\n[FINISHED]")
            return

        self._process_chunk(chunk)

    def closeEvent(self, event: object) -> None:
        """Clean up on window close."""
        if self._playing:
            self._timer.stop()
            self._audio.stop()
            self._stop_orig_stream()
            if self._is_mic:
                self._reader.stop()
        if self._reader:
            self._reader.close()
        super().closeEvent(event)


def run_live(
    path: Path | None = None,
    fmin: float = 15_000,
    fmax: float = 130_000,
    division: int = 10,
    method: Literal["time-expansion", "heterodyne", "spectral-shift", "vocoder"] = "spectral-shift",
    tune_hz: float = 45_000,
    highpass_hz: float = 10_000,
    gain: float = 1.0,
    fps: int = 20,
    nperseg: int = 512,
    loop: bool = False,
    mute: bool = False,
    mic: bool = False,
    mic_sr: int = 192_000,
    mic_device: int | str | None = None,
) -> None:
    """Launch the live spectrogram application."""
    import sys

    app = QApplication(sys.argv)
    window = LiveApp(
        path=path,
        fmin=fmin,
        fmax=fmax,
        division=division,
        method=method,
        tune_hz=tune_hz,
        highpass_hz=highpass_hz,
        gain=gain,
        fps=fps,
        nperseg=nperseg,
        loop=loop,
        mute=mute,
        mic=mic,
        mic_sr=mic_sr,
        mic_device=mic_device,
    )
    window.show()
    window.start()
    sys.exit(app.exec())
