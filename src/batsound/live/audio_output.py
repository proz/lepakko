"""Audio output with frequency division for audible bat call playback.

Four methods:
- time-expansion: plays all samples at sr/division rate (audio is division× slower)
- heterodyne: multiply by tunable oscillator, lowpass, decimate (real-time, narrowband)
- spectral-shift: STFT → highpass → remap frequency bins → IFFT (real-time, wideband)
- vocoder: phase vocoder with proper phase tracking (real-time, wideband, phase-coherent)
"""

import threading
from collections import deque
from typing import Literal

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class TimeExpansionProcessor:
    """Passthrough processor — sounddevice plays at reduced sample rate.

    Output sample rate is input_sr / division, so a 50 kHz tone plays as
    50/division kHz. Audio plays division× slower than real-time.
    """

    def __init__(self, sample_rate: int, division: int) -> None:
        self.output_sr = sample_rate // division

    def process(self, chunk: np.ndarray) -> np.ndarray:
        return chunk


class HeterodyneProcessor:
    """Mixes input with a tunable oscillator to shift a narrow band to audible.

    Classic bat detector method: multiply by cos(2*pi*tune_hz*t), lowpass
    filter, decimate. Listens to a ~14 kHz band centered on tune_hz.
    Real-time, but only one frequency band at a time.

    Args:
        sample_rate: Input sample rate in Hz.
        division: Decimation factor (default 10).
        tune_hz: Center frequency of the oscillator in Hz (default 45000).
    """

    def __init__(
        self,
        sample_rate: int,
        division: int,
        tune_hz: float = 45_000,
    ) -> None:
        self.sample_rate = sample_rate
        self.division = division
        self.output_sr = sample_rate // division
        self.tune_hz = tune_hz

        # Phase accumulator for continuous oscillator across chunks
        self._phase_index = 0

        # Lowpass at output Nyquist for anti-aliasing before decimation
        nyquist = sample_rate / 2
        cutoff = self.output_sr / 2
        sos = butter(4, cutoff / nyquist, btype="low", output="sos")
        self._sos = sos
        self._zi = sosfilt_zi(sos)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        n = len(chunk)
        t = (np.arange(n) + self._phase_index) / self.sample_rate
        self._phase_index += n

        # Mix down: shift tune_hz to DC
        mixed = chunk * np.cos(2 * np.pi * self.tune_hz * t)

        # Lowpass filter (stateful across chunks)
        filtered, self._zi = sosfilt(self._sos, mixed, zi=self._zi)

        # Decimate (take every Nth sample)
        return filtered[:: self.division]


class SpectralShiftProcessor:
    """Shifts ultrasonic content down by division factor in real-time.

    Uses STFT with overlap-add: for each frame, frequency bins above the
    highpass cutoff are remapped to bin k/division with linear interpolation
    for anti-aliasing. Output runs at the original sample rate.
    """

    def __init__(
        self,
        sample_rate: int,
        division: int,
        nperseg: int = 512,
        highpass_hz: float = 10_000,
        gain: float = 1.0,
    ) -> None:
        self.output_sr = sample_rate
        self.division = division
        self.gain = gain
        self.nperseg = nperseg
        self.hop = nperseg // 4
        self.noverlap = nperseg - self.hop

        self._window = np.hanning(nperseg)
        n_bins = nperseg // 2 + 1

        # Precompute bin mapping arrays for vectorized processing
        freq_per_bin = sample_rate / nperseg
        highpass_bin = int(np.ceil(highpass_hz / freq_per_bin))

        src_bins = np.arange(highpass_bin, n_bins)
        targets = src_bins / division

        # Only keep source bins whose target falls within output range
        valid = targets < n_bins
        src_bins = src_bins[valid]
        targets = targets[valid]

        self._src_bins = src_bins
        self._dst_lo = np.floor(targets).astype(int)
        self._dst_hi = np.minimum(np.ceil(targets).astype(int), n_bins - 1)
        frac = targets - self._dst_lo
        self._w_lo = 1.0 - frac
        self._w_hi = frac

        # Streaming state
        self._in_buf = np.array([], dtype=np.float64)
        self._ola_buf = np.zeros(self.noverlap, dtype=np.float64)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        self._in_buf = np.concatenate([self._in_buf, chunk])

        output_parts: list[np.ndarray] = []

        while len(self._in_buf) >= self.nperseg:
            frame_in = self._in_buf[: self.nperseg]
            self._in_buf = self._in_buf[self.hop :]

            # STFT: window + rfft
            spectrum = np.fft.rfft(frame_in * self._window)

            # Remap bins with gain
            shifted = np.zeros_like(spectrum)
            scaled = spectrum[self._src_bins] * self.gain
            np.add.at(shifted, self._dst_lo, scaled * self._w_lo)
            np.add.at(shifted, self._dst_hi, scaled * self._w_hi)

            # IFFT
            frame_out = np.fft.irfft(shifted, n=self.nperseg)

            # Overlap-add
            frame_out[: self.noverlap] += self._ola_buf
            output_parts.append(frame_out[: self.hop])
            self._ola_buf = frame_out[self.hop :].copy()

        if output_parts:
            return np.concatenate(output_parts)
        return np.array([], dtype=np.float64)


class PhaseVocoderProcessor:
    """Shifts ultrasonic content down with proper phase tracking.

    Standard phase vocoder: decomposes each STFT frame into magnitude +
    instantaneous frequency, remaps both to lower bins (k/division),
    and accumulates output phase from the shifted frequencies. Produces
    cleaner output than SpectralShiftProcessor for sustained tones.

    Args:
        sample_rate: Input sample rate in Hz.
        division: Frequency division factor.
        nperseg: STFT window size in samples.
        highpass_hz: Ignore input below this frequency.
    """

    def __init__(
        self,
        sample_rate: int,
        division: int,
        nperseg: int = 512,
        highpass_hz: float = 10_000,
    ) -> None:
        self.output_sr = sample_rate
        self.division = division
        self.nperseg = nperseg
        self.hop = nperseg // 4
        self.noverlap = nperseg - self.hop

        self._ana_window = np.hanning(nperseg)
        self._syn_window = np.hanning(nperseg)
        n_bins = nperseg // 2 + 1

        # COLA normalization: sum of (ana * syn window) across overlapping frames
        # With Hann*Hann at 75% overlap, 4 frames overlap at any point
        w2 = self._ana_window * self._syn_window
        n_frames = nperseg // self.hop  # 4 for 75% overlap
        cola_sum = sum(w2[i * self.hop : (i + 1) * self.hop].sum() for i in range(n_frames))
        self._cola_norm = cola_sum / self.hop  # average (should be constant)

        # Expected phase advance per bin per hop
        self._expected_advance = 2 * np.pi * np.arange(n_bins) * self.hop / nperseg

        # Precompute bin mapping (same as SpectralShiftProcessor)
        freq_per_bin = sample_rate / nperseg
        highpass_bin = int(np.ceil(highpass_hz / freq_per_bin))

        src_bins = np.arange(highpass_bin, n_bins)
        targets = src_bins / division

        valid = targets < n_bins
        src_bins = src_bins[valid]
        targets = targets[valid]

        self._src_bins = src_bins
        self._dst_lo = np.floor(targets).astype(int)
        self._dst_hi = np.minimum(np.ceil(targets).astype(int), n_bins - 1)
        frac = targets - self._dst_lo
        self._w_lo = 1.0 - frac
        self._w_hi = frac

        # Magnitude threshold: ignore bins below noise floor to prevent phase drift
        self._mag_threshold = 1e-6

        # Phase state
        self._prev_phase = np.zeros(n_bins)
        self._synth_phase = np.zeros(n_bins)
        self._n_bins = n_bins

        # Streaming state
        self._in_buf = np.array([], dtype=np.float64)
        self._ola_buf = np.zeros(self.noverlap, dtype=np.float64)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        self._in_buf = np.concatenate([self._in_buf, chunk])

        output_parts: list[np.ndarray] = []

        while len(self._in_buf) >= self.nperseg:
            frame_in = self._in_buf[: self.nperseg]
            self._in_buf = self._in_buf[self.hop :]

            # Analysis: window + FFT
            spectrum = np.fft.rfft(frame_in * self._ana_window)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Instantaneous frequency via phase difference
            dphi = phase - self._prev_phase
            self._prev_phase = phase.copy()
            dev = dphi - self._expected_advance
            dev -= 2 * np.pi * np.round(dev / (2 * np.pi))
            inst_freq = self._expected_advance + dev

            # Shift magnitudes to lower bins
            out_mag = np.zeros(self._n_bins)
            np.add.at(out_mag, self._dst_lo, magnitude[self._src_bins] * self._w_lo)
            np.add.at(out_mag, self._dst_hi, magnitude[self._src_bins] * self._w_hi)

            # Shift frequencies (magnitude-weighted for proper averaging)
            out_freq_num = np.zeros(self._n_bins)
            scaled_freq = inst_freq[self._src_bins] / self.division
            np.add.at(
                out_freq_num, self._dst_lo,
                scaled_freq * magnitude[self._src_bins] * self._w_lo,
            )
            np.add.at(
                out_freq_num, self._dst_hi,
                scaled_freq * magnitude[self._src_bins] * self._w_hi,
            )

            # Average frequency weighted by magnitude (threshold to prevent noise drift)
            out_freq = np.zeros(self._n_bins)
            mask = out_mag > self._mag_threshold
            out_freq[mask] = out_freq_num[mask] / out_mag[mask]

            # Accumulate synthesis phase, wrap to prevent numerical drift
            self._synth_phase += out_freq
            self._synth_phase = np.remainder(self._synth_phase + np.pi, 2 * np.pi) - np.pi

            # Zero DC bin to prevent low-frequency rumble
            out_mag[0] = 0.0

            # Reconstruct spectrum with new magnitudes and accumulated phases
            out_spectrum = out_mag * np.exp(1j * self._synth_phase)

            # IFFT + synthesis window (smooths frame-to-frame phase discontinuities)
            frame_out = np.fft.irfft(out_spectrum, n=self.nperseg)
            frame_out *= self._syn_window / self._cola_norm

            # Overlap-add
            frame_out[: self.noverlap] += self._ola_buf
            output_parts.append(frame_out[: self.hop])
            self._ola_buf = frame_out[self.hop :].copy()

        if output_parts:
            return np.concatenate(output_parts)
        return np.array([], dtype=np.float64)


class AudioOutput:
    """Processes ultrasonic audio and plays it through speakers.

    Args:
        sample_rate: Input sample rate in Hz (e.g. 281000).
        division: Frequency division factor (default 10).
        method: 'time-expansion', 'heterodyne', 'spectral-shift', or 'vocoder'.
        tune_hz: Oscillator frequency for heterodyne in Hz.
        highpass_hz: Highpass cutoff for spectral-shift/vocoder in Hz.
        gain: Output gain for spectral-shift (default 1.0).
        blocksize: Sounddevice output block size.
        mute: If True, skip playback entirely.
    """

    def __init__(
        self,
        sample_rate: int,
        division: int = 10,
        method: Literal["time-expansion", "heterodyne", "spectral-shift", "vocoder"] = "spectral-shift",
        tune_hz: float = 45_000,
        highpass_hz: float = 10_000,
        gain: float = 1.0,
        blocksize: int = 1024,
        mute: bool = False,
    ) -> None:
        self.mute = mute
        self.blocksize = blocksize

        if method == "time-expansion":
            self.processor = TimeExpansionProcessor(sample_rate, division)
        elif method == "heterodyne":
            self.processor = HeterodyneProcessor(
                sample_rate, division, tune_hz=tune_hz,
            )
        elif method == "vocoder":
            self.processor = PhaseVocoderProcessor(
                sample_rate, division, highpass_hz=highpass_hz,
            )
        else:
            self.processor = SpectralShiftProcessor(
                sample_rate, division, highpass_hz=highpass_hz, gain=gain,
            )

        self.output_sr = self.processor.output_sr

        self._buffer: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._stream = None

    def start(self) -> None:
        """Start the audio output stream."""
        if self.mute:
            return

        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.output_sr,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        """Feed a chunk of high-sample-rate audio for processing and playback.

        Always processes the chunk and returns the output (for visualization).
        Only buffers for playback when not muted.
        """
        output = self.processor.process(chunk)
        if len(output) > 0:
            if not self.mute:
                with self._lock:
                    self._buffer.append(output.astype(np.float32))
            return output
        return None

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Sounddevice output callback. Fills outdata from buffer."""
        filled = 0
        while filled < frames:
            with self._lock:
                if not self._buffer:
                    break
                block = self._buffer[0]

            needed = frames - filled
            if len(block) <= needed:
                outdata[filled : filled + len(block), 0] = block
                filled += len(block)
                with self._lock:
                    self._buffer.popleft()
            else:
                outdata[filled:frames, 0] = block[:needed]
                filled = frames
                with self._lock:
                    self._buffer[0] = block[needed:]

        # Fill any remaining with silence
        if filled < frames:
            outdata[filled:, 0] = 0.0

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
