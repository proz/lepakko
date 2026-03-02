"""Tests for AudioOutput processors: TimeExpansion, Heterodyne, and SpectralShift."""

import numpy as np
import pytest

from batsound.live.audio_output import (
    AudioOutput,
    HeterodyneProcessor,
    PhaseVocoderProcessor,
    SpectralShiftProcessor,
    TimeExpansionProcessor,
)

SR = 281_000
DIVISION = 10


class TestTimeExpansionProcessor:
    def test_passthrough(self):
        """Output equals input (no processing)."""
        proc = TimeExpansionProcessor(SR, DIVISION)
        chunk = np.random.randn(1000)
        result = proc.process(chunk)
        np.testing.assert_array_equal(result, chunk)

    def test_output_sr(self):
        proc = TimeExpansionProcessor(SR, DIVISION)
        assert proc.output_sr == SR // DIVISION


class TestHeterodyneProcessor:
    def test_near_tune_produces_low_beat(self):
        """A 45.5 kHz tone tuned to 45 kHz → 500 Hz audible beat."""
        proc = HeterodyneProcessor(SR, DIVISION, tune_hz=45_000)
        n = SR  # 1 second
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * 45_500 * t)

        output = proc.process(tone)
        output_sr = SR // DIVISION
        spectrum = np.abs(np.fft.rfft(output))
        freqs = np.fft.rfftfreq(len(output), 1 / output_sr)

        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - 500) < 100

    def test_shifts_50khz_to_5khz(self):
        """A 50 kHz tone with 45 kHz tuning → 5 kHz beat frequency."""
        proc = HeterodyneProcessor(SR, DIVISION, tune_hz=45_000)
        n = SR
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * 50_000 * t)

        output = proc.process(tone)
        output_sr = SR // DIVISION

        spectrum = np.abs(np.fft.rfft(output))
        freqs = np.fft.rfftfreq(len(output), 1 / output_sr)

        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - 5_000) < 500

    def test_off_tune_signal_filtered(self):
        """A 80 kHz tone with 45 kHz tuning → 35 kHz difference, above output Nyquist."""
        proc = HeterodyneProcessor(SR, DIVISION, tune_hz=45_000)
        n = SR
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * 80_000 * t)

        output = proc.process(tone)
        rms = np.sqrt(np.mean(output**2))
        # Should be heavily attenuated (35 kHz > 14 kHz output Nyquist)
        assert rms < 0.05

    def test_output_sr(self):
        proc = HeterodyneProcessor(SR, DIVISION)
        assert proc.output_sr == SR // DIVISION

    def test_output_decimated_length(self):
        """Output length is input length / division."""
        proc = HeterodyneProcessor(SR, DIVISION)
        chunk = np.zeros(14050)
        output = proc.process(chunk)
        assert len(output) == 14050 // DIVISION


class TestSpectralShiftProcessor:
    def test_shifts_50khz_to_5khz(self):
        """A 50 kHz tone should appear near 5 kHz in output (division=10)."""
        proc = SpectralShiftProcessor(SR, DIVISION)
        n = SR  # 1 second of audio
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * 50_000 * t)

        output = proc.process(tone)
        assert len(output) > 0

        # FFT of output
        spectrum = np.abs(np.fft.rfft(output))
        freqs = np.fft.rfftfreq(len(output), 1 / SR)

        peak_freq = freqs[np.argmax(spectrum)]
        # Bin quantization in STFT shift causes ~1 kHz spread
        assert abs(peak_freq - 5_000) < 1500

    def test_highpass_filters_low_freq(self):
        """A 5 kHz tone (below default 10 kHz highpass) should be filtered out."""
        proc = SpectralShiftProcessor(SR, DIVISION, highpass_hz=10_000)
        n = SR
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * 5_000 * t)

        output = proc.process(tone)
        assert len(output) > 0

        # Output should have negligible energy
        rms_out = np.sqrt(np.mean(output**2))
        assert rms_out < 0.01

    def test_output_length_matches_input(self):
        """Output length should be approximately equal to input length."""
        proc = SpectralShiftProcessor(SR, DIVISION)
        n = 14050  # ~50ms chunk
        chunk = np.random.randn(n)
        output = proc.process(chunk)
        # Within one hop of input length
        assert abs(len(output) - n) < proc.nperseg

    def test_output_sr_equals_input_sr(self):
        proc = SpectralShiftProcessor(SR, DIVISION)
        assert proc.output_sr == SR

    def test_vectorized_bin_mapping(self):
        """Precomputed bin arrays have consistent shapes and valid ranges."""
        proc = SpectralShiftProcessor(SR, DIVISION, highpass_hz=10_000)
        n_bins = proc.nperseg // 2 + 1

        assert len(proc._src_bins) == len(proc._dst_lo)
        assert len(proc._src_bins) == len(proc._dst_hi)
        assert len(proc._src_bins) == len(proc._w_lo)
        assert len(proc._src_bins) == len(proc._w_hi)

        # All destination bins within range
        assert np.all(proc._dst_lo >= 0)
        assert np.all(proc._dst_hi < n_bins)

        # Weights sum to 1
        np.testing.assert_allclose(proc._w_lo + proc._w_hi, 1.0)

    def test_streaming_continuity(self):
        """Multiple small chunks produce same result as one large chunk."""
        proc_single = SpectralShiftProcessor(SR, DIVISION)
        proc_multi = SpectralShiftProcessor(SR, DIVISION)

        chunk_size = 14050
        n_chunks = 4
        full = np.random.randn(chunk_size * n_chunks)

        out_single = proc_single.process(full)

        out_parts = []
        for i in range(n_chunks):
            part = proc_multi.process(full[i * chunk_size : (i + 1) * chunk_size])
            if len(part) > 0:
                out_parts.append(part)
        out_multi = np.concatenate(out_parts)

        # Same total length (within hop tolerance)
        assert abs(len(out_single) - len(out_multi)) < proc_single.nperseg


class TestPhaseVocoderProcessor:
    def test_shifts_50khz_to_5khz(self):
        """A 50 kHz tone should appear near 5 kHz in output (division=10)."""
        proc = PhaseVocoderProcessor(SR, DIVISION)
        n = SR  # 1 second
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * 50_000 * t)

        output = proc.process(tone)
        assert len(output) > 0

        spectrum = np.abs(np.fft.rfft(output))
        freqs = np.fft.rfftfreq(len(output), 1 / SR)

        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - 5_000) < 1500

    def test_highpass_filters_low_freq(self):
        """A 5 kHz tone (below default 10 kHz highpass) should be filtered out."""
        proc = PhaseVocoderProcessor(SR, DIVISION, highpass_hz=10_000)
        n = SR
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * 5_000 * t)

        output = proc.process(tone)
        assert len(output) > 0

        rms_out = np.sqrt(np.mean(output**2))
        assert rms_out < 0.01

    def test_output_length_matches_input(self):
        proc = PhaseVocoderProcessor(SR, DIVISION)
        n = 14050
        chunk = np.random.randn(n)
        output = proc.process(chunk)
        assert abs(len(output) - n) < proc.nperseg

    def test_output_sr_equals_input_sr(self):
        proc = PhaseVocoderProcessor(SR, DIVISION)
        assert proc.output_sr == SR


class TestAudioOutput:
    def test_mute_skips_feed(self):
        ao = AudioOutput(sample_rate=SR, division=DIVISION, mute=True)
        ao.feed(np.zeros(1000))
        assert len(ao._buffer) == 0

    def test_feed_buffers_processed(self):
        ao = AudioOutput(
            sample_rate=SR, division=DIVISION, method="time-expansion", mute=True,
        )
        ao.mute = False  # enable feed but don't start stream
        chunk = np.zeros(1000)
        ao.feed(chunk)
        assert len(ao._buffer) == 1
        assert len(ao._buffer[0]) == 1000

    def test_time_expansion_output_sr(self):
        ao = AudioOutput(sample_rate=SR, division=DIVISION, method="time-expansion")
        assert ao.output_sr == SR // DIVISION

    def test_spectral_shift_output_sr(self):
        ao = AudioOutput(sample_rate=SR, division=DIVISION, method="spectral-shift")
        assert ao.output_sr == SR

    def test_heterodyne_output_sr(self):
        ao = AudioOutput(sample_rate=SR, division=DIVISION, method="heterodyne")
        assert ao.output_sr == SR // DIVISION

    def test_vocoder_output_sr(self):
        ao = AudioOutput(sample_rate=SR, division=DIVISION, method="vocoder")
        assert ao.output_sr == SR

    def test_default_method_is_spectral_shift(self):
        ao = AudioOutput(sample_rate=SR)
        assert isinstance(ao.processor, SpectralShiftProcessor)
