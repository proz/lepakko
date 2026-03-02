"""Tests for ChunkReader streaming WAV reader."""

import numpy as np
import soundfile as sf
import pytest

from lepakko.live.stream import ChunkReader


@pytest.fixture
def mono_wav(tmp_path):
    """Create a short mono WAV file (1000 samples at 281 kHz)."""
    path = tmp_path / "mono.wav"
    data = np.sin(2 * np.pi * 50000 * np.arange(1000) / 281000)
    sf.write(str(path), data, 281000, subtype="PCM_16")
    return path


@pytest.fixture
def stereo_wav(tmp_path):
    """Create a short stereo WAV file (1000 samples at 281 kHz)."""
    path = tmp_path / "stereo.wav"
    t = np.arange(1000) / 281000
    left = np.sin(2 * np.pi * 50000 * t)
    right = np.sin(2 * np.pi * 30000 * t)
    data = np.column_stack([left, right])
    sf.write(str(path), data, 281000, subtype="PCM_16")
    return path


class TestChunkReader:
    def test_basic_chunk_read(self, mono_wav):
        with ChunkReader(mono_wav, chunk_size=200) as reader:
            assert reader.sample_rate == 281000
            assert reader.channels == 1
            assert reader.frames == 1000

            chunk = reader.read_chunk()
            assert chunk is not None
            assert chunk.shape == (200,)
            assert chunk.dtype == np.float64

    def test_reads_all_data(self, mono_wav):
        """Reading all chunks should cover the entire file."""
        with ChunkReader(mono_wav, chunk_size=200) as reader:
            chunks = []
            while True:
                chunk = reader.read_chunk()
                if chunk is None:
                    break
                chunks.append(chunk)
            # 1000 samples / 200 = 5 exact chunks
            assert len(chunks) == 5
            assert all(c.shape == (200,) for c in chunks)

    def test_zero_pad_last_chunk(self, mono_wav):
        """Final chunk is zero-padded when file length isn't divisible."""
        with ChunkReader(mono_wav, chunk_size=300) as reader:
            chunks = []
            while True:
                chunk = reader.read_chunk()
                if chunk is None:
                    break
                chunks.append(chunk)
            # 1000 / 300 = 3 full chunks + 1 padded (100 samples + 200 zeros)
            assert len(chunks) == 4
            assert chunks[-1].shape == (300,)
            # Last 200 samples should be zero
            assert np.all(chunks[-1][100:] == 0.0)

    def test_finished_property(self, mono_wav):
        with ChunkReader(mono_wav, chunk_size=1000) as reader:
            assert not reader.finished
            reader.read_chunk()
            assert not reader.finished
            result = reader.read_chunk()
            assert result is None
            assert reader.finished

    def test_loop_mode(self, mono_wav):
        """Loop mode wraps back to beginning."""
        with ChunkReader(mono_wav, chunk_size=400, loop=True) as reader:
            chunks = []
            for _ in range(5):
                chunk = reader.read_chunk()
                assert chunk is not None
                assert chunk.shape == (400,)
                chunks.append(chunk)
            # Should never finish in loop mode
            assert not reader.finished

    def test_loop_wraps_partial_chunk(self, mono_wav):
        """In loop mode, partial chunks at EOF are filled from file start."""
        with ChunkReader(mono_wav, chunk_size=300, loop=True) as reader:
            chunks = []
            for _ in range(4):
                chunk = reader.read_chunk()
                chunks.append(chunk)
            # All chunks should be full (no zero padding in loop mode)
            assert all(c.shape == (300,) for c in chunks)
            # 4th chunk wraps: last 200 samples should come from file start
            # (not zeros)
            assert not np.all(chunks[3][100:] == 0.0)

    def test_stereo_downmix(self, stereo_wav):
        """Stereo files should be downmixed to mono (first channel)."""
        with ChunkReader(stereo_wav, chunk_size=500) as reader:
            assert reader.channels == 2
            chunk = reader.read_chunk()
            assert chunk.ndim == 1
            assert chunk.shape == (500,)

    def test_seek(self, mono_wav):
        with ChunkReader(mono_wav, chunk_size=200) as reader:
            reader.read_chunk()  # read first 200
            reader.seek(0)
            chunk = reader.read_chunk()
            assert chunk is not None
            assert chunk.shape == (200,)

    def test_context_manager(self, mono_wav):
        """Context manager properly closes file."""
        reader = ChunkReader(mono_wav, chunk_size=200)
        reader.close()
        # After close, file operations should fail
        with pytest.raises(Exception):
            reader.read_chunk()
