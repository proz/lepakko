"""Streaming WAV chunk reader for simulated real-time input."""

from pathlib import Path

import numpy as np
import soundfile as sf


class ChunkReader:
    """Reads a WAV file in fixed-size chunks, simulating real-time audio input.

    Uses sf.SoundFile for streaming reads (no full-file load).

    Args:
        path: Path to WAV file.
        chunk_size: Number of samples per chunk.
        loop: Whether to restart from beginning when file ends.
    """

    def __init__(self, path: Path, chunk_size: int, loop: bool = False) -> None:
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.loop = loop
        self._file = sf.SoundFile(str(self.path))
        self.sample_rate: int = self._file.samplerate
        self.channels: int = self._file.channels
        self.frames: int = self._file.frames
        self._finished = False

    def read_chunk(self) -> np.ndarray | None:
        """Read the next chunk of audio data.

        Returns:
            1D float64 array of chunk_size samples, or None if EOF
            (when loop=False). Zero-pads the final chunk if needed.
            Stereo files are downmixed to mono (first channel).
        """
        if self._finished:
            return None

        data = self._file.read(self.chunk_size, dtype="float64")

        # EOF: no data read
        if len(data) == 0:
            if self.loop:
                self._file.seek(0)
                data = self._file.read(self.chunk_size, dtype="float64")
                if len(data) == 0:
                    self._finished = True
                    return None
            else:
                self._finished = True
                return None

        # Stereo -> mono (first channel)
        if data.ndim > 1:
            data = data[:, 0]

        # Zero-pad final partial chunk
        if len(data) < self.chunk_size:
            if self.loop:
                # Wrap around: fill remainder from start of file
                remaining = self.chunk_size - len(data)
                self._file.seek(0)
                extra = self._file.read(remaining, dtype="float64")
                if extra.ndim > 1:
                    extra = extra[:, 0]
                data = np.concatenate([data, extra])
            else:
                padded = np.zeros(self.chunk_size, dtype=np.float64)
                padded[: len(data)] = data
                data = padded

        return data

    @property
    def finished(self) -> bool:
        """Whether the reader has reached EOF (only when loop=False)."""
        return self._finished

    def seek(self, frame: int) -> None:
        """Seek to a specific frame position."""
        self._file.seek(frame)
        self._finished = False

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()

    def __enter__(self) -> "ChunkReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
