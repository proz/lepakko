"""Streaming audio sources: WAV file reader and live microphone input."""

import threading
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
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


class MicReader:
    """Reads audio from a microphone in fixed-size chunks.

    Uses sounddevice InputStream with a callback that fills a thread-safe
    buffer. ``read_chunk`` pops one chunk from the buffer (blocking until
    data is available or timeout expires).

    Args:
        chunk_size: Number of samples per chunk.
        sample_rate: Requested sample rate in Hz (e.g. 192000 for TeensyBat).
        device: Sounddevice device index or name.  ``None`` uses the default.
        channels: Number of input channels (default 1).
    """

    def __init__(
        self,
        chunk_size: int,
        sample_rate: int | None = None,
        device: int | str | None = None,
        channels: int = 1,
    ) -> None:
        # Query device default sample rate if none specified
        if sample_rate is None:
            info = sd.query_devices(device, kind="input")
            sample_rate = int(info["default_samplerate"])

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = 0  # unknown for live input

        self._buffer: deque[np.ndarray] = deque(maxlen=64)
        self._lock = threading.Lock()
        self._overflow = False
        self._finished = False

        self._stream = sd.InputStream(
            samplerate=sample_rate,
            device=device,
            channels=channels,
            dtype="float32",
            blocksize=chunk_size,
            callback=self._callback,
        )

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Sounddevice callback — pushes chunks into the buffer."""
        if status.input_overflow:
            self._overflow = True
        # Take first channel if multi-channel
        data = indata[:, 0].copy() if indata.ndim > 1 else indata.ravel().copy()
        with self._lock:
            self._buffer.append(data.astype(np.float64))

    def start(self) -> None:
        """Start the audio stream."""
        self._finished = False
        self._stream.start()

    def stop(self) -> None:
        """Stop the audio stream (can be restarted)."""
        self._stream.stop()

    def read_chunk(self) -> np.ndarray | None:
        """Pop one chunk from the buffer.

        Returns:
            1D float64 array of chunk_size samples, or None if stopped
            and buffer is empty.
        """
        if self._finished:
            return None

        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
        return None

    @property
    def finished(self) -> bool:
        """Whether the reader has been stopped."""
        return self._finished

    @property
    def overflow(self) -> bool:
        """Whether an input overflow was detected since last check."""
        if self._overflow:
            self._overflow = False
            return True
        return False

    def seek(self, frame: int) -> None:
        """No-op for live input (kept for interface compatibility)."""

    def close(self) -> None:
        """Stop and close the audio stream."""
        self._finished = True
        self._stream.stop()
        self._stream.close()

    def __enter__(self) -> "MicReader":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
