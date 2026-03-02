# Lepakko

Bat echolocation analysis and acquisition software. Real-time spectrograms, frequency-divided audio playback, and offline analysis of ultrasonic bat recordings.

## Features

- **Live spectrogram** — scrolling waterfall display of ultrasonic WAV files using PyQt6 + pyqtgraph
- **Frequency division** — four methods to shift bat calls into the audible range:
  - **Time expansion** — plays audio at 1/N speed, perfect frequency mapping
  - **Heterodyne** — classic bat detector, tunable narrowband mixer
  - **Spectral shift** — STFT bin remapping, real-time wideband
  - **Phase vocoder** — phase-coherent frequency shifting
- **Offline analysis** — spectrogram generation, bandpass filtering, call parameter extraction
- **WAV I/O** — handles high sample rates (192-500 kHz) via libsndfile

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### Live spectrogram

```bash
lepakko live recording.wav
lepakko live recording.wav --method spectral-shift --division 10
lepakko live recording.wav --method heterodyne --tune 45000
lepakko live recording.wav --method time-expansion --loop
lepakko live recording.wav --mute  # spectrogram only, no audio
```

### File info

```bash
lepakko info recording.wav
lepakko info /path/to/directory/  # list all WAV files
```

### Static spectrogram

```bash
lepakko spectrogram recording.wav
lepakko spectrogram recording.wav -o output.png --fmin 15000 --fmax 130000
```

## Hardware

The companion hardware is a TeensyBat USB detector (~60-80 EUR): Teensy 4.1 + Audio Shield + MEMS mic, streaming 192 kHz audio over USB. See `hardware/` for BOM and build instructions.

## License

MIT
