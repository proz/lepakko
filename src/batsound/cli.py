"""CLI entry point for BatSound."""

from pathlib import Path

import click

from batsound import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """BatSound - Bat echolocation analysis and acquisition."""


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def info(path: Path) -> None:
    """Show WAV file metadata."""
    from batsound.io import load_wav, wav_info

    if path.is_file():
        meta = wav_info(path)
        click.echo(f"File:        {meta['path']}")
        click.echo(f"Sample rate: {meta['sample_rate'] / 1000:.1f} kHz")
        click.echo(f"Channels:    {meta['channels']}")
        click.echo(f"Frames:      {meta['frames']}")
        click.echo(f"Duration:    {meta['duration']:.3f} s")
        click.echo(f"Subtype:     {meta['subtype']}")
    elif path.is_dir():
        wav_files = sorted(path.glob("*.wav")) + sorted(path.glob("*.WAV"))
        if not wav_files:
            click.echo("No WAV files found.")
            return
        for f in wav_files:
            meta = wav_info(f)
            click.echo(
                f"{f.name:40s}  {meta['sample_rate']/1000:.0f} kHz  "
                f"{meta['duration']:.2f}s  {meta['channels']}ch"
            )


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Save to file instead of displaying.")
@click.option("--fmin", type=float, default=15000, help="Min frequency in Hz (default: 15000).")
@click.option("--fmax", type=float, default=130000, help="Max frequency in Hz (default: 130000).")
def spectrogram(path: Path, output: Path | None, fmin: float, fmax: float) -> None:
    """Generate spectrogram from a WAV file."""
    from batsound.io import load_wav
    from batsound.analysis import compute_spectrogram
    from batsound.visualization import plot_spectrogram

    data, sample_rate = load_wav(path)
    freqs, times, sxx = compute_spectrogram(data, sample_rate)
    plot_spectrogram(freqs, times, sxx, fmin=fmin, fmax=fmax, title=path.name, output=output)


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--fmin", type=float, default=15000, help="Min frequency in Hz (default: 15000).")
@click.option("--fmax", type=float, default=130000, help="Max frequency in Hz (default: 130000).")
@click.option("--division", type=int, default=10, help="Frequency division factor (default: 10).")
@click.option(
    "--method",
    type=click.Choice(["time-expansion", "heterodyne", "spectral-shift", "vocoder"]),
    default="spectral-shift",
    help="Audio output method (default: spectral-shift).",
)
@click.option("--tune", type=float, default=45000, help="Heterodyne tuning frequency in Hz (default: 45000).")
@click.option("--highpass", type=float, default=10000, help="Highpass cutoff in Hz for spectral-shift (default: 10000).")
@click.option("--gain", type=float, default=1.0, help="Output gain for spectral-shift (default: 1.0).")
@click.option("--fps", type=int, default=20, help="Display update rate in Hz (default: 20).")
@click.option("--nperseg", type=int, default=512, help="STFT window size in samples (default: 512).")
@click.option("--loop", is_flag=True, help="Loop the WAV file.")
@click.option("--mute", is_flag=True, help="Disable audio output.")
def live(
    path: Path,
    fmin: float,
    fmax: float,
    division: int,
    method: str,
    tune: float,
    highpass: float,
    gain: float,
    fps: int,
    nperseg: int,
    loop: bool,
    mute: bool,
) -> None:
    """Live scrolling spectrogram with audible frequency-divided output."""
    from batsound.live.app import run_live

    run_live(
        path=path,
        fmin=fmin,
        fmax=fmax,
        division=division,
        method=method,
        tune_hz=tune,
        highpass_hz=highpass,
        gain=gain,
        fps=fps,
        nperseg=nperseg,
        loop=loop,
        mute=mute,
    )
