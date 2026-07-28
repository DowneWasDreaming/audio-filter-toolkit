"""Generate a WAV, design FIR/IIR filters, and write filtered results."""

import argparse
from pathlib import Path

import torch

from audio_filter_toolkit import (
    apply_filter,
    design_butterworth_iir,
    design_fir,
    fir_from_taps,
    filter_path,
    save_wav,
)


def main() -> None:
    """Run a deterministic tensor and file filtering example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("filter_example_output"),
        help="Directory for generated FLOAT WAV files.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Torch device used by tensor and path filtering.",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")

    sample_rate = 48_000
    length = sample_rate
    time = torch.arange(length, dtype=torch.float64) / sample_rate
    low_tone = 0.6 * torch.sin(2.0 * torch.pi * 500.0 * time)
    high_tone = 0.25 * torch.sin(2.0 * torch.pi * 8_000.0 * time)
    waveform = torch.stack((low_tone + high_tone, low_tone - high_tone))

    fir = design_fir(
        kind="lowpass",
        sample_rate=sample_rate,
        cutoff_hz=2_000.0,
        num_taps=257,
        window=("kaiser", 8.0),
    )
    iir = design_butterworth_iir(
        kind="highpass",
        sample_rate=sample_rate,
        critical_hz=4_000.0,
        order=6,
    )
    one_sample_delay = fir_from_taps([0.0, 1.0], sample_rate=sample_rate)

    device_waveform = waveform.to(args.device)
    fir_direct = apply_filter(
        device_waveform,
        fir,
        sample_rate=sample_rate,
        method="direct",
        alignment="causal",
    )
    fir_fft = apply_filter(
        device_waveform,
        fir,
        sample_rate=sample_rate,
        method="fft",
        alignment="causal",
    )
    iir_recursive = apply_filter(
        device_waveform,
        iir,
        sample_rate=sample_rate,
        method="recursive",
    )
    delayed = apply_filter(
        device_waveform,
        one_sample_delay,
        sample_rate=sample_rate,
        method="direct",
    )
    direct_fft_error = (fir_direct - fir_fft).abs().max().item()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "input.wav"
    save_wav(input_path, waveform, sample_rate, subtype="DOUBLE", overwrite=True)
    save_wav(
        args.output_dir / "fir_direct.wav",
        fir_direct,
        sample_rate,
        subtype="DOUBLE",
        overwrite=True,
    )
    save_wav(
        args.output_dir / "iir_recursive.wav",
        iir_recursive,
        sample_rate,
        subtype="DOUBLE",
        overwrite=True,
    )
    save_wav(
        args.output_dir / "one_sample_delay.wav",
        delayed,
        sample_rate,
        subtype="DOUBLE",
        overwrite=True,
    )

    report = filter_path(
        input_path,
        args.output_dir / "fir_fft_path.wav",
        fir,
        method="fft",
        alignment="centered",
        device=args.device,
        overwrite=True,
        decode_dtype="float64",
        output_subtype="DOUBLE",
    )

    print(f"FIR taps shape: {tuple(fir.taps.shape)}")
    print(f"IIR SOS shape: {tuple(iir.sos.shape)}")
    print(f"custom delay taps: {one_sample_delay.taps.tolist()}")
    print(f"direct/FFT maximum error: {direct_fft_error:.3e}")
    print(f"path outputs: {[str(path) for path in report.outputs]}")


if __name__ == "__main__":
    main()
