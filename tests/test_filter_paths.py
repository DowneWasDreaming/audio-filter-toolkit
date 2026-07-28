"""Numerical tests for WAV and directory filtering."""

from pathlib import Path

import numpy as np
import pytest
import torch

from audio_filter_toolkit import (
    apply_filter,
    design_butterworth_iir,
    design_fir,
    filter_path,
    load_wav,
    save_wav,
)


def test_filter_path_matches_array_filter_numerically(tmp_path: Path) -> None:
    """A file result must contain the same samples as direct array filtering."""
    sample_rate = 16_000
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    generator = torch.Generator().manual_seed(17)
    waveform = torch.randn(2, 1_024, generator=generator) * 0.1
    designed_filter = design_fir(
        kind="lowpass",
        sample_rate=sample_rate,
        cutoff_hz=2_000.0,
        num_taps=65,
        window="hann",
    )
    save_wav(input_path, waveform, sample_rate, subtype="FLOAT")

    report = filter_path(
        input_path,
        output_path,
        designed_filter,
        method="fft",
        output_subtype="FLOAT",
    )
    actual, actual_rate = load_wav(output_path)
    expected = apply_filter(
        waveform,
        designed_filter,
        sample_rate=sample_rate,
        method="fft",
    )

    assert report.outputs == (output_path,)
    assert report.errors == ()
    assert actual_rate == sample_rate
    assert actual.shape == waveform.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_directory_filtering_preserves_tree_and_reports_bad_files(
    tmp_path: Path,
) -> None:
    """Recursive jobs must write numeric outputs and report invalid WAV data."""
    sample_rate = 8_000
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    nested = source / "nested"
    nested.mkdir(parents=True)
    first_path = source / "first.wav"
    second_path = nested / "second.WAV"
    broken_path = source / "broken.wav"
    first = torch.linspace(-0.4, 0.4, 257)
    second = torch.sin(torch.arange(257) * 0.1).unsqueeze(0)
    save_wav(first_path, first, sample_rate, subtype="FLOAT")
    save_wav(second_path, second, sample_rate, subtype="FLOAT")
    broken_path.write_bytes(b"not a WAV")
    designed_filter = design_butterworth_iir(
        kind="highpass",
        sample_rate=sample_rate,
        critical_hz=500.0,
        order=4,
    )

    report = filter_path(
        source,
        destination,
        designed_filter,
        method="recursive",
        recursive=True,
        on_error="skip",
        output_subtype="FLOAT",
    )
    first_output = destination / "first.wav"
    second_output = destination / "nested" / "second.wav"
    actual_first, _ = load_wav(first_output)
    actual_second, _ = load_wav(second_output)
    expected_first = apply_filter(
        first.unsqueeze(0),
        designed_filter,
        sample_rate=sample_rate,
        method="recursive",
    )
    expected_second = apply_filter(
        second,
        designed_filter,
        sample_rate=sample_rate,
        method="recursive",
    )

    assert report.outputs == (first_output, second_output)
    assert len(report.errors) == 1
    assert report.errors[0][0] == broken_path
    assert "LibsndfileError" in report.errors[0][1]
    torch.testing.assert_close(actual_first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(actual_second, expected_second, rtol=0, atol=0)


def test_in_place_directory_filtering_requires_consent_and_is_numeric(
    tmp_path: Path,
) -> None:
    """In-place filtering must require overwrite and atomically replace samples."""
    sample_rate = 8_000
    source = tmp_path / "in-place"
    source.mkdir()
    audio_path = source / "audio.wav"
    waveform = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    save_wav(audio_path, waveform, sample_rate, subtype="DOUBLE")
    designed_filter = design_fir(
        kind="lowpass",
        sample_rate=sample_rate,
        cutoff_hz=1_000.0,
        num_taps=5,
        window="boxcar",
    )
    expected = apply_filter(
        waveform,
        designed_filter,
        sample_rate=sample_rate,
        method="direct",
    )

    with pytest.raises(ValueError, match="requires overwrite=True"):
        filter_path(
            source,
            source,
            designed_filter,
            method="direct",
        )

    report = filter_path(
        source,
        source,
        designed_filter,
        method="direct",
        overwrite=True,
        decode_dtype="float64",
        output_subtype="DOUBLE",
    )
    actual, actual_rate = load_wav(audio_path, dtype="float64")

    assert report.outputs == (audio_path,)
    assert report.errors == ()
    assert actual_rate == sample_rate
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_directory_sample_rate_mismatch_is_reported_without_output(
    tmp_path: Path,
) -> None:
    """A filter must never silently process a file at another sample rate."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    wrong_rate_path = source / "wrong-rate.wav"
    save_wav(wrong_rate_path, torch.zeros(64), 8_000, subtype="FLOAT")
    designed_filter = design_fir(
        kind="lowpass",
        sample_rate=16_000,
        cutoff_hz=2_000.0,
        num_taps=31,
        window="hann",
    )

    report = filter_path(
        source,
        destination,
        designed_filter,
        method="direct",
        on_error="skip",
    )

    assert report.outputs == ()
    assert len(report.errors) == 1
    assert report.errors[0][0] == wrong_rate_path
    assert "designed for 16000 Hz, got 8000 Hz" in report.errors[0][1]
    assert not (destination / "wrong-rate.wav").exists()


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    """A separate existing output file must not be replaced without consent."""
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    waveform = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
    save_wav(input_path, waveform, 8_000, subtype="FLOAT")
    save_wav(output_path, np.zeros(64, dtype=np.float32), 8_000, subtype="FLOAT")
    designed_filter = design_fir(
        kind="highpass",
        sample_rate=8_000,
        cutoff_hz=1_000.0,
        num_taps=31,
        window="hann",
    )

    with pytest.raises(RuntimeError, match="FileExistsError"):
        filter_path(
            input_path,
            output_path,
            designed_filter,
            method="direct",
        )
    unchanged, _ = load_wav(output_path)
    torch.testing.assert_close(unchanged, torch.zeros_like(unchanged), rtol=0, atol=0)
