"""Numerical tests for FIR/IIR design and array filtering."""

import numpy as np
import pytest
from scipy import signal
import torch

from audio_filter_toolkit import (
    IIRFilter,
    apply_filter,
    design_butterworth_iir,
    design_fir,
    fir_from_taps,
)


def make_sine(
    sample_rate: int,
    frequency: float,
    length: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Create a deterministic sine wave."""
    time = torch.arange(length, dtype=dtype) / sample_rate
    return torch.sin(2.0 * torch.pi * frequency * time)


def steady_state_gain(
    filtered: torch.Tensor,
    original: torch.Tensor,
    *,
    discard: int = 2_048,
) -> float:
    """Measure RMS gain after discarding startup transients."""
    filtered_rms = torch.sqrt(torch.mean(filtered[..., discard:].square()))
    original_rms = torch.sqrt(torch.mean(original[..., discard:].square()))
    return float(filtered_rms / original_rms)


def test_fir_taps_are_forward_time_impulse_response() -> None:
    """A causal impulse response must expose taps in their returned order."""
    designed_filter = design_fir(
        kind="lowpass",
        sample_rate=16_000,
        cutoff_hz=2_000.0,
        num_taps=65,
        window="hann",
    )
    impulse = torch.zeros(256, dtype=torch.float64)
    impulse[0] = 1.0

    direct = apply_filter(
        impulse,
        designed_filter,
        sample_rate=16_000,
        method="direct",
    )
    fft = apply_filter(
        impulse,
        designed_filter,
        sample_rate=16_000,
        method="fft",
    )

    torch.testing.assert_close(
        direct[: designed_filter.num_taps],
        designed_filter.taps,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        direct[designed_filter.num_taps :],
        torch.zeros_like(direct[designed_filter.num_taps :]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(fft, direct, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["direct", "fft"])
def test_custom_fir_two_taps_delays_by_one_sample(method: str) -> None:
    """Forward-time taps `[0, 1]` must produce one exact causal sample delay."""
    designed_filter = fir_from_taps([0, 1], sample_rate=16_000)
    waveform = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-0.5, 0.25, 0.75, -1.0]],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [0.0, -0.5, 0.25, 0.75]],
        dtype=torch.float64,
    )

    actual = apply_filter(
        waveform,
        designed_filter,
        sample_rate=16_000,
        method=method,  # type: ignore[arg-type]
    )

    assert designed_filter.kind == "custom"
    assert designed_filter.cutoff_hz is None
    assert designed_filter.window is None
    assert designed_filter.scale is None
    assert designed_filter.num_taps == 2
    assert designed_filter.taps.device.type == "cpu"
    assert designed_filter.taps.dtype == torch.float64
    torch.testing.assert_close(
        designed_filter.taps,
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)


def test_custom_fir_preserves_coefficients_without_normalization() -> None:
    """Custom coefficients must be copied exactly without scaling or reversal."""
    source_taps = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    designed_filter = fir_from_taps(source_taps, sample_rate=8_000)
    source_taps[:] = 0.0
    impulse = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    actual = apply_filter(
        impulse,
        designed_filter,
        sample_rate=8_000,
        method="direct",
    )

    np.testing.assert_array_equal(
        designed_filter.taps.numpy(),
        np.array([2.0, -1.0, 0.5], dtype=np.float64),
    )
    np.testing.assert_array_equal(actual, np.array([2.0, -1.0, 0.5, 0.0]))


def test_even_custom_fir_rejects_centered_alignment() -> None:
    """Even-length FIR filters must reject ambiguous centered alignment."""
    designed_filter = fir_from_taps([0.0, 1.0], sample_rate=16_000)

    with pytest.raises(ValueError, match="half-sample alignment ambiguity"):
        apply_filter(
            torch.ones(16),
            designed_filter,
            sample_rate=16_000,
            method="direct",
            alignment="centered",
        )


@pytest.mark.parametrize(
    ("taps", "error_type", "message"),
    [
        ([], ValueError, "at least one coefficient"),
        (1.0, ValueError, "one-dimensional"),
        ([[1.0, 0.0]], ValueError, "one-dimensional"),
        ([0.0, np.nan], ValueError, "only finite values"),
        ([True, False], TypeError, "real numeric coefficients"),
        ([1.0 + 1.0j], TypeError, "real numeric coefficients"),
    ],
)
def test_custom_fir_rejects_invalid_taps(
    taps: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Custom taps must reject empty, ranked, non-finite, and non-real inputs."""
    with pytest.raises(error_type, match=message):
        fir_from_taps(taps, sample_rate=16_000)  # type: ignore[arg-type]


@pytest.mark.parametrize("alignment", ["causal", "centered"])
def test_fir_direct_and_fft_match_for_batched_audio(alignment: str) -> None:
    """Direct and FFT linear convolution must agree for every leading signal."""
    generator = torch.Generator().manual_seed(41)
    waveform = torch.randn(2, 3, 511, generator=generator, dtype=torch.float64)
    designed_filter = design_fir(
        kind="bandstop",
        sample_rate=24_000,
        cutoff_hz=(2_000.0, 4_000.0),
        num_taps=81,
        window=("kaiser", 8.0),
    )

    direct = apply_filter(
        waveform,
        designed_filter,
        sample_rate=24_000,
        method="direct",
        alignment=alignment,  # type: ignore[arg-type]
    )
    fft = apply_filter(
        waveform,
        designed_filter,
        sample_rate=24_000,
        method="fft",
        alignment=alignment,  # type: ignore[arg-type]
    )

    assert direct.shape == waveform.shape
    assert fft.shape == waveform.shape
    torch.testing.assert_close(direct, fft, rtol=1e-11, atol=1e-11)
    for batch in range(2):
        for channel in range(3):
            independent = apply_filter(
                waveform[batch, channel],
                designed_filter,
                sample_rate=24_000,
                method="direct",
                alignment=alignment,  # type: ignore[arg-type]
            )
            torch.testing.assert_close(
                direct[batch, channel],
                independent,
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    ("kind", "cutoff", "pass_frequency", "stop_frequency"),
    [
        ("lowpass", 1_500.0, 400.0, 3_500.0),
        ("highpass", 1_500.0, 3_500.0, 400.0),
        ("bandpass", (1_000.0, 2_500.0), 1_600.0, 400.0),
        ("bandstop", (1_000.0, 2_500.0), 3_500.0, 1_600.0),
    ],
)
def test_all_fir_kinds_have_expected_numeric_gain(
    kind: str,
    cutoff: float | tuple[float, float],
    pass_frequency: float,
    stop_frequency: float,
) -> None:
    """Each FIR kind must pass and reject representative sinusoids."""
    sample_rate = 16_000
    designed_filter = design_fir(
        kind=kind,  # type: ignore[arg-type]
        sample_rate=sample_rate,
        cutoff_hz=cutoff,
        num_taps=257,
        window="hann",
    )
    pass_tone = make_sine(sample_rate, pass_frequency, sample_rate)
    stop_tone = make_sine(sample_rate, stop_frequency, sample_rate)

    passed = apply_filter(
        pass_tone,
        designed_filter,
        sample_rate=sample_rate,
        method="direct",
    )
    stopped = apply_filter(
        stop_tone,
        designed_filter,
        sample_rate=sample_rate,
        method="direct",
    )

    assert steady_state_gain(passed, pass_tone) > 0.95
    assert steady_state_gain(stopped, stop_tone) < 0.02


@pytest.mark.parametrize(
    ("kind", "critical", "pass_frequency", "stop_frequency"),
    [
        ("lowpass", 1_500.0, 400.0, 3_500.0),
        ("highpass", 1_500.0, 3_500.0, 400.0),
        ("bandpass", (1_000.0, 2_500.0), 1_600.0, 400.0),
        ("bandstop", (1_000.0, 2_500.0), 3_500.0, 1_600.0),
    ],
)
def test_all_iir_kinds_have_expected_numeric_gain(
    kind: str,
    critical: float | tuple[float, float],
    pass_frequency: float,
    stop_frequency: float,
) -> None:
    """Each IIR kind must pass and reject representative sinusoids."""
    sample_rate = 16_000
    designed_filter = design_butterworth_iir(
        kind=kind,  # type: ignore[arg-type]
        sample_rate=sample_rate,
        critical_hz=critical,
        order=6,
    )
    pass_tone = make_sine(sample_rate, pass_frequency, sample_rate)
    stop_tone = make_sine(sample_rate, stop_frequency, sample_rate)

    passed = apply_filter(
        pass_tone,
        designed_filter,
        sample_rate=sample_rate,
        method="recursive",
    )
    stopped = apply_filter(
        stop_tone,
        designed_filter,
        sample_rate=sample_rate,
        method="recursive",
    )

    assert steady_state_gain(passed, pass_tone) > 0.95
    assert steady_state_gain(stopped, stop_tone) < 0.05


def test_iir_matches_scipy_sosfilt_for_multiple_signals() -> None:
    """The recursive Torch implementation must match SciPy's SOS reference."""
    generator = np.random.default_rng(29)
    waveform = generator.normal(0.0, 0.2, size=(3, 1_024)).astype(np.float64)
    designed_filter = design_butterworth_iir(
        kind="bandpass",
        sample_rate=16_000,
        critical_hz=(500.0, 3_000.0),
        order=5,
    )

    actual = apply_filter(
        waveform,
        designed_filter,
        sample_rate=16_000,
        method="recursive",
    )
    expected = signal.sosfilt(designed_filter.sos.numpy(), waveform, axis=-1)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == waveform.shape
    assert actual.dtype == waveform.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_numpy_fir_preserves_type_dtype_and_values() -> None:
    """NumPy input must return the same type and match the Torch path."""
    generator = np.random.default_rng(7)
    waveform = generator.normal(size=(2, 257)).astype(np.float32)
    designed_filter = design_fir(
        kind="highpass",
        sample_rate=8_000,
        cutoff_hz=900.0,
        num_taps=63,
        window="blackman",
    )

    numpy_result = apply_filter(
        waveform,
        designed_filter,
        sample_rate=8_000,
        method="fft",
    )
    torch_result = apply_filter(
        torch.from_numpy(waveform),
        designed_filter,
        sample_rate=8_000,
        method="fft",
    )

    assert isinstance(numpy_result, np.ndarray)
    assert numpy_result.dtype == np.float32
    np.testing.assert_allclose(numpy_result, torch_result.numpy(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("family", ["fir", "iir"])
def test_cuda_matches_cpu_numerically(family: str) -> None:
    """CPU and CUDA paths must preserve values for FIR and IIR filters."""
    generator = torch.Generator().manual_seed(13)
    waveform = torch.randn(2, 1_024, generator=generator, dtype=torch.float32) * 0.1
    if family == "fir":
        designed_filter = design_fir(
            kind="lowpass",
            sample_rate=16_000,
            cutoff_hz=2_500.0,
            num_taps=65,
            window="hann",
        )
        method = "fft"
    else:
        designed_filter = design_butterworth_iir(
            kind="lowpass",
            sample_rate=16_000,
            critical_hz=2_500.0,
            order=4,
        )
        method = "recursive"

    cpu_result = apply_filter(
        waveform,
        designed_filter,
        sample_rate=16_000,
        method=method,  # type: ignore[arg-type]
    )
    cuda_result = apply_filter(
        waveform.cuda(),
        designed_filter,
        sample_rate=16_000,
        method=method,  # type: ignore[arg-type]
    ).cpu()

    assert cuda_result.shape == waveform.shape
    torch.testing.assert_close(cuda_result, cpu_result, rtol=2e-5, atol=2e-6)


def test_band_filter_exposes_effective_iir_order() -> None:
    """Band transformations must disclose their doubled final order."""
    designed_filter = design_butterworth_iir(
        kind="bandstop",
        sample_rate=48_000,
        critical_hz=(3_000.0, 7_000.0),
        order=5,
    )

    assert isinstance(designed_filter, IIRFilter)
    assert designed_filter.order == 5
    assert designed_filter.effective_order == 10
    assert designed_filter.sos.shape == (5, 6)


@pytest.mark.parametrize(
    ("waveform", "error_type", "message"),
    [
        ([0.0, 1.0], TypeError, "torch.Tensor or numpy.ndarray"),
        (torch.tensor(1.0), ValueError, "at least one dimension"),
        (torch.empty(0), ValueError, "at least one audio sample"),
        (torch.ones(16, dtype=torch.int16), TypeError, "torch.float32"),
        (
            np.array([0.0, np.inf], dtype=np.float32),
            ValueError,
            "only finite values",
        ),
    ],
)
def test_apply_filter_rejects_invalid_waveforms(
    waveform: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Invalid type, rank, length, dtype, and values must be diagnosed."""
    designed_filter = design_fir(
        kind="lowpass",
        sample_rate=8_000,
        cutoff_hz=1_000.0,
        num_taps=31,
        window="hann",
    )

    with pytest.raises(error_type, match=message):
        apply_filter(  # type: ignore[call-overload]
            waveform,
            designed_filter,
            sample_rate=8_000,
            method="direct",
        )


def test_design_and_application_reject_invalid_parameters() -> None:
    """Cutoffs, tap counts, sample rates, and method combinations are strict."""
    with pytest.raises(ValueError, match="odd integer"):
        design_fir(
            kind="lowpass",
            sample_rate=16_000,
            cutoff_hz=2_000.0,
            num_taps=64,
            window="hann",
        )
    with pytest.raises(ValueError, match="Nyquist"):
        design_fir(
            kind="highpass",
            sample_rate=16_000,
            cutoff_hz=8_000.0,
            num_taps=65,
            window="hann",
        )
    with pytest.raises(ValueError, match="lower < upper"):
        design_butterworth_iir(
            kind="bandpass",
            sample_rate=16_000,
            critical_hz=(3_000.0, 2_000.0),
            order=4,
        )

    fir = design_fir(
        kind="lowpass",
        sample_rate=16_000,
        cutoff_hz=2_000.0,
        num_taps=65,
        window="hann",
    )
    iir = design_butterworth_iir(
        kind="lowpass",
        sample_rate=16_000,
        critical_hz=2_000.0,
        order=4,
    )
    waveform = torch.ones(128)
    with pytest.raises(ValueError, match="designed for 16000 Hz"):
        apply_filter(
            waveform,
            fir,
            sample_rate=8_000,
            method="direct",
        )
    with pytest.raises(ValueError, match="FIR filters require"):
        apply_filter(
            waveform,
            fir,
            sample_rate=16_000,
            method="recursive",
        )
    with pytest.raises(ValueError, match="IIR filters require"):
        apply_filter(
            waveform,
            iir,
            sample_rate=16_000,
            method="fft",
        )
    with pytest.raises(ValueError, match="only support alignment='causal'"):
        apply_filter(
            waveform,
            iir,
            sample_rate=16_000,
            method="recursive",
            alignment="centered",
        )
