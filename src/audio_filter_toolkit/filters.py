"""FIR/IIR design and filtering for arrays and WAV paths."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal, TypeAlias, overload

import numpy as np
from scipy import signal
import torch
import torch.nn.functional as torch_functional
import torchaudio.functional as audio_functional

from ._validation import validate_audio_array, validate_positive_int
from .file_processing import OnError, ProcessingReport, process_audio_path
from .io import AmplitudePolicy, AudioDType, WavSubtype

AudioArray: TypeAlias = torch.Tensor | np.ndarray
FilterKind: TypeAlias = Literal["lowpass", "highpass", "bandpass", "bandstop"]
FIRKind: TypeAlias = FilterKind | Literal["custom"]
FilterMethod: TypeAlias = Literal["direct", "fft", "recursive"]
Alignment: TypeAlias = Literal["causal", "centered"]
CutoffHz: TypeAlias = float | tuple[float, float]
FIRInput: TypeAlias = Sequence[float] | np.ndarray | torch.Tensor
FIRWindow: TypeAlias = (
    Literal["hann", "hamming", "blackman", "bartlett", "boxcar"]
    | tuple[Literal["kaiser"], float]
)

_FILTER_KINDS = ("lowpass", "highpass", "bandpass", "bandstop")
_FIR_WINDOWS = ("hann", "hamming", "blackman", "bartlett", "boxcar")


@dataclass(frozen=True, slots=True)
class FIRFilter:
    """Store a linear-phase FIR design and forward-time impulse response."""

    kind: FIRKind
    sample_rate: int
    cutoff_hz: CutoffHz | None
    num_taps: int
    window: FIRWindow | None
    scale: bool | None
    taps: torch.Tensor


@dataclass(frozen=True, slots=True)
class IIRFilter:
    """Store a Butterworth IIR design as stable second-order sections."""

    kind: FilterKind
    sample_rate: int
    critical_hz: CutoffHz
    prototype: Literal["butterworth"]
    order: int
    effective_order: int
    sos: torch.Tensor


DesignedFilter: TypeAlias = FIRFilter | IIRFilter


def _validate_kind(kind: object) -> FilterKind:
    """Validate a public filter-kind string."""
    if kind not in _FILTER_KINDS:
        raise ValueError(
            "kind must be 'lowpass', 'highpass', 'bandpass', or 'bandstop', "
            f"got {kind!r}."
        )
    return kind  # type: ignore[return-value]


def _finite_frequency(value: object, *, name: str) -> float:
    """Validate one finite real-valued frequency."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    frequency = float(value)
    if not math.isfinite(frequency):
        raise ValueError(f"{name} must be finite, got {value}.")
    return frequency


def _validate_cutoff(
    cutoff: object,
    *,
    kind: FilterKind,
    sample_rate: int,
    name: str,
) -> CutoffHz:
    """Validate scalar or two-edge frequencies against Nyquist."""
    nyquist = sample_rate / 2.0
    if kind in ("lowpass", "highpass"):
        frequency = _finite_frequency(cutoff, name=name)
        if not 0.0 < frequency < nyquist:
            raise ValueError(
                f"{name} must be between 0 and Nyquist ({nyquist:g} Hz), "
                f"got {frequency:g} Hz."
            )
        return frequency

    if not isinstance(cutoff, tuple) or len(cutoff) != 2:
        raise TypeError(f"{name} must be a two-item tuple for {kind}, got {cutoff!r}.")
    lower = _finite_frequency(cutoff[0], name=f"{name}[0]")
    upper = _finite_frequency(cutoff[1], name=f"{name}[1]")
    if not 0.0 < lower < upper < nyquist:
        raise ValueError(
            f"{name} must satisfy 0 < lower < upper < {nyquist:g} Hz, "
            f"got ({lower:g}, {upper:g})."
        )
    return (lower, upper)


def _validate_window(window: object) -> FIRWindow:
    """Validate a supported FIR window specification."""
    if isinstance(window, str):
        if window not in _FIR_WINDOWS:
            raise ValueError(
                f"window must be one of {_FIR_WINDOWS} or ('kaiser', beta), "
                f"got {window!r}."
            )
        return window  # type: ignore[return-value]
    if (
        isinstance(window, tuple)
        and len(window) == 2
        and window[0] == "kaiser"
    ):
        beta = _finite_frequency(window[1], name="Kaiser beta")
        if beta < 0.0:
            raise ValueError(f"Kaiser beta must be non-negative, got {beta}.")
        return ("kaiser", beta)
    raise TypeError(
        "window must be a supported string or ('kaiser', beta), "
        f"got {window!r}."
    )


def design_fir(
    *,
    kind: FilterKind,
    sample_rate: int,
    cutoff_hz: CutoffHz,
    num_taps: int,
    window: FIRWindow,
    scale: bool = True,
) -> FIRFilter:
    """Design an odd-length linear-phase FIR filter with the window method."""
    validated_kind = _validate_kind(kind)
    validated_rate = validate_positive_int(sample_rate, name="sample_rate")
    validated_taps = validate_positive_int(num_taps, name="num_taps")
    if validated_taps < 3 or validated_taps % 2 == 0:
        raise ValueError(
            f"num_taps must be an odd integer of at least 3, got {validated_taps}."
        )
    if not isinstance(scale, bool):
        raise TypeError(f"scale must be bool, got {type(scale).__name__}.")
    validated_cutoff = _validate_cutoff(
        cutoff_hz,
        kind=validated_kind,
        sample_rate=validated_rate,
        name="cutoff_hz",
    )
    validated_window = _validate_window(window)
    coefficients = signal.firwin(
        validated_taps,
        validated_cutoff,
        window=validated_window,
        pass_zero=validated_kind,
        scale=scale,
        fs=validated_rate,
    )
    taps = torch.from_numpy(np.ascontiguousarray(coefficients, dtype=np.float64))
    return FIRFilter(
        kind=validated_kind,
        sample_rate=validated_rate,
        cutoff_hz=validated_cutoff,
        num_taps=validated_taps,
        window=validated_window,
        scale=scale,
        taps=taps,
    )


def fir_from_taps(
    taps: FIRInput,
    *,
    sample_rate: int,
) -> FIRFilter:
    """Create a custom causal FIR filter from forward-time coefficients."""
    validated_rate = validate_positive_int(sample_rate, name="sample_rate")
    if isinstance(taps, torch.Tensor):
        if taps.ndim != 1:
            raise ValueError(
                f"taps must be one-dimensional, got shape {tuple(taps.shape)}."
            )
        if taps.numel() == 0:
            raise ValueError("taps must contain at least one coefficient.")
        if taps.dtype == torch.bool or taps.is_complex():
            raise TypeError("taps must contain real numeric coefficients.")
        coefficients = taps.detach().to(device="cpu", dtype=torch.float64).clone()
    else:
        try:
            array = np.asarray(taps)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "taps must be a numeric sequence, numpy.ndarray, or torch.Tensor."
            ) from error
        if array.ndim != 1:
            raise ValueError(
                f"taps must be one-dimensional, got shape {array.shape}."
            )
        if array.size == 0:
            raise ValueError("taps must contain at least one coefficient.")
        if (
            not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError("taps must contain real numeric coefficients.")
        coefficients = torch.from_numpy(
            np.ascontiguousarray(array, dtype=np.float64)
        ).clone()

    if not torch.isfinite(coefficients).all().item():
        raise ValueError("taps must contain only finite values.")
    return FIRFilter(
        kind="custom",
        sample_rate=validated_rate,
        cutoff_hz=None,
        num_taps=coefficients.numel(),
        window=None,
        scale=None,
        taps=coefficients,
    )


def design_butterworth_iir(
    *,
    kind: FilterKind,
    sample_rate: int,
    critical_hz: CutoffHz,
    order: int,
) -> IIRFilter:
    """Design a digital Butterworth IIR filter in second-order sections."""
    validated_kind = _validate_kind(kind)
    validated_rate = validate_positive_int(sample_rate, name="sample_rate")
    validated_order = validate_positive_int(order, name="order")
    validated_critical = _validate_cutoff(
        critical_hz,
        kind=validated_kind,
        sample_rate=validated_rate,
        name="critical_hz",
    )
    coefficients = signal.butter(
        validated_order,
        validated_critical,
        btype=validated_kind,
        output="sos",
        fs=validated_rate,
    )
    sos = torch.from_numpy(np.ascontiguousarray(coefficients, dtype=np.float64))
    effective_order = (
        2 * validated_order
        if validated_kind in ("bandpass", "bandstop")
        else validated_order
    )
    return IIRFilter(
        kind=validated_kind,
        sample_rate=validated_rate,
        critical_hz=validated_critical,
        prototype="butterworth",
        order=validated_order,
        effective_order=effective_order,
        sos=sos,
    )


def _validate_designed_filter(designed_filter: object) -> DesignedFilter:
    """Validate a public filter coefficient container."""
    if isinstance(designed_filter, FIRFilter):
        validate_positive_int(designed_filter.sample_rate, name="filter sample_rate")
        validate_positive_int(designed_filter.num_taps, name="filter num_taps")
        if (
            not isinstance(designed_filter.taps, torch.Tensor)
            or designed_filter.taps.ndim != 1
            or designed_filter.taps.numel() != designed_filter.num_taps
        ):
            raise ValueError("FIR taps must be one-dimensional and match num_taps.")
        if designed_filter.taps.dtype not in (torch.float32, torch.float64):
            raise TypeError("FIR taps must use torch.float32 or torch.float64.")
        if not torch.isfinite(designed_filter.taps).all().item():
            raise ValueError("FIR taps must contain only finite values.")
        return designed_filter
    if isinstance(designed_filter, IIRFilter):
        validate_positive_int(designed_filter.sample_rate, name="filter sample_rate")
        if (
            not isinstance(designed_filter.sos, torch.Tensor)
            or designed_filter.sos.ndim != 2
            or designed_filter.sos.shape[1] != 6
            or designed_filter.sos.shape[0] == 0
        ):
            raise ValueError("IIR sos must have shape [num_sections, 6].")
        if designed_filter.sos.dtype not in (torch.float32, torch.float64):
            raise TypeError("IIR sos must use torch.float32 or torch.float64.")
        if not torch.isfinite(designed_filter.sos).all().item():
            raise ValueError("IIR sos must contain only finite values.")
        if torch.any(designed_filter.sos[:, 3] == 0).item():
            raise ValueError("every IIR section must have a non-zero a0 coefficient.")
        return designed_filter
    raise TypeError(
        "designed_filter must be FIRFilter or IIRFilter, "
        f"got {type(designed_filter).__name__}."
    )


def _validate_application_mode(
    designed_filter: DesignedFilter,
    *,
    method: object,
    alignment: object,
) -> tuple[FilterMethod, Alignment]:
    """Validate algorithm and output alignment against filter family."""
    if method not in ("direct", "fft", "recursive"):
        raise ValueError(
            "method must be 'direct', 'fft', or 'recursive', "
            f"got {method!r}."
        )
    if alignment not in ("causal", "centered"):
        raise ValueError(
            f"alignment must be 'causal' or 'centered', got {alignment!r}."
        )
    if isinstance(designed_filter, FIRFilter) and method == "recursive":
        raise ValueError("FIR filters require method='direct' or method='fft'.")
    if (
        isinstance(designed_filter, FIRFilter)
        and alignment == "centered"
        and designed_filter.num_taps % 2 == 0
    ):
        raise ValueError(
            "alignment='centered' requires an odd number of FIR taps; "
            "even-length filters have a half-sample alignment ambiguity."
        )
    if isinstance(designed_filter, IIRFilter):
        if method != "recursive":
            raise ValueError("IIR filters require method='recursive'.")
        if alignment != "causal":
            raise ValueError("IIR filters only support alignment='causal'.")
    return method, alignment  # type: ignore[return-value]


def _fir_direct(
    waveform: torch.Tensor,
    taps: torch.Tensor,
    *,
    alignment: Alignment,
) -> torch.Tensor:
    """Apply forward-time taps using conv1d's cross-correlation primitive."""
    time = waveform.shape[-1]
    kernel_size = taps.numel()
    flattened = waveform.reshape(-1, 1, time)
    if alignment == "causal":
        padded = torch_functional.pad(flattened, (kernel_size - 1, 0))
    else:
        half = (kernel_size - 1) // 2
        padded = torch_functional.pad(flattened, (half, half))

    # conv1d evaluates cross-correlation. Reversing h[k] here produces
    # y[n] = sum_k h[k] * x[n-k], which is mathematical convolution.
    kernel = taps.flip(0).reshape(1, 1, kernel_size)
    filtered = torch_functional.conv1d(padded, kernel)
    return filtered.reshape(waveform.shape)


def _fir_fft(
    waveform: torch.Tensor,
    taps: torch.Tensor,
    *,
    alignment: Alignment,
) -> torch.Tensor:
    """Apply linear FIR convolution using PyTorch real FFT primitives."""
    time = waveform.shape[-1]
    full_length = time + taps.numel() - 1
    fft_length = 1 << (full_length - 1).bit_length()
    waveform_spectrum = torch.fft.rfft(waveform, n=fft_length, dim=-1)
    filter_spectrum = torch.fft.rfft(taps, n=fft_length)
    full = torch.fft.irfft(
        waveform_spectrum * filter_spectrum,
        n=fft_length,
        dim=-1,
    )[..., :full_length]
    start = 0 if alignment == "causal" else (taps.numel() - 1) // 2
    return full[..., start : start + time]


def _iir_recursive(waveform: torch.Tensor, sos: torch.Tensor) -> torch.Tensor:
    """Apply cascaded second-order sections with zero initial state."""
    filtered = waveform
    for section in sos:
        numerator = section[:3]
        denominator = section[3:]
        filtered = audio_functional.lfilter(
            filtered,
            denominator,
            numerator,
            clamp=False,
            batching=False,
        )
    return filtered


@overload
def apply_filter(
    waveform: torch.Tensor,
    designed_filter: DesignedFilter,
    *,
    sample_rate: int,
    method: FilterMethod,
    alignment: Alignment = "causal",
) -> torch.Tensor:
    """Filter a Torch waveform while preserving its shape and device."""
    ...


@overload
def apply_filter(
    waveform: np.ndarray,
    designed_filter: DesignedFilter,
    *,
    sample_rate: int,
    method: FilterMethod,
    alignment: Alignment = "causal",
) -> np.ndarray:
    """Filter a NumPy waveform while preserving its shape and dtype."""
    ...


def apply_filter(
    waveform: AudioArray,
    designed_filter: DesignedFilter,
    *,
    sample_rate: int,
    method: FilterMethod,
    alignment: Alignment = "causal",
) -> AudioArray:
    """Filter `[time]` or `[..., time]` while preserving type and shape."""
    validated_waveform = validate_audio_array(waveform)
    validated_filter = _validate_designed_filter(designed_filter)
    validated_rate = validate_positive_int(sample_rate, name="sample_rate")
    if validated_rate != validated_filter.sample_rate:
        raise ValueError(
            "sample rate mismatch: "
            f"filter was designed for {validated_filter.sample_rate} Hz, "
            f"got {validated_rate} Hz."
        )
    validated_method, validated_alignment = _validate_application_mode(
        validated_filter,
        method=method,
        alignment=alignment,
    )

    is_numpy = isinstance(validated_waveform, np.ndarray)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(validated_waveform))
        if is_numpy
        else validated_waveform
    )
    if isinstance(validated_filter, FIRFilter):
        taps = validated_filter.taps.to(device=tensor.device, dtype=tensor.dtype)
        filtered = (
            _fir_direct(tensor, taps, alignment=validated_alignment)
            if validated_method == "direct"
            else _fir_fft(tensor, taps, alignment=validated_alignment)
        )
    else:
        sos = validated_filter.sos.to(device=tensor.device, dtype=tensor.dtype)
        filtered = _iir_recursive(tensor, sos)

    if is_numpy:
        return filtered.detach().cpu().numpy()
    return filtered


def _validate_device(device: str | torch.device) -> torch.device:
    """Validate a CPU or available CUDA processing device."""
    try:
        validated = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid torch device {device!r}.") from error
    if validated.type not in ("cpu", "cuda"):
        raise ValueError(
            f"device must select CPU or CUDA, got device type {validated.type!r}."
        )
    if validated.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA processing was requested but CUDA is unavailable.")
    return validated


def filter_path(
    source: str | Path,
    destination: str | Path,
    designed_filter: DesignedFilter,
    *,
    method: FilterMethod,
    alignment: Alignment = "causal",
    device: str | torch.device = "cpu",
    recursive: bool = False,
    overwrite: bool = False,
    on_error: OnError = "raise",
    show_progress: bool = False,
    decode_dtype: AudioDType = "float32",
    output_subtype: WavSubtype = "FLOAT",
    amplitude_policy: AmplitudePolicy = "error",
) -> ProcessingReport:
    """Filter one WAV or a directory tree and return a processing report."""
    validated_filter = _validate_designed_filter(designed_filter)
    validated_method, validated_alignment = _validate_application_mode(
        validated_filter,
        method=method,
        alignment=alignment,
    )
    validated_device = _validate_device(device)

    def transform(
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor, int]:
        """Move, filter, and return one loaded waveform for atomic saving."""
        device_waveform = waveform.to(validated_device)
        filtered = apply_filter(
            device_waveform,
            validated_filter,
            sample_rate=sample_rate,
            method=validated_method,
            alignment=validated_alignment,
        )
        return filtered.cpu(), sample_rate

    return process_audio_path(
        source,
        destination,
        transform,
        recursive=recursive,
        overwrite=overwrite,
        on_error=on_error,
        show_progress=show_progress,
        decode_dtype=decode_dtype,
        output_subtype=output_subtype,
        amplitude_policy=amplitude_policy,
    )
