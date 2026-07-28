"""WAV input and atomic output helpers."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, TypeAlias

import numpy as np
import soundfile as sf
import torch

from ._validation import validate_path, validate_positive_int

AudioDType: TypeAlias = Literal["float32", "float64"]
WavSubtype: TypeAlias = Literal["PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"]
AmplitudePolicy: TypeAlias = Literal["error", "clip"]


def load_wav(
    path: str | Path,
    *,
    expected_sample_rate: int | None = None,
    dtype: AudioDType = "float32",
) -> tuple[torch.Tensor, int]:
    """Load a WAV as `[channel, time]` and return its sample rate."""
    wav_path = validate_path(path, name="path")
    if wav_path.suffix.lower() != ".wav":
        raise ValueError(f"path must point to a .wav file, got {wav_path.name!r}.")
    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}.")
    if expected_sample_rate is not None:
        validate_positive_int(expected_sample_rate, name="expected_sample_rate")

    samples, sample_rate = sf.read(wav_path, dtype=dtype, always_2d=True)
    sample_rate = validate_positive_int(sample_rate, name="file sample rate")
    if samples.shape[0] == 0:
        raise ValueError(f"WAV file contains no audio samples: {wav_path}.")
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise ValueError(
            "sample rate mismatch: "
            f"expected {expected_sample_rate} Hz, got {sample_rate} Hz."
        )

    return torch.from_numpy(np.ascontiguousarray(samples.T)), sample_rate


def _waveform_to_numpy(waveform: object) -> np.ndarray:
    """Validate file-compatible audio and return channel-first NumPy data."""
    if isinstance(waveform, torch.Tensor):
        if waveform.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "waveform dtype must be torch.float32 or torch.float64, "
                f"got {waveform.dtype}."
            )
        samples = waveform.detach().cpu().numpy()
    elif isinstance(waveform, np.ndarray):
        if waveform.dtype not in (np.float32, np.float64):
            raise TypeError(
                "waveform dtype must be numpy.float32 or numpy.float64, "
                f"got {waveform.dtype}."
            )
        samples = waveform
    else:
        raise TypeError(
            "waveform must be a torch.Tensor or numpy.ndarray, "
            f"got {type(waveform).__name__}."
        )

    if samples.ndim not in (1, 2):
        raise ValueError(
            "waveform must have shape [time] or [channel, time], "
            f"got {samples.shape}."
        )
    if samples.size == 0 or samples.shape[-1] == 0:
        raise ValueError("waveform must contain at least one audio sample.")
    if not np.isfinite(samples).all():
        raise ValueError("waveform must contain only finite values.")
    return np.ascontiguousarray(samples)


def save_wav(
    path: str | Path,
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int,
    *,
    subtype: WavSubtype = "FLOAT",
    overwrite: bool = False,
    amplitude_policy: AmplitudePolicy = "error",
) -> Path:
    """Atomically save `[time]` or `[channel, time]` floating audio as WAV."""
    wav_path = validate_path(path, name="path")
    if wav_path.suffix.lower() != ".wav":
        raise ValueError(f"path must end with .wav, got {wav_path.name!r}.")
    sample_rate = validate_positive_int(sample_rate, name="sample_rate")
    if not isinstance(overwrite, bool):
        raise TypeError(f"overwrite must be bool, got {type(overwrite).__name__}.")
    if amplitude_policy not in ("error", "clip"):
        raise ValueError(
            "amplitude_policy must be 'error' or 'clip', "
            f"got {amplitude_policy!r}."
        )

    available_subtypes = sf.available_subtypes("WAV")
    if subtype not in available_subtypes:
        raise ValueError(
            f"unsupported WAV subtype {subtype!r}; "
            f"available values are {sorted(available_subtypes)}."
        )
    if wav_path.exists():
        if wav_path.is_dir():
            raise IsADirectoryError(f"output WAV path is a directory: {wav_path}.")
        if not overwrite:
            raise FileExistsError(f"output WAV already exists: {wav_path}.")

    channel_first = _waveform_to_numpy(waveform)
    if subtype.startswith("PCM"):
        peak_amplitude = float(np.max(np.abs(channel_first)))
        if peak_amplitude > 1.0:
            if amplitude_policy == "error":
                raise ValueError(
                    "PCM output requires waveform amplitudes within [-1, 1], "
                    f"got peak amplitude {peak_amplitude:.6g}; "
                    "use amplitude_policy='clip' to clip explicitly."
                )
            channel_first = np.clip(channel_first, -1.0, 1.0)

    soundfile_samples = (
        channel_first if channel_first.ndim == 1 else channel_first.T
    )
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            prefix=f".{wav_path.stem}.",
            suffix=".tmp.wav",
            dir=wav_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        sf.write(
            temporary_path,
            soundfile_samples,
            sample_rate,
            subtype=subtype,
            format="WAV",
        )
        os.replace(temporary_path, wav_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return wav_path
