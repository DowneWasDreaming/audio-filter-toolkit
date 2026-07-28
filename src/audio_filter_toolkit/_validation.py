"""Shared validation helpers for public audio APIs."""

from pathlib import Path

import numpy as np
import torch


def validate_positive_int(value: object, *, name: str) -> int:
    """Validate and return a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def validate_path(path: object, *, name: str) -> Path:
    """Validate and return a filesystem path."""
    if not isinstance(path, (str, Path)):
        raise TypeError(f"{name} must be str or Path, got {type(path).__name__}.")
    return Path(path)


def validate_audio_array(
    waveform: object,
) -> torch.Tensor | np.ndarray:
    """Validate finite floating audio shaped `[time]` or `[..., time]`."""
    if isinstance(waveform, torch.Tensor):
        if waveform.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "waveform dtype must be torch.float32 or torch.float64, "
                f"got {waveform.dtype}."
            )
        if waveform.ndim < 1:
            raise ValueError("waveform must have at least one dimension for time.")
        if waveform.numel() == 0 or waveform.shape[-1] == 0:
            raise ValueError("waveform must contain at least one audio sample.")
        if not torch.isfinite(waveform).all().item():
            raise ValueError("waveform must contain only finite values.")
        return waveform

    if isinstance(waveform, np.ndarray):
        if waveform.dtype not in (np.float32, np.float64):
            raise TypeError(
                "waveform dtype must be numpy.float32 or numpy.float64, "
                f"got {waveform.dtype}."
            )
        if waveform.ndim < 1:
            raise ValueError("waveform must have at least one dimension for time.")
        if waveform.size == 0 or waveform.shape[-1] == 0:
            raise ValueError("waveform must contain at least one audio sample.")
        if not np.isfinite(waveform).all():
            raise ValueError("waveform must contain only finite values.")
        return waveform

    raise TypeError(
        "waveform must be a torch.Tensor or numpy.ndarray, "
        f"got {type(waveform).__name__}."
    )
