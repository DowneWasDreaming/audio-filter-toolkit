"""Public API for the audio filter toolkit."""

from .file_processing import ProcessingReport, discover_audio_files
from .filters import (
    Alignment,
    AudioArray,
    CutoffHz,
    DesignedFilter,
    FIRInput,
    FIRKind,
    FilterKind,
    FilterMethod,
    FIRFilter,
    FIRWindow,
    IIRFilter,
    apply_filter,
    design_butterworth_iir,
    design_fir,
    fir_from_taps,
    filter_path,
)
from .io import load_wav, save_wav

__all__ = [
    "Alignment",
    "AudioArray",
    "CutoffHz",
    "DesignedFilter",
    "FIRInput",
    "FIRKind",
    "FilterKind",
    "FilterMethod",
    "FIRFilter",
    "FIRWindow",
    "IIRFilter",
    "ProcessingReport",
    "apply_filter",
    "design_butterworth_iir",
    "design_fir",
    "fir_from_taps",
    "discover_audio_files",
    "filter_path",
    "load_wav",
    "save_wav",
]
