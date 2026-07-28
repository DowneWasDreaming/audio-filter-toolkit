"""Reusable single-file and directory WAV processing workflows."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import torch
from tqdm.auto import tqdm

from ._validation import validate_path
from .io import AmplitudePolicy, AudioDType, WavSubtype, load_wav, save_wav

OnError: TypeAlias = Literal["raise", "skip"]
AudioTransform: TypeAlias = Callable[
    [torch.Tensor, int], tuple[torch.Tensor, int]
]


@dataclass(frozen=True, slots=True)
class ProcessingReport:
    """Record successful output paths and per-file error messages."""

    outputs: tuple[Path, ...]
    errors: tuple[tuple[Path, str], ...]


def _normalize_extensions(extensions: Sequence[str]) -> frozenset[str]:
    """Validate and normalize filename extensions."""
    normalized: set[str] = set()
    for extension in extensions:
        if not isinstance(extension, str):
            raise TypeError(
                "extensions must contain only strings, "
                f"got {type(extension).__name__}."
            )
        if not extension.startswith(".") or len(extension) < 2:
            raise ValueError(
                f"extension must begin with '.' and contain a suffix, got {extension!r}."
            )
        normalized.add(extension.lower())
    if not normalized:
        raise ValueError("extensions must contain at least one filename extension.")
    return frozenset(normalized)


def discover_audio_files(
    directory: str | Path,
    *,
    extensions: Sequence[str] = (".wav",),
    recursive: bool = False,
) -> list[Path]:
    """Return deterministically sorted audio files from a directory."""
    directory_path = validate_path(directory, name="directory")
    if not isinstance(recursive, bool):
        raise TypeError(f"recursive must be bool, got {type(recursive).__name__}.")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"audio directory does not exist: {directory_path}.")

    normalized_extensions = _normalize_extensions(extensions)
    candidates = directory_path.rglob("*") if recursive else directory_path.glob("*")
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in normalized_extensions
        ),
        key=lambda path: path.as_posix().lower(),
    )


def process_audio_path(
    source: str | Path,
    destination: str | Path,
    transform: AudioTransform,
    *,
    recursive: bool = False,
    overwrite: bool = False,
    on_error: OnError = "raise",
    show_progress: bool = False,
    decode_dtype: AudioDType = "float32",
    output_subtype: WavSubtype = "FLOAT",
    amplitude_policy: AmplitudePolicy = "error",
) -> ProcessingReport:
    """Transform one WAV or a directory tree, including explicit in-place jobs."""
    source_path = validate_path(source, name="source")
    destination_path = validate_path(destination, name="destination")
    if not callable(transform):
        raise TypeError(f"transform must be callable, got {type(transform).__name__}.")
    for value, name in (
        (recursive, "recursive"),
        (overwrite, "overwrite"),
        (show_progress, "show_progress"),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool, got {type(value).__name__}.")
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}.")
    if decode_dtype not in ("float32", "float64"):
        raise ValueError(
            f"decode_dtype must be 'float32' or 'float64', got {decode_dtype!r}."
        )
    if output_subtype not in ("PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"):
        raise ValueError(f"unsupported WAV output_subtype {output_subtype!r}.")
    if amplitude_policy not in ("error", "clip"):
        raise ValueError(
            "amplitude_policy must be 'error' or 'clip', "
            f"got {amplitude_policy!r}."
        )

    if source_path.is_file():
        if source_path.suffix.lower() != ".wav":
            raise ValueError(f"unsupported input extension: {source_path.suffix!r}.")
        if destination_path.exists() and destination_path.is_dir():
            raise IsADirectoryError(
                "destination must be an output file when source is a file: "
                f"{destination_path}."
            )
        if destination_path.suffix.lower() != ".wav":
            raise ValueError(
                f"destination must end with '.wav', got {destination_path.name!r}."
            )
        same_location = source_path.resolve() == destination_path.resolve()
        if same_location and not overwrite:
            raise ValueError("in-place file processing requires overwrite=True.")
        jobs = [(source_path, destination_path)]
        is_directory_job = False
    elif source_path.is_dir():
        if destination_path.exists() and not destination_path.is_dir():
            raise NotADirectoryError(
                "destination must be a directory when source is a directory: "
                f"{destination_path}."
            )
        resolved_source = source_path.resolve()
        resolved_destination = destination_path.resolve()
        same_location = resolved_destination == resolved_source
        if same_location and not overwrite:
            raise ValueError("in-place directory processing requires overwrite=True.")
        if not same_location and resolved_destination.is_relative_to(resolved_source):
            raise ValueError(
                "destination directory must not be a descendant of the source directory."
            )

        input_paths = discover_audio_files(source_path, recursive=recursive)
        if not input_paths:
            raise ValueError(f"no WAV files found in {source_path}.")
        jobs = [
            (
                input_path,
                destination_path / input_path.relative_to(source_path).with_suffix(".wav"),
            )
            for input_path in input_paths
        ]
        is_directory_job = True
    else:
        raise FileNotFoundError(f"source path does not exist: {source_path}.")

    output_owners: dict[str, Path] = {}
    for input_path, output_path in jobs:
        output_key = str(output_path.resolve()).casefold()
        if output_key in output_owners:
            raise ValueError(
                "multiple input files map to the same output path: "
                f"{output_owners[output_key]} and {input_path} -> {output_path}."
            )
        output_owners[output_key] = input_path

    outputs: list[Path] = []
    errors: list[tuple[Path, str]] = []
    iterator = (
        tqdm(jobs, desc="Filtering audio", unit="file")
        if is_directory_job and show_progress
        else jobs
    )
    for input_path, output_path in iterator:
        try:
            waveform, sample_rate = load_wav(input_path, dtype=decode_dtype)
            transformed, transformed_rate = transform(waveform, sample_rate)
            outputs.append(
                save_wav(
                    output_path,
                    transformed,
                    transformed_rate,
                    subtype=output_subtype,
                    overwrite=overwrite,
                    amplitude_policy=amplitude_policy,
                )
            )
        except Exception as error:
            if on_error == "raise":
                raise RuntimeError(
                    f"failed to process audio file {input_path}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            errors.append((input_path, f"{type(error).__name__}: {error}"))

    return ProcessingReport(outputs=tuple(outputs), errors=tuple(errors))
