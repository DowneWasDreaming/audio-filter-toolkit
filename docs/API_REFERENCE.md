# API reference

This document describes every public name exported by `audio_filter_toolkit`.
It covers purpose, call form, parameters, return values, dimensions, numerical
behavior, and common errors.

## Shared conventions

### Waveform dimensions

Numerical filtering accepts:

- `[time]` for one signal.
- `[..., time]` for any number of independent leading dimensions.

Examples of valid shapes include `[time]`, `[channel, time]`, and
`[batch, channel, time]`. Filtering is always performed independently along
the final time dimension. The returned array has exactly the same shape.

`load_wav` always returns `[channel, time]`, including `[1, time]` for mono.
`save_wav` accepts only `[time]` or `[channel, time]` because a WAV file has no
batch dimension.

### Array type, dtype, and device

- Numerical waveforms must be finite `float32` or `float64`.
- A NumPy input returns a NumPy array with the same dtype and runs on CPU.
- A Torch input returns a Torch tensor with the same dtype and device.
- Torch filtering supports CPU and CUDA.
- Filter coefficients are stored canonically as CPU `torch.float64` tensors
  and are copied to the waveform dtype/device when applied.
- No filtering function clamps amplitudes.

### Sampling rates and frequencies

Sample rates are positive integers in hertz. A filter stores the sample rate
used during design or custom construction. `apply_filter` and `filter_path`
require an exact match and never resample implicitly.

All cutoff and critical frequencies use hertz and must lie strictly between
zero and the Nyquist frequency. Band edges must also be strictly increasing.

## Public type aliases

The aliases below are exported for type annotations and editor completion.
They do not perform runtime conversion.

### `AudioArray`

```python
AudioArray = torch.Tensor | numpy.ndarray
```

Represents a floating waveform shaped `[time]` or `[..., time]`.

### `FilterKind`

```python
FilterKind = Literal[
    "lowpass",
    "highpass",
    "bandpass",
    "bandstop",
]
```

Selects which frequency region a designed FIR or IIR filter passes.

### `FIRKind`

```python
FIRKind = FilterKind | Literal["custom"]
```

Describes the origin/type of an `FIRFilter`. Filters returned by
`fir_from_taps` use `"custom"`.

### `FilterMethod`

```python
FilterMethod = Literal["direct", "fft", "recursive"]
```

| Value | Meaning | Valid filters |
| --- | --- | --- |
| `"direct"` | Linear FIR convolution through `torch.nn.functional.conv1d`. | FIR |
| `"fft"` | Full linear FIR convolution through `torch.fft.rfft/irfft`. | FIR |
| `"recursive"` | Causal SOS difference-equation filtering. | IIR |

The method must be supplied explicitly. The toolkit never chooses an algorithm
from signal length.

### `Alignment`

```python
Alignment = Literal["causal", "centered"]
```

| Value | Meaning |
| --- | --- |
| `"causal"` | Uses zero history and returns the first input-length samples of linear convolution. |
| `"centered"` | Compensates the integer group delay of an odd-length FIR by taking a centered input-length slice. |

IIR filters accept only `"causal"`. Even-length FIR filters reject
`"centered"` because their nominal linear-phase delay lies between samples.

### `CutoffHz`

```python
CutoffHz = float | tuple[float, float]
```

A scalar is used for low-pass/high-pass filters. A `(lower, upper)` pair is
used for band-pass/band-stop filters.

### `FIRInput`

```python
FIRInput = Sequence[float] | numpy.ndarray | torch.Tensor
```

One-dimensional custom FIR coefficients accepted by `fir_from_taps`.

### `FIRWindow`

```python
FIRWindow = (
    Literal["hann", "hamming", "blackman", "bartlett", "boxcar"]
    | tuple[Literal["kaiser"], float]
)
```

Supported window specifications for `design_fir`. Kaiser beta must be finite
and non-negative.

### `DesignedFilter`

```python
DesignedFilter = FIRFilter | IIRFilter
```

The common filter-object type accepted by `apply_filter` and `filter_path`.

## Filter result objects

### `FIRFilter`

Stores an FIR impulse response and its design metadata.

```python
from audio_filter_toolkit import FIRFilter
```

| Field | Type | Meaning |
| --- | --- | --- |
| `kind` | `FIRKind` | Designed response kind or `"custom"`. |
| `sample_rate` | `int` | Required application sample rate in Hz. |
| `cutoff_hz` | `CutoffHz \| None` | Window-design cutoff; `None` for custom taps. |
| `num_taps` | `int` | Number of FIR coefficients. |
| `window` | `FIRWindow \| None` | Design window; `None` for custom taps. |
| `scale` | `bool \| None` | Whether `firwin` scaled the response; `None` for custom taps. |
| `taps` | `torch.Tensor` | CPU float64 coefficients shaped `[num_taps]`. |

`taps[k]` is the forward-time impulse response `h[k]`. It is not reversed.
For causal filtering:

```text
y[n] = sum_k h[k] * x[n-k]
```

The direct implementation reverses the taps only while adapting mathematical
convolution to PyTorch's cross-correlation primitive. The FFT implementation
uses the forward-time taps directly.

Normally create this object with `design_fir` or `fir_from_taps` instead of
constructing it manually.

### `IIRFilter`

Stores a Butterworth filter as cascaded second-order sections.

```python
from audio_filter_toolkit import IIRFilter
```

| Field | Type | Meaning |
| --- | --- | --- |
| `kind` | `FilterKind` | Low-pass, high-pass, band-pass, or band-stop. |
| `sample_rate` | `int` | Required application sample rate in Hz. |
| `critical_hz` | `CutoffHz` | Butterworth -3 dB critical frequency/frequencies. |
| `prototype` | `Literal["butterworth"]` | IIR prototype used for design. |
| `order` | `int` | Requested low/high-pass prototype order. |
| `effective_order` | `int` | Final order; band transforms produce `2 * order`. |
| `sos` | `torch.Tensor` | CPU float64 matrix shaped `[num_sections, 6]`. |

Each SOS row uses:

```text
[b0, b1, b2, a0, a1, a2]
```

Normally create this object with `design_butterworth_iir`.

### `ProcessingReport`

Immutable summary returned by `filter_path`.

```python
from audio_filter_toolkit import ProcessingReport
```

| Field | Type | Meaning |
| --- | --- | --- |
| `outputs` | `tuple[pathlib.Path, ...]` | Successfully written output WAV paths in processing order. |
| `errors` | `tuple[tuple[pathlib.Path, str], ...]` | `(input_path, diagnostic)` entries retained by `on_error="skip"`. |

An empty `errors` tuple means every selected input succeeded.

## Filter design

### `design_fir`

Designs an odd-length linear-phase FIR filter using SciPy's window method.

```python
def design_fir(
    *,
    kind: FilterKind,
    sample_rate: int,
    cutoff_hz: CutoffHz,
    num_taps: int,
    window: FIRWindow,
    scale: bool = True,
) -> FIRFilter
```

Example:

```python
from audio_filter_toolkit import design_fir

fir = design_fir(
    kind="bandpass",
    sample_rate=48_000,
    cutoff_hz=(300.0, 3_400.0),
    num_taps=257,
    window=("kaiser", 8.0),
    scale=True,
)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `kind` | Response type. Low/high-pass require one cutoff; band filters require two. |
| `sample_rate` | Positive design/application rate in Hz. |
| `cutoff_hz` | FIR half-amplitude cutoff, approximately -6 dB. |
| `num_taps` | Odd coefficient count of at least 3. Larger values generally sharpen transitions but increase cost and delay. |
| `window` | Supported window string or `("kaiser", beta)`. |
| `scale` | If true, normalize the response to unity at SciPy's reference pass-band frequency. |

Returns:

An `FIRFilter` whose forward-time `taps` are shaped `[num_taps]`.

Common errors:

- `TypeError` for non-integer rates/tap counts or invalid parameter types.
- `ValueError` for an unsupported kind/window, even or too-small `num_taps`,
  invalid band order, or frequencies outside `(0, Nyquist)`.

### `fir_from_taps`

Creates a custom FIR filter without redesigning or normalizing the supplied
impulse response.

```python
def fir_from_taps(
    taps: FIRInput,
    *,
    sample_rate: int,
) -> FIRFilter
```

Example—one-sample delay:

```python
from audio_filter_toolkit import fir_from_taps

delay = fir_from_taps([0.0, 1.0], sample_rate=16_000)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `taps` | Non-empty one-dimensional real numeric coefficients in forward-time order. |
| `sample_rate` | Positive rate required when the filter is applied. |

Returns:

An `FIRFilter` with `kind="custom"`, `cutoff_hz=None`, `window=None`,
`scale=None`, and an independent CPU float64 copy of the taps. Changing the
original NumPy/Torch input later does not mutate the filter.

Notes and errors:

- Integer coefficient sequences such as `[0, 1]` are accepted and converted.
- Boolean, complex, empty, non-finite, scalar, and multi-dimensional taps are
  rejected.
- A Torch coefficient tensor is detached before it is copied; custom filter
  coefficients are not trainable parameters.
- Even-length filters support causal direct/FFT filtering but reject centered
  alignment.

### `design_butterworth_iir`

Designs a digital Butterworth IIR filter in SOS form.

```python
def design_butterworth_iir(
    *,
    kind: FilterKind,
    sample_rate: int,
    critical_hz: CutoffHz,
    order: int,
) -> IIRFilter
```

Example:

```python
from audio_filter_toolkit import design_butterworth_iir

iir = design_butterworth_iir(
    kind="bandstop",
    sample_rate=48_000,
    critical_hz=(5_000.0, 7_000.0),
    order=6,
)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `kind` | Response type. |
| `sample_rate` | Positive design/application rate in Hz. |
| `critical_hz` | Butterworth -3 dB critical frequency/frequencies. |
| `order` | Positive prototype order. Band-pass and band-stop final order is doubled. |

Returns:

An `IIRFilter` containing float64 SOS coefficients. SOS is used instead of one
high-order numerator/denominator pair to reduce numerical sensitivity.

Common errors:

- `TypeError` for non-integer rate/order or invalid frequency types.
- `ValueError` for unsupported kinds, non-positive order/rate, reversed band
  edges, or critical frequencies outside `(0, Nyquist)`.

## Array filtering

### `apply_filter`

Applies a designed/custom filter to NumPy or Torch audio while preserving array
type, dtype, device, and shape.

```python
@overload
def apply_filter(
    waveform: torch.Tensor,
    designed_filter: DesignedFilter,
    *,
    sample_rate: int,
    method: FilterMethod,
    alignment: Alignment = "causal",
) -> torch.Tensor

@overload
def apply_filter(
    waveform: numpy.ndarray,
    designed_filter: DesignedFilter,
    *,
    sample_rate: int,
    method: FilterMethod,
    alignment: Alignment = "causal",
) -> numpy.ndarray
```

FIR examples:

```python
direct = apply_filter(
    waveform,
    fir,
    sample_rate=48_000,
    method="direct",
    alignment="causal",
)

fft = apply_filter(
    waveform,
    fir,
    sample_rate=48_000,
    method="fft",
    alignment="centered",
)
```

IIR example:

```python
recursive = apply_filter(
    waveform,
    iir,
    sample_rate=48_000,
    method="recursive",
)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `waveform` | Finite float32/float64 NumPy or Torch audio shaped `[time]` or `[..., time]`. |
| `designed_filter` | An `FIRFilter` or `IIRFilter`. |
| `sample_rate` | Must exactly match `designed_filter.sample_rate`. |
| `method` | FIR: `"direct"`/`"fft"`; IIR: `"recursive"`. Required explicitly. |
| `alignment` | FIR causal or centered output slicing. IIR requires `"causal"`. |

Returns:

- Torch input: a Torch tensor with the same shape, dtype, and device.
- NumPy input: a CPU NumPy array with the same shape and dtype.

Numerical behavior:

- Direct FIR uses zero padding and mathematical linear convolution.
- FFT FIR zero-pads to compute full linear rather than circular convolution.
- Centered FIR is offline/non-causal and uses zero extension at both ends.
- Recursive IIR applies each SOS causally with zero initial state and
  `clamp=False`.
- Output length always equals the input time length.

Common errors:

- `TypeError` for unsupported array/filter types or integer/complex waveforms.
- `ValueError` for scalar/empty/non-finite audio, rate mismatch, invalid method,
  incompatible method/filter pairs, centered IIR, or centered even-length FIR.

Memory boundary:

The FFT method transforms the complete input and filter at once. It can use
substantial CPU/GPU memory for long files because this version does not yet
implement overlap-add blocks.

## WAV I/O

### `load_wav`

Reads one `.wav` file with SoundFile and returns channel-first Torch audio.

```python
def load_wav(
    path: str | pathlib.Path,
    *,
    expected_sample_rate: int | None = None,
    dtype: Literal["float32", "float64"] = "float32",
) -> tuple[torch.Tensor, int]
```

Example:

```python
from audio_filter_toolkit import load_wav

waveform, sample_rate = load_wav(
    "input.wav",
    expected_sample_rate=48_000,
    dtype="float32",
)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `path` | Existing input path whose suffix is `.wav` case-insensitively. |
| `expected_sample_rate` | Optional exact required rate; no resampling occurs. |
| `dtype` | SoundFile decode precision. |

Returns:

`(waveform, sample_rate)`, where `waveform` is a CPU Torch tensor shaped
`[channel, time]` and `sample_rate` is a positive integer. Mono returns
`[1, time]`.

Common errors:

- `ValueError` for a non-WAV suffix, invalid dtype, empty audio, or rate
  mismatch.
- SoundFile/file-system exceptions for missing, unreadable, or invalid WAV
  content.

### `save_wav`

Atomically writes one floating waveform to WAV.

```python
def save_wav(
    path: str | pathlib.Path,
    waveform: torch.Tensor | numpy.ndarray,
    sample_rate: int,
    *,
    subtype: Literal[
        "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"
    ] = "FLOAT",
    overwrite: bool = False,
    amplitude_policy: Literal["error", "clip"] = "error",
) -> pathlib.Path
```

Example:

```python
from audio_filter_toolkit import save_wav

saved = save_wav(
    "filtered.wav",
    waveform,
    48_000,
    subtype="FLOAT",
    overwrite=False,
)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `path` | Output `.wav` path. Missing parent directories are created. |
| `waveform` | Finite float32/float64 `[time]` or `[channel, time]` array. |
| `sample_rate` | Positive integer rate written to the WAV header. |
| `subtype` | WAV sample encoding. FLOAT/DOUBLE preserve out-of-range floats. |
| `overwrite` | Whether an existing output file may be atomically replaced. |
| `amplitude_policy` | For PCM only: reject out-of-range values or explicitly clip to `[-1, 1]`. |

Returns:

The destination as `pathlib.Path`.

Write behavior:

- Torch inputs are detached and copied to CPU before SoundFile writing.
- Channel-first two-dimensional audio is transposed to SoundFile's
  `[time, channel]` convention.
- Data is written to a temporary WAV beside the destination, then replaced
  atomically with `os.replace`.

Common errors:

- `TypeError` for unsupported array/dtype or non-boolean `overwrite`.
- `ValueError` for bad shape, empty/non-finite samples, unsupported subtype,
  invalid sample rate, or PCM overflow under `amplitude_policy="error"`.
- `FileExistsError` when the target exists and `overwrite=False`.

## File discovery and path filtering

### `discover_audio_files`

Finds audio files in one directory with deterministic ordering.

```python
def discover_audio_files(
    directory: str | pathlib.Path,
    *,
    extensions: Sequence[str] = (".wav",),
    recursive: bool = False,
) -> list[pathlib.Path]
```

Example:

```python
from audio_filter_toolkit import discover_audio_files

files = discover_audio_files(
    "input_wavs",
    extensions=(".wav",),
    recursive=True,
)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `directory` | Existing directory to search. |
| `extensions` | Non-empty suffix collection; every value must begin with `"."`. Matching is case-insensitive. |
| `recursive` | If true, search all descendants; otherwise only direct children. |

Returns:

A list of matching file paths sorted case-insensitively by POSIX-style path.
The default public workflow discovers only WAV files.

Common errors:

- `TypeError` for invalid path/extension/recursive types.
- `ValueError` for an empty or malformed extension collection.
- `NotADirectoryError` when the directory does not exist or is not a directory.

### `filter_path`

Filters one WAV or a directory tree and writes WAV output.

```python
def filter_path(
    source: str | pathlib.Path,
    destination: str | pathlib.Path,
    designed_filter: DesignedFilter,
    *,
    method: FilterMethod,
    alignment: Alignment = "causal",
    device: str | torch.device = "cpu",
    recursive: bool = False,
    overwrite: bool = False,
    on_error: Literal["raise", "skip"] = "raise",
    show_progress: bool = False,
    decode_dtype: Literal["float32", "float64"] = "float32",
    output_subtype: Literal[
        "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"
    ] = "FLOAT",
    amplitude_policy: Literal["error", "clip"] = "error",
) -> ProcessingReport
```

Single-file example:

```python
report = filter_path(
    "input.wav",
    "filtered.wav",
    fir,
    method="fft",
    alignment="centered",
    device="cuda",
    output_subtype="FLOAT",
)
```

Directory example:

```python
report = filter_path(
    "input_wavs",
    "filtered_wavs",
    fir,
    method="direct",
    recursive=True,
    overwrite=False,
    on_error="skip",
    show_progress=True,
)

for input_path, diagnostic in report.errors:
    print(input_path, diagnostic)
```

Explicit in-place example:

```python
report = filter_path(
    "input_wavs",
    "input_wavs",
    fir,
    method="direct",
    recursive=True,
    overwrite=True,
)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `source` | One WAV or an existing directory containing WAV files. |
| `destination` | Output WAV for file input, or output directory for directory input. |
| `designed_filter` | FIR/IIR filter to apply to every selected file. |
| `method` | Required algorithm; compatibility matches `apply_filter`. |
| `alignment` | FIR output alignment; IIR requires causal. |
| `device` | `"cpu"`, `"cuda"`, `"cuda:N"`, or matching `torch.device`. |
| `recursive` | Include descendant WAV files for a directory source. |
| `overwrite` | Allow existing outputs. Required when source and destination are exactly the same path. |
| `on_error` | `"raise"` stops at the first failed file; `"skip"` records it and continues. |
| `show_progress` | Show a tqdm progress bar for directory jobs. |
| `decode_dtype` | SoundFile decode/intermediate precision. |
| `output_subtype` | WAV encoding passed to atomic saving. |
| `amplitude_policy` | PCM overflow behavior passed to atomic saving. |

Returns:

A `ProcessingReport` listing successfully written paths and skipped errors.
Filtering does not change the sample rate.

Path rules and boundaries:

- File source requires a `.wav` file destination.
- Directory output preserves relative subdirectories.
- Inputs are enumerated before any output is written.
- A destination directory inside the source is rejected to prevent output
  rediscovery.
- Exact in-place file/directory processing requires `overwrite=True`.
- Every file is loaded completely into memory and processed independently.
- Writes remain atomic, including in-place replacement.
- A file whose sample rate differs from the filter is rejected or recorded.
- `device="cuda"` is rejected before processing when CUDA is unavailable.

Common errors:

- Public parameter/method/filter errors are validated before processing.
- In `"raise"` mode, a per-file failure is wrapped in `RuntimeError` containing
  the input path, original exception class, and diagnostic.
- In `"skip"` mode, the same information appears in `report.errors`.

## Complete example

`examples/filter_audio.py` generates deterministic stereo audio and exercises:

- designed FIR direct and FFT filtering;
- custom `[0, 1]` one-sample delay filtering;
- Butterworth SOS IIR filtering;
- tensor CPU/CUDA execution;
- SoundFile input/output and `filter_path`.

Run it with:

```bash
python examples/filter_audio.py --device cpu
python examples/filter_audio.py --device cuda
```
