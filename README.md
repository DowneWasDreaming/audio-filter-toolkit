# Audio Filter Toolkit

A compact, typed Python package for designing and applying low-pass, high-pass,
band-pass, and band-stop FIR/IIR audio filters. Numerical filtering uses
PyTorch, accepts NumPy or Torch arrays, supports CPU/CUDA Torch tensors, and
preserves every leading dimension before the final time axis.

The WAV I/O, atomic saving, deterministic directory discovery, and processing
report conventions follow the author's
[`audio-dsp-toolkit`](https://github.com/DowneWasDreaming/audio-dsp-toolkit).

## Dimension convention

- Array filtering accepts `[time]` or `[..., time]`.
- Every leading dimension is an independent signal.
- Output shape is identical to input shape.
- `load_wav` returns `[channel, time]`, including `[1, time]` for mono.
- NumPy input returns NumPy on CPU.
- Torch input preserves its dtype and CPU/CUDA device.
- Path filtering loads `[channel, time]` with SoundFile and uses the explicitly
  selected Torch device.

Numerical arrays must be finite `float32` or `float64` values with at least one
time sample. Filtering never resamples implicitly: the call-time or file sample
rate must match the filter design rate.

## Installation

```bash
python -m pip install -e ".[test]"
```

SoundFile relies on libsndfile. The Python wheels normally include it.

## Filter design

```python
from audio_filter_toolkit import design_butterworth_iir, design_fir

fir = design_fir(
    kind="bandpass",
    sample_rate=48_000,
    cutoff_hz=(300.0, 3_400.0),
    num_taps=257,
    window=("kaiser", 8.0),
    scale=True,
)

iir = design_butterworth_iir(
    kind="bandstop",
    sample_rate=48_000,
    critical_hz=(5_000.0, 7_000.0),
    order=6,
)

print(fir.taps.shape)  # [257]
print(iir.sos.shape)   # [num_sections, 6]
```

`fir.taps[k]` is the standard forward-time impulse response `h[k]`; the public
tensor is **not reversed**. Mathematical FIR filtering is convolution:

```text
y[n] = sum_k h[k] * x[n-k]
```

PyTorch `conv1d` evaluates cross-correlation, so the direct implementation
reverses `h` internally exactly once before calling it. The FFT implementation
multiplies the spectra of `x` and the forward-time `h` directly. Numerical
tests verify that both implementations return the same linear convolution.

Window-method FIR cutoff frequencies are the half-amplitude points (about
-6 dB). Butterworth critical frequencies are -3 dB points. They use distinct
parameter names to make this difference visible.

FIR designs require an odd `num_taps >= 3`. Supported windows are `hann`,
`hamming`, `blackman`, `bartlett`, `boxcar`, and `("kaiser", beta)`.

### Custom FIR coefficients

Use `fir_from_taps` when the impulse response is already known:

```python
from audio_filter_toolkit import apply_filter, fir_from_taps

one_sample_delay = fir_from_taps(
    [0.0, 1.0],
    sample_rate=48_000,
)
delayed = apply_filter(
    waveform,
    one_sample_delay,
    sample_rate=48_000,
    method="direct",  # "fft" returns the same causal linear convolution
)
```

The coefficients are stored exactly in forward-time order. They are not
reversed, normalized, or otherwise redesigned. Python sequences, NumPy arrays,
and Torch tensors are accepted; the filter stores an independent CPU float64
copy and moves it to the waveform dtype/device when applied.

Custom taps may have any positive length. Even-length filters such as
`[0.0, 1.0]` support causal direct and FFT filtering. They reject
`alignment="centered"` because their linear-phase delay lies on a half sample
and there is no unique integer output slice.

## Array filtering

```python
from audio_filter_toolkit import apply_filter

causal = apply_filter(
    waveform,
    fir,
    sample_rate=48_000,
    method="direct",
    alignment="causal",
)

centered = apply_filter(
    waveform,
    fir,
    sample_rate=48_000,
    method="fft",
    alignment="centered",
)

recursive = apply_filter(
    waveform,
    iir,
    sample_rate=48_000,
    method="recursive",
)
```

- FIR accepts `method="direct"` or `method="fft"`.
- `alignment="causal"` uses zero initial history and retains FIR group delay.
- `alignment="centered"` compensates the integer group delay of the odd-length
  FIR. It is an offline, non-causal operation with zero extension at both ends.
- IIR accepts only `method="recursive", alignment="causal"`.
- IIR coefficients use second-order sections and zero initial state.
- Filtering never clamps amplitudes. PCM clipping is controlled explicitly
  only when saving.

The FFT path calls `torch.fft.rfft` and `torch.fft.irfft`; it does not contain
a custom FFT implementation. It computes full linear convolution and slices
the requested alignment, rather than returning circular convolution.

## WAV file and directory filtering

```python
from audio_filter_toolkit import filter_path

report = filter_path(
    "input_wavs",
    "filtered_wavs",
    fir,
    method="fft",
    alignment="centered",
    device="cuda",
    recursive=True,
    overwrite=False,
    on_error="skip",
    show_progress=True,
    decode_dtype="float32",
    output_subtype="FLOAT",
    amplitude_policy="error",
)

for input_path, message in report.errors:
    print(f"{input_path}: {message}")
```

Directory output preserves relative subdirectories. A destination below its
source is rejected so newly written files cannot be rediscovered. Exact
in-place processing is supported only with `overwrite=True`:

```python
filter_path(
    "input_wavs",
    "input_wavs",
    fir,
    method="direct",
    overwrite=True,
)
```

Files are enumerated before processing and each output is written to a
temporary WAV beside its destination before atomic replacement. With
`on_error="skip"`, a failed input is recorded and its original file remains
unchanged.

Path processing currently reads each complete WAV into memory. Streaming and
overlap-add FFT blocks are outside the first milestone.

## Runnable example and tests

The example generates its own float64 stereo input and writes DOUBLE WAV
results:

```bash
python examples/filter_audio.py --device cpu
python examples/filter_audio.py --device cuda
```

Run the numerical suite:

```bash
pytest
```

The tests check actual impulse responses, pass/stop-band gains, direct/FFT
agreement, SciPy SOS reference values, NumPy/Torch agreement, CPU/CUDA
agreement, batch independence, WAV values, directory behavior, and validation
errors.
