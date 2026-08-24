# vpcm-cnn

Apple Silicon port of the NavCim core simulation path for VGG11 on CIFAR-10.
The original PACT 2024 NavCim source and pinned submodules remain in the repository,
while `navcim_m4` provides a maintained ARM64 path that does not require CUDA or
CrossSim.

## Supported path

```text
VGG11-CIFAR10 -> ONNX -> TVM validation -> NeuroSim PPA
                                     \-> BookSim2 NoC -> ranked result
```

- macOS ARM64 (tested on Apple M4)
- Python 3.13
- PyTorch 2.13, ONNX 1.22, Apache TVM 0.25
- Native Apple Clang builds for NeuroSim and BookSim2
- Four simulation workers by default for a 16 GB machine
- Optional CrossSim accuracy simulation, with no CUDA or CuPy requirement

## Setup

Initialize only the required submodules. TVM is installed as a native ARM64
wheel; its legacy source submodule is not required.

```bash
git submodule update --init booksim2 TOPSIS-Python third_party/cross-sim-3.2
./scripts/bootstrap_macos_arm64.sh
```

Activate the environment and inspect the host:

```bash
source .venv/bin/activate
python -m navcim_m4 doctor
```

## Run

For independently runnable and measurable input/output stages, including the
ordered flowchart and manifest contracts, see [`PIPELINE.md`](PIPELINE.md).
The complete M4 location map is in [`M4_STRUCTURE.md`](M4_STRUCTURE.md), and the
isolated BookSim prediction stages are in [`BOOKSIM_PIPELINE.md`](BOOKSIM_PIPELINE.md).
The independently restartable NeuroSim stages are documented in
[`NEUROSIM_PIPELINE.md`](NEUROSIM_PIPELINE.md).
The paired NeuroSim+BookSim training and stacked PPA flow is documented in
[`META_PIPELINE.md`](META_PIPELINE.md).

Build the simulators:

```bash
python -m navcim_m4 build --jobs 8
```

Run the bounded VGG11-CIFAR10 smoke search:

```bash
python -m navcim_m4 run --workers 4 --output-dir outputs/vgg11-cifar10
```

The command exports an ONNX model, validates it with TVM, creates deterministic
NeuroSim layer records, runs NeuroSim and BookSim2, and writes `results.json`.

The default search point is:

```text
subarray = 128x128
PE = 4
tile = 8
ADC = 5 bits
cell = 2 bits
```

Additional values can be supplied as comma-separated lists:

```bash
python -m navcim_m4 run --sa 128,256 --pe 4 --tile 8,16 --workers 4
```

## Test

```bash
python -m pytest
python -m navcim_m4 run --output-dir outputs/smoke
```

## Optional CrossSim accuracy path

CrossSim 3.2.1 is pinned in `third_party/cross-sim-3.2` and is used only for
PyTorch inference accuracy and analog non-ideality analysis. It does not
calculate PPA: latency, energy, and area remain the responsibility of
NeuroSim and BookSim2. The legacy `cross-sim` submodule is untouched for
NavCim reproduction.

On Apple Silicon, CrossSim runs its NumPy CPU backend with `useGPU=False`.
MPS is used only for PyTorch training and normal digital inference; `auto`
falls back to CPU when MPS is unavailable. CrossSim 3.2.1's CPU PyTorch API
imports successfully on the supported Python 3.13 ARM64 setup, so no CUDA,
CuPy, TensorFlow, or separate Python environment is required. Install its
small CPU-only optional dependency before the first CrossSim command:

```bash
.venv/bin/pip install '.[crosssim]'
```

Train once (an existing checkpoint is reused unless `--force` is supplied):

```bash
python -m navcim_m4 train --device auto --epochs 100 \
  --checkpoint outputs/checkpoints/vgg11-cifar10.pt
```

For a fast training-path smoke check, append `--epochs 1 --train-samples 100`.
Training writes a checkpoint and prints official CIFAR-10 validation accuracy
every 10 epochs. Resume an interrupted run up to the same total epoch target:

```bash
python -m navcim_m4 train --device auto --epochs 100 --resume \
  --checkpoint outputs/checkpoints/vgg11-cifar10.pt
```

Checkpoints and generated outputs are intentionally excluded from Git. Publish
models through a GitHub Release, Git LFS, or an external artifact store, and
record the training command and `shasum -a 256` digest alongside the artifact.

Run a reproducible 100-image CPU smoke evaluation, then a full CIFAR-10
evaluation when ready:

```bash
python -m navcim_m4 crosssim --checkpoint outputs/checkpoints/vgg11-cifar10.pt \
  --samples 100 --runs 3 --adc-bits 5 --cell-bits 2 --subarray 128 \
  --output outputs/crosssim
python -m navcim_m4 crosssim --checkpoint outputs/checkpoints/vgg11-cifar10.pt \
  --samples 10000 --runs 3 --output outputs/crosssim-full
```

`crosssim_results.json` contains digital accuracy, CrossSim mean/std accuracy,
percentage-point drop, classified-image counts and timing per run, the exact
hardware configuration, random seeds, and checkpoint SHA-256. The settings
include ADC/DAC/cell precision, array dimensions, weight/input bit slicing,
programming error, read noise, wire resistance, and seed. CrossSim 3.2.1 has
no validated generic conductance-drift model, so a nonzero `--drift-time` is
rejected rather than ignored.

To merge cached accuracy with every PPA candidate, add `--crosssim` and a
checkpoint to the existing search command:

```bash
python -m navcim_m4 run --checkpoint outputs/checkpoints/vgg11-cifar10.pt \
  --crosssim --crosssim-samples 100 --crosssim-runs 3
```

Predictor searches use a full-candidate TOPSIS validation pool of 100 by
default; override `--validation-top-k` only with a recorded justification.

The cache key includes the checkpoint SHA-256, sample count, run count, and
CrossSim configuration. CPU analog simulation is substantially slower than
digital inference, especially with noise or parasitic resistance; use the
100-image cache for search and re-run selected Pareto candidates with 10,000
samples. Set all precision/noise/resistance effects to their ideal values for
the numerical digital-versus-CrossSim equivalence check.

Run the prescribed 42-condition screening sweep on 100 test images in batches
of 100. It uses ADC bits 4/5/8, 7-bit-cell/1-slice and 1-bit-cell/7-slice
weight mappings, then independently sweeps programming error and read noise
at 0.01/0.03 and wire resistance at 0.25/1.0 ohm. Non-ideal conditions use
three seeds; ideal conditions use one.

```bash
python -m navcim_m4 crosssim-sweep \
  --checkpoint outputs/checkpoints/vgg11-cifar10.pt \
  --output outputs/crosssim-sweep
```

## Legacy source

The original NavCim scripts remain under `Inference_pytorch`. Their documented
Ubuntu 18.04, CUDA, Python 3.6/3.10 and CrossSim flow is retained for research
comparison, but it is not the default M4 execution path.

NavCim paper: <https://doi.org/10.1145/3656019.3676946>
