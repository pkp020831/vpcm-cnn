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
- CrossSim and accuracy-aware search intentionally disabled

## Setup

Initialize only the required submodules. TVM is installed as a native ARM64
wheel; its legacy source submodule is not required.

```bash
git submodule update --init booksim2 TOPSIS-Python
./scripts/bootstrap_macos_arm64.sh
```

Activate the environment and inspect the host:

```bash
source .venv/bin/activate
python -m navcim_m4 doctor
```

## Run

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

## Legacy source

The original NavCim scripts remain under `Inference_pytorch`. Their documented
Ubuntu 18.04, CUDA, Python 3.6/3.10 and CrossSim flow is retained for research
comparison, but it is not the default M4 execution path.

NavCim paper: <https://doi.org/10.1145/3656019.3676946>
