# M4 Implementation Structure

This document is the location map for the maintained Apple Silicon path. The
legacy research implementation under `Inference_pytorch` remains unchanged.

## Maintained source

| Location | Responsibility |
|---|---|
| `navcim_m4/models.py` | Fixed VGG11-CIFAR10 model, checkpoint, ONNX export |
| `navcim_m4/layers.py` | Deterministic NeuroSim workload artifacts |
| `navcim_m4/graph.py` | TVM graph validation |
| `navcim_m4/simulators.py` | NeuroSim/BookSim build, execution, output parsing |
| `navcim_m4/booksim_mapping.py` | NeuroSim floorplan to BookSim traffic mapping |
| `navcim_m4/accuracy.py` | Training and CrossSim accuracy evaluation |
| `navcim_m4/search.py` | All-in-one compatibility search path |
| `navcim_m4/pipeline.py` | File-manifest simulation pipeline |
| `navcim_m4/booksim/` | Isolated BookSim predictor and ranking pipeline |
| `navcim_m4/neurosim/` | Isolated NeuroSim execution, layer analysis, and ranking pipeline |
| `navcim_m4/meta/` | Paired NeuroSim+BookSim data, meta training, evaluation, and bundle pipeline |
| `navcim_m4/__main__.py` | Top-level CLI |

## BookSim predictor package

| Location | Responsibility |
|---|---|
| `navcim_m4/booksim/contracts.py` | Manifest schema, SHA-256 input validation |
| `navcim_m4/booksim/features.py` | Feature and target contracts, deterministic sampling |
| `navcim_m4/booksim/pipeline.py` | Independently executable stages |
| `navcim_m4/booksim/models.py` | Target training, bundle assembly/evaluation, top-K comparison |
| `navcim_m4/booksim/inference.py` | Candidate feature extraction and layerwise prediction |
| `navcim_m4/booksim/meta.py` | Latency/energy meta-learner stages |
| `navcim_m4/booksim/ppa.py` | Unit-safe NeuroSim/BookSim PPA composition |
| `navcim_m4/booksim/ranking.py` | Pareto-front and TOPSIS ranking |

## Simulator and external sources

| Location | Responsibility |
|---|---|
| `Inference_pytorch/NeuroSIM/` | NeuroSim C++ source used by the M4 build |
| `booksim2/` | BookSim2 source and power model |
| `third_party/cross-sim-3.2/` | Pinned CPU CrossSim accuracy simulator |
| `TOPSIS-Python/` | Legacy TOPSIS dependency; the M4 pipeline uses its tested local implementation |

## Tests, scripts, data, and generated files

| Location | Responsibility |
|---|---|
| `tests/` | Unit and integration tests for the maintained path |
| `scripts/` | Bootstrap, plotting, and report scripts |
| `data/` | CIFAR-10 and local input data |
| `build/neurosim/` | Generated native NeuroSim build |
| `booksim2/src/booksim` | Generated native BookSim executable |
| `outputs/` | Generated manifests, simulator logs, datasets, models, and reports |
| `README.md` | Supported setup and high-level commands |
| `PIPELINE.md` | Main simulation stage contracts |
| `BOOKSIM_PIPELINE.md` | BookSim prediction stage contracts and commands |
| `NEUROSIM_PIPELINE.md` | NeuroSim candidate execution and analysis contracts |
| `META_PIPELINE.md` | NeuroSim+BookSim stacking, training, and validation contracts |

## Legacy boundary

`Inference_pytorch/` except for its `NeuroSIM` C++ source is the original
Ubuntu/CUDA research workflow. The maintained Apple Silicon path reuses and
extends that C++ source for its simulator binary, but does not use the legacy
Python/shell workflow as an execution pipeline.
