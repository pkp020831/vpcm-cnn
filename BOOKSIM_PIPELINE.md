# BookSim Prediction Pipeline

Each stage consumes immutable manifests, writes a new manifest, verifies input
file hashes, and records `performance.wall_seconds`. A failed sample is isolated
under its own directory and can be rerun without repeating successful samples.

BookSim's `Total Power` line contains dynamic switching/clock power; leakage is
reported separately by `Total leak Power`. The normalized dataset maps
`total_power_w` directly to `dynamic_power_w` without subtraction.
BookSim's unlabeled `Total Area` value is in `mm^2`; the historical JSON/CSV
field remains named `area_m2` for artifact compatibility, and PPA composition
converts it to NeuroSim `um^2` with a factor of `1e6`.

## Output layout

```text
outputs/booksim-pipeline/
├── samples/
│   ├── samples.csv
│   └── samples.json
├── simulator/
│   ├── booksim_predictor.cfg
│   └── simulator.json
├── simulations/
│   ├── simulation.json
│   └── samples/sample-000000/
│       ├── booksim.log
│       └── result.json
├── dataset/
│   ├── dataset.csv
│   ├── dataset.json
│   └── split.json
├── models/
│   ├── latency_cycles.joblib
│   ├── dynamic_power_w.joblib
│   ├── area_m2.joblib
│   ├── leakage_power_w.joblib
│   └── *.json
├── bundle/bundle.json
└── reports/
    ├── evaluation.json
    └── top-k.json
```

## Stages

The simulator must already be built. Building remains an independent stage in
`navcim_m4.pipeline`.

`python -m navcim_m4 booksim-data` and `booksim-train` are intentionally not
provided. Use the stages below so every sample, split, model, and report has a
separate manifest and restart path.

```bash
python -m navcim_m4.booksim.pipeline create-samples \
  --count 256 --output-dir outputs/booksim-pipeline/samples

python -m navcim_m4.booksim.pipeline prepare-simulator \
  --booksim-binary booksim2/src/booksim \
  --output-dir outputs/booksim-pipeline/simulator

python -m navcim_m4.booksim.pipeline simulate-pending \
  --samples-manifest outputs/booksim-pipeline/samples/samples.json \
  --simulator-manifest outputs/booksim-pipeline/simulator/simulator.json \
  --output-dir outputs/booksim-pipeline/simulations
```

Rerun one failed sample only:

```bash
python -m navcim_m4.booksim.pipeline simulate-sample \
  --samples-manifest outputs/booksim-pipeline/samples/samples.json \
  --simulator-manifest outputs/booksim-pipeline/simulator/simulator.json \
  --sample-id sample-000017 \
  --output-dir outputs/booksim-pipeline/simulations/samples/sample-000017
```

Collect completed samples and freeze one shared split:

```bash
python -m navcim_m4.booksim.pipeline collect-dataset \
  --samples-manifest outputs/booksim-pipeline/samples/samples.json \
  --simulation-manifest outputs/booksim-pipeline/simulations/simulation.json \
  --output-dir outputs/booksim-pipeline/dataset

# Alternatively, normalize an existing BookSim CSV into the same contract.
python -m navcim_m4.booksim.pipeline import-dataset \
  --input-csv outputs/booksim-predictor/dataset-37143.csv \
  --output-dir outputs/booksim-pipeline/dataset

python -m navcim_m4.booksim.pipeline create-split \
  --dataset-manifest outputs/booksim-pipeline/dataset/dataset.json \
  --output outputs/booksim-pipeline/dataset/split.json
```

Train each target independently. Latency and dynamic power use all seven
traffic features. Area and leakage use only network size and tile width.
Latency defaults to the validated ExtraTrees model; pass
`--model-family paper-mlp` to reproduce the paper-style BatchNorm MLP as a
separate experiment without changing the other target models.

```bash
for target in latency_cycles dynamic_power_w area_m2 leakage_power_w; do
  python -m navcim_m4.booksim.pipeline train-target \
    --dataset-manifest outputs/booksim-pipeline/dataset/dataset.json \
    --split-manifest outputs/booksim-pipeline/dataset/split.json \
    --target "$target" --output-dir outputs/booksim-pipeline/models
done
```

Assemble, evaluate, and compare predicted versus actual Pareto/TOPSIS top-K:

```bash
python -m navcim_m4.booksim.pipeline assemble-bundle \
  --model-manifest outputs/booksim-pipeline/models/latency_cycles.json \
  --model-manifest outputs/booksim-pipeline/models/dynamic_power_w.json \
  --model-manifest outputs/booksim-pipeline/models/area_m2.json \
  --model-manifest outputs/booksim-pipeline/models/leakage_power_w.json \
  --output-dir outputs/booksim-pipeline/bundle

python -m navcim_m4.booksim.pipeline evaluate-bundle \
  --bundle-manifest outputs/booksim-pipeline/bundle/bundle.json \
  --dataset-manifest outputs/booksim-pipeline/dataset/dataset.json \
  --split-manifest outputs/booksim-pipeline/dataset/split.json \
  --output outputs/booksim-pipeline/reports/evaluation.json

python -m navcim_m4.booksim.pipeline compare-top-k \
  --bundle-manifest outputs/booksim-pipeline/bundle/bundle.json \
  --dataset-manifest outputs/booksim-pipeline/dataset/dataset.json \
  --split-manifest outputs/booksim-pipeline/dataset/split.json \
  --top-k 5 --validation-pool 100 \
  --output outputs/booksim-pipeline/reports/top-k.json
```

`top-k` is the strict predicted Pareto/TOPSIS result. `validation-pool` is a
wider full-candidate TOPSIS pool that should be rerun with the real simulator;
the separation protects final selection from predictor errors near a Pareto
boundary.

## Meta-learners

The maintained paired NeuroSim+BookSim stacking flow is a separate pipeline.
See [`META_PIPELINE.md`](META_PIPELINE.md).

## Search integration

```bash
python -m navcim_m4 run \
  --booksim-mode predictor \
  --predictor-bundle outputs/booksim-pipeline/bundle/bundle.json \
  --ranking-mode pareto-topsis \
  --validation-top-k 5
```

Out-of-distribution candidate features fall back to the real simulator by
default. Use `--no-predictor-fallback` only when strict failure is preferred.
