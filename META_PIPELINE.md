# NeuroSim + BookSim Meta-Learner Pipeline

This pipeline keeps NeuroSim simulation, actual BookSim validation, BookSim
prediction, paired-data collection, model training, evaluation, ranking, and
end-to-end search as separate restartable stages.

## Dependency flow

```text
registered VGG11 workload
-> NeuroSim candidates and layer analysis
-> actual BookSim layer runs + BookSim predictor outputs
-> paired layer dataset
-> config-disjoint train/test split
-> overall-latency model
-> overall-energy model
-> meta bundle
-> candidate Pareto/TOPSIS comparison
-> predictor+meta end-to-end search
```

## Output layout

```text
outputs/meta-pipeline/
├── neurosim/
│   ├── search-space-source.json
│   ├── search-space.json
│   ├── simulations/
│   └── analysis/
├── registered/upstream.json
├── pairing/
│   ├── pairing.json
│   └── pairs/<config-key>/
│       ├── pair.json
│       ├── booksim.log
│       └── booksim_layers.json
├── dataset/
│   ├── dataset.csv
│   ├── dataset.json
│   └── split.json
├── models/
│   ├── overall_latency_ns.joblib
│   ├── overall_latency_ns.json
│   ├── overall_dynamic_energy_pj.joblib
│   └── overall_dynamic_energy_pj.json
├── bundle/bundle.json
├── reports/
│   ├── evaluation.json
│   └── candidate-comparison.json
└── search-demo/results.json
```

## Pairing

Register exact NeuroSim, predictor, simulator, and workload inputs:

```bash
python -m navcim_m4.meta.pipeline register-upstream \
  --neurosim-simulation-manifest outputs/meta-pipeline/neurosim/simulations/simulation.json \
  --neurosim-analysis-manifest outputs/meta-pipeline/neurosim/analysis/analysis.json \
  --neurosim-workload-manifest outputs/neurosim-pipeline/registered/workload.json \
  --booksim-predictor-bundle outputs/booksim-pipeline/bundle/bundle.json \
  --booksim-binary booksim2/src/booksim \
  --output outputs/meta-pipeline/registered/upstream.json

python -m navcim_m4.meta.pipeline run-pending \
  --upstream-manifest outputs/meta-pipeline/registered/upstream.json \
  --workers 2 --output-dir outputs/meta-pipeline/pairing
```

Each pair is isolated by config. Completed pairs are retained across an
interrupted batch. If a long-running batch is stopped, record the usable subset
without rerunning it:

```bash
python -m navcim_m4.meta.pipeline summarize-pairs \
  --upstream-manifest outputs/meta-pipeline/registered/upstream.json \
  --output-dir outputs/meta-pipeline/pairing
```

## Dataset and training

```bash
python -m navcim_m4.meta.pipeline collect-dataset \
  --pairing-manifest outputs/meta-pipeline/pairing/pairing.json \
  --output-dir outputs/meta-pipeline/dataset

python -m navcim_m4.meta.pipeline create-split \
  --dataset-manifest outputs/meta-pipeline/dataset/dataset.json \
  --test-ratio 0.33 --seed 117 \
  --output outputs/meta-pipeline/dataset/split.json

python -m navcim_m4.meta.pipeline train-latency \
  --dataset-manifest outputs/meta-pipeline/dataset/dataset.json \
  --split-manifest outputs/meta-pipeline/dataset/split.json \
  --output-dir outputs/meta-pipeline/models

python -m navcim_m4.meta.pipeline train-energy \
  --dataset-manifest outputs/meta-pipeline/dataset/dataset.json \
  --split-manifest outputs/meta-pipeline/dataset/split.json \
  --latency-model-manifest outputs/meta-pipeline/models/overall_latency_ns.json \
  --model-family auto --seed 117 \
  --output-dir outputs/meta-pipeline/models
```

The split is by `config_key`, not by layer row. Layers from one hardware
configuration never appear in both train and test sets. Latency uses degree-2
polynomial regression. Energy `auto` compares MLP, polynomial regression, and
ExtraTrees using config-group cross-validation before fitting the selected
family.

## Bundle and evaluation

```bash
python -m navcim_m4.meta.pipeline assemble-bundle \
  --latency-model-manifest outputs/meta-pipeline/models/overall_latency_ns.json \
  --energy-model-manifest outputs/meta-pipeline/models/overall_dynamic_energy_pj.json \
  --output outputs/meta-pipeline/bundle/bundle.json

python -m navcim_m4.meta.pipeline evaluate-bundle \
  --bundle-manifest outputs/meta-pipeline/bundle/bundle.json \
  --dataset-manifest outputs/meta-pipeline/dataset/dataset.json \
  --split-manifest outputs/meta-pipeline/dataset/split.json \
  --output outputs/meta-pipeline/reports/evaluation.json

python -m navcim_m4.meta.pipeline compare-candidates \
  --bundle-manifest outputs/meta-pipeline/bundle/bundle.json \
  --dataset-manifest outputs/meta-pipeline/dataset/dataset.json \
  --top-k 3 --output outputs/meta-pipeline/reports/candidate-comparison.json
```

## Search integration

Meta inference is applied per layer. The first NeuroSim layer, which has no
preceding NoC transition in the current mapping, is retained as a compute-only
residual. Final simulator validation uses actual layerwise BookSim energy and
does not feed actual BookSim values back through the meta learner.

```bash
python -m navcim_m4 run \
  --output-dir outputs/meta-pipeline/search-demo \
  --sa 128 --pe 2,4 --tile 8,16 --adc 5 --cell 2 --mux 8 \
  --booksim-mode predictor \
  --predictor-bundle outputs/booksim-pipeline/bundle/bundle.json \
  --meta-bundle outputs/meta-pipeline/bundle/bundle.json \
  --allow-predictor-ood --ranking-mode pareto-topsis
```

`--allow-predictor-ood` is explicit because the current VGG11 paired domain
contains traffic values outside the original BookSim pretraining ranges. Do not
enable it for an unvalidated search space.

## Current measured scope

- Six completed hardware configurations, 48 paired layer rows.
- Two SA64/tile8 actual BookSim candidates exceeded 15 minutes and remain
  recorded as incomplete rather than being silently discarded.
- Holdout latency R2 is approximately 0.9999; holdout energy R2 is approximately
  0.868. Candidate-level predicted top-3 recall is 100% over the six completed
  configurations, but this is not yet broad enough for production claims.
- The next quality step is broader in-domain BookSim pretraining and more paired
  hardware configurations, especially for energy generalization.
