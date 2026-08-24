# NeuroSim Pipeline

The maintained NeuroSim path is split into registration, execution, analysis,
collection, and ranking stages. Every stage writes an immutable manifest with
input paths, SHA-256 file references where applicable, and
`performance.wall_seconds`.

## Output layout

```text
outputs/neurosim-pipeline/
├── registered/
│   ├── workload.json
│   ├── search-space.json
│   └── simulator.json
├── simulations/
│   ├── simulation.json
│   └── runs/<config-key>/
│       ├── candidate.json
│       ├── neurosim.log
│       └── neurosim_floorplan.csv
├── analysis/
│   ├── analysis.json
│   └── <config-key>/
│       ├── analysis.json
│       └── layers.csv
└── results/
    ├── results.json
    ├── results.csv
    └── ranking.json
```

## Upstream artifacts

Model/workload generation, simulator building, and search-space expansion stay
independent. They can be run or replaced without rerunning the other stages.

```bash
python -m navcim_m4.pipeline export-model \
  --output-dir outputs/neurosim-pipeline/upstream/model

python -m navcim_m4.pipeline create-workload \
  --model-manifest outputs/neurosim-pipeline/upstream/model/model.json \
  --output-dir outputs/neurosim-pipeline/upstream/workload

python -m navcim_m4.pipeline create-search-space \
  --sa 128,256 --pe 4 --tile 8,16 --adc 5 --cell 2 --mux 8 \
  --output outputs/neurosim-pipeline/upstream/search-space.json
```

The native simulator build remains a separate operation:

```bash
python -m navcim_m4.pipeline build-simulators \
  --output outputs/neurosim-pipeline/upstream/build.json --jobs 8
```

## Registration

Registration converts upstream manifests into NeuroSim-specific immutable
contracts. A changed workload CSV, weight, input, or binary fails hash
validation before simulation begins.

```bash
python -m navcim_m4.neurosim.pipeline register-workload \
  --workload-manifest outputs/neurosim-pipeline/upstream/workload/workload.json \
  --output outputs/neurosim-pipeline/registered/workload.json

python -m navcim_m4.neurosim.pipeline register-search-space \
  --search-space-manifest outputs/neurosim-pipeline/upstream/search-space.json \
  --output outputs/neurosim-pipeline/registered/search-space.json

python -m navcim_m4.neurosim.pipeline prepare-simulator \
  --binary build/neurosim/neurosim \
  --output outputs/neurosim-pipeline/registered/simulator.json
```

## Candidate execution

Run one candidate while debugging:

```bash
python -m navcim_m4.neurosim.pipeline run-candidate \
  --workload-manifest outputs/neurosim-pipeline/registered/workload.json \
  --search-space-manifest outputs/neurosim-pipeline/registered/search-space.json \
  --simulator-manifest outputs/neurosim-pipeline/registered/simulator.json \
  --config-key sa128x128_pe4_tile8_adc5_cell2_mux8 \
  --output-dir outputs/neurosim-pipeline/simulations/runs/sa128x128_pe4_tile8_adc5_cell2_mux8
```

Run every missing or invalid candidate. Existing valid candidate manifests are
reused, so rerunning this command repairs only failed configurations.
Use `--force` only when every candidate must be simulated again.

```bash
python -m navcim_m4.neurosim.pipeline run-pending \
  --workload-manifest outputs/neurosim-pipeline/registered/workload.json \
  --search-space-manifest outputs/neurosim-pipeline/registered/search-space.json \
  --simulator-manifest outputs/neurosim-pipeline/registered/simulator.json \
  --workers 2 --output-dir outputs/neurosim-pipeline/simulations
```

## Layerwise analysis

Analysis consumes only saved logs. Parser fixes and new breakdown metrics do
not rerun NeuroSim.

```bash
python -m navcim_m4.neurosim.pipeline analyze-pending \
  --simulation-manifest outputs/neurosim-pipeline/simulations/simulation.json \
  --output-dir outputs/neurosim-pipeline/analysis
```

Pass `--force` to `analyze-pending` after changing parser behavior. This
rebuilds analysis manifests from existing logs without rerunning the simulator.

Each `layers.csv` reports layer latency, dynamic energy, leakage, buffer,
interconnect, H-tree, ADC, accumulation, and other peripheral breakdowns.

## Collection and ranking

```bash
python -m navcim_m4.neurosim.pipeline collect-results \
  --simulation-manifest outputs/neurosim-pipeline/simulations/simulation.json \
  --analysis-manifest outputs/neurosim-pipeline/analysis/analysis.json \
  --output-dir outputs/neurosim-pipeline/results

python -m navcim_m4.neurosim.pipeline rank-results \
  --results-manifest outputs/neurosim-pipeline/results/results.json \
  --latency-weight 1 --power-weight 1 --area-weight 1 \
  --output outputs/neurosim-pipeline/results/ranking.json
```

Ranking uses chip latency, derived total power, and chip area with Pareto-front
filtering followed by TOPSIS. Raw simulation, analysis, and ranking timing stay
separate in the result manifests.
