# Project Execution Rules

- Do not run commands, simulations, builds, training, searches, or destructive operations without an explicit user request.
- If a requested operation cannot run, fails, or requires an unavailable dependency, stop that operation and report the blocker, command, and relevant error to the user. Do not silently substitute an alternative execution path.

# User Priorities

- Treat pipeline design as a primary requirement, not an implementation detail.
- Split work into independently executable stages with explicit inputs and outputs. A failure or performance issue in one stage must be fixable and rerunnable without repeating successful upstream work.
- Make performance evaluation equally granular. Record stage wall time and stage-specific metrics so simulation, parsing, prediction, ranking, and validation costs can be measured separately.
- Prefer immutable JSON manifests and SHA-256 file references between stages. Reuse an artifact only when its registered inputs and generated outputs still match.
- Isolate candidate/sample outputs in their own directories. Batch commands must preserve completed entries, report individual failures, and rerun only missing, invalid, or explicitly forced entries.
- Keep raw simulation, analysis, model training, ranking, and final validation as separate stages. Parser or ranking changes must not require simulator reruns.
- Document every maintained pipeline, output layout, stage contract, repair command, and generated artifact location.
- Do not describe a partially connected scaffold as complete. Report separately whether code exists, data was generated, models were trained, and end-to-end validation was run.

# Parallel Session Collaboration

- Assume the user may run multiple agent sessions against the same worktree.
- Work in the smallest relevant module and do not revert, rewrite, or clean unrelated changes from another session.
- Use stage-specific output directories and manifests to avoid sessions overwriting each other's artifacts.
- Before extending an existing pipeline, read its structure document and manifests rather than rebuilding the same functionality in another location.
- When project direction or the remaining work changes, update this file so new sessions inherit the latest priorities and status.
- Always keep this file current. After any material change to user priorities, architecture, pipeline status, artifact locations, completed work, or remaining work, update `AGENTS.md` before ending the session.

# Maintained M4 Architecture

- `navcim_m4` is the maintained Apple Silicon path. See `M4_STRUCTURE.md` for the complete location map.
- The general file-based pipeline is documented in `PIPELINE.md`.
- The BookSim prediction pipeline is under `navcim_m4/booksim/` and documented in `BOOKSIM_PIPELINE.md`.
- The NeuroSim execution and layer-analysis pipeline is under `navcim_m4/neurosim/` and documented in `NEUROSIM_PIPELINE.md`.
- The NeuroSim+BookSim stacking pipeline is under `navcim_m4/meta/` and documented in `META_PIPELINE.md`.
- Generated BookSim artifacts are under `outputs/booksim-pipeline/`.
- Generated NeuroSim artifacts are under `outputs/neurosim-pipeline/`.
- Generated paired meta artifacts are under `outputs/meta-pipeline/`.

# Simulator Dependency Model

- BookSim predictor pretraining can run independently from a particular NeuroSim candidate by sampling NoC features and collecting BookSim targets.
- Actual candidate evaluation is not independent: NeuroSim runs first and produces the floorplan and NoC physical parameters consumed by either BookSim or the BookSim predictor.
- Preserve the candidate dependency chain:

```text
model/workload
-> NeuroSim compute PPA + floorplan + NoC parameters
-> BookSim features
-> BookSim simulator or predictor
-> NeuroSim/BookSim PPA composition
-> Pareto/TOPSIS
-> real-simulator validation pool
```

- Do not create a separate BookSim candidate path that invents floorplan or NoC values when registered NeuroSim outputs should be used.

# Current Project Status

- BookSim sample/simulation/dataset/split/target-training/bundle/evaluation/top-K stages are implemented.
- The trained BookSim bundle is at `outputs/booksim-pipeline/bundle/bundle.json` and uses a shared 37,143-sample dataset with a 7,429-sample holdout.
- BookSim holdout results currently recorded in `outputs/booksim-pipeline/reports/evaluation.json` are approximately: latency R2 0.946, dynamic-power R2 0.996, area R2 0.9996, and leakage R2 0.9995.
- Strict predicted top-5 recall was insufficient near the Pareto boundary. Use a full-candidate TOPSIS validation pool of 100; the recorded demo recovered all actual top-5 candidates in that pool.
- NeuroSim workload/search-space/simulator registration, candidate execution/restart, saved-log layer analysis, collection, and Pareto/TOPSIS stages are implemented.
- The paired M4/VGG11 meta pipeline is implemented and has generated 48 layer rows from six completed hardware configurations. Two SA64/tile8 BookSim pairs remain explicitly incomplete: both reached the 1,800-second per-layer timeout at BookSim layer 9 on a 13x13 mesh, including retries under `caffeinate -i`.
- The trained meta bundle is at `outputs/meta-pipeline/bundle/bundle.json`. Current config-disjoint holdout results are approximately latency R2 0.9999 and energy R2 0.868; candidate-level predicted top-3 recall was 100% over the six completed candidates.
- A four-candidate predictor+meta Pareto/TOPSIS search was run end to end at `outputs/meta-pipeline/search-demo/results.json`.
- Do not describe the meta learner as production-complete: energy generalization is below the paper result, the paired dataset is small, and actual VGG11 traffic exceeds the original BookSim predictor range in several features.
- The next major PPA quality task is to expand in-domain BookSim pretraining and paired hardware configurations, improve energy holdout accuracy, and repeat top-K validation on a substantially larger unseen candidate pool.
- BookSim `Total Area` is in mm2 even though the retained compatibility field is named `area_m2`; convert it to NeuroSim um2 with `1e6`, not `1e12`.
- Current pre-push hardening work has fixed BookSim mode selection, layerwise predictor energy composition, full-candidate TOPSIS validation-pool selection, candidate cache contracts, NeuroSim mesh column emission, staged provenance checks, split/cache integrity, and local artifact portability. The repaired four-candidate predictor+meta search validated all four candidates and reproduced the top-1 with latency relative error about 0.013% and total-power relative error about 0.046%.
- Do not push to main until the remaining pre-push review findings are addressed or explicitly accepted: the all-in-one hardware sweep still rebuilds upstream artifacts, staged manifest migration requires regeneration of old manifests, and large in-domain/meta dataset expansion remains incomplete.
- The staged M4 pipeline implementation and current hardening changes were pushed to `main` at commit `d77832c3`. The remaining review findings above are follow-up work, not evidence that the pushed PPA results are production-complete.

# Scope Boundaries

- Completing the current homogeneous VGG11 PPA path does not mean the full NavCim paper is reproduced.
- Accuracy-aware search, heterogeneous tile mapping, hierarchical beam search, and multi-model evolutionary optimization remain separate follow-up scopes unless the user explicitly prioritizes them.
