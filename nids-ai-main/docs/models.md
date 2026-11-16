# Models, training configurations and the hyperparameter search

Results are in `docs/results.md`. This file gives what was trained, how, and
which configurations were tried. Every configuration that was actually run
is recorded, with its validation score, in the committed metric files:
`artifacts/metrics/track_a/*.json` (`search`, `grid_results`) and the
per-run cache `artifacts/metrics/track_{a,b}/runs/`.

## Selection protocol (all models)

1. Hyperparameters are searched with seed 0 and scored on **validation data
   only**. Classifiers are scored by macro-F1; anomaly detectors by ROC-AUC
   of the anomaly score.
2. The selected configuration is retrained with seeds 0, 1 and 2.
3. Each seed is evaluated **once** on the test set. Tables report seed 0,
   the seed standard deviation, and a 95 % bootstrap interval.

Early stopping always monitors validation loss and restores the best
epoch. The test sets never influence any choice.

## Compute environment and the reduced search

Everything ran on one laptop CPU: Intel i5-1135G7 (4 cores / 8 threads),
7.7 GB RAM, no GPU, with 4 PyTorch threads.

The first Track A deep-learning run used the full planned grid. The
operating system stopped it for low memory while the MLP search was
partway through. Other desktop applications held most of the RAM at the
time. Measured during the restarted run, the training process itself peaks
at about 0.6 GB resident and 1.2 GB committed memory. None of that run's
results are used. Its log is kept, for transparency only, in
`artifacts/preliminary/deep_full_grid_interrupted.log`.

The experiments were then restarted with a **reduced search**, approved by
the project owner, and made resumable: each finished run is saved
immediately, so an interruption loses at most the run in progress. The
reductions were:

| Model | Planned search | Search actually run | Dropped |
|---|---|---|---|
| MLP | hidden {256-128, 512-256-128} × dropout {0.2, 0.4} × lr {1e-3, 3e-4} (8), then imbalance {none, balanced, oversample}, then batch size {256, 2048} | (128-64, 0.2, 1e-3), (256-128, 0.2, 1e-3), (256-128, 0.2, 3e-4), (256-128, 0.4, 1e-3); then imbalance {none, balanced, oversample} | the 3-layer 512-unit network; the batch-size study |
| 1D-CNN | channels {32-64, 64-128} × kernel {3, 5} (4) | (32-64, k3), (32-64, k5), (64-128, k3) | 64-128 channels with kernel 5 |
| Autoencoder | bottleneck {8, 16, 32} | unchanged | – |
| LSTM (Track B) | L ∈ {4, 16, 32}, ≤ 25 epochs, all attacks + 15 % of normal flows per epoch | L ∈ {4, 16}, ≤ 15 epochs (patience 3), at most 30,000 normal and 15,000 flows of each attack class per epoch (halved with the 50 % training-data reduction, `docs/sampling.md`) | L = 32 |
| Per-flow MLP / RF (Track B) | as planned | same per-class epoch sampling as the LSTM | – |
| RF, Isolation Forest | as planned (run before the restart) | unchanged | – |

All four required neural model families (MLP, 1D-CNN, Autoencoder, LSTM)
were kept, as were three seeds for every final model. The search is
therefore narrower than planned, and a larger network or longer sequence
window might score somewhat better. `docs/report.md` lists this as a
limitation.

## Track A models (186 preprocessed features)

| Model | Architecture | Training |
|---|---|---|
| Random Forest | 150 trees, Gini. Grid: `min_samples_leaf` {1, 3} × `max_features` {sqrt, 0.3} × imbalance {none, balanced_subsample, oversample} | scikit-learn, 4 jobs |
| RF (binary) | selected multiclass configuration, attack vs normal | |
| Isolation Forest | grid: `n_estimators` {100, 300} × `max_samples` {256, 2048} × `max_features` {1.0, 0.5} | normal training records only; threshold = 95th percentile of validation normal scores |
| MLP | [Linear → BatchNorm → ReLU → Dropout] × depth → Linear (10 classes) | Adam, batch 512, ≤ 30 epochs, patience 4, cross-entropy with 1/√frequency class weights (or the strategy selected in the imbalance study) |
| 1D-CNN | numeric features (grouped by meaning: basic, load/content, TCP, timing, counters, flags) as a 1-channel sequence: Conv-BN-ReLU ×2 → MaxPool(2) → Conv-BN-ReLU → global average + max pooling; one-hot categorical features through Dense(32); joint Dense(128) head | same optimiser and loss as the selected MLP |
| 1D-CNN, random order (ablation) | identical, but the numeric features are permuted randomly (seed 1234) | tests whether the meaningful feature order matters |
| Autoencoder | 186 → 128 → 64 → bottleneck → 64 → 128 → 186, ReLU | MSE on normal training records only; score = reconstruction MSE; threshold = 95th percentile of validation normal scores |

## Track B models (194 features, chronological blocks)

| Model | Architecture | Notes |
|---|---|---|
| FlowLSTM | per-flow Linear(194 → 64) + ReLU → 1-layer LSTM (64) → Dropout(0.2) → Linear (10) on the last step; packed sequences, so padding is ignored | input = target flow plus up to L−1 earlier flows from the same source IP in the same block |
| Per-flow MLP | the architecture selected on Track A, with 194 inputs | no history |
| Per-flow RF | the Track A RF configuration | no history |
| Context baseline | logistic regression on [log per-flow MLP probabilities, mean MLP probabilities of the host's previous L−1 flows, has-history flag], fitted on the validation block | uses context but not order |
| Oracle context | the same with the previous flows' TRUE labels | upper bound only, not deployable |

Track B training samples are drawn fresh every epoch, per class (see the
table above). Loss weights multiply each class by n_class / n_sampled, so
the expected loss equals the full-data loss, and then by 1/√frequency.
Validation loss uses the same per-class fractions with a fixed seed.
Validation and test predictions always cover every flow.

## Execution log

| Date (UTC) | Event |
|---|---|
| 2026-11-15 | Reduced-search run: Track A MLP search, imbalance study and final seeds 0–2 completed (`mlp.json`). 1D-CNN search config (32-64, k3) completed (val macro-F1 0.5857). |
| 2026-11-15/16 | Run stopped by the operator during 1D-CNN config (32-64, k5), after epoch 23. Per-run caching meant only that run was lost. |
| 2026-11-16 04:17 | Resumed with `run_experiments.py --from 2` (log: `artifacts/pipeline.log`). Before resuming: (1) fixed a bug in `train_deep.search()` that compared a stale/undefined `rep` instead of the cached row's score, which would have crashed or mis-selected on resume; (2) added per-epoch resume checkpoints (`trainer.fit(resume_path=...)`, files in git-ignored `artifacts/cache/resume/`, deleted once a run is saved; `tests/test_trainer_resume.py` checks an interrupted run ends bit-identical to an uninterrupted one), used by Track A and Track B training; (3) `train_deep.py` skips the MLP when `mlp.json` exists (`--force` recomputes), so completed artifacts are not rewritten. |
| 2026-11-16 | 1D-CNN search completed: (32-64, k3) 0.5857, (32-64, k5) **0.6082** (selected), (64-128, k3) 0.6050 validation macro-F1. |
| 2026-11-16 08:03 | Pipeline stopped by Claude Code because the whole system was critically low on memory (other desktop applications), during final CNN seed 0 after epoch 3. Epoch checkpoint `artifacts/cache/resume/track_a/cnn/01d1d47968d8e2b0.pt` kept; rerunning `run_experiments.py --from 2` continues from epoch 4. |
| 2026-11-16 | Resumed; CNN seed 0 continued from epoch 4. Final 1D-CNN (32-64, k5) test macro-F1: seeds 0/1/2 = 0.4688 / 0.4662 / 0.4707 (`cnn.json`). |
| 2026-11-16 | **Reduced training data** for the remaining experiments, at the owner's request (`docs/sampling.md`). Ablation and Autoencoder kept on full data. Track B: most recent 50 % of each session's training period; epoch caps halved to 30,000 / 15,000; validation and test blocks memory-mapped. LOCO: 50 % minority-preserving stratified sample, confirmed on validation by the new step `scripts/sampling_study.py`; preprocessor refitted per fold without the held-out class. |
| 2026-11-16 | Random-order ablation seeds 0/1/2: 0.4676 / 0.4590 / 0.4576. Autoencoder search (bottleneck 8/16/32: val ROC-AUC 0.908 / 0.930 / **0.943**); a Windows "Access is denied" error when renaming an epoch checkpoint (file briefly held by antivirus/indexer) stopped the run; fixed with a retrying `store.atomic_replace`, resumed from epoch 13. Autoencoder test ROC-AUC 0.943 / 0.931 / 0.932. |
| 2026-11-16 | Track B (reduced data, 715 s): LSTM L = 4 selected (val 0.483 vs 0.478); sampling study chose 50 % (RF −0.011, MLP −0.015 val macro-F1); leave-one-class-out (288 s), explainability (229 s) and comparison (44 s) completed. Pipeline finished, exit 0. Results: `docs/results.md`. |

## Reproducing

```
.venv\Scripts\python.exe scripts\prepare_data.py
.venv\Scripts\python.exe scripts\run_experiments.py      # steps 1-7, resumable (per run and per epoch)
.venv\Scripts\python.exe scripts\make_package.py [--with-models]   # submission zip (never contains the dataset)
```
