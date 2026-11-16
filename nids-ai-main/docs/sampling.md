# Reduced training data for the remaining experiments

Added on 2026-11-16, at the project owner's request, to shorten the
remaining experiments on the 7.7 GB laptop. **Only training data is
reduced.** The official UNSW-NB15 test set, the Track A validation split
and the Track B validation and test blocks are complete and unchanged. No
test result was used to choose a sampling method or fraction.

## What uses which data

| Experiment | Training data | Reason |
|---|---|---|
| Random Forest, Isolation Forest, MLP, 1D-CNN (completed) | **full** | finished before this change; not retrained |
| 1D-CNN random-order ablation | **full** | It is a paired comparison with the full-data 1D-CNN. Training it on less data would mix the effect of feature order with the effect of data size and make the ablation meaningless. It was already running. |
| Autoencoder | **full** normal training traffic (48,000 records) | Measured ~2 s per epoch, so a sample would save only minutes (estimate). Full data keeps it comparable with the full-data Isolation Forest. |
| Track B: LSTM, per-flow MLP, per-flow RF, context baselines | **reduced**: most recent 50 % of each session's training period | the most expensive step, both in time and memory |
| Leave-one-attack-class-out (RF, MLP; 9 folds) | **reduced**: 50 % minority-preserving stratified sample (confirmed on validation data, see below) | 18 trainings |
| Explainability, model comparison | no training | – |

Results tables (`reports/results_tables.md`, `artifacts/metrics/comparison.json`)
show a "Training data" column, and every metrics file written by a
reduced-data experiment contains a `training_data` record (strategy,
fraction, seed, class counts).

## Track A: minority-preserving stratified sample

`deepshield.data.sampling.stratified_subsample(y, fraction, seed=0, floor=2000)`
keeps a random `fraction` of each class's training records. A class is never
reduced below 2,000 records, and smaller classes are kept whole. Seed 0, so
the sample is identical in every run and in every LOCO fold.

| Class | Full training split | 70 % | 50 % | 30 % |
|---|---:|---:|---:|---:|
| Normal | 48,000 | 33,600 (70%) | 24,000 (50%) | 14,400 (30%) |
| Generic | 34,285 | 24,000 (70%) | 17,142 (50%) | 10,286 (30%) |
| Exploits | 28,622 | 20,035 (70%) | 14,311 (50%) | 8,587 (30%) |
| Fuzzers | 15,586 | 10,910 (70%) | 7,793 (50%) | 4,676 (30%) |
| DoS | 10,512 | 7,358 (70%) | 5,256 (50%) | 3,154 (30%) |
| Reconnaissance | 8,992 | 6,294 (70%) | 4,496 (50%) | 2,698 (30%) |
| Analysis | 1,715 | 1,715 (100%) | 1,715 (100%) | 1,715 (100%) |
| Backdoor | 1,497 | 1,497 (100%) | 1,497 (100%) | 1,497 (100%) |
| Shellcode | 971 | 971 (100%) | 971 (100%) | 971 (100%) |
| Worms | 111 | 111 (100%) | 111 (100%) | 111 (100%) |
| **Total** | 150,291 | 106,491 (70.9%) | 77,292 (51.4%) | 48,095 (32.0%) |

**Choosing the fraction.** `scripts/sampling_study.py` trains the already
selected RF and MLP configurations (seed 0) on each sample and scores them
on the unchanged validation split. The rule was fixed before the study ran:
use 50 % (the owner's preferred setting) if both models lose at most 0.02
validation macro-F1 against the same configuration trained on all training
data; otherwise 70 % under the same rule; otherwise full data. **Outcome:
50 %** (validation macro-F1 drop: RF 0.011, MLP 0.015; at 30 % the RF lost
0.026, failing the rule). Results: `artifacts/metrics/track_a/sampling_study.json`
and `docs/results.md` §6.

**Leave-one-class-out.** For each held-out class, its records are removed
from the sample and from validation before any fitting. The Track A
preprocessor (imputation medians, means and standard deviations, one-hot
levels) is **refitted per fold** on that fold's training records. This
removes the earlier caveat that the shared preprocessor had seen the
held-out class's (unlabelled) feature statistics. The held-out class stays
complete in the test set.

## Track B: chronological reduction

`deepshield.data.sampling.recent_fraction` keeps, for each capture session
(22 Jan and 18 Feb 2015), the training flows whose end time is in the most
recent 50 % of that session's training period. These are the flows next to
the validation block. Flows are not shuffled. Host windows are built
afterwards from the kept flows in time order, and the 300 s gaps before
validation and test are unchanged, so no window crosses a block boundary
or uses future traffic. The per-class epoch sampling caps are scaled by
the same fraction: at most 30,000 Normal and 15,000 of each attack class
per epoch, instead of 60,000 and 30,000. All Track B models use the same
reduced block, so they stay comparable with each other.

| Class | Full training block | latest 70 % | **latest 50 % (used)** | latest 30 % |
|---|---:|---:|---:|---:|
| Normal | 1,545,381 | 1,142,107 (74%) | 819,358 (53%) | 453,555 (29%) |
| Generic | 153,630 | 128,316 (84%) | 84,726 (55%) | 40,026 (26%) |
| Exploits | 33,206 | 21,209 (64%) | 14,571 (44%) | 7,998 (24%) |
| Fuzzers | 18,926 | 10,454 (55%) | 7,436 (39%) | 4,708 (25%) |
| DoS | 11,469 | 7,455 (65%) | 5,304 (46%) | 3,068 (27%) |
| Reconnaissance | 10,498 | 6,575 (63%) | 4,721 (45%) | 2,845 (27%) |
| Analysis | 2,008 | 1,129 (56%) | 790 (39%) | 350 (17%) |
| Backdoor | 1,666 | 787 (47%) | 722 (43%) | 383 (23%) |
| Shellcode | 1,146 | 686 (60%) | 490 (43%) | 291 (25%) |
| Worms | 132 | 81 (61%) | 57 (43%) | 36 (27%) |
| **Total** | 1,778,062 | 1,318,799 | 938,175 (52.8%) | 513,260 |

50 % was chosen a priori, without validation runs: it is the owner's
preferred setting, and every class keeps at least 39 % of its flows. 30 %
would leave Analysis with 350 training flows and Worms with 36. A
validation-based choice would need several extra LSTM runs, which is
exactly the cost the reduction is meant to avoid. The Track B preprocessor
is the one already fitted on the full training block. That is still
training data only, so it is not leakage, and it keeps the deployed
dashboard model consistent.

**Memory.** The reduced training block is held in RAM as float16, about
360 MB instead of 690 MB. The validation and test blocks are now read on
demand from the memory-mapped feature file (`LazyRows`), with the same
float16 rounding, instead of being copied into RAM (about 285 MB). The OS
can reclaim those pages under memory pressure. `tests/test_sampling.py`
checks that both paths produce identical windows. Measured peak memory is
stored per run as `peak_memory_mb`.

## Limitations

* Reduced-data models may score somewhat lower than full-data models
  would. Track B has no full-data counterpart, so its numbers are not
  directly comparable with full-data Track A results.
* The Track B reduction keeps only the later part of each training period.
  The model sees less of the early traffic, and the classes are not reduced
  equally (39–55 %).
* In the Track A sample, the classes kept whole (Analysis, Backdoor,
  Shellcode, Worms) form a larger share of training than in the full data.
  The 1/√frequency class weights are computed on the sample, which partly
  offsets this, but the effective class balance differs from the full-data
  models.
* LOCO results use one sample (seed 0). Variation due to the sample itself
  is not measured.
* Time savings are reported only where measured (`fit_seconds`,
  `seconds`, epoch times). The estimates above are not results.
