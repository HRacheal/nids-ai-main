# Preprocessing Report (Phase 2)

Code: `deepshield/data/{preprocess,splits,sequences,imbalance}.py`;
build script: `scripts/prepare_data.py`; numbers below come from
`reports/preprocessing/split_summary.json`.

Raw files are never modified. Derived data go to `data/interim/` (parsed
caches) and `data/processed/` (model-ready arrays), both git-ignored. The
fitted preprocessors are saved as `artifacts/models/preprocessor_track_{a,b}.joblib`
and are loaded unchanged by every model and by the dashboard.

## 1. Splits

### Track A (official partition)

| Split | Records | Source |
|---|---:|---|
| train | 150,291 | official training set (175,341-record file) minus validation fold |
| validation | 25,050 (14.3 %) | one fold of `StratifiedGroupKFold(7)` on the official training set |
| test | 82,332 | official test set, untouched until final evaluation |

- **Stratified** by attack category: Worms has 111 training / 19 validation records.
- **Grouped by feature vector**: 38.6 % of training records are exact
  duplicates. A random split would put copies of the same record on both sides
  and inflate validation scores. Measured overlap after grouping: **0**
  feature vectors shared between train and validation.
- The validation set is used for early stopping, hyperparameter choice,
  imbalance-strategy choice and every decision threshold. The test set is
  used once per final model.
- 8,541 test records (10.4 %) have a feature vector that also occurs in
  the official training set. This is a property of the official partition. Test
  metrics are reported both on the full test set and on the 73,791
  non-overlapping records.

### Track B (chronological)

Each capture session is cut by flow end time into 70 % / 15 % / 15 % of its
records, with a **300 s gap** after each cut (29,140 records dropped):

| Block | Records | Attacks | Source hosts |
|---|---:|---:|---:|
| train | 1,778,062 | 232,681 | 43 |
| validation | 366,323 | 46,068 | 38 |
| test | 366,522 | 41,450 | 37 |

All ten classes occur in every block (Worms: 132 / 20 / 20). No class was
forced into a block. If one had been missing, its metrics would be reported
as unavailable rather than changing the split.

**Residual leakage risk.** The `ct_*` counters partly count later
connections, and their window could not be reproduced (dataset profile §7).
The gap removes records near each cut, but it cannot be proven to remove all
look-ahead through these counters.

## 2. Feature transformation

| Step | Applies to | Fitted on | Why |
|---|---|---|---|
| Median imputation + `<col>_missing` flag | Track B: `ct_flw_http_mthd`, `ct_ftp_cmd`, `is_ftp_login` (blank in 53–56 % of records) | train | Blankness depends on the source file, not on the traffic, so it is imputed (median 0) and flagged rather than dropped |
| `log1p` | 32 heavy-tailed non-negative numerics (bytes, packets, loads, times, sequence numbers, counters) | none (fixed function) | Skew up to 167. Without it, standardisation is dominated by a few huge flows |
| No log | `sttl`, `dttl`, `swin`, `dwin`, `ct_state_ttl`, `is_ftp_login`, `is_sm_ips_ports` | none | Bounded ranges whose raw values are meaningful (TTL 0–255, flags) |
| Standardisation | all numerics | train mean / std | Neural networks need comparable input scales |
| Clip to ±10 | standardised numerics | none | Protects neural networks from extreme unseen test values |
| One-hot + `other` | `proto`, `service`, `state` | train level counts (≥ 50 records) | Unseen or rare levels map to `other`, e.g. `state = ACC/CLO`, which occur only in the Track A test set |

Retained features: **all 42** Track A features (41 for Track B, which has no
`rate`). No feature was removed. Features whose Spearman ρ ≈ 1 (e.g.
`is_ftp_login` and `ct_ftp_cmd`) were kept because removing them would be
assumption-based feature selection. Tree models are unaffected, and neural
networks handle redundant inputs.

Resulting inputs: Track A 186 columns (39 numeric + 147 one-hot); Track B 194
columns (38 numeric + 3 missing flags + 153 one-hot). The same matrix feeds
every model. Tree models ignore the scaling, and using one matrix guarantees
that all models see identical information.

**Identity columns are not inputs.** `srcip`, `dstip`, ports and timestamps
order and group Track B flows (LSTM windows, replay). They are never model
features: every attack comes from four testbed IPs, so a model given
addresses would learn the testbed, not the attacks.

**Feature order** is grouped by meaning: basic flow, load/content, TCP,
timing, counters, flags, missing flags, categorical. This puts related
quantities next to each other for the 1D-CNN (Phase 4), whose random-order
ablation tests whether that adjacency matters.

## 3. Targets

- Multiclass: index into `Normal, Generic, Exploits, Fuzzers, DoS,
  Reconnaissance, Analysis, Backdoor, Shellcode, Worms`.
- Binary: `label` (0 normal, 1 attack), consistent with `attack_cat` in
  100 % of records.

## 4. Class imbalance

Strategies implemented in `deepshield/data/imbalance.py`:

- none
- 1/√frequency loss weights
- fully balanced weights (1/frequency)
- random oversampling of minority classes

Each model family uses whichever strategy wins on **validation** macro-F1
(Phases 3–4; results in §6).

SMOTE is not used. It interpolates between records, but here many features
are discrete counters and one-hot categories, and 38 % of records are
duplicates, so interpolated points are not valid flows. Resampling, if
used, touches only training data. Validation and test keep their natural
distributions.

## 5. Leakage controls (summary)

1. Every fitted statistic (medians, means, standard deviations, category
   levels) comes from the training portion only. Unit test:
   `test_preprocessor_uses_training_statistics_only`.
2. Validation is grouped by feature vector (no duplicate crosses the split).
3. The official test set is never used for fitting, tuning or thresholds.
4. Track B is split by time, with gaps. LSTM windows never cross block
   boundaries (`test_windows_stay_within_host_and_order`).
5. IPs, ports and timestamps are not model inputs.
6. Leave-one-class-out experiments (Phase 5) remove the held-out class from
   the training and validation data before any resampling or weighting.

## 6. Imbalance strategy results (validation)

See `reports/preprocessing/imbalance_study.json`; summarised in
`docs/results.md`.
