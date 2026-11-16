# DeepShield AI: architecture

DeepShield AI extends the original NIDS-AI project. The original code
(`core/`, `ml/`, `dashboard/`, `alerts/`, `mitre/`, `reporting/`, `main.py`)
is kept unchanged and runnable. A byte-for-byte copy is in
`legacy/nids-ai-original/` and is checked by `scripts/verify_legacy.py`. All new
work is in the `deepshield/` package, `scripts/`, `tests/` and `docs/`.

```
               UNSW-NB15 CSVs (data/raw/unsw_nb15, read-only, SHA-256 registry)
                                     │
             deepshield.data.files / loaders      (checksums, parsing repairs, parquet cache)
                                     │
        ┌────────────────────────────┴─────────────────────────────┐
   Track A: official partition                         Track B: 2.54 M full records
   175,341 train / 82,332 test                         chronological blocks per session
   StratifiedGroupKFold validation                     70 / 15 / 15 with 300 s gaps
        │                                                           │
   Preprocessor (fit on train only): impute + missing flags, log1p, standardise, clip ±10,
   one-hot (rare levels → "other"), features ordered by group for the 1D-CNN
        │                                                           │
   RF, Isolation Forest, MLP, 1D-CNN,                  LSTM over per-source-IP flow windows,
   Autoencoder (scripts/train_*.py)                    per-flow MLP / RF, context baselines
        │                                                           │
   deepshield.evaluation: metrics, bootstrap CIs, McNemar + Holm, figures
   deepshield.xai: SHAP (RF), Integrated Gradients (MLP/CNN, Captum), AE error share
   deepshield.intel.mitre: candidate ATT&CK techniques, validated against the STIX bundle
        │
   deepshield.inference.engine
     TrackAEngine   validates uploaded records → predictions + local explanations
     ReplayEngine   streams the Track B test block in time order (FlowRecord only)
     EvaluationMonitor  the only component that sees GroundTruth (scores the demo)
        │
   deepshield.monitoring: SQLite prediction/alert history, severity, PDF/CSV reports
        │
   deepshield.app: Flask + Socket.IO dashboard on 127.0.0.1:5000
```

## Packages

| Module | Responsibility |
|---|---|
| `deepshield/config.py` | project root, `config/default.yaml`, seeding, thread count |
| `deepshield/data/schema.py` | column names, class order, label normalisation |
| `deepshield/data/files.py` | dataset file registry identified by SHA-256 (the train/test files are resolved by content, not name) |
| `deepshield/data/loaders.py` | CSV parsing (BOM, hex ports, blank fields), parquet cache |
| `deepshield/data/preprocess.py` | `Preprocessor` (fit/transform/save/load) |
| `deepshield/data/splits.py` | leakage-free validation split, chronological blocks |
| `deepshield/data/sequences.py` | `WindowIndex`: per-host causal windows of past flows |
| `deepshield/data/imbalance.py` | class weights, random oversampling |
| `deepshield/models/networks.py` | MLP, 1D-CNN, Autoencoder, FlowLSTM |
| `deepshield/training/` | training loop with early stopping, checkpoints, artifact store |
| `deepshield/evaluation/` | metrics, statistical tests, plots |
| `deepshield/xai/explain.py` | SHAP, Integrated Gradients, aggregation of one-hot attributions |
| `deepshield/intel/mitre.py` | category → candidate technique mapping, STIX validation |
| `deepshield/inference/engine.py` | input validation, inference engines, replay, ground-truth isolation |
| `deepshield/monitoring/` | history database, severity rules, reports |
| `deepshield/app/` | dashboard server, template, JS/CSS, vendored Socket.IO and Chart.js |

## Design decisions

* **Two tracks.** Track A uses the official train/test partition, so results
  can be compared with published work. Track B uses the full records in time
  order, which makes per-host sequence modelling and a realistic replay
  possible. The two tracks are never mixed.
* **Leakage control.** The preprocessor is fitted on the training data only.
  In Track A, exact duplicate feature vectors stay in the same fold
  (grouped K-fold). Track B blocks are separated by 300 s gaps, and LSTM
  windows look only backwards. Model selection uses validation data only;
  the test sets are evaluated once, after selection.
* **Ground-truth isolation.** Replay sends models `FlowRecord` objects that
  contain no label. Labels travel separately as `GroundTruth` objects to the
  `EvaluationMonitor`. A label-shuffle test confirms that predictions don't
  change when the labels do.
* **Honest scope.** The dashboard analyses uploaded records and replays a
  dataset. It does not capture live traffic. The original project's scapy
  capture code is untouched but is not part of DeepShield.
* **Local only.** The server binds to 127.0.0.1 by default, accepts uploads
  of up to 25 MB / 50,000 rows, escapes all values in the browser and loads
  no external resources at runtime (JavaScript libraries are vendored).
