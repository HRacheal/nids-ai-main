# Development Environment (Phase 0)

Verified on 15 Nov 2026. All DeepShield work runs inside the project virtual
environment `.venv`; the global Python installation is not modified.

## Platform

| Item | Value |
|---|---|
| OS | Windows 11 Pro (10.0.26300), x64 |
| Python | 3.12.0 (`C:\Program Files\Python312`), used only to create `.venv` |
| Hardware acceleration | None — CPU-only PyTorch build |

## Pinned direct dependencies

| Package | Version | Purpose |
|---|---|---|
| torch | 2.13.0+cpu | Deep learning (MLP, CNN, LSTM, Autoencoder) |
| captum | 0.9.0 | Integrated Gradients |
| shap | 0.50.0 | TreeExplainer (RF), GradientExplainer (neural networks) |
| scikit-learn | 1.9.1 | Random Forest, Isolation Forest, metrics |
| scipy | 1.18.1 | Statistics |
| numpy | 2.2.6 | Arrays |
| pandas | 2.3.3 | Tables |
| pyarrow | 21.0.0 | Parquet storage of processed data |
| joblib | 1.4.2 | Saving scikit-learn models |
| PyYAML | 6.0.3 | Configuration |
| matplotlib | 3.9.4 | Plots |
| Flask / Flask-SocketIO | 3.1.0 / 5.7.2 | Dashboard |
| reportlab | 5.1.0 | PDF reports |
| rich | 15.0.0 | Terminal output |
| scapy | 2.8.0 | Optional live capture |
| pytest | 8.4.1 | Tests |

`requirements.txt` holds these pins; `requirements-lock.txt` holds the full
`pip freeze` (60 packages) of the verified environment.

## Compatibility tests

`tests/test_environment.py` (30 tests) checks the features DeepShield
depends on, not just imports:

| Area | What is checked |
|---|---|
| Python / venv | Python 3.12; tests run inside `.venv` |
| Imports | All 15 libraries import; torch imports **after** scikit-learn and shap in a fresh process |
| PyTorch | MLP trains (loss falls by > 50 %); save → load gives identical outputs; seeded training is bit-for-bit reproducible; Conv1d and 2-layer LSTM with padded/masked sequences run forward and backward |
| Captum | Integrated Gradients completeness: attributions sum to f(x) − f(baseline) within 1 % relative error (512 steps); reported convergence delta equals that gap |
| SHAP | TreeExplainer on a multiclass RF returns shape (samples, features, classes) and satisfies additivity (sum + expected value = predicted probabilities, within 1e-6); GradientExplainer works on a PyTorch model |
| scikit-learn / joblib | RF and Isolation Forest save → load give identical outputs |
| Flask-SocketIO | HTTP route and Socket.IO event round trip via test clients |
| ReportLab | Builds a valid PDF |
| pandas / pyarrow | Parquet round trip preserves float32 and categorical types |
| matplotlib | Headless PNG export |
| scapy | Builds and parses a TCP packet (no capture) |

Run: `.venv\Scripts\python.exe -m pytest`

## Issues found and resolved

1. **scikit-learn 1.5.0 breaks torch 2.13 on Windows.** The scikit-learn 1.5.0
   wheel bundles `msvcp140.dll` version 14.32. If scikit-learn (or shap, which
   imports it) is imported before torch, that old C++ runtime is already
   loaded and torch's `c10.dll` fails with `WinError 1114`. scikit-learn 1.9.1
   bundles version 14.51 (matching the system copy) and fixes this. Pinned
   1.9.1 inside `.venv`; a regression test imports in the failing order. The
   global environment still has scikit-learn 1.5.0 and is unaffected because
   it is not used.
2. **Integrated Gradients tolerance.** On ReLU networks the integral has kinks,
   so completeness holds only up to numerical-integration error (≈ 0.2 % at
   128 steps). The test uses a relative tolerance and more steps rather than an
   absolute 1e-3 bound.

## Legacy verification

`scripts/verify_legacy.py` (run inside `.venv`) — all checks passed:

| Check | Result |
|---|---|
| Integrity before runs | 18/18 files match `legacy/nids-ai-original.sha256`; none missing, extra or changed |
| `main.py train` (synthetic, 300/class) | Exit 0 in 5.4 s; all 7 model files written |
| `main.py run --no-dashboard` (75 s) | Exit 0; alerts printed; valid PDF report written on interrupt |
| `main.py run` dashboard (30 s, bound to 127.0.0.1 for the test) | `/`, `/api/stats`, `/api/alerts`, `/api/model_info` and the Socket.IO handshake all returned HTTP 200 |
| Integrity after runs | 18/18 files still identical (all runs used a temporary copy) |

Known limitations of the **original** code observed during verification
(recorded, not fixed — the legacy copy stays unmodified):

1. On Windows, the CLI crashes with `UnicodeEncodeError` when its output is
   not UTF-8 (pipes, files, cp1252 consoles), because `rich` prints
   box-drawing characters. The verification runs it with `PYTHONUTF8=1`.
2. Normal simulated hosts (192.168.1.x) are flagged as `HIGH vuln_scan`,
   consistent with the train/serve feature mismatch identified in the audit.
3. scikit-learn warns "X does not have valid feature names" on every
   prediction (model fitted on a DataFrame, called with arrays).
