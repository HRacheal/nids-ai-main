# Credits and Attribution

DeepShield AI (DSA4050_XA Deep Learning project) extends an existing
open-source project:

**NIDS — AI-Powered Network Intrusion Detection System**
by **M. Mohamed Nabeel** (B.E. Computer Science, Cybersecurity, SRM Madurai
College for Engineering and Technology) —
<https://github.com/mohamednabeelM/nids-ai> — released under the **MIT License**.

## How the original is preserved

| Where | What |
|---|---|
| Git tag `original-nids-ai` (commit `37397e4`) | The original project exactly as received, before any change |
| `legacy/nids-ai-original/` | A complete, unmodified, runnable copy of the original |
| `legacy/nids-ai-original.sha256` | SHA-256 manifest proving the copy is byte-identical to the baseline |
| `LICENSE`, `legacy/nids-ai-original/LICENSE` | The original MIT copyright and permission notice, kept as the licence requires |

`scripts/verify_legacy.py` re-checks the manifest and confirms the legacy copy
still trains, detects, reports and serves its dashboard (in a temporary copy,
so the legacy folder is never written to).

## Reuse status

Updated at the end of every phase.

| Component | Origin | Status (Phase 0) |
|---|---|---|
| `core/`, `alerts/`, `mitre/`, `dashboard/`, `reporting/`, `main.py`, `download_dataset.py` | Original | Unchanged |
| `ml/` | Original | Unchanged; will be superseded by `deepshield/` |
| `LICENSE`, `README.md` | Original | Unchanged |
| `requirements.txt` | Original, modified | Exact, tested version pins |
| `.gitignore` | Original, modified | Broken first/last lines removed; ignores `.venv`, data, models |
| `deepshield/`, `tests/`, `scripts/`, `config/`, `docs/`, `CREDITS.md`, `pytest.ini`, `requirements-lock.txt` | New (DeepShield AI) | Phase 0 skeleton, environment tests, legacy verification |

## Datasets

UNSW-NB15 and CIC-IDS2017 citations are added when the datasets are
introduced (Phases 1 and 6).
