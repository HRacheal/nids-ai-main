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

**UNSW-NB15** — N. Moustafa and J. Slay, Australian Centre for Cyber Security,
UNSW Canberra. Official page: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>.
Free for academic research use; users must cite the five papers listed in
`docs/dataset_profile.md` §11.

The files used were obtained from Kaggle mirrors (the official download
returned HTTP 403): `mrwellsdavid/unsw-nb15` (primary) and
`alextamboli/unsw-nb15` (cross-check and ground-truth table). Their
verification against the official documentation, and the differences found,
are recorded in `docs/dataset_profile.md` §2.

**Open issue — dataset redistribution (UNRESOLVED).** The UNSW-NB15 terms grant
free academic use but say nothing about redistribution, and the Kaggle
mirrors state no licence. No UNSW-NB15 data is committed to this repository
or included in the submission package; users download it themselves
(`docs/dataset_setup.md`).
