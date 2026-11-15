#!/usr/bin/env python3
"""Verify the preserved legacy copy of the original NIDS-AI project.

Checks, WITHOUT modifying legacy/nids-ai-original/:
  1. Integrity  - every file matches legacy/nids-ai-original.sha256
                  (SHA-256 manifest recorded when the copy was made from the
                  `original-nids-ai` git baseline), with no files added/removed.
  2. Training   - `main.py train` (synthetic data) completes.
  3. Detection  - `main.py run --no-dashboard` runs for a fixed time, is
                  stopped with a KeyboardInterrupt (as Ctrl+C would) and
                  writes its PDF report.
  4. Dashboard  - `main.py run` serves /, /api/stats, /api/alerts,
                  /api/model_info and the Socket.IO handshake. For this test
                  only, the server is bound to 127.0.0.1 instead of 0.0.0.0
                  by wrapping dashboard.app.run at runtime (no file changes).

Steps 2-4 run in a throwaway copy under the system temp directory, so model
files and PDFs produced by the legacy code never land in the legacy folder.

Usage:  .venv\\Scripts\\python.exe scripts\\verify_legacy.py [--out result.json]
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy" / "nids-ai-original"
MANIFEST = ROOT / "legacy" / "nids-ai-original.sha256"

DETECT_SECONDS = 75     # attacks start after 10-20 s; flows expire after 30 s idle
DASHBOARD_SECONDS = 30
DASHBOARD_PORT = 5057

DRIVER = r"""
import json, sys, threading, time, urllib.request, _thread
ARGS, DURATION, PROBE_PORT = {args!r}, {duration!r}, {probe_port!r}
sys.argv = ["main.py"] + ARGS
sys.path.insert(0, ".")
import main

if PROBE_PORT:
    import dashboard.app as dash
    _orig_run = dash.run
    def _loopback_run(host="0.0.0.0", port=5000, debug=False):
        return _orig_run(host="127.0.0.1", port=port, debug=debug)
    dash.run = _loopback_run

    def _probe():
        time.sleep(DURATION - 8)
        base = "http://127.0.0.1:%d" % PROBE_PORT
        out = {{}}
        for path in ["/", "/api/stats", "/api/alerts?n=5", "/api/model_info",
                     "/socket.io/?EIO=4&transport=polling"]:
            try:
                with urllib.request.urlopen(base + path, timeout=5) as r:
                    body = r.read()
                    out[path] = {{"status": r.status, "bytes": len(body),
                                  "head": body[:120].decode("utf-8", "replace")}}
            except Exception as e:
                out[path] = {{"error": repr(e)}}
        with open("probe.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    threading.Thread(target=_probe, daemon=True).start()

_timer = threading.Timer(DURATION, _thread.interrupt_main)
_timer.daemon = True          # a crash must end the process, not leave it waiting
_timer.start()
main.main()
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check_integrity() -> dict:
    expected = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        expected[rel] = digest
    actual = {p.relative_to(LEGACY).as_posix(): sha256(p)
              for p in LEGACY.rglob("*") if p.is_file()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(k for k in expected.keys() & actual.keys()
                     if expected[k] != actual[k])
    return {"files_expected": len(expected), "files_found": len(actual),
            "missing": missing, "extra": extra, "changed": changed,
            "passed": not (missing or extra or changed)}


def run_driver(workdir: Path, args: list, duration: float,
               probe_port: int | None, timeout: float) -> dict:
    code = DRIVER.format(args=args, duration=duration, probe_port=probe_port)
    # The legacy CLI prints Unicode box-drawing characters through `rich`. On
    # Windows, output that is not a UTF-8 terminal (pipes, files, cp1252
    # consoles) makes it crash with UnicodeEncodeError before doing anything.
    # This is a known limitation of the original code (not changed here); the
    # verification therefore runs it with UTF-8 output, as a modern terminal
    # would provide.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", code], cwd=workdir, env=env,
                          capture_output=True, timeout=timeout)
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    return {"args": args, "returncode": proc.returncode,
            "seconds": round(time.time() - t0, 1),
            "stdout_tail": out[-1500:], "stderr_tail": err[-1500:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write the JSON result to this path")
    ap.add_argument("--integrity-only", action="store_true")
    opts = ap.parse_args()

    result = {"python": sys.executable,
              "legacy_run_env": "PYTHONUTF8=1, PYTHONIOENCODING=utf-8",
              "integrity": check_integrity()}

    if not opts.integrity_only:
        with tempfile.TemporaryDirectory(prefix="deepshield_legacy_") as tmp:
            work = Path(tmp) / "nids-ai-original"
            shutil.copytree(LEGACY, work)

            result["train"] = run_driver(work, ["train", "--samples", "300"],
                                         duration=600, probe_port=None,
                                         timeout=660)
            models = sorted(p.name for p in (work / "ml" / "models").glob("*.pkl"))
            result["train"]["model_files"] = models
            result["train"]["passed"] = (result["train"]["returncode"] == 0
                                         and len(models) == 7)

            det = run_driver(work, ["run", "--no-dashboard"],
                             duration=DETECT_SECONDS, probe_port=None,
                             timeout=DETECT_SECONDS + 240)
            pdfs = sorted((work / "reports").glob("nids_report_*.pdf"))
            det["pdf_reports"] = [p.name for p in pdfs]
            det["pdf_valid"] = bool(pdfs) and pdfs[-1].read_bytes()[:5] == b"%PDF-"
            det["status_lines"] = det["stdout_tail"].count("flows=")
            det["passed"] = det["returncode"] == 0 and det["pdf_valid"]
            result["detect"] = det

            dash = run_driver(work, ["run", "--port", str(DASHBOARD_PORT)],
                              duration=DASHBOARD_SECONDS,
                              probe_port=DASHBOARD_PORT,
                              timeout=DASHBOARD_SECONDS + 240)
            probe_file = work / "probe.json"
            dash["probe"] = (json.loads(probe_file.read_text(encoding="utf-8"))
                             if probe_file.exists() else None)
            dash["passed"] = (dash["returncode"] == 0 and dash["probe"] is not None
                              and all(v.get("status") == 200
                                      for v in dash["probe"].values()))
            result["dashboard"] = dash

    # The legacy folder itself must still be untouched after all runs.
    result["integrity_after_runs"] = check_integrity()
    steps = ["integrity", "integrity_after_runs"] + (
        [] if opts.integrity_only else ["train", "detect", "dashboard"])
    result["passed"] = all(result[s]["passed"] for s in steps)

    text = json.dumps(result, indent=2)
    if opts.out:
        Path(opts.out).write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
