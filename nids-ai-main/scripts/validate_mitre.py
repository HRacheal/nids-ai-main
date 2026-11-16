#!/usr/bin/env python3
"""Validate the candidate ATT&CK mappings against MITRE's official STIX data.

Downloads enterprise-attack.json (if not cached in data/external/mitre/),
checks each technique ID exists and is neither revoked nor deprecated,
records official names, tactics and the ATT&CK version, and writes
deepshield/intel/mitre_validated.json (committed; used by the dashboard).

Usage:  .venv\\Scripts\\python.exe scripts\\validate_mitre.py
"""

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield.config import PROJECT_ROOT, ensure_dir  # noqa: E402
from deepshield.intel.mitre import VALIDATED_PATH, validate_against_stix  # noqa: E402

URL = ("https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
       "enterprise-attack/enterprise-attack.json")


def main():
    stix = ensure_dir(PROJECT_ROOT / "data" / "external" / "mitre") / "enterprise-attack.json"
    if not stix.exists():
        print("downloading", URL)
        urllib.request.urlretrieve(URL, stix)
    result = validate_against_stix(stix)
    VALIDATED_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ATT&CK {result['attack_version']} ({result['attack_modified']}); all valid: {result['all_valid']}")
    for cat, e in result["categories"].items():
        names = ", ".join(f"{t['id']} {t['name']}" + ("" if t["valid"] else " [INVALID]")
                          for t in e["techniques"]) or "-"
        print(f"  {cat:15s} [{e['confidence']}] {names}")
    sys.exit(0 if result["all_valid"] else 1)


if __name__ == "__main__":
    main()
