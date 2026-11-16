"""Candidate MITRE ATT&CK associations for UNSW-NB15 categories.

These are CANDIDATE associations, not detections of attacker behaviour:
UNSW-NB15 categories were produced by a traffic generator (IXIA
PerfectStorm) and a predicted category says only that a flow resembles that
category's flows. Each mapping states its basis and a confidence level:

* medium - the UNSW category definition describes the technique's behaviour
* low    - plausible but the category is broad or the link indirect
* none   - no defensible technique-level mapping (shown as such)

Technique IDs and names are validated against MITRE's official STIX bundle
(`scripts/validate_mitre.py`), which records the ATT&CK version used.
Category definitions: Moustafa & Slay (2015, 2016).
"""

from __future__ import annotations

import json
from pathlib import Path

VALIDATED_PATH = Path(__file__).with_name("mitre_validated.json")

CANDIDATES: dict[str, dict] = {
    "Reconnaissance": {
        "definition": "Attacks that gather information about a target (probes and scans).",
        "techniques": ["T1595", "T1046"], "confidence": "medium",
        "basis": "Scanning hosts and services is the behaviour ATT&CK calls Active Scanning "
                 "(external) or Network Service Discovery (internal).",
    },
    "DoS": {
        "definition": "Attempts to make a server or network resource unavailable.",
        "techniques": ["T1498", "T1499"], "confidence": "medium",
        "basis": "Direct match to the Impact techniques for network- and endpoint-level denial of service.",
    },
    "Exploits": {
        "definition": "The attacker exploits a known vulnerability of an operating system or software.",
        "techniques": ["T1190", "T1210"], "confidence": "medium",
        "basis": "Exploitation of exposed services; whether initial access (T1190) or lateral "
                 "movement (T1210) depends on context the flow record does not contain.",
    },
    "Worms": {
        "definition": "Self-replicating code that spreads to other computers.",
        "techniques": ["T1210", "T1570"], "confidence": "medium",
        "basis": "Worm propagation exploits remote services and transfers its code between hosts.",
    },
    "Fuzzers": {
        "definition": "Feeding randomly generated data to a program or network to make it crash.",
        "techniques": ["T1595.002"], "confidence": "low",
        "basis": "Fuzzing a remote service is a form of vulnerability discovery; ATT&CK has no "
                 "fuzzing-specific technique.",
    },
    "Analysis": {
        "definition": "Port scans, spam and HTML-file penetration attacks.",
        "techniques": ["T1046"], "confidence": "low",
        "basis": "Only the port-scan part of this mixed category maps to a technique.",
    },
    "Shellcode": {
        "definition": "A small payload used to exploit a software vulnerability.",
        "techniques": ["T1203", "T1190"], "confidence": "low",
        "basis": "Shellcode is the payload of an exploit; the technique depends on the target "
                 "(client application or exposed service).",
    },
    "Backdoor": {
        "definition": "Bypassing normal authentication to gain covert remote access.",
        "techniques": ["T1133", "T1219"], "confidence": "low",
        "basis": "Covert remote access resembles abuse of remote services or remote access tools; "
                 "the flow record cannot show which.",
    },
    "Generic": {
        "definition": "A technique that works against all block ciphers (with a given block and key size), "
                      "regardless of the cipher's structure.",
        "techniques": [], "confidence": "none",
        "basis": "A cryptanalytic attack class; ATT&CK describes adversary behaviour, not cipher attacks, "
                 "so no technique-level mapping is defensible.",
    },
    "Normal": {"definition": "Benign traffic.", "techniques": [], "confidence": "none", "basis": ""},
}

SEVERITY = {  # default alert severity per predicted category (dashboard)
    "Normal": "CLEAN", "Generic": "MEDIUM", "Exploits": "HIGH", "Fuzzers": "MEDIUM",
    "DoS": "HIGH", "Reconnaissance": "MEDIUM", "Analysis": "MEDIUM",
    "Backdoor": "CRITICAL", "Shellcode": "CRITICAL", "Worms": "CRITICAL",
}


def validate_against_stix(stix_path: Path) -> dict:
    """Check every candidate technique against the official ATT&CK STIX bundle."""
    bundle = json.loads(Path(stix_path).read_text(encoding="utf-8"))
    collection = next(o for o in bundle["objects"] if o["type"] == "x-mitre-collection")
    techniques = {}
    for o in bundle["objects"]:
        if o["type"] != "attack-pattern":
            continue
        ref = next((r for r in o.get("external_references", [])
                    if r.get("source_name") == "mitre-attack"), None)
        if ref:
            techniques[ref["external_id"]] = {
                "name": o["name"], "url": ref.get("url"),
                "tactics": [p["phase_name"] for p in o.get("kill_chain_phases", [])],
                "revoked": o.get("revoked", False),
                "deprecated": o.get("x_mitre_deprecated", False)}
    out = {"attack_version": collection.get("x_mitre_version"),
           "attack_modified": collection.get("modified"),
           "source": "https://github.com/mitre-attack/attack-stix-data (enterprise-attack.json)",
           "categories": {}}
    for cat, entry in CANDIDATES.items():
        rows = []
        for tid in entry["techniques"]:
            t = techniques.get(tid)
            rows.append({"id": tid, "valid": bool(t) and not t["revoked"] and not t["deprecated"],
                         **(t or {"name": None})})
        out["categories"][cat] = {**entry, "techniques": rows}
    out["all_valid"] = all(r["valid"] for c in out["categories"].values() for r in c["techniques"])
    return out


def load_validated() -> dict:
    if VALIDATED_PATH.exists():
        return json.loads(VALIDATED_PATH.read_text(encoding="utf-8"))
    return {"attack_version": None, "categories": {c: {**e, "techniques": [
        {"id": t, "name": None, "valid": None} for t in e["techniques"]]} for c, e in CANDIDATES.items()}}


def enrich(category: str) -> dict:
    """Display-ready MITRE context for a predicted category."""
    v = load_validated()
    entry = v["categories"].get(category, v["categories"]["Normal"])
    return {"category": category, "confidence": entry["confidence"], "basis": entry["basis"],
            "techniques": [{"id": t["id"], "name": t.get("name"), "url": t.get("url"),
                            "tactics": t.get("tactics", [])} for t in entry["techniques"]],
            "attack_version": v.get("attack_version"),
            "disclaimer": "Candidate association for a predicted traffic category; "
                          "not confirmed attacker behaviour."}
