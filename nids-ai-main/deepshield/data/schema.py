"""UNSW-NB15 column schema for both experimental tracks.

Track A = the official training/testing partition (45 columns, with header).
Track B = the four main files UNSW-NB15_1..4.csv (49 columns, no header);
their column names come from NUSW-NB15_features.csv.

Names below were taken from the actual downloaded files (Phase 1), not from
online descriptions. Raw names are lower-cased and stripped; the features
file contains one typo with an embedded space ("ct_src_ ltm").
"""

from __future__ import annotations

import pandas as pd

# Official class labels. "Backdoor" is used as the canonical spelling because
# the Track A files use it; the main files contain both "Backdoor" and
# "Backdoors" (verified in Phase 1) and are normalised to it.
NORMAL = "Normal"
CLASS_ORDER = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS",
               "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
ATTACK_CLASSES = CLASS_ORDER[1:]

CATEGORICAL = ["proto", "service", "state"]
TARGET_BINARY = "label"
TARGET_MULTI = "attack_cat"

# Track A feature columns in file order (excluding id, attack_cat, label).
TRACK_A_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]
TRACK_A_NUMERIC = [c for c in TRACK_A_FEATURES if c not in CATEGORICAL]

# Track B raw column names (normalised from NUSW-NB15_features.csv, in order).
RAW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload",
    "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt",
    "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]

# Raw name -> Track A name, where they differ (same quantity, different name).
RAW_TO_TRACK_A = {
    "smeansz": "smean", "dmeansz": "dmean", "res_bdy_len": "response_body_len",
    "sintpkt": "sinpkt", "dintpkt": "dinpkt",
}

# Identity / time columns present only in Track B. They are used to order and
# group flows (LSTM, replay) and for display, never as model inputs: IPs and
# ports would let a model memorise the testbed's attacker addresses.
TRACK_B_ID_COLUMNS = ["srcip", "sport", "dstip", "dsport", "stime", "ltime"]

# Track B model features: the Track A features that exist in the raw files
# (all except `rate`, which only the training/testing files contain).
TRACK_B_FEATURES = [c for c in TRACK_A_FEATURES if c != "rate"]
TRACK_B_NUMERIC = [c for c in TRACK_B_FEATURES if c not in CATEGORICAL]


def normalise_feature_name(name: str) -> str:
    """Normalise a name from NUSW-NB15_features.csv to the RAW_COLUMNS form."""
    return name.strip().lower().replace(" ", "")


def normalise_attack_cat(s: pd.Series) -> pd.Series:
    """Canonical attack-category labels.

    Strips surrounding whitespace (" Fuzzers ", found in the main files),
    maps "Backdoors" -> "Backdoor", and maps empty/missing (how the main
    files mark normal traffic) -> "Normal".
    """
    out = s.astype("string").str.strip()
    out = out.replace({"Backdoors": "Backdoor", "": NORMAL}).fillna(NORMAL)
    return out.astype(str)
