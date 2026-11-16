#!/usr/bin/env python3
"""Phase 5: compare all trained models (reads saved predictions; trains nothing).

Writes
  artifacts/metrics/comparison.json   tables, paired tests, error analysis
  reports/results_tables.md           markdown tables used in docs/results.md
  reports/figures/models/*.png        ROC/PR curves, per-class F1, error analysis

Statistics (seed-0 models, Track A official test set unless stated):
  * 95 % stratified bootstrap CI of macro-F1 per model;
  * paired bootstrap CI of the macro-F1 difference for every pair of
    multiclass models (Track A) and Track B models (time-block bootstrap);
  * McNemar exact test on paired errors (Track A only: records are
    independent draws there, unlike Track B's temporally correlated flows);
  * Holm correction across each family of comparisons.

Usage:  .venv\\Scripts\\python.exe scripts\\compare_models.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield import viz  # noqa: E402
from deepshield.config import PROJECT_ROOT, path  # noqa: E402
from deepshield.data.schema import CLASS_ORDER  # noqa: E402
from deepshield.evaluation import metrics as M  # noqa: E402
from deepshield.evaluation import plots  # noqa: E402
from deepshield.training import store  # noqa: E402

A_MULTI = {"rf_multiclass": "Random Forest", "mlp": "MLP", "cnn": "1D-CNN",
           "cnn_random_order": "1D-CNN (random order)"}
A_BINARY = {"rf_binary": "RF (binary)", "autoencoder": "Autoencoder", "iforest": "Isolation Forest"}
B_MULTI = {"lstm": "LSTM", "mlp_flow": "Per-flow MLP", "rf_flow": "Per-flow RF",
           "context_baseline": "Context baseline", "oracle_context": "Oracle context (upper bound)"}


def exists(track, name):
    return store.metrics_path(track, name).exists()


def training_data_label(m: dict) -> str:
    """'full' or 'reduced (...)' from a metrics record (records without the field used all training data)."""
    td = m.get("training_data") or {}
    if td.get("full_or_reduced", "full") == "full":
        return "full"
    if "train_fraction_of_time_span" in td:
        return f"reduced: latest {td['train_fraction_of_time_span']:.0%} of training period"
    return f"reduced: {td.get('train_fraction', 0):.0%} stratified sample"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def f3(v):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def main():
    d = store.load_track_a_arrays()
    yA, ybA, ov = d["y_test"], d["yb_test"], d["test_overlap"]
    out: dict = {"track_a": {}, "track_b": {}}
    md: list[str] = []
    figs = store.FIG_DIR

    # ── Track A multiclass ────────────────────────────────────────────────────
    rows, preds, reps = [], {}, {}
    for name, title in A_MULTI.items():
        if not exists("A", name):
            continue
        m = store.load_metrics("A", name)
        p = store.load_predictions("A", f"{name}_s0", "test")["proba"]
        preds[title] = p.argmax(1)
        t = m["test"]
        reps[title] = t
        ss = m.get("seed_summary", {})
        rows.append({"Model": title, "Training data": training_data_label(m), "Accuracy": f3(t["accuracy"]),
                     "Macro-F1": f"{t['macro_f1']:.3f} ± {ss.get('macro_f1_std', 0):.3f}",
                     "95% CI": f"[{t['bootstrap_macro_f1']['low']:.3f}, {t['bootstrap_macro_f1']['high']:.3f}]",
                     "Macro precision": f3(t.get("macro_precision")), "Macro recall": f3(t.get("macro_recall")),
                     "Weighted F1": f3(t["weighted_f1"]), "ROC-AUC (OvR)": f3(t.get("roc_auc_ovr_macro")),
                     "PR-AUC": f3(t.get("pr_auc_macro")),
                     "Macro-F1 (non-overlap test)": f3(t["non_overlapping_test"]["macro_f1"]),
                     "Attack-vs-normal F1": f3(t["binary_view"]["f1"]),
                     "Train s": f"{m['fit_seconds']:.0f}", "Predict µs/rec": f"{m['predict_us_per_record']:.1f}",
                     "Parameters": f"{m['n_parameters']:,}" if m.get("n_parameters") else "–"})
    tab = pd.DataFrame(rows)
    out["track_a"]["multiclass_table"] = rows
    md += ["## Track A: multiclass (official test set, 82,332 records)", "", md_table(tab), ""]

    # per-class F1 table + figure
    pc = pd.DataFrame({t: {c: r["per_class"][c].get("f1") for c in CLASS_ORDER} for t, r in reps.items()})
    pc.insert(0, "Support", [reps[next(iter(reps))]["per_class"][c]["support"] for c in CLASS_ORDER])
    out["track_a"]["per_class_f1"] = pc.to_dict()
    md += ["### Per-class F1 (Track A test)", "",
           md_table(pc.reset_index().rename(columns={"index": "Class"}).round(3)), ""]
    plots.per_class_f1_plot({t: reps[t] for t in reps if "random" not in t}, "Per-class F1, Track A test (seed 0)",
                            figs / "per_class_f1_track_a.png")

    # paired tests
    pairs, mc, boot = {}, {}, {}
    for (a, pa), (b, pb) in itertools.combinations(preds.items(), 2):
        k = f"{a} vs {b}"
        boot[k] = M.paired_bootstrap_diff(yA, pa, pb, n_boot=1000)
        mc[k] = M.mcnemar(yA, pa, pb)
    holm = M.holm({k: v["p_value"] for k, v in mc.items()})
    for k in mc:
        pairs[k] = {**boot[k], "mcnemar": mc[k], "mcnemar_holm": holm[k]}
    out["track_a"]["paired_tests"] = pairs
    md += ["### Paired comparisons (Track A test, seed 0)", "",
           md_table(pd.DataFrame([{"Comparison": k, "ΔMacro-F1": f"{v['diff_macro_f1']:+.3f}",
                                   "95% CI": f"[{v['low']:+.3f}, {v['high']:+.3f}]",
                                   "McNemar p (Holm)": f"{v['mcnemar_holm']['p_holm']:.2g}",
                                   "A right / B right": f"{v['mcnemar']['a_right_b_wrong']:,} / {v['mcnemar']['a_wrong_b_right']:,}"}
                                  for k, v in pairs.items()])), ""]

    # error analysis: most frequent confusions of the best model
    best = max(reps, key=lambda t: reps[t]["macro_f1"])
    cm = np.array(reps[best]["confusion_matrix"])
    conf = [(CLASS_ORDER[i], CLASS_ORDER[j], int(cm[i, j])) for i in range(10) for j in range(10) if i != j]
    conf.sort(key=lambda x: -x[2])
    out["track_a"]["top_confusions_best_model"] = {"model": best, "pairs": conf[:10]}
    bv = reps[best]["binary_view"]
    out["track_a"]["false_alarms_and_misses_best_model"] = {
        "false_positives": bv["confusion_matrix"][0][1], "false_negatives": bv["confusion_matrix"][1][0],
        "missed_by_category": {CLASS_ORDER[i]: int(cm[i, 0]) for i in range(1, 10)}}
    md += [f"### Most frequent confusions ({best})", "",
           md_table(pd.DataFrame([{"True": a, "Predicted": b, "Records": f"{n:,}"} for a, b, n in conf[:10]])), ""]

    # ── Track A binary / anomaly ─────────────────────────────────────────────
    rows, curves = [], {}
    for name, title in list(A_MULTI.items())[:3]:
        if exists("A", name):
            p = store.load_predictions("A", f"{name}_s0", "test")["proba"]
            curves[f"{title} (1 − P(Normal))"] = (ybA, 1 - p[:, 0])
            bv = store.load_metrics("A", name)["test"]["binary_view"]
            rows.append({"Model": f"{title} (from multiclass)", "Type": "supervised", "Precision": f3(bv["precision"]),
                         "Detection rate": f3(bv["recall_detection_rate"]), "F1": f3(bv["f1"]),
                         "FPR": f"{bv['false_positive_rate']:.3f}", "ROC-AUC": f3(bv.get("roc_auc")),
                         "PR-AUC": f3(bv.get("pr_auc"))})
    for name, title in A_BINARY.items():
        if not exists("A", name):
            continue
        m = store.load_metrics("A", name)
        t = m["test"]
        curves[title] = (ybA, store.load_predictions("A", f"{name}_s0", "test")["score"])
        rows.append({"Model": title, "Type": "supervised" if name == "rf_binary" else "anomaly (normal-only training)",
                     "Precision": f3(t["precision"]), "Detection rate": f3(t["recall_detection_rate"]),
                     "F1": f3(t["f1"]), "FPR": f"{t['false_positive_rate']:.3f}",
                     "ROC-AUC": f3(t["roc_auc"]), "PR-AUC": f3(t["pr_auc"])})
    out["track_a"]["binary_table"] = rows
    md += ["## Track A: attack vs normal", "", md_table(pd.DataFrame(rows)), ""]
    plots.roc_pr_plot(curves, "Attack vs normal, Track A test", figs / "roc_pr_track_a.png")

    # anomaly detectors: detection rate per category
    det = {}
    for name, title in (("autoencoder", "Autoencoder"), ("iforest", "Isolation Forest")):
        if exists("A", name):
            det[title] = {k: v["flagged_fraction"] for k, v in store.load_metrics("A", name)["test"]["by_category"].items()}
    if det:
        df = pd.DataFrame(det).round(3)
        out["track_a"]["anomaly_detection_by_category"] = det
        md += ["### Anomaly detectors: share of each category flagged (Normal row = false-positive rate)", "",
               md_table(df.reset_index().rename(columns={"index": "Category"})), ""]

    # ── Track B ───────────────────────────────────────────────────────────────
    meta = pd.read_parquet(path("processed") / "track_b" / "meta.parquet", columns=["ltime", "block"])
    yB = np.load(path("processed") / "track_b" / "y.npy")[(meta["block"] == "test").to_numpy()]
    binsB = (meta.loc[meta["block"] == "test", "ltime"].to_numpy() // 600)
    rows, predsB, repsB = [], {}, {}
    for name, title in B_MULTI.items():
        if not exists("B", name):
            continue
        m = store.load_metrics("B", name)
        t = m["test"]
        repsB[title] = t
        predsB[title] = store.load_predictions("B", f"{name}_s0", "test")["proba"].argmax(1)
        ss = m.get("seed_summary", {})
        ci = t["bootstrap_macro_f1_time_blocks"]
        rows.append({"Model": title, "Training data": training_data_label(m), "Accuracy": f3(t["accuracy"]),
                     "Macro-F1": f"{t['macro_f1']:.3f} ± {ss.get('macro_f1_std', 0):.3f}",
                     "Macro precision": f3(t.get("macro_precision")), "Macro recall": f3(t.get("macro_recall")),
                     "95% CI (time blocks)": f"[{ci['low']:.3f}, {ci['high']:.3f}]",
                     "Weighted F1": f3(t["weighted_f1"]), "PR-AUC": f3(t.get("pr_auc_macro")),
                     "Attack-vs-normal F1": f3(t["binary_view"]["f1"]),
                     "FPR": f"{t['binary_view']['false_positive_rate']:.4f}"})
    if rows:
        out["track_b"]["table"] = rows
        md += ["## Track B: chronological test block (366,522 flows)", "",
               "All Track B models were trained on reduced training data (docs/sampling.md); the test "
               "block is complete. Track B has no full-data counterpart, so these numbers are not "
               "directly comparable with the full-data Track A results.", "", md_table(pd.DataFrame(rows)), ""]
        pcB = pd.DataFrame({t: {c: r["per_class"][c].get("f1") for c in CLASS_ORDER} for t, r in repsB.items()})
        pcB.insert(0, "Support", [repsB[next(iter(repsB))]["per_class"][c]["support"] for c in CLASS_ORDER])
        md += ["### Per-class F1 (Track B test)", "", md_table(pcB.reset_index().rename(columns={"index": "Class"}).round(3)), ""]
        out["track_b"]["per_class_f1"] = pcB.to_dict()
        plots.per_class_f1_plot({t: repsB[t] for t in ["LSTM", "Per-flow MLP", "Per-flow RF", "Context baseline"] if t in repsB},
                                "Per-class F1, Track B chronological test block", figs / "per_class_f1_track_b.png")
        pairsB = {}
        if "LSTM" in predsB:
            for other in [t for t in predsB if t != "LSTM" and "Oracle" not in t]:
                pairsB[f"LSTM vs {other}"] = M.paired_bootstrap_diff(yB, predsB["LSTM"], predsB[other],
                                                                     n_boot=500, blocks=binsB)
        out["track_b"]["paired_bootstrap_time_blocks"] = pairsB
        if pairsB:
            md += ["### LSTM vs non-sequential models (paired time-block bootstrap)", "",
                   md_table(pd.DataFrame([{"Comparison": k, "ΔMacro-F1": f"{v['diff_macro_f1']:+.3f}",
                                           "95% CI": f"[{v['low']:+.3f}, {v['high']:+.3f}]",
                                           "Excludes 0": "yes" if v["excludes_zero"] else "no"}
                                          for k, v in pairsB.items()])), ""]

    # ── Leave one class out ───────────────────────────────────────────────────
    if exists("A", "leave_one_class_out"):
        L = store.load_metrics("A", "leave_one_class_out")
        ae = L["anomaly_detector_detection_rate_by_category"]["autoencoder"]
        td = L.get("training_data", {})
        md += ["## Leave one attack class out (Track A)", "",
               f"Training data: {training_data_label(L)} (seed 0, classes under "
               f"{td.get('floor_per_class', '–')} records kept whole); preprocessor refitted per fold "
               "without the held-out class; full validation and official test sets.", "", md_table(pd.DataFrame([{
            "Held-out class": r["held_out"], "Model": r["model"].upper(), "Test records": f"{r['held_out_test_support']:,}",
            "→ Normal (missed)": f"{r['held_out_predicted_as_normal']:.1%}",
            "→ other attack": f"{r['held_out_predicted_as_other_attack']:.1%}",
            "Most common": r["most_common_wrong_attack"],
            "Rejected unknown": f"{r['held_out_rejected_as_unknown']:.1%}",
            "Known rejected": f"{r['known_classes_rejected_as_unknown']:.1%}",
            "Autoencoder flags": f"{ae.get(r['held_out'], float('nan')):.1%}"} for r in L["results"]])), ""]

    (PROJECT_ROOT / "artifacts" / "metrics" / "comparison.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    (PROJECT_ROOT / "reports" / "results_tables.md").write_text(
        "# Results tables (generated by scripts/compare_models.py)\n\n" + "\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
