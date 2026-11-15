#!/usr/bin/env python3
"""Phase 1: exploratory figures for docs/dataset_profile.md.

Counts come from reports/dataset/profile.json (run scripts/profile_dataset.py
first); distributions and correlations are computed from the Track A
TRAINING data only. Output: reports/figures/dataset/*.png

Usage:  .venv\\Scripts\\python.exe scripts\\make_dataset_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield import viz  # noqa: E402
from deepshield.config import PROJECT_ROOT  # noqa: E402
from deepshield.data import schema  # noqa: E402
from deepshield.data.loaders import load_track_a  # noqa: E402

plt = viz.plt
OUT = PROJECT_ROOT / "reports" / "figures" / "dataset"
PROFILE = json.loads((PROJECT_ROOT / "reports/dataset/profile.json").read_text(encoding="utf-8"))
SETS = [("Track A train", PROFILE["track_a"]["train"]),
        ("Track A test", PROFILE["track_a"]["test"]),
        ("Track B (4 main files)", PROFILE["track_b"])]


def class_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (name, d) in zip(axes, SETS):
        n, a = d["label_counts"]["0"], d["label_counts"]["1"]
        bars = ax.bar(["Normal", "Attack"], [n, a], width=0.55,
                      color=[viz.NORMAL_COLOUR, viz.ATTACK_COLOUR])
        for b, v in zip(bars, (n, a)):
            ax.annotate(f"{v:,}\n({v / (n + a):.1%})", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8, color=viz.INK_2,
                        xytext=(0, 2), textcoords="offset points")
        ax.set_title(name)
        ax.set_ylim(0, max(n, a) * 1.3)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f}k")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Normal vs attack records", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return viz.save(fig, OUT / "class_distribution.png")


def attack_categories():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    order = schema.CLASS_ORDER[::-1]
    for ax, (name, d) in zip(axes, SETS):
        counts = [d["attack_cat_counts"][c] for c in order]
        ax.barh(order, counts, height=0.62, color=viz.SINGLE)
        ax.set_xscale("log")
        for y, v in enumerate(counts):
            ax.annotate(f"{v:,}", (v, y), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=viz.INK_2)
        ax.set_xlim(10, max(counts) * 12)
        ax.set_title(name)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("records (log scale)")
    fig.suptitle("Records per category: imbalance spans more than two orders of magnitude",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return viz.save(fig, OUT / "attack_categories.png")


def missing_values():
    miss = PROFILE["track_b"]["missing_by_column"]
    total = PROFILE["track_b"]["rows"]
    cols = sorted(miss, key=miss.get)
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.barh(cols, [miss[c] / total * 100 for c in cols], height=0.5, color=viz.SINGLE)
    for y, c in enumerate(cols):
        ax.annotate(f"{miss[c]:,} ({miss[c] / total:.1%})", (miss[c] / total * 100, y),
                    xytext=(3, 0), textcoords="offset points", va="center", fontsize=8,
                    color=viz.INK_2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of Track B records with a blank value")
    ax.grid(axis="y", visible=False)
    ax.set_title("Missing values: only three Track B columns; Track A has none")
    return viz.save(fig, OUT / "missing_values.png")


def numeric_distributions(tr: pd.DataFrame):
    feats = ["dur", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload",
             "smean", "sinpkt", "tcprtt", "ct_srv_src", "ct_state_ttl"]
    fig, axes = plt.subplots(3, 4, figsize=(11, 6.8))
    for ax, f in zip(axes.ravel(), feats):
        v = np.log1p(tr[f].to_numpy(dtype=float))
        bins = np.linspace(v.min(), v.max(), 40)
        for lab, colour, name in ((0, viz.NORMAL_COLOUR, "Normal"), (1, viz.ATTACK_COLOUR, "Attack")):
            ax.hist(v[tr["label"].to_numpy() == lab], bins=bins, density=True,
                    histtype="step", linewidth=1.6, color=colour, label=name)
        ax.set_title(f"log1p({f})", fontsize=9)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Selected numeric features, Track A training set (density, log1p scale)",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return viz.save(fig, OUT / "numeric_distributions.png")


def correlation(tr: pd.DataFrame):
    sample = tr[schema.TRACK_A_NUMERIC].sample(50_000, random_state=0)
    corr = sample.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(corr.to_numpy(), cmap=viz.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman rho")
    cb.outline.set_visible(False)
    ax.set_title("Spearman correlation of numeric features (Track A train, 50,000-record sample)")
    return viz.save(fig, OUT / "correlation_heatmap.png")


def timeline():
    h = PROFILE["track_b"]["hourly_counts"]
    t = pd.to_datetime(h["hour_epoch"], unit="s", utc=True)
    df = pd.DataFrame({"normal": h["normal"], "attack": h["attack"]}, index=t)
    sessions = [g for _, g in df.groupby(df.index.strftime("%Y-%m") )]
    fig, axes = plt.subplots(2, len(sessions), figsize=(11, 4.4), sharex="col",
                             gridspec_kw={"width_ratios": [len(s) for s in sessions]})
    for j, s in enumerate(sessions):
        for i, (col, colour) in enumerate((("normal", viz.NORMAL_COLOUR), ("attack", viz.ATTACK_COLOUR))):
            ax = axes[i, j]
            ax.bar(s.index, s[col], width=1 / 24 * 0.85, color=colour, align="edge")
            ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f}k")
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(f"{col} / hour")
        axes[0, j].set_title(f"Capture session: {s.index[0]:%d %b %Y}")
        axes[1, j].tick_params(axis="x", rotation=30)
        axes[1, j].xaxis.set_major_formatter(viz.matplotlib.dates.DateFormatter("%H:%M"))
    fig.suptitle("Track B records per hour (UTC). Two capture sessions; attacks concentrate on 18 Feb",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return viz.save(fig, OUT / "track_b_timeline.png")


def duplicates():
    rows = [("Track A train", PROFILE["track_a"]["train"]),
            ("Track A test", PROFILE["track_a"]["test"]),
            ("Track B", PROFILE["track_b"])]
    names = [r[0] for r in rows]
    exact = [r[1].get("exact_duplicates_excl_id", r[1].get("exact_duplicates")) / r[1]["rows"] * 100 for r in rows]
    conf = [r[1]["rows_in_conflicting_groups"] / r[1]["rows"] * 100 for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.4), sharey=True)
    for ax, vals, title in ((axes[0], exact, "Exact duplicate records (% of rows)"),
                            (axes[1], conf, "Rows whose identical features carry different labels (%)")):
        ax.barh(names[::-1], vals[::-1], height=0.5, color=viz.SINGLE)
        for y, v in enumerate(vals[::-1]):
            ax.annotate(f"{v:.1f}%", (v, y), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=8, color=viz.INK_2)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return viz.save(fig, OUT / "duplicates.png")


def main():
    viz.apply_style()
    import matplotlib.dates  # noqa: F401  (used via viz.matplotlib.dates)
    tr = load_track_a("train")
    for p in (class_distribution(), attack_categories(), missing_values(),
              numeric_distributions(tr), correlation(tr), timeline(), duplicates()):
        print("wrote", p.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
