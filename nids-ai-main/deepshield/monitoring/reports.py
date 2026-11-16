"""PDF and CSV security reports built from the stored prediction history.

Reuses the original NIDS-AI report look (ReportLab, same palette and
severity colours). Every number comes from the history database or the
saved experiment metrics; nothing is hard-coded.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

from deepshield.monitoring.history import History

# palette of the original NIDS-AI report
CLR_DARK = colors.HexColor("#040d1a")
CLR_MUTED = colors.HexColor("#64748b")
CLR_HEADER = colors.HexColor("#1a2940")
CLR_LIGHT = colors.HexColor("#f1f5f9")
SEV_CLR = {"CRITICAL": colors.HexColor("#ef4444"), "HIGH": colors.HexColor("#f59e0b"),
           "MEDIUM": colors.HexColor("#3b82f6"), "LOW": colors.HexColor("#10b981")}


def _fmt_time(ts) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


_CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=10)
_HEAD = ParagraphStyle("head", parent=_CELL, textColor=colors.white)
_SEV = {k: ParagraphStyle(f"sev_{k}", parent=_CELL, textColor=c) for k, c in SEV_CLR.items()}


def _cell(value, style) -> Paragraph:
    """Every cell is an escaped Paragraph, so long text wraps inside its column."""
    return Paragraph(escape(str(value)).replace("\n", "<br/>"), style)


def _table(rows, widths, header=True, sev_col=None):
    cells = []
    for i, r in enumerate(rows):
        row = []
        for j, v in enumerate(r):
            style = _HEAD if header and i == 0 else (
                _SEV.get(str(v), _CELL) if sev_col is not None and j == sev_col else _CELL)
            row.append(_cell(v, style))
        cells.append(row)
    t = Table(cells, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLR_LIGHT])]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), CLR_HEADER)]
    t.setStyle(TableStyle(style))
    return t


def build_pdf(history: History, metrics: dict, explanations: dict | None = None,
              local_explanation: dict | None = None) -> bytes:
    base = getSampleStyleSheet()
    st = {"title": ParagraphStyle("t", parent=base["Title"], fontSize=20, textColor=CLR_DARK),
          "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9, textColor=CLR_MUTED),
          "h": ParagraphStyle("h", parent=base["Heading2"], fontSize=12, textColor=CLR_DARK,
                              spaceBefore=10, spaceAfter=4),
          "b": ParagraphStyle("b", parent=base["Normal"], fontSize=8.5, leading=12)}
    ov = history.overview()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="DeepShield AI security report")
    W = A4[0] - 32 * mm
    s = [Paragraph("DeepShield AI — Security Report", st["title"]),
         Paragraph(f"Generated {_fmt_time(datetime.now(timezone.utc).timestamp())}. "
                   "Source data: UNSW-NB15 dataset replay and/or uploaded records. "
                   "This is an academic demonstration, not monitoring of a live network.", st["sub"]),
         Spacer(1, 6)]

    s.append(Paragraph("1. Summary of detected activity", st["h"]))
    s.append(_table([["Records analysed", "Normal", "Malicious", "Attack share",
                      "Time period (flow time)", "Analysis period"],
                     [f"{ov['total']:,}", f"{ov['normal']:,}", f"{ov['malicious']:,}",
                      f"{ov['attack_share']:.1%}",
                      f"{_fmt_time(ov['flow_time_start'])}\nto {_fmt_time(ov['flow_time_end'])}",
                      f"{_fmt_time(ov['first_prediction'])}\nto {_fmt_time(ov['last_prediction'])}"]],
                    [W * x for x in (.13, .1, .11, .11, .28, .27)]))

    s.append(Paragraph("2. Attack categories and severity", st["h"]))
    dist = ov["attack_distribution"]
    rows = [["Category", "Predictions", "Share of malicious"]] + [
        [k, f"{v:,}", f"{v / max(ov['malicious'], 1):.1%}"] for k, v in dist.items()]
    s.append(_table(rows, [W * .4, W * .3, W * .3]))
    s.append(Spacer(1, 4))
    s.append(_table([["Severity"] + list(ov["severity"]), ["Alerts"] + [f"{v:,}" for v in ov["severity"].values()]],
                    [W / (len(ov["severity"]) + 1)] * (len(ov["severity"]) + 1), header=False))

    s.append(Paragraph("3. Prediction statistics by model", st["h"]))
    rows = [["Model", "Source", "Predictions", "Flagged malicious"]] + [
        [r["model"], r["source"], f"{r['n']:,}", f"{(r['a'] or 0):,}"] for r in ov["by_model"]]
    s.append(_table(rows, [W * .35, W * .2, W * .2, W * .25]))

    s.append(Paragraph("4. Model information (offline evaluation on held-out test data)", st["h"]))
    rows = [["Model", "Track", "Task", "Test macro-F1 / ROC-AUC", "Accuracy"]]
    for track, ms in metrics.items():
        for name, m in ms.items():
            t = m.get("test") or {}
            if "macro_f1" in t:
                rows.append([m.get("model", name), track.upper(), "multiclass",
                             f"{t['macro_f1']:.3f}", f"{t['accuracy']:.3f}"])
            elif "roc_auc" in t:
                rows.append([m.get("model", name), track.upper(), m.get("task", "binary"),
                             f"ROC-AUC {t['roc_auc']:.3f}", f"{t['accuracy']:.3f}"])
    s.append(_table(rows, [W * .38, W * .1, W * .22, W * .18, W * .12]))

    alerts = history.alerts(limit=25)["rows"]
    s.append(Paragraph(f"5. Most recent alerts (25 of {history.alerts(limit=1)['total']:,})", st["h"]))
    rows = [["Time (flow)", "Severity", "Category", "Source → destination", "Model", "Conf.", "ATT&CK (candidate)"]]
    for a in alerts:
        tech = ", ".join(t["id"] for t in (a.get("mitre") or {}).get("techniques", [])) or "-"
        rows.append([_fmt_time(a.get("flow_time") or a["created"])[5:19], a["severity"], a["category"],
                     f"{a['srcip'] or '-'} → {a['dstip'] or '-'}", a["model"],
                     f"{a['confidence']:.2f}" if a["confidence"] is not None else "-", tech])
    s.append(_table(rows, [W * x for x in (.15, .1, .16, .2, .13, .07, .19)], sev_col=1))
    s.append(Paragraph("ATT&amp;CK techniques are candidate associations for the predicted category, "
                       "validated against the official MITRE STIX data; they are not confirmed attacker "
                       "behaviour.", st["sub"]))

    if explanations or local_explanation:
        s.append(Paragraph("6. Explanations", st["h"]))
    if explanations:
        for method in ("RF - SHAP", "MLP - Integrated Gradients"):
            top = explanations.get("top10", {}).get(method)
            if top:
                s.append(Paragraph(f"<b>{method}</b>, globally most influential features: "
                                   + ", ".join(top), st["b"]))
    if local_explanation:
        le = local_explanation
        s.append(Paragraph(f"<b>Highest-severity alert</b> ({le['prediction']['category']}, model "
                           f"{le['model']}, {le['method']}):", st["b"]))
        rows = [["Feature", "Value", "Contribution"]] + [
            [c["feature"], str(c.get("value", "")), f"{c['contribution']:+.4f}"] for c in le["contributions"]]
        s.append(_table(rows, [W * .4, W * .3, W * .3]))
    if explanations or local_explanation:
        s.append(Paragraph("Attributions describe what the model relied on, not the cause of the traffic.",
                           st["sub"]))
    doc.build(s)
    return buf.getvalue()


def build_csv(history: History) -> str:
    rows = history.export_rows()
    out = io.StringIO()
    if rows:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["created"] = _fmt_time(r["created"])
            r["flow_time"] = _fmt_time(r["flow_time"]) if r["flow_time"] else ""
            w.writerow(r)
    else:
        out.write("no predictions recorded\n")
    return out.getvalue()
