"""
report_generator.py
=====================
Generates a downloadable PDF research report for a completed analysis,
summarizing detected candidates, their classifications, evidence, and
key signal parameters — addressing the "exportable reports" dashboard
requirement.
"""

from __future__ import annotations
import io
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=20, leading=24, spaceAfter=6,
        textColor=colors.HexColor("#1a1206"), fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", fontSize=11, leading=14, textColor=colors.HexColor("#555555"),
        spaceAfter=16,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader", fontSize=13, leading=16, spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#2c2410"), fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="CandidateHeader", fontSize=12, leading=15, spaceBefore=10, spaceAfter=4,
        textColor=colors.HexColor("#7a4f00"), fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="Body", fontSize=9.5, leading=13.5, textColor=colors.HexColor("#222222"),
        alignment=TA_LEFT, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Small", fontSize=8, leading=11, textColor=colors.HexColor("#777777"),
    ))
    return styles


def generate_analysis_report_pdf(analysis: dict, dataset_name: str) -> bytes:
    """
    analysis: dict matching the shape returned by db.get_analysis() / AnalyzeResponse,
    i.e. {analysis_id, dataset_id, model_version, processing_time_seconds,
          observation_summary, candidates: [...]}
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = _build_styles()
    story = []

    # ---- Header ----
    story.append(Paragraph("Exoplanet Transit Detection Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Target: {dataset_name} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Model: {analysis.get('model_version', 'unknown')}",
        styles["ReportSubtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dddddd")))
    story.append(Spacer(1, 10))

    # ---- Observation summary ----
    story.append(Paragraph("Observation Summary", styles["SectionHeader"]))
    summary_text = (analysis.get("observation_summary") or "No summary available.").replace("\n", "<br/>")
    story.append(Paragraph(summary_text, styles["Body"]))

    candidates = analysis.get("candidates", [])

    # ---- Candidate ranking table ----
    story.append(Paragraph("Candidate Ranking", styles["SectionHeader"]))
    table_data = [["#", "Classification", "Confidence", "Period (d)", "Depth (ppm)", "SNR", "FP Risk"]]
    for c in candidates:
        f = c["features"]
        table_data.append([
            str(c["rank"]), c["final_label"].replace("_", " "),
            f"{c['final_confidence']*100:.1f}%", f"{f['period_days']:.4f}",
            f"{f['depth_ppm']:.0f}", f"{f['snr']:.1f}",
            "Yes" if c["is_likely_false_positive"] else "No",
        ])
    tbl = Table(table_data, colWidths=[0.3*inch, 1.4*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.6*inch, 0.6*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1206")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f5f0")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # ---- Per-candidate detail ----
    for c in candidates:
        f = c["features"]
        story.append(PageBreak())
        story.append(Paragraph(f"Candidate #{c['rank']}: {c['final_label'].replace('_', ' ')}", styles["CandidateHeader"]))
        story.append(Paragraph(
            f"Calibrated confidence: <b>{c['final_confidence']*100:.1f}%</b> &nbsp;|&nbsp; "
            f"False-positive risk flags: {', '.join(c['false_positive_flags']) if c['false_positive_flags'] else 'None'}",
            styles["Body"],
        ))

        param_data = [
            ["Period", f"{f['period_days']:.5f} days", "Transit duration", f"{f['duration_hours']:.2f} hours"],
            ["Transit depth", f"{f['depth_ppm']:.1f} ± {f['depth_err_ppm']:.1f} ppm", "Detection SNR", f"{f['snr']:.2f}"],
            ["Est. planet radius", f"{f['planet_radius_re_estimate']:.2f} R⊕", "Transits observed", f"{f['n_transits_observed']}"],
            ["Odd/even depth diff", f"{f['odd_even_depth_diff_sigma']:.2f}σ", "Secondary eclipse sig.", f"{f['secondary_eclipse_sig_sigma']:.2f}σ"],
            ["Transit shape score", f"{f['transit_shape_score']:.2f} (0=V-shape, 1=flat)", "Periodicity strength", f"{f['periodicity_strength']:.2f}"],
        ]
        ptbl = Table(param_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 1.5*inch])
        ptbl.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0ede5")),
            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f0ede5")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ptbl)
        story.append(Spacer(1, 8))

        story.append(Paragraph("Scientific Explanation", styles["SectionHeader"]))
        story.append(Paragraph(c.get("explanation", ""), styles["Body"]))

        story.append(Paragraph("Evidence & Vetting", styles["SectionHeader"]))
        for e in c.get("evidence", []):
            mark = "✓ SUPPORTS" if e["direction"] == "supports" else "✗ AGAINST"
            color = "#2e7d32" if e["direction"] == "supports" else "#c62828"
            story.append(Paragraph(f'<font color="{color}"><b>{mark}</b></font> — {e["reason"]}', styles["Body"]))

        if c.get("data_quality_warning"):
            story.append(Paragraph(f"<b>Data quality note:</b> {c['data_quality_warning']}", styles["Small"]))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
    story.append(Paragraph(
        "Generated by ExoNova — AI-Powered Exoplanet Discovery and Analysis Platform. "
        "This report is produced by an automated pipeline and should be reviewed by a qualified "
        "astronomer before scientific publication or follow-up resource allocation.",
        styles["Small"],
    ))

    doc.build(story)
    return buf.getvalue()
