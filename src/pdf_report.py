"""
pdf_report.py
-------------
Generates a one-page PDF prediction report: patient details, the
prediction + confidence, a SHAP "reasons" chart (optional), and doctor
recommendations. Used by the Streamlit app's "Download PDF Report" button.
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)


def _reasons_chart_png(reasons):
    """Small horizontal bar chart of the top SHAP reasons, as PNG bytes."""
    if not reasons:
        return None

    features = [r["feature"] for r in reasons][::-1]
    values = [r["pct"] * (1 if r["direction"] == "+" else -1) for r in reasons][::-1]
    bar_colors = ["#C44E52" if v > 0 else "#4C72B0" for v in values]

    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.barh(features, values, color=bar_colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Influence on prediction (%)")
    ax.set_title("Top Contributing Factors", fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf_report(dataset_label, disease_label, patient_values: dict,
                      prediction: int, confidence, model_label, doctor,
                      recommendations, reasons=None):
    """
    Builds the PDF in memory and returns it as bytes, ready for a
    Streamlit download_button.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=18, spaceAfter=4)
    section_style = ParagraphStyle("SectionCustom", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    normal = styles["Normal"]

    story = []

    story.append(Paragraph("Disease Prediction Report", title_style))
    story.append(Paragraph(f"Dataset: {dataset_label} &nbsp;|&nbsp; Model: {model_label}", normal))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#CCCCCC"), spaceBefore=8, spaceAfter=8))

    # --- Patient details ---
    story.append(Paragraph("Patient Input", section_style))
    table_data = [["Feature", "Value"]] + [[k, f"{v:.2f}" if isinstance(v, float) else str(v)]
                                            for k, v in patient_values.items()]
    t = Table(table_data, colWidths=[3 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # --- Prediction result ---
    story.append(Paragraph("Prediction Result", section_style))
    result_label = disease_label if prediction == 1 else "No Disease"
    conf_text = f"{confidence:.1%}" if confidence is not None else "N/A"
    result_color = colors.HexColor("#C44E52") if prediction == 1 else colors.HexColor("#55A868")

    result_style = ParagraphStyle("Result", parent=styles["Heading2"], textColor=result_color)
    story.append(Paragraph(f"Prediction: {result_label}", result_style))
    story.append(Paragraph(f"Confidence: {conf_text}", normal))

    if prediction == 1:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Suggested Doctor:</b> {doctor}", normal))
        story.append(Paragraph("<b>Recommendations:</b>", normal))
        for rec in recommendations:
            story.append(Paragraph(f"&bull; {rec}", normal))
    else:
        story.append(Spacer(1, 6))
        story.append(Paragraph("No immediate concern based on this input, but routine "
                                "checkups are still worthwhile.", normal))

    # --- SHAP reasons chart ---
    if reasons:
        story.append(Paragraph("Why This Prediction", section_style))
        chart_buf = _reasons_chart_png(reasons)
        if chart_buf:
            story.append(Image(chart_buf, width=5.8 * inch, height=2.1 * inch))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#CCCCCC")))
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8,
                                       textColor=colors.grey, spaceBefore=6)
    story.append(Paragraph(
        "This is an educational demo, not a medical diagnostic tool. "
        "Always consult a qualified healthcare professional.", disclaimer_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
