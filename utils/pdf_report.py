"""PDF reporting utilities for Valvulin optimization runs."""
from __future__ import annotations

import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _format_numeric(value: Any) -> str:
    """Return a nicely formatted string for numeric values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 100:
        return f"{numeric:,.0f}"  # Large numbers without decimals
    return f"{numeric:,.3f}"  # Default precision for metrics


def generate_pdf_report(
    best_params: Dict[str, float],
    backtest_result: Dict[str, Any],
    optimizer_df: pd.DataFrame,
    timestamp: str,
) -> str:
    """Generate a professional PDF report summarising an optimisation run."""

    folder = os.path.join("results", "runs", timestamp)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "Valvulin_Report.pdf")

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    subtitle_style = ParagraphStyle(
        name="Subtitle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1f4f7f"),
    )
    elements = []

    elements.append(
        Paragraph("<b>Valvulin Trading Bot — Optimization Report</b>", styles["Title"])
    )
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>📊 Parámetros Óptimos</b>", subtitle_style))
    for key, value in best_params.items():
        elements.append(Paragraph(f"{key}: {_format_numeric(value)}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    metrics = backtest_result.get("metrics", {})
    elements.append(Paragraph("<b>📈 Métricas del Mejor Backtest</b>", subtitle_style))
    for key, value in metrics.items():
        elements.append(Paragraph(f"{key}: {_format_numeric(value)}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>🏆 Top 10 Resultados de Optimización</b>", subtitle_style))
    if optimizer_df.empty:
        elements.append(Paragraph("No se registraron resultados de optimización.", styles["Normal"]))
    else:
        top10 = optimizer_df.sort_values("expectancy_R", ascending=False).head(10).copy()
        top10.fillna(0, inplace=True)
        table_data = [list(top10.columns)]
        for _, row in top10.iterrows():
            table_data.append([_format_numeric(value) for value in row])
        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0d3055")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        elements.append(table)
    elements.append(Spacer(1, 20))

    if not optimizer_df.empty:
        plot_path = os.path.join(folder, "expectancy_drawdown.png")
        plot_df = optimizer_df.sort_values("expectancy_R", ascending=False).head(100)
        plt.figure(figsize=(6, 4))
        plt.scatter(plot_df["expectancy_R"], plot_df["max_drawdown"], c="#1f77b4")
        plt.xlabel("Expectancy (R)")
        plt.ylabel("Drawdown (%)")
        plt.title("Expectancy vs Drawdown")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

        elements.append(Image(plot_path, width=400, height=250))
        elements.append(Spacer(1, 20))

    elements.append(Paragraph("📅 Reporte generado automáticamente", styles["Italic"]))

    doc.build(elements)
    return file_path


__all__ = ["generate_pdf_report"]
