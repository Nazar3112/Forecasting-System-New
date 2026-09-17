"""
core/visualizer.py
==================
Kelas DashboardVisualizer: pembuatan grafik interaktif Plotly,
kartu metrik KPI Streamlit, tabel analitik, dan ekspor CSV UTF-8.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_C_ACTUAL   = "#eaecef"
_C_PROPHET  = "#929aa5"
_C_HYBRID   = "#0ecb81"
_C_FUTURE   = "#fcd535"
_C_RESIDUAL = "#3b82f6"

logger = logging.getLogger(__name__)


class DashboardVisualizer:
    """Lapisan visual dasbor Streamlit — Binance Design System."""

    @staticmethod
    def render_kpi_cards(
        total_sales: int,
        mape_val:    float,
        rmse_val:    float,
        mae_val:     float,
        trend_pct:   float,
        mape_label:  str,
        mape_emoji:  str,
    ) -> None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="📦 Total Penjualan Historis",
                value=f"{total_sales:,} unit",
                delta=f"{trend_pct:+.1f}% tren",
                help="Jumlah kumulatif kuantitas terjual seluruh periode historis.",
            )
        with col2:
            st.metric(
                label=f"{mape_emoji} MAPE Model Hibrida",
                value=f"{mape_val:.2f}%",
                delta=mape_label,
                delta_color="off",
                help="Mean Absolute Percentage Error pada data uji. < 10% = Sangat Baik.",
            )
        with col3:
            st.metric(
                label="📐 RMSE",
                value=f"{rmse_val:,.2f}",
                help="Root Mean Squared Error dalam satuan unit barang.",
            )
        with col4:
            st.metric(
                label="📏 MAE",
                value=f"{mae_val:,.2f}",
                help="Mean Absolute Error rata-rata galat per periode.",
            )

    @staticmethod
    def plot_forecast(
        historical_df:   pd.DataFrame,
        test_results:    pd.DataFrame,
        future_forecast: pd.DataFrame,
        title:           str  = "Peramalan Penjualan",
        category_label:  str  = "",
        freq_label:      str  = "Mingguan",
        show_prophet:    bool = True,
    ) -> go.Figure:
        fig = go.Figure()

        # Trace 1: Aktual
        fig.add_trace(go.Scatter(
            x=historical_df["ds"],
            y=historical_df["y"],
            name="Penjualan Aktual",
            mode="lines+markers",
            line=dict(color=_C_ACTUAL, width=2),
            marker=dict(size=4, color=_C_ACTUAL),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Aktual: <b>%{y:,.0f}</b> unit<extra></extra>",
        ))

        # Trace 2: Prophet tunggal
        if show_prophet and not test_results.empty and "y_prophet" in test_results.columns:
            fig.add_trace(go.Scatter(
                x=test_results["ds"],
                y=test_results["y_prophet"],
                name="Prediksi Prophet (Data Uji)",
                mode="lines+markers",
                line=dict(color=_C_PROPHET, width=1.8, dash="dot"),
                marker=dict(size=5, symbol="diamond", color=_C_PROPHET),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Prophet: <b>%{y:,.0f}</b> unit<extra></extra>",
            ))

        # Trace 3: Hibrida test
        if not test_results.empty and "y_hybrid" in test_results.columns:
            fig.add_trace(go.Scatter(
                x=test_results["ds"],
                y=test_results["y_hybrid"],
                name="Prediksi Hibrida (Data Uji)",
                mode="lines+markers",
                line=dict(color=_C_HYBRID, width=2.5),
                marker=dict(size=6, symbol="circle", color=_C_HYBRID),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Hibrida: <b>%{y:,.0f}</b> unit<extra></extra>",
            ))

        # Trace 4: Proyeksi masa depan
        if not future_forecast.empty:
            fig.add_trace(go.Scatter(
                x=future_forecast["ds"],
                y=future_forecast["y_hybrid"],
                name="Proyeksi Masa Depan",
                mode="lines+markers",
                line=dict(color=_C_FUTURE, width=2.5, dash="dash"),
                marker=dict(size=7, symbol="star", color=_C_FUTURE),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Proyeksi: <b>%{y:,.0f}</b> unit<extra></extra>",
            ))

        # Zona Uji
        if not test_results.empty:
            fig.add_vrect(
                x0=test_results["ds"].min(),
                x1=test_results["ds"].max(),
                fillcolor="rgba(252,213,53,0.06)",
                line_width=0,
                annotation_text="Zona Uji",
                annotation_position="top left",
                annotation=dict(font=dict(size=11, color="#707a8a")),
            )

        # Garis pemisah masa depan
        if not future_forecast.empty and not historical_df.empty:
            fig.add_vline(
                x=historical_df["ds"].max(),
                line_dash="dot",
                line_color="#2b3139",
                annotation_text="Mulai Proyeksi",
                annotation_position="top right",
                annotation=dict(font=dict(size=11, color="#707a8a")),
            )

        subtitle = f"Kategori: {category_label}" if category_label else ""
        fig.update_layout(**_dark_layout(
            title=f"<b>{title}</b>" + (
                f"<br><span style='font-size:12px;color:#707a8a'>{subtitle}</span>"
                if subtitle else ""
            ),
            xaxis_title=f"Periode ({freq_label})",
            yaxis_title="Kuantitas Penjualan (unit)",
            height=480,
        ))
        return fig

    @staticmethod
    def plot_residual_analysis(test_results: pd.DataFrame, category_label: str = "") -> go.Figure:
        fig = go.Figure()
        if test_results.empty or "y" not in test_results.columns:
            return fig

        y_true      = test_results["y"]
        err_prophet = (y_true - test_results.get("y_prophet", y_true)).abs()
        err_hybrid  = (y_true - test_results.get("y_hybrid",  y_true)).abs()

        fig.add_trace(go.Bar(x=test_results["ds"], y=err_prophet,
                             name="|Galat Prophet|", marker_color=_C_PROPHET, opacity=0.85))
        fig.add_trace(go.Bar(x=test_results["ds"], y=err_hybrid,
                             name="|Galat Hibrida|", marker_color=_C_HYBRID, opacity=0.9))

        subtitle = f"Kategori: {category_label}" if category_label else ""
        fig.update_layout(**_dark_layout(
            title=f"<b>Analisis Galat Absolut pada Data Uji</b>" + (
                f"<br><span style='font-size:12px;color:#707a8a'>{subtitle}</span>"
                if subtitle else ""
            ),
            xaxis_title="Periode",
            yaxis_title="Galat Absolut (unit)",
            height=360,
            barmode="group",
        ))
        return fig

    @staticmethod
    def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        top_df = fi_df.head(top_n).sort_values("importance")
        fig = go.Figure(go.Bar(
            x=top_df["importance"], y=top_df["feature"],
            orientation="h", marker_color=_C_RESIDUAL,
        ))
        fig.update_layout(**_dark_layout(
            title="<b>Pentingnya Fitur LightGBM (Feature Importance)</b>",
            xaxis_title="Importance (Gain)",
            yaxis_title="Fitur",
            height=max(300, top_n * 25),
        ))
        return fig

    @staticmethod
    def render_forecast_table(future_forecast: pd.DataFrame, freq_label: str = "Mingguan") -> None:
        if future_forecast.empty:
            st.info("Belum ada data proyeksi.")
            return
        display_df = future_forecast[["ds", "y_hybrid"]].copy()
        display_df.columns = ["Periode", "Proyeksi Penjualan (unit)"]
        display_df["Periode"] = display_df["Periode"].dt.strftime("%d %b %Y")
        display_df["Proyeksi Penjualan (unit)"] = (
            display_df["Proyeksi Penjualan (unit)"].round(0).astype(int)
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    @staticmethod
    def render_comparison_table(comp_df: pd.DataFrame) -> None:
        if comp_df.empty:
            return
        styled_df = comp_df.copy()
        for col in ["MAE", "MAPE_%", "RMSE"]:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(
                    lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x
                )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    @staticmethod
    def export_csv_bytes(future_forecast: pd.DataFrame, category_label: str = "") -> bytes:
        export_df = future_forecast.copy()
        if "ds" in export_df.columns:
            export_df["Periode"] = export_df["ds"].dt.strftime("%d-%m-%Y")
            export_df.drop(columns=["ds"], inplace=True)

        rename_map = {
            "y_hybrid":        "Proyeksi_Penjualan_Unit",
            "y_prophet":       "Estimasi_Prophet_Unit",
            "y_residual_lgbm": "Koreksi_LightGBM_Unit",
        }
        export_df.rename(columns=rename_map, inplace=True)

        buffer = io.StringIO()
        buffer.write("# Laporan Proyeksi Penjualan Retail\n")
        buffer.write(f"# Kategori/Divisi : {category_label}\n")
        buffer.write("# Model           : Hibrida Prophet + LightGBM (MySQL Persistent)\n")
        buffer.write("# Indomarco Prismatama — Sistem Prediksi Tren Penjualan\n")
        buffer.write("#\n")
        export_df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")

    @staticmethod
    def calculate_trend_pct(series_df: pd.DataFrame, window: int = 4) -> float:
        y = series_df["y"].values
        if len(y) < window * 2:
            return 0.0
        avg_recent   = np.mean(y[-window:])
        avg_previous = np.mean(y[-window * 2:-window])
        if avg_previous == 0:
            return 0.0
        return round((avg_recent - avg_previous) / avg_previous * 100, 2)


def _dark_layout(title: str = "", xaxis_title: str = "", yaxis_title: str = "", height: int = 400, **kwargs) -> dict:
    base = dict(
        title=dict(text=title, font=dict(size=15, color="#ffffff", family="Inter, sans-serif")),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=12, color="#eaecef")),
            gridcolor="#2b3139", linecolor="#2b3139",
            tickfont=dict(size=11, color="#707a8a"),
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=12, color="#eaecef")),
            gridcolor="#2b3139", linecolor="#2b3139",
            tickfont=dict(size=11, color="#707a8a"),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=12, color="#eaecef"),
            bgcolor="rgba(30,35,41,0.9)", bordercolor="#2b3139", borderwidth=1,
        ),
        hovermode="x unified",
        height=height,
        paper_bgcolor="#0b0e11",
        plot_bgcolor="#1e2329",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#eaecef"),
        margin=dict(l=40, r=20, t=80, b=40),
    )
    base.update(kwargs)
    return base
