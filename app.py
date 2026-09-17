"""
app.py
======
Streamlit Dashboard — Antarmuka Pengguna Operasional
Sistem Prediksi Tren Penjualan Retail Berbasis Kategori
Model Hibrida Prophet + LightGBM (PT. Indomarco Prismatama)

Arsitektur: MySQL Native (XAMPP / Standalone MySQL) — Tanpa Docker
  - Driver: Pure Python PyMySQL (tanpa kebutuhan C++ build tools)
  - Auto-create database `db_forecasting` & auto-create tables
  - Query deret waktu instan (<0.1s) dari tabel `sales_aggregations`
  - Upload delta bulanan inkremental (~15-25 MB)

Desain: Binance Design System
  - Near-black canvas (#0b0e11), Binance Yellow (#fcd535)
  - Inter typography, JetBrains Mono untuk angka
"""

import logging
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (
    APP_TITLE, APP_SUBTITLE, APP_ICON,
    FREQ_OPTIONS, SEASONALITY_MODES,
    DEFAULT_HORIZON, MIN_HORIZON, MAX_HORIZON,
    MIN_DATA_POINTS,
    DeltaPreprocessor,
    HybridForecastingEngine,
    ModelEvaluator,
    DashboardVisualizer,
)

# Impor lapisan database MySQL
try:
    from database.connection import check_connection, init_database
    from database import repository
    _DB_AVAILABLE = True
except ImportError as _db_err:
    _DB_AVAILABLE = False
    _DB_IMPORT_ERROR = str(_db_err)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi Halaman Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecast — Indomarco (MySQL)",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Binance Design System CSS
# ---------------------------------------------------------------------------
BINANCE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --primary:          #fcd535;
  --primary-active:   #f0b90b;
  --primary-disabled: #3a3a1f;
  --on-primary:       #181a20;
  --on-dark:          #ffffff;
  --body:             #eaecef;
  --muted:            #707a8a;
  --muted-strong:     #929aa5;
  --canvas-dark:      #0b0e11;
  --surface-card:     #1e2329;
  --surface-elevated: #2b3139;
  --hairline:         #2b3139;
  --trading-up:       #0ecb81;
  --trading-down:     #f6465d;
  --info:             #3b82f6;
  --r-sm: 4px; --r-md: 6px; --r-lg: 8px; --r-xl: 12px;
}
html, body, [class*="css"], .stApp {
  font-family: 'Inter', -apple-system, sans-serif !important;
  background-color: var(--canvas-dark) !important;
  color: var(--body) !important;
}
.main .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1280px !important; }
h1 { font-size: 30px !important; font-weight: 600 !important; color: var(--on-dark) !important; margin-bottom: 4px !important; }
h2 { font-size: 22px !important; font-weight: 600 !important; color: var(--on-dark) !important; margin-top: 2rem !important; }
h3 { font-size: 18px !important; font-weight: 600 !important; color: var(--on-dark) !important; }
p, li { font-size: 14px !important; color: var(--body) !important; }
small { font-size: 12px !important; color: var(--muted) !important; }
hr { border: none !important; border-top: 1px solid var(--hairline) !important; margin: 1.5rem 0 !important; }
[data-testid="stSidebar"] { background-color: var(--surface-card) !important; border-right: 1px solid var(--hairline) !important; }
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.25rem 2rem !important; }
.stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
  background-color: var(--primary) !important; color: var(--on-primary) !important;
  border: none !important; border-radius: var(--r-md) !important;
  font-size: 14px !important; font-weight: 600 !important; padding: 12px 24px !important;
}
.stButton > button[kind="primary"]:hover { background-color: var(--primary-active) !important; }
.stButton > button[kind="secondary"], button[data-testid="baseButton-secondary"] {
  background-color: var(--surface-card) !important; color: var(--on-dark) !important;
  border: 1px solid var(--hairline) !important; border-radius: var(--r-md) !important;
  font-size: 14px !important; font-weight: 600 !important; padding: 12px 24px !important;
}
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--hairline) !important; }
[data-testid="stTabs"] button[role="tab"] {
  font-size: 14px !important; font-weight: 500 !important; color: var(--muted) !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  padding: 10px 20px !important; background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  font-weight: 600 !important; color: var(--primary) !important;
  border-bottom: 2px solid var(--primary) !important;
}
[data-testid="stMetric"] {
  background: var(--surface-card) !important; border: 1px solid var(--hairline) !important;
  border-radius: var(--r-xl) !important; padding: 18px 20px 14px !important;
}
[data-testid="stMetricLabel"] { font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 700 !important; color: var(--primary) !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stSelectbox"] > div > div:first-child {
  background: var(--surface-elevated) !important; border: 1px solid var(--hairline) !important;
  border-radius: var(--r-lg) !important; padding: 8px 12px !important; color: var(--body) !important;
}
[data-testid="stFileUploader"] {
  border: 1px dashed var(--hairline) !important; border-radius: var(--r-xl) !important;
  background: var(--surface-card) !important; padding: 0.75rem !important;
}
.bn-eyebrow { font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.bn-info-card { background: var(--surface-card); border-left: 3px solid var(--primary); border-radius: var(--r-lg); padding: 18px 22px; margin: 1rem 0; }
.bn-info-card h4 { font-size: 15px; font-weight: 600; color: var(--primary); margin: 0 0 4px; }
.bn-info-card p  { font-size: 13px; color: var(--body); margin: 0; }
.db-status-ok   { color: var(--trading-up); font-weight: 600; font-size: 13px; }
.db-status-fail { color: var(--trading-down); font-weight: 600; font-size: 13px; }
</style>
"""
st.markdown(BINANCE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
_SESSION_KEYS = [
    "series_df", "test_results", "future_forecast",
    "eval_report", "engine", "selected_cat",
    "freq", "group_type", "horizon", "db_summary",
]
for _k in _SESSION_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ---------------------------------------------------------------------------
# Inisialisasi Database MySQL (Auto-Create DB & Tables)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _init_mysql_database():
    """Auto-create MySQL DB dan tabel, lalu cek koneksi."""
    if not _DB_AVAILABLE:
        return False, f"Modul database gagal dimuat: {_DB_IMPORT_ERROR if '_DB_IMPORT_ERROR' in globals() else 'Tidak diketahui'}"
    try:
        init_database()
        ok, msg = check_connection()
        return ok, msg
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    with st.sidebar:
        # Header Brand
        st.markdown(
            '<p class="bn-eyebrow" style="margin-top:0">Forecasting System (MySQL)</p>'
            '<p style="font-size:12px;color:#707a8a;margin:0">Prophet + LightGBM Hybrid · XAMPP Native</p>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)

        # ── SEKSI 1: Status Koneksi MySQL ──────────────────────────────
        st.markdown('<p class="bn-eyebrow">Status Basis Data MySQL</p>', unsafe_allow_html=True)

        db_ok, db_msg = _init_mysql_database()
        if db_ok:
            st.markdown(f'<p class="db-status-ok">🟢 {db_msg}</p>', unsafe_allow_html=True)
            db_sum = repository.get_database_summary()
            st.caption(
                f"**{db_sum['total_raw_transactions']:,}** transaksi raw  |  "
                f"**{db_sum['total_divisions']}** divisi  |  **{db_sum['total_categories']}** kategori"
            )
            if db_sum["min_date"] and db_sum["max_date"]:
                st.caption(f"Rentang: `{db_sum['min_date']}` s.d. `{db_sum['max_date']}`")
            st.session_state["db_summary"] = db_sum
        else:
            st.markdown(
                f'<p class="db-status-fail">🔴 Koneksi MySQL Gagal</p>'
                f'<p style="font-size:11px;color:#707a8a">{db_msg}</p>',
                unsafe_allow_html=True,
            )
            st.info("💡 Pastikan service MySQL aktif di **XAMPP Control Panel** (klik Start pada MySQL).")

        st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)

        # ── SEKSI 2: Upload Delta Bulanan ─────────────────────────────
        st.markdown('<p class="bn-eyebrow">Pembaruan Inkremental Bulanan</p>', unsafe_allow_html=True)
        st.caption("Unggah berkas log transaksi kasir bulanan baru (.csv). Hanya delta yang diproses.")
        uploaded_delta = st.file_uploader(
            "Unggah Transaksi Kasir Bulanan (.csv)",
            type=["csv"],
            label_visibility="collapsed",
            help="Kolom wajib: TANGGAL (DD-MM-YYYY), QTY, DIV, CAT_COD",
        )

        sync_clicked = st.button(
            "🔄 Sinkronkan ke Database",
            use_container_width=True,
            disabled=(uploaded_delta is None or not db_ok),
            type="secondary",
        )

        if sync_clicked and uploaded_delta is not None and db_ok:
            _handle_delta_sync(uploaded_delta)

        st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)

        # ── SEKSI 3: Kontrol Parameter Peramalan ──────────────────────
        st.markdown('<p class="bn-eyebrow">Pengaturan Peramalan</p>', unsafe_allow_html=True)

        group_type_label = st.radio(
            "Hierarki Kategori",
            options=["Divisi (DIV)", "Sub-Kategori (CAT_COD)"],
            index=0,
            label_visibility="collapsed",
        )
        group_type = "DIV" if "DIV" in group_type_label else "CAT_COD"

        freq_label = st.selectbox(
            "Interval Waktu",
            options=list(FREQ_OPTIONS.keys()),
            index=0,
        )
        freq = FREQ_OPTIONS[freq_label]

        categories = []
        if db_ok:
            try:
                categories = repository.get_available_categories(group_type)
            except Exception:
                categories = []

        if categories:
            selected_cat = st.selectbox(
                "Kode Kategori / Divisi",
                options=categories,
                label_visibility="collapsed",
                help=f"{len(categories)} pilihan tersedia di basis data",
            )
        else:
            if db_ok:
                st.caption("⚠️ Belum ada data. Jalankan `python -m scripts.seed_historical --file <path_csv>`.")
            selected_cat = None

        horizon = st.slider(
            "Horizon Proyeksi (Periode)",
            min_value=MIN_HORIZON,
            max_value=MAX_HORIZON,
            value=DEFAULT_HORIZON,
            step=1,
            help=f"Maksimal {MAX_HORIZON} periode ke depan",
        )

        with st.expander("⚙️ Parameter Lanjutan", expanded=False):
            seasonality_mode = st.selectbox(
                "Mode Musiman Prophet",
                options=SEASONALITY_MODES,
                index=0,
                format_func=lambda x: "Aditif" if x == "additive" else "Multiplikatif",
            )

        st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)

        run_forecast = st.button(
            "🚀 Jalankan Peramalan",
            use_container_width=True,
            type="primary",
            disabled=(selected_cat is None or not db_ok),
        )

        st.markdown(
            '<p style="font-size:11px;color:#707a8a;margin-top:1rem;line-height:1.6">'
            'MySQL Native Edition (Pure Python PyMySQL)<br>'
            '<strong style="color:#fcd535">Alfiyan Nazar</strong> · 220511053<br>'
            'Teknik Informatika – UMC 2026'
            '</p>',
            unsafe_allow_html=True,
        )

    return {
        "freq":             freq,
        "freq_label":       freq_label,
        "group_type":       group_type,
        "selected_cat":     selected_cat,
        "seasonality_mode": seasonality_mode,
        "horizon":          horizon,
        "run_forecast":     run_forecast,
        "db_ok":            db_ok,
    }


# ---------------------------------------------------------------------------
# Handler Sinkronisasi Data Delta
# ---------------------------------------------------------------------------
def _handle_delta_sync(uploaded_file) -> None:
    preprocessor = DeltaPreprocessor()
    try:
        with st.spinner(f"Memvalidasi {uploaded_file.name} …"):
            clean_df = preprocessor.clean_delta(uploaded_file)
            summary  = preprocessor.get_delta_summary(clean_df)

        batch_id = f"DELTA_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with st.spinner("Menyimpan transaksi baru ke MySQL (chunk 25.000) …"):
            n_inserted = repository.fast_bulk_insert_raw(clean_df, batch_id=batch_id, chunksize=25_000)

        with st.spinner("Menjalankan kalkulasi agregasi inkremental (ON DUPLICATE KEY UPDATE) …"):
            start_date = summary["tanggal_awal"].date() if hasattr(summary["tanggal_awal"], "date") else summary["tanggal_awal"]
            repository.sync_aggregations(start_date=start_date)

        _init_mysql_database.clear()

        st.success(
            f"✅ **{n_inserted:,} baris transaksi baru** berhasil disimpan ke MySQL!\n\n"
            f"- Rentang: {summary['tanggal_awal'].strftime('%d %b %Y')} s.d. {summary['tanggal_akhir'].strftime('%d %b %Y')}\n"
            f"- Total QTY: {summary['total_qty']:,} unit\n"
            f"- Deret waktu siap diramalkan tanpa membaca ulang CSV master."
        )
        st.rerun()

    except ValueError as e:
        st.error(f"Validasi gagal: {e}")
    except Exception as e:
        st.error(f"Gagal sinkronisasi data: {e}")
        logger.exception("Delta sync error.")


# ---------------------------------------------------------------------------
# Handler Eksekusi Peramalan Hibrida
# ---------------------------------------------------------------------------
def handle_forecast_execution(params: dict) -> None:
    if not params["run_forecast"]:
        return

    selected_cat = params["selected_cat"]
    group_type   = params["group_type"]
    freq         = params["freq"]

    if selected_cat is None:
        st.warning("Pilih kategori atau divisi terlebih dahulu.")
        return

    evaluator = ModelEvaluator()

    with st.spinner(f"Mengambil data {group_type}={selected_cat} dari MySQL (<0.1 detik) …"):
        # Kueri instan dari MySQL sales_aggregations
        series_df = repository.fetch_series(
            group_type=group_type,
            group_code=selected_cat,
            freq=freq,
        )

    if series_df.empty or len(series_df) < MIN_DATA_POINTS:
        st.warning(
            f"Data tidak mencukupi untuk {group_type}=**{selected_cat}** "
            f"(hanya {len(series_df)} periode). Dibutuhkan minimal {MIN_DATA_POINTS} observasi."
        )
        return

    with st.spinner(
        f"Melatih model hibrida Prophet + LightGBM untuk {group_type}={selected_cat} …"
    ):
        try:
            engine = HybridForecastingEngine(
                seasonality_mode=params["seasonality_mode"],
                test_size=0.20,
            )
            engine.fit(series_df, freq=freq)

            test_results    = engine.predict_test()
            future_forecast = engine.forecast_future(series_df, horizon=params["horizon"])
            eval_report     = evaluator.build_evaluation_report(test_results)

            st.session_state.update({
                "series_df":       series_df,
                "test_results":    test_results,
                "future_forecast": future_forecast,
                "eval_report":     eval_report,
                "engine":          engine,
                "selected_cat":    selected_cat,
                "group_type":      group_type,
                "freq":            freq,
                "horizon":         params["horizon"],
            })

            mape = eval_report["metrics_hybrid"]["MAPE_%"]
            label, emoji = evaluator.interpret_mape(mape)
            st.success(
                f"{emoji} Peramalan selesai · {group_type}=**{selected_cat}** · "
                f"MAPE Hibrida: **{mape:.2f}%** ({label})"
            )

        except Exception as e:
            st.error(f"Peramalan gagal: {e}")
            logger.exception("Forecast error.")


# ---------------------------------------------------------------------------
# Welcome Screen
# ---------------------------------------------------------------------------
def _render_welcome() -> None:
    st.markdown(
        """
        <div class="bn-info-card" style="margin-top:0">
          <h4>Selamat Datang di Sistem Peramalan Retail (MySQL Native Edition)</h4>
          <p>Sistem ini menggunakan basis data MySQL lokal (XAMPP) tanpa Docker daemon.
          Pilih kategori dari sidebar dan klik <strong>Jalankan Peramalan</strong> untuk memulai peramalan dua tahap.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    db_sum = st.session_state.get("db_summary")
    if db_sum and db_sum["total_raw_transactions"] > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Transaksi Raw", f"{db_sum['total_raw_transactions']:,}")
        with col2:
            st.metric("🗂️ Jumlah Divisi", db_sum["total_divisions"])
        with col3:
            st.metric("🏷️ Jumlah Kategori", db_sum["total_categories"])
        with col4:
            if db_sum["min_date"] and db_sum["max_date"]:
                n_days = (db_sum["max_date"] - db_sum["min_date"]).days
                st.metric("📅 Rentang Data", f"{n_days} hari")
    else:
        st.info(
            "📌 Basis data MySQL masih kosong. Jalankan perintah seeding awal:\n"
            "```bash\npython -m scripts.seed_historical --file path/to/merged_MTRAN_with_desc.csv\n```"
        )


# ---------------------------------------------------------------------------
# AREA UTAMA (4 TABS)
# ---------------------------------------------------------------------------
def render_main_area(params: dict) -> None:
    series_df       = st.session_state.get("series_df")
    test_results    = st.session_state.get("test_results")
    future_forecast = st.session_state.get("future_forecast")
    eval_report     = st.session_state.get("eval_report")
    engine          = st.session_state.get("engine")
    selected_cat    = st.session_state.get("selected_cat")
    group_type      = st.session_state.get("group_type", "DIV")
    freq            = st.session_state.get("freq", "W")

    vis = DashboardVisualizer()
    evaluator = ModelEvaluator()
    freq_label = params.get("freq_label", "Mingguan")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tab 1 — Data Explorer",
        "📈 Tab 2 — Grafik Hibrida",
        "🎯 Tab 3 — Evaluasi Akurasi",
        "📥 Tab 4 — Ekspor Proyeksi",
    ])

    # ── TAB 1: Data Explorer ───────────────────────────────────────────
    with tab1:
        if series_df is None:
            _render_welcome()
        else:
            st.markdown(f"### Deret Waktu Penjualan — {group_type}: `{selected_cat}` ({freq_label})")
            st.caption(
                f"Data diambil langsung dari tabel `sales_aggregations` (MySQL) · "
                f"{len(series_df)} periode · "
                f"{series_df['ds'].min().strftime('%d %b %Y')} s.d. {series_df['ds'].max().strftime('%d %b %Y')}"
            )

            fig_ov = go.Figure()
            fig_ov.add_trace(go.Scatter(
                x=series_df["ds"], y=series_df["y"],
                name="Penjualan Aktual", mode="lines+markers",
                line=dict(color="#eaecef", width=2), marker=dict(size=4),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:,.0f} unit<extra></extra>",
            ))
            fig_ov.update_layout(
                title=f"<b>Deret Waktu Historis — {group_type}: {selected_cat}</b>",
                height=340, paper_bgcolor="#0b0e11", plot_bgcolor="#1e2329",
                font=dict(color="#eaecef"), hovermode="x unified",
                xaxis=dict(gridcolor="#2b3139"), yaxis=dict(gridcolor="#2b3139"),
                margin=dict(l=40, r=20, t=60, b=40),
            )
            st.plotly_chart(fig_ov, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Total Penjualan", f"{series_df['y'].sum():,.0f} unit")
            with c2: st.metric("Rata-rata/Periode", f"{series_df['y'].mean():,.0f} unit")
            with c3: st.metric("Kuantitas Maks", f"{series_df['y'].max():,.0f} unit")
            with c4: st.metric("Kuantitas Min", f"{series_df['y'].min():,.0f} unit")

            with st.expander("🔍 Pratinjau Tabel Deret Waktu"):
                disp = series_df.copy()
                disp["ds"] = disp["ds"].dt.strftime("%d %b %Y")
                disp.columns = ["Periode", "Kuantitas Penjualan (unit)"]
                disp["Kuantitas Penjualan (unit)"] = disp["Kuantitas Penjualan (unit)"].round(0).astype(int)
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── TAB 2: Grafik Hibrida ──────────────────────────────────────────
    with tab2:
        if test_results is None or series_df is None:
            st.info("Jalankan peramalan terlebih dahulu via tombol di sidebar.")
        else:
            eval_rep  = eval_report or {}
            metrics_h = eval_rep.get("metrics_hybrid", {})
            mape_val  = metrics_h.get("MAPE_%", 0)
            rmse_val  = metrics_h.get("RMSE", 0)
            mae_val   = metrics_h.get("MAE", 0)
            trend_pct = vis.calculate_trend_pct(series_df)
            mape_label, mape_emoji = evaluator.interpret_mape(mape_val)

            vis.render_kpi_cards(
                total_sales = int(series_df["y"].sum()),
                mape_val    = mape_val,
                rmse_val    = rmse_val,
                mae_val     = mae_val,
                trend_pct   = trend_pct,
                mape_label  = mape_label,
                mape_emoji  = mape_emoji,
            )
            st.markdown("")

            fig = vis.plot_forecast(
                historical_df   = series_df,
                test_results    = test_results,
                future_forecast = future_forecast or pd.DataFrame(),
                title           = "Kurva Peramalan Penjualan Hibrida Prophet + LightGBM",
                category_label  = f"{group_type}: {selected_cat}",
                freq_label      = freq_label,
                show_prophet    = True,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Analisis Galat: Baseline Prophet vs Model Hibrida")
            st.caption("Perbandingan galat absolut |y − ŷ| menunjukkan reduksi galat berkat koreksi residual LightGBM.")
            fig_res = vis.plot_residual_analysis(test_results, category_label=f"{selected_cat}")
            st.plotly_chart(fig_res, use_container_width=True)

    # ── TAB 3: Evaluasi Akurasi ────────────────────────────────────────
    with tab3:
        if eval_report is None:
            st.info("Jalankan peramalan terlebih dahulu.")
        else:
            metrics_p = eval_report["metrics_prophet"]
            metrics_h = eval_report["metrics_hybrid"]
            mape_val  = metrics_h["MAPE_%"]
            mape_label, mape_emoji = evaluator.interpret_mape(mape_val)

            st.markdown("#### Metrik Akurasi pada Data Uji (20% Temporal Split)")
            st.caption(f"Evaluasi dihitung terhadap **{eval_report['n_test_points']} titik observasi** tanpa data leakage.")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    label=f"{mape_emoji} MAPE Hibrida",
                    value=f"{mape_val:.2f}%",
                    delta=mape_label,
                    delta_color="off",
                    help="< 10% = Sangat Baik, 10-20% = Baik, 20-50% = Layak.",
                )
            with c2:
                st.metric(
                    label="📐 RMSE",
                    value=f"{metrics_h['RMSE']:,.2f}",
                    delta=f"Prophet: {metrics_p['RMSE']:,.2f}",
                )
            with c3:
                st.metric(
                    label="📏 MAE",
                    value=f"{metrics_h['MAE']:,.2f}",
                    delta=f"Prophet: {metrics_p['MAE']:,.2f}",
                )

            st.markdown("#### Tabel Perbandingan Kinerja Model")
            vis.render_comparison_table(eval_report["comparison_table"])

            if engine is not None:
                fi_df = engine.get_feature_importance()
                if fi_df is not None and not fi_df.empty:
                    st.markdown("#### Kontribusi Fitur Residual (LightGBM Feature Importance)")
                    st.caption("Peringkat fitur lag dan kalender yang paling berkontribusi dalam mengoreksi residual Prophet.")
                    fig_fi = vis.plot_feature_importance(fi_df)
                    st.plotly_chart(fig_fi, use_container_width=True)

    # ── TAB 4: Ekspor Proyeksi ─────────────────────────────────────────
    with tab4:
        if future_forecast is None:
            st.info("Jalankan peramalan terlebih dahulu untuk mengekspor data proyeksi.")
        else:
            st.markdown(f"#### Hasil Proyeksi Masa Depan ({len(future_forecast)} Periode)")
            st.caption(f"Model: Hibrida Prophet + LightGBM · Basis Data: MySQL · Kategori: {group_type}={selected_cat}")

            vis.render_forecast_table(future_forecast, freq_label=freq_label)

            csv_bytes = vis.export_csv_bytes(
                future_forecast,
                category_label=f"{group_type}: {selected_cat}",
            )
            cat_clean = selected_cat.replace("/", "-") if selected_cat else "hasil"
            st.download_button(
                label="⬇️ Unduh Berkas Proyeksi (CSV)",
                data=csv_bytes,
                file_name=f"proyeksi_{cat_clean}_{freq}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

            st.markdown("---")
            st.markdown(
                """
                <div class="bn-info-card">
                  <h4>Formulasi Matematis Model Hibrida</h4>
                  <p>
                    1. <strong>Tahap 1 (Prophet)</strong>: Dekomposisi aditif deret waktu:
                    <code>y(t) = g(t) + s(t) + ε(t)</code><br>
                    2. <strong>Tahap 2 (LightGBM)</strong>: Estimasi residual non-linier:
                    <code>ê(t) ≈ y_aktual(t) − ŷ_Prophet(t)</code><br>
                    3. <strong>Rekonsiliasi Akhir</strong>:
                    <code>ŷ_hybrid(t) = max(0, ŷ_Prophet(t) + ê_LightGBM(t))</code>
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(
        f"""
        <div style="padding: 1rem 0 0.5rem; border-bottom: 1px solid #2b3139; margin-bottom: 1.5rem;">
          <h1 style="margin:0">📦 {APP_TITLE}</h1>
          <p style="color:#707a8a;margin:4px 0 0;font-size:13px">{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    params = render_sidebar()
    handle_forecast_execution(params)
    render_main_area(params)


if __name__ == "__main__":
    main()
