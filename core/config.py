"""
core/config.py
==============
Konfigurasi global sistem peramalan — versi MySQL Native.
"""

# ---------------------------------------------------------------------------
# PARAMETER DERET WAKTU
# ---------------------------------------------------------------------------
FREQ_OPTIONS: dict[str, str] = {
    "Mingguan (W)": "W",
    "Bulanan (M)":  "M",
}
DEFAULT_FREQ:     str = "W"
MIN_DATA_POINTS:  int = 24

# ---------------------------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
DEFAULT_TEST_SIZE: float = 0.20   # 20% data terakhir sebagai data uji

# ---------------------------------------------------------------------------
# HORIZON PROYEKSI
# ---------------------------------------------------------------------------
DEFAULT_HORIZON: int = 8
MIN_HORIZON:     int = 1
MAX_HORIZON:     int = 24

# ---------------------------------------------------------------------------
# PARAMETER MODEL PROPHET
# Dekomposisi aditif: y(t) = g(t) + s(t) + h(t) + ε(t)
# Referensi: Taylor & Letham (2018) — "Forecasting at Scale"
# ---------------------------------------------------------------------------
PROPHET_PARAMS: dict = {
    "yearly_seasonality":      True,
    "weekly_seasonality":      True,
    "daily_seasonality":       False,
    "seasonality_mode":        "additive",
    "interval_width":          0.80,
    "changepoint_prior_scale": 0.05,
}
SEASONALITY_MODES: list[str] = ["additive", "multiplicative"]

# ---------------------------------------------------------------------------
# PARAMETER MODEL LIGHTGBM
# Target: residual ε(t) = y_aktual(t) − ŷ_Prophet(t)
# Fitur: lag kuantitas + kalender waktu
# ---------------------------------------------------------------------------
LIGHTGBM_PARAMS: dict = {
    "n_estimators":      200,
    "learning_rate":     0.05,
    "num_leaves":        15,
    "max_depth":         5,
    "min_child_samples": 5,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "verbosity":         -1,
}
RESIDUAL_LAG_STEPS: list[int] = [1, 2, 3, 4]
ROLLING_WINDOWS:    list[int] = [4, 8]

# ---------------------------------------------------------------------------
# PARAMETER UI
# ---------------------------------------------------------------------------
APP_TITLE:    str = "Sistem Prediksi Tren Penjualan Retail"
APP_SUBTITLE: str = "Model Hibrida Prophet + LightGBM"
APP_ICON:     str = "📦"

# Warna grafik Plotly (Binance Design System)
COLOR_ACTUAL:   str = "#eaecef"
COLOR_PROPHET:  str = "#929aa5"
COLOR_HYBRID:   str = "#0ecb81"
COLOR_FUTURE:   str = "#fcd535"
COLOR_RESIDUAL: str = "#3b82f6"

MAPE_THRESHOLDS: dict[str, tuple] = {
    "Sangat Baik": (0.0,  10.0),
    "Baik":        (10.0, 20.0),
    "Layak":       (20.0, 50.0),
    "Kurang":      (50.0, float("inf")),
}

# Validasi delta CSV upload
REQUIRED_DELTA_COLUMNS: list[str] = ["TANGGAL", "QTY", "DIV", "CAT_COD"]
DATE_FORMAT: str = "%d-%m-%Y"
VALID_RTYPE: str = "J"
