"""
core/feature_engine.py
======================
Modul rekayasa fitur (Feature Engineering) untuk LightGBM.

Fitur yang dihasilkan untuk Tahap 2 Residual Regression:
  1. Kalender waktu dari kolom 'ds': month, weekofyear, quarter, dll.
  2. Lag kuantitas penjualan: lag_1, lag_2, lag_3, lag_4
  3. Rolling statistics: roll_mean_4, roll_std_4, roll_mean_8, roll_std_8

Semua fitur ini digunakan LightGBM untuk memprediksi:
    residual(t) = y_aktual(t) − ŷ_Prophet(t)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


class FeatureEngine:
    """Membangun matriks fitur independen (X) untuk LightGBM residual model."""

    def __init__(
        self,
        lags:            List[int] | None = None,
        rolling_windows: List[int] | None = None,
    ):
        from core.config import RESIDUAL_LAG_STEPS, ROLLING_WINDOWS
        self.lags            = lags or RESIDUAL_LAG_STEPS
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS

    @staticmethod
    def extract_exogenous_features(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
        """
        Ekstrak fitur kalender dari kolom tanggal 'ds'.

        Fitur: month, weekofyear, quarter, day, dayofweek,
               is_month_end, is_month_start
        """
        dt = pd.to_datetime(dates)
        return pd.DataFrame({
            "month":          dt.dt.month.values,
            "weekofyear":     dt.dt.isocalendar().week.astype(int).values,
            "quarter":        dt.dt.quarter.values,
            "day":            dt.dt.day.values,
            "dayofweek":      dt.dt.dayofweek.values,
            "is_month_end":   dt.dt.is_month_end.astype(int).values,
            "is_month_start": dt.dt.is_month_start.astype(int).values,
        })

    def build_feature_matrix(
        self,
        df:       pd.DataFrame,
        y_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Bangun matriks fitur lengkap: kalender + lag + rolling.

        Fitur lag menggunakan shift() untuk mencegah data leakage.

        Parameters
        ----------
        df       : pd.DataFrame  Harus mengandung kolom 'ds'.
        y_series : pd.Series     Nilai kuantitas untuk lag/rolling.

        Returns
        -------
        pd.DataFrame  (baris awal NaN dari lag/rolling — perlu dropna)
        """
        exog = self.extract_exogenous_features(df["ds"])
        exog.index = df.index

        if y_series is not None:
            s = pd.Series(y_series.values, index=df.index)

            # ─── Fitur Lag ─────────────────────────────────────────────
            # lag_k = penjualan k periode sebelumnya (autoregresif)
            for lag in self.lags:
                exog[f"lag_{lag}"] = s.shift(lag).values

            # ─── Fitur Rolling ─────────────────────────────────────────
            # Rata-rata & std bergerak dari periode sebelumnya
            for win in self.rolling_windows:
                exog[f"roll_mean_{win}"] = s.shift(1).rolling(win).mean().values
                exog[f"roll_std_{win}"]  = s.shift(1).rolling(win).std().fillna(0).values

        return exog

    def get_max_lookback(self) -> int:
        """Baris awal yang harus dibuang karena NaN lag/rolling."""
        return max(
            max(self.lags) if self.lags else 0,
            max(self.rolling_windows) if self.rolling_windows else 0,
        )
