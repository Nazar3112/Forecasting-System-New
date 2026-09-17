"""
core/engine.py
==============
Kelas HybridForecastingEngine: Orkestrator pipeline peramalan dua tahap.

═══════════════════════════════════════════════════════════════════════
METODE HIBRIDA: PROPHET + LIGHTGBM (MySQL Persistent)
═══════════════════════════════════════════════════════════════════════
Pipeline dua tahap sesuai proposal skripsi Bab 3 (hal. 13–14):

  TAHAP 1 — Facebook Prophet (Baseline Decomposition):
    Mendekomposisi deret waktu penjualan menjadi komponen:
      ŷ_Prophet(t) = g(t) + s(t)
    di mana g(t) = tren, s(t) = musiman (aditif/multiplikatif)

  TAHAP 2 — LightGBM (Residual Regression):
    Meregresikan sisa galat (residual) dari Prophet menggunakan
    fitur lag kuantitas + kalender waktu:
      ê_LightGBM(t) ≈ residual(t) = y_aktual(t) − ŷ_Prophet(t)

  REKONSILIASI HIBRIDA (Non-Negatif):
    ŷ_hybrid(t) = max(0, ŷ_Prophet(t) + ê_LightGBM(t))

  Perlindungan batas bawah max(0, ...) memastikan prediksi penjualan
  tidak pernah bernilai negatif (jumlah barang terjual selalu ≥ 0).

Data dibaca langsung dari database MySQL (tabel sales_aggregations),
menghilangkan kebutuhan membaca CSV 686 MB ke memori.

Referensi: PROPOSAL SKRIPSI — Model Hibrida Prophet + LightGBM
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import (
    DEFAULT_TEST_SIZE,
    DEFAULT_FREQ,
    DEFAULT_HORIZON,
    RESIDUAL_LAG_STEPS,
    ROLLING_WINDOWS,
)
from core.feature_engine import FeatureEngine
from core.prophet_engine import ProphetBaselineEngine
from core.lgbm_engine import LightGBMResidualEngine

logger = logging.getLogger(__name__)


class HybridForecastingEngine:
    """
    Orkestrator pipeline peramalan hibrida Prophet + LightGBM.

    ── ALUR KOMPUTASI ────────────────────────────────────────────────
    1. fit(series_df)
       a. train_test_split()           → Bagi data kronologis 80/20
       b. ProphetBaselineEngine.fit()  → Tahap 1: latih Prophet
       c. extract_train_residuals()    → residual = y - ŷ_Prophet
       d. FeatureEngine.build()        → Bangun matriks fitur lag+kalender
       e. LightGBMResidualEngine.fit() → Tahap 2: latih LightGBM pada residual

    2. predict_test()
       a. Prophet predict pada test set
       b. LightGBM predict koreksi residual pada test set
       c. ŷ_hybrid = max(0, ŷ_Prophet + ê_LightGBM)

    3. forecast_future(horizon)
       a. Prediksi Prophet untuk n periode ke depan
       b. Prediksi LightGBM rekursif (autoregresif)
       c. ŷ_hybrid = max(0, ŷ_Prophet + ê_LightGBM) per periode
    """

    def __init__(
        self,
        seasonality_mode: str              = "additive",
        test_size:        float            = DEFAULT_TEST_SIZE,
        lags:             List[int] | None = None,
        rolling_windows:  List[int] | None = None,
        prophet_params:   Dict | None      = None,
        lgbm_params:      Dict | None      = None,
    ):
        self.seasonality_mode = seasonality_mode
        self.test_size        = test_size
        self.lags             = lags or RESIDUAL_LAG_STEPS
        self.rolling_windows  = rolling_windows or ROLLING_WINDOWS

        # ── Inisialisasi Sub-Engine ───────────────────────────────────
        self._prophet_engine = ProphetBaselineEngine(
            seasonality_mode=seasonality_mode,
            prophet_params=prophet_params,
        )

        self._lgbm_engine = LightGBMResidualEngine(
            lgbm_params=lgbm_params,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
        )

        self._feature_engine = FeatureEngine(
            lags=self.lags,
            rolling_windows=self.rolling_windows,
        )

        self._train_df: Optional[pd.DataFrame] = None
        self._test_df:  Optional[pd.DataFrame] = None
        self._freq:     str                    = DEFAULT_FREQ
        self._is_fitted: bool                  = False

    # ------------------------------------------------------------------
    # Public API: Pelatihan
    # ------------------------------------------------------------------

    def fit(
        self,
        series_df: pd.DataFrame,
        freq:      str = DEFAULT_FREQ,
    ) -> "HybridForecastingEngine":
        """
        Jalankan pipeline pelatihan hibrida lengkap.

        series_df diambil dari MySQL sales_aggregations via repository.fetch_series().

        Parameters
        ----------
        series_df : pd.DataFrame  Kolom ['ds', 'y'] dari database.
        freq      : str           Frekuensi deret waktu ('W' atau 'M').

        Returns
        -------
        self
        """
        self._freq = freq
        series_df  = series_df[["ds", "y"]].dropna().sort_values("ds").reset_index(drop=True)

        # ─── 1. Partisi Train / Test (Kronologis) ─────────────────────
        self._train_df, self._test_df = self._train_test_split(series_df)
        logger.info(
            "Train/Test split: train=%d | test=%d periode",
            len(self._train_df), len(self._test_df),
        )

        # ─── 2. TAHAP 1: Pelatihan Prophet Baseline ───────────────────
        self._prophet_engine.fit(self._train_df)
        residuals = self._prophet_engine.extract_train_residuals()

        # ─── 3. Rekayasa Fitur ─────────────────────────────────────────
        full_train_feat = self._feature_engine.build_feature_matrix(
            self._train_df, self._train_df["y"]
        )
        max_lookback = self._feature_engine.get_max_lookback()

        X_train = full_train_feat.iloc[max_lookback:].reset_index(drop=True)
        y_train = residuals[max_lookback:]

        # ─── 4. TAHAP 2: Pelatihan LightGBM Residual ──────────────────
        self._lgbm_engine.fit(X_train, y_train)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Public API: Prediksi pada Test Set
    # ------------------------------------------------------------------

    def predict_test(self) -> pd.DataFrame:
        """Hasilkan prediksi hibrida pada data uji."""
        self._require_fitted()
        return self._generate_hybrid_predictions(self._test_df, is_future=False)

    # ------------------------------------------------------------------
    # Public API: Proyeksi Masa Depan
    # ------------------------------------------------------------------

    def forecast_future(
        self,
        series_df: pd.DataFrame,
        horizon:   int = DEFAULT_HORIZON,
    ) -> pd.DataFrame:
        """
        Proyeksikan penjualan untuk n periode ke masa depan (rekursif).

        Parameters
        ----------
        series_df : pd.DataFrame  Seluruh deret waktu historis ['ds', 'y'].
        horizon   : int           Jumlah periode ke depan (1 s.d. 24).

        Returns
        -------
        pd.DataFrame  Kolom: ['ds', 'y_prophet', 'y_residual_lgbm', 'y_hybrid']
        """
        self._require_fitted()

        last_date = series_df["ds"].max()
        freq_map = {"W": "W-MON", "M": "ME"}
        pandas_freq = freq_map.get(self._freq, self._freq)

        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(pandas_freq),
            periods=horizon,
            freq=pandas_freq,
        )
        future_df = pd.DataFrame({"ds": future_dates, "y": np.nan})
        return self._generate_hybrid_predictions(future_df, is_future=True)

    # ------------------------------------------------------------------
    # Private: Generasi Prediksi Hibrida
    # ------------------------------------------------------------------

    def _generate_hybrid_predictions(
        self,
        period_df: pd.DataFrame,
        is_future: bool = False,
    ) -> pd.DataFrame:
        """
        Rekonsiliasi dua tahap:
        ŷ_hybrid(t) = max(0, ŷ_Prophet(t) + ê_LightGBM(t))
        """
        result = period_df[["ds"]].copy()
        if "y" in period_df.columns:
            result["y"] = period_df["y"].values

        # Tahap 1: Prophet Baseline
        result["y_prophet"] = self._prophet_engine.predict(result)

        # Tahap 2: LightGBM Residual
        if self._lgbm_engine._is_fitted:
            if not is_future:
                combined_df = pd.concat([self._train_df, period_df], ignore_index=True)
                full_feat   = self._feature_engine.build_feature_matrix(
                    combined_df, combined_df["y"]
                )
                X_test = full_feat.tail(len(period_df))[self._lgbm_engine.feature_names]
                X_test = X_test.reset_index(drop=True)
                result["y_residual_lgbm"] = self._lgbm_engine.predict(X_test)
            else:
                full_hist = pd.concat([self._train_df, self._test_df], ignore_index=True)
                y_buffer  = list(full_hist["y"].fillna(0).values)

                result["y_residual_lgbm"] = self._lgbm_engine.predict_recursive(
                    future_dates     = pd.DatetimeIndex(result["ds"]),
                    y_prophet_future = result["y_prophet"].values,
                    y_history_buffer = y_buffer,
                    feature_engine   = self._feature_engine,
                )
        else:
            result["y_residual_lgbm"] = 0.0

        # Formula hibrida non-negatif
        result["y_hybrid"] = (result["y_prophet"] + result["y_residual_lgbm"]).clip(lower=0)
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private: Train/Test Split Kronologis
    # ------------------------------------------------------------------

    def _train_test_split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Partisi temporal 80% Train / 20% Test (tanpa data leakage)."""
        n_test   = max(1, int(len(df) * self.test_size))
        n_train  = len(df) - n_test
        train_df = df.iloc[:n_train].reset_index(drop=True)
        test_df  = df.iloc[n_train:].reset_index(drop=True)
        return train_df, test_df

    # ------------------------------------------------------------------
    # Utilitas Publik
    # ------------------------------------------------------------------

    def get_prophet_components(self) -> Optional[pd.DataFrame]:
        if not self._is_fitted:
            return None
        full_df = pd.concat([self._train_df, self._test_df], ignore_index=True)
        return self._prophet_engine.get_components(full_df)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return self._lgbm_engine.get_feature_importance()

    # ------------------------------------------------------------------
    # Backward Compatibility
    # ------------------------------------------------------------------

    def train_test_split(self, df: pd.DataFrame):
        return self._train_test_split(df)

    def build_feature_matrix(self, df, y_series=None):
        return self._feature_engine.build_feature_matrix(df, y_series)

    @staticmethod
    def extract_exogenous_features(dates):
        return FeatureEngine.extract_exogenous_features(dates)

    def fit_lightgbm_residual(self, X_train, y_train):
        return self._lgbm_engine.fit(X_train, y_train)

    def generate_hybrid_predictions(self, period_df, is_future=False):
        return self._generate_hybrid_predictions(period_df, is_future)

    @property
    def prophet_model(self):
        return self._prophet_engine.model

    @property
    def lgbm_model(self):
        return self._lgbm_engine.model

    @property
    def _feature_names(self):
        return self._lgbm_engine.feature_names

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "HybridForecastingEngine belum dilatih. Panggil .fit() terlebih dahulu."
            )
