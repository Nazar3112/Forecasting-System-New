"""
core/lgbm_engine.py
===================
Modul Tahap 2 — Residual Regression menggunakan LightGBM.

═══════════════════════════════════════════════════════════════════════
METODE: LIGHTGBM RESIDUAL REGRESSION
═══════════════════════════════════════════════════════════════════════
LightGBM digunakan pada Tahap 2 untuk meregresikan RESIDUAL
dari prediksi Prophet Tahap 1.

Formulasi:
  Target pelatihan:
    ê(t) = y_aktual(t) − ŷ_Prophet(t)      [residual = galat Prophet]

  Input fitur (X):
    - Kalender: month, quarter, weekofyear, dayofweek, ...
    - Lag: lag_1, lag_2, lag_3, lag_4        [pola autoregresif]
    - Rolling: roll_mean_4, roll_std_4, roll_mean_8, roll_std_8

  Formula prediksi hibrida akhir:
    ŷ_hybrid(t) = max(0, ŷ_Prophet(t) + ê_LightGBM(t))

Algoritma LightGBM:
  - Gradient Boosting berbasis pohon keputusan (Decision Tree Ensemble)
  - Histogram-based learning: efisien untuk data besar
  - Leaf-wise tree growth: akurasi lebih tinggi dari level-wise

Referensi Skripsi:
  - Proposal Skripsi Alfiyan Nazar (220511053) — Bab III hal. 13–14
  - Teknik Informatika UMC 2026
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from core.config import LIGHTGBM_PARAMS

logger = logging.getLogger(__name__)


class LightGBMResidualEngine:
    """
    Mesin Tahap 2: Regresi residual menggunakan LightGBM.

    Belajar memprediksi pola galat yang tidak tertangkap Prophet,
    sehingga hasil hibrida lebih akurat dari Prophet tunggal.
    """

    def __init__(
        self,
        lgbm_params:     Dict | None = None,
        lags:            List[int] | None = None,
        rolling_windows: List[int] | None = None,
    ):
        from core.config import RESIDUAL_LAG_STEPS, ROLLING_WINDOWS
        _params = {**LIGHTGBM_PARAMS}
        if lgbm_params:
            _params.update(lgbm_params)
        self._lgbm_params    = _params
        self.lags            = lags or RESIDUAL_LAG_STEPS
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.model:          Optional[LGBMRegressor] = None
        self._feature_names: List[str]               = []
        self._is_fitted:     bool                    = False

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "LightGBMResidualEngine":
        """
        Latih LGBMRegressor pada matriks fitur dan target residual.

        ── METODE LIGHTGBM (TAHAP 2) ─────────────────────────────────
        Target y_train = ê(t) = y_aktual(t) − ŷ_Prophet(t)

        LightGBM membangun ensemble pohon keputusan secara berurutan
        (gradient boosting), setiap pohon memperbaiki galat pohon sebelumnya.
        """
        if len(X_train) == 0:
            return self
        self._feature_names = list(X_train.columns)
        self.model = LGBMRegressor(**self._lgbm_params)
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("LightGBM dilatih: %d sampel, %d fitur", len(X_train), X_train.shape[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediksi koreksi residual pada test set."""
        self._require_fitted()
        return self.model.predict(X[self._feature_names])

    def predict_recursive(
        self,
        future_dates:     pd.DatetimeIndex,
        y_prophet_future: np.ndarray,
        y_history_buffer: list,
        feature_engine,
    ) -> np.ndarray:
        """
        Peramalan autoregresif rekursif untuk masa depan.

        ── PERAMALAN AUTOREGRESIF ONE-STEP-AHEAD ─────────────────────
        Untuk t = T+1, ..., T+horizon:
          1. Bangun fitur x(t) dari buffer historis
          2. ê_LightGBM(t) = model.predict(x(t))
          3. ŷ_hybrid(t)   = max(0, ŷ_Prophet(t) + ê_LightGBM(t))
          4. Tambahkan ŷ_hybrid(t) ke buffer (sebagai lag t berikutnya)
        """
        self._require_fitted()
        res_preds = []

        for i, dt in enumerate(future_dates):
            feat_row = {
                "month":          dt.month,
                "weekofyear":     int(dt.isocalendar().week),
                "quarter":        dt.quarter,
                "day":            dt.day,
                "dayofweek":      dt.dayofweek,
                "is_month_end":   int(dt.is_month_end),
                "is_month_start": int(dt.is_month_start),
            }
            for lag in self.lags:
                feat_row[f"lag_{lag}"] = float(y_history_buffer[-lag]) if len(y_history_buffer) >= lag else 0.0
            for win in self.rolling_windows:
                w = y_history_buffer[-win:] if len(y_history_buffer) >= win else y_history_buffer
                feat_row[f"roll_mean_{win}"] = float(np.mean(w)) if w else 0.0
                feat_row[f"roll_std_{win}"]  = float(np.std(w)) if len(w) > 1 else 0.0

            X_step   = pd.DataFrame([feat_row])[self._feature_names]
            step_res = float(self.model.predict(X_step)[0])
            res_preds.append(step_res)

            # ŷ_hybrid(t) = max(0, ŷ_Prophet(t) + ê_LightGBM(t))
            step_hybrid = max(0.0, float(y_prophet_future[i]) + step_res)
            y_history_buffer.append(step_hybrid)

        return np.array(res_preds)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Kembalikan tabel feature importance LightGBM (gain-based)."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return None
        return (
            pd.DataFrame({
                "feature":    self._feature_names,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("LightGBMResidualEngine belum dilatih. Panggil .fit() dulu.")
