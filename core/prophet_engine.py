"""
core/prophet_engine.py
======================
Modul Tahap 1 — Baseline Decomposition menggunakan Facebook Prophet.

═══════════════════════════════════════════════════════════════════════
METODE: FACEBOOK PROPHET (Taylor & Letham, 2018)
═══════════════════════════════════════════════════════════════════════
Prophet adalah model dekomposisi deret waktu aditif:

    y(t) = g(t) + s(t) + h(t) + ε(t)

di mana:
  g(t) : Fungsi tren — piecewise linear atau logistic growth
  s(t) : Komponen musiman — dimodelkan dengan Fourier series
           s(t) = Σ [a_n·cos(2πnt/P) + b_n·sin(2πnt/P)]
  h(t) : Efek hari libur / event khusus
  ε(t) : Error term (noise residual)

Output utama:
  ŷ_Prophet(t)  → Prediksi baseline dari dekomposisi tren + musiman
  residual(t)   = y_aktual(t) − ŷ_Prophet(t)  → Target LightGBM (Tahap 2)

Referensi Skripsi:
  - Proposal Skripsi Alfiyan Nazar (220511053) — Bab III hal. 13–14
  - Teknik Informatika UMC 2026
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from core.config import PROPHET_PARAMS

logger = logging.getLogger(__name__)


class ProphetBaselineEngine:
    """
    Mesin Tahap 1: Pelatihan & prediksi baseline Prophet.

    Mengimplementasikan dekomposisi tren + musiman aditif/multiplikatif
    sesuai proposal skripsi Bagian 3.3.1.
    """

    def __init__(
        self,
        seasonality_mode: str = "additive",
        prophet_params:   Dict | None = None,
    ):
        _params = {**PROPHET_PARAMS, "seasonality_mode": seasonality_mode}
        if prophet_params:
            _params.update(prophet_params)
        self._prophet_params: Dict = _params
        self.model:      Optional[Prophet]      = None
        self._train_df:  Optional[pd.DataFrame] = None
        self._is_fitted: bool                   = False

    def fit(self, train_df: pd.DataFrame) -> "ProphetBaselineEngine":
        """
        Latih model Prophet pada data training.

        ── METODE PROPHET (TAHAP 1) ──────────────────────────────────
        Prophet melakukan dekomposisi aditif:
          ŷ(t) = g(t) + s(t)    [mode aditif: amplitudo musiman konstan]
        atau:
          ŷ(t) = g(t) · s(t)   [mode multiplikatif: tumbuh proporsional]

        Parameter utama:
          yearly_seasonality=True  : Musiman tahunan
          weekly_seasonality=True  : Musiman mingguan
          changepoint_prior_scale  : Fleksibilitas tren (0.05 = konservatif)
        """
        import logging as _log
        _log.getLogger("prophet").setLevel(_log.WARNING)
        _log.getLogger("cmdstanpy").setLevel(_log.WARNING)

        self._train_df = train_df[["ds", "y"]].dropna().reset_index(drop=True)
        self.model     = Prophet(**self._prophet_params)
        self.model.fit(self._train_df)
        self._is_fitted = True

        logger.info(
            "ProphetBaselineEngine: %d periode, mode=%s",
            len(self._train_df),
            self._prophet_params.get("seasonality_mode"),
        )
        return self

    def predict_train(self) -> np.ndarray:
        """Prediksi ŷ_Prophet pada periode training (untuk ekstraksi residual)."""
        self._require_fitted()
        return self.model.predict(self._train_df[["ds"]])["yhat"].values.clip(min=0)

    def predict(self, period_df: pd.DataFrame) -> np.ndarray:
        """Prediksi ŷ_Prophet pada periode arbitrer."""
        self._require_fitted()
        return self.model.predict(period_df[["ds"]])["yhat"].values.clip(min=0)

    def extract_train_residuals(self) -> np.ndarray:
        """
        Hitung residual training:
            residual(t) = y_aktual(t) − ŷ_Prophet(t)

        Residual inilah yang menjadi TARGET pelatihan LightGBM Tahap 2.
        """
        self._require_fitted()
        residuals = self._train_df["y"].values - self.predict_train()
        logger.debug("Residual: mean=%.2f std=%.2f", residuals.mean(), residuals.std())
        return residuals

    def get_components(self, full_df: pd.DataFrame) -> pd.DataFrame:
        """Kembalikan komponen dekomposisi Prophet."""
        self._require_fitted()
        return self.model.predict(full_df[["ds"]])[
            ["ds", "trend", "yhat", "yhat_lower", "yhat_upper"]
        ]

    @property
    def train_df(self):
        return self._train_df

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("ProphetBaselineEngine belum dilatih. Panggil .fit() dulu.")
