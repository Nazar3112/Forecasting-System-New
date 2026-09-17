"""
core/evaluator.py
=================
Kelas ModelEvaluator: kalkulasi metrik akurasi peramalan
(MAE, MAPE, RMSE), perbandingan model tunggal vs hibrida,
dan interpretasi kualitas hasil peramalan.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from core.config import MAPE_THRESHOLDS

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluasi galat prediksi menggunakan tiga metrik standar ilmiah:
      1. MAE  : (1/n) * Σ |y_t - ŷ_t|
      2. MAPE : (1/n) * Σ |y_t - ŷ_t| / max(|y_t|, ε) * 100%
      3. RMSE : sqrt((1/n) * Σ (y_t - ŷ_t)²)
    """

    @staticmethod
    def calculate_mae(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def calculate_mape(
        y_true:  np.ndarray | pd.Series,
        y_pred:  np.ndarray | pd.Series,
        epsilon: float = 1e-6,
    ) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) == 0:
            return 0.0
        denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
        return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

    @staticmethod
    def calculate_rmse(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) == 0:
            return 0.0
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def evaluate(
        self,
        y_true:     np.ndarray | pd.Series,
        y_pred:     np.ndarray | pd.Series,
        model_name: str = "Model",
    ) -> Dict[str, float | str]:
        mae  = self.calculate_mae(y_true, y_pred)
        mape = self.calculate_mape(y_true, y_pred)
        rmse = self.calculate_rmse(y_true, y_pred)
        return {
            "model":   model_name,
            "MAE":     round(mae, 4),
            "MAPE_%":  round(mape, 4),
            "RMSE":    round(rmse, 4),
        }

    def compare_models(
        self,
        y_true:       np.ndarray | pd.Series,
        prophet_pred: np.ndarray | pd.Series,
        hybrid_pred:  np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        rows = [
            self.evaluate(y_true, prophet_pred, "Prophet (Tunggal)"),
            self.evaluate(y_true, hybrid_pred,  "Hibrida Prophet + LightGBM"),
        ]
        comp_df = pd.DataFrame(rows).set_index("model").reset_index()

        for metric in ["MAE", "MAPE_%", "RMSE"]:
            col_delta = f"Δ {metric} (%)"
            delta_vals = []
            for _, row in comp_df.iterrows():
                if row["model"] == "Hibrida Prophet + LightGBM":
                    prophet_row = comp_df[comp_df["model"] == "Prophet (Tunggal)"]
                    if not prophet_row.empty:
                        v_p = float(prophet_row[metric].values[0])
                        v_h = float(row[metric])
                        if v_p > 0:
                            d = (v_p - v_h) / v_p * 100
                            delta_vals.append(f"+{d:.2f}%" if d >= 0 else f"{d:.2f}%")
                        else:
                            delta_vals.append("N/A")
                    else:
                        delta_vals.append("N/A")
                else:
                    delta_vals.append("-")
            comp_df[col_delta] = delta_vals

        return comp_df

    def interpret_mape(self, mape_val: float) -> Tuple[str, str]:
        emoji_map = {
            "Sangat Baik": "🟢",
            "Baik":        "🔵",
            "Layak":       "🟡",
            "Kurang":      "🔴",
        }
        for label, (lo, hi) in MAPE_THRESHOLDS.items():
            if lo <= mape_val < hi:
                return label, emoji_map.get(label, "⚪")
        return "Kurang", "🔴"

    def build_evaluation_report(
        self,
        test_df:     pd.DataFrame,
        prophet_col: str = "y_prophet",
        hybrid_col:  str = "y_hybrid",
        actual_col:  str = "y",
    ) -> Dict:
        y_true    = test_df[actual_col].values
        y_prophet = test_df[prophet_col].values
        y_hybrid  = test_df[hybrid_col].values

        metrics_prophet = self.evaluate(y_true, y_prophet, "Prophet (Tunggal)")
        metrics_hybrid  = self.evaluate(y_true, y_hybrid,  "Hibrida Prophet + LightGBM")

        mape_val     = metrics_hybrid["MAPE_%"]
        label, emoji = self.interpret_mape(mape_val)
        comp_table   = self.compare_models(y_true, y_prophet, y_hybrid)

        return {
            "metrics_prophet":     metrics_prophet,
            "metrics_hybrid":      metrics_hybrid,
            "comparison_table":    comp_table,
            "mape_interpretation": label,
            "mape_emoji":          emoji,
            "n_test_points":       len(y_true),
        }
