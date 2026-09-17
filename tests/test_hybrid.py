"""
tests/test_hybrid.py
====================
Unit test untuk HybridForecastingEngine pada versi MySQL.
Memvalidasi pipeline dua tahap Prophet + LightGBM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def weekly_series() -> pd.DataFrame:
    dates = pd.date_range(start="2021-01-04", periods=156, freq="W-MON")
    np.random.seed(42)
    t = np.arange(156)
    trend = 1000 + 2 * t
    seasonal = 200 * np.sin(2 * np.pi * t / 52)
    noise = np.random.normal(0, 50, 156)
    y = np.maximum(0, trend + seasonal + noise)
    return pd.DataFrame({"ds": dates, "y": y})


@pytest.fixture
def monthly_series() -> pd.DataFrame:
    dates = pd.date_range(start="2021-01-01", periods=36, freq="ME")
    np.random.seed(42)
    t = np.arange(36)
    y = np.maximum(0, 5000 + 50 * t + 1000 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 200, 36))
    return pd.DataFrame({"ds": dates, "y": y})


@pytest.fixture
def engine_weekly(weekly_series):
    from core.engine import HybridForecastingEngine
    engine = HybridForecastingEngine(seasonality_mode="additive", test_size=0.20)
    engine.fit(weekly_series, freq="W")
    return engine


class TestHybridEngineInit:
    def test_default_init(self):
        from core.engine import HybridForecastingEngine
        engine = HybridForecastingEngine()
        assert engine.seasonality_mode == "additive"
        assert engine.test_size == 0.20
        assert not engine._is_fitted

    def test_require_fitted_raises(self):
        from core.engine import HybridForecastingEngine
        engine = HybridForecastingEngine()
        with pytest.raises(RuntimeError, match="belum dilatih"):
            engine.predict_test()


class TestTrainTestSplit:
    def test_split_ratio(self, weekly_series):
        from core.engine import HybridForecastingEngine
        engine = HybridForecastingEngine(test_size=0.20)
        train, test = engine.train_test_split(weekly_series)
        n_total = len(weekly_series)
        assert len(train) + len(test) == n_total
        assert abs(len(test) / n_total - 0.20) < 0.02

    def test_split_chronological(self, weekly_series):
        from core.engine import HybridForecastingEngine
        engine = HybridForecastingEngine(test_size=0.20)
        train, test = engine.train_test_split(weekly_series)
        assert train["ds"].max() < test["ds"].min()


class TestFeatureEngineering:
    def test_exogenous_features(self, weekly_series):
        from core.engine import HybridForecastingEngine
        engine = HybridForecastingEngine()
        exog = engine.extract_exogenous_features(weekly_series["ds"])
        assert {"month", "weekofyear", "quarter", "day", "dayofweek"}.issubset(exog.columns)

    def test_feature_matrix_lags_and_rolling(self, weekly_series):
        from core.engine import HybridForecastingEngine
        engine = HybridForecastingEngine(lags=[1, 2], rolling_windows=[4])
        feat = engine.build_feature_matrix(weekly_series, weekly_series["y"])
        assert "lag_1" in feat.columns
        assert "roll_mean_4" in feat.columns


class TestFitAndPredict:
    def test_predict_test_columns(self, engine_weekly):
        test_results = engine_weekly.predict_test()
        required = {"ds", "y", "y_prophet", "y_residual_lgbm", "y_hybrid"}
        assert required.issubset(test_results.columns)

    def test_predict_test_non_negative(self, engine_weekly):
        test_results = engine_weekly.predict_test()
        assert (test_results["y_hybrid"] >= 0).all()

    def test_forecast_future_length(self, engine_weekly, weekly_series):
        horizon = 8
        future = engine_weekly.forecast_future(weekly_series, horizon=horizon)
        assert len(future) == horizon
        assert (future["y_hybrid"] >= 0).all()

    def test_monthly_fit_and_forecast(self, monthly_series):
        from core.engine import HybridForecastingEngine
        engine = HybridForecastingEngine()
        engine.fit(monthly_series, freq="M")
        assert engine._is_fitted
        future = engine.forecast_future(monthly_series, horizon=6)
        assert len(future) == 6
        assert (future["y_hybrid"] >= 0).all()
