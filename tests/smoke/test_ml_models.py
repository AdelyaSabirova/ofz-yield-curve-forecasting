from __future__ import annotations

import numpy as np
import pandas as pd

from ofz_curve.ml_models import (
    forecast_factors_catboost,
    forecast_factors_random_forest,
)


def _make_synthetic_factors(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-06", periods=n)
    beta0 = 12.0 + np.cumsum(rng.normal(0.0, 0.08, size=n))
    beta1 = -3.0 + 0.7 * np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0.0, 0.05, size=n)
    beta2 = 1.5 + 0.4 * np.cos(np.linspace(0, 3 * np.pi, n)) + rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame({"beta0": beta0, "beta1": beta1, "beta2": beta2}, index=dates)


def test_random_forest_and_catboost_walk_forward_zero_future_leakage():
    base_df = _make_synthetic_factors(90)
    shocked_df = base_df.copy()
    # Corrupt future observations strictly after forecast origin index 75
    pivot_date = shocked_df.index[75]
    shocked_df.loc[shocked_df.index > pivot_date, ["beta0", "beta1", "beta2"]] += 500.0

    rf_base, _ = forecast_factors_random_forest(
        base_df,
        horizons=[5],
        lags=3,
        oos_step=1,
        oos_max_points=20,
        n_estimators=40,
        retrain_every=1,
        min_train_rows=25,
    )
    rf_shocked, _ = forecast_factors_random_forest(
        shocked_df,
        horizons=[5],
        lags=3,
        oos_step=1,
        oos_max_points=20,
        n_estimators=40,
        retrain_every=1,
        min_train_rows=25,
    )

    assert not rf_base.empty
    at_pivot_base = rf_base[rf_base["forecast_made_on_date"] == pivot_date].sort_values("factor")
    at_pivot_shock = rf_shocked[rf_shocked["forecast_made_on_date"] == pivot_date].sort_values("factor")
    assert len(at_pivot_base) == 3
    np.testing.assert_allclose(
        at_pivot_base["forecast_value_rf"].to_numpy(),
        at_pivot_shock["forecast_value_rf"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    cb_base, _ = forecast_factors_catboost(
        base_df,
        horizons=[5],
        lags=3,
        oos_step=2,
        oos_max_points=12,
        cb_iterations=50,
        retrain_every=2,
        min_train_rows=25,
    )
    assert not cb_base.empty
    assert {"forecast_value_catboost", "actual_value", "factor", "horizon_days"} <= set(cb_base.columns)
