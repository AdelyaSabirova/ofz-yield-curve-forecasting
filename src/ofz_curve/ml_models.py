"""
Machine learning models (Random Forest & CatBoost) for strict walk-forward
forecasting of Nelson-Siegel yield curve factors without data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from .config import FORECAST_HORIZONS_DAYS

try:
    from catboost import CatBoostRegressor, Pool

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def prepare_ml_features_for_factors(
    factors_df: pd.DataFrame,
    oil_series_full: pd.Series | None = None,
    usd_series_full: pd.Series | None = None,
    lags: int = 5,
    include_external: bool = True,
    dropna_targets: bool = True,
    return_separate: bool = False,
):
    """
    Build lagged factor and macro features strictly available at time t,
    paired with direct h-step ahead targets y_{t + h}.
    """
    features_df = pd.DataFrame(index=factors_df.index)

    for factor in ["beta0", "beta1", "beta2"]:
        features_df[f"{factor}_current"] = factors_df[factor]
        for lag in range(1, lags + 1):
            features_df[f"{factor}_lag_{lag}"] = factors_df[factor].shift(lag)

    if include_external and oil_series_full is not None and usd_series_full is not None:
        oil_aligned = oil_series_full.reindex(factors_df.index)
        usd_aligned = usd_series_full.reindex(factors_df.index)

        for lag in range(0, lags + 1):
            if lag == 0:
                features_df["ln_oil_current"] = oil_aligned
                features_df["ln_usd_current"] = usd_aligned
            else:
                features_df[f"ln_oil_lag_{lag}"] = oil_aligned.shift(lag)
                features_df[f"ln_usd_lag_{lag}"] = usd_aligned.shift(lag)

        features_df["ln_oil_diff"] = oil_aligned.diff()
        features_df["ln_usd_diff"] = usd_aligned.diff()

    targets_df = pd.DataFrame(index=factors_df.index)
    for horizon in FORECAST_HORIZONS_DAYS:
        for factor in ["beta0", "beta1", "beta2"]:
            targets_df[f"{factor}_h_{horizon}"] = factors_df[factor].shift(-horizon)

    if dropna_targets and not return_separate:
        return pd.concat([features_df, targets_df], axis=1).dropna()

    features_df_clean = features_df.dropna()
    targets_df_aligned = targets_df.loc[features_df_clean.index]

    if return_separate:
        return features_df_clean, targets_df_aligned

    return pd.concat([features_df_clean, targets_df_aligned], axis=1)


def _walk_forward_ml_forecast(
    factors_df: pd.DataFrame,
    oil_series_full: pd.Series | None,
    usd_series_full: pd.Series | None,
    horizons: list[int],
    lags: int,
    oos_step: int,
    oos_max_points: int | None,
    retrain_every: int,
    model_builder,
    forecast_col: str,
    desc: str,
    min_train_rows: int = 30,
):
    """
    Expanding-window walk-forward loop with strict horizon purging:
    at forecast index i, only training rows j <= i - horizon have realized
    h-step targets y_{j + h} observable on or before forecast_date[i].
    """
    include_ext = oil_series_full is not None and usd_series_full is not None
    features_df, targets_df = prepare_ml_features_for_factors(
        factors_df,
        oil_series_full,
        usd_series_full,
        lags=lags,
        include_external=include_ext,
        dropna_targets=False,
        return_separate=True,
    )
    for horizon in horizons:
        for factor in ["beta0", "beta1", "beta2"]:
            col = f"{factor}_h_{horizon}"
            if col not in targets_df.columns:
                targets_df[col] = factors_df[factor].shift(-horizon).reindex(features_df.index)

    if features_df.empty:
        return pd.DataFrame(), {}

    forecasts_list = []
    max_h = max(horizons) if horizons else 1
    min_start = max(min_train_rows + max_h, len(features_df) // 5)
    if oos_max_points is None:
        start_idx = min_start
    else:
        start_idx = max(min_start, len(features_df) - oos_max_points)

    model_cache = {}

    for i in tqdm(range(start_idx, len(features_df), max(1, oos_step)), desc=desc, disable=True):
        current_features = features_df.iloc[i : i + 1]
        if current_features.empty or current_features.isna().any(axis=None):
            continue

        forecast_date = features_df.index[i]
        try:
            forecast_pos = factors_df.index.get_loc(forecast_date)
        except KeyError:
            continue

        for factor in ["beta0", "beta1", "beta2"]:
            for horizon in horizons:
                target_col = f"{factor}_h_{horizon}"
                # Strict purge: row j in features_df uses target at j + horizon,
                # so j + horizon <= i  <=>  j <= i - horizon  <=>  slice [: i - horizon + 1]
                purge_end = i - horizon + 1
                if purge_end < min_train_rows:
                    continue

                X_hist = features_df.iloc[:purge_end]
                y_hist = targets_df[target_col].iloc[:purge_end]
                valid_mask = y_hist.notna()
                X_train = X_hist[valid_mask]
                y_train = y_hist[valid_mask]

                if len(X_train) < min_train_rows:
                    continue

                cache_key = (factor, horizon)
                need_retrain = (cache_key not in model_cache) or (
                    (i - start_idx) % max(1, retrain_every) == 0
                )
                if need_retrain:
                    model = model_builder(X_train, y_train)
                    model_cache[cache_key] = model
                else:
                    model = model_cache[cache_key]

                pred_val = float(model.predict(current_features)[0])

                target_pos = forecast_pos + horizon
                if target_pos >= len(factors_df):
                    continue

                target_date = factors_df.index[target_pos]
                actual_value = float(factors_df.iloc[target_pos][factor])

                forecasts_list.append(
                    {
                        "forecast_made_on_date": forecast_date,
                        "forecast_target_date": target_date,
                        "horizon_days": horizon,
                        "factor": factor,
                        forecast_col: pred_val,
                        "actual_value": actual_value,
                    }
                )

    forecasts_df = pd.DataFrame(forecasts_list)
    if not forecasts_df.empty:
        forecasts_df.dropna(subset=[forecast_col], inplace=True)
    return forecasts_df, model_cache


def train_random_forest_factor_models(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    horizons=FORECAST_HORIZONS_DAYS,
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: int | None = 8,
    random_state: int = 42,
):
    """Train Random Forest factor models with chronological train/test split."""
    models = {}
    for factor in ["beta0", "beta1", "beta2"]:
        models[factor] = {}
        for horizon in horizons:
            target_col = f"{factor}_h_{horizon}"
            if target_col not in targets_df.columns:
                continue
            valid = targets_df[target_col].notna()
            X = features_df[valid]
            y = targets_df.loc[valid, target_col]
            if len(X) < 30:
                continue
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            models[factor][horizon] = {
                "model": rf,
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                "mae": float(mean_absolute_error(y_test, preds)),
                "feature_importance": dict(zip(X.columns, rf.feature_importances_)),
            }
    return models


def forecast_factors_random_forest(
    factors_df: pd.DataFrame,
    oil_series_full: pd.Series | None = None,
    usd_series_full: pd.Series | None = None,
    horizons=FORECAST_HORIZONS_DAYS,
    lags: int = 5,
    oos_step: int = 3,
    oos_max_points: int | None = 300,
    n_estimators: int = 200,
    max_depth: int | None = 8,
    retrain_every: int = 10,
    min_train_rows: int = 30,
):
    """Strict walk-forward factor forecasting using RandomForestRegressor."""

    def _build_rf(X_train: pd.DataFrame, y_train: pd.Series):
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        return rf

    return _walk_forward_ml_forecast(
        factors_df=factors_df,
        oil_series_full=oil_series_full,
        usd_series_full=usd_series_full,
        horizons=list(horizons),
        lags=lags,
        oos_step=oos_step,
        oos_max_points=oos_max_points,
        retrain_every=retrain_every,
        model_builder=_build_rf,
        forecast_col="forecast_value_rf",
        desc="RandomForest OOS Forecasting",
        min_train_rows=min_train_rows,
    )


def train_catboost_factor_models(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    horizons=FORECAST_HORIZONS_DAYS,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train CatBoost factor models with chronological train/test split."""
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost is not installed.")

    models = {}
    for factor in ["beta0", "beta1", "beta2"]:
        models[factor] = {}
        for horizon in horizons:
            target_col = f"{factor}_h_{horizon}"
            if target_col not in targets_df.columns:
                continue
            valid = targets_df[target_col].notna()
            X = features_df[valid]
            y = targets_df.loc[valid, target_col]
            if len(X) < 30:
                continue
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            cb = CatBoostRegressor(
                iterations=400,
                learning_rate=0.06,
                depth=6,
                loss_function="RMSE",
                random_seed=random_state,
                verbose=False,
            )
            cb.fit(Pool(X_train, y_train))
            preds = cb.predict(Pool(X_test))
            models[factor][horizon] = {
                "model": cb,
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                "mae": float(mean_absolute_error(y_test, preds)),
                "feature_importance": dict(zip(X.columns, cb.get_feature_importance())),
            }
    return models


def forecast_factors_catboost(
    factors_df: pd.DataFrame,
    oil_series_full: pd.Series | None = None,
    usd_series_full: pd.Series | None = None,
    horizons=FORECAST_HORIZONS_DAYS,
    lags: int = 5,
    oos_step: int = 3,
    oos_max_points: int | None = 300,
    cb_iterations: int = 300,
    cb_learning_rate: float = 0.06,
    retrain_every: int = 10,
    min_train_rows: int = 30,
):
    """Strict walk-forward factor forecasting using CatBoostRegressor."""
    if not CATBOOST_AVAILABLE:
        return pd.DataFrame(), {}

    def _build_cb(X_train: pd.DataFrame, y_train: pd.Series):
        cb = CatBoostRegressor(
            iterations=cb_iterations,
            learning_rate=cb_learning_rate,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )
        cb.fit(Pool(X_train, y_train))
        return cb

    return _walk_forward_ml_forecast(
        factors_df=factors_df,
        oil_series_full=oil_series_full,
        usd_series_full=usd_series_full,
        horizons=list(horizons),
        lags=lags,
        oos_step=oos_step,
        oos_max_points=oos_max_points,
        retrain_every=retrain_every,
        model_builder=_build_cb,
        forecast_col="forecast_value_catboost",
        desc="CatBoost OOS Forecasting",
        min_train_rows=min_train_rows,
    )


def compare_ml_vs_econometric(
    ar_forecasts_df: pd.DataFrame,
    catboost_forecasts_df: pd.DataFrame,
    factor: str = "beta0",
    horizon: int = 21,
):
    """Compare AR(1) and CatBoost factor forecast errors on identical target dates."""
    if ar_forecasts_df.empty or catboost_forecasts_df.empty:
        return pd.DataFrame()

    ar_data = ar_forecasts_df[
        (ar_forecasts_df["factor"] == factor) & (ar_forecasts_df["horizon_days"] == horizon)
    ].copy()
    cb_data = catboost_forecasts_df[
        (catboost_forecasts_df["factor"] == factor) & (catboost_forecasts_df["horizon_days"] == horizon)
    ].copy()

    if ar_data.empty or cb_data.empty:
        return pd.DataFrame()

    comparison_df = pd.merge(
        ar_data[["forecast_target_date", "forecast_value_ar1", "actual_value"]],
        cb_data[["forecast_target_date", "forecast_value_catboost"]],
        on="forecast_target_date",
        how="inner",
    )
    if comparison_df.empty:
        return pd.DataFrame()

    rmse_ar1 = float(
        np.sqrt(mean_squared_error(comparison_df["actual_value"], comparison_df["forecast_value_ar1"]))
    )
    rmse_cb = float(
        np.sqrt(mean_squared_error(comparison_df["actual_value"], comparison_df["forecast_value_catboost"]))
    )
    mae_ar1 = float(
        mean_absolute_error(comparison_df["actual_value"], comparison_df["forecast_value_ar1"])
    )
    mae_cb = float(
        mean_absolute_error(comparison_df["actual_value"], comparison_df["forecast_value_catboost"])
    )

    return pd.DataFrame(
        {
            "Model": ["AR(1)", "CatBoost"],
            "RMSE": [rmse_ar1, rmse_cb],
            "MAE": [mae_ar1, mae_cb],
            "N_observations": [len(comparison_df)] * 2,
        }
    )


def compare_ml_vs_econometric_all(
    ar_forecasts_df: pd.DataFrame,
    catboost_forecasts_df: pd.DataFrame,
    factors: list[str] | None = None,
    horizons: list[int] | None = None,
):
    """Batch comparison across all overlapping factors and horizons."""
    if ar_forecasts_df.empty or catboost_forecasts_df.empty:
        return pd.DataFrame()

    factors_list = factors or sorted(
        set(ar_forecasts_df.get("factor", [])).intersection(catboost_forecasts_df.get("factor", []))
    )
    horizons_list = horizons or sorted(
        set(ar_forecasts_df.get("horizon_days", [])).intersection(
            catboost_forecasts_df.get("horizon_days", [])
        )
    )

    results_all = []
    for factor in factors_list:
        for horizon in horizons_list:
            res = compare_ml_vs_econometric(
                ar_forecasts_df, catboost_forecasts_df, factor=factor, horizon=horizon
            )
            if not res.empty:
                res = res.copy()
                res["factor"] = factor
                res["horizon_days"] = horizon
                results_all.append(res)

    if not results_all:
        return pd.DataFrame()
    return pd.concat(results_all, ignore_index=True)
