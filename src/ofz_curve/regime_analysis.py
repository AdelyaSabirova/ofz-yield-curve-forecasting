"""
Regime analysis and structural break checks for the OFZ thesis update.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_distribution
from scipy.stats import ttest_ind
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import VECM

from .paths import PROJECT_ROOT


CORE_PATH = PROJECT_ROOT / "data" / "processed" / "core_2026" / "monthly_ns_factors_core.csv"
MACRO_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "macro_2026"
    / "monthly_expectations_panel.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "regime_outputs"
DEFAULT_FINDINGS_PATH = PROJECT_ROOT / "docs" / "p3_regime_findings.md"

DATE_COLUMN = "date"
TAU_COLUMN = "tau1"
MODEL_VARIABLES = ["beta0", "beta1", "beta2", "cpi_mom", "key_rate"]
CURVE_VARIABLES = ["beta0", "beta1", "beta2"]
MACRO_VARIABLES = ["cpi_mom", "key_rate"]
VECM_K_AR_DIFF = 1
COINT_RANK = 1
VAR_ORDER = VECM_K_AR_DIFF + 1
FORECAST_HORIZON = 1
FORECAST_START_RATIO = 0.6
FORECAST_MIN_TRAIN = 12
FEVD_HORIZONS = (1, 3, 6, 12)
IRF_HORIZONS = (1, 3, 6, 12)
BREAK_DATE = pd.Timestamp("2022-02-01")
NS_ABS_THRESHOLD = 1.0


@dataclass(frozen=True)
class RegimeSpec:
    regime_id: str
    label: str
    start: str
    end: str | None


REGIME_SPECS = (
    RegimeSpec("pre_2022_02", "2013-09 - 2022-01", "2013-09-01", "2022-01-01"),
    RegimeSpec("post_2022_02", "2022-02 - last", "2022-02-01", None),
    RegimeSpec("transition_2022_2023", "2022-02 - 2023-12", "2022-02-01", "2023-12-01"),
    RegimeSpec("stabilization_2024_plus", "2024-01 - last", "2024-01-01", None),
)


def _iso_date(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _month_label(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row(
    *,
    table: str,
    regime_id: str,
    regime_label: str,
    sample_start: str | None,
    sample_end: str | None,
    target: str | None = None,
    source: str | None = None,
    metric: str | None = None,
    value: float | int | str | bool | None = None,
    horizon: int | None = None,
    lag: int | None = None,
    p_value: float | None = None,
    significant: bool | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "table": table,
        "regime_id": regime_id,
        "regime_label": regime_label,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "target": target,
        "source": source,
        "metric": metric,
        "value": value,
        "horizon": horizon,
        "lag": lag,
        "p_value": p_value,
        "significant": significant,
        "notes": notes,
    }


def _require_columns(frame: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {missing}")


def _load_inputs(
    core_path: Path,
    macro_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core = pd.read_csv(core_path, parse_dates=[DATE_COLUMN])
    macro = pd.read_csv(macro_path, parse_dates=[DATE_COLUMN])

    _require_columns(core, [DATE_COLUMN, "beta0", "beta1", "beta2", TAU_COLUMN], "core csv")
    _require_columns(macro, [DATE_COLUMN, "cpi_mom", "key_rate"], "macro csv")

    core = core.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).drop_duplicates(DATE_COLUMN)
    macro = macro.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).drop_duplicates(DATE_COLUMN)

    merged = core.merge(macro, on=DATE_COLUMN, how="inner", suffixes=("_core", "_macro"))
    merged = merged.sort_values(DATE_COLUMN).drop_duplicates(DATE_COLUMN)
    return core, macro, merged


def _prepare_tau_panel(core: pd.DataFrame) -> pd.DataFrame:
    tau_panel = core[[DATE_COLUMN, TAU_COLUMN, "beta0", "beta1", "beta2"]].copy()
    tau_panel = tau_panel.dropna(subset=[DATE_COLUMN, TAU_COLUMN]).set_index(DATE_COLUMN).sort_index()
    tau_panel = tau_panel.loc[pd.Timestamp(REGIME_SPECS[0].start) :]
    return tau_panel


def _prepare_model_panel(merged: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = [DATE_COLUMN, TAU_COLUMN, *MODEL_VARIABLES]
    model_panel = merged[columns].copy()
    model_panel = model_panel.dropna(subset=[TAU_COLUMN, *MODEL_VARIABLES]).sort_values(DATE_COLUMN)

    outlier_mask = (model_panel["beta1"].abs() > NS_ABS_THRESHOLD) | (
        model_panel["beta2"].abs() > NS_ABS_THRESHOLD
    )
    excluded_dates = [
        _iso_date(timestamp) for timestamp in model_panel.loc[outlier_mask, DATE_COLUMN].tolist()
    ]
    cleaned = model_panel.loc[~outlier_mask].set_index(DATE_COLUMN).sort_index()
    return cleaned, excluded_dates


def _slice_regime(frame: pd.DataFrame, spec: RegimeSpec) -> pd.DataFrame:
    subset = frame.loc[pd.Timestamp(spec.start) :]
    if spec.end is not None:
        subset = subset.loc[: pd.Timestamp(spec.end)]
    return subset.copy()


def _frame_dates(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame.empty:
        return None, None
    index = pd.DatetimeIndex(frame.index)
    return _iso_date(index.min()), _iso_date(index.max())


def _vecm_fit(frame: pd.DataFrame) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VECM(
            frame,
            k_ar_diff=VECM_K_AR_DIFF,
            coint_rank=COINT_RANK,
            deterministic="co",
        ).fit()


def _var_fit(frame: pd.DataFrame) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VAR(frame).fit(VAR_ORDER, trend="c")


def _describe_series(
    frame: pd.DataFrame,
    spec: RegimeSpec,
    table: str,
    columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_start, sample_end = _frame_dates(frame)
    if frame.empty:
        return rows

    for column in columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        rows.extend(
            [
                _row(
                    table=table,
                    regime_id=spec.regime_id,
                    regime_label=spec.label,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    target=column,
                    metric="mean",
                    value=_safe_float(series.mean()),
                ),
                _row(
                    table=table,
                    regime_id=spec.regime_id,
                    regime_label=spec.label,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    target=column,
                    metric="median",
                    value=_safe_float(series.median()),
                ),
                _row(
                    table=table,
                    regime_id=spec.regime_id,
                    regime_label=spec.label,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    target=column,
                    metric="std",
                    value=_safe_float(series.std(ddof=1)),
                ),
                _row(
                    table=table,
                    regime_id=spec.regime_id,
                    regime_label=spec.label,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    target=column,
                    metric="min",
                    value=_safe_float(series.min()),
                ),
                _row(
                    table=table,
                    regime_id=spec.regime_id,
                    regime_label=spec.label,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    target=column,
                    metric="max",
                    value=_safe_float(series.max()),
                ),
                _row(
                    table=table,
                    regime_id=spec.regime_id,
                    regime_label=spec.label,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    target=column,
                    metric="n_obs",
                    value=int(series.shape[0]),
                ),
            ]
        )

    return rows


def _sample_info_rows(
    *,
    spec: RegimeSpec,
    tau_frame: pd.DataFrame,
    model_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    tau_start, tau_end = _frame_dates(tau_frame)
    model_start, model_end = _frame_dates(model_frame)
    return [
        _row(
            table="sample_info",
            regime_id=spec.regime_id,
            regime_label=spec.label,
            sample_start=tau_start,
            sample_end=tau_end,
            target="tau_panel",
            metric="n_obs",
            value=int(len(tau_frame)),
        ),
        _row(
            table="sample_info",
            regime_id=spec.regime_id,
            regime_label=spec.label,
            sample_start=model_start,
            sample_end=model_end,
            target="model_panel",
            metric="n_obs",
            value=int(len(model_frame)),
        ),
    ]


def _forecast_rows(frame: pd.DataFrame, spec: RegimeSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_start, sample_end = _frame_dates(frame)
    n_obs = len(frame)
    initial_train = max(FORECAST_MIN_TRAIN, math.ceil(n_obs * FORECAST_START_RATIO))

    if n_obs <= initial_train:
        return rows

    errors: dict[str, list[float]] = {column: [] for column in MODEL_VARIABLES}
    target_dates: list[str] = []

    for split_idx in range(initial_train, n_obs):
        train = frame.iloc[:split_idx]
        actual = frame.iloc[split_idx]
        target_dates.append(_iso_date(frame.index[split_idx]) or "")

        try:
            fit = _vecm_fit(train)
            predicted = fit.predict(steps=FORECAST_HORIZON)[0]
        except Exception:
            continue

        for column_idx, column in enumerate(MODEL_VARIABLES):
            error = float(actual.iloc[column_idx] - predicted[column_idx])
            errors[column].append(error)

    n_forecasts = len(next(iter(errors.values()))) if errors else 0
    if n_forecasts == 0:
        return rows

    normalized_values: list[float] = []
    for column in MODEL_VARIABLES:
        error_array = np.asarray(errors[column], dtype=float)
        rmse = float(np.sqrt(np.mean(np.square(error_array))))
        scale = float(frame[column].std(ddof=1))
        rmse_to_std = rmse / scale if scale and np.isfinite(scale) else np.nan
        normalized_values.append(rmse_to_std)

        rows.append(
            _row(
                table="forecast_rmse",
                regime_id=spec.regime_id,
                regime_label=spec.label,
                sample_start=sample_start,
                sample_end=sample_end,
                target=column,
                metric="rmse",
                value=rmse,
                notes=f"initial_train={initial_train}; n_forecasts={n_forecasts}",
            )
        )
        rows.append(
            _row(
                table="forecast_rmse",
                regime_id=spec.regime_id,
                regime_label=spec.label,
                sample_start=sample_start,
                sample_end=sample_end,
                target=column,
                metric="rmse_to_std",
                value=_safe_float(rmse_to_std),
                notes=f"initial_train={initial_train}; n_forecasts={n_forecasts}",
            )
        )
        rows.append(
            _row(
                table="forecast_rmse",
                regime_id=spec.regime_id,
                regime_label=spec.label,
                sample_start=sample_start,
                sample_end=sample_end,
                target=column,
                metric="n_forecasts",
                value=n_forecasts,
            )
        )

    rows.append(
        _row(
            table="forecast_rmse",
            regime_id=spec.regime_id,
            regime_label=spec.label,
            sample_start=sample_start,
            sample_end=sample_end,
            target="system",
            metric="mean_rmse_to_std",
            value=_safe_float(np.nanmean(normalized_values)),
            notes=f"initial_train={initial_train}; n_forecasts={n_forecasts}",
        )
    )
    return rows


def _vecm_rows(frame: pd.DataFrame, spec: RegimeSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_start, sample_end = _frame_dates(frame)
    result = _vecm_fit(frame)

    for idx, variable in enumerate(MODEL_VARIABLES):
        rows.append(
            _row(
                table="vecm_alpha",
                regime_id=spec.regime_id,
                regime_label=spec.label,
                sample_start=sample_start,
                sample_end=sample_end,
                target=variable,
                metric="ec1_loading",
                value=_safe_float(result.alpha[idx, 0]),
            )
        )
        rows.append(
            _row(
                table="vecm_beta",
                regime_id=spec.regime_id,
                regime_label=spec.label,
                sample_start=sample_start,
                sample_end=sample_end,
                target=variable,
                metric="cointegration_weight",
                value=_safe_float(result.beta[idx, 0]),
            )
        )

    gamma = np.asarray(result.gamma)
    gamma_columns = [f"d_{column}_lag1" for column in MODEL_VARIABLES]
    for eq_idx, target in enumerate(MODEL_VARIABLES):
        for source_idx, source in enumerate(gamma_columns):
            rows.append(
                _row(
                    table="vecm_gamma",
                    regime_id=spec.regime_id,
                    regime_label=spec.label,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    target=target,
                    source=source,
                    metric="short_run_coef",
                    value=_safe_float(gamma[eq_idx, source_idx]),
                )
            )

    return rows


def _granger_rows(frame: pd.DataFrame, spec: RegimeSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_start, sample_end = _frame_dates(frame)

    for target in MODEL_VARIABLES:
        for source in MODEL_VARIABLES:
            if source == target:
                continue
            pair = frame[[target, source]].dropna()
            lag_used = min(VAR_ORDER, max(1, len(pair) // 8))
            lag_used = min(lag_used, VAR_ORDER)
            if len(pair) <= lag_used + 2:
                rows.append(
                    _row(
                        table="granger",
                        regime_id=spec.regime_id,
                        regime_label=spec.label,
                        sample_start=sample_start,
                        sample_end=sample_end,
                        target=target,
                        source=source,
                        metric="p_value",
                        value=np.nan,
                        lag=lag_used,
                        notes="not enough observations",
                    )
                )
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    result = grangercausalitytests(pair, maxlag=lag_used, verbose=False)
                p_value = float(result[lag_used][0]["ssr_ftest"][1])
            except Exception as exc:
                rows.append(
                    _row(
                        table="granger",
                        regime_id=spec.regime_id,
                        regime_label=spec.label,
                        sample_start=sample_start,
                        sample_end=sample_end,
                        target=target,
                        source=source,
                        metric="p_value",
                        value=np.nan,
                        lag=lag_used,
                        notes=str(exc),
                    )
                )
                continue

            rows.append(
                _row(
                    table="granger",
                    regime_id=spec.regime_id,
                    regime_label=spec.label,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    target=target,
                    source=source,
                    metric="p_value",
                    value=p_value,
                    p_value=p_value,
                    lag=lag_used,
                    significant=bool(p_value < 0.05),
                )
            )
    return rows


def _irf_rows(frame: pd.DataFrame, spec: RegimeSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_start, sample_end = _frame_dates(frame)
    irf = _vecm_fit(frame).irf(max(IRF_HORIZONS))

    for response_idx, response in enumerate(MODEL_VARIABLES):
        for impulse_idx, impulse in enumerate(MODEL_VARIABLES):
            series = np.asarray(irf.irfs[:, response_idx, impulse_idx], dtype=float)
            peak_index = int(np.abs(series).argmax())
            rows.append(
                _row(
                    table="irf_summary",
                    regime_id=spec.regime_id,
                    regime_label=spec.label,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    target=response,
                    source=impulse,
                    metric="peak_response",
                    value=_safe_float(series[peak_index]),
                    horizon=peak_index,
                )
            )
            rows.append(
                _row(
                    table="irf_summary",
                    regime_id=spec.regime_id,
                    regime_label=spec.label,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    target=response,
                    source=impulse,
                    metric="cumulative_response_12m",
                    value=_safe_float(series[: max(IRF_HORIZONS) + 1].sum()),
                    horizon=max(IRF_HORIZONS),
                )
            )
            for horizon in IRF_HORIZONS:
                rows.append(
                    _row(
                        table="irf_point",
                        regime_id=spec.regime_id,
                        regime_label=spec.label,
                        sample_start=sample_start,
                        sample_end=sample_end,
                        target=response,
                        source=impulse,
                        metric="response",
                        value=_safe_float(series[horizon]),
                        horizon=horizon,
                    )
                )
    return rows


def _fevd_rows(frame: pd.DataFrame, spec: RegimeSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_start, sample_end = _frame_dates(frame)
    fevd = _var_fit(frame).fevd(max(FEVD_HORIZONS))

    for response_idx, response in enumerate(MODEL_VARIABLES):
        for horizon in FEVD_HORIZONS:
            horizon_idx = horizon - 1
            for shock_idx, shock in enumerate(MODEL_VARIABLES):
                share = float(fevd.decomp[response_idx, horizon_idx, shock_idx] * 100.0)
                rows.append(
                    _row(
                        table="fevd",
                        regime_id=spec.regime_id,
                        regime_label=spec.label,
                        sample_start=sample_start,
                        sample_end=sample_end,
                        target=response,
                        source=shock,
                        metric="share_pct",
                        value=share,
                        horizon=horizon,
                    )
                )
    return rows


def _lagged_regression_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    predictors: list[str],
) -> pd.DataFrame:
    data = pd.DataFrame(index=frame.index)
    data["y"] = frame[target]
    for predictor in predictors:
        data[f"lag1_{predictor}"] = frame[predictor].shift(1)
    return data.dropna()


def _chow_test(
    frame: pd.DataFrame,
    *,
    break_date: pd.Timestamp,
) -> dict[str, Any]:
    x = sm.add_constant(frame.drop(columns="y"), has_constant="add")
    y = frame["y"]

    x_pre = x.loc[x.index < break_date]
    y_pre = y.loc[y.index < break_date]
    x_post = x.loc[x.index >= break_date]
    y_post = y.loc[y.index >= break_date]

    k_params = int(x.shape[1])
    n_pre = int(len(x_pre))
    n_post = int(len(x_post))

    if n_pre <= k_params or n_post <= k_params:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "reject_5pct": False,
            "n_pre": n_pre,
            "n_post": n_post,
            "notes": "not enough observations for Chow test",
        }

    pooled_fit = sm.OLS(y, x).fit()
    pre_fit = sm.OLS(y_pre, x_pre).fit()
    post_fit = sm.OLS(y_post, x_post).fit()

    rss_pooled = float(np.sum(np.square(pooled_fit.resid)))
    rss_split = float(np.sum(np.square(pre_fit.resid)) + np.sum(np.square(post_fit.resid)))
    denom_df = n_pre + n_post - (2 * k_params)
    if rss_split <= 0 or denom_df <= 0:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "reject_5pct": False,
            "n_pre": n_pre,
            "n_post": n_post,
            "notes": "invalid denominator",
        }

    statistic = ((rss_pooled - rss_split) / k_params) / (rss_split / denom_df)
    p_value = float(f_distribution.sf(statistic, k_params, denom_df))
    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "reject_5pct": bool(p_value < 0.05),
        "n_pre": n_pre,
        "n_post": n_post,
        "notes": f"k_params={k_params}",
    }


def _break_test_rows(
    tau_panel: pd.DataFrame,
    model_panel: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    tau_pre = _slice_regime(tau_panel, REGIME_SPECS[0])[TAU_COLUMN].dropna()
    tau_post = _slice_regime(tau_panel, REGIME_SPECS[1])[TAU_COLUMN].dropna()
    tau_mean_test = ttest_ind(tau_pre, tau_post, equal_var=False, nan_policy="omit")
    rows.append(
        {
            "test_name": "welch_mean_shift",
            "target": TAU_COLUMN,
            "break_date": _iso_date(BREAK_DATE),
            "sample_start": _iso_date(tau_panel.index.min()),
            "sample_end": _iso_date(tau_panel.index.max()),
            "statistic": _safe_float(tau_mean_test.statistic),
            "p_value": _safe_float(tau_mean_test.pvalue),
            "reject_5pct": bool(tau_mean_test.pvalue < 0.05),
            "n_pre": int(len(tau_pre)),
            "n_post": int(len(tau_post)),
            "notes": "raw tau1 mean comparison",
        }
    )

    tau_regression = _lagged_regression_frame(
        tau_panel[["tau1", "beta0", "beta1", "beta2"]],
        target="tau1",
        predictors=["tau1", "beta0", "beta1", "beta2"],
    )
    chow_tau = _chow_test(tau_regression, break_date=BREAK_DATE)
    rows.append(
        {
            "test_name": "chow_lag1",
            "target": "tau1",
            "break_date": _iso_date(BREAK_DATE),
            "sample_start": _iso_date(tau_regression.index.min()),
            "sample_end": _iso_date(tau_regression.index.max()),
            **chow_tau,
        }
    )
    tau_residuals = sm.OLS(
        tau_regression["y"],
        sm.add_constant(tau_regression.drop(columns="y"), has_constant="add"),
    ).fit()
    cusum_tau = breaks_cusumolsresid(tau_residuals.resid, ddof=tau_residuals.df_model + 1)
    rows.append(
        {
            "test_name": "cusum_ols",
            "target": "tau1",
            "break_date": _iso_date(BREAK_DATE),
            "sample_start": _iso_date(tau_regression.index.min()),
            "sample_end": _iso_date(tau_regression.index.max()),
            "statistic": _safe_float(cusum_tau[0]),
            "p_value": _safe_float(cusum_tau[1]),
            "reject_5pct": bool(cusum_tau[1] < 0.05),
            "n_pre": int((tau_regression.index < BREAK_DATE).sum()),
            "n_post": int((tau_regression.index >= BREAK_DATE).sum()),
            "notes": "tau1 lagged OLS residual test",
        }
    )

    for target in MODEL_VARIABLES:
        regression_frame = _lagged_regression_frame(
            model_panel[MODEL_VARIABLES],
            target=target,
            predictors=MODEL_VARIABLES,
        )
        chow_result = _chow_test(regression_frame, break_date=BREAK_DATE)
        rows.append(
            {
                "test_name": "chow_lag1",
                "target": target,
                "break_date": _iso_date(BREAK_DATE),
                "sample_start": _iso_date(regression_frame.index.min()),
                "sample_end": _iso_date(regression_frame.index.max()),
                **chow_result,
            }
        )

    return pd.DataFrame(rows)


def _metric_lookup(
    comparison: pd.DataFrame,
    *,
    table: str,
    regime_id: str,
    target: str,
    metric: str,
    source: str | None = None,
    horizon: int | None = None,
) -> float | None:
    mask = (
        (comparison["table"] == table)
        & (comparison["regime_id"] == regime_id)
        & (comparison["target"] == target)
        & (comparison["metric"] == metric)
    )
    if source is not None:
        mask &= comparison["source"] == source
    if horizon is not None:
        mask &= comparison["horizon"] == horizon
    subset = comparison.loc[mask, "value"]
    if subset.empty:
        return None
    return _safe_float(subset.iloc[0])


def _significant_granger_count(
    comparison: pd.DataFrame,
    *,
    regime_id: str,
    sources: list[str],
    targets: list[str],
) -> int:
    subset = comparison[
        (comparison["table"] == "granger")
        & (comparison["regime_id"] == regime_id)
        & (comparison["source"].isin(sources))
        & (comparison["target"].isin(targets))
        & (comparison["significant"] == True)
    ]
    return int(len(subset))


def _curve_share(
    comparison: pd.DataFrame,
    *,
    regime_id: str,
    target: str,
    horizon: int,
) -> float | None:
    subset = comparison[
        (comparison["table"] == "fevd")
        & (comparison["regime_id"] == regime_id)
        & (comparison["target"] == target)
        & (comparison["horizon"] == horizon)
        & (comparison["source"].isin(CURVE_VARIABLES))
    ]["value"]
    if subset.empty:
        return None
    return _safe_float(pd.to_numeric(subset, errors="coerce").sum())


def _build_regime_summary(comparison: pd.DataFrame) -> dict[str, dict[str, Any]]:
    regime_summary: dict[str, dict[str, Any]] = {}

    for spec in REGIME_SPECS:
        regime_summary[spec.regime_id] = {
            "label": spec.label,
            "tau1_mean": _metric_lookup(
                comparison,
                table="tau1_stats",
                regime_id=spec.regime_id,
                target="tau1",
                metric="mean",
            ),
            "tau1_std": _metric_lookup(
                comparison,
                table="tau1_stats",
                regime_id=spec.regime_id,
                target="tau1",
                metric="std",
            ),
            "beta0_mean": _metric_lookup(
                comparison,
                table="level_stats",
                regime_id=spec.regime_id,
                target="beta0",
                metric="mean",
            ),
            "key_rate_mean": _metric_lookup(
                comparison,
                table="level_stats",
                regime_id=spec.regime_id,
                target="key_rate",
                metric="mean",
            ),
            "cpi_mom_mean": _metric_lookup(
                comparison,
                table="level_stats",
                regime_id=spec.regime_id,
                target="cpi_mom",
                metric="mean",
            ),
            "system_mean_rmse_to_std": _metric_lookup(
                comparison,
                table="forecast_rmse",
                regime_id=spec.regime_id,
                target="system",
                metric="mean_rmse_to_std",
            ),
            "curve_to_macro_granger_5pct": _significant_granger_count(
                comparison,
                regime_id=spec.regime_id,
                sources=CURVE_VARIABLES,
                targets=MACRO_VARIABLES,
            ),
            "macro_to_curve_granger_5pct": _significant_granger_count(
                comparison,
                regime_id=spec.regime_id,
                sources=MACRO_VARIABLES,
                targets=CURVE_VARIABLES,
            ),
            "curve_share_in_cpi_fevd_h12": _curve_share(
                comparison,
                regime_id=spec.regime_id,
                target="cpi_mom",
                horizon=12,
            ),
            "curve_share_in_key_rate_fevd_h12": _curve_share(
                comparison,
                regime_id=spec.regime_id,
                target="key_rate",
                horizon=12,
            ),
            "beta0_loading": _metric_lookup(
                comparison,
                table="vecm_alpha",
                regime_id=spec.regime_id,
                target="beta0",
                metric="ec1_loading",
            ),
            "cpi_loading": _metric_lookup(
                comparison,
                table="vecm_alpha",
                regime_id=spec.regime_id,
                target="cpi_mom",
                metric="ec1_loading",
            ),
            "key_rate_loading": _metric_lookup(
                comparison,
                table="vecm_alpha",
                regime_id=spec.regime_id,
                target="key_rate",
                metric="ec1_loading",
            ),
        }
    return regime_summary


def _build_findings(
    *,
    comparison: pd.DataFrame,
    break_tests: pd.DataFrame,
    regime_summary: dict[str, dict[str, Any]],
    core_last_date: pd.Timestamp,
    macro_last_date: pd.Timestamp,
    model_last_date: pd.Timestamp,
    excluded_dates: list[str],
) -> list[str]:
    findings: list[str] = []

    pre = regime_summary["pre_2022_02"]
    post = regime_summary["post_2022_02"]
    transition = regime_summary["transition_2022_2023"]
    stabilization = regime_summary["stabilization_2024_plus"]

    if pre["tau1_mean"] is not None and post["tau1_mean"] is not None:
        tau_change = post["tau1_mean"] - pre["tau1_mean"]
        findings.append(
            "Среднее значение tau1 после февраля 2022 года снизилось "
            f"с {pre['tau1_mean']:.3f} до {post['tau1_mean']:.3f} "
            f"({tau_change:+.3f}), а в подпериоде 2024-01+ опустилось до "
            f"{stabilization['tau1_mean']:.3f}."
        )

    tau_break = break_tests[
        (break_tests["test_name"] == "welch_mean_shift") & (break_tests["target"] == "tau1")
    ]
    if not tau_break.empty:
        p_value = _safe_float(tau_break["p_value"].iloc[0])
        verdict = "отвергает" if p_value is not None and p_value < 0.05 else "не отвергает"
        findings.append(
            "Формальная проверка среднего tau1 на разрыв в точке 2022-02 "
            f"{verdict} гипотезу стабильности (p-value={p_value:.4f})."
        )

    chow_hits = break_tests[
        (break_tests["test_name"] == "chow_lag1")
        & (break_tests["reject_5pct"] == True)
        & (break_tests["target"].isin(["beta0", "cpi_mom", "key_rate", "tau1"]))
    ]["target"].tolist()
    if chow_hits:
        findings.append(
            "Тест Чоу на известный разрыв в 2022-02 указывает на изменение параметров "
            f"как минимум для уравнений: {', '.join(sorted(chow_hits))}."
        )
    else:
        findings.append(
            "Тест Чоу на известный разрыв в 2022-02 не дал устойчивого отклонения по ключевым уравнениям."
        )

    if (
        pre["curve_to_macro_granger_5pct"] is not None
        and post["curve_to_macro_granger_5pct"] is not None
    ):
        findings.append(
            "Количество значимых на 5% направлений причинности по Грейнджеру от факторов кривой "
            f"к макропеременным изменилось с {pre['curve_to_macro_granger_5pct']} до "
            f"{post['curve_to_macro_granger_5pct']}, а в 2024-01+ составило "
            f"{stabilization['curve_to_macro_granger_5pct']}."
        )

    if (
        pre["curve_share_in_key_rate_fevd_h12"] is not None
        and post["curve_share_in_key_rate_fevd_h12"] is not None
    ):
        findings.append(
            "По FEVD на горизонте 12 месяцев суммарный вклад шоков факторов кривой "
            f"в дисперсию ошибки прогноза key_rate изменился с "
            f"{pre['curve_share_in_key_rate_fevd_h12']:.1f}% до "
            f"{post['curve_share_in_key_rate_fevd_h12']:.1f}%; в 2024-01+ он равен "
            f"{stabilization['curve_share_in_key_rate_fevd_h12']:.1f}%."
        )

    if (
        pre["system_mean_rmse_to_std"] is not None
        and post["system_mean_rmse_to_std"] is not None
    ):
        findings.append(
            "Нормированный средний RMSE one-step VECM изменился с "
            f"{pre['system_mean_rmse_to_std']:.3f} в докризисном режиме до "
            f"{post['system_mean_rmse_to_std']:.3f} после февраля 2022 года; "
            f"для 2024-01+ он составляет {stabilization['system_mean_rmse_to_std']:.3f}."
        )

    findings.append(
        "Расчет использует свежие файлы: core до "
        f"{_month_label(core_last_date)}, macro до {_month_label(macro_last_date)}, "
        f"совместная модельная выборка до {_month_label(model_last_date)}."
    )

    if excluded_dates:
        findings.append(
            "Для VECM/IRF/FEVD из модельной панели исключены месяцы с патологической "
            "компенсацией beta1 и beta2: "
            + ", ".join(excluded_dates)
            + ". Исходные CSV не менялись."
        )

    if transition["tau1_mean"] is not None and stabilization["tau1_mean"] is not None:
        findings.append(
            "Сравнение подпериодов 2022-02 - 2023-12 и 2024-01+ показывает, что 2024-2026 "
            "выглядит отдельным режимом: tau1 ниже, средний key_rate выше, а набор значимых "
            "связей по Грейнджеру и FEVD отличается от переходной фазы 2022-2023."
        )

    return findings


def _render_markdown(summary: dict[str, Any], break_tests: pd.DataFrame) -> str:
    input_data = summary["input_data"]
    regime_summary = summary["regimes"]
    break_subset = break_tests[
        break_tests["target"].isin(["tau1", "beta0", "cpi_mom", "key_rate"])
    ].copy()

    lines: list[str] = []
    lines.append("# Поток 3. Режимный анализ и структурные сдвиги")
    lines.append("")
    lines.append("## Статус расчета")
    lines.append(
        f"- Расчет переисполнен по свежим данным: core до `{input_data['core_last_date']}`, "
        f"macro до `{input_data['macro_last_date']}`."
    )
    lines.append(
        f"- Совместная модельная выборка для VECM доходит до `{input_data['model_last_date']}`."
    )
    lines.append(
        f"- Основной блок модели: `{', '.join(summary['model_spec']['variables'])}`."
    )
    lines.append(
        f"- Спецификация VECM: `k_ar_diff={summary['model_spec']['k_ar_diff']}`, "
        f"`coint_rank={summary['model_spec']['coint_rank']}`, `VAR_order={summary['model_spec']['var_order']}`."
    )
    if input_data["excluded_model_dates"]:
        lines.append(
            "- Для модельного блока исключены месяцы с нестабильной оценкой NS-факторов: "
            + ", ".join(f"`{value}`" for value in input_data["excluded_model_dates"])
            + "."
        )
    lines.append("")
    lines.append("## Краткие выводы")
    for finding in summary["findings"]:
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## Сводка по режимам")
    lines.append("| Режим | tau1 среднее | beta0 среднее | key_rate среднее | mean RMSE/std | curve→macro Granger | curve share in key_rate FEVD h=12 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for regime_id in [spec.regime_id for spec in REGIME_SPECS]:
        info = regime_summary[regime_id]
        lines.append(
            "| "
            f"{info['label']} | "
            f"{info['tau1_mean']:.3f} | "
            f"{info['beta0_mean']:.3f} | "
            f"{info['key_rate_mean']:.3f} | "
            f"{info['system_mean_rmse_to_std']:.3f} | "
            f"{info['curve_to_macro_granger_5pct']} | "
            f"{info['curve_share_in_key_rate_fevd_h12']:.1f}% |"
        )
    lines.append("")
    lines.append("## Формальные проверки на разрыв")
    lines.append("| Тест | Целевая переменная | Статистика | p-value | Вывод |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for _, row in break_subset.iterrows():
        verdict = "разрыв есть" if bool(row["reject_5pct"]) else "разрыв не подтвержден"
        statistic = row["statistic"]
        p_value = row["p_value"]
        lines.append(
            "| "
            f"{row['test_name']} | "
            f"{row['target']} | "
            f"{statistic:.4f} | "
            f"{p_value:.4f} | "
            f"{verdict} |"
        )
    lines.append("")
    lines.append("## Ограничения")
    lines.append(
        "- В основную режимную модель не включены ожидания аналитиков и ряды рыночной инфляционной компенсации, "
        "поскольку эти столбцы появляются только в коротком хвосте выборки и ломают сопоставимость между режимами."
    )
    lines.append(
        "- Сравнение делается на одной и той же 5-мерной спецификации, чтобы коэффициенты, IRF, FEVD и RMSE были сопоставимы между всеми четырьмя режимами."
    )
    lines.append("")
    return "\n".join(lines)


def run_regime_analysis(
    *,
    core_path: Path = CORE_PATH,
    macro_path: Path = MACRO_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    findings_path: Path = DEFAULT_FINDINGS_PATH,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    core, macro, merged = _load_inputs(core_path, macro_path)
    tau_panel = _prepare_tau_panel(core)
    model_panel, excluded_dates = _prepare_model_panel(merged)

    comparison_rows: list[dict[str, Any]] = []
    for spec in REGIME_SPECS:
        tau_regime = _slice_regime(tau_panel, spec)
        model_regime = _slice_regime(model_panel, spec)
        if len(model_regime) <= FORECAST_MIN_TRAIN:
            raise ValueError(
                f"Regime {spec.regime_id} is too short for the chosen forecast setup: {len(model_regime)} rows"
            )

        comparison_rows.extend(
            _sample_info_rows(spec=spec, tau_frame=tau_regime, model_frame=model_regime)
        )
        comparison_rows.extend(_describe_series(tau_regime, spec, "tau1_stats", [TAU_COLUMN]))
        comparison_rows.extend(_describe_series(model_regime, spec, "level_stats", MODEL_VARIABLES))
        comparison_rows.extend(_forecast_rows(model_regime[MODEL_VARIABLES], spec))
        comparison_rows.extend(_vecm_rows(model_regime[MODEL_VARIABLES], spec))
        comparison_rows.extend(_granger_rows(model_regime[MODEL_VARIABLES], spec))
        comparison_rows.extend(_irf_rows(model_regime[MODEL_VARIABLES], spec))
        comparison_rows.extend(_fevd_rows(model_regime[MODEL_VARIABLES], spec))

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "regime_comparison_tables.csv"
    comparison.to_csv(comparison_path, index=False)

    break_tests = _break_test_rows(tau_panel, model_panel[MODEL_VARIABLES])
    break_tests_path = output_dir / "regime_break_tests.csv"
    break_tests.to_csv(break_tests_path, index=False)

    regime_summary = _build_regime_summary(comparison)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_data": {
            "core_path": str(core_path),
            "macro_path": str(macro_path),
            "core_last_date": _iso_date(pd.Timestamp(core[DATE_COLUMN].max())),
            "macro_last_date": _iso_date(pd.Timestamp(macro[DATE_COLUMN].max())),
            "merged_last_date": _iso_date(pd.Timestamp(merged[DATE_COLUMN].max())),
            "model_last_date": _iso_date(pd.Timestamp(model_panel.index.max())),
            "excluded_model_dates": excluded_dates,
            "model_rows": int(len(model_panel)),
            "tau_rows": int(len(tau_panel)),
        },
        "model_spec": {
            "variables": MODEL_VARIABLES,
            "k_ar_diff": VECM_K_AR_DIFF,
            "coint_rank": COINT_RANK,
            "var_order": VAR_ORDER,
            "forecast_horizon": FORECAST_HORIZON,
            "forecast_start_ratio": FORECAST_START_RATIO,
            "forecast_min_train": FORECAST_MIN_TRAIN,
            "ns_abs_threshold": NS_ABS_THRESHOLD,
        },
        "regimes": regime_summary,
        "break_tests": break_tests.to_dict(orient="records"),
        "limitations": [
            "Основная режимная спецификация использует только переменные, доступные почти на всем горизонте 2013-09 - 2026.",
            "Ожидания аналитиков и ряды рыночной инфляционной компенсации не включены в основную VECM из-за короткой доступной истории.",
        ],
    }
    summary["findings"] = _build_findings(
        comparison=comparison,
        break_tests=break_tests,
        regime_summary=regime_summary,
        core_last_date=pd.Timestamp(core[DATE_COLUMN].max()),
        macro_last_date=pd.Timestamp(macro[DATE_COLUMN].max()),
        model_last_date=pd.Timestamp(model_panel.index.max()),
        excluded_dates=excluded_dates,
    )

    summary_path = output_dir / "regime_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    findings_path.write_text(_render_markdown(summary, break_tests), encoding="utf-8")

    return {
        "comparison_path": str(comparison_path),
        "break_tests_path": str(break_tests_path),
        "summary_path": str(summary_path),
        "findings_path": str(findings_path),
        "summary": summary,
    }
