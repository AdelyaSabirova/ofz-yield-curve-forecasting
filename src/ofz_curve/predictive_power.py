from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nelson_siegel_svensson import NelsonSiegelCurve
from scipy import stats


DEFAULT_CORE_PATH = Path("data/processed/core_2026/monthly_ns_factors_core.csv")
DEFAULT_MACRO_PATH = Path("data/processed/macro_2026/monthly_expectations_panel.csv")
DEFAULT_OFFICIAL_CURVE_PATH = Path("data/processed/cbr_zero_curve/monthly_cbr_zero_curve.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/predictive_outputs")
DEFAULT_FINDINGS_PATH = Path("docs/p4_predictive_findings.md")

HORIZONS = (1, 3, 6, 12)
CURVE_FACTORS = ("beta0", "beta1", "beta2")
TAU1_FEATURE = "log_tau1"
MODEL_SPREAD_FEATURES = ("model_spread_10_1", "model_spread_5_1", "model_spread_10_3")
OFFICIAL_SHAPE_FEATURES = ("official_level", "official_slope", "official_curvature")
DELAYED_MONTHLY_FEATURES = (
    "cpi_yoy",
    "ofz_nonresident_share",
    "general_government_debt_growth_12m",
)
REGIME_BREAK_DATE = pd.Timestamp("2022-02-01")
DEFAULT_START_DATE = pd.Timestamp("2013-09-01")
STABILIZATION_START_DATE = pd.Timestamp("2024-01-01")

SPLIT_DEFINITIONS: tuple[tuple[str, str, pd.Timestamp, pd.Timestamp | None], ...] = (
    ("all", "Вся выборка", DEFAULT_START_DATE, None),
    ("pre_2022_02", "До февраля 2022 года", DEFAULT_START_DATE, pd.Timestamp("2022-01-01")),
    ("post_2022_02", "После февраля 2022 года", REGIME_BREAK_DATE, None),
)

MARKET_CONTROL_SPLIT_DEFINITIONS: tuple[tuple[str, str, pd.Timestamp, pd.Timestamp | None], ...] = (
    ("all", "Вся выборка", DEFAULT_START_DATE, None),
    ("post_2022_02", "После февраля 2022 года", REGIME_BREAK_DATE, None),
    ("stabilization_2024_plus", "Устойчивый режим 2024--2026", STABILIZATION_START_DATE, None),
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    base_features: tuple[str, ...]
    baseline_description: str
    augmented_description: str


@dataclass(frozen=True)
class TargetSpec:
    name: str
    label: str
    source_column: str
    lag_feature_column: str
    target_type: str
    average_future: bool
    model_specs: tuple[ModelSpec, ...]
    actual_notes: str


@dataclass(frozen=True)
class CurveBenchmarkSpec:
    name: str
    label: str
    feature_cols: tuple[str, ...]
    comparison_family: str
    baseline_description: str
    augmented_description: str


INFLATION_SPECS = (
    ModelSpec(
        name="ar",
        label="Авторегрессия",
        base_features=("lag_target",),
        baseline_description="Последняя опубликованная инфляция",
        augmented_description="Последняя опубликованная инфляция + beta0,beta1,beta2",
    ),
    ModelSpec(
        name="key_rate",
        label="Ключевая ставка",
        base_features=("lag_target", "key_rate"),
        baseline_description="Последняя опубликованная инфляция + ключевая ставка",
        augmented_description="Последняя опубликованная инфляция + ключевая ставка + beta0,beta1,beta2",
    ),
    ModelSpec(
        name="usd",
        label="Курс USD/RUB",
        base_features=("lag_target", "log_usd_rub"),
        baseline_description="Последняя опубликованная инфляция + log USD/RUB",
        augmented_description="Последняя опубликованная инфляция + log USD/RUB + beta0,beta1,beta2",
    ),
    ModelSpec(
        name="expectations",
        label="Ожидания аналитиков",
        base_features=("lag_target", "analyst_cpi_2026"),
        baseline_description="Последняя опубликованная инфляция + ожидания аналитиков по ИПЦ на 2026 год",
        augmented_description="Последняя опубликованная инфляция + ожидания аналитиков + beta0,beta1,beta2",
    ),
)

KEY_RATE_SPECS = (
    ModelSpec(
        name="ar",
        label="Авторегрессия",
        base_features=("lag_target",),
        baseline_description="Текущая ключевая ставка",
        augmented_description="Текущая ключевая ставка + beta0,beta1,beta2",
    ),
    ModelSpec(
        name="inflation",
        label="Инфляция",
        base_features=("lag_target", "cpi_yoy_lag1"),
        baseline_description="Текущая ключевая ставка + последняя опубликованная инфляция",
        augmented_description="Текущая ключевая ставка + последняя опубликованная инфляция + beta0,beta1,beta2",
    ),
    ModelSpec(
        name="usd",
        label="Курс USD/RUB",
        base_features=("lag_target", "log_usd_rub"),
        baseline_description="Текущая ключевая ставка + log USD/RUB",
        augmented_description="Текущая ключевая ставка + log USD/RUB + beta0,beta1,beta2",
    ),
    ModelSpec(
        name="expectations",
        label="Ожидания аналитиков",
        base_features=("lag_target", "analyst_key_rate_2026"),
        baseline_description="Текущая ключевая ставка + ожидания аналитиков по средней ставке 2026",
        augmented_description="Текущая ключевая ставка + ожидания аналитиков + beta0,beta1,beta2",
    ),
)

KEY_RATE_MARKET_CONTROL_SPECS = (
    ModelSpec(
        name="inflation_breakeven10",
        label="Инфляция + 10-летняя инфляционная компенсация",
        base_features=("lag_target", "cpi_yoy_lag1", "breakeven_10y"),
        baseline_description="Текущая ключевая ставка + последняя опубликованная инфляция + 10-летняя рыночная инфляционная компенсация",
        augmented_description="Текущая ключевая ставка + последняя опубликованная инфляция + 10-летняя рыночная инфляционная компенсация + beta0,beta1,beta2",
    ),
    ModelSpec(
        name="inflation_debt_growth",
        label="Инфляция + рост долга сектора госуправления",
        base_features=("lag_target", "cpi_yoy_lag1", "general_government_debt_growth_12m_lag1"),
        baseline_description="Текущая ключевая ставка + последняя опубликованная инфляция + рост долга сектора государственного управления",
        augmented_description="Текущая ключевая ставка + последняя опубликованная инфляция + рост долга сектора государственного управления + beta0,beta1,beta2",
    ),
    ModelSpec(
        name="inflation_nonresident_share",
        label="Инфляция + доля нерезидентов",
        base_features=("lag_target", "cpi_yoy_lag1", "ofz_nonresident_share_lag1"),
        baseline_description="Текущая ключевая ставка + последняя опубликованная инфляция + доля нерезидентов в ОФЗ",
        augmented_description="Текущая ключевая ставка + последняя опубликованная инфляция + доля нерезидентов в ОФЗ + beta0,beta1,beta2",
    ),
    ModelSpec(
        name="inflation_debt_and_nonresident",
        label="Инфляция + рост долга + доля нерезидентов",
        base_features=("lag_target", "cpi_yoy_lag1", "general_government_debt_growth_12m_lag1", "ofz_nonresident_share_lag1"),
        baseline_description="Текущая ключевая ставка + последняя опубликованная инфляция + рост долга сектора государственного управления + доля нерезидентов",
        augmented_description="Текущая ключевая ставка + последняя опубликованная инфляция + рост долга сектора государственного управления + доля нерезидентов + beta0,beta1,beta2",
    ),
    ModelSpec(
        name="inflation_full_controls",
        label="Инфляция + все наблюдаемые контроли",
        base_features=(
            "lag_target",
            "cpi_yoy_lag1",
            "breakeven_10y",
            "general_government_debt_growth_12m_lag1",
            "ofz_nonresident_share_lag1",
        ),
        baseline_description=(
            "Текущая ключевая ставка + последняя опубликованная инфляция + 10-летняя рыночная инфляционная компенсация "
            "+ рост долга сектора государственного управления + доля нерезидентов"
        ),
        augmented_description=(
            "Текущая ключевая ставка + последняя опубликованная инфляция + 10-летняя рыночная инфляционная компенсация "
            "+ рост долга сектора государственного управления + доля нерезидентов + beta0,beta1,beta2"
        ),
    ),
)

TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(
        name="inflation",
        label="Инфляция",
        source_column="cpi_yoy",
        lag_feature_column="cpi_yoy_lag1",
        target_type="future_level",
        average_future=False,
        model_specs=INFLATION_SPECS,
        actual_notes="Прогнозируется годовая инфляция `cpi_yoy` на горизонте `h` месяцев.",
    ),
    TargetSpec(
        name="key_rate",
        label="Ключевая ставка",
        source_column="key_rate",
        lag_feature_column="key_rate",
        target_type="future_average",
        average_future=True,
        model_specs=KEY_RATE_SPECS,
        actual_notes="Прогнозируется средняя ключевая ставка за следующие `h` месяцев.",
    ),
)

CURVE_BENCHMARK_SPECS: tuple[CurveBenchmarkSpec, ...] = (
    CurveBenchmarkSpec(
        name="simple_spreads",
        label="Простые спреды авторской кривой",
        feature_cols=MODEL_SPREAD_FEATURES,
        comparison_family="ns_vs_simple_spreads",
        baseline_description="Базовая модель + beta0,beta1,beta2",
        augmented_description="Базовая модель + простые спреды авторской кривой",
    ),
    CurveBenchmarkSpec(
        name="official_shape",
        label="Простые характеристики официальной КБД",
        feature_cols=OFFICIAL_SHAPE_FEATURES,
        comparison_family="ns_vs_official_shape",
        baseline_description="Базовая модель + beta0,beta1,beta2",
        augmented_description="Базовая модель + уровень, наклон и кривизна официальной КБД",
    ),
)


def load_analysis_panel(
    core_path: Path | str = DEFAULT_CORE_PATH,
    macro_path: Path | str = DEFAULT_MACRO_PATH,
    official_curve_path: Path | str = DEFAULT_OFFICIAL_CURVE_PATH,
) -> pd.DataFrame:
    core_df = pd.read_csv(core_path, parse_dates=["date"])
    macro_df = pd.read_csv(macro_path, parse_dates=["date"])
    data = core_df.merge(macro_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    data = data.loc[data["date"] >= DEFAULT_START_DATE].copy()
    data["log_usd_rub"] = np.log(data["usd_rub"])
    data["log_tau1"] = np.where(data["tau1"] > 0, np.log(data["tau1"]), np.nan)
    for feature in DELAYED_MONTHLY_FEATURES:
        data[f"{feature}_lag1"] = data[feature].shift(1)
    data = _add_curve_benchmark_features(data=data, core_df=core_df, official_curve_path=official_curve_path)
    return data


def _build_model_curve_monthly(core_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    node_map = {
        "model_y1_00": 1.0,
        "model_y3_00": 3.0,
        "model_y5_00": 5.0,
        "model_y10_00": 10.0,
    }
    for row in core_df.loc[:, ["date", "curve_source_date", "beta0", "beta1", "beta2", "tau1"]].itertuples(index=False):
        if any(pd.isna(value) for value in (row.beta0, row.beta1, row.beta2, row.tau1)) or row.tau1 <= 0:
            records.append(
                {
                    "date": row.date,
                    "curve_source_date": row.curve_source_date,
                    **{column: np.nan for column in node_map},
                }
            )
            continue
        curve = NelsonSiegelCurve(beta0=float(row.beta0), beta1=float(row.beta1), beta2=float(row.beta2), tau=float(row.tau1))
        record = {"date": row.date, "curve_source_date": row.curve_source_date}
        for column, maturity in node_map.items():
            record[column] = float(curve(maturity))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _add_curve_benchmark_features(
    data: pd.DataFrame,
    core_df: pd.DataFrame,
    official_curve_path: Path | str,
) -> pd.DataFrame:
    result = data.copy()
    monthly_curve = _build_model_curve_monthly(core_df=core_df)
    monthly_curve["model_spread_10_1"] = monthly_curve["model_y10_00"] - monthly_curve["model_y1_00"]
    monthly_curve["model_spread_5_1"] = monthly_curve["model_y5_00"] - monthly_curve["model_y1_00"]
    monthly_curve["model_spread_10_3"] = monthly_curve["model_y10_00"] - monthly_curve["model_y3_00"]
    monthly_curve = monthly_curve.loc[:, ["date", *MODEL_SPREAD_FEATURES]].copy()
    result = result.merge(monthly_curve, on="date", how="left")

    official_curve = pd.read_csv(official_curve_path, parse_dates=["date"])
    official_curve = official_curve.loc[:, ["date", "y1_00", "y5_00", "y10_00"]].copy()
    for column in ("y1_00", "y5_00", "y10_00"):
        official_curve[column] = official_curve[column] / 100.0
    official_curve["official_level"] = (
        official_curve["y1_00"] + official_curve["y5_00"] + official_curve["y10_00"]
    ) / 3.0
    official_curve["official_slope"] = official_curve["y10_00"] - official_curve["y1_00"]
    official_curve["official_curvature"] = 2.0 * official_curve["y5_00"] - official_curve["y1_00"] - official_curve["y10_00"]
    result = result.merge(
        official_curve.loc[:, ["date", *OFFICIAL_SHAPE_FEATURES]],
        on="date",
        how="left",
    )
    return result


def _forward_average(series: pd.Series, horizon: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    result = np.full(shape=len(values), fill_value=np.nan, dtype=float)
    for idx in range(len(values) - horizon):
        window = values[idx + 1 : idx + horizon + 1]
        if np.isnan(window).any():
            continue
        result[idx] = float(window.mean())
    return pd.Series(result, index=series.index)


def _lookup_split(
    split_key: str,
    split_definitions: tuple[tuple[str, str, pd.Timestamp, pd.Timestamp | None], ...],
) -> tuple[str, pd.Timestamp, pd.Timestamp | None]:
    for key, label, start, end in split_definitions:
        if key == split_key:
            return label, start, end
    raise KeyError(f"Не найден разрез выборки: {split_key}")


def _prepare_split_frame(
    data: pd.DataFrame,
    target_spec: TargetSpec,
    horizon: int,
    split_key: str,
    split_definitions: tuple[tuple[str, str, pd.Timestamp, pd.Timestamp | None], ...] = SPLIT_DEFINITIONS,
) -> pd.DataFrame:
    _, split_start, split_end = _lookup_split(split_key=split_key, split_definitions=split_definitions)
    split_df = data.loc[data["date"] >= split_start].copy()
    if split_end is not None:
        split_df = split_df.loc[split_df["date"] <= split_end].copy()
    split_df = split_df.sort_values("date").reset_index(drop=True)

    split_df["lag_target"] = split_df[target_spec.lag_feature_column]
    if target_spec.average_future:
        split_df["target_value"] = _forward_average(split_df[target_spec.source_column], horizon)
    else:
        split_df["target_value"] = split_df[target_spec.source_column].shift(-horizon)
    split_df["target_date"] = split_df["date"].shift(-horizon)
    return split_df


def _initial_train_size(feature_count: int, horizon: int) -> int:
    return max(12, horizon, feature_count * 2)


def _fit_ols_predict(train_df: pd.DataFrame, test_row: pd.Series, feature_cols: tuple[str, ...]) -> float:
    x_train = train_df.loc[:, list(feature_cols)].to_numpy(dtype=float)
    y_train = train_df["target_value"].to_numpy(dtype=float)
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    beta_hat, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    x_test = np.concatenate(([1.0], test_row.loc[list(feature_cols)].to_numpy(dtype=float)))
    return float(np.dot(x_test, beta_hat))


def _forecast_with_expanding_window(
    split_df: pd.DataFrame,
    feature_cols: tuple[str, ...],
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    empty_forecasts = pd.DataFrame(
        columns=[
            "forecast_date",
            "target_date",
            "actual_value",
            "forecast_value",
            "error",
            "squared_error",
            "absolute_error",
        ]
    )
    required_cols = ["date", "target_date", "target_value", *feature_cols]
    working_df = split_df.loc[:, required_cols].dropna().reset_index(drop=True)
    min_train = _initial_train_size(feature_count=len(feature_cols), horizon=horizon)
    if len(working_df) < min_train + 3:
        return empty_forecasts, {
            "status": "insufficient_data",
            "complete_rows": int(len(working_df)),
            "min_train_size": int(min_train),
        }

    records: list[dict[str, Any]] = []
    for idx in range(min_train, len(working_df)):
        train_df = working_df.iloc[:idx]
        test_row = working_df.iloc[idx]
        prediction = _fit_ols_predict(train_df=train_df, test_row=test_row, feature_cols=feature_cols)
        actual = float(test_row["target_value"])
        records.append(
            {
                "forecast_date": test_row["date"],
                "target_date": test_row["target_date"],
                "actual_value": actual,
                "forecast_value": prediction,
                "error": actual - prediction,
                "squared_error": (actual - prediction) ** 2,
                "absolute_error": abs(actual - prediction),
            }
        )

    return pd.DataFrame.from_records(records), {
        "status": "ok",
        "complete_rows": int(len(working_df)),
        "min_train_size": int(min_train),
    }


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors.to_numpy(dtype=float)))))


def diebold_mariano_test(
    baseline_errors: pd.Series,
    augmented_errors: pd.Series,
    horizon: int,
) -> dict[str, float | int | str]:
    loss_diff = np.square(baseline_errors.to_numpy(dtype=float)) - np.square(augmented_errors.to_numpy(dtype=float))
    sample_size = int(loss_diff.shape[0])
    if sample_size < max(5, horizon + 1):
        return {
            "status": "insufficient_data",
            "n_obs": sample_size,
            "mean_loss_diff": float(np.mean(loss_diff)) if sample_size else np.nan,
            "dm_stat": np.nan,
            "p_value": np.nan,
        }

    centered = loss_diff - loss_diff.mean()
    gamma0 = float(np.dot(centered, centered) / sample_size)
    long_run_variance = gamma0
    for lag in range(1, horizon):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / sample_size)
        weight = 1.0 - lag / horizon
        long_run_variance += 2.0 * weight * cov

    if long_run_variance <= 0:
        return {
            "status": "non_positive_variance",
            "n_obs": sample_size,
            "mean_loss_diff": float(loss_diff.mean()),
            "dm_stat": np.nan,
            "p_value": np.nan,
        }

    dm_stat = float(loss_diff.mean() / np.sqrt(long_run_variance / sample_size))
    harvey_adjustment = np.sqrt((sample_size + 1 - 2 * horizon + (horizon * (horizon - 1) / sample_size)) / sample_size)
    dm_stat_adjusted = dm_stat * harvey_adjustment
    p_value = float(2.0 * stats.t.sf(abs(dm_stat_adjusted), df=sample_size - 1))
    return {
        "status": "ok",
        "n_obs": sample_size,
        "mean_loss_diff": float(loss_diff.mean()),
        "dm_stat": dm_stat_adjusted,
        "p_value": p_value,
    }


def _comparison_rows_from_forecasts(
    *,
    target_name: str,
    target_label: str,
    target_type: str,
    split_key: str,
    split_label: str,
    horizon: int,
    model_name: str,
    model_label: str,
    baseline_description: str,
    augmented_description: str,
    baseline_features: tuple[str, ...],
    augmented_features: tuple[str, ...],
    baseline_forecasts: pd.DataFrame,
    baseline_meta: dict[str, Any],
    augmented_forecasts: pd.DataFrame,
    augmented_meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = baseline_forecasts.rename(
        columns={
            "forecast_value": "baseline_forecast",
            "error": "baseline_error",
            "squared_error": "baseline_squared_error",
            "absolute_error": "baseline_absolute_error",
        }
    )
    aug = augmented_forecasts.rename(
        columns={
            "forecast_value": "augmented_forecast",
            "error": "augmented_error",
            "squared_error": "augmented_squared_error",
            "absolute_error": "augmented_absolute_error",
        }
    )
    merged = base.merge(
        aug,
        on=["forecast_date", "target_date", "actual_value"],
        how="inner",
        validate="one_to_one",
    )

    status = "ok" if not merged.empty else "insufficient_data"
    note_parts: list[str] = []
    if baseline_meta.get("status") != "ok":
        note_parts.append(f"baseline: {baseline_meta.get('status')}")
    if augmented_meta.get("status") != "ok":
        note_parts.append(f"augmented: {augmented_meta.get('status')}")
    if merged.empty:
        note_parts.append("no overlapping forecasts")

    dm_result = diebold_mariano_test(
        baseline_errors=merged["baseline_error"] if not merged.empty else pd.Series(dtype=float),
        augmented_errors=merged["augmented_error"] if not merged.empty else pd.Series(dtype=float),
        horizon=horizon,
    )
    if dm_result["status"] != "ok":
        note_parts.append(f"dm: {dm_result['status']}")

    if not merged.empty:
        baseline_rmse = _rmse(merged["baseline_error"])
        augmented_rmse = _rmse(merged["augmented_error"])
        baseline_mae = float(merged["baseline_absolute_error"].mean())
        augmented_mae = float(merged["augmented_absolute_error"].mean())
        rmse_diff = baseline_rmse - augmented_rmse
        rmse_gain_pct = (rmse_diff / baseline_rmse * 100.0) if baseline_rmse else np.nan
        forecast_start = merged["forecast_date"].min()
        forecast_end = merged["forecast_date"].max()
        target_start = merged["target_date"].min()
        target_end = merged["target_date"].max()
    else:
        baseline_rmse = np.nan
        augmented_rmse = np.nan
        baseline_mae = np.nan
        augmented_mae = np.nan
        rmse_diff = np.nan
        rmse_gain_pct = np.nan
        forecast_start = pd.NaT
        forecast_end = pd.NaT
        target_start = pd.NaT
        target_end = pd.NaT

    comparison_row = {
        "target_name": target_name,
        "target_label": target_label,
        "target_type": target_type,
        "split": split_key,
        "split_label": split_label,
        "horizon_months": int(horizon),
        "model_name": model_name,
        "model_label": model_label,
        "baseline_description": baseline_description,
        "augmented_description": augmented_description,
        "baseline_features": ",".join(baseline_features),
        "augmented_features": ",".join(augmented_features),
        "n_baseline_forecasts": int(len(baseline_forecasts)),
        "n_augmented_forecasts": int(len(augmented_forecasts)),
        "n_common_forecasts": int(len(merged)),
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "target_start": target_start,
        "target_end": target_end,
        "baseline_rmse": baseline_rmse,
        "augmented_rmse": augmented_rmse,
        "baseline_mae": baseline_mae,
        "augmented_mae": augmented_mae,
        "rmse_diff": rmse_diff,
        "rmse_gain_pct": rmse_gain_pct,
        "curve_helps": bool(rmse_diff > 0) if pd.notna(rmse_diff) else False,
        "status": status,
        "note": "; ".join(note_parts),
    }

    dm_row = {
        "target_name": target_name,
        "target_label": target_label,
        "split": split_key,
        "split_label": split_label,
        "horizon_months": int(horizon),
        "model_name": model_name,
        "model_label": model_label,
        "n_common_forecasts": int(len(merged)),
        "mean_loss_diff": dm_result["mean_loss_diff"],
        "dm_stat": dm_result["dm_stat"],
        "p_value": dm_result["p_value"],
        "curve_helps_on_average": bool(dm_result["mean_loss_diff"] > 0) if pd.notna(dm_result["mean_loss_diff"]) else False,
        "significant_10pct": bool(dm_result["p_value"] < 0.10) if pd.notna(dm_result["p_value"]) else False,
        "status": dm_result["status"],
        "note": "; ".join(note_parts),
    }
    return comparison_row, dm_row


def _comparison_row(
    target_spec: TargetSpec,
    model_spec: ModelSpec,
    split_key: str,
    split_label: str,
    horizon: int,
    baseline_forecasts: pd.DataFrame,
    baseline_meta: dict[str, Any],
    augmented_forecasts: pd.DataFrame,
    augmented_meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_features = model_spec.base_features
    augmented_features = (*model_spec.base_features, *CURVE_FACTORS)
    return _comparison_rows_from_forecasts(
        target_name=target_spec.name,
        target_label=target_spec.label,
        target_type=target_spec.target_type,
        split_key=split_key,
        split_label=split_label,
        horizon=horizon,
        model_name=model_spec.name,
        model_label=model_spec.label,
        baseline_description=model_spec.baseline_description,
        augmented_description=model_spec.augmented_description,
        baseline_features=baseline_features,
        augmented_features=augmented_features,
        baseline_forecasts=baseline_forecasts,
        baseline_meta=baseline_meta,
        augmented_forecasts=augmented_forecasts,
        augmented_meta=augmented_meta,
    )


def evaluate_target_models(data: pd.DataFrame, target_spec: TargetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []

    for split_key, split_label, _, _ in SPLIT_DEFINITIONS:
        for horizon in HORIZONS:
            split_df = _prepare_split_frame(data=data, target_spec=target_spec, horizon=horizon, split_key=split_key)
            for model_spec in target_spec.model_specs:
                baseline_forecasts, baseline_meta = _forecast_with_expanding_window(
                    split_df=split_df,
                    feature_cols=model_spec.base_features,
                    horizon=horizon,
                )
                augmented_forecasts, augmented_meta = _forecast_with_expanding_window(
                    split_df=split_df,
                    feature_cols=(*model_spec.base_features, *CURVE_FACTORS),
                    horizon=horizon,
                )
                comparison_row, dm_row = _comparison_row(
                    target_spec=target_spec,
                    model_spec=model_spec,
                    split_key=split_key,
                    split_label=split_label,
                    horizon=horizon,
                    baseline_forecasts=baseline_forecasts,
                    baseline_meta=baseline_meta,
                    augmented_forecasts=augmented_forecasts,
                    augmented_meta=augmented_meta,
                )
                comparison_rows.append(comparison_row)
                dm_rows.append(dm_row)

    comparison_df = pd.DataFrame.from_records(comparison_rows)
    dm_df = pd.DataFrame.from_records(dm_rows)
    return comparison_df, dm_df


def evaluate_tau1_increment_for_key_rate(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_spec = next(target for target in TARGET_SPECS if target.name == "key_rate")
    comparison_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []

    for split_key, split_label, _, _ in SPLIT_DEFINITIONS:
        for horizon in HORIZONS:
            split_df = _prepare_split_frame(data=data, target_spec=target_spec, horizon=horizon, split_key=split_key)
            for model_spec in target_spec.model_specs:
                baseline_features = (*model_spec.base_features, *CURVE_FACTORS)
                augmented_features = (*baseline_features, TAU1_FEATURE)
                baseline_forecasts, baseline_meta = _forecast_with_expanding_window(
                    split_df=split_df,
                    feature_cols=baseline_features,
                    horizon=horizon,
                )
                augmented_forecasts, augmented_meta = _forecast_with_expanding_window(
                    split_df=split_df,
                    feature_cols=augmented_features,
                    horizon=horizon,
                )
                comparison_row, dm_row = _comparison_rows_from_forecasts(
                    target_name=target_spec.name,
                    target_label=target_spec.label,
                    target_type=target_spec.target_type,
                    split_key=split_key,
                    split_label=split_label,
                    horizon=horizon,
                    model_name=model_spec.name,
                    model_label=model_spec.label,
                    baseline_description=model_spec.augmented_description,
                    augmented_description=f"{model_spec.augmented_description} + ln(tau1)",
                    baseline_features=baseline_features,
                    augmented_features=augmented_features,
                    baseline_forecasts=baseline_forecasts,
                    baseline_meta=baseline_meta,
                    augmented_forecasts=augmented_forecasts,
                    augmented_meta=augmented_meta,
                )
                comparison_row["comparison_family"] = "ns_vs_ns_tau1"
                dm_row["comparison_family"] = "ns_vs_ns_tau1"
                comparison_rows.append(comparison_row)
                dm_rows.append(dm_row)

    comparison_df = pd.DataFrame.from_records(comparison_rows)
    dm_df = pd.DataFrame.from_records(dm_rows)
    return comparison_df, dm_df


def evaluate_curve_benchmarks_for_key_rate(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_spec = next(target for target in TARGET_SPECS if target.name == "key_rate")
    comparison_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []

    for split_key, split_label, _, _ in SPLIT_DEFINITIONS:
        for horizon in HORIZONS:
            split_df = _prepare_split_frame(data=data, target_spec=target_spec, horizon=horizon, split_key=split_key)
            for model_spec in target_spec.model_specs:
                baseline_features = (*model_spec.base_features, *CURVE_FACTORS)
                baseline_forecasts, baseline_meta = _forecast_with_expanding_window(
                    split_df=split_df,
                    feature_cols=baseline_features,
                    horizon=horizon,
                )
                for benchmark_spec in CURVE_BENCHMARK_SPECS:
                    augmented_features = (*model_spec.base_features, *benchmark_spec.feature_cols)
                    augmented_forecasts, augmented_meta = _forecast_with_expanding_window(
                        split_df=split_df,
                        feature_cols=augmented_features,
                        horizon=horizon,
                    )
                    comparison_row, dm_row = _comparison_rows_from_forecasts(
                        target_name=target_spec.name,
                        target_label=target_spec.label,
                        target_type=target_spec.target_type,
                        split_key=split_key,
                        split_label=split_label,
                        horizon=horizon,
                        model_name=model_spec.name,
                        model_label=model_spec.label,
                        baseline_description=benchmark_spec.baseline_description,
                        augmented_description=benchmark_spec.augmented_description,
                        baseline_features=baseline_features,
                        augmented_features=augmented_features,
                        baseline_forecasts=baseline_forecasts,
                        baseline_meta=baseline_meta,
                        augmented_forecasts=augmented_forecasts,
                        augmented_meta=augmented_meta,
                    )
                    comparison_row["comparison_family"] = benchmark_spec.comparison_family
                    comparison_row["benchmark_name"] = benchmark_spec.name
                    comparison_row["benchmark_label"] = benchmark_spec.label
                    dm_row["comparison_family"] = benchmark_spec.comparison_family
                    dm_row["benchmark_name"] = benchmark_spec.name
                    dm_row["benchmark_label"] = benchmark_spec.label
                    comparison_rows.append(comparison_row)
                    dm_rows.append(dm_row)

    comparison_df = pd.DataFrame.from_records(comparison_rows)
    dm_df = pd.DataFrame.from_records(dm_rows)
    return comparison_df, dm_df


def evaluate_market_control_models_for_key_rate(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_spec = next(target for target in TARGET_SPECS if target.name == "key_rate")
    comparison_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []

    for split_key, split_label, _, _ in MARKET_CONTROL_SPLIT_DEFINITIONS:
        for horizon in HORIZONS:
            split_df = _prepare_split_frame(
                data=data,
                target_spec=target_spec,
                horizon=horizon,
                split_key=split_key,
                split_definitions=MARKET_CONTROL_SPLIT_DEFINITIONS,
            )
            for model_spec in KEY_RATE_MARKET_CONTROL_SPECS:
                baseline_features = model_spec.base_features
                augmented_features = (*model_spec.base_features, *CURVE_FACTORS)
                baseline_forecasts, baseline_meta = _forecast_with_expanding_window(
                    split_df=split_df,
                    feature_cols=baseline_features,
                    horizon=horizon,
                )
                augmented_forecasts, augmented_meta = _forecast_with_expanding_window(
                    split_df=split_df,
                    feature_cols=augmented_features,
                    horizon=horizon,
                )
                comparison_row, dm_row = _comparison_rows_from_forecasts(
                    target_name=target_spec.name,
                    target_label=target_spec.label,
                    target_type=target_spec.target_type,
                    split_key=split_key,
                    split_label=split_label,
                    horizon=horizon,
                    model_name=model_spec.name,
                    model_label=model_spec.label,
                    baseline_description=model_spec.baseline_description,
                    augmented_description=model_spec.augmented_description,
                    baseline_features=baseline_features,
                    augmented_features=augmented_features,
                    baseline_forecasts=baseline_forecasts,
                    baseline_meta=baseline_meta,
                    augmented_forecasts=augmented_forecasts,
                    augmented_meta=augmented_meta,
                )
                comparison_row["comparison_family"] = "market_controls_vs_ns"
                dm_row["comparison_family"] = "market_controls_vs_ns"
                comparison_rows.append(comparison_row)
                dm_rows.append(dm_row)

    comparison_df = pd.DataFrame.from_records(comparison_rows)
    dm_df = pd.DataFrame.from_records(dm_rows)
    return comparison_df, dm_df


def summarise_curve_benchmark_results(
    comparison_df: pd.DataFrame,
    dm_df: pd.DataFrame,
) -> pd.DataFrame:
    ok_comparison = comparison_df.loc[comparison_df["status"] == "ok"].copy()
    ok_dm = dm_df.loc[dm_df["status"] == "ok"].copy()
    if ok_comparison.empty:
        return pd.DataFrame(
            columns=[
                "comparison_family",
                "benchmark_name",
                "benchmark_label",
                "split",
                "split_label",
                "evaluated_combinations",
                "benchmark_better",
                "mean_rmse_gain_pct",
                "median_rmse_gain_pct",
                "evaluated_horizons",
                "significant_benchmark_improvements",
            ]
        )

    summary = (
        ok_comparison.groupby(
            ["comparison_family", "benchmark_name", "benchmark_label", "split", "split_label"],
            as_index=False,
        )
        .agg(
            evaluated_combinations=("curve_helps", "size"),
            benchmark_better=("curve_helps", "sum"),
            mean_rmse_gain_pct=("rmse_gain_pct", "mean"),
            median_rmse_gain_pct=("rmse_gain_pct", "median"),
            evaluated_horizons=("horizon_months", lambda values: ",".join(str(int(value)) for value in sorted(set(values)))),
        )
    )
    significant = (
        ok_dm.loc[ok_dm["significant_10pct"] & ok_dm["curve_helps_on_average"]]
        .groupby(["comparison_family", "benchmark_name", "split"], as_index=False)
        .agg(significant_benchmark_improvements=("significant_10pct", "size"))
    )
    summary = summary.merge(
        significant,
        on=["comparison_family", "benchmark_name", "split"],
        how="left",
    )
    summary["significant_benchmark_improvements"] = summary["significant_benchmark_improvements"].fillna(0).astype(int)
    return summary.sort_values(["comparison_family", "split"]).reset_index(drop=True)


def summarise_market_control_results(
    comparison_df: pd.DataFrame,
    dm_df: pd.DataFrame,
) -> pd.DataFrame:
    ok_comparison = comparison_df.loc[comparison_df["status"] == "ok"].copy()
    ok_dm = dm_df.loc[dm_df["status"] == "ok"].copy()
    if ok_comparison.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "split_label",
                "model_name",
                "model_label",
                "evaluated_combinations",
                "curve_better",
                "mean_rmse_gain_pct",
                "median_rmse_gain_pct",
                "evaluated_horizons",
                "significant_curve_improvements",
            ]
        )

    summary = (
        ok_comparison.groupby(["split", "split_label", "model_name", "model_label"], as_index=False)
        .agg(
            evaluated_combinations=("curve_helps", "size"),
            curve_better=("curve_helps", "sum"),
            mean_rmse_gain_pct=("rmse_gain_pct", "mean"),
            median_rmse_gain_pct=("rmse_gain_pct", "median"),
            evaluated_horizons=("horizon_months", lambda values: ",".join(str(int(value)) for value in sorted(set(values)))),
        )
    )
    significant = (
        ok_dm.loc[ok_dm["significant_10pct"] & ok_dm["curve_helps_on_average"]]
        .groupby(["split", "model_name"], as_index=False)
        .agg(significant_curve_improvements=("significant_10pct", "size"))
    )
    summary = summary.merge(significant, on=["split", "model_name"], how="left")
    summary["significant_curve_improvements"] = summary["significant_curve_improvements"].fillna(0).astype(int)
    return summary.sort_values(["split", "model_name"]).reset_index(drop=True)


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}%"


def _format_float(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.4f}"


def _markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if df.empty:
        return "| Нет данных |\n| --- |\n| Нет оценимых комбинаций |\n"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_findings_markdown(
    data: pd.DataFrame,
    inflation_df: pd.DataFrame,
    key_rate_df: pd.DataFrame,
    dm_df: pd.DataFrame,
    key_rate_tau1_df: pd.DataFrame,
    key_rate_tau1_dm_df: pd.DataFrame,
    curve_benchmark_summary_df: pd.DataFrame,
    market_control_comparison_df: pd.DataFrame,
    market_control_dm_df: pd.DataFrame,
    market_control_summary_df: pd.DataFrame,
    summary_metadata: dict[str, Any],
) -> str:
    latest_input_date = data["date"].max().strftime("%Y-%m-%d")
    inflation_actual_end = data.loc[data["cpi_yoy"].notna(), "date"].max().strftime("%Y-%m-%d")
    key_rate_actual_end = data.loc[data["key_rate"].notna(), "date"].max().strftime("%Y-%m-%d")
    analyst_first = data.loc[data["analyst_cpi_2026"].notna(), "date"].min().strftime("%Y-%m-%d")

    ok_inflation = inflation_df.loc[inflation_df["status"] == "ok"].copy()
    ok_key_rate = key_rate_df.loc[key_rate_df["status"] == "ok"].copy()
    ok_dm = dm_df.loc[dm_df["status"] == "ok"].copy()
    positive_dm = ok_dm.loc[ok_dm["significant_10pct"] & ok_dm["curve_helps_on_average"]].copy()
    negative_dm = ok_dm.loc[ok_dm["significant_10pct"] & ~ok_dm["curve_helps_on_average"]].copy()

    inflation_wins = ok_inflation.groupby("split")["curve_helps"].sum().to_dict()
    key_rate_wins = ok_key_rate.groupby("split")["curve_helps"].sum().to_dict()

    strongest_inflation = (
        ok_inflation.sort_values("rmse_gain_pct", ascending=False)
        .loc[:, ["split_label", "horizon_months", "model_label", "rmse_gain_pct", "baseline_rmse", "augmented_rmse"]]
        .head(6)
        .copy()
    )
    strongest_key_rate = (
        ok_key_rate.sort_values("rmse_gain_pct", ascending=False)
        .loc[:, ["split_label", "horizon_months", "model_label", "rmse_gain_pct", "baseline_rmse", "augmented_rmse"]]
        .head(6)
        .copy()
    )
    significant_positive_dm = (
        positive_dm
        .sort_values(["target_name", "p_value", "mean_loss_diff"], ascending=[True, True, False])
        .loc[:, ["target_label", "split_label", "horizon_months", "model_label", "mean_loss_diff", "dm_stat", "p_value"]]
        .head(10)
        .copy()
    )

    for frame in (strongest_inflation, strongest_key_rate):
        if not frame.empty:
            frame["rmse_gain_pct"] = frame["rmse_gain_pct"].map(_format_pct)
            frame["baseline_rmse"] = frame["baseline_rmse"].map(_format_float)
            frame["augmented_rmse"] = frame["augmented_rmse"].map(_format_float)
    if not significant_positive_dm.empty:
        significant_positive_dm["mean_loss_diff"] = significant_positive_dm["mean_loss_diff"].map(_format_float)
        significant_positive_dm["dm_stat"] = significant_positive_dm["dm_stat"].map(_format_float)
        significant_positive_dm["p_value"] = significant_positive_dm["p_value"].map(_format_float)

    inflation_positive_dm = positive_dm.loc[positive_dm["target_name"] == "inflation"].copy()
    key_rate_positive_dm = positive_dm.loc[positive_dm["target_name"] == "key_rate"].copy()
    tau1_ok = key_rate_tau1_df.loc[key_rate_tau1_df["status"] == "ok"].copy()
    tau1_positive_dm = key_rate_tau1_dm_df.loc[
        (key_rate_tau1_dm_df["status"] == "ok")
        & key_rate_tau1_dm_df["significant_10pct"]
        & key_rate_tau1_dm_df["curve_helps_on_average"]
    ].copy()
    tau1_wins = tau1_ok.groupby("split")["curve_helps"].sum().to_dict()
    strongest_tau1 = (
        tau1_ok.sort_values("rmse_gain_pct", ascending=False)
        .loc[:, ["split_label", "horizon_months", "model_label", "rmse_gain_pct", "baseline_rmse", "augmented_rmse"]]
        .head(6)
        .copy()
    )
    benchmark_summary = curve_benchmark_summary_df.copy()
    market_control_summary = market_control_summary_df.copy()
    if not benchmark_summary.empty:
        benchmark_summary["mean_rmse_gain_pct"] = benchmark_summary["mean_rmse_gain_pct"].map(_format_pct)
        benchmark_summary["median_rmse_gain_pct"] = benchmark_summary["median_rmse_gain_pct"].map(_format_pct)
    if not market_control_summary.empty:
        market_control_summary["mean_rmse_gain_pct"] = market_control_summary["mean_rmse_gain_pct"].map(_format_pct)
        market_control_summary["median_rmse_gain_pct"] = market_control_summary["median_rmse_gain_pct"].map(_format_pct)

    post_simple_row = curve_benchmark_summary_df.loc[
        (curve_benchmark_summary_df["benchmark_name"] == "simple_spreads")
        & (curve_benchmark_summary_df["split"] == "post_2022_02")
    ]
    post_official_row = curve_benchmark_summary_df.loc[
        (curve_benchmark_summary_df["benchmark_name"] == "official_shape")
        & (curve_benchmark_summary_df["split"] == "post_2022_02")
    ]
    market_control_ok = market_control_comparison_df.loc[market_control_comparison_df["status"] == "ok"].copy()
    market_control_positive_dm = market_control_dm_df.loc[
        (market_control_dm_df["status"] == "ok")
        & market_control_dm_df["significant_10pct"]
        & market_control_dm_df["curve_helps_on_average"]
    ].copy()
    market_control_wins = market_control_ok.groupby("split")["curve_helps"].sum().to_dict()
    strongest_market_controls = (
        market_control_ok.sort_values("rmse_gain_pct", ascending=False)
        .loc[:, ["split_label", "horizon_months", "model_label", "rmse_gain_pct", "baseline_rmse", "augmented_rmse"]]
        .head(8)
        .copy()
    )
    market_control_pairs_by_split = {
        split_key: {
            (int(row.horizon_months), str(row.model_name))
            for row in frame.loc[:, ["horizon_months", "model_name"]].itertuples(index=False)
        }
        for split_key, frame in market_control_ok.groupby("split")
    }
    comparable_market_control_pairs = sorted(
        market_control_pairs_by_split.get("post_2022_02", set())
        & market_control_pairs_by_split.get("stabilization_2024_plus", set())
    )
    comparable_market_control_horizons = sorted({horizon for horizon, _ in comparable_market_control_pairs})

    def describe_positive_cases(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "статистически значимых улучшений не найдено"
        parts = []
        for row in frame.loc[:, ["split_label", "horizon_months", "model_label"]].head(4).itertuples(index=False):
            parts.append(f"{row.split_label}, {row.horizon_months} мес., {row.model_label}")
        return "; ".join(parts)

    if not strongest_tau1.empty:
        strongest_tau1["rmse_gain_pct"] = strongest_tau1["rmse_gain_pct"].map(_format_pct)
        strongest_tau1["baseline_rmse"] = strongest_tau1["baseline_rmse"].map(_format_float)
        strongest_tau1["augmented_rmse"] = strongest_tau1["augmented_rmse"].map(_format_float)
    if not strongest_market_controls.empty:
        strongest_market_controls["rmse_gain_pct"] = strongest_market_controls["rmse_gain_pct"].map(_format_pct)
        strongest_market_controls["baseline_rmse"] = strongest_market_controls["baseline_rmse"].map(_format_float)
        strongest_market_controls["augmented_rmse"] = strongest_market_controls["augmented_rmse"].map(_format_float)

    def describe_benchmark_summary(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "нет оценимых комбинаций"
        row = frame.iloc[0]
        return (
            f"выигрыш в `{int(row['benchmark_better'])}` из `{int(row['evaluated_combinations'])}` комбинаций, "
            f"среднее изменение RMSE `{_format_pct(float(row['mean_rmse_gain_pct']))}`, "
            f"статистически значимых улучшений `{int(row['significant_benchmark_improvements'])}`"
        )

    def describe_market_control_split(split_key: str) -> str:
        subset = market_control_ok.loc[
            (market_control_ok["split"] == split_key)
            & market_control_ok.apply(
                lambda row: (int(row["horizon_months"]), str(row["model_name"])) in comparable_market_control_pairs,
                axis=1,
            )
        ]
        if subset.empty:
            return "нет сопоставимых оценок"
        return f"`{int(subset['curve_helps'].sum())}` из `{int(len(subset))}`"

    comparable_horizons_text = ", ".join(str(int(h)) for h in comparable_market_control_horizons) or "нет"

    return "\n".join(
        [
            "# Поток 4. Добавочная прогностическая сила факторов кривой",
            "",
            "## Что было пересчитано",
            f"- Расчёт переисполнен по входам до `{latest_input_date}`.",
            f"- Объединённая панель покрывает период с `{data['date'].min().strftime('%Y-%m-%d')}` по `{latest_input_date}`.",
            f"- Фактическая годовая инфляция доступна до `{inflation_actual_end}`, ключевая ставка до `{key_rate_actual_end}`.",
            f"- Ожидания аналитиков по 2026 году появляются только с `{analyst_first}`, поэтому модели с ожиданиями оцениваются на коротком окне.",
            "",
            "## Постановка",
            "- Для инфляции прогнозируется `cpi_yoy` на горизонтах 1, 3, 6 и 12 месяцев.",
            "- Для ключевой ставки прогнозируется среднее значение `key_rate` за следующие 1, 3, 6 и 12 месяцев.",
            "- Базовые модели сравниваются с расширенными версиями, в которые добавлены `beta0`, `beta1`, `beta2`.",
            "- Официальные месячные макропеременные и наблюдаемые ряды Банка России, публикуемые с задержкой, входят в модели со сдвигом на один месяц назад; рыночные ряды кривой, курса, инфляционной компенсации и результаты макроэкономического опроса аналитиков используются в месяце их публикации.",
            "- Вневыборочные прогнозы строятся регрессиями с расширяющимся окном по трём разрезам: вся выборка, до `2022-02`, после `2022-02`.",
            "",
            "## Краткий итог",
            f"- По инфляции число комбинаций, где RMSE снижается после добавления кривой: `all={int(inflation_wins.get('all', 0))}`, `pre_2022_02={int(inflation_wins.get('pre_2022_02', 0))}`, `post_2022_02={int(inflation_wins.get('post_2022_02', 0))}`.",
            f"- По ключевой ставке число комбинаций со снижением RMSE: `all={int(key_rate_wins.get('all', 0))}`, `pre_2022_02={int(key_rate_wins.get('pre_2022_02', 0))}`, `post_2022_02={int(key_rate_wins.get('post_2022_02', 0))}`.",
            f"- На уровне значимости `10%` тест Диболда-Мариано подтверждает выигрыш кривой в `{int(len(positive_dm))}` комбинациях из `{int(len(ok_dm))}` оценённых; ещё в `{int(len(negative_dm))}` комбинациях кривая статистически ухудшает прогноз.",
            f"- Для инфляции статистически значимые улучшения сосредоточены в следующих случаях: {describe_positive_cases(inflation_positive_dm)}.",
            f"- Для средней ключевой ставки статистически значимые улучшения сосредоточены в следующих случаях: {describe_positive_cases(key_rate_positive_dm)}.",
            f"- Дополнительная проверка `NS` против `NS + ln(tau1)` для средней ключевой ставки даёт число комбинаций со снижением RMSE: `all={int(tau1_wins.get('all', 0))}`, `pre_2022_02={int(tau1_wins.get('pre_2022_02', 0))}`, `post_2022_02={int(tau1_wins.get('post_2022_02', 0))}`.",
            f"- Статистически значимые случаи, где добавление `tau1` улучшает прогноз средней ключевой ставки: {describe_positive_cases(tau1_positive_dm)}.",
            f"- Сравнение `NS` с простыми спредами авторской кривой после `2022-02`: {describe_benchmark_summary(post_simple_row)}.",
            f"- Сравнение `NS` с простыми характеристиками официальной КБД после `2022-02`: {describe_benchmark_summary(post_official_row)}.",
            f"- Для корректного сравнения режимов в блоке с долговыми и инвестиционными контролями используются общие горизонты `{comparable_horizons_text}` месяцев.",
            f"- После `2022-02` на этих сопоставимых горизонтах добавление `beta0,beta1,beta2` снижает RMSE в {describe_market_control_split('post_2022_02')} комбинаций.",
            f"- В устойчивом режиме `2024-2026` тот же результат наблюдается в {describe_market_control_split('stabilization_2024_plus')} комбинаций.",
            f"- Статистически значимые случаи, где кривая улучшает прогноз даже после таких контролей: {describe_positive_cases(market_control_positive_dm)}.",
            "",
            "## Сильнейшие случаи по инфляции",
            _markdown_table(
                strongest_inflation,
                ["split_label", "horizon_months", "model_label", "rmse_gain_pct", "baseline_rmse", "augmented_rmse"],
                ["Разрез", "Горизонт", "Модель", "Снижение RMSE", "RMSE база", "RMSE с кривой"],
            ),
            "## Сильнейшие случаи по ключевой ставке",
            _markdown_table(
                strongest_key_rate,
                ["split_label", "horizon_months", "model_label", "rmse_gain_pct", "baseline_rmse", "augmented_rmse"],
                ["Разрез", "Горизонт", "Модель", "Снижение RMSE", "RMSE база", "RMSE с кривой"],
            ),
            "## Положительные результаты теста Диболда-Мариано",
            _markdown_table(
                significant_positive_dm,
                ["target_label", "split_label", "horizon_months", "model_label", "mean_loss_diff", "dm_stat", "p_value"],
                ["Цель", "Разрез", "Горизонт", "Модель", "Среднее d", "DM", "p-value"],
            ),
            "## Где `tau1` усиливает прогноз средней ключевой ставки",
            _markdown_table(
                strongest_tau1,
                ["split_label", "horizon_months", "model_label", "rmse_gain_pct", "baseline_rmse", "augmented_rmse"],
                ["Разрез", "Горизонт", "Модель", "Снижение RMSE", "RMSE NS", "RMSE NS+tau1"],
            ),
            "## Сопоставление NS с простыми альтернативами",
            _markdown_table(
                benchmark_summary,
                [
                    "benchmark_label",
                    "split_label",
                    "evaluated_combinations",
                    "benchmark_better",
                    "mean_rmse_gain_pct",
                    "significant_benchmark_improvements",
                ],
                ["Альтернатива", "Разрез", "Комбинаций", "Альтернатива лучше", "Средний выигрыш RMSE", "Значимых улучшений"],
            ),
            "## Проверка после учета долговых и инвестиционных показателей",
            _markdown_table(
                market_control_summary,
                [
                    "split_label",
                    "model_label",
                    "evaluated_combinations",
                    "curve_better",
                    "mean_rmse_gain_pct",
                    "significant_curve_improvements",
                ],
                ["Разрез", "Базовая модель", "Комбинаций", "Кривая лучше", "Средний выигрыш RMSE", "Значимых улучшений"],
            ),
            "## Сильнейшие случаи после учета долговых и инвестиционных показателей",
            _markdown_table(
                strongest_market_controls,
                ["split_label", "horizon_months", "model_label", "rmse_gain_pct", "baseline_rmse", "augmented_rmse"],
                ["Разрез", "Горизонт", "Базовая модель", "Снижение RMSE", "RMSE без кривой", "RMSE с кривой"],
            ),
            "## Ограничения",
            "- До февраля 2022 года модели с ожиданиями аналитиков не оцениваются: в панели нет соответствующих наблюдений.",
            "- Для инфляции используется `cpi_yoy`, потому что `cpi_mom` в доступной панели обрывается раньше и не даёт полного обновления до 2026 года.",
            "- Часть результатов на длинных горизонтах опирается на короткие вневыборочные окна, особенно в послешоковом периоде и для моделей с ожиданиями.",
            "- Переменная `tau1` в дополнительной проверке трактуется как вспомогательный параметр формы кривой и не заменяет базовую интерпретацию через `beta0`, `beta1`, `beta2`.",
            "- Простые характеристики официальной КБД используются только как внешний ориентир; повторная калибровка Нельсона--Сигеля по официальной кривой не выполняется.",
            "- В моделях с инфляцией, долгом сектора государственного управления и долей нерезидентов используется месячный лаг публикации, чтобы не опираться на информацию, еще не доступную в момент прогноза; для опроса аналитиков такой лаг не вводится, поскольку результаты публикуются в том же месяце.",
            "- Наблюдаемые долговые и инвестиционные показатели берутся из официальных месячных рядов Банка России; они полезны как контроль, но не дают прямого разложения всей сроковой премии по компонентам.",
            "- Полная спецификация со всеми наблюдаемыми контролями в режиме `2024-2026` не оценивается ни на одном горизонте из-за короткого перекрытия этого подпериода с 10-летним рядом рыночной инфляционной компенсации.",
            "",
            "## Файлы",
            f"- `{summary_metadata['artifacts']['inflation_model_comparison_csv']}`",
            f"- `{summary_metadata['artifacts']['key_rate_model_comparison_csv']}`",
            f"- `{summary_metadata['artifacts']['dm_tests_csv']}`",
            f"- `{summary_metadata['artifacts']['key_rate_tau1_increment_csv']}`",
            f"- `{summary_metadata['artifacts']['key_rate_tau1_dm_tests_csv']}`",
            f"- `{summary_metadata['artifacts']['key_rate_curve_benchmark_comparison_csv']}`",
            f"- `{summary_metadata['artifacts']['key_rate_curve_benchmark_dm_tests_csv']}`",
            f"- `{summary_metadata['artifacts']['key_rate_curve_benchmark_summary_csv']}`",
            f"- `{summary_metadata['artifacts']['key_rate_market_controls_comparison_csv']}`",
            f"- `{summary_metadata['artifacts']['key_rate_market_controls_dm_tests_csv']}`",
            f"- `{summary_metadata['artifacts']['key_rate_market_controls_summary_csv']}`",
            f"- `{summary_metadata['artifacts']['summary_json']}`",
        ]
    ) + "\n"


def run_predictive_power(
    core_path: Path | str = DEFAULT_CORE_PATH,
    macro_path: Path | str = DEFAULT_MACRO_PATH,
    official_curve_path: Path | str = DEFAULT_OFFICIAL_CURVE_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    findings_path: Path | str = DEFAULT_FINDINGS_PATH,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    findings_path = Path(findings_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_analysis_panel(core_path=core_path, macro_path=macro_path, official_curve_path=official_curve_path)
    inflation_df, inflation_dm = evaluate_target_models(data=data, target_spec=TARGET_SPECS[0])
    key_rate_df, key_rate_dm = evaluate_target_models(data=data, target_spec=TARGET_SPECS[1])
    key_rate_tau1_df, key_rate_tau1_dm = evaluate_tau1_increment_for_key_rate(data=data)
    curve_benchmark_df, curve_benchmark_dm_df = evaluate_curve_benchmarks_for_key_rate(data=data)
    curve_benchmark_summary_df = summarise_curve_benchmark_results(
        comparison_df=curve_benchmark_df,
        dm_df=curve_benchmark_dm_df,
    )
    market_control_df, market_control_dm_df = evaluate_market_control_models_for_key_rate(data=data)
    market_control_summary_df = summarise_market_control_results(
        comparison_df=market_control_df,
        dm_df=market_control_dm_df,
    )
    dm_df = pd.concat([inflation_dm, key_rate_dm], ignore_index=True)

    inflation_output = output_dir / "inflation_model_comparison.csv"
    key_rate_output = output_dir / "key_rate_model_comparison.csv"
    dm_output = output_dir / "dm_tests.csv"
    key_rate_tau1_output = output_dir / "key_rate_tau1_increment.csv"
    key_rate_tau1_dm_output = output_dir / "key_rate_tau1_dm_tests.csv"
    curve_benchmark_output = output_dir / "key_rate_curve_benchmark_comparison.csv"
    curve_benchmark_dm_output = output_dir / "key_rate_curve_benchmark_dm_tests.csv"
    curve_benchmark_summary_output = output_dir / "key_rate_curve_benchmark_summary.csv"
    market_control_output = output_dir / "key_rate_market_controls_comparison.csv"
    market_control_dm_output = output_dir / "key_rate_market_controls_dm_tests.csv"
    market_control_summary_output = output_dir / "key_rate_market_controls_summary.csv"
    summary_json_output = output_dir / "predictive_summary.json"

    inflation_df.to_csv(inflation_output, index=False)
    key_rate_df.to_csv(key_rate_output, index=False)
    dm_df.to_csv(dm_output, index=False)
    key_rate_tau1_df.to_csv(key_rate_tau1_output, index=False)
    key_rate_tau1_dm.to_csv(key_rate_tau1_dm_output, index=False)
    curve_benchmark_df.to_csv(curve_benchmark_output, index=False)
    curve_benchmark_dm_df.to_csv(curve_benchmark_dm_output, index=False)
    curve_benchmark_summary_df.to_csv(curve_benchmark_summary_output, index=False)
    market_control_df.to_csv(market_control_output, index=False)
    market_control_dm_df.to_csv(market_control_dm_output, index=False)
    market_control_summary_df.to_csv(market_control_summary_output, index=False)

    summary_metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_files": {
            "core_path": str(Path(core_path)),
            "macro_path": str(Path(macro_path)),
            "official_curve_path": str(Path(official_curve_path)),
        },
        "input_coverage": {
            "panel_start": data["date"].min().strftime("%Y-%m-%d"),
            "panel_end": data["date"].max().strftime("%Y-%m-%d"),
            "inflation_actual_end": data.loc[data["cpi_yoy"].notna(), "date"].max().strftime("%Y-%m-%d"),
            "key_rate_actual_end": data.loc[data["key_rate"].notna(), "date"].max().strftime("%Y-%m-%d"),
            "expectations_start": data.loc[data["analyst_cpi_2026"].notna(), "date"].min().strftime("%Y-%m-%d"),
        },
        "artifacts": {
            "inflation_model_comparison_csv": str(inflation_output),
            "key_rate_model_comparison_csv": str(key_rate_output),
            "dm_tests_csv": str(dm_output),
            "key_rate_tau1_increment_csv": str(key_rate_tau1_output),
            "key_rate_tau1_dm_tests_csv": str(key_rate_tau1_dm_output),
            "key_rate_curve_benchmark_comparison_csv": str(curve_benchmark_output),
            "key_rate_curve_benchmark_dm_tests_csv": str(curve_benchmark_dm_output),
            "key_rate_curve_benchmark_summary_csv": str(curve_benchmark_summary_output),
            "key_rate_market_controls_comparison_csv": str(market_control_output),
            "key_rate_market_controls_dm_tests_csv": str(market_control_dm_output),
            "key_rate_market_controls_summary_csv": str(market_control_summary_output),
            "summary_json": str(summary_json_output),
            "findings_markdown": str(findings_path),
        },
        "row_counts": {
            "inflation_model_comparison": int(len(inflation_df)),
            "key_rate_model_comparison": int(len(key_rate_df)),
            "dm_tests": int(len(dm_df)),
            "key_rate_tau1_increment": int(len(key_rate_tau1_df)),
            "key_rate_tau1_dm_tests": int(len(key_rate_tau1_dm)),
            "key_rate_curve_benchmark_comparison": int(len(curve_benchmark_df)),
            "key_rate_curve_benchmark_dm_tests": int(len(curve_benchmark_dm_df)),
            "key_rate_curve_benchmark_summary": int(len(curve_benchmark_summary_df)),
            "key_rate_market_controls_comparison": int(len(market_control_df)),
            "key_rate_market_controls_dm_tests": int(len(market_control_dm_df)),
            "key_rate_market_controls_summary": int(len(market_control_summary_df)),
        },
        "targets": {
            target.name: {
                "label": target.label,
                "source_column": target.source_column,
                "target_type": target.target_type,
                "horizons": list(HORIZONS),
                "note": target.actual_notes,
            }
            for target in TARGET_SPECS
        },
    }
    summary_json_output.write_text(json.dumps(summary_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    findings_markdown = build_findings_markdown(
        data=data,
        inflation_df=inflation_df,
        key_rate_df=key_rate_df,
        dm_df=dm_df,
        key_rate_tau1_df=key_rate_tau1_df,
        key_rate_tau1_dm_df=key_rate_tau1_dm,
        curve_benchmark_summary_df=curve_benchmark_summary_df,
        market_control_comparison_df=market_control_df,
        market_control_dm_df=market_control_dm_df,
        market_control_summary_df=market_control_summary_df,
        summary_metadata=summary_metadata,
    )
    findings_path.write_text(findings_markdown, encoding="utf-8")

    return {
        "data": data,
        "inflation_model_comparison": inflation_df,
        "key_rate_model_comparison": key_rate_df,
        "dm_tests": dm_df,
        "key_rate_tau1_increment": key_rate_tau1_df,
        "key_rate_tau1_dm_tests": key_rate_tau1_dm,
        "key_rate_curve_benchmark_comparison": curve_benchmark_df,
        "key_rate_curve_benchmark_dm_tests": curve_benchmark_dm_df,
        "key_rate_curve_benchmark_summary": curve_benchmark_summary_df,
        "key_rate_market_controls_comparison": market_control_df,
        "key_rate_market_controls_dm_tests": market_control_dm_df,
        "key_rate_market_controls_summary": market_control_summary_df,
        "summary_metadata": summary_metadata,
        "findings_path": findings_path,
    }
