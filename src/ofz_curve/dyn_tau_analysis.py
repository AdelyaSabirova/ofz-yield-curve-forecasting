from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from nelson_siegel_svensson import NelsonSiegelCurve
from tqdm.auto import tqdm

from .paths import DYNAMIC_TAU_PARAMS_CSV


def load_dyn_params_df(csv_path: str = str(DYNAMIC_TAU_PARAMS_CSV)) -> pd.DataFrame:
    """
    Загружает результаты динамической калибровки tau1 из CSV.
    Ожидаются колонки: date, beta0, beta1, beta2, tau1_opt, rmse_pp, rmse_pp_static.
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except FileNotFoundError:
        print(f"❌ Файл с динамическими параметрами не найден: {csv_path}")
        return pd.DataFrame()

    if df.empty:
        print(f"⚠️ Файл {csv_path} пуст, динамические параметры недоступны")
        return pd.DataFrame()

    if "date" not in df.columns:
        print(f"⚠️ В {csv_path} нет колонки 'date', динамические параметры недоступны")
        return pd.DataFrame()

    return df.sort_values("date").set_index("date")


def run_dyn_tau_eda(dyn_params_df: pd.DataFrame) -> dict:
    """
    EDA по динамическому tau1: график tau1, сравнение RMSE и доли улучшений.
    Возвращает словарь с агрегированными метриками.
    """
    if dyn_params_df is None or dyn_params_df.empty:
        print("⚠️ Нет результатов динамической калибровки (dyn_params_df), EDA пропущен")
        return {}

    plt.figure(figsize=(14, 6))
    plt.plot(dyn_params_df.index, dyn_params_df["tau1_opt"], label="tau1 (dynamic)", color="navy")
    plt.axvline(pd.Timestamp("2020-03-01"), color="crimson", linestyle="--", label="COVID-19 (2020)")
    plt.axvline(pd.Timestamp("2022-02-24"), color="orange", linestyle="--", label="События 2022")
    plt.title("Динамическая оценка tau1 по дням")
    plt.xlabel("Дата")
    plt.ylabel("tau1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    rmse_daily = dyn_params_df[["rmse_pp", "rmse_pp_static"]].dropna()
    rmse_mean_dyn = rmse_daily["rmse_pp"].mean()
    rmse_mean_static = rmse_daily["rmse_pp_static"].mean()

    print(f"Средняя ежедневная RMSE (динамическая tau1): {rmse_mean_dyn:.4f} п.п.")
    print(f"Средняя ежедневная RMSE (статическая фиксированная tau1): {rmse_mean_static:.4f} п.п.")

    plt.figure(figsize=(14, 5))
    plt.plot(rmse_daily.index, rmse_daily["rmse_pp"], label="Dynamic tau1 RMSE", color="green")
    plt.plot(
        rmse_daily.index,
        rmse_daily["rmse_pp_static"],
        label="Static fixed tau1 RMSE",
        color="gray",
        alpha=0.7,
    )
    plt.title("Ежедневная RMSE подгонки: динамическая vs статическая")
    plt.xlabel("Дата")
    plt.ylabel("RMSE, п.п.")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    improvement_share = (rmse_daily["rmse_pp"] <= rmse_daily["rmse_pp_static"]).mean()
    print(f"Дней, где динамическая tau1 лучше/не хуже статической: {improvement_share:.1%}")

    return {
        "rmse_mean_dyn": rmse_mean_dyn,
        "rmse_mean_static": rmse_mean_static,
        "improvement_share": improvement_share,
    }


def forecast_dynamic_tau1(
    dyn_params_df: pd.DataFrame,
    df_filtered: pd.DataFrame,
    horizons_days,
    out_of_sample_ratio: float,
    rolling_window: int = 5,
    eval_maturities_years=(0.25, 1.0, 3.0, 5.0, 10.0),
):
    """
    Делает OOS AR(1) прогноз факторов beta0/1/2 и ln(tau1),
    строит прогноз кривой доходности и возвращает:
    (dyn_factor_forecasts_df, yield_forecasts_dyn_df, rmse_dyn_yield_df).
    """
    if dyn_params_df is None or dyn_params_df.empty:
        print("⚠️ Нет dyn_params_df для прогнозирования")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if df_filtered is None or df_filtered.empty:
        print("⚠️ df_filtered пуст, прогноз по динамическому tau1 невозможен")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    tau1_series = dyn_params_df["tau1_opt"].dropna()
    ln_tau1_series = np.log(tau1_series)
    ln_tau1_smoothed = ln_tau1_series.rolling(rolling_window, min_periods=1).mean()

    factors_dyn_fc_df = dyn_params_df[["beta0", "beta1", "beta2"]].copy()
    factors_dyn_fc_df["ln_tau1"] = ln_tau1_smoothed
    factors_dyn_fc_df.dropna(inplace=True)
    print(f"Длина ряда для динамического прогноза: {len(factors_dyn_fc_df)}")

    dyn_factor_fc_list = []
    n_obs_total_dyn = len(factors_dyn_fc_df)
    n_train_initial_dyn = int(n_obs_total_dyn * out_of_sample_ratio)

    for t_idx in tqdm(range(n_train_initial_dyn, n_obs_total_dyn), desc="OOS AR(1) Dyn factors"):
        current_train = factors_dyn_fc_df.iloc[:t_idx]
        date_fc_made = factors_dyn_fc_df.index[t_idx - 1]
        for h_days in horizons_days:
            target_idx = t_idx + h_days - 1
            if target_idx >= n_obs_total_dyn:
                continue
            target_date = factors_dyn_fc_df.index[target_idx]
            for factor in ["beta0", "beta1", "beta2", "ln_tau1"]:
                Y = current_train[factor].iloc[h_days:].values
                X_raw = current_train[factor].iloc[:-h_days].values
                forecast_val = np.nan
                if len(X_raw) >= 2:
                    model = sm.OLS(Y, sm.add_constant(X_raw)).fit()
                    cur_val = factors_dyn_fc_df[factor].iloc[t_idx - 1]
                    forecast_val = model.predict([1, cur_val])[0]
                dyn_factor_fc_list.append(
                    {
                        "forecast_made_on_date": date_fc_made,
                        "forecast_target_date": target_date,
                        "horizon_days": h_days,
                        "factor": factor,
                        "forecast_value_dyn_ar1": forecast_val,
                        "actual_value": factors_dyn_fc_df.loc[target_date, factor],
                    }
                )

    dyn_factor_forecasts_df = pd.DataFrame(dyn_factor_fc_list)
    dyn_factor_forecasts_df.dropna(subset=["forecast_value_dyn_ar1", "actual_value"], inplace=True)
    print(f"Получено {len(dyn_factor_forecasts_df)} OOS прогнозов факторов (beta + ln_tau1)")

    oos_yield_dyn_results = []
    grouped_dyn = dyn_factor_forecasts_df.groupby(["forecast_target_date", "horizon_days"])

    for (target_date, horizon), group_fc in tqdm(grouped_dyn, desc="OOS Yield (Dynamic tau1)"):
        factor_map = {}
        for fac in ["beta0", "beta1", "beta2", "ln_tau1"]:
            row = group_fc[group_fc["factor"] == fac]
            if row.empty or pd.isna(row["forecast_value_dyn_ar1"].iloc[0]):
                factor_map = {}
                break
            factor_map[fac] = row["forecast_value_dyn_ar1"].iloc[0]

        if len(factor_map) != 4:
            continue

        tau_fc = np.exp(factor_map["ln_tau1"])
        try:
            dyn_ns_curve = NelsonSiegelCurve(
                beta0=factor_map["beta0"],
                beta1=factor_map["beta1"],
                beta2=factor_map["beta2"],
                tau=tau_fc,
            )
        except Exception:
            continue

        actual_market = df_filtered[df_filtered["TRADEDATE"] == target_date]
        if actual_market.empty:
            continue
        actual_ttms = actual_market["TTM_years"].values
        actual_yields = actual_market["YIELDCLOSE"].values
        sort_idx = np.argsort(actual_ttms)
        actual_ttms, actual_yields = actual_ttms[sort_idx], actual_yields[sort_idx]

        for mat_y in eval_maturities_years:
            res_row = {
                "forecast_target_date": target_date,
                "horizon_days": horizon,
                "maturity_years": mat_y,
            }
            res_row["yield_forecast_ns_dyn"] = dyn_ns_curve(mat_y) * 100

            actual_val_interp = np.nan
            if len(actual_ttms) >= 2:
                actual_val_interp = np.interp(
                    mat_y, actual_ttms, actual_yields, left=actual_yields[0], right=actual_yields[-1]
                )
            elif len(actual_ttms) == 1:
                actual_val_interp = actual_yields[0] if np.isclose(mat_y, actual_ttms[0]) else np.nan

            res_row["actual_yield"] = actual_val_interp
            oos_yield_dyn_results.append(res_row)

    yield_forecasts_dyn_df = pd.DataFrame(oos_yield_dyn_results)
    if not yield_forecasts_dyn_df.empty:
        yield_forecasts_dyn_df["error_ns_dyn"] = (
            yield_forecasts_dyn_df["actual_yield"] - yield_forecasts_dyn_df["yield_forecast_ns_dyn"]
        )
        yield_forecasts_dyn_df.dropna(subset=["actual_yield"], inplace=True)
        print(f"Получено {len(yield_forecasts_dyn_df)} OOS прогнозов доходностей (динамическая tau1)")
    else:
        print("Не удалось получить прогнозы доходностей для динамической tau1")

    rmse_dyn_yield_list = []
    if not yield_forecasts_dyn_df.empty:
        for horizon_val in horizons_days:
            for mat_y_val in eval_maturities_years:
                subset = yield_forecasts_dyn_df[
                    (yield_forecasts_dyn_df["horizon_days"] == horizon_val)
                    & (yield_forecasts_dyn_df["maturity_years"] == mat_y_val)
                ]
                rmse_val = np.nan
                n_obs = len(subset)
                if n_obs > 0:
                    errs = subset["error_ns_dyn"].dropna()
                    if len(errs) > 0:
                        rmse_val = np.sqrt(np.mean(errs**2))
                rmse_dyn_yield_list.append(
                    {"horizon_days": horizon_val, "maturity_years": mat_y_val, "n_obs": n_obs, "RMSE_NS_DYNAMIC": rmse_val}
                )

    rmse_dyn_yield_df = pd.DataFrame(rmse_dyn_yield_list)
    print("RMSE динамического NS (п.п.):")
    if not rmse_dyn_yield_df.empty:
        display(rmse_dyn_yield_df)
    else:
        print("RMSE таблица пуста")

    return dyn_factor_forecasts_df, yield_forecasts_dyn_df, rmse_dyn_yield_df
