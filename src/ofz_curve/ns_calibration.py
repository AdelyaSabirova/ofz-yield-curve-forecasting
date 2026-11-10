"""
Функции для калибровки модели Nelson-Siegel
"""

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from .config import *


def calibrate_ns_for_tau1(df_filtered, tau1_value, forecast_horizons_days=FORECAST_HORIZONS_DAYS,
                         out_of_sample_start_ratio=OUT_OF_SAMPLE_START_RATIO):
    """
    Калибрует модель Nelson-Siegel для заданного значения tau1 и выполняет прогнозирование факторов

    Args:
        df_filtered (pd.DataFrame): Отфильтрованные данные ОФЗ
        tau1_value (float): Значение параметра tau1
        forecast_horizons_days (list): Горизонты прогнозирования в днях
        out_of_sample_start_ratio (float): Доля данных для обучения

    Returns:
        tuple: (params_successful, ar_factor_forecasts_df, yield_forecasts_df, rmse_df)
    """
    print(f"Анализ для tau1 = {tau1_value}")

    results_list = []
    unique_dates = sorted(df_filtered['TRADEDATE'].unique())

    # Калибровка NS по дням
    for current_date in tqdm(unique_dates, desc=f"Калибровка NS (tau1={tau1_value})"):
        df_today = df_filtered[df_filtered['TRADEDATE'] == current_date].copy()

        if len(df_today) < MIN_DATA_POINTS_FOR_FIT:
            results_list.append({
                'date': current_date,
                'beta0': np.nan, 'beta1': np.nan, 'beta2': np.nan,
                'tau1': tau1_value, 'success': False,
                'error_message': f'Too few data points ({len(df_today)})'
            })
            continue

        ttm_years = df_today['TTM_years'].values
        yields = df_today['YIELDCLOSE'].values / 100.0

        tau_div_tau1 = ttm_years / tau1_value
        exp_term = np.exp(-tau_div_tau1)

        # Нагрузки на факторы
        load_beta1 = np.where(
            np.abs(tau_div_tau1) < 1e-6,
            1.0 - tau_div_tau1 / 2.0 + tau_div_tau1**2 / 6.0,
            (1 - exp_term) / tau_div_tau1
        )
        load_beta2 = load_beta1 - exp_term

        # Матрица регрессоров
        X = np.vstack([np.ones_like(ttm_years), load_beta1, load_beta2]).T

        beta_params, residuals, rank, singular_values = np.linalg.lstsq(X, yields, rcond=None)

        if rank < X.shape[1]:
            results_list.append({
                'date': current_date,
                'beta0': np.nan, 'beta1': np.nan, 'beta2': np.nan,
                'tau1': tau1_value, 'success': False,
                'error_message': f'Rank deficiency in regressors (rank={rank})'
            })
            continue

        results_list.append({
            'date': current_date,
            'beta0': beta_params[0],
            'beta1': beta_params[1],
            'beta2': beta_params[2],
            'tau1': tau1_value,
            'success': True,
            'error_message': None
        })

    # Постобработка результатов калибровки
    params_df_raw = pd.DataFrame(results_list)
    params_df_raw.set_index('date', inplace=True)

    successful_fits_count = params_df_raw['success'].sum()
    print(f"Калибровка tau1 = {tau1_value}: {successful_fits_count} успешных дат")

    # Отфильтровываем только успешные
    params_successful = params_df_raw[params_df_raw['success']].drop(columns=['success', 'error_message']).copy()
    params_successful.dropna(subset=['beta0', 'beta1', 'beta2'], inplace=True)

    print(f"Доступно {len(params_successful)} дат для анализа (tau1 = {tau1_value})")

    # Прогнозирование факторов AR(1)
    factors_df = params_successful[['beta0', 'beta1', 'beta2']].copy()
    ar_factor_forecasts_oos_list = []

    n_obs_total = len(factors_df)
    n_train_initial = int(n_obs_total * out_of_sample_start_ratio)
    n_test = n_obs_total - n_train_initial

    max_h_forecast = max(forecast_horizons_days)

    # Рекурсивное out-of-sample прогнозирование факторов
    for t_idx in tqdm(range(n_train_initial, n_obs_total), desc=f"OOS AR(1) Factor Forecasting (tau1={tau1_value})"):
        current_train_set = factors_df.iloc[:t_idx]

        # Для каждого горизонта прогнозирования
        for h_days in forecast_horizons_days:
            target_forecast_idx = t_idx + h_days - 1
            if target_forecast_idx >= n_obs_total:
                continue

            forecast_target_date_actual = factors_df.index[target_forecast_idx]

            # Для каждого фактора
            for factor_name in factors_df.columns:
                Y_ar_h_step = current_train_set[factor_name].iloc[h_days:].values
                X_ar_h_step_raw = current_train_set[factor_name].iloc[:-h_days].values

                forecast_val = np.nan
                if len(X_ar_h_step_raw) >= 2:
                    import statsmodels.api as sm
                    X_ar_h_step = sm.add_constant(X_ar_h_step_raw)
                    ar_model_direct_h = sm.OLS(Y_ar_h_step, X_ar_h_step).fit()
                    current_factor_val_for_pred = factors_df[factor_name].iloc[t_idx-1]
                    forecast_val = ar_model_direct_h.predict([1, current_factor_val_for_pred])[0]

                ar_factor_forecasts_oos_list.append({
                    'forecast_made_on_date': factors_df.index[t_idx-1],
                    'forecast_target_date': forecast_target_date_actual,
                    'horizon_days': h_days,
                    'factor': factor_name,
                    'forecast_value_ar1': forecast_val,
                    'actual_value': factors_df.loc[forecast_target_date_actual, factor_name],
                    'tau1': tau1_value
                })

    ar_factor_forecasts_df = pd.DataFrame(ar_factor_forecasts_oos_list)
    ar_factor_forecasts_df.dropna(subset=['forecast_value_ar1', 'actual_value'], inplace=True)

    # Прогнозирование доходности
    oos_yield_forecast_results_list = []
    ar_fc_df = ar_factor_forecasts_df.rename(columns={'forecast_value_ar1': 'forecast_beta_ar1'})

    grouped_fc = ar_fc_df.groupby(['forecast_target_date', 'horizon_days'])
    current_tau1_for_ns_curves = tau1_value
    eval_maturities_years = np.array(EVAL_MATURITIES_YEARS)

    from nelson_siegel_svensson import NelsonSiegelCurve

    for (target_date_of_yield, horizon), group_fc in tqdm(grouped_fc, desc=f"OOS Yield Forecasting (tau1={tau1_value})"):
        beta_hats_ar1 = {}

        # Собираем прогнозы всех факторов
        for factor_name in ['beta0', 'beta1', 'beta2']:
            factor_row = group_fc[group_fc['factor'] == factor_name]
            if factor_row.empty or pd.isna(factor_row['forecast_beta_ar1'].iloc[0]):
                beta_hats_ar1 = {}
                break
            beta_hats_ar1[factor_name] = factor_row['forecast_beta_ar1'].iloc[0]

        if len(beta_hats_ar1) != 3:
            continue

        # Создаем кривую НС с прогнозными факторами
        predicted_ns_curve = NelsonSiegelCurve(
            beta0=beta_hats_ar1['beta0'],
            beta1=beta_hats_ar1['beta1'],
            beta2=beta_hats_ar1['beta2'],
            tau=current_tau1_for_ns_curves
        )

        # Получаем фактические данные
        actual_market_data_today = df_filtered[df_filtered['TRADEDATE'] == target_date_of_yield]

        actual_ttms_today = actual_market_data_today['TTM_years'].values
        actual_yields_today = actual_market_data_today['YIELDCLOSE'].values
        sort_idx = np.argsort(actual_ttms_today)
        actual_ttms_sorted, actual_yields_sorted = actual_ttms_today[sort_idx], actual_yields_today[sort_idx]

        # Прогноз для каждого срока погашения
        for mat_y in eval_maturities_years:
            result_row_dict = {
                'forecast_target_date': target_date_of_yield,
                'horizon_days': horizon,
                'maturity_years': mat_y,
                'tau1': tau1_value
            }

            # Прогноз НС
            result_row_dict['yield_forecast_ns_ar1'] = predicted_ns_curve(mat_y) * 100

            # Фактическое значение (интерполяция)
            actual_val_interp = np.nan
            if len(actual_ttms_sorted) >= 2:
                actual_val_interp = np.interp(mat_y, actual_ttms_sorted, actual_yields_sorted,
                                              left=actual_yields_sorted[0], right=actual_yields_sorted[-1])
            elif len(actual_ttms_sorted) == 1:
                actual_val_interp = actual_yields_sorted[0] if np.isclose(mat_y, actual_ttms_sorted[0]) else np.nan

            result_row_dict['actual_yield'] = actual_val_interp

            oos_yield_forecast_results_list.append(result_row_dict)

    yield_forecasts_df = pd.DataFrame(oos_yield_forecast_results_list)

    # Расчет ошибок
    yield_forecasts_df['error_ns_ar1'] = yield_forecasts_df['actual_yield'] - yield_forecasts_df['yield_forecast_ns_ar1']

    # Удаляем строки без фактических значений
    yield_forecasts_df.dropna(subset=['actual_yield'], inplace=True)

    # Расчет RMSE
    rmse_results_list = []
    for horizon_val in forecast_horizons_days:
        for mat_y_val in eval_maturities_years:
            subset_for_rmse = yield_forecasts_df[
                (yield_forecasts_df['horizon_days'] == horizon_val) &
                (yield_forecasts_df['maturity_years'] == mat_y_val)
            ]

            n_obs = len(subset_for_rmse)
            rmse_val = np.nan
            if n_obs > 0:
                valid_errors = subset_for_rmse['error_ns_ar1'].dropna()
                if len(valid_errors) > 0:
                    rmse_val = np.sqrt(np.mean(valid_errors**2))

            rmse_results_list.append({
                'tau1': tau1_value,
                'horizon_days': horizon_val,
                'maturity_years': mat_y_val,
                'n_obs': n_obs,
                'rmse': rmse_val
            })

    rmse_df = pd.DataFrame(rmse_results_list)

    return params_successful, ar_factor_forecasts_df, yield_forecasts_df, rmse_df

