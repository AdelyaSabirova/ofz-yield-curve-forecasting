"""
Функции для прогнозирования факторов и доходностей
"""

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from .config import *
import warnings


def forecast_factors_ar1(factors_df, forecast_horizons_days=FORECAST_HORIZONS_DAYS,
                        out_of_sample_start_ratio=OUT_OF_SAMPLE_START_RATIO):
    """
    Прогнозирование факторов AR(1) моделью

    Args:
        factors_df (pd.DataFrame): DataFrame с факторами NS
        forecast_horizons_days (list): Горизонты прогнозирования
        out_of_sample_start_ratio (float): Доля данных для обучения

    Returns:
        pd.DataFrame: Прогнозы факторов
    """
    ar_factor_forecasts_oos_list = []

    n_obs_total = len(factors_df)
    n_train_initial = int(n_obs_total * out_of_sample_start_ratio)
    n_test = n_obs_total - n_train_initial

    max_h_forecast = max(forecast_horizons_days)

    # Рекурсивное out-of-sample прогнозирование факторов
    for t_idx in tqdm(range(n_train_initial, n_obs_total), desc="OOS AR(1) Factor Forecasting"):
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
                    'actual_value': factors_df.loc[forecast_target_date_actual, factor_name]
                })

    ar_factor_forecasts_df = pd.DataFrame(ar_factor_forecasts_oos_list)
    ar_factor_forecasts_df.dropna(subset=['forecast_value_ar1', 'actual_value'], inplace=True)

    print(f"Получено {len(ar_factor_forecasts_df)} out-of-sample прогнозов факторов AR(1)")

    return ar_factor_forecasts_df


def forecast_factors_var_level(factors_df, oil_series_full, usd_series_full,
                              forecast_horizons_days=FORECAST_HORIZONS_DAYS,
                              out_of_sample_start_ratio=OUT_OF_SAMPLE_START_RATIO):
    """
    Прогнозирование факторов VAR в уровнях

    Args:
        factors_df (pd.DataFrame): DataFrame с факторами NS
        oil_series_full (pd.Series): Серия с логарифмами цен на нефть
        usd_series_full (pd.Series): Серия с логарифмами курса USD
        forecast_horizons_days (list): Горизонты прогнозирования
        out_of_sample_start_ratio (float): Доля данных для обучения

    Returns:
        pd.DataFrame: Прогнозы факторов VAR в уровнях
    """
    # Выравниваем по датам факторов
    oil_series_aligned_for_var = oil_series_full.reindex(factors_df.index).ffill().bfill()
    usd_series_aligned_for_var = usd_series_full.reindex(factors_df.index).ffill().bfill()

    var_endog_df = factors_df.copy()
    var_endog_df['ln_oil'] = oil_series_aligned_for_var
    var_endog_df['ln_usd'] = usd_series_aligned_for_var
    var_endog_df.dropna(inplace=True)

    print(f"VAR (levels): наблюдений после склейки: {len(var_endog_df)}")

    var_level_factor_forecasts_oos_list = []

    n_obs_total = len(var_endog_df)
    n_train_initial = int(n_obs_total * out_of_sample_start_ratio)
    n_test = n_obs_total - n_train_initial

    # VAR в уровнях
    for t_idx_var_level in tqdm(range(n_train_initial, n_obs_total), desc="OOS VAR_level Factor Forecasting"):
        current_train_set_var_level = var_endog_df.iloc[:t_idx_var_level]

        var_model_level = VAR(current_train_set_var_level)

        # Ручной выбор количества лагов с trend='n'
        try:
            lag_selection = var_model_level.select_order(maxlags=min(VAR_MAX_LAGS, len(current_train_set_var_level)//2), trend='n')
            optimal_lags = lag_selection.selected_orders['aic'] if hasattr(lag_selection, 'selected_orders') and lag_selection.selected_orders['aic'] > 0 else 1
        except:
            optimal_lags = 1

        var_results_level = var_model_level.fit(maxlags=optimal_lags, trend='n')
        if var_results_level.k_ar == 0:
            var_results_level = var_model_level.fit(maxlags=1, trend='n')

        var_col_pos_level = {col: idx for idx, col in enumerate(var_results_level.names)}
        last_known_values_for_var_level = current_train_set_var_level.values[-var_results_level.k_ar:]

        all_h_forecasts_from_var_level = var_results_level.forecast(y=last_known_values_for_var_level,
                                                                  steps=max(forecast_horizons_days))

        date_of_forecast_made_var_level = var_endog_df.index[t_idx_var_level - 1]

        for h_idx, h_days_var_level in enumerate(forecast_horizons_days):
            target_forecast_idx_var_level = t_idx_var_level + h_days_var_level - 1
            if target_forecast_idx_var_level >= n_obs_total:
                continue
            forecast_target_date_actual_var_level = var_endog_df.index[target_forecast_idx_var_level]
            current_h_forecast_vector_level = all_h_forecasts_from_var_level[h_days_var_level - 1]

            for factor_name_var_level in ['beta0', 'beta1', 'beta2', 'ln_oil', 'ln_usd']:
                if factor_name_var_level not in var_col_pos_level:
                    continue
                var_level_factor_forecasts_oos_list.append({
                    'forecast_made_on_date': date_of_forecast_made_var_level,
                    'forecast_target_date': forecast_target_date_actual_var_level,
                    'horizon_days': h_days_var_level,
                    'factor': factor_name_var_level,
                    'forecast_value_var_level': current_h_forecast_vector_level[var_col_pos_level[factor_name_var_level]],
                    'actual_value': var_endog_df.loc[forecast_target_date_actual_var_level, factor_name_var_level]
                })

    var_factor_forecasts_df = pd.DataFrame(var_level_factor_forecasts_oos_list)
    var_factor_forecasts_df.dropna(subset=['forecast_value_var_level', 'actual_value'], inplace=True)

    print(f"Получено {len(var_factor_forecasts_df)} OOS прогнозов факторов VAR в УРОВНЯХ")

    return var_factor_forecasts_df


def forecast_factors_var_dl(factors_df, oil_series_full, usd_series_full,
                           forecast_horizons_days=FORECAST_HORIZONS_DAYS,
                           out_of_sample_start_ratio=OUT_OF_SAMPLE_START_RATIO):
    """
    Прогнозирование факторов VAR с разностями (VAR_diff_level)

    Args:
        factors_df (pd.DataFrame): DataFrame с факторами NS
        oil_series_full (pd.Series): Серия с логарифмами цен на нефть
        usd_series_full (pd.Series): Серия с логарифмами курса USD
        forecast_horizons_days (list): Горизонты прогнозирования
        out_of_sample_start_ratio (float): Доля данных для обучения

    Returns:
        pd.DataFrame: Прогнозы факторов VAR_diff_level
    """
    var_dl_factor_forecasts_oos_list = []

    # Создаем датафрейм с разностями для beta0/beta1 и уровнями для beta2 + ln_oil/ln_usd
    factors_df_for_var_dl = pd.DataFrame(index=factors_df.index)
    factors_df_for_var_dl['d_beta0'] = factors_df['beta0'].diff()
    factors_df_for_var_dl['d_beta1'] = factors_df['beta1'].diff()
    factors_df_for_var_dl['beta2'] = factors_df['beta2']
    factors_df_for_var_dl['ln_oil'] = oil_series_full.reindex(factors_df.index).ffill().bfill()
    factors_df_for_var_dl['ln_usd'] = usd_series_full.reindex(factors_df.index).ffill().bfill()
    factors_df_for_var_dl.dropna(inplace=True)

    print("Факторы для VAR_diff_level (d_beta0, d_beta1, beta2, ln_oil, ln_usd)")

    # Переопределяем параметры
    n_obs_total_dl = len(factors_df_for_var_dl)
    n_train_initial_dl = int(n_obs_total_dl * out_of_sample_start_ratio)
    n_test_dl = n_obs_total_dl - n_train_initial_dl

    for t_idx_dl in tqdm(range(n_train_initial_dl, n_obs_total_dl), desc="OOS VAR_diff_level Factor Forecasting"):
        current_train_set_var_dl = factors_df_for_var_dl.iloc[:t_idx_dl]

        # Исходные уровни факторов на момент t-1
        date_of_last_known_levels = factors_df_for_var_dl.index[t_idx_dl - 1]
        last_known_level_beta0 = factors_df.loc[date_of_last_known_levels, 'beta0']
        last_known_level_beta1 = factors_df.loc[date_of_last_known_levels, 'beta1']

        var_model_dl = VAR(current_train_set_var_dl)
        var_results_dl = var_model_dl.fit(maxlags=VAR_MAX_LAGS, ic='aic', trend='n')
        if var_results_dl.k_ar == 0:
            var_results_dl = var_model_dl.fit(maxlags=1, trend='n')

        var_col_pos_dl = {col: idx for idx, col in enumerate(var_results_dl.names)}
        last_known_values_for_var_dl = current_train_set_var_dl.values[-var_results_dl.k_ar:]
        all_h_forecasts_from_var_dl = var_results_dl.forecast(y=last_known_values_for_var_dl,
                                                           steps=max(forecast_horizons_days))

        # Интегрируем прогнозы разностей обратно в уровни
        forecasted_d_beta0_series = all_h_forecasts_from_var_dl[:, var_col_pos_dl['d_beta0']]
        forecasted_d_beta1_series = all_h_forecasts_from_var_dl[:, var_col_pos_dl['d_beta1']]
        forecasted_beta2_series = all_h_forecasts_from_var_dl[:, var_col_pos_dl['beta2']]
        forecasted_ln_oil_series = all_h_forecasts_from_var_dl[:, var_col_pos_dl['ln_oil']] if 'ln_oil' in var_col_pos_dl else None
        forecasted_ln_usd_series = all_h_forecasts_from_var_dl[:, var_col_pos_dl['ln_usd']] if 'ln_usd' in var_col_pos_dl else None

        # Кумулятивная сумма прогнозов разностей + последний известный уровень
        forecasted_beta0_levels = last_known_level_beta0 + np.cumsum(forecasted_d_beta0_series)
        forecasted_beta1_levels = last_known_level_beta1 + np.cumsum(forecasted_d_beta1_series)

        for h_idx, h_days_var_dl in enumerate(forecast_horizons_days):
            target_forecast_idx_var_dl = t_idx_dl + h_days_var_dl - 1
            if target_forecast_idx_var_dl >= n_obs_total_dl:
                continue
            # Дата, НА которую прогноз
            forecast_target_date_actual_var_dl = factors_df_for_var_dl.index[target_forecast_idx_var_dl]

            # Сохраняем прогнозы в УРОВНЯХ для beta0, beta1, beta2
            fc_beta0_level = forecasted_beta0_levels[h_days_var_dl-1]
            fc_beta1_level = forecasted_beta1_levels[h_days_var_dl-1]
            fc_beta2_level = forecasted_beta2_series[h_days_var_dl-1]
            fc_ln_oil_level = forecasted_ln_oil_series[h_days_var_dl-1] if forecasted_ln_oil_series is not None else np.nan
            fc_ln_usd_level = forecasted_ln_usd_series[h_days_var_dl-1] if forecasted_ln_usd_series is not None else np.nan

            var_dl_factor_forecasts_oos_list.extend([
                {'forecast_made_on_date': date_of_last_known_levels,
                 'forecast_target_date': forecast_target_date_actual_var_dl,
                 'horizon_days': h_days_var_dl, 'factor': 'beta0',
                 'forecast_value_var_dl': fc_beta0_level,
                 'actual_value': factors_df.loc[forecast_target_date_actual_var_dl, 'beta0']},
                {'forecast_made_on_date': date_of_last_known_levels,
                 'forecast_target_date': forecast_target_date_actual_var_dl,
                 'horizon_days': h_days_var_dl, 'factor': 'beta1',
                 'forecast_value_var_dl': fc_beta1_level,
                 'actual_value': factors_df.loc[forecast_target_date_actual_var_dl, 'beta1']},
                {'forecast_made_on_date': date_of_last_known_levels,
                 'forecast_target_date': forecast_target_date_actual_var_dl,
                 'horizon_days': h_days_var_dl, 'factor': 'beta2',
                 'forecast_value_var_dl': fc_beta2_level,
                 'actual_value': factors_df.loc[forecast_target_date_actual_var_dl, 'beta2']},
                {'forecast_made_on_date': date_of_last_known_levels,
                 'forecast_target_date': forecast_target_date_actual_var_dl,
                 'horizon_days': h_days_var_dl, 'factor': 'ln_oil',
                 'forecast_value_var_dl': fc_ln_oil_level,
                 'actual_value': factors_df_for_var_dl.loc[forecast_target_date_actual_var_dl, 'ln_oil']},
                {'forecast_made_on_date': date_of_last_known_levels,
                 'forecast_target_date': forecast_target_date_actual_var_dl,
                 'horizon_days': h_days_var_dl, 'factor': 'ln_usd',
                 'forecast_value_var_dl': fc_ln_usd_level,
                 'actual_value': factors_df_for_var_dl.loc[forecast_target_date_actual_var_dl, 'ln_usd']}
            ])

    var_dl_factor_forecasts_df = pd.DataFrame(var_dl_factor_forecasts_oos_list)
    var_dl_factor_forecasts_df.dropna(subset=['forecast_value_var_dl', 'actual_value'], inplace=True)

    if var_dl_factor_forecasts_df.empty:
        print("Не удалось получить ни одного валидного OOS прогноза факторов VAR_diff_level.")
    else:
        print(f"Получено {len(var_dl_factor_forecasts_df)} OOS прогнозов факторов VAR_diff_level.")

    return var_dl_factor_forecasts_df


def forecast_factors_vecm(factors_for_vecm_df, forecast_horizons_days=FORECAST_HORIZONS_DAYS,
                         out_of_sample_start_ratio=OUT_OF_SAMPLE_START_RATIO):
    """
    Прогнозирование факторов VECM моделью

    Args:
        factors_for_vecm_df (pd.DataFrame): Подготовленный DataFrame для VECM
        forecast_horizons_days (list): Горизонты прогнозирования
        out_of_sample_start_ratio (float): Доля данных для обучения

    Returns:
        tuple: (vecm_factor_forecasts_df, vecm_fit_success, vecm_fit_failures, johansen_msg, trace_rank, max_eig_rank)
    """
    vecm_factor_forecasts_oos_list = []
    vecm_fit_success = 0
    vecm_fit_failures = []

    endog_cols = [c for c in ['beta0', 'beta1', 'beta2', 'ln_oil', 'ln_usd'] if c in factors_for_vecm_df.columns]
    exog_cols = []

    endog_df = factors_for_vecm_df[endog_cols].copy()
    exog_df = None

    endog_df.dropna(inplace=True)

    n_obs_total_vecm = len(endog_df)
    n_train_initial_vecm = int(n_obs_total_vecm * out_of_sample_start_ratio)
    n_test_vecm = n_obs_total_vecm - n_train_initial_vecm

    print(f"VECM: Всего наблюдений: {n_obs_total_vecm}")
    print(f"VECM: Размер тестовой выборки: {n_test_vecm}")

    dim_vecm = endog_df.shape[1]
    MIN_OBS_FOR_VECM_ATTEMPT = max(VAR_MAX_LAGS * dim_vecm * 2, 60)

    # Тест Йохансена
    trace_rank = 0
    max_eig_rank = 0
    johansen_msg = ""

    if n_obs_total_vecm >= MIN_OBS_FOR_VECM_ATTEMPT:
        try:
            johansen_res = coint_johansen(endog_df, det_order=0, k_ar_diff=1)
            trace_stats, trace_crit = johansen_res.lr1, johansen_res.cvt[:, 1]
            maxeig_stats, maxeig_crit = johansen_res.lr2, johansen_res.cvm[:, 1]

            for r, (stat, crit) in enumerate(zip(trace_stats, trace_crit)):
                if stat > crit:
                    trace_rank = r + 1

            for r, (stat, crit) in enumerate(zip(maxeig_stats, maxeig_crit)):
                if stat > crit:
                    max_eig_rank = r + 1

            johansen_msg = "Тест Йохансена успешно выполнен"
        except Exception as e:
            johansen_msg = f"⚠️ Ошибка теста Йохансена: {e}"
    else:
        johansen_msg = "⚠️ Недостаточно наблюдений для теста Йохансена"

    max_rank_allowed = max(1, dim_vecm - 1)
    COINT_RANK_FOR_VECM = max(trace_rank, max_eig_rank, 1)
    COINT_RANK_FOR_VECM = min(COINT_RANK_FOR_VECM, max_rank_allowed)

    print(johansen_msg)
    print(f"Trace rank = {trace_rank}, Max eigenvalue rank = {max_eig_rank}")
    print(f"Используется COINT_RANK_FOR_VECM = {COINT_RANK_FOR_VECM}")

    if n_test_vecm < 1:
        print("Ошибка: Тестовая выборка для VECM пуста")
        return pd.DataFrame(), vecm_fit_success, vecm_fit_failures, johansen_msg, trace_rank, max_eig_rank
    elif n_train_initial_vecm <= MIN_OBS_FOR_VECM_ATTEMPT:
        print(f"Ошибка: Начальный размер обучающей выборки слишком мал")
        return pd.DataFrame(), vecm_fit_success, vecm_fit_failures, johansen_msg, trace_rank, max_eig_rank

    col_pos = {col: idx for idx, col in enumerate(endog_df.columns)}

    for t_idx_vecm in tqdm(range(n_train_initial_vecm, n_obs_total_vecm), desc="OOS VECM Factor Forecasting"):
        current_endog_train = endog_df.iloc[:t_idx_vecm]
        current_exog_train = exog_df.iloc[:t_idx_vecm] if exog_df is not None else None
        date_of_forecast_made = current_endog_train.index[-1]
        current_train_set_ar_beta2 = factors_for_vecm_df.loc[:date_of_forecast_made, 'beta2'].dropna()

        forecasts_beta0_vecm = {}
        forecasts_beta1_vecm = {}
        forecasts_beta2_vecm = {}
        forecasts_beta2_ar = {}
        vecm_fit_ok = False

        # VECM для endog_cols
        if len(current_endog_train) >= MIN_OBS_FOR_VECM_ATTEMPT:
            try:
                var_model_for_lags = VAR(current_endog_train)
                max_lags_aic_select = min(VAR_MAX_LAGS, int(len(current_endog_train) / (dim_vecm + 2)) - 1)

                lags_in_levels_p = 1
                if max_lags_aic_select > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        selected_order_results = var_model_for_lags.select_order(maxlags=max_lags_aic_select)
                    lags_in_levels_p = selected_order_results.aic
                    if lags_in_levels_p == 0:
                        lags_in_levels_p = 1

                k_ar_diff_vecm = max(lags_in_levels_p - 1, 1)  # не даём лагов меньше 1

                exog_future_block = None
                if exog_df is not None:
                    exog_future_block = exog_df.iloc[t_idx_vecm : t_idx_vecm + max(forecast_horizons_days)]
                    if len(exog_future_block) < max(forecast_horizons_days):
                        if exog_future_block.empty:
                            exog_future_block = pd.DataFrame(np.zeros((max(forecast_horizons_days), len(exog_cols))), columns=exog_cols)
                        else:
                            last_row = exog_future_block.iloc[[-1]]
                            rows_needed = max(forecast_horizons_days) - len(exog_future_block)
                            pad_block = pd.concat([last_row] * rows_needed)
                            exog_future_block = pd.concat([exog_future_block, pad_block])

                deterministic_param = "ci" if COINT_RANK_FOR_VECM > 0 else "n"

                vecm_model = VECM(endog=current_endog_train,
                                  exog=current_exog_train,
                                  k_ar_diff=k_ar_diff_vecm,
                                  coint_rank=COINT_RANK_FOR_VECM,
                                  deterministic=deterministic_param)
                vecm_results = vecm_model.fit()
                all_h_forecasts_from_vecm = vecm_results.predict(steps=max(forecast_horizons_days),
                                                               exog_fc=exog_future_block)
                vecm_fit_ok = True
                vecm_fit_success += 1
            except Exception as e:
                vecm_fit_failures.append(str(e))
                all_h_forecasts_from_vecm = None

            if vecm_fit_ok and all_h_forecasts_from_vecm is not None:
                for h_days_vecm_fc in forecast_horizons_days:
                    if 'beta0' in col_pos:
                        forecasts_beta0_vecm[h_days_vecm_fc] = all_h_forecasts_from_vecm[h_days_vecm_fc - 1, col_pos['beta0']]
                    if 'beta1' in col_pos:
                        forecasts_beta1_vecm[h_days_vecm_fc] = all_h_forecasts_from_vecm[h_days_vecm_fc - 1, col_pos['beta1']]
                    if 'beta2' in col_pos:
                        forecasts_beta2_vecm[h_days_vecm_fc] = all_h_forecasts_from_vecm[h_days_vecm_fc - 1, col_pos['beta2']]

        # AR(1) для beta2 (fallback, если VECM не дала прогноз)
        if len(current_train_set_ar_beta2) > 0 and date_of_forecast_made in factors_for_vecm_df.index:
            last_known_beta2_val = factors_for_vecm_df.loc[date_of_forecast_made, 'beta2']
            if pd.notna(last_known_beta2_val):
                for h_days_ar_b2_fc in forecast_horizons_days:
                    if len(current_train_set_ar_beta2) >= h_days_ar_b2_fc + 5:
                        Y_ar_b2 = current_train_set_ar_beta2.iloc[h_days_ar_b2_fc:].values
                        X_ar_b2_raw = current_train_set_ar_beta2.iloc[:-h_days_ar_b2_fc].values
                        if len(X_ar_b2_raw) >= 2:
                            X_ar_b2_const = sm.add_constant(X_ar_b2_raw)
                            ar_model_b2 = sm.OLS(Y_ar_b2, X_ar_b2_const).fit()
                            forecasts_beta2_ar[h_days_ar_b2_fc] = ar_model_b2.predict([1, last_known_beta2_val])[0]

        # Сохранение результатов
        for h_days_oos_fc in forecast_horizons_days:
            target_forecast_idx_oos = t_idx_vecm + h_days_oos_fc - 1
            if target_forecast_idx_oos >= n_obs_total_vecm:
                continue
            forecast_target_date_actual_oos = endog_df.index[target_forecast_idx_oos]

            actual_b0 = factors_for_vecm_df.loc[forecast_target_date_actual_oos, 'beta0'] if forecast_target_date_actual_oos in factors_for_vecm_df.index else np.nan
            actual_b1 = factors_for_vecm_df.loc[forecast_target_date_actual_oos, 'beta1'] if forecast_target_date_actual_oos in factors_for_vecm_df.index else np.nan
            actual_b2 = factors_for_vecm_df.loc[forecast_target_date_actual_oos, 'beta2'] if forecast_target_date_actual_oos in factors_for_vecm_df.index else np.nan

            fc_b0_component = forecasts_beta0_vecm.get(h_days_oos_fc, np.nan)
            fc_b1_component = forecasts_beta1_vecm.get(h_days_oos_fc, np.nan)
            fc_b2_component = forecasts_beta2_vecm.get(h_days_oos_fc, np.nan)
            if pd.isna(fc_b2_component):
                fc_b2_component = forecasts_beta2_ar.get(h_days_oos_fc, np.nan)

            final_fc_b0 = fc_b0_component if pd.notna(fc_b0_component) and pd.notna(fc_b1_component) and pd.notna(fc_b2_component) else np.nan
            final_fc_b1 = fc_b1_component if pd.notna(fc_b0_component) and pd.notna(fc_b1_component) and pd.notna(fc_b2_component) else np.nan
            final_fc_b2 = fc_b2_component if pd.notna(fc_b0_component) and pd.notna(fc_b1_component) and pd.notna(fc_b2_component) else np.nan

            vecm_factor_forecasts_oos_list.extend([
                {'forecast_made_on_date': date_of_forecast_made, 'forecast_target_date': forecast_target_date_actual_oos,
                 'horizon_days': h_days_oos_fc, 'factor': 'beta0', 'forecast_value_vecm': final_fc_b0, 'actual_value': actual_b0},
                {'forecast_made_on_date': date_of_forecast_made, 'forecast_target_date': forecast_target_date_actual_oos,
                 'horizon_days': h_days_oos_fc, 'factor': 'beta1', 'forecast_value_vecm': final_fc_b1, 'actual_value': actual_b1},
                {'forecast_made_on_date': date_of_forecast_made, 'forecast_target_date': forecast_target_date_actual_oos,
                 'horizon_days': h_days_oos_fc, 'factor': 'beta2', 'forecast_value_vecm': final_fc_b2, 'actual_value': actual_b2}
            ])

    print(f"Успешных подгонок VECM: {vecm_fit_success}; с ошибками: {len(vecm_fit_failures)}")
    if vecm_fit_failures:
        print("Первые ошибки VECM (до 3):")
        for err in vecm_fit_failures[:3]:
            print(f"- {err}")

    vecm_factor_forecasts_df = pd.DataFrame(vecm_factor_forecasts_oos_list)

    if vecm_factor_forecasts_df.empty or 'forecast_value_vecm' not in vecm_factor_forecasts_df.columns:
        vecm_factor_forecasts_df = pd.DataFrame(columns=['forecast_made_on_date', 'forecast_target_date', 'horizon_days', 'factor', 'forecast_value_vecm', 'actual_value'])
        print("Не удалось получить ни одной строки прогнозов факторов VECM/AR")
    else:
        vecm_factor_forecasts_df.dropna(subset=['forecast_value_vecm'], inplace=True)

        if vecm_factor_forecasts_df.empty:
            print("Не удалось получить валидных OOS прогнозов факторов с VECM/AR")
        else:
            num_complete_vecm_sets = len(vecm_factor_forecasts_df.groupby(['forecast_target_date', 'horizon_days']).filter(lambda x: len(x['factor'].unique()) == 3 and x['forecast_value_vecm'].notna().all())) // 3
            print(f"Получено {len(vecm_factor_forecasts_df)} строк прогнозов факторов с VECM/AR")
            print(f"Из них {num_complete_vecm_sets} полных наборов (beta0,1,2)")

    return vecm_factor_forecasts_df, vecm_fit_success, vecm_fit_failures, johansen_msg, trace_rank, max_eig_rank


def analyze_irf(factors_df, oil_series_full, usd_series_full,
                forecast_horizons_days=FORECAST_HORIZONS_DAYS):
    """
    Анализ импульсных откликов (IRF) для факторов и внешних переменных

    Args:
        factors_df (pd.DataFrame): DataFrame с факторами NS
        oil_series_full (pd.Series): Серия с логарифмами цен на нефть
        usd_series_full (pd.Series): Серия с логарифмами курса USD
        forecast_horizons_days (list): Горизонты для IRF

    Returns:
        tuple: (irf_yield_df, var_res_irf)
    """
    # Объединяем ln_oil, ln_usd и факторы для VAR/IRF
    irf_merge_df = pd.DataFrame({"ln_oil": oil_series_full}).reindex(factors_df.index)
    for col in ["beta0", "beta1", "beta2"]:
        irf_merge_df[col] = factors_df[col]
    irf_merge_df["ln_usd"] = usd_series_full.reindex(factors_df.index)
    irf_merge_df.dropna(inplace=True)

    # Фит VAR
    k_vars = irf_merge_df.shape[1]
    maxlags_irf = min(VAR_MAX_LAGS, max(1, int(len(irf_merge_df) / (k_vars + 2)) - 1))
    var_model_irf = VAR(irf_merge_df)
    try:
        selected = var_model_irf.select_order(maxlags=maxlags_irf)
        lags_irf = selected.selected_orders['aic'] if hasattr(selected, 'selected_orders') and selected.selected_orders['aic'] > 0 else 1
    except Exception:
        lags_irf = 1
    var_res_irf = var_model_irf.fit(maxlags=lags_irf)

    # Горизонты: 6 и 12 месяцев в торговых днях
    horizons_days = [126, 252]
    max_h = max(horizons_days)
    irf_obj = var_res_irf.irf(max_h)

    names = list(var_res_irf.names)
    idx_map = {n: i for i, n in enumerate(names)}

    # Матуритеты и tau для НС
    current_tau_for_ns = 1.8  # Можно параметризовать

    def ns_loadings(mat_y, tau):
        tau_div = mat_y / tau
        exp_term = np.exp(-tau_div)
        load_b1 = (1 - exp_term) / tau_div
        load_b2 = load_b1 - exp_term
        return load_b1, load_b2

    def shock_to_yield(shock_var):
        rows = []
        shock_idx = idx_map[shock_var]
        b0_idx = idx_map["beta0"]; b1_idx = idx_map["beta1"]; b2_idx = idx_map["beta2"]
        for h in horizons_days:
            b0_resp = irf_obj.irfs[h, b0_idx, shock_idx]
            b1_resp = irf_obj.irfs[h, b1_idx, shock_idx]
            b2_resp = irf_obj.irfs[h, b2_idx, shock_idx]
            for mat_y in EVAL_MATURITIES_YEARS:
                load_b1, load_b2 = ns_loadings(mat_y, current_tau_for_ns)
                delta_yield_pp = (b0_resp + load_b1 * b1_resp + load_b2 * b2_resp) * 100
                rows.append({"shock": shock_var, "horizon_days": h, "maturity_years": mat_y, "delta_yield_pp": delta_yield_pp})
        return pd.DataFrame(rows)

    irf_yield_df = pd.concat([shock_to_yield("ln_oil"), shock_to_yield("ln_usd")], ignore_index=True)

    return irf_yield_df, var_res_irf
