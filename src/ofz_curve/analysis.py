"""
Функции для анализа результатов прогнозирования
"""

import pandas as pd
import numpy as np
from nelson_siegel_svensson import NelsonSiegelCurve
from .config import *


def _rmse_column_name(prefix: str) -> str:
    """
    Единый маппинг названий колонок RMSE для всех моделей.
    """
    mapping = {
        'ns_ar1': 'RMSE_AR1',
        'ns_var_level': 'RMSE_VAR_LEVEL',
        'ns_var_dl': 'RMSE_VAR_DL',
        'ns_vecm': 'RMSE_VECM',
        'ns_catboost': 'RMSE_CatBoost',
        'rw': 'RMSE_RW',
    }
    return mapping.get(prefix, f'RMSE_{prefix.upper().replace("NS_", "")}')


def calculate_yield_forecasts(merged_fc_df, current_tau1, df_filtered=None,
                             eval_maturities_years=EVAL_MATURITIES_YEARS,
                             forecast_horizons_days=FORECAST_HORIZONS_DAYS):
    """
    Расчет прогнозов доходностей на основе прогнозов факторов

    Args:
        merged_fc_df (pd.DataFrame): Объединенный DataFrame с прогнозами факторов
        current_tau1 (float): Значение tau1 для NS кривой
        eval_maturities_years (list): Сроки погашения для оценки
        forecast_horizons_days (list): Горизонты прогнозирования

    Returns:
        tuple: (yield_forecasts_df, model_forecast_col_map)
    """
    model_forecast_col_map = {
        'ar1': 'forecast_value_ar1',
        'var_level': 'forecast_value_var_level',
        'var_dl': 'forecast_value_var_dl',
        'vecm': 'forecast_value_vecm',
        'catboost': 'forecast_value_catboost'
    }

    if merged_fc_df is None or merged_fc_df.empty:
        return pd.DataFrame(), model_forecast_col_map
    if df_filtered is None or df_filtered.empty:
        print("df_filtered не передан или пуст. Расчёт прогнозов доходности невозможен.")
        return pd.DataFrame(), model_forecast_col_map

    required_market_cols = {'TRADEDATE', 'TTM_years', 'YIELDCLOSE'}
    if not required_market_cols.issubset(df_filtered.columns):
        missing_cols = sorted(required_market_cols - set(df_filtered.columns))
        print(f"В df_filtered отсутствуют обязательные колонки: {missing_cols}")
        return pd.DataFrame(), model_forecast_col_map

    grouped_all_fc = merged_fc_df.groupby(['forecast_target_date', 'horizon_days'])

    oos_yield_forecast_results_list = []

    for (target_date_of_yield, horizon), group_fc in grouped_all_fc:
        beta_hats_all_models = {key: {} for key in model_forecast_col_map.keys()}
        model_validity = {key: True for key in model_forecast_col_map.keys()}

        for factor_name in ['beta0', 'beta1', 'beta2']:
            factor_row = group_fc[group_fc['factor'] == factor_name]
            if factor_row.empty:
                model_validity = {k: False for k in model_validity}
                break

            for model_key, forecast_col_name in model_forecast_col_map.items():
                if model_validity[model_key]:
                    if forecast_col_name not in factor_row.columns or pd.isna(factor_row[forecast_col_name].iloc[0]):
                        model_validity[model_key] = False
                    else:
                        beta_hats_all_models[model_key][factor_name] = factor_row[forecast_col_name].iloc[0]

        if not any(model_validity.values()):
            continue

        predicted_ns_curves_all = {}
        for model_key in model_forecast_col_map.keys():
            if model_validity[model_key] and len(beta_hats_all_models[model_key]) == 3:
                try:
                    predicted_ns_curves_all[model_key] = NelsonSiegelCurve(
                        beta0=beta_hats_all_models[model_key]['beta0'],
                        beta1=beta_hats_all_models[model_key]['beta1'],
                        beta2=beta_hats_all_models[model_key]['beta2'],
                        tau=current_tau1
                    )
                except KeyError:
                    model_validity[model_key] = False
            else:
                 model_validity[model_key] = False

        # Фактическая рыночная кривая на дату цели прогноза
        actual_market_data_today = df_filtered[df_filtered['TRADEDATE'] == target_date_of_yield]
        if actual_market_data_today.empty:
            continue
        actual_ttms_today = actual_market_data_today['TTM_years'].values
        actual_yields_today = actual_market_data_today['YIELDCLOSE'].values
        sort_idx = np.argsort(actual_ttms_today)
        actual_ttms_sorted = actual_ttms_today[sort_idx]
        actual_yields_sorted = actual_yields_today[sort_idx]

        # Random Walk: берём кривую на дату формирования прогноза
        date_t_for_rw = None
        if 'forecast_made_on_date' in group_fc.columns:
            valid_dates_rw = group_fc['forecast_made_on_date'].dropna()
            if not valid_dates_rw.empty:
                date_t_for_rw = valid_dates_rw.iloc[0]

        rw_ttms_sorted = None
        rw_yields_sorted = None
        if date_t_for_rw is not None:
            market_data_rw = df_filtered[df_filtered['TRADEDATE'] == date_t_for_rw]
            if not market_data_rw.empty:
                rw_ttms = market_data_rw['TTM_years'].values
                rw_yields = market_data_rw['YIELDCLOSE'].values
                rw_sort_idx = np.argsort(rw_ttms)
                rw_ttms_sorted = rw_ttms[rw_sort_idx]
                rw_yields_sorted = rw_yields[rw_sort_idx]

        # Для каждого срока погашения
        for mat_y in eval_maturities_years:
            result_row_dict = {
                'forecast_target_date': target_date_of_yield,
                'horizon_days': horizon,
                'maturity_years': mat_y
            }

            for model_key_iter in model_forecast_col_map.keys():
                ns_curve_object = predicted_ns_curves_all.get(model_key_iter)
                if ns_curve_object:
                     result_row_dict[f'yield_forecast_ns_{model_key_iter}'] = ns_curve_object(mat_y) * 100
                else:
                     result_row_dict[f'yield_forecast_ns_{model_key_iter}'] = np.nan

            # Фактическое значение (интерполяция)
            actual_val_interp = np.nan
            if len(actual_ttms_sorted) >= 2:
                actual_val_interp = np.interp(
                    mat_y,
                    actual_ttms_sorted,
                    actual_yields_sorted,
                    left=actual_yields_sorted[0],
                    right=actual_yields_sorted[-1]
                )
            elif len(actual_ttms_sorted) == 1:
                actual_val_interp = actual_yields_sorted[0] if np.isclose(mat_y, actual_ttms_sorted[0]) else np.nan
            result_row_dict['actual_yield'] = actual_val_interp

            # Random Walk (интерполяция на дату формирования прогноза)
            rw_val_interp = np.nan
            if rw_ttms_sorted is not None and len(rw_ttms_sorted) >= 2:
                rw_val_interp = np.interp(
                    mat_y,
                    rw_ttms_sorted,
                    rw_yields_sorted,
                    left=rw_yields_sorted[0],
                    right=rw_yields_sorted[-1]
                )
            elif rw_ttms_sorted is not None and len(rw_ttms_sorted) == 1:
                rw_val_interp = rw_yields_sorted[0] if np.isclose(mat_y, rw_ttms_sorted[0]) else np.nan
            result_row_dict['yield_forecast_rw'] = rw_val_interp

            oos_yield_forecast_results_list.append(result_row_dict)

    yield_forecasts_df = pd.DataFrame(oos_yield_forecast_results_list)

    # Расчет ошибок
    model_name_prefixes_for_errors = ['ns_ar1', 'ns_var_level', 'ns_var_dl', 'ns_vecm', 'ns_catboost', 'rw']
    for prefix in model_name_prefixes_for_errors:
        forecast_col_name = f'yield_forecast_{prefix}'
        error_col_name = f'error_{prefix}'
        if forecast_col_name in yield_forecasts_df.columns and 'actual_yield' in yield_forecasts_df.columns:
            yield_forecasts_df[error_col_name] = yield_forecasts_df['actual_yield'] - yield_forecasts_df[forecast_col_name]
        else:
            yield_forecasts_df[error_col_name] = np.nan

    # Если расчет не дал строк, возвращаем пустую таблицу без аварии
    if yield_forecasts_df.empty:
        print("Не удалось получить валидных OOS прогнозов доходностей")
        return yield_forecasts_df, model_forecast_col_map

    # Удаляем строки без фактических значений
    if 'actual_yield' in yield_forecasts_df.columns:
        yield_forecasts_df.dropna(subset=['actual_yield'], how='any', inplace=True)
    else:
        print("В таблице прогнозов нет колонки actual_yield, возврат пустого результата")
        return pd.DataFrame(), model_forecast_col_map

    if yield_forecasts_df.empty:
        print("Не удалось получить валидных OOS прогнозов доходностей")
    else:
        print(f"Получено {len(yield_forecasts_df)} OOS прогнозов доходностей")

    return yield_forecasts_df, model_forecast_col_map


def calculate_rmse_summary(yield_forecasts_df, eval_maturities_years=EVAL_MATURITIES_YEARS,
                          forecast_horizons_days=FORECAST_HORIZONS_DAYS):
    """
    Расчет сводки RMSE по моделям

    Args:
        yield_forecasts_df (pd.DataFrame): DataFrame с прогнозами доходностей
        eval_maturities_years (list): Сроки погашения
        forecast_horizons_days (list): Горизонты прогнозирования

    Returns:
        pd.DataFrame: Сводка RMSE
    """
    model_name_prefixes_for_errors = ['ns_ar1', 'ns_var_level', 'ns_var_dl', 'ns_vecm', 'ns_catboost', 'rw']
    rmse_results_list = []

    required_cols = {'horizon_days', 'maturity_years'}
    if yield_forecasts_df is None or yield_forecasts_df.empty or not required_cols.issubset(yield_forecasts_df.columns):
        for horizon_val in forecast_horizons_days:
            for mat_y_val in eval_maturities_years:
                row = {'Horizon (days)': horizon_val, 'Maturity (years)': mat_y_val, 'N_obs': 0}
                for prefix in model_name_prefixes_for_errors:
                    row[_rmse_column_name(prefix)] = np.nan
                rmse_results_list.append(row)
        rmse_summary_df = pd.DataFrame(rmse_results_list)
        ordered_rmse_cols = [
            'Horizon (days)', 'Maturity (years)', 'N_obs',
            'RMSE_AR1', 'RMSE_VAR_LEVEL', 'RMSE_VAR_DL', 'RMSE_VECM', 'RMSE_CatBoost', 'RMSE_RW'
        ]
        print("RMSE для прогнозов доходности (Out-of-Sample, в п.п.):")
        print(rmse_summary_df[ordered_rmse_cols].to_string())
        return rmse_summary_df

    for horizon_val in forecast_horizons_days:
        for mat_y_val in eval_maturities_years:
            subset_for_rmse_calc = yield_forecasts_df[
                (yield_forecasts_df['horizon_days'] == horizon_val) &
                (yield_forecasts_df['maturity_years'] == mat_y_val)
            ]

            n_obs_for_table = len(subset_for_rmse_calc)
            res_rmse_dict = {'Horizon (days)': horizon_val, 'Maturity (years)': mat_y_val, 'N_obs': n_obs_for_table}

            if n_obs_for_table == 0:
                for prefix in model_name_prefixes_for_errors:
                    res_rmse_dict[_rmse_column_name(prefix)] = np.nan
            else:
                for prefix in model_name_prefixes_for_errors:
                    error_col = f'error_{prefix}'
                    rmse_col_name = _rmse_column_name(prefix)

                    if error_col in subset_for_rmse_calc.columns:
                        valid_errors_for_model = subset_for_rmse_calc[error_col].dropna()
                        if len(valid_errors_for_model) > 0:
                            res_rmse_dict[rmse_col_name] = np.sqrt(np.mean(valid_errors_for_model**2))
                        else:
                            res_rmse_dict[rmse_col_name] = np.nan
                    else:
                        res_rmse_dict[rmse_col_name] = np.nan
            rmse_results_list.append(res_rmse_dict)

    rmse_summary_df = pd.DataFrame(rmse_results_list)
    for prefix in model_name_prefixes_for_errors:
        expected_rmse_col = _rmse_column_name(prefix)
        if expected_rmse_col not in rmse_summary_df.columns:
            rmse_summary_df[expected_rmse_col] = np.nan

    ordered_rmse_cols = ['Horizon (days)', 'Maturity (years)', 'N_obs',
                        'RMSE_AR1', 'RMSE_VAR_LEVEL', 'RMSE_VAR_DL', 'RMSE_VECM', 'RMSE_CatBoost', 'RMSE_RW']
    existing_ordered_cols = [col for col in ordered_rmse_cols if col in rmse_summary_df.columns]

    print("RMSE для прогнозов доходности (Out-of-Sample, в п.п.):")
    print(rmse_summary_df[existing_ordered_cols].to_string())

    return rmse_summary_df


def merge_factor_forecasts(ar_factor_forecasts_df, var_factor_forecasts_df,
                          var_dl_factor_forecasts_df, vecm_factor_forecasts_df,
                          catboost_factor_forecasts_df=None):
    """
    Объединяет прогнозы факторов из разных моделей в один датафрейм

    Args:
        ar_factor_forecasts_df (pd.DataFrame): Прогнозы AR(1)
        var_factor_forecasts_df (pd.DataFrame): Прогнозы VAR в уровнях
        var_dl_factor_forecasts_df (pd.DataFrame): Прогнозы VAR_diff_level
        vecm_factor_forecasts_df (pd.DataFrame): Прогнозы VECM
        catboost_factor_forecasts_df (pd.DataFrame, optional): Прогнозы CatBoost

    Returns:
        pd.DataFrame: Объединенный датафрейм с прогнозами факторов
    """
    forecast_dfs_to_merge = [
        (ar_factor_forecasts_df, ['forecast_value_ar1']),
        (var_factor_forecasts_df, ['forecast_value_var_level']),
        (var_dl_factor_forecasts_df, ['forecast_value_var_dl']),
        (vecm_factor_forecasts_df, ['forecast_value_vecm'])
    ]

    # Добавляем CatBoost если доступен
    if catboost_factor_forecasts_df is not None and not catboost_factor_forecasts_df.empty:
        forecast_dfs_to_merge.append((catboost_factor_forecasts_df, ['forecast_value_catboost']))

    merged_fc_df = pd.DataFrame()

    # Находим первый непустой датафрейм для инициализации
    for idx, (df_candidate, fc_cols_candidate) in enumerate(forecast_dfs_to_merge):
        if not df_candidate.empty:
            cols_to_select = ['forecast_target_date', 'horizon_days', 'factor', 'forecast_made_on_date', 'actual_value'] + \
                             [fc_col for fc_col in fc_cols_candidate if fc_col in df_candidate.columns]
            merged_fc_df = df_candidate[cols_to_select].copy()
            base_df_id = idx
            break
    else:
        return pd.DataFrame()  # Все датафреймы пустые

    # Объединяем остальные датафреймы
    for idx, (df_to_merge, forecast_cols) in enumerate(forecast_dfs_to_merge):
        if idx == base_df_id or df_to_merge.empty:
            continue

        cols_for_current_merge = ['forecast_target_date', 'horizon_days', 'factor'] + \
                                 [fc_col for fc_col in forecast_cols if fc_col in df_to_merge.columns] + \
                                 ['actual_value', 'forecast_made_on_date']

        existing_cols_for_merge = [col for col in cols_for_current_merge if col in df_to_merge.columns]

        merged_fc_df = pd.merge(
            merged_fc_df,
            df_to_merge[existing_cols_for_merge],
            on=['forecast_target_date', 'horizon_days', 'factor'],
            how='outer',
            suffixes=('', '_y')
        )

        # Обрабатываем дублированные колонки
        for col_base in ['actual_value', 'forecast_made_on_date']:
            col_merged = col_base + '_y'
            if col_merged in merged_fc_df.columns:
                merged_fc_df[col_base] = merged_fc_df[col_base].fillna(merged_fc_df[col_merged])
                merged_fc_df.drop(columns=[col_merged], inplace=True)

    # Финальная обработка
    if not merged_fc_df.empty:
        key_cols_for_dedup = ['forecast_target_date', 'horizon_days', 'factor']
        for col_fill in ['actual_value', 'forecast_made_on_date']:
            if col_fill in merged_fc_df.columns:
                # Используем transform чтобы сохранить исходный индекс без переиндексации
                merged_fc_df[col_fill] = (
                    merged_fc_df
                    .groupby(key_cols_for_dedup, group_keys=False)[col_fill]
                    .transform(lambda x: x.ffill().bfill())
                    .reset_index(drop=True)
                )
        merged_fc_df.drop_duplicates(subset=key_cols_for_dedup, keep='first', inplace=True)

    return merged_fc_df


def analyze_tau1_results(all_tau1_rmse_results):
    """
    Анализ результатов для различных tau1

    Args:
        all_tau1_rmse_results (dict): Словарь с результатами RMSE для каждого tau1

    Returns:
        tuple: (combined_rmse_df, summary_df, best_tau1, best_avg_rmse)
    """
    combined_rmse_df = pd.concat(
        [df.assign(tau1=tau) for tau, df in all_tau1_rmse_results.items() if not df.empty],
        ignore_index=True
    )

    if combined_rmse_df.empty:
        raise ValueError("combined_rmse_df пуст: нет RMSE для визуализации.")

    summary_df = (
        combined_rmse_df.groupby('tau1')['rmse']
        .mean()
        .reset_index()
        .rename(columns={'rmse': 'avg_rmse'})
    )

    best_tau1_idx = summary_df['avg_rmse'].idxmin()
    best_tau1 = summary_df.loc[best_tau1_idx, 'tau1']
    best_avg_rmse = summary_df.loc[best_tau1_idx, 'avg_rmse']

    return combined_rmse_df, summary_df, best_tau1, best_avg_rmse
