"""
Функции для автоподбора оптимального параметра tau1 модели Nelson-Siegel
"""

import numpy as np
from scipy.optimize import minimize_scalar
from .config import *


def optimize_tau1(df_filtered, tau1_values_to_test=None):
    """
    Автоматический подбор статичной tau1 для модели Nelson-Siegel

    Args:
        df_filtered (pd.DataFrame): Отфильтрованные данные ОФЗ
        tau1_values_to_test (list, optional): Предварительные значения tau1 для тестирования

    Returns:
        list: Оптимальные значения tau1 для тестирования
    """
    if tau1_values_to_test and len(tau1_values_to_test) > 0:
        print(f"Используются заданные tau1: {tau1_values_to_test}")
        return tau1_values_to_test

    print("Запуск автоподбора tau1...")

    rng = np.random.default_rng(TAU1_OPTIM_SEED)
    unique_trade_dates = np.array(sorted(df_filtered['TRADEDATE'].unique()))
    sample_size = min(len(unique_trade_dates), TAU1_OPTIM_SAMPLE_DATES)
    sample_dates = rng.choice(unique_trade_dates, size=sample_size, replace=False) if sample_size else np.array([])

    grouped = df_filtered.groupby('TRADEDATE')
    sampled_groups = [grouped.get_group(dt) for dt in sample_dates if len(grouped.get_group(dt)) >= MIN_DATA_POINTS_FOR_FIT]

    def tau1_fit_rmse(tau1_value: float) -> float:
        """
        Функция потерь для оптимизации tau1
        """
        if tau1_value is None or tau1_value <= 1e-4:
            return np.inf
        total_sq = 0.0
        total_obs = 0
        for df_day in sampled_groups:
            ttm_years = df_day['TTM_years'].values
            yields = df_day['YIELDCLOSE'].values / 100.0

            tau_div = ttm_years / tau1_value
            exp_term = np.exp(-tau_div)
            load_beta1 = np.where(
                np.abs(tau_div) < 1e-6,
                1.0 - tau_div / 2.0 + tau_div**2 / 6.0,
                (1 - exp_term) / tau_div
            )
            load_beta2 = load_beta1 - exp_term

            X = np.vstack([np.ones_like(ttm_years), load_beta1, load_beta2]).T
            beta_params, _, rank, _ = np.linalg.lstsq(X, yields, rcond=None)
            if rank < X.shape[1]:
                continue

            fitted = X @ beta_params
            resid = yields - fitted
            total_sq += float(np.sum(resid ** 2))
            total_obs += len(resid)

        return np.sqrt(total_sq / total_obs) if total_obs else np.inf

    if len(sampled_groups) == 0:
        best_tau1_auto = TAU0_INITIAL_GUESS
        tau1_values_to_test = [round(float(best_tau1_auto), 4)]
        print(f"Недостаточно данных для автоподбора tau1, используем TAU0_INITIAL_GUESS={best_tau1_auto}")
        return tau1_values_to_test

    rmse_cache = {}
    def rmse_cached(tau_val: float) -> float:
        if tau_val in rmse_cache:
            return rmse_cache[tau_val]
        rmse_val = tau1_fit_rmse(tau_val)
        rmse_cache[tau_val] = rmse_val
        return rmse_val

    # Грубая сетка по диапазону
    grid_taus = np.linspace(TAU1_SEARCH_BOUNDS[0], TAU1_SEARCH_BOUNDS[1], TAU1_OPTIM_GRID_SIZE)
    grid_rmses = [(float(t), rmse_cached(float(t))) for t in grid_taus]
    grid_rmses = [(t, r) for t, r in grid_rmses if np.isfinite(r)]

    # Точечная оптимизация для получения глобального минимума
    optim_res = minimize_scalar(tau1_fit_rmse, bounds=TAU1_SEARCH_BOUNDS, method='bounded')
    if optim_res.success and np.isfinite(optim_res.fun):
        rmse_cached(float(optim_res.x))
    else:
        print("Автоподбор tau1 не сошёлся, используем TAU0_INITIAL_GUESS в качестве кандидата")
        rmse_cached(float(TAU0_INITIAL_GUESS))

    # Сортируем по RMSE, берём победителя + 4 случайных из просмотренных
    sorted_candidates = [
        (t, r) for t, r in sorted(rmse_cache.items(), key=lambda kv: kv[1])
        if np.isfinite(r)
    ]

    if not sorted_candidates:
        best_tau1_auto = TAU0_INITIAL_GUESS
        tau1_values_to_test = [round(float(best_tau1_auto), 4)]
        print("Не удалось получить валидные кандидаты tau1, используем TAU0_INITIAL_GUESS")
        return tau1_values_to_test

    best_tau1_auto, best_rmse_auto = sorted_candidates[0]
    other_taus = [t for t, _ in sorted_candidates[1:]]
    num_rand = min(4, len(other_taus))
    random_taus = list(rng.choice(other_taus, size=num_rand, replace=False)) if num_rand else []
    tau1_values_to_test = [round(float(best_tau1_auto), 4)] + [round(float(t), 4) for t in random_taus]

    print(
        f"Автоподбор tau1 завершён. Победитель: {tau1_values_to_test[0]} (RMSE={best_rmse_auto:.5f}). "
        f"Добавлены случайные tau1 из просмотренных: {tau1_values_to_test[1:]}. "
        f"Дат в оценке: {len(sampled_groups)}"
    )

    return tau1_values_to_test

