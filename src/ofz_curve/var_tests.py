"""
Вспомогательные тесты для VAR/VECM анализа.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def adf_test(series, name: str, alpha: float = 0.05, autolag: str = "AIC") -> bool:
    """
    Расширенный тест Дики-Фуллера.
    Возвращает `True`, если ряд стационарен на уровне `alpha`.
    """

    ser = pd.Series(series).dropna()
    if ser.empty:
        print(f"\nADF тест для {name}: ряд пустой после удаления NaN")
        return False

    result = adfuller(ser, autolag=autolag)

    print(f"\nADF тест для {name}:")
    print(f"ADF статистика: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Критические значения:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.4f}")

    stationary = bool(result[1] <= alpha)
    if stationary:
        print(f"Результат: {name} стационарен (отвергаем H0)")
    else:
        print(f"Результат: {name} нестационарен (не отвергаем H0)")
    return stationary


def granger_causality_test(
    data,
    optimal_lags: int | None = None,
    max_lags: int = 4,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """
    Проводит тесты причинности по Грейнджеру для всех пар переменных.
    """

    df = pd.DataFrame(data).dropna()
    variables = df.columns.tolist()
    results: dict[str, dict[str, Any]] = {}

    for i, target in enumerate(variables):
        for j, source in enumerate(variables):
            if i == j:
                continue

            test_data = df[[target, source]].dropna()
            key = f"{source} → {target}"

            if len(test_data) < max_lags + 2:
                results[key] = {
                    "p_value": np.nan,
                    "significant": False,
                    "lag_used": None,
                    "error": "not enough observations",
                }
                continue

            try:
                gc_test = grangercausalitytests(test_data, max_lags, verbose=False)
                lag_used = (
                    min(int(optimal_lags), max_lags)
                    if optimal_lags is not None
                    else max_lags
                )
                lag_used = max(1, lag_used)
                p_value = float(gc_test[lag_used][0]["ssr_ftest"][1])
                results[key] = {
                    "p_value": p_value,
                    "significant": bool(p_value < alpha),
                    "lag_used": lag_used,
                }
            except Exception as exc:
                print(f"Ошибка в тесте {key}: {exc}")
                results[key] = {
                    "p_value": np.nan,
                    "significant": False,
                    "lag_used": None,
                    "error": str(exc),
                }

    return results


def interpret_johansen_result(
    johansen_result,
    alpha_level: str = "95%",
) -> dict[str, Any]:
    """
    Красивый вывод и краткая интерпретация результатов теста Йохансена.

    Возвращает словарь с решениями по trace/max-eig и оценкой ранга.
    """

    alpha_to_idx = {"90%": 0, "95%": 1, "99%": 2}
    if alpha_level not in alpha_to_idx:
        raise ValueError("alpha_level must be one of: '90%', '95%', '99%'")
    crit_idx = alpha_to_idx[alpha_level]

    trace_stats = np.asarray(
        getattr(johansen_result, "trace_stat", getattr(johansen_result, "lr1"))
    )
    trace_crit = np.asarray(
        getattr(
            johansen_result,
            "trace_stat_crit_vals",
            getattr(johansen_result, "cvt"),
        )
    )
    maxeig_stats = np.asarray(
        getattr(johansen_result, "max_eig_stat", getattr(johansen_result, "lr2", []))
    )
    maxeig_crit = np.asarray(
        getattr(johansen_result, "max_eig_stat_crit_vals", getattr(johansen_result, "cvm", []))
    )

    trace_decisions = []
    print("\nРезультаты теста Йохансена (Trace statistic):")
    for r, stat in enumerate(trace_stats):
        crit = float(trace_crit[r][crit_idx])
        reject = bool(stat > crit)
        trace_decisions.append(
            {
                "rank_null": r,
                "stat": float(stat),
                "crit": crit,
                "reject_h0": reject,
            }
        )
        verdict = "отвергаем H0" if reject else "не отвергаем H0"
        print(f"r <= {r}: stat={float(stat):.4f}, crit({alpha_level})={crit:.4f} -> {verdict}")

    maxeig_decisions = []
    if len(maxeig_stats) and len(maxeig_crit):
        print("\nРезультаты теста Йохансена (Max-Eigen statistic):")
        for r, stat in enumerate(maxeig_stats):
            crit = float(maxeig_crit[r][crit_idx])
            reject = bool(stat > crit)
            maxeig_decisions.append(
                {
                    "rank_null": r,
                    "stat": float(stat),
                    "crit": crit,
                    "reject_h0": reject,
                }
            )
            verdict = "отвергаем H0" if reject else "не отвергаем H0"
            print(f"r = {r}: stat={float(stat):.4f}, crit({alpha_level})={crit:.4f} -> {verdict}")

    trace_rank = 0
    for row in trace_decisions:
        if row["reject_h0"]:
            trace_rank = row["rank_null"] + 1
        else:
            break

    maxeig_rank = None
    if maxeig_decisions:
        rank_val = 0
        for row in maxeig_decisions:
            if row["reject_h0"]:
                rank_val = row["rank_null"] + 1
            else:
                break
        maxeig_rank = rank_val

    print("\nИтоговая интерпретация:")
    print(f"Оценка коинтеграционного ранга по Trace ({alpha_level}): {trace_rank}")
    if maxeig_rank is not None:
        print(f"Оценка коинтеграционного ранга по Max-Eigen ({alpha_level}): {maxeig_rank}")

    return {
        "alpha_level": alpha_level,
        "trace_rank": trace_rank,
        "max_eig_rank": maxeig_rank,
        "trace_decisions": trace_decisions,
        "max_eig_decisions": maxeig_decisions,
    }

