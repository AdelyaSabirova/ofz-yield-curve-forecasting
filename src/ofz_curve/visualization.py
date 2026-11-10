"""
Функции для визуализации результатов анализа
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from .config import *


def _format_tau1_label(tau1_value):
    tau1_float = float(tau1_value)
    if tau1_float.is_integer():
        return f"{tau1_float:.1f}"
    return f"{tau1_float:g}"


def _select_representative_tau1_values(summary_df, best_tau1, max_values=5):
    tau1_values = sorted(float(value) for value in summary_df["tau1"].dropna().unique())
    if len(tau1_values) <= max_values:
        return tau1_values

    best_index = min(range(len(tau1_values)), key=lambda idx: abs(tau1_values[idx] - float(best_tau1)))
    anchor_indices = [
        0,
        round((len(tau1_values) - 1) * 0.25),
        round((len(tau1_values) - 1) * 0.50),
        best_index,
        len(tau1_values) - 1,
    ]

    representative_values = []
    for idx in anchor_indices:
        tau1_value = tau1_values[int(idx)]
        if tau1_value not in representative_values:
            representative_values.append(tau1_value)
    return representative_values


def plot_rmse_comparison(combined_rmse_df, summary_df, best_tau1, best_avg_rmse, save_path=None):
    """
    Визуализация сравнения RMSE по различным tau1

    Args:
        combined_rmse_df (pd.DataFrame): Объединенный DataFrame с RMSE
        summary_df (pd.DataFrame): Суммарная статистика по tau1
        best_tau1 (float): Лучшее значение tau1
        best_avg_rmse (float): Лучшее среднее RMSE
        save_path (str | Path | None): Путь для сохранения рисунка
    """
    def set_ylim_with_margin(ax, values, frac=0.3, min_pad=0.05):
        vals = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            return
        spread = float(vals.max() - vals.min())
        pad = max(min_pad, spread * frac)
        ax.set_ylim(float(vals.min()) - pad, float(vals.max()) + pad)

    fig, axes = plt.subplots(SUBPLOT_SIZE[0], SUBPLOT_SIZE[1], figsize=FIGURE_SIZE)
    ax1, ax2, ax3, ax4 = axes.flatten()

    representative_tau1_values = _select_representative_tau1_values(summary_df, best_tau1)
    comparison_palette = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#b279a2"]
    comparison_colors = {}
    color_idx = 0
    for tau1_value in representative_tau1_values:
        if np.isclose(float(tau1_value), float(best_tau1)):
            comparison_colors[tau1_value] = "#c1121f"
            continue
        comparison_colors[tau1_value] = comparison_palette[color_idx % len(comparison_palette)]
        color_idx += 1

    # График 1: Средняя RMSE по tau1
    bar_colors = [
        "#c1121f" if np.isclose(float(tau1_value), float(best_tau1)) else "#2c7fb8"
        for tau1_value in summary_df["tau1"]
    ]
    ax1.bar(summary_df["tau1"].map(_format_tau1_label), summary_df["avg_rmse"], color=bar_colors)
    ax1.set_title('Средняя RMSE по значениям tau1')
    ax1.set_xlabel('tau1')
    ax1.set_ylabel('RMSE (п.п.)')
    ax1.grid(True, alpha=0.3)
    set_ylim_with_margin(ax1, summary_df['avg_rmse'], frac=0.4)

    # График 2: RMSE по горизонтам для репрезентативных tau1
    horizon_vals = []
    for tau1 in representative_tau1_values:
        tau1_data = combined_rmse_df[combined_rmse_df['tau1'] == tau1]
        if tau1_data.empty:
            continue
        horizon_means = tau1_data.groupby('horizon_days')['rmse'].mean()
        horizon_vals.extend(horizon_means.values)
        is_best_tau1 = np.isclose(float(tau1), float(best_tau1))
        ax2.plot(
            horizon_means.index,
            horizon_means.values,
            marker='o',
            label=f'tau1={_format_tau1_label(tau1)}',
            linewidth=3.0 if is_best_tau1 else 1.8,
            markersize=7 if is_best_tau1 else 5,
            color=comparison_colors[tau1],
            alpha=1.0 if is_best_tau1 else 0.9,
            zorder=5 if is_best_tau1 else 3,
        )

    ax2.set_title('RMSE по горизонтам прогнозирования')
    ax2.set_xlabel('Горизонт (дни)')
    ax2.set_ylabel('RMSE (п.п.)')
    ax2.legend(title='Показанные tau1')
    ax2.grid(True, alpha=0.3)
    set_ylim_with_margin(ax2, horizon_vals, frac=0.25)

    # График 3: RMSE по срокам погашения для репрезентативных tau1
    maturity_vals = []
    for tau1 in representative_tau1_values:
        tau1_data = combined_rmse_df[combined_rmse_df['tau1'] == tau1]
        if tau1_data.empty:
            continue
        maturity_means = tau1_data.groupby('maturity_years')['rmse'].mean()
        maturity_vals.extend(maturity_means.values)
        is_best_tau1 = np.isclose(float(tau1), float(best_tau1))
        ax3.plot(
            maturity_means.index,
            maturity_means.values,
            marker='s',
            label=f'tau1={_format_tau1_label(tau1)}',
            linewidth=3.0 if is_best_tau1 else 1.8,
            markersize=7 if is_best_tau1 else 5,
            color=comparison_colors[tau1],
            alpha=1.0 if is_best_tau1 else 0.9,
            zorder=5 if is_best_tau1 else 3,
        )

    ax3.set_title('RMSE по срокам погашения')
    ax3.set_xlabel('Срок погашения (лет)')
    ax3.set_ylabel('RMSE (п.п.)')
    ax3.legend(title='Показанные tau1')
    ax3.grid(True, alpha=0.3)
    set_ylim_with_margin(ax3, maturity_vals, frac=0.25)

    # График 4: Тепловая карта для лучшего tau1
    best_tau1_data = combined_rmse_df[combined_rmse_df['tau1'] == best_tau1]
    if best_tau1_data.empty:
        ax4.text(0.5, 0.5, f'Нет данных RMSE для tau1 = {best_tau1}',
                ha='center', va='center', transform=ax4.transAxes)
    else:
        pivot_table = best_tau1_data.pivot_table(
            values='rmse',
            index='horizon_days',
            columns='maturity_years'
        )
        im = ax4.imshow(pivot_table.values, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=ax4, label='RMSE (п.п.)')
        ax4.set_title(f'Тепловая карта RMSE для tau1={_format_tau1_label(best_tau1)}')
        ax4.set_xlabel('Срок погашения')
        ax4.set_ylabel('Горизонт (дни)')
        ax4.set_xticks(range(len(pivot_table.columns)), [f'{col}y' for col in pivot_table.columns])
        ax4.set_yticks(range(len(pivot_table.index)), pivot_table.index)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches='tight')
    backend_name = plt.get_backend().lower()
    if "agg" in backend_name:
        plt.close(fig)
    else:
        plt.show()

    # Сводная таблица по лучшему tau1
    if not best_tau1_data.empty:
        print(f"🏆 ЛУЧШЕЕ ЗНАЧЕНИЕ TAU1: {_format_tau1_label(best_tau1)} (средний RMSE = {best_avg_rmse:.4f})")
        print("Сводная таблица RMSE по лучшему tau1:")
        print(pivot_table)

        # Рекомендации
        print("📊 РЕКОМЕНДАЦИИ:")
        print(f"1. Оптимальное значение tau1: {_format_tau1_label(best_tau1)}")
        print(f"2. Средняя RMSE: {best_avg_rmse:.4f} п.п.")

        theoretical_optimal = 2.5 / 1.789
        print(f"3. Теоретический оптимум: {theoretical_optimal:.3f}")

        if abs(best_tau1 - theoretical_optimal) < 0.3:
            print("✅ Эмпирический оптимум близок к теоретическому!")
        else:
            print("⚠️ Эмпирический оптимум отличается от теоретического")


def plot_irf_yields(irf_yield_df, horizons_days):
    """
    Визуализация импульсных откликов доходностей на шоки нефти и валюты

    Args:
        irf_yield_df (pd.DataFrame): DataFrame с результатами IRF
        horizons_days (list): Горизонты прогнозирования
    """
    shock_list = sorted(irf_yield_df["shock"].unique())
    fig, axes = plt.subplots(len(shock_list), len(horizons_days), figsize=(12, 4 * len(shock_list)), sharey=True)
    if len(shock_list) == 1:
        axes = np.array([axes])
    if len(horizons_days) == 1:
        axes = axes.reshape(len(shock_list), 1)
    colors = {"ln_oil": "#1f77b4", "ln_usd": "#ff7f0e"}
    for i, shock in enumerate(shock_list):
        for j, h in enumerate(horizons_days):
            ax = axes[i, j]
            subset = irf_yield_df[(irf_yield_df["shock"] == shock) & (irf_yield_df["horizon_days"] == h)]
            ax.bar(subset["maturity_years"].astype(str), subset["delta_yield_pp"], color=colors.get(shock, "#444"))
            ax.set_title(f"{shock}: h={h}д")
            ax.set_xlabel("Срок (лет)")
            ax.set_ylabel("Δ доходности, п.п.")
            ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_yield_forecast_rmse(rmse_summary_df_v4, eval_maturities_years, forecast_horizons_days):
    """
    Визуализация RMSE прогнозов доходности по всем моделям

    Args:
        rmse_summary_df_v4 (pd.DataFrame): DataFrame с RMSE по моделям
        eval_maturities_years (list): Сроки погашения
        forecast_horizons_days (list): Горизонты прогнозирования
    """
    if not rmse_summary_df_v4.empty:
        model_plot_configs = {
            'RMSE_AR1': {'marker': 'o', 'linestyle': '-', 'color': 'blue', 'label': 'NS-AR1'},
            'RMSE_VAR_LEVEL': {'marker': 's', 'linestyle': '--', 'color': 'green', 'label': 'NS-VAR_lev'},
            'RMSE_VAR_DL': {'marker': '^', 'linestyle': '-.', 'color': 'red', 'label': 'NS-VAR_dl'},
            'RMSE_VECM': {'marker': 'D', 'linestyle': ':', 'color': 'purple', 'label': 'NS-VECM'},
            'RMSE_CatBoost': {'marker': '*', 'linestyle': ':', 'color': 'orange', 'label': 'NS-CatBoost'},
            'RMSE_RW': {'marker': 'x', 'linestyle': '--', 'color': 'black', 'label': 'Random Walk'}
        }

        for mat_y_plot in eval_maturities_years:
            plt.figure(figsize=(12, 7))

            subset_maturity_data = rmse_summary_df_v4[rmse_summary_df_v4['Maturity (years)'] == mat_y_plot]

            if subset_maturity_data.empty:
                print(f"Нет данных RMSE для срока погашения {mat_y_plot}y для построения графика.")
                plt.close()
                continue

            any_model_plotted_on_current_fig = False
            for rmse_col_name, config in model_plot_configs.items():
                if rmse_col_name in subset_maturity_data.columns and subset_maturity_data[rmse_col_name].notna().any():
                    plt.plot(subset_maturity_data['Horizon (days)'], subset_maturity_data[rmse_col_name],
                             marker=config['marker'],
                             linestyle=config['linestyle'],
                             color=config['color'],
                             label=config['label'],
                             linewidth=1.5, markersize=7)
                    any_model_plotted_on_current_fig = True

            if not any_model_plotted_on_current_fig:
                print(f"Нет валидных данных моделей для срока погашения {mat_y_plot}y для построения графика.")
                plt.close()
                continue

            plt.title(f'Сравнение RMSE для срока погашения: {mat_y_plot}y', fontsize=16)
            plt.xlabel('Горизонт прогнозирования (торговых дней)', fontsize=14)
            plt.ylabel('RMSE (процентные пункты доходности)', fontsize=14)

            horizon_labels = [f'{int(h//21)}м' if h%21==0 and h//21 > 0 else f'{int(h)}д' for h in forecast_horizons_days]
            if len(forecast_horizons_days) == 1 and forecast_horizons_days[0] == 21:
                 horizon_labels = ['1м']
            plt.xticks(forecast_horizons_days, horizon_labels, fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylim(bottom=0)
            plt.legend(title="Типы моделей", fontsize=10, title_fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.show()

    else:
        print("Нет данных для построения графика RMSE")


def plot_factor_forecasts_combined(merged_fc_df, factor='beta0', horizon_days=21,
                                   start_date=None, end_date=None,
                                   last_n_points=250, thin_every=2):
    """
    Один график для фактора: фактические значения vs прогнозы всех моделей + столбцы RMSE/MAE.

    Args:
        merged_fc_df (pd.DataFrame): объединенный датафрейм из merge_factor_forecasts
        factor (str): фактор ('beta0'/'beta1'/'beta2')
        horizon_days (int): горизонт прогнозирования
        start_date (str|pd.Timestamp|None): опционально ограничить начало периода
        end_date (str|pd.Timestamp|None): опционально ограничить конец периода
        last_n_points (int): взять хвост последних N точек для читаемости
        thin_every (int): брать каждую k-ую точку для отображения линий (снижение шумности)
    """
    if merged_fc_df is None or merged_fc_df.empty:
        print("Нет данных для визуализации прогнозов факторов.")
        return

    df = merged_fc_df.copy()
    df = df[(df['factor'] == factor) & (df['horizon_days'] == horizon_days)].copy()
    if df.empty:
        print(f"Нет данных для фактора {factor} и горизонта {horizon_days} дней.")
        return

    if start_date is not None:
        df = df[df['forecast_target_date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['forecast_target_date'] <= pd.to_datetime(end_date)]

    df = df.sort_values('forecast_target_date')
    df.set_index('forecast_target_date', inplace=True)

    # Ограничиваем хвостом, чтобы график был читабельным
    if last_n_points is not None and last_n_points > 0 and len(df) > last_n_points:
        df = df.tail(last_n_points)

    # Проредим для отрисовки, если нужно
    if thin_every and thin_every > 1 and len(df) > thin_every:
        df_plot = df.iloc[::thin_every].copy()
    else:
        df_plot = df

    model_cols = {
        'AR(1)': 'forecast_value_ar1',
        'VAR_level': 'forecast_value_var_level',
        'VAR_diff_level': 'forecast_value_var_dl',
        'VECM': 'forecast_value_vecm',
        'CatBoost': 'forecast_value_catboost'
    }

    available_models = {name: col for name, col in model_cols.items() if col in df.columns}
    if not available_models:
        print("В merged_fc_df нет прогнозов моделей для отрисовки.")
        return

    actual = df['actual_value']

    def calc_metrics(actual_series, pred_series):
        mask = actual_series.notna() & pred_series.notna()
        if mask.sum() == 0:
            return np.nan, np.nan
        errors = pred_series[mask] - actual_series[mask]
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        mae = float(np.mean(np.abs(errors)))
        return rmse, mae

    metrics = {}
    for name, col in available_models.items():
        rmse, mae = calc_metrics(actual, df[col])
        metrics[name] = {'rmse': rmse, 'mae': mae}

    colors = {
        'Actual': '#000000',
        'AR(1)': '#1f77b4',
        'VAR_level': '#2ca02c',
        'VAR_diff_level': '#d62728',
        'VECM': '#9467bd',
        'CatBoost': '#ff7f0e'
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=False,
        gridspec_kw={'height_ratios': [3.2, 1]}
    )

    # Верхний график: фактические vs прогнозы
    actual_mask = df_plot['actual_value'].notna()
    if actual_mask.any():
        ax1.plot(df_plot.index[actual_mask], df_plot['actual_value'][actual_mask],
                 label='Факт', color=colors['Actual'], linewidth=2.0, zorder=5)
    for name, col in available_models.items():
        series = df_plot[col]
        mask = series.notna()
        if not mask.any():
            continue
        label = name
        if not np.isnan(metrics[name]['rmse']):
            label += f" (RMSE={metrics[name]['rmse']:.3f}, MAE={metrics[name]['mae']:.3f})"
        # Если всего одна точка - рисуем scatter, иначе линию
        if mask.sum() == 1:
            ax1.scatter(series.index[mask], series[mask],
                        label=label, color=colors.get(name, '#ff7f0e'),
                        s=30, zorder=4)
        else:
            ax1.plot(series.index[mask], series[mask],
                     label=label, color=colors.get(name, '#ff7f0e'),
                     linewidth=1.6, alpha=0.95, zorder=4)

    ax1.set_title(f'Фактор {factor}: факт и прогнозы (h={horizon_days} дн.)')
    ax1.set_ylabel('Уровень фактора')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Нижний график: столбцы RMSE и MAE по моделям (горизонтальные для читаемости)
    model_names = list(available_models.keys())
    rmse_vals = [metrics[m]['rmse'] for m in model_names]
    mae_vals = [metrics[m]['mae'] for m in model_names]
    y = np.arange(len(model_names))
    height = 0.35

    ax2.barh(y - height / 2, rmse_vals, height, label='RMSE', color='#4c78a8')
    ax2.barh(y + height / 2, mae_vals, height, label='MAE', color='#f58518')

    ax2.set_yticks(y)
    ax2.set_yticklabels(model_names)
    ax2.set_xlabel('Ошибка')
    ax2.set_title('Ошибки моделей на совпадающих датах')
    ax2.grid(True, axis='x', linestyle=':', alpha=0.6)
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax2.legend(loc='lower right')

    xmax = max([v for v in rmse_vals + mae_vals if not np.isnan(v)] + [0]) * 1.3 if model_names else 0.01
    ax2.set_xlim(0, xmax if xmax > 0 else 0.01)

    # подписи на конце столбцов
    for yi, (r, m) in enumerate(zip(rmse_vals, mae_vals)):
        if not np.isnan(r):
            ax2.text(r + xmax * 0.015, yi - height / 2, f"{r:.3f}", va='center', fontsize=8)
        if not np.isnan(m):
            ax2.text(m + xmax * 0.015, yi + height / 2, f"{m:.3f}", va='center', fontsize=8)

    plt.tight_layout()
    plt.show()
