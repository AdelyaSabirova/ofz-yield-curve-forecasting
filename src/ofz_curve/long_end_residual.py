from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nelson_siegel_svensson import NelsonSiegelCurve
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import statsmodels.api as sm

from .paths import (
    CBR_ZERO_CURVE_MONTHLY_CSV,
    LONG_END_RESIDUAL_DIR,
    LONG_END_RESIDUAL_FIGURE,
    PROJECT_ROOT,
)


DEFAULT_CORE_PATH = PROJECT_ROOT / "data" / "processed" / "core_2026" / "monthly_ns_factors_core.csv"
DEFAULT_MACRO_PATH = PROJECT_ROOT / "data" / "processed" / "macro_2026" / "monthly_expectations_panel.csv"
DEFAULT_OFFICIAL_CURVE_PATH = CBR_ZERO_CURVE_MONTHLY_CSV
DEFAULT_OUTPUT_DIR = LONG_END_RESIDUAL_DIR
DEFAULT_FINDINGS_PATH = PROJECT_ROOT / "docs" / "p4_long_end_residual_findings.md"
DEFAULT_FIGURE_PATH = LONG_END_RESIDUAL_FIGURE

DATE_COLUMN = "date"
ANALYSIS_START_DATE = pd.Timestamp("2018-04-01")
OLD_REGIME_END_DATE = pd.Timestamp("2022-01-01")

REGIME_ORDER = (
    ("pre_2022_02", "До февраля 2022 года", pd.Timestamp("2018-04-01"), pd.Timestamp("2022-01-01")),
    ("post_2022_02", "После февраля 2022 года", pd.Timestamp("2022-02-01"), None),
    ("stabilization_2024_plus", "Устойчивый режим 2024--2026", pd.Timestamp("2024-01-01"), None),
)


@dataclass(frozen=True)
class ResidualSpec:
    name: str
    label: str
    features: tuple[str, ...]


RESIDUAL_SPECS = (
    ResidualSpec(
        name="macro_only",
        label="Старая макромодель",
        features=("key_rate_pct", "cpi_yoy_lag1_pct", "usd_rub"),
    ),
    ResidualSpec(
        name="macro_plus_market_structure",
        label="Старая макромодель + структура рынка",
        features=(
            "key_rate_pct",
            "cpi_yoy_lag1_pct",
            "usd_rub",
            "breakeven_10y_pct",
            "ofz_nonresident_share_lag1_pct",
            "general_government_debt_growth_12m_lag1_pct",
        ),
    ),
)


TARGET_SPECS = (
    ("official", "Официальная КБД Банка России", "official_slope_10_1"),
    ("model", "Авторская кривая Нельсона--Сигеля", "model_slope_10_1"),
)


def _model_yield(beta0: float, beta1: float, beta2: float, tau1: float, maturity: float) -> float | None:
    if any(pd.isna(value) for value in (beta0, beta1, beta2, tau1)) or tau1 <= 0:
        return None
    curve = NelsonSiegelCurve(beta0=float(beta0), beta1=float(beta1), beta2=float(beta2), tau=float(tau1))
    return float(curve(maturity) * 100)


def load_long_end_panel(
    core_path: Path,
    macro_path: Path,
    official_curve_path: Path,
) -> pd.DataFrame:
    core = pd.read_csv(core_path, parse_dates=[DATE_COLUMN]).sort_values(DATE_COLUMN)
    macro = pd.read_csv(macro_path, parse_dates=[DATE_COLUMN]).sort_values(DATE_COLUMN)
    official = pd.read_csv(official_curve_path, parse_dates=[DATE_COLUMN]).sort_values(DATE_COLUMN)

    macro["cpi_yoy_lag1"] = macro["cpi_yoy"].shift(1)
    macro["ofz_nonresident_share_lag1"] = macro["ofz_nonresident_share"].shift(1)
    macro["general_government_debt_growth_12m_lag1"] = macro["general_government_debt_growth_12m"].shift(1)

    for source_column in (
        "key_rate",
        "cpi_yoy_lag1",
        "breakeven_10y",
        "ofz_nonresident_share_lag1",
        "general_government_debt_growth_12m_lag1",
    ):
        macro[f"{source_column}_pct"] = macro[source_column] * 100

    core["model_y1_00"] = [
        _model_yield(row.beta0, row.beta1, row.beta2, row.tau1, 1.0)
        for row in core[["beta0", "beta1", "beta2", "tau1"]].itertuples(index=False)
    ]
    core["model_y10_00"] = [
        _model_yield(row.beta0, row.beta1, row.beta2, row.tau1, 10.0)
        for row in core[["beta0", "beta1", "beta2", "tau1"]].itertuples(index=False)
    ]
    core["model_slope_10_1"] = core["model_y10_00"] - core["model_y1_00"]

    official = official.rename(columns={"y1_00": "official_y1_00", "y10_00": "official_y10_00"})
    official["official_slope_10_1"] = official["official_y10_00"] - official["official_y1_00"]

    panel = (
        core[[DATE_COLUMN, "beta0", "beta1", "beta2", "tau1", "model_slope_10_1"]]
        .merge(
            macro[
                [
                    DATE_COLUMN,
                    "key_rate",
                    "key_rate_pct",
                    "cpi_yoy",
                    "cpi_yoy_lag1",
                    "cpi_yoy_lag1_pct",
                    "usd_rub",
                    "breakeven_10y",
                    "breakeven_10y_pct",
                    "ofz_nonresident_share",
                    "ofz_nonresident_share_lag1",
                    "ofz_nonresident_share_lag1_pct",
                    "general_government_debt_growth_12m",
                    "general_government_debt_growth_12m_lag1",
                    "general_government_debt_growth_12m_lag1_pct",
                ]
            ],
            on=DATE_COLUMN,
            how="inner",
        )
        .merge(
            official[[DATE_COLUMN, "curve_source_date", "official_y1_00", "official_y10_00", "official_slope_10_1"]],
            on=DATE_COLUMN,
            how="inner",
        )
    )
    panel = panel.loc[panel[DATE_COLUMN] >= ANALYSIS_START_DATE].reset_index(drop=True)
    return panel


def _slice_regime(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    subset = frame.loc[frame[DATE_COLUMN] >= start].copy()
    if end is not None:
        subset = subset.loc[subset[DATE_COLUMN] <= end].copy()
    return subset


def _fit_old_regime_ols(frame: pd.DataFrame, target_column: str, features: tuple[str, ...]) -> Any:
    train = _slice_regime(frame, start=ANALYSIS_START_DATE, end=OLD_REGIME_END_DATE)
    train = train.dropna(subset=[target_column, *features]).copy()
    X = sm.add_constant(train[list(features)], has_constant="add")
    return sm.OLS(train[target_column], X).fit()


def _residual_summary_rows(
    residual_frame: pd.DataFrame,
    *,
    target_name: str,
    target_label: str,
    spec_name: str,
    spec_label: str,
    fit: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime_id, regime_label, regime_start, regime_end in REGIME_ORDER:
        subset = _slice_regime(residual_frame, start=regime_start, end=regime_end)
        residuals = subset["residual_pp"].dropna()
        if residuals.empty:
            continue
        t_stat = None
        p_value = None
        if regime_id != "pre_2022_02":
            t_stat_raw, p_value_raw = ttest_1samp(residuals, popmean=0.0, alternative="greater")
            t_stat = float(t_stat_raw)
            p_value = float(p_value_raw)
        rows.append(
            {
                "target_name": target_name,
                "target_label": target_label,
                "spec_name": spec_name,
                "spec_label": spec_label,
                "regime_id": regime_id,
                "regime_label": regime_label,
                "sample_start": subset[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                "sample_end": subset[DATE_COLUMN].max().strftime("%Y-%m-%d"),
                "n_obs": int(residuals.shape[0]),
                "train_r2": float(fit.rsquared),
                "mean_residual_pp": float(residuals.mean()),
                "median_residual_pp": float(residuals.median()),
                "std_residual_pp": float(residuals.std(ddof=1)),
                "mean_abs_residual_pp": float(residuals.abs().mean()),
                "positive_share": float((residuals > 0).mean()),
                "t_stat_positive": t_stat,
                "p_value_positive": p_value,
                "significant_positive_5pct": bool(p_value is not None and p_value < 0.05),
            }
        )
    return rows


def _coefficient_rows(
    fit: Any,
    *,
    target_name: str,
    target_label: str,
    spec_name: str,
    spec_label: str,
    train_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parameter_name, coefficient in fit.params.items():
        rows.append(
            {
                "target_name": target_name,
                "target_label": target_label,
                "spec_name": spec_name,
                "spec_label": spec_label,
                "parameter_name": parameter_name,
                "coefficient": float(coefficient),
                "p_value": float(fit.pvalues.get(parameter_name, np.nan)),
                "train_n": int(train_n),
                "train_r2": float(fit.rsquared),
            }
        )
    return rows


def evaluate_long_end_residuals(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    residual_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for target_name, target_label, target_column in TARGET_SPECS:
        for spec in RESIDUAL_SPECS:
            current = panel[[DATE_COLUMN, target_column, *spec.features]].dropna().copy()
            fit = _fit_old_regime_ols(frame=current, target_column=target_column, features=spec.features)
            X = sm.add_constant(current[list(spec.features)], has_constant="add")
            current["fitted_old_regime_pp"] = fit.predict(X)
            current["residual_pp"] = current[target_column] - current["fitted_old_regime_pp"]
            current["target_name"] = target_name
            current["target_label"] = target_label
            current["spec_name"] = spec.name
            current["spec_label"] = spec.label
            current = current.rename(columns={target_column: "actual_pp"})
            residual_rows.append(current)

            train_n = int(_slice_regime(current, start=ANALYSIS_START_DATE, end=OLD_REGIME_END_DATE).shape[0])
            summary_rows.extend(
                _residual_summary_rows(
                    current[[DATE_COLUMN, "residual_pp"]].copy(),
                    target_name=target_name,
                    target_label=target_label,
                    spec_name=spec.name,
                    spec_label=spec.label,
                    fit=fit,
                )
            )
            coefficient_rows.extend(
                _coefficient_rows(
                    fit,
                    target_name=target_name,
                    target_label=target_label,
                    spec_name=spec.name,
                    spec_label=spec.label,
                    train_n=train_n,
                )
            )

    residual_df = pd.concat(residual_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    coefficients_df = pd.DataFrame(coefficient_rows)
    return residual_df, summary_df, coefficients_df


def plot_long_end_residuals(
    residual_df: pd.DataFrame,
    figure_path: Path,
) -> Path:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_spec = residual_df.loc[residual_df["spec_name"] == "macro_only"].copy()
    official = plot_spec.loc[plot_spec["target_name"] == "official"].sort_values(DATE_COLUMN)
    author = plot_spec.loc[plot_spec["target_name"] == "model"].sort_values(DATE_COLUMN)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.1), sharex=True)

    axes[0].plot(
        official[DATE_COLUMN],
        official["actual_pp"],
        label="Официальный наклон 10Y-1Y",
        color="#1d3557",
        linewidth=2.2,
    )
    axes[0].plot(
        official[DATE_COLUMN],
        official["fitted_old_regime_pp"],
        label="Старая макромодель",
        color="#e76f51",
        linewidth=2.0,
        linestyle="--",
    )
    axes[0].set_ylabel("п.п.", labelpad=8)
    axes[0].set_title("Наклон длинного конца и его оценка по докризисной связи")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].plot(
        official[DATE_COLUMN],
        official["residual_pp"],
        label="Официальная КБД",
        color="#1d3557",
        linewidth=2.2,
    )
    axes[1].plot(
        author[DATE_COLUMN],
        author["residual_pp"],
        label="Авторская кривая",
        color="#457b9d",
        linewidth=2.0,
    )
    axes[1].axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
    axes[1].set_ylabel("п.п.", labelpad=8)
    axes[1].set_title("Остаточный наклон длинного конца относительно докризисной связи")
    axes[1].legend(frameon=False, loc="upper left")

    for axis in axes:
        axis.axvline(pd.Timestamp("2022-02-01"), color="#c1121f", linewidth=1.2, linestyle="--")
        axis.axvline(pd.Timestamp("2024-01-01"), color="#6c757d", linewidth=1.0, linestyle="--")
        axis.grid(alpha=0.25, linestyle=":")
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", labelbottom=True, labelrotation=35)
        axis.tick_params(axis="y", labelsize=10)

    axes[1].set_xlabel("Дата")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.12, hspace=0.36)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def build_findings_markdown(
    panel: pd.DataFrame,
    summary_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    figure_path: Path,
    output_dir: Path,
) -> str:
    def extract_summary(target_name: str, spec_name: str, regime_id: str) -> pd.Series:
        mask = (
            (summary_df["target_name"] == target_name)
            & (summary_df["spec_name"] == spec_name)
            & (summary_df["regime_id"] == regime_id)
        )
        return summary_df.loc[mask].iloc[0]

    official_post = extract_summary("official", "macro_plus_market_structure", "post_2022_02")
    official_stab = extract_summary("official", "macro_plus_market_structure", "stabilization_2024_plus")
    author_post = extract_summary("model", "macro_plus_market_structure", "post_2022_02")
    author_stab = extract_summary("model", "macro_plus_market_structure", "stabilization_2024_plus")
    macro_official_post = extract_summary("official", "macro_only", "post_2022_02")
    macro_official_stab = extract_summary("official", "macro_only", "stabilization_2024_plus")
    macro_author_post = extract_summary("model", "macro_only", "post_2022_02")
    macro_author_stab = extract_summary("model", "macro_only", "stabilization_2024_plus")

    def fmt(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "н/д"
        return f"{float(value):.2f}"

    def fmt_p(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "н/д"
        if float(value) < 0.001:
            return "< 0.001"
        return f"= {float(value):.3f}"

    top_coefficients = coefficients_df.loc[
        (coefficients_df["target_name"] == "official")
        & (coefficients_df["spec_name"] == "macro_plus_market_structure")
        & (coefficients_df["parameter_name"] != "const")
    ].copy()
    top_coefficients["abs_coefficient"] = top_coefficients["coefficient"].abs()
    top_coefficients = top_coefficients.sort_values("abs_coefficient", ascending=False).head(4)

    lines = [
        "# Поток 4б. Остаточный наклон длинного конца",
        "",
        "## Что сделано",
        "- Построен месячный ряд наклона длинного конца `10Y-1Y` по официальной КБД Банка России и по авторской кривой Нельсона--Сигеля.",
        "- На старом режиме `2018-04 .. 2022-01` оценены две линейные модели наклона: узкая макромодель и расширенная модель с наблюдаемой структурой рынка.",
        "- Узкая макромодель продолжается до `2026-02`, а расширенная --- до `2025-05`, поскольку 10-летняя рыночная инфляционная компенсация обрывается раньше остальных рядов.",
        "- Затем коэффициенты старого режима перенесены на весь доступный период, а разность между фактическим и оцененным наклоном интерпретируется как остаточный компонент длинного конца.",
        "",
        "## Главный результат",
        f"- По официальной КБД даже узкая модель, оцененная на докризисной связи, недооценивает наклон `10Y-1Y` после февраля 2022 года в среднем на `{fmt(macro_official_post['mean_residual_pp'])}` п.п. при доле положительных месяцев `{fmt(macro_official_post['positive_share'] * 100)}%` и одностороннем `p {fmt_p(macro_official_post['p_value_positive'])}`.",
        f"- В более устойчивом режиме `2024-2026` узкая модель оставляет положительный остаток `{fmt(macro_official_stab['mean_residual_pp'])}` п.п. по официальной КБД и `{fmt(macro_author_stab['mean_residual_pp'])}` п.п. по авторской кривой.",
        f"- Более строгая расширенная модель с наблюдаемой структурой рынка подтверждает тот же вывод на перекрытии до `2025-05`: по официальной КБД средний остаток после февраля 2022 года составляет `{fmt(official_post['mean_residual_pp'])}` п.п., а в режиме `2024-2025` `{fmt(official_stab['mean_residual_pp'])}` п.п.",
        f"- По авторской кривой знак и порядок величины сохраняются и в узкой, и в расширенной постановке: `{fmt(macro_author_post['mean_residual_pp'])}` п.п. в узкой модели после февраля 2022 года и `{fmt(author_post['mean_residual_pp'])}` п.п. в расширенной.",
        "",
        "## Как это читать",
        "- До 2022 года старая связь между коротким концом, инфляцией, курсом и длинным концом описывает наклон кривой приемлемо.",
        "- После 2022 года длинный конец систематически оказывается выше того, что следовало бы из старой макрофинансовой связи. Это и есть отдельный остаточный компонент длинного конца.",
        "- Такой результат не равен чистой сроковой премии в строгом безарбитражном смысле, но согласуется с идеей, что в новом режиме в длинном конце выросла роль дополнительной премии за риск и бюджетно-долгового контура.",
        "",
        "## Коэффициенты расширенной модели по официальной КБД",
    ]
    for row in top_coefficients.itertuples(index=False):
        lines.append(
            f"- `{row.parameter_name}`: коэффициент `{fmt(row.coefficient)}`, `p {fmt_p(row.p_value)}`."
        )
    lines.extend(
        [
            "",
            "## Артефакты",
            f"- Таблица остатков: `{output_dir / 'long_end_residual_series.csv'}`",
            f"- Сводка по режимам: `{output_dir / 'long_end_residual_summary.csv'}`",
            f"- Коэффициенты моделей: `{output_dir / 'long_end_residual_coefficients.csv'}`",
            f"- Рисунок: `{figure_path}`",
        ]
    )
    return "\n".join(lines)


def run_long_end_residual_analysis(
    core_path: Path = DEFAULT_CORE_PATH,
    macro_path: Path = DEFAULT_MACRO_PATH,
    official_curve_path: Path = DEFAULT_OFFICIAL_CURVE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    findings_path: Path = DEFAULT_FINDINGS_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    panel = load_long_end_panel(
        core_path=core_path,
        macro_path=macro_path,
        official_curve_path=official_curve_path,
    )
    residual_df, summary_df, coefficients_df = evaluate_long_end_residuals(panel=panel)

    residual_output = output_dir / "long_end_residual_series.csv"
    summary_output = output_dir / "long_end_residual_summary.csv"
    coefficients_output = output_dir / "long_end_residual_coefficients.csv"

    residual_df.to_csv(residual_output, index=False)
    summary_df.to_csv(summary_output, index=False)
    coefficients_df.to_csv(coefficients_output, index=False)
    figure_path = plot_long_end_residuals(residual_df=residual_df, figure_path=figure_path)

    findings_text = build_findings_markdown(
        panel=panel,
        summary_df=summary_df,
        coefficients_df=coefficients_df,
        figure_path=figure_path,
        output_dir=output_dir,
    )
    findings_path.write_text(findings_text, encoding="utf-8")

    metadata = {
        "coverage": {
            "panel_start": panel[DATE_COLUMN].min().strftime("%Y-%m-%d"),
            "panel_end": panel[DATE_COLUMN].max().strftime("%Y-%m-%d"),
            "old_regime_train_end": OLD_REGIME_END_DATE.strftime("%Y-%m-%d"),
        },
        "artifacts": {
            "residual_series_csv": str(residual_output),
            "summary_csv": str(summary_output),
            "coefficients_csv": str(coefficients_output),
            "figure": str(figure_path),
            "findings_markdown": str(findings_path),
        },
    }
    metadata_path = output_dir / "long_end_residual_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["artifacts"]["metadata_json"] = str(metadata_path)

    return {
        "panel": panel,
        "residuals": residual_df,
        "summary": summary_df,
        "coefficients": coefficients_df,
        "metadata": metadata,
        "findings_text": findings_text,
    }
