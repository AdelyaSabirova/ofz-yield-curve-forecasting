from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from .paths import CBR_ZERO_CURVE_MONTHLY_CSV, POLICY_PATH_GAP_DIR, POLICY_PATH_GAP_FIGURE, PROJECT_ROOT


DATE_COLUMN = "date"
SURVEY_YEAR_START = 2021
SURVEY_YEAR_END = 2028
DEFAULT_SURVEY_WORKBOOK_PATH = PROJECT_ROOT / "data" / "raw" / "macro" / "expectations" / "cbr_macro_survey_full.xlsx"
DEFAULT_OFFICIAL_CURVE_PATH = CBR_ZERO_CURVE_MONTHLY_CSV
DEFAULT_OUTPUT_DIR = POLICY_PATH_GAP_DIR
DEFAULT_FINDINGS_PATH = PROJECT_ROOT / "docs" / "p4_policy_path_gap_findings.md"
DEFAULT_FIGURE_PATH = POLICY_PATH_GAP_FIGURE
CURRENT_DATE = pd.Timestamp(date.today())

REGIME_ORDER = (
    ("pre_2022_02", "До февраля 2022 года", pd.Timestamp("2021-05-01"), pd.Timestamp("2022-01-01")),
    ("transition_2022_2023", "Переходный режим 2022--2023", pd.Timestamp("2022-02-01"), pd.Timestamp("2023-12-01")),
    ("post_2022_02", "После февраля 2022 года", pd.Timestamp("2022-02-01"), None),
    ("stabilization_2024_plus", "Устойчивый режим 2024--2026", pd.Timestamp("2024-01-01"), None),
)


@dataclass(frozen=True)
class GapSpec:
    column: str
    label: str


GAP_SPECS = (
    GapSpec(column="nominal_gap_2y_pp", label="2-летняя доходность"),
)


def _to_month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _normalize_fractional_percent(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.abs().median(skipna=True)
    if pd.notna(median) and median > 100:
        numeric = numeric / 100.0
    return numeric / 100.0


def _month_range_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="MS")


def parse_survey_target_year(
    workbook_path: Path,
    *,
    sheet_name: str,
    column_name: str,
    target_year: int,
    panel_start: pd.Timestamp,
    panel_end: pd.Timestamp,
) -> pd.DataFrame:
    sheet = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)
    survey_dates = _to_month_start(pd.Series(sheet.iloc[5, 4:].tolist()))
    label_col_1 = sheet.iloc[:, 1].astype(str)
    label_col_2 = sheet.iloc[:, 2].astype(str)
    median_idx_candidates = sheet.index[label_col_2.str.contains("Median", na=False)].tolist()
    if not median_idx_candidates:
        raise ValueError(f"Не найден блок Median в листе {sheet_name}")
    median_start = median_idx_candidates[0]
    stop_idx = len(sheet)
    for idx in range(median_start + 1, len(sheet)):
        label_text = f"{label_col_1.iloc[idx]} {label_col_2.iloc[idx]}"
        if "Average" in label_text or "Среднее" in label_text:
            stop_idx = idx
            break
    median_block = sheet.iloc[median_start:stop_idx].copy()
    target_row = None
    for _, row in median_block.iterrows():
        marker = row.iloc[3]
        if isinstance(marker, datetime) and marker.year == target_year:
            target_row = row
            break
    if target_row is None:
        raise ValueError(f"В листе {sheet_name} нет строки для {target_year} года")
    target_values = _normalize_fractional_percent(pd.Series(target_row.iloc[4:].tolist()))
    output = pd.DataFrame({DATE_COLUMN: survey_dates.tolist(), column_name: target_values.tolist()})
    output = output.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN)
    output = output[(output[DATE_COLUMN] >= panel_start) & (output[DATE_COLUMN] <= panel_end)]
    output = output.drop_duplicates(subset=[DATE_COLUMN], keep="last")
    return output


def expand_irregular_monthly_series(
    frame: pd.DataFrame,
    *,
    value_column: str,
    panel_start: pd.Timestamp,
    panel_end: pd.Timestamp,
) -> pd.DataFrame:
    monthly = pd.DataFrame({DATE_COLUMN: _month_range_index(panel_start, panel_end)})
    monthly = monthly.merge(frame[[DATE_COLUMN, value_column]], on=DATE_COLUMN, how="left")
    monthly[value_column] = monthly[value_column].ffill()
    return monthly


def build_monthly_survey_panel(
    workbook_path: Path,
    *,
    panel_start: pd.Timestamp | None = None,
    panel_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    resolved_start = panel_start or pd.Timestamp("2021-05-01")
    resolved_end = panel_end or (CURRENT_DATE.to_period("M") - 1).to_timestamp()
    panel = pd.DataFrame({DATE_COLUMN: _month_range_index(resolved_start, resolved_end)})
    for year in range(SURVEY_YEAR_START, SURVEY_YEAR_END + 1):
        key_rate_raw = parse_survey_target_year(
            workbook_path,
            sheet_name="3",
            column_name=f"analyst_key_rate_{year}",
            target_year=year,
            panel_start=resolved_start,
            panel_end=resolved_end,
        )
        cpi_raw = parse_survey_target_year(
            workbook_path,
            sheet_name="1",
            column_name=f"analyst_cpi_{year}",
            target_year=year,
            panel_start=resolved_start,
            panel_end=resolved_end,
        )
        key_rate_monthly = expand_irregular_monthly_series(
            key_rate_raw,
            value_column=f"analyst_key_rate_{year}",
            panel_start=resolved_start,
            panel_end=resolved_end,
        )
        cpi_monthly = expand_irregular_monthly_series(
            cpi_raw,
            value_column=f"analyst_cpi_{year}",
            panel_start=resolved_start,
            panel_end=resolved_end,
        )
        panel = panel.merge(key_rate_monthly, on=DATE_COLUMN, how="left").merge(cpi_monthly, on=DATE_COLUMN, how="left")
    return panel


def _weighted_forward_average(row: pd.Series, *, prefix: str, horizon_months: int) -> float | None:
    current_date = row[DATE_COLUMN]
    current_year = int(current_date.year)
    current_month = int(current_date.month)
    remaining_horizon = horizon_months
    weights: list[int] = []
    values: list[float] = []

    current_year_months = min(12 - current_month, remaining_horizon)
    if current_year_months > 0:
        weights.append(current_year_months)
        values.append(row.get(f"{prefix}_{current_year}"))
        remaining_horizon -= current_year_months

    next_year = current_year + 1
    while remaining_horizon > 0:
        current_weight = min(12, remaining_horizon)
        weights.append(current_weight)
        values.append(row.get(f"{prefix}_{next_year}"))
        remaining_horizon -= current_weight
        next_year += 1

    if any(pd.isna(value) for value in values):
        return None
    return float(np.average(values, weights=weights))


def _slice_regime(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    subset = frame.loc[frame[DATE_COLUMN] >= start].copy()
    if end is not None:
        subset = subset.loc[subset[DATE_COLUMN] <= end].copy()
    return subset


def build_policy_path_gap_panel(
    survey_workbook_path: Path,
    official_curve_path: Path,
) -> pd.DataFrame:
    official = pd.read_csv(official_curve_path, parse_dates=[DATE_COLUMN]).sort_values(DATE_COLUMN)
    survey = build_monthly_survey_panel(
        survey_workbook_path,
        panel_start=official[DATE_COLUMN].min(),
        panel_end=official[DATE_COLUMN].max(),
    )

    survey["analyst_key_rate_24m"] = survey.apply(
        _weighted_forward_average,
        axis=1,
        prefix="analyst_key_rate",
        horizon_months=24,
    )
    survey["analyst_cpi_24m"] = survey.apply(
        _weighted_forward_average,
        axis=1,
        prefix="analyst_cpi",
        horizon_months=24,
    )
    survey["analyst_real_rate_24m"] = survey["analyst_key_rate_24m"] - survey["analyst_cpi_24m"]
    survey["analyst_key_rate_24m_pct"] = survey["analyst_key_rate_24m"] * 100.0
    survey["analyst_cpi_24m_pct"] = survey["analyst_cpi_24m"] * 100.0
    survey["analyst_real_rate_24m_pct"] = survey["analyst_real_rate_24m"] * 100.0

    panel = official.merge(
        survey[
            [
                DATE_COLUMN,
                "analyst_key_rate_24m",
                "analyst_cpi_24m",
                "analyst_real_rate_24m",
                "analyst_key_rate_24m_pct",
                "analyst_cpi_24m_pct",
                "analyst_real_rate_24m_pct",
            ]
        ],
        on=DATE_COLUMN,
        how="left",
    )
    panel["nominal_gap_2y_pp"] = panel["y2_00"] - panel["analyst_key_rate_24m_pct"]
    return panel


def summarize_policy_path_gap(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gap_spec in GAP_SPECS:
        for regime_id, regime_label, regime_start, regime_end in REGIME_ORDER:
            subset = _slice_regime(panel, start=regime_start, end=regime_end)
            series = subset[gap_spec.column].dropna()
            if series.empty:
                continue
            t_stat_raw, p_value_raw = ttest_1samp(series, popmean=0.0, alternative="greater")
            rows.append(
                {
                    "gap_column": gap_spec.column,
                    "gap_label": gap_spec.label,
                    "regime_id": regime_id,
                    "regime_label": regime_label,
                    "sample_start": subset.loc[subset[gap_spec.column].notna(), DATE_COLUMN].min().strftime("%Y-%m-%d"),
                    "sample_end": subset.loc[subset[gap_spec.column].notna(), DATE_COLUMN].max().strftime("%Y-%m-%d"),
                    "n_obs": int(series.shape[0]),
                    "mean_gap_pp": float(series.mean()),
                    "median_gap_pp": float(series.median()),
                    "std_gap_pp": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
                    "positive_share": float((series > 0).mean()),
                    "t_stat_positive": float(t_stat_raw),
                    "p_value_positive": float(p_value_raw),
                    "significant_positive_5pct": bool(p_value_raw < 0.05),
                }
            )
    return pd.DataFrame(rows)


def _format_p_value(value: float) -> str:
    if pd.isna(value):
        return "н/д"
    if value < 0.001:
        return "<0.001"
    return f"={value:.3f}"


def write_findings(summary: pd.DataFrame, findings_path: Path) -> None:
    def pick(gap_column: str, regime_id: str) -> pd.Series:
        return summary.loc[
            (summary["gap_column"] == gap_column) & (summary["regime_id"] == regime_id)
        ].iloc[0]

    stable_2 = pick("nominal_gap_2y_pp", "stabilization_2024_plus")
    post_2 = pick("nominal_gap_2y_pp", "post_2022_02")
    transition_2 = pick("nominal_gap_2y_pp", "transition_2022_2023")

    lines = [
        "# Поток 4е. Сопоставление 2-летней доходности ОФЗ с опросной траекторией ключевой ставки",
        "",
        "- Для каждого месяца из ежегодных прогнозов аналитиков Банка России собрана приближенная средняя ключевая ставка на ближайшие 24 месяца: использовались оставшиеся месяцы текущего года и два следующих календарных года с весами по числу месяцев.",
        "- Для основного сопоставления использована именно 2-летняя доходность официальной КБД, потому что только у нее горизонт близок к 24-месячной опросной траектории; прямое сравнение 5- и 10-летних доходностей с таким ориентиром было бы методически слабее.",
        (
            "- В устойчивом режиме 2024--2026 2-летняя официальная доходность превышает эту "
            f"опросную траекторию в среднем на {stable_2['mean_gap_pp']:.2f} п.п."
        ),
        (
            "- Доля месяцев с положительным разрывом в 2024--2026 составляет "
            f"{stable_2['positive_share'] * 100:.1f}%, а односторонний тест дает "
            f"p{_format_p_value(stable_2['p_value_positive'])}."
        ),
        (
            "- Уже в переходном режиме 2022--2023 разрыв остается положительным и составляет "
            f"{transition_2['mean_gap_pp']:.2f} п.п., а на всей послешоковой выборке "
            f"после февраля 2022 года он составляет {post_2['mean_gap_pp']:.2f} п.п."
        ),
        "- Интерпретация: даже при почти совпадающем горизонте 2-летняя рыночная доходность остается выше опросной средней траектории ключевой ставки. Значит, новый рыночный компонент после 2022 года нельзя свести только к ожиданию того, что ставка долго останется высокой.",
    ]
    findings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_policy_path_gap(panel: pd.DataFrame, figure_path: Path) -> None:
    figure_end = panel.loc[panel["analyst_key_rate_24m"].notna(), DATE_COLUMN].max()
    subset = panel.loc[(panel[DATE_COLUMN] >= pd.Timestamp("2024-01-01")) & (panel[DATE_COLUMN] <= figure_end)].copy()
    if subset.empty:
        raise ValueError("Недостаточно данных для рисунка разрыва относительно опросной траектории ставки")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True)

    axes[0].plot(subset[DATE_COLUMN], subset["y2_00"], label="Официальная КБД, 2 года", linewidth=2.2, color="#1f77b4")
    axes[0].plot(
        subset[DATE_COLUMN],
        subset["analyst_key_rate_24m_pct"],
        label="Опросная средняя ключевая ставка на 24 месяца",
        linewidth=2.0,
        color="#2ca02c",
    )
    axes[0].set_ylabel("% годовых", labelpad=10)
    axes[0].set_title("2-летняя доходность ОФЗ и опросная 24-месячная траектория ставки")
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].plot(
        subset[DATE_COLUMN],
        subset["nominal_gap_2y_pp"],
        label="Разрыв 2 года",
        linewidth=2.2,
        color="#1f77b4",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].axvline(pd.Timestamp("2024-01-01"), color="#7f7f7f", linewidth=1.0, linestyle=":")
    axes[1].set_ylabel("п.п.", labelpad=10)
    axes[1].set_title("Разрыв 2-летней доходности относительно опросной 24-месячной траектории ставки")
    axes[1].legend(loc="upper right", frameon=True)

    for axis in axes:
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", labelbottom=True, labelrotation=45, labelsize=9)
        axis.tick_params(axis="y", labelsize=10)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.11, right=0.985, top=0.94, bottom=0.14, hspace=0.38)
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)


def _coverage(panel: pd.DataFrame) -> dict[str, Any]:
    non_null = panel.loc[panel["analyst_key_rate_24m"].notna(), DATE_COLUMN]
    return {
        "panel_start": panel[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "panel_end": panel[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "survey_path_start": non_null.min().strftime("%Y-%m-%d"),
        "survey_path_end": non_null.max().strftime("%Y-%m-%d"),
    }


def run_policy_path_gap_analysis(
    *,
    survey_workbook_path: Path = DEFAULT_SURVEY_WORKBOOK_PATH,
    official_curve_path: Path = DEFAULT_OFFICIAL_CURVE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    findings_path: Path = DEFAULT_FINDINGS_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, Any]:
    if not survey_workbook_path.exists():
        raise FileNotFoundError(f"Не найден файл макроэкономического опроса: {survey_workbook_path}")
    if not official_curve_path.exists():
        raise FileNotFoundError(f"Не найден месячный ряд официальной КБД: {official_curve_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    panel = build_policy_path_gap_panel(
        survey_workbook_path=survey_workbook_path,
        official_curve_path=official_curve_path,
    )
    summary = summarize_policy_path_gap(panel)

    panel_path = output_dir / "policy_path_gap_panel.csv"
    summary_path = output_dir / "policy_path_gap_summary.csv"
    metadata_path = output_dir / "policy_path_gap_metadata.json"

    panel.to_csv(panel_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_findings(summary, findings_path)
    plot_policy_path_gap(panel, figure_path)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "coverage": _coverage(panel),
        "artifacts": {
            "panel_csv": str(panel_path),
            "summary_csv": str(summary_path),
            "figure": str(figure_path),
            "findings_markdown": str(findings_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "panel": panel,
        "summary": summary,
        "metadata": metadata,
    }
