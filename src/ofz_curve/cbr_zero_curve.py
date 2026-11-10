"""
Загрузка официальной КБД Банка России и внешняя проверка собственной кривой.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import json
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from nelson_siegel_svensson import NelsonSiegelCurve
from pandas.api.types import is_datetime64_any_dtype

from .paths import (
    CBR_ZERO_CURVE_DAILY_CSV,
    CBR_ZERO_CURVE_MONTHLY_CSV,
    CORE_DAILY_NS_FACTORS_CSV,
    OFFICIAL_CURVE_VALIDATION_DIR,
    ASSETS_DIR,
)


CBR_ZERO_CURVE_URL = "https://www.cbr.ru/hd_base/zcyc_params/"
DEFAULT_START_DATE = pd.Timestamp("2003-01-01")
DEFAULT_REPORT_START_DATE = pd.Timestamp("2013-09-01")
DEFAULT_END_DATE = pd.Timestamp.today().normalize()
DEFAULT_FRESHNESS_TOLERANCE_DAYS = 7
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRIES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 1.5
# Внешняя проверка сосредоточена на устойчивом средне- и долгосрочном сегменте.
DEFAULT_COMPARISON_MATURITIES = (1.0, 3.0, 5.0, 10.0)
DEFAULT_FIGURE_PATH = ASSETS_DIR / "official_curve_validation_summary.png"

MATURITY_SPECS = (
    ("0,25", 0.25, "y0_25"),
    ("0,5", 0.50, "y0_50"),
    ("0,75", 0.75, "y0_75"),
    ("1", 1.00, "y1_00"),
    ("2", 2.00, "y2_00"),
    ("3", 3.00, "y3_00"),
    ("5", 5.00, "y5_00"),
    ("7", 7.00, "y7_00"),
    ("10", 10.00, "y10_00"),
    ("15", 15.00, "y15_00"),
    ("20", 20.00, "y20_00"),
    ("30", 30.00, "y30_00"),
)
MATURITY_LABEL_TO_COLUMN = {label: column for label, _, column in MATURITY_SPECS}
MATURITY_YEAR_TO_COLUMN = {years: column for _, years, column in MATURITY_SPECS}


@dataclass(frozen=True)
class YearSlice:
    start: pd.Timestamp
    end: pd.Timestamp


def _to_timestamp(value: pd.Timestamp | str | None, fallback: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return fallback
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(None)
    return stamp.normalize()


def _daterange_by_year(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[YearSlice]:
    slices: list[YearSlice] = []
    for year in range(start_date.year, end_date.year + 1):
        year_start = max(start_date, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end_date, pd.Timestamp(year=year, month=12, day=31))
        if year_start <= year_end:
            slices.append(YearSlice(start=year_start, end=year_end))
    return slices


def _build_cbr_query_url(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    return (
        f"{CBR_ZERO_CURVE_URL}?UniDbQuery.Posted=True"
        f"&UniDbQuery.From={start_date.strftime('%d.%m.%Y')}"
        f"&UniDbQuery.To={end_date.strftime('%d.%m.%Y')}"
    )


def _flatten_columns(columns: Iterable[object]) -> list[str]:
    flattened: list[str] = []
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part).strip() for part in column if str(part).strip() and not str(part).startswith("Unnamed")]
            flattened.append(parts[-1] if parts else "")
        else:
            flattened.append(str(column).strip())
    return flattened


def normalize_cbr_zero_curve_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(columns=["date", *MATURITY_LABEL_TO_COLUMN.values()])

    normalized = table.copy()
    normalized.columns = _flatten_columns(normalized.columns)
    normalized = normalized.rename(columns={"Дата": "date"})
    if "date" not in normalized.columns:
        raise ValueError("Не удалось найти колонку даты в таблице архива КБД Банка России.")

    available_labels = [label for label in MATURITY_LABEL_TO_COLUMN if label in normalized.columns]
    if not available_labels:
        raise ValueError("В таблице КБД не найдено ни одной колонки со сроком до погашения.")

    normalized = normalized[["date", *available_labels]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], dayfirst=True, errors="coerce")
    normalized = normalized.dropna(subset=["date"])

    rename_map = {label: output_name for label, _, output_name in MATURITY_SPECS}
    normalized = normalized.rename(columns=rename_map)

    for _, _, column in MATURITY_SPECS:
        if column not in normalized.columns:
            normalized[column] = np.nan
        series = pd.to_numeric(normalized[column], errors="coerce")
        non_null = series.dropna()
        if not non_null.empty and non_null.abs().median() > 100:
            series = series / 100.0
        normalized[column] = series

    ordered_columns = ["date", *MATURITY_LABEL_TO_COLUMN.values()]
    normalized = normalized[ordered_columns].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return normalized


def _fetch_cbr_zero_curve_slice(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    session: requests.Session | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    local_session = session or requests.Session()
    url = _build_cbr_query_url(start_date, end_date)
    headers = {"User-Agent": "nir-adelya-ofz-curve/1.0"}

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = local_session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            if not tables:
                return pd.DataFrame(columns=["date", *MATURITY_LABEL_TO_COLUMN.values()])
            return normalize_cbr_zero_curve_table(tables[0])
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt == retries:
                break
            time.sleep(DEFAULT_RETRY_SLEEP_SECONDS * attempt)

    raise RuntimeError(
        f"Не удалось загрузить КБД Банка России за период {start_date.date()}..{end_date.date()}"
    ) from last_error


def download_cbr_zero_curve_history(
    *,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    start = _to_timestamp(start_date, DEFAULT_START_DATE)
    end = _to_timestamp(end_date, DEFAULT_END_DATE)
    if start > end:
        raise ValueError("start_date не может быть позже end_date")

    session = requests.Session()
    chunks: list[pd.DataFrame] = []
    for year_slice in _daterange_by_year(start, end):
        chunk = _fetch_cbr_zero_curve_slice(
            year_slice.start,
            year_slice.end,
            session=session,
            retries=retries,
            timeout=timeout,
        )
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=["date", *MATURITY_LABEL_TO_COLUMN.values()])

    combined = (
        pd.concat(chunks, ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    return combined


def _load_cbr_zero_curve_csv(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path, parse_dates=["date"])
    daily = daily.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return daily


def ensure_cbr_zero_curve_daily(
    *,
    raw_daily_path: Path = CBR_ZERO_CURVE_DAILY_CSV,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    start = _to_timestamp(start_date, DEFAULT_START_DATE)
    end = _to_timestamp(end_date, DEFAULT_END_DATE)

    if raw_daily_path.exists() and not refresh:
        existing = _load_cbr_zero_curve_csv(raw_daily_path)
        if not existing.empty:
            max_date = pd.Timestamp(existing["date"].max())
            min_date = pd.Timestamp(existing["date"].min())
            if min_date <= start and max_date >= end - pd.Timedelta(days=DEFAULT_FRESHNESS_TOLERANCE_DAYS):
                return existing

    downloaded = download_cbr_zero_curve_history(start_date=start, end_date=end)
    raw_daily_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded.to_csv(raw_daily_path, index=False)
    return downloaded


def build_monthly_cbr_zero_curve(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "curve_source_date", *MATURITY_LABEL_TO_COLUMN.values()])

    ordered = daily.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    ordered = ordered.dropna(subset=["date"]).sort_values("date")
    ordered["_month"] = ordered["date"].dt.to_period("M").dt.to_timestamp()

    last_rows = ordered.groupby("_month", sort=True).tail(1).copy()
    last_rows.insert(1, "curve_source_date", last_rows["date"])
    last_rows["date"] = last_rows["_month"]
    last_rows = last_rows.drop(columns=["_month"]).reset_index(drop=True)
    return last_rows


def _build_model_curve_daily(
    core_daily: pd.DataFrame,
    maturities: tuple[float, ...],
) -> pd.DataFrame:
    required = {"date", "beta0", "beta1", "beta2", "tau1"}
    missing = required - set(core_daily.columns)
    if missing:
        raise ValueError(f"В дневном файле факторов отсутствуют обязательные колонки: {sorted(missing)}")

    rows: list[dict[str, float | pd.Timestamp]] = []
    for row in core_daily.itertuples(index=False):
        date = pd.Timestamp(getattr(row, "date"))
        tau1 = float(getattr(row, "tau1"))
        if not np.isfinite(tau1) or tau1 <= 0:
            continue
        curve = NelsonSiegelCurve(
            beta0=float(getattr(row, "beta0")),
            beta1=float(getattr(row, "beta1")),
            beta2=float(getattr(row, "beta2")),
            tau=tau1,
        )
        result_row: dict[str, float | pd.Timestamp] = {"date": date}
        for maturity in maturities:
            column = MATURITY_YEAR_TO_COLUMN[maturity]
            result_row[f"model_{column}"] = float(curve(maturity) * 100.0)
        rows.append(result_row)

    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date").reset_index(drop=True)


def build_curve_comparison(
    core_daily: pd.DataFrame,
    official_daily: pd.DataFrame,
    *,
    maturities: tuple[float, ...] = DEFAULT_COMPARISON_MATURITIES,
    report_start_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    start = _to_timestamp(report_start_date, DEFAULT_REPORT_START_DATE)

    valid_core = core_daily.copy()
    if "success" in valid_core.columns:
        valid_core = valid_core[valid_core["success"]].copy()
    valid_core["date"] = pd.to_datetime(valid_core["date"], errors="coerce")
    valid_core = valid_core.dropna(subset=["date"]).sort_values("date")

    official = official_daily.copy()
    official["date"] = pd.to_datetime(official["date"], errors="coerce")
    official = official.dropna(subset=["date"]).sort_values("date")

    model_curve = _build_model_curve_daily(valid_core, maturities)
    overlap = model_curve.merge(official, on="date", how="inner")
    overlap = overlap[overlap["date"] >= start].copy()

    rows: list[dict[str, object]] = []
    for maturity in maturities:
        maturity_column = MATURITY_YEAR_TO_COLUMN[maturity]
        model_column = f"model_{maturity_column}"
        official_column = maturity_column
        subset = overlap[["date", model_column, official_column]].dropna()
        for item in subset.itertuples(index=False):
            model_yield = float(getattr(item, model_column))
            official_yield = float(getattr(item, official_column))
            error_pp = model_yield - official_yield
            rows.append(
                {
                    "date": pd.Timestamp(item.date),
                    "maturity_years": maturity,
                    "model_yield": model_yield,
                    "official_yield": official_yield,
                    "error_pp": error_pp,
                    "abs_error_pp": abs(error_pp),
                }
            )

    comparison = pd.DataFrame(rows).sort_values(["date", "maturity_years"]).reset_index(drop=True)
    return comparison


def summarize_curve_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(
            columns=[
                "maturity_years",
                "n_obs",
                "mean_error_pp",
                "median_abs_error_pp",
                "rmse_pp",
                "correlation",
            ]
        )

    rows: list[dict[str, float | int]] = []
    for maturity, group in comparison.groupby("maturity_years", sort=True):
        errors = group["error_pp"].dropna()
        model = group["model_yield"].dropna()
        official = group["official_yield"].dropna()
        correlation = np.nan
        if len(group[["model_yield", "official_yield"]].dropna()) >= 2:
            correlation = float(group["model_yield"].corr(group["official_yield"]))

        rows.append(
            {
                "maturity_years": float(maturity),
                "n_obs": int(len(group)),
                "mean_error_pp": float(errors.mean()) if len(errors) else np.nan,
                "median_abs_error_pp": float(group["abs_error_pp"].median()) if len(group) else np.nan,
                "rmse_pp": float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
                "correlation": correlation,
            }
        )

    return pd.DataFrame(rows).sort_values("maturity_years").reset_index(drop=True)


def plot_curve_validation_summary(summary: pd.DataFrame, figure_path: Path) -> None:
    if summary.empty:
        return

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{value:g}" for value in summary["maturity_years"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(labels, summary["rmse_pp"], color="#1f77b4")
    axes[0].set_title("RMSE по срокам")
    axes[0].set_xlabel("Срок до погашения, лет")
    axes[0].set_ylabel("п.п.")

    axes[1].bar(labels, summary["median_abs_error_pp"], color="#ff7f0e")
    axes[1].set_title("Медианная абсолютная ошибка")
    axes[1].set_xlabel("Срок до погашения, лет")
    axes[1].set_ylabel("п.п.")

    fig.suptitle("Сопоставление собственной кривой ОФЗ и официальной КБД Банка России")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _serialize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    serialized = frame.copy()
    for column in serialized.columns:
        if is_datetime64_any_dtype(serialized[column]):
            serialized[column] = serialized[column].dt.strftime("%Y-%m-%d")
    return serialized


def run_cbr_zero_curve_validation(
    *,
    core_daily_path: Path | str = CORE_DAILY_NS_FACTORS_CSV,
    raw_daily_path: Path | str = CBR_ZERO_CURVE_DAILY_CSV,
    monthly_path: Path | str = CBR_ZERO_CURVE_MONTHLY_CSV,
    output_dir: Path | str = OFFICIAL_CURVE_VALIDATION_DIR,
    figure_path: Path | str = DEFAULT_FIGURE_PATH,
    start_date: pd.Timestamp | str | None = None,
    report_start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    refresh: bool = False,
) -> dict[str, object]:
    core_daily_path = Path(core_daily_path)
    raw_daily_path = Path(raw_daily_path)
    monthly_path = Path(monthly_path)
    output_dir = Path(output_dir)
    figure_path = Path(figure_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path.parent.mkdir(parents=True, exist_ok=True)

    official_daily = ensure_cbr_zero_curve_daily(
        raw_daily_path=raw_daily_path,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    official_monthly = build_monthly_cbr_zero_curve(official_daily)
    _serialize_dates(official_monthly).to_csv(monthly_path, index=False)

    core_daily = pd.read_csv(core_daily_path, parse_dates=["date"])
    comparison = build_curve_comparison(
        core_daily,
        official_daily,
        report_start_date=report_start_date,
    )
    summary = summarize_curve_comparison(comparison)
    plot_curve_validation_summary(summary, figure_path)

    daily_comparison_path = output_dir / "official_vs_model_daily.csv"
    summary_path = output_dir / "official_vs_model_summary.csv"
    metadata_path = output_dir / "official_vs_model_summary.json"

    _serialize_dates(comparison).to_csv(daily_comparison_path, index=False)
    summary.to_csv(summary_path, index=False)

    report_start = _to_timestamp(report_start_date, DEFAULT_REPORT_START_DATE)
    overlap_start = None
    overlap_end = None
    if not comparison.empty:
        overlap_start = pd.Timestamp(comparison["date"].min()).strftime("%Y-%m-%d")
        overlap_end = pd.Timestamp(comparison["date"].max()).strftime("%Y-%m-%d")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_archive": {
            "source": CBR_ZERO_CURVE_URL,
            "start_date": pd.Timestamp(official_daily["date"].min()).strftime("%Y-%m-%d")
            if not official_daily.empty
            else None,
            "end_date": pd.Timestamp(official_daily["date"].max()).strftime("%Y-%m-%d")
            if not official_daily.empty
            else None,
            "raw_daily_csv": str(raw_daily_path),
            "monthly_csv": str(monthly_path),
        },
        "comparison": {
            "report_start_date": report_start.strftime("%Y-%m-%d"),
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "maturities_years": list(DEFAULT_COMPARISON_MATURITIES),
            "daily_rows": int(len(comparison)),
            "summary_rows": int(len(summary)),
            "artifacts": {
                "daily_comparison_csv": str(daily_comparison_path),
                "summary_csv": str(summary_path),
                "summary_figure": str(figure_path),
            },
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "official_daily_path": raw_daily_path,
        "official_monthly_path": monthly_path,
        "daily_comparison_path": daily_comparison_path,
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "figure_path": figure_path,
        "metadata": metadata,
    }
