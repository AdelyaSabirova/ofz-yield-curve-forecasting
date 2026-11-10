from __future__ import annotations

import concurrent.futures
import html
import io
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ofz_curve.config import MIN_DATA_POINTS_FOR_FIT, TAU1_SEARCH_BOUNDS  # noqa: E402
from download_ofz_history import (  # noqa: E402
    HistoryDownloadError,
    get_historical_trades_and_info,
    get_ofz_list,
    get_static_bond_info,
)


RAW_OFZ_PATH = ROOT / "data" / "raw" / "market" / "ofz" / "ofz_full_history.csv"
RAW_CBR_CPI_PATH = ROOT / "data" / "raw" / "macro" / "cbr" / "ipc_mes_05-2025.xlsx"
RAW_CBR_INFL_RATE_PATH = (
    ROOT
    / "data"
    / "raw"
    / "macro"
    / "cbr"
    / "Инфляция и ключевая ставка Банка России_F17_09_2013_T06_06_2025.xlsx"
)
RAW_USD_PATH = ROOT / "data" / "raw" / "macro" / "usd" / "RC_F01_01_2012_T06_12_2025.xlsx"

OUT_DIR = ROOT / "data" / "processed" / "core_2026"
DAILY_NS_PATH = OUT_DIR / "daily_ns_factors_core.csv"
MONTHLY_NS_PATH = OUT_DIR / "monthly_ns_factors_core.csv"
FIXED_TAU1_NS_PATH = OUT_DIR / "daily_ns_factors_fixed_tau1.csv"
SAMPLE_SPLITS_PATH = OUT_DIR / "sample_splits.json"

CBR_INFLATION_PAGE_URL = "https://www.cbr.ru/hd_base/infl/"
CBR_USD_XML_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
USD_SERIES_ID = "R01235"

START_OFZ_DATE = pd.Timestamp("2012-02-13")
START_CBR_INFLATION_DATE = date(2013, 9, 17)
START_USD_DATE = date(2012, 1, 1)
STATIC_TAU1 = 3.6
MAX_MOEX_WORKERS = 8
MOEX_RETRY_ROUNDS = 2
REQUEST_TIMEOUT = 60
PROGRESS_EVERY_DATES = 250

# Monthly CPI updates are based on official Rosstat monthly releases.
# Direct automated access to rosstat.gov.ru is unstable in the current environment,
# so the latest official values are pinned here for reproducibility.
MANUAL_CPI_MOM_UPDATES = [
    {
        "date": "2025-06-01",
        "cpi_mom": 0.0020,
        "source_label": "Rosstat monthly CPI release for June 2025",
    },
    {
        "date": "2025-07-01",
        "cpi_mom": 0.0057,
        "source_label": "Rosstat monthly CPI release for July 2025",
    },
    {
        "date": "2025-08-01",
        "cpi_mom": -0.0040,
        "source_label": "Rosstat monthly CPI release for August 2025",
    },
    {
        "date": "2025-09-01",
        "cpi_mom": 0.0034,
        "source_label": "Rosstat monthly CPI release for September 2025",
    },
    {
        "date": "2025-10-01",
        "cpi_mom": 0.0050,
        "source_label": "Rosstat monthly CPI release for October 2025",
    },
    {
        "date": "2025-11-01",
        "cpi_mom": 0.0042,
        "source_label": "Rosstat monthly CPI release for November 2025",
    },
    {
        "date": "2025-12-01",
        "cpi_mom": 0.0032,
        "source_label": "Rosstat monthly CPI release for December 2025",
    },
    {
        "date": "2026-01-01",
        "cpi_mom": 0.0162,
        "source_label": "Rosstat monthly CPI release for January 2026",
    },
]


@dataclass
class UpdateSummary:
    ofz_rows_added: int
    ofz_last_date: str
    ofz_failed_secids: list[str]
    cpi_last_date: str
    key_rate_last_date: str | None
    usd_last_date: str | None
    daily_last_date: str
    monthly_last_date: str
    monthly_last_curve_source_date: str


def _iso_date(value: pd.Timestamp | datetime | date | str | None) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_cbr_download_url(start_date: date, end_date: date) -> str:
    response = requests.get(CBR_INFLATION_PAGE_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    match = re.search(
        r'href="(?P<href>/Queries/UniDbQuery/DownloadExcel/\d+\?[^"]+)"',
        response.text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise RuntimeError("Could not locate CBR inflation export link.")

    href = html.unescape(match.group("href"))
    absolute_url = urljoin(CBR_INFLATION_PAGE_URL, href)
    parsed = urlparse(absolute_url)
    query = parse_qs(parsed.query)
    query["FromDate"] = [start_date.strftime("%m/%d/%Y")]
    query["ToDate"] = [end_date.strftime("%m/%d/%Y")]
    query["posted"] = ["False"]
    updated_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=updated_query))


def update_cbr_inflation_and_key_rate(today: date) -> pd.DataFrame:
    download_url = _build_cbr_download_url(
        start_date=START_CBR_INFLATION_DATE,
        end_date=today,
    )
    response = requests.get(download_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    _ensure_parent(RAW_CBR_INFL_RATE_PATH)
    RAW_CBR_INFL_RATE_PATH.write_bytes(response.content)

    df = pd.read_excel(io.BytesIO(response.content))
    df = df.rename(
        columns={
            "Дата": "date_token",
            "Ключевая ставка, % годовых": "key_rate",
            "Инфляция, % г/г": "cpi_yoy",
        }
    )
    if "date_token" not in df.columns:
        raise RuntimeError("CBR export does not contain the expected date column.")

    df["date_token"] = (
        df["date_token"]
        .astype(str)
        .str.strip()
        .str.replace(r"^(\d{1,2})\.202$", r"\1.2020", regex=True)
    )
    df["date"] = pd.to_datetime(df["date_token"], format="%m.%Y", errors="coerce")
    df["key_rate"] = pd.to_numeric(df["key_rate"], errors="coerce")
    df["cpi_yoy"] = pd.to_numeric(df["cpi_yoy"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return df[["date", "key_rate", "cpi_yoy"]]


def update_cpi_monthly_workbook() -> pd.DataFrame:
    existing = pd.read_excel(RAW_CBR_CPI_PATH, sheet_name="a")
    existing = existing.rename(columns={str(col).strip().lower(): str(col).strip().lower() for col in existing.columns})
    if "date" not in existing.columns or "cpi_mom" not in existing.columns:
        raise RuntimeError("Monthly CPI workbook does not contain sheet 'a' with columns date and cpi_mom.")

    base = existing[["date", "cpi_mom"]].copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base["cpi_mom"] = pd.to_numeric(base["cpi_mom"], errors="coerce")

    manual = pd.DataFrame(MANUAL_CPI_MOM_UPDATES)
    manual["date"] = pd.to_datetime(manual["date"])
    manual["cpi_mom"] = pd.to_numeric(manual["cpi_mom"], errors="coerce")

    updated = (
        pd.concat([base[["date", "cpi_mom"]], manual[["date", "cpi_mom"]]], ignore_index=True)
        .dropna(subset=["date", "cpi_mom"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )

    with pd.ExcelWriter(
        RAW_CBR_CPI_PATH,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        updated.to_excel(writer, sheet_name="a", index=False)
        manual.to_excel(writer, sheet_name="manual_updates_2026", index=False)

    return updated


def update_usd_workbook(today: date) -> pd.DataFrame:
    xml_url = (
        f"{CBR_USD_XML_URL}?date_req1={START_USD_DATE.strftime('%d/%m/%Y')}"
        f"&date_req2={today.strftime('%d/%m/%Y')}&VAL_NM_RQ={USD_SERIES_ID}"
    )
    df = pd.read_xml(xml_url, xpath="//Record")
    if df.empty:
        raise RuntimeError("CBR USD XML response is empty.")

    df["data"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["nominal"] = pd.to_numeric(df["Nominal"], errors="coerce")
    df["curs"] = pd.to_numeric(df["Value"].astype(str).str.replace(",", "."), errors="coerce")
    out = (
        df.rename(columns={"Date": "date_token"})[["nominal", "data", "curs"]]
        .dropna(subset=["data", "curs"])
        .sort_values("data", ascending=False)
        .reset_index(drop=True)
    )
    out["cdx"] = "Доллар США"

    _ensure_parent(RAW_USD_PATH)
    with pd.ExcelWriter(RAW_USD_PATH, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="RC", index=False)

    return out


def _fetch_incremental_history(secid: str, start_date: str) -> tuple[str, list[dict], str | None]:
    try:
        rows = get_historical_trades_and_info(secid, start_date=start_date)
        return secid, rows, None
    except HistoryDownloadError as exc:
        return secid, [], str(exc)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return secid, [], f"{type(exc).__name__}: {exc}"


def _enrich_new_ofz_rows(new_history: pd.DataFrame) -> pd.DataFrame:
    if new_history.empty:
        return new_history

    static_payload: list[dict] = []
    secids = sorted(new_history["SECID"].dropna().unique().tolist())
    for index, secid in enumerate(secids, start=1):
        if index % 20 == 0 or index == len(secids):
            print(f"[OFZ] static {index}/{len(secids)}")
        static_payload.append(get_static_bond_info(secid))
        time.sleep(0.2)

    static_df = pd.DataFrame(static_payload)
    merged = pd.merge(new_history, static_df, on="SECID", how="left")
    merged["ISSUESIZEPLACED"] = pd.to_numeric(merged["ISSUESIZEPLACED"], errors="coerce")
    merged["ISSUESIZEPLACED"] = merged["ISSUESIZEPLACED"].replace(0, np.nan)
    merged["TURNOVER_RATIO"] = pd.to_numeric(merged["VOLUME"], errors="coerce") / merged["ISSUESIZEPLACED"]
    merged["TURNOVER_RATIO"] = merged["TURNOVER_RATIO"].replace([np.inf, -np.inf], np.nan)
    return merged


def update_ofz_history(today: date) -> tuple[pd.DataFrame, dict]:
    if not RAW_OFZ_PATH.exists():
        raise FileNotFoundError(f"Missing OFZ raw file: {RAW_OFZ_PATH}")

    existing = pd.read_csv(RAW_OFZ_PATH, parse_dates=["TRADEDATE", "MATDATE"])
    existing["TRADEDATE"] = pd.to_datetime(existing["TRADEDATE"], errors="coerce")
    last_date = pd.Timestamp(existing["TRADEDATE"].max())
    next_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[OFZ] existing last trade date: {last_date.date()}")

    secids = get_ofz_list()
    if not secids:
        secids = sorted(existing["SECID"].dropna().unique().tolist())
        print(f"[OFZ] fallback to {len(secids)} SECIDs from existing raw history")
    failed: list[str] = []
    collected: list[dict] = []
    pending_secids = secids

    for round_no in range(1, MOEX_RETRY_ROUNDS + 1):
        target_secids = pending_secids
        failed = []
        if round_no > 1:
            print(f"[OFZ] retry round {round_no} for {len(target_secids)} secids")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_MOEX_WORKERS) as executor:
            futures = {
                executor.submit(_fetch_incremental_history, secid, next_date): secid
                for secid in target_secids
            }
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                secid = futures[future]
                _, rows, error_message = future.result()
                if rows:
                    collected.extend(rows)
                if error_message:
                    failed.append(secid)
                if index % 15 == 0 or index == len(target_secids):
                    print(
                        f"[OFZ] round {round_no}: {index}/{len(target_secids)} "
                        f"completed, rows added so far: {len(collected)}, failed: {len(failed)}"
                    )
        if not failed:
            break
        pending_secids = failed

    if collected:
        new_history = (
            pd.DataFrame(collected)
            .drop_duplicates(subset=["SECID", "TRADEDATE"], keep="last")
            .sort_values(["SECID", "TRADEDATE"])
        )
        new_history = _enrich_new_ofz_rows(new_history)
        combined = pd.concat([existing, new_history], ignore_index=True, sort=False)
    else:
        new_history = pd.DataFrame(columns=existing.columns)
        combined = existing.copy()

    combined["TRADEDATE"] = pd.to_datetime(combined["TRADEDATE"], errors="coerce")
    combined["MATDATE"] = pd.to_datetime(combined["MATDATE"], errors="coerce")
    for column in ["NEXTCOUPON_hist", "NEXTCOUPON_static"]:
        if column in combined.columns:
            combined[column] = pd.to_datetime(combined[column], errors="coerce")

    combined = combined.drop_duplicates(subset=["SECID", "TRADEDATE"], keep="last")
    combined = combined.sort_values(["SECID", "TRADEDATE"]).reset_index(drop=True)

    ordered_columns = existing.columns.tolist()
    for column in combined.columns:
        if column not in ordered_columns:
            ordered_columns.append(column)
    combined = combined[ordered_columns]

    _ensure_parent(RAW_OFZ_PATH)
    combined.to_csv(RAW_OFZ_PATH, index=False)

    summary = {
        "rows_added": int(len(new_history)),
        "failed_secids": failed,
        "last_trade_date": _iso_date(combined["TRADEDATE"].max()),
    }
    return combined, summary


def _ns_design_matrix(ttm_years: np.ndarray, tau1: float) -> np.ndarray:
    tau_ratio = ttm_years / tau1
    exp_term = np.exp(-tau_ratio)
    beta1_loading = np.where(
        np.abs(tau_ratio) < 1e-6,
        1.0 - tau_ratio / 2.0 + tau_ratio**2 / 6.0,
        (1.0 - exp_term) / tau_ratio,
    )
    beta2_loading = beta1_loading - exp_term
    return np.column_stack([np.ones_like(ttm_years), beta1_loading, beta2_loading])


def _fit_ns_given_tau(ttm_years: np.ndarray, yields_decimal: np.ndarray, tau1: float) -> tuple[np.ndarray, float] | None:
    if tau1 <= 0 or not np.isfinite(tau1):
        return None

    X = _ns_design_matrix(ttm_years, tau1)
    beta_params, _, rank, _ = np.linalg.lstsq(X, yields_decimal, rcond=None)
    if rank < X.shape[1]:
        return None

    fitted = X @ beta_params
    rmse = float(np.sqrt(np.mean((yields_decimal - fitted) ** 2)))
    if not np.isfinite(rmse):
        return None
    return beta_params, rmse


def _fit_dynamic_day(trade_date: pd.Timestamp, df_day: pd.DataFrame) -> dict:
    row = {
        "date": trade_date,
        "curve_source_date": trade_date,
        "beta0": np.nan,
        "beta1": np.nan,
        "beta2": np.nan,
        "tau1": np.nan,
        "n_bonds": int(len(df_day)),
        "rmse_pp": np.nan,
        "rmse_pp_static": np.nan,
        "success": False,
        "error_message": None,
    }

    if len(df_day) < MIN_DATA_POINTS_FOR_FIT:
        row["error_message"] = f"too_few_bonds:{len(df_day)}"
        return row

    ttm_years = df_day["TTM_years"].to_numpy(dtype=float)
    yields_decimal = df_day["YIELDCLOSE"].to_numpy(dtype=float) / 100.0
    static_fit = _fit_ns_given_tau(ttm_years, yields_decimal, STATIC_TAU1)
    if static_fit is not None:
        row["rmse_pp_static"] = static_fit[1] * 100.0

    def objective(tau1: float) -> float:
        fitted = _fit_ns_given_tau(ttm_years, yields_decimal, tau1)
        if fitted is None:
            return np.inf
        return fitted[1]

    optimization = minimize_scalar(
        objective,
        bounds=TAU1_SEARCH_BOUNDS,
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not optimization.success:
        row["error_message"] = "tau1_optimization_failed"
        return row

    best_fit = _fit_ns_given_tau(ttm_years, yields_decimal, float(optimization.x))
    if best_fit is None:
        row["error_message"] = "invalid_best_fit"
        return row

    beta_params, rmse = best_fit
    row.update(
        {
            "beta0": float(beta_params[0]),
            "beta1": float(beta_params[1]),
            "beta2": float(beta_params[2]),
            "tau1": float(optimization.x),
            "rmse_pp": rmse * 100.0,
            "success": True,
            "error_message": None,
        }
    )
    return row


def build_daily_ns_factors(ofz_history: pd.DataFrame) -> pd.DataFrame:
    df = ofz_history.copy()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df["MATDATE"] = pd.to_datetime(df["MATDATE"], errors="coerce")
    df["YIELDCLOSE"] = pd.to_numeric(df["YIELDCLOSE"], errors="coerce")

    if "TTM_years" not in df.columns:
        df["TTM_years"] = (df["MATDATE"] - df["TRADEDATE"]).dt.days / 365.25
    else:
        df["TTM_years"] = pd.to_numeric(df["TTM_years"], errors="coerce")

    df = df.dropna(subset=["TRADEDATE", "TTM_years", "YIELDCLOSE"])
    df = df[df["TRADEDATE"] >= START_OFZ_DATE]
    df = df[df["TTM_years"] > 1e-6]
    df = df[df["YIELDCLOSE"] >= 0]
    df = df.sort_values(["TRADEDATE", "TTM_years", "SECID"])

    results: list[dict] = []
    grouped = list(df.groupby("TRADEDATE", sort=True))
    total_groups = len(grouped)
    for index, (trade_date, df_day) in enumerate(grouped, start=1):
        results.append(_fit_dynamic_day(pd.Timestamp(trade_date), df_day))
        if index % PROGRESS_EVERY_DATES == 0 or index == total_groups:
            print(f"[NS] fitted {index}/{total_groups} dates")

    daily = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
    return daily


def build_fixed_tau1_factors(ofz_history: pd.DataFrame, tau1: float = STATIC_TAU1) -> pd.DataFrame:
    df = ofz_history.copy()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df["MATDATE"] = pd.to_datetime(df["MATDATE"], errors="coerce")
    df["YIELDCLOSE"] = pd.to_numeric(df["YIELDCLOSE"], errors="coerce")

    if "TTM_years" not in df.columns:
        df["TTM_years"] = (df["MATDATE"] - df["TRADEDATE"]).dt.days / 365.25
    else:
        df["TTM_years"] = pd.to_numeric(df["TTM_years"], errors="coerce")

    df = df.dropna(subset=["TRADEDATE", "TTM_years", "YIELDCLOSE"])
    df = df[df["TRADEDATE"] >= START_OFZ_DATE]
    df = df[df["TTM_years"] > 1e-6]
    df = df[df["YIELDCLOSE"] >= 0]
    df = df.sort_values(["TRADEDATE", "TTM_years", "SECID"])

    rows: list[dict[str, object]] = []
    for trade_date, df_day in df.groupby("TRADEDATE", sort=True):
        if len(df_day) < MIN_DATA_POINTS_FOR_FIT:
            continue

        fitted = _fit_ns_given_tau(
            df_day["TTM_years"].to_numpy(dtype=float),
            df_day["YIELDCLOSE"].to_numpy(dtype=float) / 100.0,
            tau1,
        )
        if fitted is None:
            continue

        beta_params, _ = fitted
        rows.append(
            {
                "date": pd.Timestamp(trade_date),
                "beta0": float(beta_params[0]),
                "beta1": float(beta_params[1]),
                "beta2": float(beta_params[2]),
                "tau1": float(tau1),
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_monthly_ns_factors(daily_ns: pd.DataFrame) -> pd.DataFrame:
    valid = daily_ns[daily_ns["success"]].copy()
    valid["date"] = pd.to_datetime(valid["date"], errors="coerce")
    valid["curve_source_date"] = pd.to_datetime(valid["curve_source_date"], errors="coerce")
    valid = valid.dropna(subset=["date"]).set_index("date").sort_index()

    rows: list[dict] = []
    for month_start, group in valid.groupby(pd.Grouper(freq="MS")):
        if group.empty:
            continue
        last_row = group.iloc[-1]
        rows.append(
            {
                "date": month_start,
                "beta0": float(last_row["beta0"]),
                "beta1": float(last_row["beta1"]),
                "beta2": float(last_row["beta2"]),
                "tau1": float(last_row["tau1"]),
                "curve_source_date": pd.Timestamp(last_row["curve_source_date"]),
            }
        )

    monthly = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return monthly


def build_sample_splits(daily_ns: pd.DataFrame, monthly_ns: pd.DataFrame) -> dict:
    daily_ok = daily_ns[daily_ns["success"]].copy()
    daily_ok["date"] = pd.to_datetime(daily_ok["date"], errors="coerce")
    monthly = monthly_ns.copy()
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
    monthly["curve_source_date"] = pd.to_datetime(monthly["curve_source_date"], errors="coerce")

    last_daily_date = pd.Timestamp(daily_ok["date"].max())
    last_monthly_date = pd.Timestamp(monthly["date"].max())

    split_specs = [
        ("2013-09_to_2022-01", pd.Timestamp("2013-09-01"), pd.Timestamp("2022-01-31")),
        ("2022-02_to_latest", pd.Timestamp("2022-02-01"), None),
        ("2024-01_to_latest", pd.Timestamp("2024-01-01"), None),
    ]

    payload: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "daily_last_date": _iso_date(last_daily_date),
        "monthly_last_date": _iso_date(last_monthly_date),
        "splits": {},
    }

    for label, start_date, end_date in split_specs:
        split_end_daily = last_daily_date if end_date is None else min(end_date, last_daily_date)
        split_end_monthly = last_monthly_date if end_date is None else min(end_date, last_monthly_date)

        daily_slice = daily_ok[
            (daily_ok["date"] >= start_date)
            & (daily_ok["date"] <= split_end_daily)
        ].copy()
        monthly_slice = monthly[
            (monthly["date"] >= start_date.to_period("M").to_timestamp())
            & (monthly["date"] <= split_end_monthly.to_period("M").to_timestamp())
        ].copy()

        payload["splits"][label] = {
            "start_month": start_date.strftime("%Y-%m"),
            "end_month": split_end_monthly.strftime("%Y-%m"),
            "daily_obs": int(len(daily_slice)),
            "monthly_obs": int(len(monthly_slice)),
            "daily_first_date": _iso_date(daily_slice["date"].min() if not daily_slice.empty else None),
            "daily_last_date": _iso_date(daily_slice["date"].max() if not daily_slice.empty else None),
            "monthly_first_date": _iso_date(monthly_slice["date"].min() if not monthly_slice.empty else None),
            "monthly_last_date": _iso_date(monthly_slice["date"].max() if not monthly_slice.empty else None),
            "monthly_last_curve_source_date": _iso_date(
                monthly_slice["curve_source_date"].max() if not monthly_slice.empty else None
            ),
        }

    return payload


def _serialize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if np.issubdtype(out[column].dtype, np.datetime64):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
    return out


def run() -> UpdateSummary:
    today = datetime.now().date()
    print(f"[RUN] start: {datetime.now().isoformat(timespec='seconds')}")

    cbr_annual = update_cbr_inflation_and_key_rate(today=today)
    cpi_monthly = update_cpi_monthly_workbook()
    usd_daily = update_usd_workbook(today=today)
    ofz_history, ofz_summary = update_ofz_history(today=today)

    daily_ns = build_daily_ns_factors(ofz_history)
    fixed_tau1_daily = build_fixed_tau1_factors(ofz_history)
    monthly_ns = build_monthly_ns_factors(daily_ns)
    sample_splits = build_sample_splits(daily_ns, monthly_ns)

    _ensure_parent(DAILY_NS_PATH)
    _ensure_parent(FIXED_TAU1_NS_PATH)
    _ensure_parent(MONTHLY_NS_PATH)
    _serialize_for_csv(daily_ns).to_csv(DAILY_NS_PATH, index=False)
    _serialize_for_csv(fixed_tau1_daily).to_csv(FIXED_TAU1_NS_PATH, index=False)
    _serialize_for_csv(monthly_ns).to_csv(MONTHLY_NS_PATH, index=False)
    SAMPLE_SPLITS_PATH.write_text(json.dumps(sample_splits, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = UpdateSummary(
        ofz_rows_added=ofz_summary["rows_added"],
        ofz_last_date=ofz_summary["last_trade_date"],
        ofz_failed_secids=ofz_summary["failed_secids"],
        cpi_last_date=_iso_date(pd.Timestamp(cpi_monthly["date"].max())),
        key_rate_last_date=_iso_date(pd.Timestamp(cbr_annual["date"].max())),
        usd_last_date=_iso_date(pd.Timestamp(usd_daily["data"].max())),
        daily_last_date=_iso_date(pd.Timestamp(daily_ns.loc[daily_ns["success"], "date"].max())),
        monthly_last_date=_iso_date(pd.Timestamp(monthly_ns["date"].max())),
        monthly_last_curve_source_date=_iso_date(pd.Timestamp(monthly_ns["curve_source_date"].max())),
    )

    print("[RUN] summary")
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run()
