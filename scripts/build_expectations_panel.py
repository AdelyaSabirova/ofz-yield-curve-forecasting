from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATH = ROOT / "scripts" / "build_expectations_panel.py"
EXPECTATIONS_RAW_DIR = ROOT / "data" / "raw" / "macro" / "expectations"
MINFIN_RAW_DIR = ROOT / "data" / "raw" / "macro" / "minfin"
OFZ_IN_RAW_DIR = ROOT / "data" / "raw" / "macro" / "ofz_in"
CBR_RAW_DIR = ROOT / "data" / "raw" / "macro" / "cbr"
PROCESSED_DIR = ROOT / "data" / "processed" / "macro_2026"
LOG_PATH = ROOT / "docs" / "p2_expectations_log.md"

LOCAL_CPI_MONTHLY_PATH = ROOT / "data" / "raw" / "macro" / "cbr" / "ipc_mes_05-2025.xlsx"
LOCAL_NOMINAL_OFZ_PATH = ROOT / "data" / "raw" / "market" / "ofz" / "ofz_full_history.csv"

PANEL_START = pd.Timestamp("2013-09-01")
CURRENT_DATE = pd.Timestamp(date.today())
PANEL_END = (CURRENT_DATE.to_period("M") - 1).to_timestamp()
TIMEOUT = 60

CBR_KEY_RATE_URL = "https://www.cbr.ru/hd_base/KeyRate/"
CBR_INFLATION_URL = "https://www.cbr.ru/hd_base/infl/"
CBR_USD_XML_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
CBR_SURVEY_URL = "https://www.cbr.ru/Content/Document/File/144490/full.xlsx"
CBR_OFZ_NONRESIDENT_URL = "https://www.cbr.ru/vfs/eng/statistics/credit_statistics/debt/table_ofz_e.xlsx"
CBR_DEBT_SECURITIES_URL = "https://www.cbr.ru/vfs/eng/statistics/debt_securities/66e-debt_securities.xlsx"
MOEX_HISTORY_URL = "https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities/{secid}.json"

OFZ_IN_SECIDS = [
    "SU52002RMFS1",
    "SU52003RMFS9",
    "SU52004RMFS7",
    "SU52005RMFS4",
]

PANEL_COLUMNS = [
    "date",
    "cpi_mom",
    "cpi_yoy",
    "key_rate",
    "analyst_cpi_2026",
    "analyst_key_rate_2026",
    "usd_rub",
    "budget_signal",
    "breakeven_5y",
    "breakeven_10y",
    "ofz_nonresident_holdings_bln",
    "ofz_market_size_bln",
    "ofz_nonresident_share",
    "general_government_debt_bln",
    "general_government_debt_growth_12m",
]


@dataclass
class SourceNote:
    id: str
    label: str
    source_type: str
    url: str | None
    local_path: str | None
    status: str
    note: str


def ensure_output_dirs() -> None:
    for path in [EXPECTATIONS_RAW_DIR, MINFIN_RAW_DIR, OFZ_IN_RAW_DIR, CBR_RAW_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
    )
    return session


def fetch_html_tables(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> list[pd.DataFrame]:
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def to_month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def clean_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    as_str = str(value).replace("\xa0", " ").strip()
    if as_str in {"", "-", "nan", "NaN", "None"}:
        return None
    as_str = as_str.replace(" ", "").replace(",", ".")
    try:
        return float(as_str)
    except ValueError:
        return None


def normalize_fractional_percent(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.abs().median(skipna=True)
    if pd.notna(median) and median > 100:
        numeric = numeric / 100.0
    return numeric / 100.0


def normalize_date_token(token: Any) -> pd.Timestamp | pd.NaT:
    if pd.isna(token):
        return pd.NaT
    token_str = str(token).strip()
    token_str = token_str.replace(" ", "")
    if "." in token_str and token_str.count(".") == 1:
        month_token, year_token = token_str.split(".")
        if year_token == "202":
            year_token = "2020"
        if month_token.isdigit() and year_token.isdigit():
            return pd.Timestamp(year=int(year_token), month=int(month_token), day=1)
    return pd.to_datetime(token_str, errors="coerce")


def month_range_index() -> pd.DatetimeIndex:
    return pd.date_range(PANEL_START, PANEL_END, freq="MS")


def fetch_key_rate_monthly(session: requests.Session) -> tuple[pd.DataFrame, SourceNote]:
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": PANEL_START.strftime("%d.%m.%Y"),
        "UniDbQuery.To": CURRENT_DATE.strftime("%d.%m.%Y"),
    }
    table = fetch_html_tables(session, CBR_KEY_RATE_URL, params=params)[0]
    table.columns = [str(col).strip() for col in table.columns]
    table = table.rename(columns={"Дата": "date", "Ставка": "key_rate_raw"})
    table["date"] = pd.to_datetime(table["date"], dayfirst=True, errors="coerce")
    table["key_rate"] = normalize_fractional_percent(table["key_rate_raw"])
    table = table.dropna(subset=["date", "key_rate"]).sort_values("date")
    monthly = (
        table.assign(month=table["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["key_rate"]
        .last()
        .rename(columns={"month": "date"})
    )
    monthly = monthly[monthly["date"].between(PANEL_START, PANEL_END)]
    note = SourceNote(
        id="cbr_key_rate_daily",
        label="Ключевая ставка Банка России",
        source_type="html_table",
        url=f"{CBR_KEY_RATE_URL}?UniDbQuery.Posted=True",
        local_path=None,
        status="ok",
        note="Дневная ставка ЦБ агрегирована в месячный ряд как последнее доступное значение месяца.",
    )
    return monthly, note


def fetch_cpi_yoy_monthly(session: requests.Session) -> tuple[pd.DataFrame, SourceNote]:
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": PANEL_START.strftime("%d.%m.%Y"),
        "UniDbQuery.To": CURRENT_DATE.strftime("%d.%m.%Y"),
    }
    table = fetch_html_tables(session, CBR_INFLATION_URL, params=params)[0]
    table.columns = [str(col).strip() for col in table.columns]
    table = table.rename(columns={"Дата": "date", "Инфляция, % г/г": "cpi_yoy_raw"})
    table["date"] = table["date"].map(normalize_date_token)
    table["cpi_yoy"] = normalize_fractional_percent(table["cpi_yoy_raw"])
    table = table.dropna(subset=["date", "cpi_yoy"]).sort_values("date")
    table = table[table["date"].between(PANEL_START, PANEL_END)]
    note = SourceNote(
        id="cbr_inflation_yoy",
        label="Годовая инфляция Банка России",
        source_type="html_table",
        url=f"{CBR_INFLATION_URL}?UniDbQuery.Posted=True",
        local_path=None,
        status="ok",
        note="Месячный ряд годовой инфляции взят из таблицы Банка России.",
    )
    return table[["date", "cpi_yoy"]], note


def load_cpi_mom_local() -> tuple[pd.DataFrame, SourceNote]:
    df = pd.read_excel(LOCAL_CPI_MONTHLY_PATH, sheet_name="a")
    df.columns = [str(col).strip().lower() for col in df.columns]
    df = df.rename(columns={"date": "date", "cpi_mom": "cpi_mom"})
    df["date"] = to_month_start(df["date"])
    df["cpi_mom"] = pd.to_numeric(df["cpi_mom"], errors="coerce")
    if df["cpi_mom"].abs().median(skipna=True) > 1:
        df["cpi_mom"] = df["cpi_mom"] / 100.0
    df = df.dropna(subset=["date", "cpi_mom"]).sort_values("date")
    df = df[df["date"].between(PANEL_START, PANEL_END)]
    note = SourceNote(
        id="cbr_cpi_mom_local",
        label="Месячный ИПЦ (м/м)",
        source_type="local_excel",
        url=None,
        local_path=str(LOCAL_CPI_MONTHLY_PATH),
        status="ok",
        note="Использован локальный файл ЦБ РФ из проекта; покрытие заканчивается маем 2025 года.",
    )
    return df[["date", "cpi_mom"]], note


def fetch_usd_rub_monthly(session: requests.Session) -> tuple[pd.DataFrame, SourceNote]:
    params = {
        "date_req1": PANEL_START.strftime("%d/%m/%Y"),
        "date_req2": CURRENT_DATE.strftime("%d/%m/%Y"),
        "VAL_NM_RQ": "R01235",
    }
    response = session.get(CBR_USD_XML_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows: list[dict[str, Any]] = []
    for record in root.findall(".//Record"):
        day = pd.to_datetime(record.attrib.get("Date"), dayfirst=True, errors="coerce")
        nominal = clean_float(record.findtext("Nominal"))
        value = clean_float(record.findtext("Value"))
        if pd.isna(day) or nominal in {None, 0} or value is None:
            continue
        rows.append({"date": day, "usd_rub": value / nominal})
    usd = pd.DataFrame(rows)
    usd = usd.dropna(subset=["date", "usd_rub"]).sort_values("date")
    monthly = (
        usd.assign(month=usd["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["usd_rub"]
        .mean()
        .rename(columns={"month": "date"})
    )
    monthly = monthly[monthly["date"].between(PANEL_START, PANEL_END)]
    note = SourceNote(
        id="cbr_usd_xml",
        label="Курс USD/RUB Банка России",
        source_type="xml_api",
        url=response.url,
        local_path=None,
        status="ok",
        note="Официальный курс доллара США усреднён по календарному месяцу.",
    )
    return monthly, note


def download_survey_workbook(session: requests.Session) -> tuple[Path, SourceNote]:
    target_path = EXPECTATIONS_RAW_DIR / "cbr_macro_survey_full.xlsx"
    response = session.get(CBR_SURVEY_URL, timeout=TIMEOUT)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    note = SourceNote(
        id="cbr_macro_survey",
        label="Макроэкономический опрос Банка России",
        source_type="excel_download",
        url=CBR_SURVEY_URL,
        local_path=str(target_path),
        status="ok",
        note="Скачан файл full.xlsx с агрегированными результатами опроса аналитиков.",
    )
    return target_path, note


def download_binary_file(session: requests.Session, url: str, target_path: Path) -> Path:
    response = session.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def parse_survey_target_year(
    workbook_path: Path,
    sheet_name: str,
    column_name: str,
    target_year: int,
) -> pd.DataFrame:
    df = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)
    survey_dates = to_month_start(pd.Series(df.iloc[5, 4:].tolist()))
    label_col_1 = df.iloc[:, 1].astype(str)
    label_col_2 = df.iloc[:, 2].astype(str)
    median_idx_candidates = df.index[label_col_2.str.contains("Median", na=False)].tolist()
    if not median_idx_candidates:
        raise ValueError(f"Не найден блок Median в листе {sheet_name}")
    median_start = median_idx_candidates[0]
    stop_idx = len(df)
    for idx in range(median_start + 1, len(df)):
        label_text = f"{label_col_1.iloc[idx]} {label_col_2.iloc[idx]}"
        if "Average" in label_text or "Среднее" in label_text:
            stop_idx = idx
            break
    median_block = df.iloc[median_start:stop_idx].copy()
    target_row = None
    for _, row in median_block.iterrows():
        target_marker = row.iloc[3]
        if isinstance(target_marker, datetime) and target_marker.year == target_year:
            target_row = row
            break
    if target_row is None:
        raise ValueError(f"В листе {sheet_name} нет строки для {target_year} года")
    target_values = normalize_fractional_percent(pd.Series(target_row.iloc[4:].tolist()))
    out = pd.DataFrame(
        {
            "date": survey_dates.tolist(),
            column_name: target_values.tolist(),
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out[(out["date"] >= PANEL_START) & (out["date"] <= PANEL_END)]
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out


def expand_irregular_monthly_series(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    monthly = pd.DataFrame({"date": month_range_index()})
    monthly = monthly.merge(df[["date", value_column]], on="date", how="left")
    monthly[value_column] = monthly[value_column].ffill()
    return monthly


def fetch_analyst_expectations(workbook_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cpi_raw = parse_survey_target_year(
        workbook_path=workbook_path,
        sheet_name="1",
        column_name="analyst_cpi_2026",
        target_year=2026,
    )
    key_rate_raw = parse_survey_target_year(
        workbook_path=workbook_path,
        sheet_name="3",
        column_name="analyst_key_rate_2026",
        target_year=2026,
    )
    cpi = expand_irregular_monthly_series(cpi_raw, "analyst_cpi_2026")
    key_rate = expand_irregular_monthly_series(key_rate_raw, "analyst_key_rate_2026")
    return cpi, key_rate


def download_ofz_in_history(session: requests.Session) -> tuple[pd.DataFrame, SourceNote]:
    rows: list[dict[str, Any]] = []
    for secid in OFZ_IN_SECIDS:
        start = 0
        while True:
            response = session.get(
                MOEX_HISTORY_URL.format(secid=secid),
                params={
                    "iss.meta": "off",
                    "from": "2015-01-01",
                    "till": CURRENT_DATE.strftime("%Y-%m-%d"),
                    "start": start,
                    "limit": 100,
                    "history.columns": (
                        "TRADEDATE,CLOSE,YIELDCLOSE,VOLUME,VALUE,"
                        "SHORTNAME,MATDATE,DURATION,ACCINT"
                    ),
                },
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json().get("history", {})
            columns = payload.get("columns", [])
            data = payload.get("data", [])
            if not data:
                break
            rows.extend(dict(zip(columns, item)) | {"SECID": secid} for item in data)
            if len(data) < 100:
                break
            start += len(data)
    df = pd.DataFrame(rows)
    if not df.empty:
        target_path = OFZ_IN_RAW_DIR / "moex_ofz_in_history.csv"
        df.to_csv(target_path, index=False)
        note = SourceNote(
            id="moex_ofz_in_history",
            label="История ОФЗ-ИН на MOEX",
            source_type="json_api",
            url="https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities/{secid}.json",
            local_path=str(target_path),
            status="ok",
            note="Скачана объединённая история доходностей ОФЗ-ИН из ISS MOEX.",
        )
    else:
        note = SourceNote(
            id="moex_ofz_in_history",
            label="История ОФЗ-ИН на MOEX",
            source_type="json_api",
            url="https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities/{secid}.json",
            local_path=None,
            status="empty",
            note="MOEX не вернула наблюдения по выбранным ОФЗ-ИН.",
        )
    return df, note


def _parse_ofz_nonresident_table(table: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Date": "date",
        "Nonresidents’ OFZ holdings, bln roubles": "ofz_nonresident_holdings_bln",
        "OFZ market size, bln roubles": "ofz_market_size_bln",
        "Nonresidents’ market share, %": "ofz_nonresident_share_pct",
    }
    out = table.rename(columns=rename_map).copy()
    required_columns = {"date", "ofz_nonresident_holdings_bln", "ofz_market_size_bln", "ofz_nonresident_share_pct"}
    missing_columns = required_columns.difference(out.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"В таблице Банка России по доле нерезидентов не найдены столбцы: {missing}")
    out["date"] = to_month_start(out["date"])
    out["ofz_nonresident_holdings_bln"] = pd.to_numeric(out["ofz_nonresident_holdings_bln"], errors="coerce")
    out["ofz_market_size_bln"] = pd.to_numeric(out["ofz_market_size_bln"], errors="coerce")
    out["ofz_nonresident_share"] = pd.to_numeric(out["ofz_nonresident_share_pct"], errors="coerce") / 100.0
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out[(out["date"] >= PANEL_START) & (out["date"] <= PANEL_END)]
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out[
        [
            "date",
            "ofz_nonresident_holdings_bln",
            "ofz_market_size_bln",
            "ofz_nonresident_share",
        ]
    ].reset_index(drop=True)


def download_ofz_nonresident_series(session: requests.Session) -> tuple[pd.DataFrame, SourceNote]:
    target_path = CBR_RAW_DIR / "cbr_ofz_nonresident_share.xlsx"
    workbook_path = download_binary_file(session=session, url=CBR_OFZ_NONRESIDENT_URL, target_path=target_path)
    table = pd.read_excel(workbook_path, sheet_name="OFZ", header=1)
    monthly = _parse_ofz_nonresident_table(table)
    note = SourceNote(
        id="cbr_ofz_nonresident_share",
        label="Доля нерезидентов на рынке ОФЗ Банка России",
        source_type="excel_download",
        url=CBR_OFZ_NONRESIDENT_URL,
        local_path=str(workbook_path),
        status="ok",
        note=(
            "Использован официальный месячный ряд Банка России по объему вложений нерезидентов в ОФЗ, "
            "размеру рынка и доле нерезидентов."
        ),
    )
    return monthly, note


def _extract_general_government_debt_from_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    if sheet.shape[0] <= 31 or sheet.shape[1] <= 2:
        raise ValueError("Лист Банка России по внутреннему долгу слишком мал для извлечения ряда General government.")
    raw_dates = pd.to_datetime(sheet.iloc[4, 1:], errors="coerce")
    row_mask = sheet.iloc[:, 0].astype(str).str.strip().eq("General government")
    row_positions = sheet.index[row_mask].tolist()
    if not row_positions:
        raise ValueError("На листе Банка России не найдена строка General government.")
    row_values = sheet.iloc[row_positions[0], 1:]
    monthly = pd.DataFrame(
        {
            "date": to_month_start(pd.Series(raw_dates)),
            "general_government_debt_mln": pd.to_numeric(row_values, errors="coerce"),
        }
    )
    monthly = monthly.dropna(subset=["date", "general_government_debt_mln"]).sort_values("date")
    monthly["general_government_debt_bln"] = monthly["general_government_debt_mln"] / 1000.0
    positive_debt = monthly["general_government_debt_bln"].where(monthly["general_government_debt_bln"] > 0)
    monthly["general_government_debt_growth_12m"] = np.log(positive_debt).diff(12)
    monthly = monthly[(monthly["date"] >= PANEL_START) & (monthly["date"] <= PANEL_END)]
    monthly = monthly.drop_duplicates(subset=["date"], keep="last")
    return monthly[["date", "general_government_debt_bln", "general_government_debt_growth_12m"]].reset_index(drop=True)


def download_general_government_debt_series(session: requests.Session) -> tuple[pd.DataFrame, SourceNote]:
    target_path = CBR_RAW_DIR / "cbr_domestic_debt_securities.xlsx"
    workbook_path = download_binary_file(session=session, url=CBR_DEBT_SECURITIES_URL, target_path=target_path)
    sheet = pd.read_excel(workbook_path, sheet_name="Month (nominal value)", header=None)
    monthly = _extract_general_government_debt_from_sheet(sheet)
    note = SourceNote(
        id="cbr_general_government_debt",
        label="Долг сектора государственного управления по долговым ценным бумагам Банка России",
        source_type="excel_download",
        url=CBR_DEBT_SECURITIES_URL,
        local_path=str(workbook_path),
        status="ok",
        note=(
            "Использован официальный месячный ряд Банка России по номинальному объему долговых ценных бумаг "
            "сектора государственного управления; в панель включены уровень и 12-месячный логарифмический прирост."
        ),
    )
    return monthly, note


def prepare_nominal_ofz_history() -> pd.DataFrame:
    usecols = ["TRADEDATE", "SECID", "SHORTNAME", "MATDATE", "YIELDCLOSE", "TTM_years", "VALUE"]
    df = pd.read_csv(LOCAL_NOMINAL_OFZ_PATH, usecols=usecols)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df["MATDATE"] = pd.to_datetime(df["MATDATE"], errors="coerce")
    df["TTM_years"] = pd.to_numeric(df["TTM_years"], errors="coerce")
    df["YIELDCLOSE"] = pd.to_numeric(df["YIELDCLOSE"], errors="coerce")
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["TRADEDATE", "TTM_years", "YIELDCLOSE"])
    df = df[(df["TTM_years"] > 0.25) & (df["YIELDCLOSE"].between(-5, 40))]
    mask_inflation_linked = df["SECID"].astype(str).str.startswith("SU52") | df["SHORTNAME"].astype(str).str.contains("ИН", na=False)
    df = df[~mask_inflation_linked].copy()
    return df


def prepare_real_ofz_in_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["TRADEDATE"] = pd.to_datetime(out["TRADEDATE"], errors="coerce")
    out["MATDATE"] = pd.to_datetime(out["MATDATE"], errors="coerce")
    out["YIELDCLOSE"] = pd.to_numeric(out["YIELDCLOSE"], errors="coerce")
    out["VALUE"] = pd.to_numeric(out["VALUE"], errors="coerce")
    out["TTM_years"] = (out["MATDATE"] - out["TRADEDATE"]).dt.days / 365.25
    out = out.dropna(subset=["TRADEDATE", "TTM_years", "YIELDCLOSE"])
    out = out[(out["TTM_years"] > 0.25) & (out["YIELDCLOSE"].between(-5, 20))]
    return out


def interpolate_curve_at_target(
    df: pd.DataFrame,
    target_ttm: float,
    max_gap: float,
    nearest_tolerance: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for trade_date, group in df.groupby("TRADEDATE"):
        current = group[["TTM_years", "YIELDCLOSE", "VALUE"]].dropna().sort_values("TTM_years")
        if current.empty:
            continue
        lower = current[current["TTM_years"] <= target_ttm].tail(1)
        upper = current[current["TTM_years"] >= target_ttm].head(1)
        chosen_yield = np.nan
        if not lower.empty and not upper.empty:
            left = lower.iloc[0]
            right = upper.iloc[0]
            if left["TTM_years"] == right["TTM_years"]:
                chosen_yield = float(left["YIELDCLOSE"])
            else:
                left_gap = target_ttm - float(left["TTM_years"])
                right_gap = float(right["TTM_years"]) - target_ttm
                if left_gap <= max_gap and right_gap <= max_gap:
                    slope = (float(right["YIELDCLOSE"]) - float(left["YIELDCLOSE"])) / (
                        float(right["TTM_years"]) - float(left["TTM_years"])
                    )
                    chosen_yield = float(left["YIELDCLOSE"]) + slope * left_gap
        if pd.isna(chosen_yield):
            nearest = current.iloc[(current["TTM_years"] - target_ttm).abs().argmin()]
            if abs(float(nearest["TTM_years"]) - target_ttm) <= nearest_tolerance:
                chosen_yield = float(nearest["YIELDCLOSE"])
        if pd.notna(chosen_yield):
            records.append({"date": trade_date, "yield_pct": chosen_yield})
    return pd.DataFrame(records)


def compute_breakeven_series(ofz_in_history: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    if ofz_in_history.empty:
        issues.append("MOEX не вернула историю ОФЗ-ИН, поэтому ряд рыночной инфляционной компенсации не заполнен.")
        return pd.DataFrame(columns=["date", "breakeven_5y", "breakeven_10y"]), issues

    nominal = prepare_nominal_ofz_history()
    real = prepare_real_ofz_in_history(ofz_in_history)
    if nominal.empty or real.empty:
        issues.append("Недостаточно наблюдений по номинальным ОФЗ или ОФЗ-ИН для оценки рыночной инфляционной компенсации.")
        return pd.DataFrame(columns=["date", "breakeven_5y", "breakeven_10y"]), issues

    nominal_5 = interpolate_curve_at_target(nominal, target_ttm=5.0, max_gap=2.0, nearest_tolerance=1.0)
    nominal_10 = interpolate_curve_at_target(nominal, target_ttm=10.0, max_gap=3.0, nearest_tolerance=1.5)
    real_5 = interpolate_curve_at_target(real, target_ttm=5.0, max_gap=3.0, nearest_tolerance=1.5)
    real_10 = interpolate_curve_at_target(real, target_ttm=10.0, max_gap=4.0, nearest_tolerance=2.0)

    nominal_5 = nominal_5.rename(columns={"yield_pct": "nominal_5"})
    nominal_10 = nominal_10.rename(columns={"yield_pct": "nominal_10"})
    real_5 = real_5.rename(columns={"yield_pct": "real_5"})
    real_10 = real_10.rename(columns={"yield_pct": "real_10"})

    daily = nominal_5.merge(real_5, on="date", how="outer").merge(nominal_10, on="date", how="outer").merge(real_10, on="date", how="outer")
    daily["breakeven_5y"] = (daily["nominal_5"] - daily["real_5"]) / 100.0
    daily["breakeven_10y"] = (daily["nominal_10"] - daily["real_10"]) / 100.0
    monthly = (
        daily.assign(month=to_month_start(daily["date"]))
        .groupby("month", as_index=False)[["breakeven_5y", "breakeven_10y"]]
        .mean()
        .rename(columns={"month": "date"})
    )
    monthly = monthly[(monthly["date"] >= PANEL_START) & (monthly["date"] <= PANEL_END)]

    if monthly["breakeven_5y"].notna().sum() == 0:
        issues.append("5-летнюю рыночную инфляционную компенсацию не удалось оценить ни для одного месяца.")
    if monthly["breakeven_10y"].notna().sum() == 0:
        issues.append("10-летнюю рыночную инфляционную компенсацию не удалось оценить ни для одного месяца.")
    valid_breakeven_dates = monthly.loc[
        monthly[["breakeven_5y", "breakeven_10y"]].notna().any(axis=1),
        "date",
    ]
    if not valid_breakeven_dates.empty and valid_breakeven_dates.max() < PANEL_END:
        issues.append(
            "Ряд рыночной инфляционной компенсации заканчивается раньше панели, потому что номинальная история ОФЗ в проекте пока доступна только до апреля 2025 года."
        )
    return monthly[["date", "breakeven_5y", "breakeven_10y"]], issues


def build_budget_signal(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, SourceNote]:
    signal = pd.DataFrame({"date": index, "budget_signal": (index >= pd.Timestamp("2022-02-01")).astype(int)})
    note = SourceNote(
        id="budget_signal_regime_dummy",
        label="Прокси бюджетно-долгового давления",
        source_type="derived_proxy",
        url=None,
        local_path=None,
        status="proxy",
        note=(
            "Использована минимальная режимная переменная: 0 до 2022-02 и 1 начиная с 2022-02. "
            "Наблюдаемый месячный ряд Минфина не был получен из среды из-за TLS-ошибки при доступе к minfin.gov.ru."
        ),
    )
    return signal, note


def merge_panel(parts: list[pd.DataFrame]) -> pd.DataFrame:
    panel = pd.DataFrame({"date": month_range_index()})
    for frame in parts:
        panel = panel.merge(frame, on="date", how="left")
    return panel[PANEL_COLUMNS]


def field_summary(panel: pd.DataFrame, column: str, source_ids: list[str], note: str) -> dict[str, Any]:
    series = panel[column]
    non_null = series.dropna()
    first_non_null = None
    last_non_null = None
    if not non_null.empty:
        first_non_null = panel.loc[non_null.index[0], "date"].strftime("%Y-%m-%d")
        last_non_null = panel.loc[non_null.index[-1], "date"].strftime("%Y-%m-%d")
    return {
        "filled": bool(non_null.shape[0]),
        "non_null_count": int(non_null.shape[0]),
        "missing_count": int(series.isna().sum()),
        "first_non_null": first_non_null,
        "last_non_null": last_non_null,
        "source_ids": source_ids,
        "note": note,
    }


def build_source_manifest(panel: pd.DataFrame, sources: list[SourceNote], breakeven_notes: list[str]) -> dict[str, Any]:
    field_notes = {
        "cpi_mom": "Месячный ИПЦ из локального файла ЦБ РФ. После мая 2025 года значения отсутствуют в доступном проекту файле.",
        "cpi_yoy": "Годовая инфляция Банка России по месяцам.",
        "key_rate": "Ключевая ставка Банка России как последнее доступное значение месяца.",
        "analyst_cpi_2026": "Медианный прогноз аналитиков Банка России по ИПЦ на конец 2026 года.",
        "analyst_key_rate_2026": "Медианный прогноз аналитиков Банка России по средней ключевой ставке за 2026 год.",
        "usd_rub": "Средний за месяц официальный курс USD/RUB Банка России.",
        "budget_signal": "Минимальный прокси бюджетно-долгового фактора: режимная переменная с февраля 2022 года.",
        "breakeven_5y": "Приближённая 5-летняя вменённая инфляция: номинальная ОФЗ минус реальная доходность ОФЗ-ИН.",
        "breakeven_10y": "Приближённая 10-летняя вменённая инфляция: номинальная ОФЗ минус реальная доходность ОФЗ-ИН.",
        "ofz_nonresident_holdings_bln": "Объем вложений нерезидентов в ОФЗ по данным Банка России, млрд руб.",
        "ofz_market_size_bln": "Размер рынка ОФЗ по данным Банка России, млрд руб.",
        "ofz_nonresident_share": "Доля нерезидентов на рынке ОФЗ по данным Банка России, доля единицы.",
        "general_government_debt_bln": "Номинальный объем долговых ценных бумаг сектора государственного управления, млрд руб.",
        "general_government_debt_growth_12m": "12-месячный логарифмический прирост долга сектора государственного управления по долговым бумагам.",
    }
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT_PATH),
        "panel": {
            "rows": int(len(panel)),
            "start": panel["date"].min().strftime("%Y-%m-%d"),
            "end": panel["date"].max().strftime("%Y-%m-%d"),
            "columns": PANEL_COLUMNS,
        },
        "sources": [
            {
                "id": source.id,
                "label": source.label,
                "source_type": source.source_type,
                "url": source.url,
                "local_path": source.local_path,
                "status": source.status,
                "note": source.note,
            }
            for source in sources
        ],
        "fields": {
            "cpi_mom": field_summary(panel, "cpi_mom", ["cbr_cpi_mom_local"], field_notes["cpi_mom"]),
            "cpi_yoy": field_summary(panel, "cpi_yoy", ["cbr_inflation_yoy"], field_notes["cpi_yoy"]),
            "key_rate": field_summary(panel, "key_rate", ["cbr_key_rate_daily"], field_notes["key_rate"]),
            "analyst_cpi_2026": field_summary(
                panel,
                "analyst_cpi_2026",
                ["cbr_macro_survey"],
                field_notes["analyst_cpi_2026"] + " Между датами опросов используется перенос последнего доступного значения вперёд.",
            ),
            "analyst_key_rate_2026": field_summary(
                panel,
                "analyst_key_rate_2026",
                ["cbr_macro_survey"],
                field_notes["analyst_key_rate_2026"] + " Между датами опросов используется перенос последнего доступного значения вперёд.",
            ),
            "usd_rub": field_summary(panel, "usd_rub", ["cbr_usd_xml"], field_notes["usd_rub"]),
            "budget_signal": field_summary(panel, "budget_signal", ["budget_signal_regime_dummy"], field_notes["budget_signal"]),
            "breakeven_5y": field_summary(panel, "breakeven_5y", ["moex_ofz_in_history"], field_notes["breakeven_5y"]),
            "breakeven_10y": field_summary(panel, "breakeven_10y", ["moex_ofz_in_history"], field_notes["breakeven_10y"]),
            "ofz_nonresident_holdings_bln": field_summary(
                panel,
                "ofz_nonresident_holdings_bln",
                ["cbr_ofz_nonresident_share"],
                field_notes["ofz_nonresident_holdings_bln"],
            ),
            "ofz_market_size_bln": field_summary(
                panel,
                "ofz_market_size_bln",
                ["cbr_ofz_nonresident_share"],
                field_notes["ofz_market_size_bln"],
            ),
            "ofz_nonresident_share": field_summary(
                panel,
                "ofz_nonresident_share",
                ["cbr_ofz_nonresident_share"],
                field_notes["ofz_nonresident_share"],
            ),
            "general_government_debt_bln": field_summary(
                panel,
                "general_government_debt_bln",
                ["cbr_general_government_debt"],
                field_notes["general_government_debt_bln"],
            ),
            "general_government_debt_growth_12m": field_summary(
                panel,
                "general_government_debt_growth_12m",
                ["cbr_general_government_debt"],
                field_notes["general_government_debt_growth_12m"],
            ),
        },
        "gaps": breakeven_notes
        + [
            "Minfin.gov.ru не удалось использовать из текущей среды: TLS-соединение обрывается на стороне сайта, поэтому вместо прямого ряда Минфина используются наблюдаемые ряды Банка России по долгу сектора государственного управления и структуре держателей ОФЗ.",
            "Доступный в проекте файл cpi_mom заканчивается 2025-05, поэтому хвост панели по cpi_mom остаётся пустым.",
        ],
    }
    return manifest


def write_log(panel: pd.DataFrame, sources: list[SourceNote], manifest: dict[str, Any]) -> None:
    filled_fields = [name for name, meta in manifest["fields"].items() if meta["filled"]]
    empty_fields = [name for name, meta in manifest["fields"].items() if not meta["filled"]]
    lines = [
        "# Поток 2. Лог сборки панели ожиданий",
        "",
        f"- Дата сборки: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`.",
        f"- Скрипт: `{SCRIPT_PATH}`.",
        f"- Период панели: `{panel['date'].min().strftime('%Y-%m-%d')}` .. `{panel['date'].max().strftime('%Y-%m-%d')}`.",
        f"- Число месячных наблюдений: `{len(panel)}`.",
        "",
        "## Источники",
        "",
    ]
    for source in sources:
        source_line = f"- `{source.id}`: {source.label}. Статус: `{source.status}`."
        if source.url:
            source_line += f" URL: `{source.url}`."
        if source.local_path:
            source_line += f" Локальный файл: `{source.local_path}`."
        source_line += f" Примечание: {source.note}"
        lines.append(source_line)
    lines.extend(
        [
            "",
            "## Покрытие полей",
            "",
        ]
    )
    for field_name, meta in manifest["fields"].items():
        lines.append(
            "- "
            f"`{field_name}`: заполнено `{meta['non_null_count']}` наблюдений, "
            f"пропусков `{meta['missing_count']}`, "
            f"первое непустое `{meta['first_non_null']}`, "
            f"последнее непустое `{meta['last_non_null']}`. "
            f"{meta['note']}"
        )
    lines.extend(
        [
            "",
            "## Пробелы и допущения",
            "",
        ]
    )
    for item in manifest["gaps"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Итог",
            "",
            f"- Поля с ненулевым покрытием: `{', '.join(filled_fields)}`.",
            f"- Поля без покрытых наблюдений: `{', '.join(empty_fields) if empty_fields else 'нет'}`.",
            "- `budget_signal` сохранён как минимальный режимный индикатор с февраля 2022 года, но прогностический блок дополнительно использует наблюдаемые ряды Банка России по долгу сектора государственного управления и доле нерезидентов в ОФЗ.",
        ]
    )
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    session = build_session()

    sources: list[SourceNote] = []
    sources.append(
        SourceNote(
            id="minfin_site_attempt",
            label="Сайт Минфина России",
            source_type="web_source",
            url="https://minfin.gov.ru/",
            local_path=None,
            status="unavailable",
            note=(
                "Из текущей среды воспроизводимо не получен из-за TLS-ошибки соединения. "
                "Поэтому наблюдаемый месячный ряд Минфина не встроен, а budget_signal задан в минимальном контракте."
            ),
        )
    )

    cpi_mom_df, cpi_mom_note = load_cpi_mom_local()
    sources.append(cpi_mom_note)

    cpi_yoy_df, cpi_yoy_note = fetch_cpi_yoy_monthly(session)
    sources.append(cpi_yoy_note)

    key_rate_df, key_rate_note = fetch_key_rate_monthly(session)
    sources.append(key_rate_note)

    usd_df, usd_note = fetch_usd_rub_monthly(session)
    sources.append(usd_note)

    survey_path, survey_note = download_survey_workbook(session)
    sources.append(survey_note)
    analyst_cpi_df, analyst_key_rate_df = fetch_analyst_expectations(survey_path)

    budget_df, budget_note = build_budget_signal(month_range_index())
    sources.append(budget_note)

    ofz_nonresident_df, ofz_nonresident_note = download_ofz_nonresident_series(session)
    sources.append(ofz_nonresident_note)

    government_debt_df, government_debt_note = download_general_government_debt_series(session)
    sources.append(government_debt_note)

    ofz_in_history, ofz_in_note = download_ofz_in_history(session)
    sources.append(ofz_in_note)
    breakeven_df, breakeven_notes = compute_breakeven_series(ofz_in_history)

    panel = merge_panel(
        [
            cpi_mom_df,
            cpi_yoy_df,
            key_rate_df,
            analyst_cpi_df,
            analyst_key_rate_df,
            usd_df,
            budget_df,
            breakeven_df,
            ofz_nonresident_df,
            government_debt_df,
        ]
    )

    output_path = PROCESSED_DIR / "monthly_expectations_panel.csv"
    panel.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    manifest = build_source_manifest(panel, sources, breakeven_notes)
    manifest_path = PROCESSED_DIR / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_log(panel, sources, manifest)

    print(f"Saved panel to {output_path}")
    print(f"Saved manifest to {manifest_path}")
    print(f"Saved log to {LOG_PATH}")


if __name__ == "__main__":
    main()
