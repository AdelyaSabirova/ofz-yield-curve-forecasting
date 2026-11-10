"""
Загрузка данных инфляции (ИПЦ) из локальных файлов ЦБ РФ.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import CBR_INFLATION_KEY_RATE_XLSX, CBR_IPC_XLSX


def _to_path(path: str | Path | None, default: Path) -> Path:
    return Path(path) if path is not None else default


def _normalize_annual_cbr_date_tokens(date_series: pd.Series) -> pd.Series:
    """
    Исправляет дефектные токены года в выгрузке ЦБ:
    `1.202`, `2.202`, ... `12.202` -> `1.2020`, ... `12.2020`.
    """

    as_str = date_series.astype(str).str.strip()
    as_str = as_str.str.replace(
        r"^(\d{1,2})\.202$",
        r"\1.2020",
        regex=True,
    )
    return as_str


def _normalize_month_start_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out[date_col] = out[date_col].dt.to_period("M").dt.to_timestamp()
    out = out.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    out = out.set_index(date_col)
    return out


def load_monthly_inflation(path: str | Path | None = None) -> pd.DataFrame | None:
    """
    Загружает месячную инфляцию (cpi_mom) в долях из файла `ipc_mes_*.xlsx`.

    Возвращает DataFrame с индексом `date` и колонкой `cpi_mom`.
    """

    xlsx_path = _to_path(path, CBR_IPC_XLSX)
    try:
        # В текущем проекте лист `a` уже содержит подготовленные `date` и `cpi_mom`.
        df = pd.read_excel(xlsx_path, sheet_name="a")
    except Exception as exc:
        print(f"❌ Не удалось загрузить месячную инфляцию из {xlsx_path}: {exc}")
        return None

    if df.empty:
        print(f"⚠️ Файл {xlsx_path} пуст")
        return None

    cols = {str(c).strip().lower(): c for c in df.columns}
    if "date" not in cols or "cpi_mom" not in cols:
        print(f"❌ В {xlsx_path} нет ожидаемых колонок 'date' и 'cpi_mom'")
        return None

    out = df[[cols["date"], cols["cpi_mom"]]].copy()
    out.columns = ["date", "cpi_mom"]
    out["cpi_mom"] = pd.to_numeric(out["cpi_mom"], errors="coerce")
    out = _normalize_month_start_index(out)
    out = out.dropna(subset=["cpi_mom"])

    # Подстраховка: если значения записаны как проценты, переводим в доли.
    if out["cpi_mom"].abs().median() > 1:
        out["cpi_mom"] = out["cpi_mom"] / 100.0

    return out[["cpi_mom"]]


def load_annual_inflation(path: str | Path | None = None) -> pd.DataFrame | None:
    """
    Загружает годовую инфляцию (cpi_yoy) в долях из файла
    `Инфляция и ключевая ставка Банка России_*.xlsx`.

    Возвращает DataFrame с индексом `date` и колонкой `cpi_yoy`.
    """

    xlsx_path = _to_path(path, CBR_INFLATION_KEY_RATE_XLSX)
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as exc:
        print(f"❌ Не удалось загрузить годовую инфляцию из {xlsx_path}: {exc}")
        return None

    if df.empty:
        print(f"⚠️ Файл {xlsx_path} пуст")
        return None

    # Нормализуем ожидаемые колонки из выгрузки ЦБ РФ.
    out = df.rename(
        columns={
            "Дата": "date",
            "Инфляция, % г/г": "cpi_yoy_pct",
        }
    )
    if "date" not in out.columns or "cpi_yoy_pct" not in out.columns:
        print(f"❌ В {xlsx_path} нет ожидаемых колонок 'Дата' и 'Инфляция, % г/г'")
        return None

    out = out[["date", "cpi_yoy_pct"]].copy()
    out["date"] = _normalize_annual_cbr_date_tokens(out["date"])
    out["date"] = pd.to_datetime(out["date"], format="%m.%Y", errors="coerce")
    out["cpi_yoy"] = pd.to_numeric(out["cpi_yoy_pct"], errors="coerce") / 100.0
    out = out.drop(columns=["cpi_yoy_pct"])
    out = _normalize_month_start_index(out)
    out = out.dropna(subset=["cpi_yoy"])

    return out[["cpi_yoy"]]


def get_cbr_inflation_data(
    monthly_path: str | Path | None = None,
    annual_path: str | Path | None = None,
) -> pd.DataFrame | None:
    """
    Собирает итоговый DataFrame CPI для анализа:
    `cpi_mom`, `cpi_yoy`, `cpi_index` (индекс с базой 100).
    """

    df_monthly = load_monthly_inflation(monthly_path)
    df_annual = load_annual_inflation(annual_path)

    if df_monthly is None or df_annual is None:
        print("❌ Не удалось собрать CPI: один из источников недоступен")
        return None

    cpi_data = pd.merge(
        df_monthly,
        df_annual,
        left_index=True,
        right_index=True,
        how="inner",
    ).sort_index()

    if cpi_data.empty:
        print("❌ После объединения месячных и годовых данных CPI получен пустой DataFrame")
        return None

    cpi_data["cpi_index"] = 100 * (1 + cpi_data["cpi_mom"]).cumprod()
    return cpi_data[["cpi_mom", "cpi_yoy", "cpi_index"]]
