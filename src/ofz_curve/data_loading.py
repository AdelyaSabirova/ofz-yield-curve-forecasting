"""
Функции для загрузки и подготовки данных
"""

import pandas as pd
import numpy as np
from datetime import datetime
from .config import *


def load_ofz_data(filename=FILENAME):
    """
    Загружает и подготавливает данные ОФЗ

    Args:
        filename (str): Путь к CSV файлу с данными ОФЗ

    Returns:
        pd.DataFrame: Подготовленный DataFrame с данными ОФЗ
    """
    # Загрузка данных
    df = pd.read_csv(filename)
    print(f"Данные загружены из {filename}. Строк: {len(df)}")

    # Преобразование дат
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'], errors='coerce')
    df['MATDATE'] = pd.to_datetime(df['MATDATE'], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=['TRADEDATE', 'MATDATE'], inplace=True)

    # Расчет TTM
    df['TTM_days'] = (df['MATDATE'] - df['TRADEDATE']).dt.days
    df['TTM_years'] = df['TTM_days'] / 365.25

    # Преобразование числовых колонок
    numeric_cols = ['YIELDCLOSE', 'CLOSE', 'VOLUME', 'VALUE']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Фильтрация данных
    df_filtered = df.dropna(subset=['TTM_years', 'YIELDCLOSE'])
    df_filtered = df_filtered[df_filtered['TTM_years'] > 1e-6]
    df_filtered = df_filtered[df_filtered['YIELDCLOSE'] >= 0]

    print(f"Данные готовы. Строк для анализа: {len(df_filtered)}")
    print(f"Информация о данных: {df_filtered[['TRADEDATE', 'MATDATE', 'TTM_years', 'YIELDCLOSE', 'VOLUME']].info()}")

    return df_filtered


def load_oil_data(oil_path=OIL_PATH):
    """
    Загружает данные о ценах на нефть Brent

    Args:
        oil_path (str): Путь к Excel файлу с данными Brent

    Returns:
        pd.Series: Серия с логарифмами цен на нефть
    """
    oil_df_raw = pd.read_excel(oil_path, sheet_name="Daily")
    oil_df_raw.columns = [col.strip().lower() for col in oil_df_raw.columns]
    oil_df_raw.rename(columns={"observation_date": "date", "dcoilbrenteu": "price"}, inplace=True)
    oil_df_raw["date"] = pd.to_datetime(oil_df_raw["date"], errors="coerce")

    oil_df_raw["price"] = pd.to_numeric(oil_df_raw["price"].astype(str).str.replace(",", "."), errors="coerce")
    oil_df_raw = oil_df_raw.dropna(subset=["date"])
    oil_df_raw = oil_df_raw[oil_df_raw["date"] >= pd.Timestamp("2012-01-01")].sort_values("date")

    missing_oil_before_fill = oil_df_raw["price"].isna().sum()

    oil_daily = oil_df_raw.set_index("date")[["price"]].resample("D").mean()
    oil_daily["price_filled"] = oil_daily["price"].ffill().bfill()
    oil_daily["ln_oil"] = np.log(oil_daily["price_filled"])

    oil_series_full = oil_daily["ln_oil"]
    print(f"Brent: пустых цен до заполнения: {missing_oil_before_fill}")

    return oil_series_full


def load_usd_data(usd_path=USD_PATH):
    """
    Загружает данные о курсе USD/RUB

    Args:
        usd_path (str): Путь к Excel файлу с данными USD

    Returns:
        pd.Series: Серия с логарифмами курса USD
    """
    usd_df_raw = pd.read_excel(usd_path)
    usd_df_raw.columns = [col.strip().lower() for col in usd_df_raw.columns]
    usd_df_raw.rename(columns={"data": "date", "curs": "usd"}, inplace=True)
    usd_df_raw["date"] = pd.to_datetime(usd_df_raw["date"], errors="coerce", dayfirst=True)

    usd_df_raw["usd"] = pd.to_numeric(usd_df_raw["usd"].astype(str).str.replace(",", "."), errors="coerce")
    usd_df_raw = usd_df_raw.dropna(subset=["date"])
    usd_df_raw = usd_df_raw[usd_df_raw["date"] >= pd.Timestamp("2012-01-01")].sort_values("date")

    missing_usd_before_fill = usd_df_raw["usd"].isna().sum()

    usd_daily = usd_df_raw.set_index("date")[["usd"]].resample("D").mean()
    usd_daily["usd_filled"] = usd_daily["usd"].ffill().bfill()
    usd_daily["ln_usd"] = np.log(usd_daily["usd_filled"])

    usd_series_full = usd_daily["ln_usd"]
    print(f"USD: пустых цен до заполнения: {missing_usd_before_fill}")

    return usd_series_full


def prepare_factors_for_vecm(factors_df, oil_series_full, usd_series_full):
    """
    Подготавливает данные факторов для VECM модели

    Args:
        factors_df (pd.DataFrame): DataFrame с факторами NS
        oil_series_full (pd.Series): Серия с логарифмами цен на нефть
        usd_series_full (pd.Series): Серия с логарифмами курса USD

    Returns:
        pd.DataFrame: Подготовленный DataFrame для VECM
    """
    # Выравниваем по датам факторов
    oil_series_aligned = oil_series_full.reindex(factors_df.index).ffill().bfill()
    usd_series_aligned = usd_series_full.reindex(factors_df.index).ffill().bfill()

    factors_for_vecm_df = factors_df[["beta0", "beta1", "beta2"]].copy()
    factors_for_vecm_df["ln_oil"] = oil_series_aligned
    factors_for_vecm_df["ln_usd"] = usd_series_aligned

    # Убираем дамми (если есть)
    factors_for_vecm_df.dropna(inplace=True)

    coverage_share = len(factors_for_vecm_df) / len(factors_df) if len(factors_df) else 0

    print("Подготовлены данные по нефти/валюте для VECM")
    print(f"Доступно наблюдений после склейки: {len(factors_for_vecm_df)} ({coverage_share:.1%} от factors_df)")

    return factors_for_vecm_df

