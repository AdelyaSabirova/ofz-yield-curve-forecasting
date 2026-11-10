import requests
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
import os
import time
import numpy as np # Для расчетов и обработки NaN

# --- Константы ---
BASE_URL = "https://iss.moex.com/iss"
ALL_SECURITIES_URL = f"{BASE_URL}/securities.json"
HISTORY_BASE_URL = f"{BASE_URL}/history/engines/stock/markets/bonds/securities"
SECURITY_INFO_URL = f"{BASE_URL}/engines/stock/markets/bonds/securities" # Для статики

ROW_CHECKPOINT = 10000
REQUEST_TIMEOUT = 30
HISTORY_TIMEOUT = 60
STATIC_INFO_TIMEOUT = 45 # Таймаут для запросов статики
RATE_LIMIT_PAUSE = 5
MAX_RETRIES = 2 # Количество повторных попыток для сбойных SECID
RETRY_PAUSE = 10 # Пауза в секундах перед каждой повторной попыткой для SECID

# --- Кастомное исключение для ошибок истории ---
class HistoryDownloadError(Exception):
    """Исключение, возникающее при неполной загрузке истории из-за ошибки."""
    def __init__(self, message, secid, partial_data=None):
        super().__init__(message)
        self.secid = secid
        self.partial_data = partial_data # Сохраняем частично загруженные данные, если есть

# --- Функции для получения данных ---

def get_ofz_list():
    """
    Получает список ВСЕХ ОФЗ (SECID), включая исторические.
    Использует эндпоинт /iss/securities с фильтрами и пагинацией.
    Фильтрует по engine=stock, market=bonds, префиксу 'SU'
    и типу 'ofz_bond'.
    """
    all_potential_securities = []
    start = 0
    limit = 100
    print("Получение полного списка облигаций (engine=stock, market=bonds)...")
    while True:
        params = {
            "iss.meta": "off", "limit": limit, "start": start,
            "engine": "stock", "market": "bonds",
            "securities.columns": "secid,type",
        }
        try:
            response = requests.get(ALL_SECURITIES_URL, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404 and start > 0: break
            if response.status_code == 429:
                print(f"429 Rate Limit (list, start={start}). Пауза {RATE_LIMIT_PAUSE} сек...")
                time.sleep(RATE_LIMIT_PAUSE)
                continue
            response.raise_for_status()
            data = response.json()
            securities_block = data.get("securities", {})
            securities_data = securities_block.get("data", [])
            if not securities_data: break
            all_potential_securities.extend(securities_data)
            if len(securities_data) < limit: break
            start += len(securities_data)
            if start % 500 == 0: print(f"Загружено {start} записей из общего списка облигаций...")
        except requests.exceptions.Timeout:
            print(f"Таймаут (list, start={start}). Повтор...")
            continue
        except requests.exceptions.RequestException as e:
            status_code = getattr(e.response, 'status_code', 'N/A')
            if status_code not in [404, 429]: print(f"Сетевая ошибка ({status_code}) (list): {e}")
            return []
        except Exception as e:
            print(f"Неожиданная ошибка (list): {e}")
            return []
    print(f"\nНайдено всего записей облигаций (stock/bonds): {len(all_potential_securities)}.")
    final_ofz_list = []
    secid_idx, type_idx = 0, 1
    if all_potential_securities and isinstance(all_potential_securities[0], list) and len(all_potential_securities[0]) >= 2:
        for sec_data in all_potential_securities:
            if isinstance(sec_data, list) and len(sec_data) > type_idx:
                secid, sec_type = sec_data[secid_idx], sec_data[type_idx]
                is_ofz_prefix = isinstance(secid, str) and secid.startswith('SU')
                is_correct_type = sec_type == 'ofz_bond'
                if is_ofz_prefix and is_correct_type: final_ofz_list.append(secid)
            else:
                 if not hasattr(get_ofz_list, 'format_warning_shown'):
                      print(f"Предупреждение (list): некорректный формат строки: {sec_data}")
                      get_ofz_list.format_warning_shown = True
    else:
        format_issue = "Формат первого элемента некорректен" if all_potential_securities else "Список пуст"
        print(f"Предупреждение (list): нет данных для фильтрации ({format_issue}).")

    if hasattr(get_ofz_list, 'format_warning_shown'): del get_ofz_list.format_warning_shown
    unique_ofz_list = sorted(list(set(final_ofz_list)))
    print(f"Отфильтровано ОФЗ (префикс SU и type='ofz_bond'): {len(unique_ofz_list)}.")
    if not unique_ofz_list: print("ПРЕДУПРЕЖДЕНИЕ: Список ОФЗ пуст.")
    return unique_ofz_list


def get_historical_trades_and_info(secid, start_date="2010-01-01"):
    """
    Получает историю + статику из API истории. Добавляет расчетные поля.
    Выбрасывает HistoryDownloadError при ошибке во время пагинации.
    """
    all_trades_combined = []
    start_index = 0
    limit = 100
    end_date = datetime.now().strftime("%Y-%m-%d")
    url = f"{HISTORY_BASE_URL}/{secid}.json"
    # Убран ISIN из запроса
    history_columns_to_request = ("TRADEDATE,CLOSE,YIELDCLOSE,VOLUME,VALUE,SHORTNAME,MATDATE,"
                                  "COUPONVALUE,COUPONPERIOD,DURATION,FACEVALUE,NEXTCOUPON,"
                                  "BOARDID,NUMTRADES,ACCINT") # ISIN убран
    static_info = {"SECID": secid}
    has_static_info = False
    columns = [] # Инициализируем columns вне цикла

    while True:
        params = {
            "iss.meta": "off", "from": start_date, "till": end_date,
            "start": start_index, "limit": limit,
            "history.columns": history_columns_to_request
        }
        response = None
        try:
            response = requests.get(url, params=params, timeout=HISTORY_TIMEOUT)
            if response.status_code == 429:
                print(f"429 Rate Limit (hist, {secid}, start={start_index}). Пауза {RATE_LIMIT_PAUSE} сек...")
                time.sleep(RATE_LIMIT_PAUSE)
                continue
            if response.status_code == 404:
                 if start_index == 0: pass
                 break
            response.raise_for_status()
            data = response.json()
            history_block = data.get("history", {})
            # Получаем columns здесь, чтобы они были доступны в блоке except ниже
            columns = history_block.get("columns", [])
            history_data = history_block.get("data", [])
            if not history_data: break

            # --- Извлечение статической информации (только один раз) ---
            if start_index == 0 and not has_static_info:
                try:
                    # Проверяем, что columns не пустой перед использованием zip
                    if not columns:
                        raise ValueError("Список колонок из API истории пуст.")
                    first_row_dict = dict(zip(columns, history_data[0]))
                    static_info["SHORTNAME"] = first_row_dict.get("SHORTNAME")
                    static_info["MATDATE"] = first_row_dict.get("MATDATE")
                    static_info["COUPONVALUE_hist"] = first_row_dict.get("COUPONVALUE")
                    static_info["COUPONPERIOD_hist"] = first_row_dict.get("COUPONPERIOD") if "COUPONPERIOD" in columns else None
                    static_info["FACEVALUE"] = first_row_dict.get("FACEVALUE")
                    static_info["NEXTCOUPON_hist"] = first_row_dict.get("NEXTCOUPON") if "NEXTCOUPON" in columns else None
                    # ISIN убран

                    if not static_info.get("SHORTNAME"):
                         print(f"Предупреждение (hist): Нет SHORTNAME для {secid}.")
                         return []
                    has_static_info = True
                except Exception as e:
                     print(f"Ошибка извлечения статики из истории для {secid}: {e}")
                     raise HistoryDownloadError(f"Static info extraction failed for {secid}: {e}", secid) from e

            # --- Обработка строк и расчеты ---
            current_page_records = []
            matdate_dt = pd.to_datetime(static_info.get("MATDATE"), errors='coerce')

            # Проверяем, что columns не пустой перед использованием zip в цикле
            if not columns:
                 print(f"Предупреждение: Список колонок пуст для {secid} на шаге {start_index}, пропуск страницы.")
                 start_index += len(history_data) # Пропускаем эту страницу
                 continue

            for row in history_data:
                try:
                    row_dict = dict(zip(columns, row))
                    close_price = pd.to_numeric(row_dict.get("CLOSE"), errors='coerce')
                    accint = pd.to_numeric(row_dict.get("ACCINT"), errors='coerce')
                    clean_price = close_price - accint if pd.notna(close_price) and pd.notna(accint) else np.nan
                    tradedate_dt = pd.to_datetime(row_dict.get("TRADEDATE"), errors='coerce')
                    ttm_years = ((matdate_dt - tradedate_dt).days / 365.25) if pd.notna(matdate_dt) and pd.notna(tradedate_dt) else np.nan

                    combined_record = {
                        "SECID": secid,
                        "SHORTNAME": static_info.get("SHORTNAME"),
                        # ISIN убран
                        "MATDATE": static_info.get("MATDATE"),
                        "FACEVALUE": static_info.get("FACEVALUE"),
                        "COUPONVALUE_hist": static_info.get("COUPONVALUE_hist"),
                        "COUPONPERIOD_hist": static_info.get("COUPONPERIOD_hist"),
                        "NEXTCOUPON_hist": static_info.get("NEXTCOUPON_hist"),
                        "TRADEDATE": row_dict.get("TRADEDATE"),
                        "BOARDID": row_dict.get("BOARDID"),
                        "CLOSE": close_price,
                        "VOLUME": pd.to_numeric(row_dict.get("VOLUME"), errors='coerce'),
                        "VALUE": pd.to_numeric(row_dict.get("VALUE"), errors='coerce'),
                        "NUMTRADES": pd.to_numeric(row_dict.get("NUMTRADES"), errors='coerce'),
                        "YIELDCLOSE": pd.to_numeric(row_dict.get("YIELDCLOSE"), errors='coerce'),
                        "ACCINT": accint,
                        "DURATION": pd.to_numeric(row_dict.get("DURATION"), errors='coerce'),
                        "CleanPrice": clean_price,
                        "TTM_years": ttm_years,
                    }
                    current_page_records.append(combined_record)
                except Exception as e:
                    print(f"Ошибка обработки строки истории для {secid} (start={start_index}): {row}. Ошибка: {e}")

            all_trades_combined.extend(current_page_records)
            if len(history_data) < limit: break
            start_index += len(history_data)

        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            status_code = getattr(response, 'status_code', 'N/A')
            error_message = f"Ошибка сети/таймаут ({status_code}) при получении истории для {secid} (start={start_index}): {e}"
            print(error_message)
            raise HistoryDownloadError(error_message, secid, partial_data=all_trades_combined) from e
        except Exception as e:
            error_message = f"Неожиданная ошибка при обработке истории для {secid} (start={start_index}): {e}"
            print(error_message)
            raise HistoryDownloadError(error_message, secid, partial_data=all_trades_combined) from e

    # Обработка случая, когда история есть, а статики из 1й стр нет
    if not has_static_info and all_trades_combined:
         print(f"Предупреждение: Собраны строки истории {secid}, но нет статики (не было 1й стр?).")
         # Обновляем записи, используя данные из самой строки истории, если возможно
         # (columns должны быть доступны с последнего успешного запроса)
         if columns: # Проверяем наличие колонок
             for record in all_trades_combined:
                 # Пытаемся заполнить из самой записи (record уже содержит данные из row_dict)
                 # Эти поля уже должны быть заполнены при создании record, проверяем на всякий случай
                 record["SHORTNAME"] = record.get("SHORTNAME", static_info.get("SHORTNAME")) # Пытаемся сохранить, если он был
                 record["MATDATE"] = record.get("MATDATE", static_info.get("MATDATE"))
                 record["FACEVALUE"] = record.get("FACEVALUE", static_info.get("FACEVALUE"))
                 # Для купонных полей из истории - они уже должны быть там или None
                 record["COUPONVALUE_hist"] = record.get("COUPONVALUE_hist")
                 record["COUPONPERIOD_hist"] = record.get("COUPONPERIOD_hist")
                 record["NEXTCOUPON_hist"] = record.get("NEXTCOUPON_hist")
         else:
              print(f"Ошибка: Не удалось обновить записи для {secid}, т.к. список колонок не определен.")
         print(f"-> Данные {secid} будут добавлены, статика попробуется позже.")

    return all_trades_combined


def get_static_bond_info(secid):
    """Получает статическую информацию для одной облигации."""
    request_url = f"{SECURITY_INFO_URL}/{secid}.json"
    params = {
        "iss.meta": "off",
        "iss.only": "securities",
        # Убран TYPE, добавлен SECNAME
        "securities.columns": ("SECID,ISIN,STATUS,SECTYPE,SECNAME,FACEVALUE," # TYPE убран, SECNAME добавлен
                               "COUPONVALUE,COUPONPERIOD,NEXTCOUPON,ISSUESIZEPLACED")
    }
    static_data = {"SECID": secid}
    try:
        response = requests.get(request_url, params=params, timeout=STATIC_INFO_TIMEOUT)
        if response.status_code == 404: return static_data
        if response.status_code == 429:
            print(f"429 Rate Limit (static, {secid}). Пауза {RATE_LIMIT_PAUSE} сек...")
            time.sleep(RATE_LIMIT_PAUSE)
            response = requests.get(request_url, params=params, timeout=STATIC_INFO_TIMEOUT)
            if response.status_code in [404, 429]: return static_data
        response.raise_for_status()
        data = response.json()
        securities_block = data.get("securities", {})
        columns = securities_block.get("columns", [])
        sec_data_list = securities_block.get("data", [])

        if sec_data_list and columns:
            sec_data = sec_data_list[0]
            sec_dict = dict(zip(columns, sec_data))

            static_data["ISIN_static"] = sec_dict.get("ISIN")
            static_data["STATUS"] = sec_dict.get("STATUS")
            static_data["SECTYPE"] = sec_dict.get("SECTYPE")
            static_data["SECNAME"] = sec_dict.get("SECNAME") # Добавлено
            static_data["FACEVALUE_static"] = sec_dict.get("FACEVALUE")
            static_data["COUPONVALUE_static"] = sec_dict.get("COUPONVALUE")
            static_data["COUPONPERIOD_static"] = sec_dict.get("COUPONPERIOD")
            static_data["NEXTCOUPON_static"] = sec_dict.get("NEXTCOUPON")
            static_data["ISSUESIZEPLACED"] = sec_dict.get("ISSUESIZEPLACED")

    except requests.exceptions.Timeout:
        print(f"Таймаут (static) для {secid}. Пропуск статики.")
    except requests.exceptions.RequestException as e:
        status_code = getattr(e.response, 'status_code', 'N/A')
        if status_code not in [404, 429]:
            print(f"Сетевая ошибка ({status_code}) (static) для {secid}: {e}. Пропуск статики.")
    except Exception as e:
        print(f"Неожиданная ошибка (static) для {secid}: {e}. Пропуск статики.")

    return static_data


# --- Основное выполнение ---

def main():
    try: max_workers = len(os.sched_getaffinity(0)); print(f"Доступных ядер CPU: {max_workers}")
    except AttributeError: max_workers = os.cpu_count(); print(f"CPU (логических ядер): {max_workers}")
    if max_workers is None: max_workers = 4; print("Не удалось определить кол-во ядер, используется 4.")
    # max_workers = min(max_workers, 10) # Ограничение

    start_time_script = datetime.now()
    print(f"Запуск скрипта: {start_time_script.strftime('%Y-%m-%d %H:%M:%S')}")

    ofz_list = get_ofz_list()
    if not ofz_list: return

    print(f"\nНачинаем основную обработку истории для {len(ofz_list)} ОФЗ ({max_workers} потоков)...")
    all_combined_data = []
    processed_securities_count = 0
    total_rows_accumulated = 0
    last_checkpoint = 0
    failed_secids_initial = []

    # --- Основной параллельный сбор истории ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_secid = {executor.submit(get_historical_trades_and_info, secid): secid for secid in ofz_list}
        for future in concurrent.futures.as_completed(future_to_secid):
            secid = future_to_secid[future]
            processed_securities_count += 1
            try:
                result_list = future.result()
                if result_list:
                    all_combined_data.extend(result_list)
                    new_rows = len(result_list)
                    total_rows_accumulated += new_rows
                    print(f"[{processed_securities_count}/{len(ofz_list)}] {secid}: +{new_rows} строк (Всего: {total_rows_accumulated})")
                    current_checkpoint = total_rows_accumulated // ROW_CHECKPOINT
                    if current_checkpoint > last_checkpoint:
                       print(f"\n--- Чекпоинт: Собрано {total_rows_accumulated} строк ({datetime.now().strftime('%H:%M:%S')}) ---")
                       print(f"--- Продолжение сбора данных... ---")
                       last_checkpoint = current_checkpoint
            except HistoryDownloadError as e:
                print(f"!!! Ошибка истории {secid}: {e}. В список для повтора.")
                failed_secids_initial.append(secid)
            except Exception as exc:
                print(f"!!! Неожиданное исключение (Future) для {secid}: {exc}")
                failed_secids_initial.append(secid)

    print(f"\nОсновная обработка истории завершена. {len(failed_secids_initial)} ОФЗ требуют повтора.")

    # --- Повторная обработка сбойных SECID (история) ---
    failed_secids_current_retry = list(set(failed_secids_initial))
    retry_attempt = 0
    while retry_attempt < MAX_RETRIES and failed_secids_current_retry:
        retry_attempt += 1
        print(f"\n--- Повтор истории #{retry_attempt}/{MAX_RETRIES} для {len(failed_secids_current_retry)} ОФЗ ---")
        time.sleep(RETRY_PAUSE / 2)
        failed_in_this_retry = []
        for i, secid in enumerate(failed_secids_current_retry):
            print(f"Повтор {retry_attempt} {secid} ({i+1}/{len(failed_secids_current_retry)})...")
            try:
                time.sleep(1)
                result_list = get_historical_trades_and_info(secid)
                if result_list:
                    all_combined_data.extend(result_list)
                    new_rows = len(result_list); total_rows_accumulated += new_rows
                    print(f"  -> Успех повтора {secid}: +{new_rows} строк (Всего: {total_rows_accumulated})")
                    current_checkpoint = total_rows_accumulated // ROW_CHECKPOINT
                    if current_checkpoint > last_checkpoint:
                        print(f"\n--- Чекпоинт (повтор): Собрано {total_rows_accumulated} строк ({datetime.now().strftime('%H:%M:%S')}) ---")
                        last_checkpoint = current_checkpoint
            except HistoryDownloadError as e:
                print(f"  -> Ошибка повтора {retry_attempt} {secid}: {e}")
                failed_in_this_retry.append(secid)
            except Exception as exc:
                print(f"  -> Исключение при повторе {retry_attempt} {secid}: {exc}")
                failed_in_this_retry.append(secid)
            time.sleep(0.5)
        failed_secids_current_retry = failed_in_this_retry
        if failed_secids_current_retry:
            print(f"--- Попытка #{retry_attempt} завершена. {len(failed_secids_current_retry)} ОФЗ не загружены. Пауза {RETRY_PAUSE} сек... ---")
            time.sleep(RETRY_PAUSE)
        else: print(f"--- Попытка #{retry_attempt} завершена. ---")

    if not all_combined_data:
        print("\nНет данных истории для обработки. Выход.")
        return

    # --- Преобразование исторических данных в DataFrame ---
    print("\nСоздание DataFrame из исторических данных...")
    try:
        df_history = pd.DataFrame(all_combined_data)
        initial_rows_hist = len(df_history)
        df_history = df_history.drop_duplicates(subset=['SECID', 'TRADEDATE'], keep='last')
        final_rows_hist = len(df_history)
        if initial_rows_hist != final_rows_hist:
            print(f"Очистка дубликатов истории: Удалено {initial_rows_hist - final_rows_hist} строк.")
        total_rows_accumulated = final_rows_hist
    except Exception as e:
         print(f"Ошибка создания DataFrame истории: {e}. Попытка сохранить сырые данные...")
         try:
             raw_filename = f"ofz_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
             with open(raw_filename, 'w', encoding='utf-8') as f:
                 import json
                 json.dump(all_combined_data, f, ensure_ascii=False, indent=2)
             print(f"!!! Сырые данные сохранены в: {raw_filename}")
         except Exception as dump_err: print(f"!!! Не удалось сохранить сырые данные: {dump_err}")
         return

    # --- Сбор статической информации ---
    unique_secids_in_history = df_history['SECID'].unique()
    print(f"\nНачинаем сбор статической информации для {len(unique_secids_in_history)} уникальных ОФЗ...")
    static_data_list = []
    processed_static_count = 0
    for secid in unique_secids_in_history:
        processed_static_count += 1
        if processed_static_count % 20 == 0 or processed_static_count == len(unique_secids_in_history):
             print(f"Запрос статики {processed_static_count}/{len(unique_secids_in_history)}: {secid}...")
        static_info = get_static_bond_info(secid)
        static_data_list.append(static_info)
        time.sleep(0.3)

    # --- Создание и объединение DataFrame статики ---
    print("\nСоздание DataFrame статики и объединение...")
    df_static = pd.DataFrame(static_data_list)
    df_static = df_static.dropna(subset=['SECID'])
    df_merged = pd.merge(df_history, df_static, on='SECID', how='left')

    # --- Финальные расчеты и обработка ---
    print("Выполнение финальных расчетов (TURNOVER_RATIO)...")
    df_merged['ISSUESIZEPLACED'] = pd.to_numeric(df_merged['ISSUESIZEPLACED'], errors='coerce')
    df_merged['ISSUESIZEPLACED'] = df_merged['ISSUESIZEPLACED'].replace(0, np.nan)
    df_merged['TURNOVER_RATIO'] = df_merged['VOLUME'] / df_merged['ISSUESIZEPLACED']
    df_merged['TURNOVER_RATIO'] = df_merged['TURNOVER_RATIO'].replace([np.inf, -np.inf], np.nan)

    # --- Сохранение результатов ---
    end_time_script = datetime.now()
    total_duration = end_time_script - start_time_script
    print(f"\nСбор и обработка данных завершены ({retry_attempt} попыток повтора).")
    if failed_secids_current_retry:
         print(f"!!! ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить ПОЛНУЮ ИСТОРИЮ для: {failed_secids_current_retry}")
    print(f"Общее время выполнения: {total_duration}")

    if not df_merged.empty:
        print(f"\nВсего строк в итоговом DataFrame: {len(df_merged)}")
        print("Подготовка и сохранение финального DataFrame в CSV...")
        try:
            # Преобразование типов данных
            df_merged['TRADEDATE'] = pd.to_datetime(df_merged['TRADEDATE'], errors='coerce')
            df_merged['MATDATE'] = pd.to_datetime(df_merged['MATDATE'], errors='coerce')
            if 'NEXTCOUPON_hist' in df_merged.columns:
                df_merged['NEXTCOUPON_hist'] = pd.to_datetime(df_merged['NEXTCOUPON_hist'], errors='coerce')
            if 'NEXTCOUPON_static' in df_merged.columns:
                df_merged['NEXTCOUPON_static'] = pd.to_datetime(df_merged['NEXTCOUPON_static'], errors='coerce')

            # Колонки с числами (TYPE убран)
            numeric_cols = ['CLOSE', 'YIELDCLOSE', 'VOLUME', 'VALUE', 'NUMTRADES', 'ACCINT', 'DURATION',
                            'FACEVALUE', 'FACEVALUE_static',
                            'COUPONVALUE_hist', 'COUPONPERIOD_hist',
                            'COUPONVALUE_static', 'COUPONPERIOD_static',
                            'ISSUESIZEPLACED', 'CleanPrice', 'TTM_years', 'TURNOVER_RATIO'] # TYPE убран
            for col in numeric_cols:
                 if col in df_merged.columns:
                    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

            df_merged = df_merged.sort_values(by=['SECID', 'TRADEDATE'], ascending=[True, True])

            # Переупорядочивание колонок (обновлено)
            ordered_cols = [
                # ID
                'SECID', 'SHORTNAME', 'ISIN_static', # ISIN убран
                # Статика
                'MATDATE', 'FACEVALUE', 'STATUS',
                'SECTYPE', 'SECNAME', # TYPE убран, SECNAME добавлен
                'COUPONVALUE_static', 'COUPONPERIOD_static', 'NEXTCOUPON_static',
                'ISSUESIZEPLACED',
                 # Статика (из истории)
                'COUPONVALUE_hist', 'COUPONPERIOD_hist', 'NEXTCOUPON_hist',
                # Ежедневные торги
                'TRADEDATE', 'BOARDID', 'CLOSE', 'VOLUME', 'VALUE', 'NUMTRADES',
                # Ежедневная оценка
                'ACCINT', 'CleanPrice', 'YIELDCLOSE', 'TTM_years',
                # Ежедневный риск
                'DURATION',
                # Относительная ликвидность
                'TURNOVER_RATIO',
                # Дополнительно
                 'FACEVALUE_static'
            ]
            final_cols = [col for col in ordered_cols if col in df_merged.columns]
            final_cols.extend([col for col in df_merged.columns if col not in final_cols])
            df_merged = df_merged[final_cols]

            # Сохранение
            filename = f"ofz_full_history_ext_{end_time_script.strftime('%Y%m%d_%H%M%S')}.csv"
            df_merged.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"Данные успешно сохранены в файл: {filename}")

            print("\nИнформация о финальном DataFrame:")
            df_merged.info(memory_usage='deep', verbose=True)

        except Exception as e:
            print(f"Ошибка при создании/сохранении финального DataFrame: {e}")
    else:
        print("\nНет данных для сохранения.")

if __name__ == "__main__":
    main()