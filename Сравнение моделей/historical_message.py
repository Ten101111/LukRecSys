import time

import pandas as pd
import datetime as dt
import re
import glob
import os

from constants import PATH_TO_SENDINGS_FILES, PATH_ALREADY_SEND_MODIF


def extract_kccc_daily_checks(excel_path: str) -> pd.DataFrame:
    """
    Читает файл формата, как у тебя, и возвращает DataFrame с колонками:
    'КССС', 'Дата', 'Совокупное кол-во суточных чеков'.

    Месяц берём из первой ячейки (A1),
    год — из названия файла (например, '12.25 Декабрь 2025.xlsx').
    """
    # ---- ГОД ИЗ НАЗВАНИЯ ФАЙЛА ----
    fname = os.path.basename(excel_path)
    m = re.search(r'(20\d{2})', fname)
    if not m:
        raise ValueError(f"Не удалось определить год из имени файла: {fname}")
    year = int(m.group(1))

    # Читаем лист БЕЗ заголовков
    df = pd.read_excel(excel_path, header=None)

    # --- 1. Определяем месяц по первой ячейке ---
    first_cell = str(df.iloc[0, 0])

    months_map = {
        'январ': 1,
        'феврал': 2,
        'март': 3,
        'апрел': 4,
        'ма': 5,
        'июн': 6,
        'июл': 7,
        'август': 8,
        'сентябр': 9,
        'октябр': 10,
        'ноябр': 11,
        'декабр': 12
    }

    month_num = None
    for key, num in months_map.items():
        if re.search(key, first_cell, re.IGNORECASE):
            month_num = num
            break

    if month_num is None:
        raise ValueError("Не удалось определить месяц из первой ячейки.")

    # --- 2. Находим строку с 'Показатель' и определяем столбцы с днями ---
    row_pok = None
    for idx, row in df.iterrows():
        if any(isinstance(x, str) and 'Показатель' in x for x in row):
            row_pok = idx
            break

    if row_pok is None:
        raise ValueError("Не найдена строка с текстом 'Показатель'.")

    # Столбцы, в которых в этой строке лежат числа 1..31 — это дни
    days_cols = []
    for col in df.columns:
        val = df.loc[row_pok, col]
        if isinstance(val, (int, float)) and not pd.isna(val):
            day = int(val)
            if 1 <= day <= 31:
                days_cols.append((col, day))

    days_cols.sort(key=lambda x: x[0])  # по номеру столбца

    # --- 3. Находим столбец с 'КССС' ---
    kccc_col = None
    for col in df.columns:
        if any(isinstance(x, str) and 'КССС' in x for x in df[col]):
            kccc_col = col
            break

    if kccc_col is None:
        raise ValueError("Не найден столбец с заголовком 'КССС'.")

    # --- 4. Строки с 'Итого прогноз чеков на сутки' ---
    total_rows_idx = []
    for idx, row in df.iterrows():
        if any(isinstance(x, str) and 'Итого прогноз чеков на сутки' in x for x in row):
            total_rows_idx.append(idx)

    records = []

    for idx in total_rows_idx:
        # Поднимаемся вверх до ближайшей ячейки в столбце КССС
        kccc_val = None
        scan_idx = idx
        while scan_idx >= 0:
            val = df.loc[scan_idx, kccc_col]
            if isinstance(val, str) or (isinstance(val, (int, float)) and not pd.isna(val)):
                # Пропускаем сам заголовок 'КССС'
                if isinstance(val, str) and 'КССС' in val:
                    kccc_val = None
                else:
                    kccc_val = val
                break
            scan_idx -= 1

        if kccc_val is None:
            # если не нашли КССС — пропускаем эту строку
            continue

        # По всем дням месяца забираем значения чеков
        for col, day in days_cols:
            checks = df.loc[idx, col]
            if pd.isna(checks):
                continue
            try:
                checks_val = float(checks)
            except Exception:
                continue

            date = dt.date(year, month_num, day)
            records.append({
                'КССС': kccc_val,
                'Дата': date,
                'Совокупное кол-во суточных чеков': checks_val
            })

    result_df = pd.DataFrame(records)
    result_df.sort_values(['КССС', 'Дата'], inplace=True)
    return result_df


def transpose_all_files(files_all):
    files_df = []
    for i in files_all:
        print('----- Приступили к обработке массива из файла -----')
        print(f"Работаем над {i.split('\\')[-1]}")
        stsrt = time.time()
        df = extract_kccc_daily_checks(i)
        files_df.append(df)
        print(f"Выполнено за {round(time.time()-stsrt, 2)} сек.\n")
    reslt_df = pd.concat(files_df, ignore_index=True)
    return reslt_df

def build_kccc_file(files_all) -> None:
    """
    Основная функция: читает исходный файл и создаёт новый Excel-файл
    с колонками: КССС, Дата, Совокупное кол-во суточных чеков.

    Год берётся из имени входного файла.
    """
    print(f"----- Начинаем создание общего массива из {len(files_all)} файлов -----\n")
    stat = time.time()
    reslt_df = transpose_all_files(files_all)
    now = dt.datetime.now()
    os.mkdir(PATH_ALREADY_SEND_MODIF)
    file_name = f'Файл с прогнозом на {now.day}.{now.month}.{now.day}.csv'
    full_path = os.path.join(PATH_ALREADY_SEND_MODIF, file_name)
    reslt_df.to_csv(full_path, index=False, sep='|')
    print(f"Выполнено за {round(time.time()-stat, 1)} сек.\n")
    return reslt_df

files_all = glob.glob(PATH_TO_SENDINGS_FILES+"*.xlsx")

build_kccc_file(files_all=files_all)

