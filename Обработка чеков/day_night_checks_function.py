from collections import defaultdict
import datetime as dt
import glob
import pandas as pd
import os
from typing import Iterable, List, Optional, Union
import time
import numpy as np  # если ещё не импортирован выше

from new_checks_add import _load_state, _pick_latest_added_file, _resolve_manual_files, _save_state

def _aggregate_file_by_day_night(file, unique=True):
    """
    Агрегация одного CSV по дням и интервалам времени (08_20 / other).
    Возвращает DataFrame с колонками:
    ['ORG_KSSS','year','month','day','time_segment','unique_cheques' | 'cheques_count'].
    """
    dtype_dict = {
        'CHEQUE_ID': 'str',
        'ORG_KSSS':  'str',
        'OPERATION_DATE': 'object'
    }
    accum = defaultdict(int)

    for chunk in pd.read_csv(
        file,
        sep='|',
        dtype=dtype_dict,
        usecols=['CHEQUE_ID', 'ORG_KSSS', 'OPERATION_DATE'],
        chunksize=100_000
    ):
        # Приводим дату-время
        chunk['OPERATION_DATE'] = pd.to_datetime(
            chunk['OPERATION_DATE'],
            errors='coerce',
            dayfirst=True,
            infer_datetime_format=True
        )
        chunk = chunk.dropna(subset=['OPERATION_DATE'])
        if chunk.empty:
            continue

        # Выделяем компоненты даты
        dt_series = chunk['OPERATION_DATE']
        chunk['year']  = dt_series.dt.year
        chunk['month'] = dt_series.dt.month
        chunk['day']   = dt_series.dt.day

        # Выделяем час и формируем интервал
        hours = dt_series.dt.hour
        chunk['time_segment'] = np.where(
            hours.between(8, 19),  # с 08:00 до 19:59
            '08_20',
            'other'
        )

        group_cols = ['ORG_KSSS', 'year', 'month', 'day', 'time_segment']
        if unique:
            grouped = chunk.groupby(group_cols)['CHEQUE_ID'].nunique()
        else:
            grouped = chunk.groupby(group_cols).size()

        for key, val in grouped.items():
            accum[key] += int(val)

    col_name = 'unique_cheques' if unique else 'cheques_count'
    df = pd.DataFrame(
        [(k[0], k[1], k[2], k[3], k[4], v) for k, v in accum.items()],
        columns=['ORG_KSSS', 'year', 'month', 'day', 'time_segment', col_name]
    )
    return df


def _read_day_night_master(master_path, unique=True):
    """
    Читает мастер-файл с агрегацией по дням и интервалам (день/ночь).
    Если файла нет/пустой, отдаёт пустой DataFrame правильной структуры.
    """
    col_name = 'unique_cheques' if unique else 'cheques_count'
    cols = ['ORG_KSSS', 'year', 'month', 'day', 'time_segment', col_name]

    if not os.path.exists(master_path):
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(
            master_path,
            sep='|',
            dtype={
                'ORG_KSSS': 'str',
                'year': 'int64',
                'month': 'int64',
                'day': 'int64',
                'time_segment': 'str'
            }
        )
        # на всякий случай убедимся, что есть все нужные колонки
        for c in cols:
            if c not in df.columns:
                df[c] = pd.Series(dtype='int64' if c == col_name else 'object')
        return df[cols]
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)


def update_by_day_night(
    input_dir: str,
    out_dir:   str,
    master_filename: str = 'День и Ночь_master.csv',
    unique: bool = True,
    append_only_missing: bool = True,
    # режимы как в update_by_day:
    mode: str = 'state',            # 'state' | 'latest' | 'manual'
    files: Optional[Union[str, Iterable[str]]] = None,
    pattern: str = '*.csv',
    skip_if_in_state: bool = True,
    write_master: bool = False,
    reset_master: bool = False
):
    """
    Универсальное обновление агрегации по дням и интервалам (08_20 / other).

    Режимы:
      - mode='state'  : берём все новые/изменённые файлы (по size+mtime).
      - mode='latest' : берём ТОЛЬКО один файл — последний добавленный в папку (по ctime).
      - mode='manual' : берём РОВНО те файлы, что переданы через параметр `files`.

    Логика append_only_missing:
      - True  -> дописываем только НОВЫЕ группы (ORG_KSSS,year,month,day,time_segment),
                 если такая комбинация уже есть в мастере — она игнорируется.
      - False -> upsert с пересчётом: мастер + новые данные, затем groupby и sum.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    col_name = 'unique_cheques' if unique else 'cheques_count'

    # отдельный реестр под день/ночь, чтобы не пересекаться с дневным мастером
    state_path = os.path.join(out_dir, '_processed_files_day_night.csv')
    state_df = _load_state(state_path)

    # 1. Определяем список файлов к обработке
    files_to_process: List[str] = []

    if mode == 'latest':
        latest = _pick_latest_added_file(input_dir, pattern=pattern)
        if latest:
            files_to_process = [latest]
        print(f"Режим: latest (day_night). Последний добавленный файл: "
              f"{os.path.basename(latest) if latest else 'не найден'}")

    elif mode == 'manual':
        if not files:
            print("Режим: manual (day_night). Не передан список/путь файлов — нечего обрабатывать.")
            return
        files_to_process = _resolve_manual_files(files, pattern=pattern)
        print(f"Режим: manual (day_night). Файлов к обработке: {len(files_to_process)}")
        for p in files_to_process:
            print("  -", os.path.basename(p))

    else:  # 'state' по умолчанию
        all_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
        seen = {(row['file'], row['size'], row['mtime']) for _, row in state_df.iterrows()}
        for f in all_files:
            try:
                size = os.path.getsize(f)
                mtime = int(os.path.getmtime(f))
            except FileNotFoundError:
                continue
            sig = (f, size, mtime)
            if sig not in seen:
                files_to_process.append(f)
        print(f"Режим: state (day_night). Новых/изменённых файлов: {len(files_to_process)}")
        for p in files_to_process:
            print("  -", os.path.basename(p))

    # 2. Фильтр по уже учтённым файлам (актуально для latest/manual)
    new_rows = []
    if skip_if_in_state and files_to_process:
        filtered = []
        seen = {(row['file'], row['size'], row['mtime']) for _, row in state_df.iterrows()}
        for f in files_to_process:
            try:
                size = os.path.getsize(f)
                mtime = int(os.path.getmtime(f))
            except FileNotFoundError:
                continue
            sig = (f, size, mtime)
            if sig in seen:
                print(f"Пропуск (day_night, уже в реестре): {os.path.basename(f)}")
                continue
            filtered.append(f)
            new_rows.append({'file': f, 'size': size, 'mtime': mtime})
        files_to_process = filtered

    if not files_to_process:
        print('Нет файлов для обработки (day_night) — выхожу.')
        return

    # 3. Читаем/создаём мастер по день/ночь
    if not master_filename:
        master_filename = 'День и Ночь_master.csv'
    master_path = os.path.join(out_dir, str(master_filename))
    if reset_master:
        master = pd.DataFrame(columns=['ORG_KSSS', 'year', 'month', 'day', 'time_segment', col_name])
    else:
        master = _read_day_night_master(master_path, unique=unique)

    # 4. Агрегируем выбранные файлы
    t_agg = time.time()
    parts = []
    for f in files_to_process:
        print(f"Агрегирую (day_night): {os.path.basename(f)}")
        df = _aggregate_file_by_day_night(f, unique=unique)
        if not df.empty:
            parts.append(df)

    if not parts:
        print('Выбранные файлы (day_night) не дали данных — завершаю.')
        if new_rows:
            state_df = pd.concat([state_df, pd.DataFrame(new_rows)], ignore_index=True)
            _save_state(state_path, state_df)
        return

    new_data = pd.concat(parts, ignore_index=True)
    new_data = (
        new_data
        .groupby(['ORG_KSSS', 'year', 'month', 'day', 'time_segment'], as_index=False)[col_name]
        .sum()
    )
    print(f'Агрегация (day_night) заняла: {time.time() - t_agg:.2f} сек')

    # 5. Обновляем мастер
    key_cols = ['ORG_KSSS', 'year', 'month', 'day', 'time_segment']

    if append_only_missing:
        if master.empty:
            to_append = new_data
        else:
            merged = new_data.merge(master[key_cols], on=key_cols, how='left', indicator=True)
            to_append = merged.loc[merged['_merge'] == 'left_only', new_data.columns]

        if to_append.empty:
            print('Все группы (day_night) уже есть в мастере — дописывать нечего.')
        else:
            master = pd.concat([master, to_append], ignore_index=True)
    else:
        combined = pd.concat([master, new_data], ignore_index=True)
        master = (
            combined
            .groupby(key_cols, as_index=False)[col_name]
            .sum()
        )

    # 6. Сохраняем мастер и снапшот
    master = master.sort_values(key_cols).reset_index(drop=True)

    now = dt.datetime.now()
    dated_path = os.path.join(
        out_dir,
        f'День и Ночь по дням_{now.day}.{now.month}.{now.year} {now.hour}.{now.minute}.csv'
    )
    master.to_csv(dated_path, index=False, sep='|')
    if write_master:
        master_dir = os.path.dirname(master_path)
        if master_dir:
            os.makedirs(master_dir, exist_ok=True)
        master.to_csv(master_path, index=False, sep='|')

    # 7. Обновляем реестр обработанных файлов
    if new_rows:
        state_df = pd.concat([state_df, pd.DataFrame(new_rows)], ignore_index=True)
        _save_state(state_path, state_df)

    print('\n--- Итог (day_night) ---')
    print(f'Файлов обработано: {len(files_to_process)}')
    print(f'Итоговый размер мастера: {master.shape}')
    print(f'Готово за {time.time() - t0:.2f} сек')

    for p in glob.glob(os.path.join(out_dir, '*_processed_files.csv')):
        try:
            os.remove(p)
            print(f'Удалил: {p}')
        except Exception as e:
            print(f'Не удалось удалить {p}: {e}')
