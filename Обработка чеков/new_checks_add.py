# -*- coding: utf-8 -*-
import pandas as pd
import time
from collections import defaultdict
import os
import glob
import datetime as dt
import warnings
from typing import Iterable, List, Optional, Union

warnings.filterwarnings('ignore')


# =========================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================
def _aggregate_file_by_day(file, unique=True):
    """
    Агрегация одного CSV по дням.
    Возвращает DataFrame с колонками:
    ['ORG_KSSS','year','month','day','unique_cheques'| 'cheques_count'].
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
        chunk['OPERATION_DATE'] = pd.to_datetime(
            chunk['OPERATION_DATE'],
            errors='coerce',
            dayfirst=True,
            infer_datetime_format=True
        )
        chunk = chunk.dropna(subset=['OPERATION_DATE'])
        if chunk.empty:
            continue

        chunk['year']  = chunk['OPERATION_DATE'].dt.year
        chunk['month'] = chunk['OPERATION_DATE'].dt.month
        chunk['day']   = chunk['OPERATION_DATE'].dt.day

        group_cols = ['ORG_KSSS', 'year', 'month', 'day']
        if unique:
            grouped = chunk.groupby(group_cols)['CHEQUE_ID'].nunique()
        else:
            grouped = chunk.groupby(group_cols).size()

        for key, val in grouped.items():
            accum[key] += int(val)

    col_name = 'unique_cheques' if unique else 'cheques_count'
    df = pd.DataFrame(
        [(k[0], k[1], k[2], k[3], v) for k, v in accum.items()],
        columns=['ORG_KSSS', 'year', 'month', 'day', col_name]
    )
    return df


def _load_state(state_path):
    """Читает реестр обработанных файлов (_processed_files.csv)."""
    if os.path.exists(state_path):
        try:
            df = pd.read_csv(state_path)
            expected = {'file', 'size', 'mtime'}
            if not expected.issubset(set(df.columns)):
                return pd.DataFrame(columns=['file', 'size', 'mtime'])
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=['file', 'size', 'mtime'])
    return pd.DataFrame(columns=['file', 'size', 'mtime'])


def _save_state(state_path, state_df):
    """Сохраняет реестр обработанных файлов."""
    state_df.to_csv(state_path, index=False)


def _read_master(master_path, unique=True):
    """Читает мастер CSV, если есть. Возвращает пустой df, если файла нет/пустой."""
    col_name = 'unique_cheques' if unique else 'cheques_count'
    if not os.path.exists(master_path):
        return pd.DataFrame(columns=['ORG_KSSS', 'year', 'month', 'day', col_name])
    try:
        df = pd.read_csv(
            master_path,
            sep='|',
            dtype={'ORG_KSSS':'str', 'year':'int64', 'month':'int64', 'day':'int64'}
        )
        if col_name not in df.columns:
            df[col_name] = pd.Series(dtype='int64')
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=['ORG_KSSS', 'year', 'month', 'day', col_name])


def _pick_latest_added_file(input_dir: str, pattern: str = '*.csv') -> Optional[str]:
    """
    Возвращает путь к ФАЙЛУ, который был добавлен в папку последним.
    На Windows используется время создания (ctime).
    """
    latest_path = None
    latest_ctime = -1.0
    for entry in os.scandir(input_dir):
        if not entry.is_file():
            continue
        if not glob.fnmatch.fnmatch(entry.name, pattern):
            continue
        try:
            st = entry.stat()
        except FileNotFoundError:
            continue
        # На Windows st.st_ctime — время создания файла.
        ctime = st.st_ctime
        # Если нужно, можно учесть st.st_mtime как tie-breaker.
        if ctime > latest_ctime:
            latest_ctime = ctime
            latest_path = entry.path
    return latest_path


def _resolve_manual_files(files: Union[str, Iterable[str]], pattern: str = '*.csv') -> List[str]:
    """
    Преобразует вход в список файлов:
    - строка с папкой -> берём *.csv в этой папке
    - glob-строка -> разворачиваем
    - один путь к файлу -> делаем список из одного элемента
    - список путей -> фильтруем существующие
    """
    result: List[str] = []
    if isinstance(files, str):
        if os.path.isdir(files):
            result = sorted(glob.glob(os.path.join(files, pattern)))
        elif any(ch in files for ch in ['*', '?', '[', ']']):
            result = sorted(glob.glob(files))
        else:
            if os.path.exists(files):
                result = [files]
    else:
        # iterable путей
        for p in files:
            if os.path.isdir(p):
                result.extend(sorted(glob.glob(os.path.join(p, pattern))))
            elif any(ch in p for ch in ['*', '?', '[', ']']):
                result.extend(sorted(glob.glob(p)))
            else:
                if os.path.exists(p):
                    result.append(p)
    # уберём дубликаты, только существующие файлы
    result = [p for p in dict.fromkeys(result) if os.path.isfile(p)]
    return result


# =========================
#  ОСНОВНАЯ ФУНКЦИЯ
# =========================
def update_by_day(
    input_dir: str = r'D:\Изменение предиктивки\Обработка чеков\Чеки',
    out_dir:   str = r'D:\Изменение предиктивки\Обработка чеков\Объединение чеков',
    master_filename: str = 'Объединение чеков по дням_master.csv',
    unique: bool = True,
    append_only_missing: bool = True,
    # новые опции:
    mode: str = 'state',            # 'state' | 'latest' | 'manual'
    files: Optional[Union[str, Iterable[str]]] = None,  # используется при mode='manual'
    pattern: str = '*.csv',
    skip_if_in_state: bool = True,  # пропускать файл, если он уже учтён в реестре (актуально для mode='latest'/'manual')
    write_master: bool = False,
    reset_master: bool = False
):
    """
    Универсальное обновление дневной агрегации чеков.

    Режимы:
      - mode='state'  : как раньше — берём все новые/изменённые файлы (по size+mtime).
      - mode='latest' : берём ТОЛЬКО один файл — последний добавленный в папку (по ctime на Windows).
      - mode='manual' : берём РОВНО те файлы, что вы передали через параметр `files` (путь/список/glob).

    Параметры хранения:
      - append_only_missing=True  : дописывает только НОВЫЕ группы (ORG_KSSS,y,m,d).
      - append_only_missing=False : upsert (сумма по совпадающим ключам). ВНИМАНИЕ: при unique=True возможен «пересчёт вверх».

    Реестр: out_dir/_processed_files.csv (file, size, mtime) — чтобы не пересчитывать старое.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    col_name = 'unique_cheques' if unique else 'cheques_count'

    # Реестр
    state_path = os.path.join(out_dir, '_processed_files.csv')
    state_df = _load_state(state_path)

    # Определяем список файлов к обработке
    files_to_process: List[str] = []
    if mode == 'latest':
        latest = _pick_latest_added_file(input_dir, pattern=pattern)
        if latest:
            files_to_process = [latest]
        print(f"Режим: latest. Последний добавленный файл: {os.path.basename(latest) if latest else 'не найден'}")

    elif mode == 'manual':
        if not files:
            print("Режим: manual. Не передан список/путь файлов — нечего обрабатывать.")
            return
        files_to_process = _resolve_manual_files(files, pattern=pattern)
        print(f"Режим: manual. Файлов к обработке: {len(files_to_process)}")
        for p in files_to_process:
            print("  -", os.path.basename(p))

    else:  # 'state' (по умолчанию)
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
        print(f"Режим: state. Новых/изменённых файлов: {len(files_to_process)}")
        for p in files_to_process:
            print("  -", os.path.basename(p))

    # Фильтр: не обрабатывать уже учтённые (актуально для latest/manual)
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
                print(f"Пропуск (уже в реестре): {os.path.basename(f)}")
                continue
            filtered.append(f)
            new_rows.append({'file': f, 'size': size, 'mtime': mtime})
        files_to_process = filtered

    if not files_to_process:
        print('Нет файлов для обработки — выхожу.')
        return

    # Читаем/создаём мастер
    if not master_filename:
        master_filename = 'Объединение чеков по дням_master.csv'
    master_path = os.path.join(out_dir, str(master_filename))
    if reset_master:
        master = pd.DataFrame(columns=['ORG_KSSS', 'year', 'month', 'day', col_name])
    else:
        master = _read_master(master_path, unique=unique)

    # Агрегируем выбранные файлы
    t_agg = time.time()
    parts = []
    for f in files_to_process:
        df = _aggregate_file_by_day(f, unique=unique)
        if not df.empty:
            parts.append(df)

    if not parts:
        print('Выбранные файлы не дали данных — завершаю.')
        # но всё равно отметим их в реестре, чтобы не гонять снова
        if new_rows:
            state_df = pd.concat([state_df, pd.DataFrame(new_rows)], ignore_index=True)
            _save_state(state_path, state_df)
        return

    new_data = pd.concat(parts, ignore_index=True)
    new_data = (new_data
                .groupby(['ORG_KSSS', 'year', 'month', 'day'], as_index=False)[col_name]
                .sum())
    print(f'Агрегация заняла: {time.time() - t_agg:.2f} сек')

    # Обновляем мастер
    if append_only_missing:
        if master.empty:
            to_append = new_data
        else:
            key_cols = ['ORG_KSSS', 'year', 'month', 'day']
            merged = new_data.merge(master[key_cols], on=key_cols, how='left', indicator=True)
            to_append = merged.loc[merged['_merge'] == 'left_only', new_data.columns]
        if to_append.empty:
            print('Все группы уже есть в мастере — дописывать нечего.')
        else:
            master = pd.concat([master, to_append], ignore_index=True)
    else:
        combined = pd.concat([master, new_data], ignore_index=True)
        master = (combined
                  .groupby(['ORG_KSSS', 'year', 'month', 'day'], as_index=False)[col_name]
                  .sum())

    # Сохраняем мастер и снимки
    master = master.sort_values(['ORG_KSSS', 'year', 'month', 'day']).reset_index(drop=True)
    if write_master:
        master_dir = os.path.dirname(master_path)
        if master_dir:
            os.makedirs(master_dir, exist_ok=True)
        master.to_csv(master_path, index=False, sep='|')

    now = dt.datetime.now()
    dated_path  = os.path.join(out_dir, f'Объединение чеков по дням_{now.day}.{now.month}.{now.year} {now.hour}.{now.minute}.csv')
    master.to_csv(dated_path,  index=False, sep='|')

    # Обновляем реестр обработанных
    if new_rows:
        state_df = pd.concat([state_df, pd.DataFrame(new_rows)], ignore_index=True)
        _save_state(state_path, state_df)

    print('\n--- Итог ---')
    print(f'Файлов обработано: {len(files_to_process)}')
    print(f'Итоговый размер мастера: {master.shape}')
    print(f'Готово за {time.time() - t0:.2f} сек')

    for p in glob.glob(os.path.join(out_dir, '*_processed_files.csv')):
        try:
            os.remove(p)
            print(f'Удалил: {p}')
        except Exception as e:
            print(f'Не удалось удалить {p}: {e}')
