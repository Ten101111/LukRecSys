import pandas as pd
import time
from collections import defaultdict
import os
import glob
import datetime as dt

import warnings
warnings.filterwarnings('ignore')

def check_by_hour_massive(all_files):
    '''
    Обрабатывает выгрузку чеков и создает массив с кол-вом чеков в каждый час
    :param all_files: файлы с выгрузками ЧЕКИ ТРК
    :return: сохраняет объединенные чеки
    '''
    time_all = time.time()
    dtype_dict = {'CHEQUE_ID': 'str', 'ORG_KSSS': 'str', 'OPERATION_DATE': 'object'}  # object для смешанных типов
    # Словарь для накопления результатов
    results_dict = defaultdict(int)
    type_counts = defaultdict(int)  # Для диагностики типов данных
    # Проход по всем файлам
    for file in all_files:
        start_time = time.time()
        filename = os.path.basename(file)
        print(f'Обработка файла: {filename}')
        # Чтение файла с сохранением исходных типов
        for chunk_idx, chunk in enumerate(pd.read_csv(
                file,
                sep='|',
                # nrows = 100,
                dtype=dtype_dict,
                chunksize=100000
        )):
            # Диагностика типов данных
            for val in chunk['OPERATION_DATE']:
                type_counts[type(val).__name__] += 1
            # Унифицированное преобразование в datetime
            chunk['OPERATION_DATE'] = pd.to_datetime(
                chunk['OPERATION_DATE'],
                errors='coerce',
                infer_datetime_format=True,
                dayfirst=True
            )
            # Удаление некорректных дат
            invalid_mask = chunk['OPERATION_DATE'].isna()
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                print(f"Чанк {chunk_idx}: Найдено {invalid_count} некорректных дат")
                chunk = chunk[~invalid_mask]
            # Извлечение компонентов даты
            chunk['year'] = chunk['OPERATION_DATE'].dt.year
            chunk['month'] = chunk['OPERATION_DATE'].dt.month
            chunk['day'] = chunk['OPERATION_DATE'].dt.day
            chunk['hour'] = chunk['OPERATION_DATE'].dt.hour
            # Группировка и агрегация
            grouped = chunk.groupby([
                'ORG_KSSS', 'year', 'month', 'day', 'hour'
            ])['CHEQUE_ID'].nunique()
            # Накопление результатов
            for key, value in grouped.items():
                results_dict[key] += value
        print(f"Файл обработан за {time.time() - start_time:.2f} сек")
    # Вывод диагностики типов
    print("\nДиагностика типов данных в OPERATION_DATE:")
    for dtype, count in type_counts.items():
        print(f"- {dtype}: {count} значений")

    # Преобразование словаря в DataFrame
    final_result = pd.DataFrame(
        [(k[0], k[1], k[2], k[3], k[4], v) for k, v in results_dict.items()],
        columns=['ORG_KSSS', 'year', 'month', 'day', 'hour', 'unique_cheques']
    )
    data_day = dt.datetime.now()
    # Сохранение результата
    final_result.to_csv(
        f'D:\Изменение предиктивки\Обработка чеков\Объединение чеков\Объединение чеков_{data_day.day}.{data_day.month}.{data_day.year}.csv',
                        index=False, sep='|')
    final_result.to_csv(
        f'D:\Изменение предиктивки\Обработка чеков\Объединение чеков {data_day.day}.csv',
        index=False, sep='|')
    # Статистика
    print(f"\nИтоговая статистика:")
    print(f"Обработано файлов: {len(all_files)}")
    print(f"Уникальных комбинаций дат: {len(results_dict)}")
    print(f"Итоговый размер данных: {final_result.shape}")
    print()
    print(f'Функция отработала за {time.time() - time_all} сек.')


def check_by_day_massive(
    all_files,
    unique: bool = True,
    out_dir: str = r'D:\Изменение предиктивки\Обработка чеков\Объединение чеков'
):
    """
    Агрегация по дням: ORG_KSSS, year, month, day.
    По умолчанию считает количество УНИКАЛЬНЫХ чеков (CHEQUE_ID) в группе.
    Если unique=False — считает все строки (все чеки), эквивалентно groupby.size().

    :param all_files: список путей к csv-файлам
    :param unique: True -> уникальные CHEQUE_ID; False -> все строки
    :param out_dir: каталог для сохранения итоговых csv
    """
    time_all = time.time()
    os.makedirs(out_dir, exist_ok=True)

    dtype_dict = {
        'CHEQUE_ID': 'str',
        'ORG_KSSS': 'str',
        'OPERATION_DATE': 'object'  # object для смешанных типов
    }

    # Накопители
    results_dict = defaultdict(int)
    type_counts = defaultdict(int)  # диагностика типов OPERATION_DATE

    for file in all_files:
        start_time = time.time()
        filename = os.path.basename(file)
        print(f'Обработка файла: {filename}')

        for chunk_idx, chunk in enumerate(pd.read_csv(
            file,
            sep='|',
            dtype=dtype_dict,
            usecols=['CHEQUE_ID', 'ORG_KSSS', 'OPERATION_DATE'],
            chunksize=100_000
        )):
            # Диагностика типов
            for val in chunk['OPERATION_DATE']:
                type_counts[type(val).__name__] += 1

            # Преобразование в datetime
            chunk['OPERATION_DATE'] = pd.to_datetime(
                chunk['OPERATION_DATE'],
                errors='coerce',
                infer_datetime_format=True,
                dayfirst=True
            )

            # Отфильтровать некорректные даты
            invalid_mask = chunk['OPERATION_DATE'].isna()
            if invalid_mask.any():
                invalid_count = int(invalid_mask.sum())
                print(f"Чанк {chunk_idx}: Найдено {invalid_count} некорректных дат")
                chunk = chunk[~invalid_mask]

            if chunk.empty:
                continue

            # Компоненты даты
            chunk['year'] = chunk['OPERATION_DATE'].dt.year
            chunk['month'] = chunk['OPERATION_DATE'].dt.month
            chunk['day'] = chunk['OPERATION_DATE'].dt.day

            # Группировка и агрегация по дням
            group_cols = ['ORG_KSSS', 'year', 'month', 'day']
            if unique:
                grouped = chunk.groupby(group_cols)['CHEQUE_ID'].nunique()  # уникальные чеки
            else:
                grouped = chunk.groupby(group_cols).size()  # все строки/чеки

            # Накопление результатов через словарь
            for key, value in grouped.items():
                results_dict[key] += int(value)

            # В формате даты ДД.ММ.ГГГГ
            chunk['date_str'] = (
                    chunk['day'].astype(str).str.zfill(2) + '.' +
                    chunk['month'].astype(str).str.zfill(2) + '.' +
                    chunk['year'].astype(str)
            )
        print(f"Файл обработан за {time.time() - start_time:.2f} сек")

    # Диагностика типов
    print("\nДиагностика типов данных в OPERATION_DATE:")
    for dtype, count in type_counts.items():
        print(f"- {dtype}: {count} значений")

    # Преобразование словаря в DataFrame
    col_name = 'unique_cheques' if unique else 'cheques_count'
    final_result = pd.DataFrame(
        [(k[0], k[1], k[2], k[3], v) for k, v in results_dict.items()],
        columns=['ORG_KSSS', 'year', 'month', 'day', col_name]
    ).sort_values(['ORG_KSSS', 'year', 'month', 'day'])

    # Сохранение результата
    now = dt.datetime.now()
    dated_path = os.path.join(
        out_dir,
        f'Объединение чеков по дням_{now.day}.{now.month}.{now.year} {now.hour}.{now.minute}.csv'
    )

    final_result.to_csv(dated_path, index=False, sep='|')

    # Статистика
    print("\nИтоговая статистика:")
    print(f"Обработано файлов: {len(all_files)}")
    print(f"Уникальных комбинаций (ORG_KSSS, year, month, day): {len(results_dict)}")
    print(f"Итоговый размер данных: {final_result.shape}")
    print(f'Функция отработала за {time.time() - time_all:.2f} сек')

from constants import PATH_CHECKS_FROM_TRK, PATH_CHECK_SAVE

all_files = glob.glob(PATH_CHECKS_FROM_TRK)

check_by_day_massive(
    all_files=all_files,
    unique=True,
    out_dir=PATH_CHECK_SAVE
)