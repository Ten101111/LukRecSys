from collections import defaultdict
import glob
import os
import time
import datetime as dt

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def check_by_day_night(all_files, out_dir):
    time_all = time.time()
    dtype_dict = {
        'CHEQUE_ID': 'str',
        'ORG_KSSS': 'str',
        'OPERATION_DATE': 'object'  # object для смешанных типов
    }

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
            # nrows=100,
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

            # Признак интервала времени:
            # True (или '08_20') для 08:00–19:59, False (или 'other') для остального
            hours = chunk['OPERATION_DATE'].dt.hour
            chunk['time_segment'] = np.where(
                hours.between(8, 19),  # 8..19 включительно = 08:00–19:59
                '08_20',
                'other'
            )

            # Группировка и агрегация:
            # теперь вместо hour используем time_segment
            grouped = chunk.groupby(
                ['ORG_KSSS', 'year', 'month', 'day', 'time_segment']
            )['CHEQUE_ID'].nunique()

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
        [
            (k[0], k[1], k[2], k[3], k[4], v)
            for k, v in results_dict.items()
        ],
        columns=[
            'ORG_KSSS',
            'year',
            'month',
            'day',
            'time_segment',      # '08_20' или 'other'
            'unique_cheques'
        ]
    )

    data_day = dt.datetime.now()

    # Сохранение результата
    os.makedirs(out_dir, exist_ok=True)
    dated_path = os.path.join(
        out_dir,
        f'День и Ночь от {data_day.day}.{data_day.month}.{data_day.year} {data_day.hour}.{data_day.minute}.csv'
    )
    final_result.to_csv(dated_path, index=False, sep='|')

    # Статистика
    print(f"\nИтоговая статистика:")
    print(f"Обработано файлов: {len(all_files)}")
    print(f"Уникальных комбинаций дат/интервалов: {len(results_dict)}")
    print(f"Итоговый размер данных: {final_result.shape}")
    print()
    print(f'Функция отработала за {time.time() - time_all} сек.')

from constants import PATH_CHECKS_FROM_TRK, PATH_CHECK_DAY_NIGHT_SAVE

all_files = glob.glob(PATH_CHECKS_FROM_TRK)

check_by_day_night(
    all_files=all_files,
    out_dir=PATH_CHECK_DAY_NIGHT_SAVE
)
