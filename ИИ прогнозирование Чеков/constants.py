import datetime as dt
from os.path import split
from pathlib import Path

import pandas as pd

# ----- ОБЩИЙ ПУТЬ -----
# Общий путь
COMMON_PATH = Path(__file__).resolve().parent
PRJ_PATH = COMMON_PATH.parent

# ----- ОБЩИЙ ПУТЬ ДО ФАЙЛОВ С ЧЕКАМИ (КРАЙНИЙ ОБНАВЛЕННЫЙ) -----
# Путь до чеков всех - путь до чеков, которые были получены крайними

from constant_functions import pick_latest_folder, pick_latest_file, pretty_display_name, last_date_indicator

root = PRJ_PATH / "Обработка чеков" / "Объединение чеков"

# Крайняя созданная папка с чеками
LATEST_CHECKS_FOLDER = pick_latest_folder(root)
SRC_PATH = pick_latest_file(LATEST_CHECKS_FOLDER)


# ----- TRIPLE_WEIGHT_SEARCH.PY -----
# Сегодняшний день
now = dt.datetime.now()
# Папка для сохранения файла с весами
PATH_NEW_FOLDER = COMMON_PATH / "Выгрузки" / "Константы" / "Веса" / f"Веса {now.day}.{now.month}.{now.year}"

# Путь до файла с весами
OUT_CSV = PATH_NEW_FOLDER / f"Веса для модели по АЗС на {now.day}.{now.month}.{now.year} {now.hour}.{now.minute}.csv"

# Доступный период взвешивания (можно расширить или изменить, но старт не может быть меньше)
# MONTH_RANGE_START = "2024-08-01"
MONTH_RANGE_END = last_date_indicator(SRC_PATH)


# ----- FUNCTIONS_ML.PY -----
# Название колонки, в которой хранится значение КССС
NAME_OF_COLUMN_OF_KSSS = 'ORG_KSSS'
NAME_OF_COLUMN_OF_DATE = 'date'
NAME_OF_COLUMN_OF_SEGMENT = 'time_segment'
# Горизонт прогноза
HORIZON = 67
# Цикличность в рамках прогнозирования
SEASONAL_PERIOD = 7
# Лаг рассылки
DAYS_UNTIL_SENDINGS = 3


# ----- MODELING.PY -----
OUTPUT_PATH_DIR = COMMON_PATH / "Выгрузки" / "Константы" / "Модели" / f"Модели на {now.day}.{now.month}.{now.year}"


# ----- ВЫБОР МОДЕЛИ.PY -----
PATH_TO_VIGRUZKI_CONSTANTS = COMMON_PATH / "Выгрузки" / "Константы"
PATH_WEIGHTS = PATH_TO_VIGRUZKI_CONSTANTS / "Веса"
PATH_MODELS = PATH_TO_VIGRUZKI_CONSTANTS / "Модели"

# Последний файл с Весами
LATEST_WEIGHT_FOLDER = pick_latest_folder(PATH_WEIGHTS)
path_init_w = pick_latest_file(LATEST_WEIGHT_FOLDER)
new_name_w = pretty_display_name(path_init_w)
LAST_FILE_FROM_PATH_WEIGHTS = path_init_w.with_name(new_name_w)

# Последний файл с Моделями
LATEST_MODEL_FOLDER = pick_latest_folder(PATH_MODELS)
path_init_m = pick_latest_file(LATEST_MODEL_FOLDER)
new_name_m = pretty_display_name(path_init_m)
LAST_FILE_FROM_PATH_MODELS = path_init_m.with_name(new_name_m)

# Последний файл с результатом сравнения моделей
OUTPUT_MODELS_COMPARISON = PATH_TO_VIGRUZKI_CONSTANTS / "Оптимальная модель" / f"Результат сравнения моделей {now.day}.{now.month}.{now.year}"


# ----- Дневные и ночные данные -----
# Файл с днями и ночами
FOLDER_TO_DAY_NIGHT = PRJ_PATH / "Обработка чеков" / "Доля чеков дневных и ночных"
folder_last = pick_latest_folder(FOLDER_TO_DAY_NIGHT)
PATH_TO_DAY_NIGHT_FILE = pick_latest_file(folder_last)


# ----- Путь для сохранения текущего прогноза с учетом всех моделей прогноза -----
PATH_TO_SAVE_CURRENT_FORCAST = PRJ_PATH / "Выгрузки" / "Прогноз текущий" / f"Прогноз от {now.day}.{now.month}.{now.year}"

root_cur = PRJ_PATH / "Сравнение моделей" / "Отклонения"
LATEST_CURRENT_FOLDER = pick_latest_folder(root_cur)
LAST_FILE_FROM_PATH_CURRENT = pick_latest_file(LATEST_CURRENT_FOLDER)

root_best_model = COMMON_PATH / "Выгрузки" / "Константы" / "Оптимальная модель"
LATEST_MODEL_OPTION_FOLDER = pick_latest_folder(root_best_model)
LATEST_FILE_OF_BEST_MODEL = pick_latest_file(LATEST_MODEL_OPTION_FOLDER)

RESULT_PREDICTIONS = COMMON_PATH / "Выгрузки" / "Итоговые чеки" / f"Прогноз от {now.day}.{now.month}.{now.year}"


# ----- ДРУГИЕ КОНСТАНТЫ -----
WEEKDAY = {
    1: "ПН",
    2: "ВТ",
    3: "СР",
    4: "ЧТ",
    5: "ПТ",
    6: "СБ",
    7: "ВС"
}

MONTH_NAMES = {
    1: "Январь",
    2: "Февраль",
    3: "Март",
    4: "Апрель",
    5: "Май",
    6: "Июнь",
    7: "Июль",
    8: "Август",
    9: "Сентябрь",
    10: "Октябрь",
    11: "Ноябрь",
    12: "Декабрь"
}

# Файл с АЗС для прогноза
AZS_PRED_FILE = PRJ_PATH / "АЗС для прогноза.xlsx"
