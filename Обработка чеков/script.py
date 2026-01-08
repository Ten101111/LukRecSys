from constants import (CHECKS_CONTAINER_FOLDER, PATH_TO_RESULT_FOLDER,
                       MASTER_FILE_PATH, OBRABOTKA_CHEKOV, PATH_CHECK_DAY_NIGHT_SAVE,
                       MASTER_FILE_PATH_DN)
from new_checks_add import update_by_day
from day_night_checks_function import update_by_day_night

import warnings
warnings.filterwarnings('ignore')

# можно убрать, если не manual
FILE_FOLDER = f'{OBRABOTKA_CHEKOV}\Чеки'
SP_TO_APPEND = [
    f'{FILE_FOLDER}\Чеки с ТРК Декабрь 2025.csv',
    f'{FILE_FOLDER}\Чеки с ТРК Ноябрь 2025.csv',
    f'{FILE_FOLDER}\Чеки с ТРК Сентябрь 2025.csv',
    f'{FILE_FOLDER}\Чеки с ТРК Октябрь 2025.csv',
]

# тип добавления в датасет новых данных
'''
Режимы:
    - mode='state'  : берём все новые/изменённые файлы (по size+mtime).
    - mode='latest' : берём ТОЛЬКО один файл — последний добавленный в папку (по ctime).
    - mode='manual' : берём РОВНО те файлы, что переданы через параметр `files`.
'''
MODE = 'manual'

# Добавляем чеки в основной массив
update_by_day(
    input_dir=CHECKS_CONTAINER_FOLDER,
    out_dir=PATH_TO_RESULT_FOLDER,
    master_filename=MASTER_FILE_PATH,
    unique=True,
    append_only_missing=True,  # дописывать только отсутствующие группы
    mode=MODE,
    files=SP_TO_APPEND,
    skip_if_in_state=True
)

# Добавляем чеки ночные и дневные
update_by_day_night(
    input_dir = CHECKS_CONTAINER_FOLDER,
    out_dir = PATH_CHECK_DAY_NIGHT_SAVE,
    master_filename = MASTER_FILE_PATH_DN,
    unique = True,
    append_only_missing = True,
    mode = MODE,
    files = SP_TO_APPEND,
    pattern = '*.csv',
    skip_if_in_state = True
)