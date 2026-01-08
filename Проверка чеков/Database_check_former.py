import runpy
import time
from pathlib import Path

from Check_check_functions import prediction_analys
from constants import PATH_TO_SAVED
from pathes_to_change import PATH_TO_CURRENT_PREDICTION_TABLE, FILE_PREFICS

import warnings
warnings.filterwarnings('ignore')

# ----- ТРАНСПОНИРУЕМ И ПРОВЕРЯЕМ ИСХОДНИК И ЕГО СОХРАНЯЕМ В ФАЙЛ ------
prediction_analys(file_name=PATH_TO_CURRENT_PREDICTION_TABLE,
                  path_to_save=PATH_TO_SAVED,
                  new_name=FILE_PREFICS)

# ----- ЗАПУСКАЕМ ФАЙЛ С АНАЛИЗОМ ИНФОРМАЦИЕЙ ИЗ HOURS_CHECK.PY ------
print(f'\nРассчитываем часы и создаем файл-шаблон для замены в итоговых рассылках')
start_time = time.time()
runpy.run_path(str(Path(__file__).resolve().parent / "Hours_check.py"), run_name="__main__")
print(f'Выполнели за {round(time.time()-start_time, 2)} сек.')