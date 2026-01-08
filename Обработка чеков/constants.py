# ----- ДЛЯ ФАЙЛА, КОТОРЫЙ СОБИРАЕТ СРАЗУ ВСЕ И ОЧЕНЬ ДОЛГО ---
# Получаем данные по дате
import datetime as dt
from pathlib import Path

year = dt.datetime.now().year
month = dt.datetime.now().month
day = dt.datetime.now().day

# Директория для чеков в проекте
OBRABOTKA_CHEKOV = Path(__file__).resolve().parent

# Дорога до файла, где лежат все чеки - файлы со всеми чеками
PATH_CHECKS_FROM_TRK = rf'{OBRABOTKA_CHEKOV}\Чеки\*.csv'

# Путь до папки, где будет лежать обнавленный массив
PATH_CHECK_SAVE = rf'{OBRABOTKA_CHEKOV}\Объединение чеков\Объединенные чеки от {day}.{month}.{year}'


# ----- ДЛЯ ПЕРЕДАЧИ В КОД ДЛЯ ОБНОВЛЕНИЯ ЧЕКОВ -----
# Папка с чеками всеми
CHECKS_CONTAINER_FOLDER = rf'{OBRABOTKA_CHEKOV}\Чеки'
# Папка для сохранения результата прогнонки чеков
PATH_TO_RESULT_FOLDER = rf'{OBRABOTKA_CHEKOV}\Объединение чеков\Объединенные чеки от {day}.{month}.{year}'

# Файл, в который мы добавляем результат новых файлов
from constants_functions import pick_latest_file, pick_latest_folder, pretty_display_name

root = Path('D:\Изменение предиктивки\Обработка чеков\Объединение чеков')
LATEST_CHECKS_FOLDER = pick_latest_folder(root)
MASTER_FILE_PATH = pick_latest_file(LATEST_CHECKS_FOLDER)

# Файл с чеками в день и ночь
PATH_CHECK_DAY_NIGHT_SAVE = rf'{OBRABOTKA_CHEKOV}\Доля чеков дневных и ночных\Абсолютные значения от {day}.{month}.{year}'

root_DN = rf'{OBRABOTKA_CHEKOV}\Доля чеков дневных и ночных'
LATEST_CHECKS_FOLDER_DN = pick_latest_folder(root_DN)
MASTER_FILE_PATH_DN = pick_latest_file(LATEST_CHECKS_FOLDER_DN)
