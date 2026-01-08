import datetime as dt
from pathlib import Path

ROOT_PRJ = Path(__file__).resolve().parent.parent
ROOT_CURRENT_FOLDER = Path(__file__).resolve().parent

PATH_TO_SENDINGS_FILES = str(
    ROOT_PRJ / "Проверка чеков" / "Анализ рассылок" / "Рассылки по месяцам - все"
) + "/"

now = dt.datetime.now()
PATH_ALREADY_SEND_MODIF = str(
    ROOT_CURRENT_FOLDER / "Чеки из прошлых рассылок" / f"Прошлые рассылки на {now.day}.{now.month}.{now.year}"
)

from constants_functions import pick_latest_folder, pick_latest_file, pretty_display_name

root = ROOT_PRJ / "Обработка чеков" / "Объединение чеков"

# Крайняя созданная папка с чеками
LATEST_CHECKS_FOLDER = pick_latest_folder(root)
SRC_PATH = pick_latest_file(LATEST_CHECKS_FOLDER)

root_licard_checks = ROOT_CURRENT_FOLDER / "Чеки из прошлых рассылок"

# Крайняя созданная папка с чеками
LATEST_CHECKS_FOLDER_LICARD = pick_latest_folder(root_licard_checks)
CHECKS_LICARD = pick_latest_file(LATEST_CHECKS_FOLDER_LICARD)

PATH_TO_CURRENT_DATA_COMP = str(
    ROOT_CURRENT_FOLDER / "Отклонения" / f"Отклонение на {now.day}.{now.month}.{now.year}"
)
