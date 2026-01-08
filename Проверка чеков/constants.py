import datetime as dt
from pathlib import Path
from pathes_to_change import CHECKS_PATH

INITIAL_PATH = Path(__file__).resolve().parent

# ------ ДЛЯ ПУТЕЙ ФАЙЛОВ К ДАННЫМ В ФАЙЛЕ Расчет чеков.PY ------
POS_KSO_PATH = str(INITIAL_PATH / "Исходники" / "POS&KSO.xlsx")
CLUSTER_PATH = str(INITIAL_PATH / "Исходники" / "Тип объекта.xlsx")
AUTO_PATH = str(INITIAL_PATH / "Исходники" / "Автоматы.xlsx")
INTENSIVE_NIGHTS = str(INITIAL_PATH / "Исходники" / "АЗС активные ночью.xlsx")

MONTHS_NUMS = {
        "январь": 1,
        "февраль": 2,
        "март": 3,
        "апрель": 4,
        "май": 5,
        "июнь": 6,
        "июль": 7,
        "август": 8,
        "сентябрь": 9,
        "октябрь": 10,
        "ноябрь": 11,
        "декабрь": 12,
    }

# --- ОЦЕНКА ДАТЫ ПРОГНОЗА И ТЕКУЩЕЙ ---
mnth = Path(CHECKS_PATH).name.split("_")[0].lower()
MONTH = MONTHS_NUMS[mnth]
YEAR = dt.datetime.now().year
if dt.datetime.now().month > MONTH:
    YEAR += 1


# НЕ МЕНЯТЬ ПУТИ
# НЕ МЕНЯТЬ ПУТЬ - дорога к сохраняемой папке
PATH_TO_SAVED = str(
    INITIAL_PATH / "Результаты анализа чеков-часов" / "Преобразованные для анализа"
)

# --- ПУТИ К ФАЙЛАМ ВЫГРУЗКИ ФАЙЛА С ЧАСАМИ ---
OUTPUT_PATH = str(
    INITIAL_PATH
    / "Результаты анализа чеков-часов"
    / "Полный анализ"
    / f"{mnth.title()}.xlsx"
)

# --- ПУТИ К ФАЙЛАМ ВЫГРУЗКИ ФАЙЛА С ОТЛИЧИЯМИ ---
OUTPUT_PATH_CHANGES = str(
    INITIAL_PATH
    / "Результаты анализа чеков-часов"
    / "Шаблоны для замены"
    / f"Шаблон с заменами ({mnth.title()}).xlsx"
)
