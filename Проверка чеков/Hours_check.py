from calendar import month
import pandas as pd
import numpy as np
import datetime as dt

from constants import (CHECKS_PATH, POS_KSO_PATH, CLUSTER_PATH, AUTO_PATH, INTENSIVE_NIGHTS,
                       OUTPUT_PATH, OUTPUT_PATH_CHANGES, MONTH, YEAR)

# --- ЗАГРУЗКА ---
checks_df = pd.read_excel(CHECKS_PATH)
pos_kso_df = pd.read_excel(POS_KSO_PATH)
cluster_df = pd.read_excel(CLUSTER_PATH)
auto_df = pd.read_excel(AUTO_PATH)
nights_azs_df = pd.read_excel(INTENSIVE_NIGHTS)

# --- ПОДГОТОВКА ЧЕКОВ ---
# Переименуем колонки в основном массиве
col_map = {
    "КССС": "KSSS",
    "Номер АЗС": "Station",
    "Дата": "Date",
    "День": "Day_num",
    "Прогноз дневных чеков (дневная смена)": "day_checks",
    "Прогноз ночных чеков (ночная смена)": "night_checks",
    'Совокупная сумма часов для дневной смены': "Чеки ЛИ (день)",
    'Совокупная сумма часов для ночной смены': "Чеки ЛИ (ночь)"
}
checks_df = checks_df[list(col_map.keys())].rename(columns=col_map)

# Переводим в тип данных, необходимый в дальнейшем исследовании
checks_df["KSSS"] = pd.to_numeric(checks_df["KSSS"], errors="coerce").astype("Int64")
if "Day_num" in checks_df.columns:
    checks_df["Day_num"] = pd.to_numeric(checks_df["Day_num"], errors="coerce").astype("Int64")

# --- СЛИЯНИЯ ---
# Добавление ПОС и КСО
df = checks_df.merge(pos_kso_df, left_on="KSSS", right_on="КССС", how="left")
if "КССС" in df.columns:
    df = df.drop(columns=["КССС"])

# Добавление импа АЗС (магазин, магазин-кафе, и др.)
df = df.merge(cluster_df, left_on="KSSS", right_on="КССС_union", how="left")
df = df.merge(auto_df, left_on="KSSS", right_on="КССС", how="left")
if "КССС" in df.columns:
    df = df.drop(columns=["КССС"])

# --- ПРИВЕДЕНИЕ К ЕДИНОМУ ВИДУ ---
df = df.rename(columns={
    "POS": "POS",
    "КСО": "KSO",
    "Кластер_по_сервису": "cluster",
    "Автомат": "automated"
})
df["POS"] = df["POS"].fillna(0).astype(int)
df["KSO"] = df["KSO"].fillna(0).astype(int)
df["cluster"] = df["cluster"].fillna("нет")

# --- ФУНКЦИИ ДЛЯ ПРИЗНАКОВ ---
# Проверка типа АЗС по сервису
def get_type_value(cluster_name: str) -> int:
    if not isinstance(cluster_name, str):
        return 0
    cl = cluster_name.strip().lower().replace("–", "-")
    if cl in ("магазин - кафе", "магазин – кафе", "магазин - фастфуд", "магазин – фастфуд"):
        return 1
    if cl == "магазин":
        return 0
    if cl in ("окно", "нет", "аазс", "автоцисцерна"):
        return 0
    return 0

# Бинарный признак наличия КСО
def kso_having(kso_num) -> int:
    try:
        kso_num = int(kso_num)
    except Exception:
        return 0
    return 1 if kso_num > 0 else 0

df["type_val"] = df["cluster"].apply(get_type_value)
df["kso_avail"] = df["KSO"].apply(kso_having)

# --- МОЩНОСТЬ ПО СМЕНАМ ---
# День: POS + kso_avail + type_val
df["capacity_count_day"] = (
    df["POS"].fillna(0).astype(int)
    + df["kso_avail"].fillna(0).astype(int)
    + df["type_val"].fillna(0).astype(int)
)
# Ночь: только POS
df["capacity_count_night"] = df["POS"].fillna(0).astype(int)

# На случай, если где-то создавалась колонка с хвостом пробела
if "capacity_count_night " in df.columns:
    df = df.rename(columns={"capacity_count_night ": "capacity_count_night"})

# отдельные целевые интенсивности по сменам
TARGET_DAY_INTENSITY = 22.0
TARGET_NIGHT_INTENSITY = 15.0
SHIFT_HOURS = 12.0
SU_TARGET = 18.0  # если нужно сохранять "Отклонение от 18" в выходе

def _choose_staff_for_shift(checks: float, cap: int, target_intensity: float) -> int:
    """
    Подбирает целочисленное число сотрудников в смене (1..cap), чтобы
    |checks / (12 * staff) - target_intensity| было минимальным.
    При равенстве — выбираем меньшее staff.
    Если checks <= 0, возвращает 0.
    """
    if checks <= 0:
        return 0
    cap = max(int(cap or 0), 1)  # если чеки есть, минимум мощность 1
    best_staff = 1
    best_diff = float("inf")
    for s in range(1, cap + 1):
        diff = abs((checks / (SHIFT_HOURS * s)) - target_intensity)
        if diff < best_diff - 1e-12 or (abs(diff - best_diff) <= 1e-12 and s < best_staff):
            best_diff = diff
            best_staff = s
    return best_staff

def optimize_staffing(row):
    # мощности по сменам
    day_cap = int(row.get("capacity_count_day", 0) or 0)
    night_cap = int(row.get("capacity_count_night", 0) or 0)

    # чеки по сменам
    day_checks = row.get("day_checks", 0) or 0
    night_checks = row.get("night_checks", 0) or 0
    total_checks = day_checks + night_checks

    # если вообще нет чеков
    if total_checks <= 0:
        return 0, 0, 0, 0, 0.0, abs(0.0 - SU_TARGET)

    # если есть чеки, но мощность 0 — поднимаем до 1
    if day_checks > 0 and day_cap == 0:
        day_cap = 1
    if night_checks > 0 and night_cap == 0:
        night_cap = 1

    # подбираем штат по сменам под их цели
    day_staff = _choose_staff_for_shift(day_checks, day_cap, TARGET_DAY_INTENSITY) if day_checks > 0 else 0
    night_staff = _choose_staff_for_shift(night_checks, night_cap, TARGET_NIGHT_INTENSITY) if night_checks > 0 else 0

    # часы
    day_hours = int(day_staff * SHIFT_HOURS)
    night_hours = int(night_staff * SHIFT_HOURS)
    total_hours = day_hours + night_hours

    # суточная достигнутая интенсивность и отклонение от 18 (для совместимости с текущими колонками)
    achieved_intensity = (total_checks / total_hours) if total_hours > 0 else 0.0
    deviation = abs(achieved_intensity - SU_TARGET)

    return day_staff, day_hours, night_staff, night_hours, achieved_intensity, deviation

# Применяем к каждой строке
result_df = df.copy()
result_df[[
    "day_staff", "day_hours",
    "night_staff", "night_hours",
    "achieved_intensity", "deviation"
]] = result_df.apply(optimize_staffing, axis=1, result_type="expand")

# --- ПРИСОЕДИНЯЕМ «НОЧНЫЕ» АЗС (по КССС) ---
# Берём только список КССС, дедуплицируем и ставим флаг night_active=1
night_flags = nights_azs_df.rename(columns={"КССС": "KSSS"})[["KSSS"]].copy()
night_flags["KSSS"] = pd.to_numeric(night_flags["KSSS"], errors="coerce").astype("Int64")
night_flags = night_flags.dropna().drop_duplicates()
night_flags["night_active"] = 1

result_df = result_df.merge(night_flags, on="KSSS", how="left")
result_df["night_active"] = result_df["night_active"].fillna(0).astype(int)

# --- ДОППРОВЕРКА ДЛЯ НОЧНЫХ АЗС: СНИЖЕНИЕ НОЧНОГО ШТАТА ДО БЛИЖАЙШЕЙ К 18 ИНТЕНСИВНОСТИ ---
def adjust_night_staff(row):
    night_active = int(row.get("night_active", 0) or 0)
    night_checks = row.get("night_checks", 0) or 0
    night_staff = int(row.get("night_staff", 0) or 0)
    day_staff = int(row.get("day_staff", 0) or 0)
    day_checks = row.get("day_checks", 0) or 0

    # если станция не в списке активных ночью, либо нет ночных чеков/сотрудников — ничего не меняем
    if night_active != 1 or night_checks <= 0 or night_staff <= 0:
        return night_staff, int(night_staff * SHIFT_HOURS), row["achieved_intensity"], row["deviation"]

    # функция ошибки по ночной интенсивности
    def night_diff(s):
        return abs((night_checks / (SHIFT_HOURS * s)) - SU_TARGET)

    # уменьшаем по одному, пока становится лучше; минимум 1 сотрудник при наличии чеков
    s = night_staff
    best_s = s
    best_diff = night_diff(s)
    while s > 1:
        new_diff = night_diff(s - 1)
        if new_diff < best_diff - 1e-12:  # строго лучше
            s -= 1
            best_s = s
            best_diff = new_diff
        else:
            break

    # если изменили — пересчёт суточной интенсивности/отклонения
    if best_s != night_staff:
        new_night_staff = best_s
        new_night_hours = int(new_night_staff * SHIFT_HOURS)
        total_hours = int(day_staff * SHIFT_HOURS + new_night_hours)
        total_checks = (day_checks + night_checks)
        new_achieved = total_checks / total_hours if total_hours > 0 else 0.0
        new_dev = abs(new_achieved - SU_TARGET)
        return new_night_staff, new_night_hours, new_achieved, new_dev
    else:
        # без изменений
        return night_staff, int(night_staff * SHIFT_HOURS), row["achieved_intensity"], row["deviation"]

# применяем корректировку
result_df[["night_staff", "night_hours", "achieved_intensity", "deviation"]] = \
    result_df.apply(adjust_night_staff, axis=1, result_type="expand")

# Убираем полностью автоматизированные объекты (если надо)
if "automated" in result_df.columns:
    mask_auto = (result_df["automated"] == 1) & ((result_df["day_checks"] + result_df["night_checks"]) > 0)
    result_df = result_df[~mask_auto]

# Сортировка
if "Day_num" in result_df.columns:
    result_df = result_df.sort_values(["KSSS", "Day_num"])
else:
    result_df = result_df.sort_values(["KSSS", "Date"])

# --- ФОРМИРОВАНИЕ ВЫХОДНОГО ДАТАСЕТА ---
output_df = result_df.rename(columns={
    "Station": "Номер АЗС",
    "KSSS": "КССС",
    "Date": "Дата",
    "day_staff": "Сотрудники (день)",
    "day_hours": "Часы (день)",
    "night_staff": "Сотрудники (ночь)",
    "night_hours": "Часы (ночь)",
    "achieved_intensity": "Интенсивность труда суточная",
    "deviation": "Отклонение от 18",
    "day_checks": "Дневные чеки",
    "night_checks": "Ночные чеки",
    "capacity_count_day": "Мощность (день)",
    "capacity_count_night": "Мощность (ночь)",

    'Чеки ЛИ (день)': 'Часы ЛИ (день)',
    'Чеки ЛИ (ночь)': 'Часы ЛИ (ночь)'
})

columns_order = [
    "Номер АЗС", "КССС", "Дата",
    "Сотрудники (день)", "Часы (день)", "Дневные чеки",
    "Сотрудники (ночь)", "Часы (ночь)", "Ночные чеки",
    "Интенсивность труда суточная", "Отклонение от 18",
    "Мощность (день)", "Мощность (ночь)",
    "POS", "KSO", "Часы ЛИ (день)", "Часы ЛИ (ночь)"
]
output_columns = [col for col in columns_order if col in output_df.columns]
final_df = output_df[output_columns]

# --- ВЫГРУЗКА В EXCEL С ФОРМАТИРОВАНИЕМ ---
with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    final_df.to_excel(writer, sheet_name="Optimal Staffing", index=False)
    ws = writer.sheets["Optimal Staffing"]

    # ширины колонок по заголовкам
    header_widths = {
        "Номер АЗС": 12,
        "КССС": 8,
        "Дата": 10,
        "Сотрудники (день)": 18,
        "Часы (день)": 12,
        "Дневные чеки": 16,
        "Сотрудники (ночь)": 18,
        "Часы (ночь)": 12,
        "Ночные чеки": 16,
        "Интенсивность труда суточная": 20,
        "Отклонение от 18": 18,
        "Мощность (день)": 16,
        "Мощность (ночь)": 16,
        "POS": 8,
        "KSO": 8,
        "Часы ЛИ (день)": 16,
        "Часы ЛИ (ночь)": 16
    }

    # применяем ширины
    for cell in ws[1]:
        header = cell.value
        width = header_widths.get(header)
        if width:
            ws.column_dimensions[cell.column_letter].width = width

    # формат чисел: по одному знаку после запятой для интенсивности и отклонения
    for cell in ws[1]:
        if cell.value in ("Интенсивность труда суточная", "Отклонение от 18"):
            col_letter = cell.column_letter
            for data_cell in ws[col_letter][1:]:
                data_cell.number_format = "0.0"

# --- ФОРМИРОВАНИЕ ФАЙЛА С ДАННЫМИ ДЛЯ ЗАМЕНЫ ---
final_df = output_df[output_columns].copy()

# Добавляем колонки соответствия часов ЛИКАРДА - часам в алгоритме
final_df.loc[:, "Соответствие дневных часов"] = np.where(
    final_df['Часы ЛИ (день)'] == final_df['Часы (день)'], "ИСТИНА", "ЛОЖЬ"
)
final_df.loc[:, "Соответствие ночных часов"] = np.where(
    final_df['Часы ЛИ (ночь)'] == final_df['Часы (ночь)'], "ИСТИНА", "ЛОЖЬ"
)


# фильтр для создания массива где нужно внести изменения
df_massive_to_change = final_df[
    (final_df['Соответствие дневных часов'] == 'ЛОЖЬ') |
    ((final_df['Соответствие дневных часов'] == 'ИСТИНА') &
     (final_df['Соответствие ночных часов'] == 'ЛОЖЬ'))
]

# подготовка и новая дата
df_changes = df_massive_to_change[['КССС', 'Дата', 'Часы (день)', 'Часы (ночь)']].copy()

# Создание даты из сцепки НомДня_ДенНед
df_changes.loc[:, 'Дата_новая'] = pd.to_datetime(
    df_changes['Дата'].str.extract(r'(\d+)')[0].str.zfill(2) + f'.{MONTH:02d}.{YEAR}',
    format='%d.%m.%Y'
)

# если нужен текст "dd.mm.yyyy" в Excel:
df_changes.loc[:, 'ДАТА'] = df_changes['Дата_новая'].dt.strftime('%d.%m.%Y')
# и можно выбрасывать 'Дата_новая'

# порядок колонок и переименования
df_changes = df_changes[['Дата_новая', 'КССС', 'Часы (день)', 'Часы (ночь)']].rename(
    columns={
        'Дата_новая': "ДАТА",
        'Часы (день)': "день",
        'Часы (ночь)': "ночь"
    }
)

# Удаление дубликатов
df_changes = df_changes.drop_duplicates(subset=['КССС', 'ДАТА'])

# Сохранение по пути файла с данными
with pd.ExcelWriter(OUTPUT_PATH_CHANGES, engine="openpyxl") as writer:
    df_changes.to_excel(writer, sheet_name="Изменение", index=False)