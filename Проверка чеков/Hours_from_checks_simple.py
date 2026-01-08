"""
Hours_from_checks_simple.py

Модуль для расчёта часов (день/ночь) по прогнозу чеков из CSV/XLSX
и формирования отдельного файла-свода сумм часов.

Вход:
  CSV/XLSX с обязательными колонками (или их синонимами):
    - ORG_KSSS (или КССС/KSSS)
    - date (или Дата)
    - Прогноз (или чеки итого/совокупные и т.п.)  [может быть, но не обязателен для расчёта]
    - Чеки дневные
    - Чеки ночные

Выход (детальный файл):
  Исходные колонки + добавляем:
    - "кол-во часов днем"
    - "кол-во часов ночью"
    - (опционально) "кол-во часов всего" = день + ночь

Выход (сводный файл):
  Отдельный файл с:
    - totals (итого по всем строкам)
    - by_ksss (суммы по КССС)

Алгоритм соответствует принципу Hours_check.py:
  - мощность день = POS + (КСО>0) + (тип=магазин-кафе/фастфуд)
  - мощность ночь = POS
  - подбор целочисленного числа сотрудников (1..мощность)
    под целевую интенсивность (день=22, ночь=15), смена=12 часов
  - для АЗС из списка "активные ночью" пробуем уменьшать ночной штат,
    чтобы ночная интенсивность стала ближе к 18 (SU_TARGET)

Функции для платформы:
  - calculate_hours_and_summary(...) -> (detail_path, summary_path)
  - simple_calculate_hours(...) -> detail_path  (backward-compatible)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Пытаемся подтянуть пути из constants.py проекта (если модуль рядом).
# ---------------------------------------------------------------------
try:
    import constants  # type: ignore
except Exception:  # pragma: no cover
    constants = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent


def _get_const(name: str, fallback: str) -> str:
    if constants is None:
        return fallback
    value = getattr(constants, name, None)
    if value:
        try:
            if Path(value).exists():
                return value
        except Exception:
            pass
    return fallback


DEFAULT_POS_KSO_PATH = _get_const(
    "POS_KSO_PATH",
    str(BASE_DIR / "Исходники" / "POS&KSO.xlsx"),
)
DEFAULT_CLUSTER_PATH = _get_const(
    "CLUSTER_PATH",
    str(BASE_DIR / "Исходники" / "Тип объекта.xlsx"),
)
DEFAULT_AUTO_PATH = _get_const(
    "AUTO_PATH",
    str(BASE_DIR / "Исходники" / "Автоматы.xlsx"),
)
DEFAULT_INTENSIVE_NIGHTS_PATH = _get_const(
    "INTENSIVE_NIGHTS",
    str(BASE_DIR / "Исходники" / "АЗС активные ночью.xlsx"),
)


# ---------------------------------------------------------------------
# Параметры алгоритма (как в Hours_check.py)
# ---------------------------------------------------------------------
TARGET_DAY_INTENSITY = 22.0
TARGET_NIGHT_INTENSITY = 15.0
SHIFT_HOURS = 12.0
SU_TARGET = 18.0


# Имена выходных колонок (для совместимости с платформой)
OUT_DAY_COL = "кол-во часов днем"
OUT_NIGHT_COL = "кол-во часов ночью"
OUT_TOTAL_COL = "кол-во часов всего"


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()


def _sniff_csv_delimiter(path: Path) -> str:
    """Определяем разделитель CSV. Если не вышло — ';'."""
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ";"


def _read_input(path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path), None
    if suf == ".csv":
        sep = _sniff_csv_delimiter(path)
        df = pd.read_csv(path, sep=sep, engine="python")
        return df, sep
    raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}. Ожидаю .csv или .xlsx")


def _detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Находит нужные колонки во входном df и возвращает mapping internal->original.
    total_checks необязателен (если есть — читаем, если нет — считаем как day+night).
    """
    cols = {_norm(c): c for c in df.columns}

    def pick(*variants: str) -> Optional[str]:
        for v in variants:
            key = _norm(v)
            if key in cols:
                return cols[key]
        return None

    ksss = pick("ORG_KSSS", "КССС", "KSSS", "org_ksss", "ksss")
    date = pick("date", "Дата", "дата")
    day = pick("Чеки дневные", "чеки дневные", "Дневные чеки", "чеки день", "day_checks")
    night = pick("Чеки ночные", "чеки ночные", "Ночные чеки", "чеки ночь", "night_checks")

    # total опционален
    total = pick("Прогноз", "чеки совокупные", "чеки итого", "чеки всего", "total_checks")

    missing = [
        name
        for name, col in (
            ("ORG_KSSS/КССС", ksss),
            ("date/Дата", date),
            ("Чеки дневные", day),
            ("Чеки ночные", night),
        )
        if col is None
    ]
    if missing:
        raise ValueError("Во входном файле не найдены обязательные колонки: " + ", ".join(missing))

    return {
        "KSSS": ksss,
        "Date": date,
        "total_checks": total,  # может быть None
        "day_checks": day,
        "night_checks": night,
    }


def _require_file(path: str | Path, name: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Не найден файл справочника {name} по пути:\n{p}\n"
            f"Проверьте constants.py или передайте путь параметром."
        )
    return p


# ---------------------------------------------------------------------
# Feature helpers (как в Hours_check.py)
# ---------------------------------------------------------------------
def _get_type_value(cluster_name: str) -> int:
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


def _kso_having(kso_num) -> int:
    try:
        kso_num = int(kso_num)
    except Exception:
        return 0
    return 1 if kso_num > 0 else 0


def _choose_staff_for_shift(checks: float, cap: int, target_intensity: float) -> int:
    """
    Подбор целого staff (1..cap) с минимальным |checks/(12*staff) - target|.
    Если checks <= 0 -> 0.
    """
    if checks <= 0:
        return 0
    cap = max(int(cap or 0), 1)
    best_staff = 1
    best_diff = float("inf")
    for s in range(1, cap + 1):
        diff = abs((checks / (SHIFT_HOURS * s)) - target_intensity)
        if diff < best_diff - 1e-12 or (abs(diff - best_diff) <= 1e-12 and s < best_staff):
            best_diff = diff
            best_staff = s
    return best_staff


def _optimize_staffing(row: pd.Series) -> Tuple[int, int, int, int, float, float]:
    """
    Возвращает:
      day_staff, day_hours, night_staff, night_hours, achieved_intensity, deviation
    """
    day_cap = int(row.get("capacity_count_day", 0) or 0)
    night_cap = int(row.get("capacity_count_night", 0) or 0)

    day_checks = float(row.get("day_checks", 0) or 0)
    night_checks = float(row.get("night_checks", 0) or 0)
    total_checks = day_checks + night_checks

    if total_checks <= 0:
        return 0, 0, 0, 0, 0.0, abs(0.0 - SU_TARGET)

    if day_checks > 0 and day_cap == 0:
        day_cap = 1
    if night_checks > 0 and night_cap == 0:
        night_cap = 1

    day_staff = _choose_staff_for_shift(day_checks, day_cap, TARGET_DAY_INTENSITY) if day_checks > 0 else 0
    night_staff = _choose_staff_for_shift(night_checks, night_cap, TARGET_NIGHT_INTENSITY) if night_checks > 0 else 0

    day_hours = int(day_staff * SHIFT_HOURS)
    night_hours = int(night_staff * SHIFT_HOURS)
    total_hours = day_hours + night_hours

    achieved_intensity = (total_checks / total_hours) if total_hours > 0 else 0.0
    deviation = abs(achieved_intensity - SU_TARGET)

    return day_staff, day_hours, night_staff, night_hours, achieved_intensity, deviation


def _adjust_night_staff(row: pd.Series) -> Tuple[int, int, float, float]:
    """
    Для night_active=1 пробуем уменьшать night_staff (минимум до 1 при наличии чеков),
    пока ночная интенсивность ближе к SU_TARGET (18).
    """
    night_active = int(row.get("night_active", 0) or 0)
    night_checks = float(row.get("night_checks", 0) or 0)
    night_staff = int(row.get("night_staff", 0) or 0)

    day_staff = int(row.get("day_staff", 0) or 0)
    day_checks = float(row.get("day_checks", 0) or 0)

    if night_active != 1 or night_checks <= 0 or night_staff <= 0:
        return (
            night_staff,
            int(night_staff * SHIFT_HOURS),
            float(row.get("achieved_intensity", 0.0) or 0.0),
            float(row.get("deviation", 0.0) or 0.0),
        )

    def night_diff(s: int) -> float:
        return abs((night_checks / (SHIFT_HOURS * s)) - SU_TARGET)

    s = night_staff
    best_s = s
    best_diff = night_diff(s)

    while s > 1:
        new_diff = night_diff(s - 1)
        if new_diff < best_diff - 1e-12:
            s -= 1
            best_s = s
            best_diff = new_diff
        else:
            break

    if best_s != night_staff:
        new_night_staff = best_s
        new_night_hours = int(new_night_staff * SHIFT_HOURS)
        total_hours = int(day_staff * SHIFT_HOURS + new_night_hours)
        total_checks = day_checks + night_checks
        new_achieved = total_checks / total_hours if total_hours > 0 else 0.0
        new_dev = abs(new_achieved - SU_TARGET)
        return new_night_staff, new_night_hours, new_achieved, new_dev

    return (
        night_staff,
        int(night_staff * SHIFT_HOURS),
        float(row.get("achieved_intensity", 0.0) or 0.0),
        float(row.get("deviation", 0.0) or 0.0),
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def calculate_hours_and_summary(
    input_path: str | Path,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    pos_kso_path: str | Path = DEFAULT_POS_KSO_PATH,
    cluster_path: str | Path = DEFAULT_CLUSTER_PATH,
    auto_path: str | Path = DEFAULT_AUTO_PATH,
    nights_path: str | Path = DEFAULT_INTENSIVE_NIGHTS_PATH,
    drop_automated: bool = False,
    add_total_column: bool = True,
) -> Tuple[Path, Path]:
    """
    Главная функция для платформы.

    Делает:
      1) детальный файл: исходник + часы день/ночь (+опционально всего)
      2) сводный файл: итого + суммы по КССС

    Возвращает: (detail_file_path, summary_file_path)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    # проверим справочники
    pos_kso_path = _require_file(pos_kso_path, "POS_KSO_PATH (POS&KSO.xlsx)")
    cluster_path = _require_file(cluster_path, "CLUSTER_PATH (Тип объекта.xlsx)")
    auto_path = _require_file(auto_path, "AUTO_PATH (Автоматы.xlsx)")
    nights_path = _require_file(nights_path, "INTENSIVE_NIGHTS (АЗС активные ночью.xlsx)")

    df_in, in_delim = _read_input(input_path)
    df_in = df_in.reset_index(drop=True)
    col_map = _detect_columns(df_in)

    # внутренний df с row_id (важно для безопасной обратной склейки)
    base = pd.DataFrame(index=df_in.index)
    base["__row_id"] = df_in.index.astype(int)
    base["KSSS"] = pd.to_numeric(df_in[col_map["KSSS"]], errors="coerce").astype("Int64")
    base["Date"] = df_in[col_map["Date"]]
    base["day_checks"] = pd.to_numeric(df_in[col_map["day_checks"]], errors="coerce").fillna(0)
    base["night_checks"] = pd.to_numeric(df_in[col_map["night_checks"]], errors="coerce").fillna(0)
    base["total_checks"] = base["day_checks"] + base["night_checks"]

    # справочники
    pos_kso_df = pd.read_excel(pos_kso_path)
    cluster_df = pd.read_excel(cluster_path)
    auto_df = pd.read_excel(auto_path)
    nights_df = pd.read_excel(nights_path)

    # дедуп ключей (иначе merge размножает строки и ломает индексы)
    if "КССС" in pos_kso_df.columns:
        pos_kso_df = pos_kso_df.drop_duplicates(subset=["КССС"], keep="first")
    if "КССС_union" in cluster_df.columns:
        cluster_df = cluster_df.drop_duplicates(subset=["КССС_union"], keep="first")
    if "КССС" in auto_df.columns:
        auto_df = auto_df.drop_duplicates(subset=["КССС"], keep="first")
    if "КССС" in nights_df.columns:
        nights_df = nights_df.drop_duplicates(subset=["КССС"], keep="first")

    merged = base.merge(pos_kso_df, left_on="KSSS", right_on="КССС", how="left")
    if "КССС" in merged.columns:
        merged = merged.drop(columns=["КССС"])

    merged = merged.merge(cluster_df, left_on="KSSS", right_on="КССС_union", how="left")
    merged = merged.merge(auto_df, left_on="KSSS", right_on="КССС", how="left")
    if "КССС" in merged.columns:
        merged = merged.drop(columns=["КССС"])

    merged = merged.rename(columns={
        "POS": "POS",
        "КСО": "KSO",
        "Кластер_по_сервису": "cluster",
        "Автомат": "automated",
    })

    if "POS" not in merged.columns:
        merged["POS"] = 0
    if "KSO" not in merged.columns:
        merged["KSO"] = 0
    if "cluster" not in merged.columns:
        merged["cluster"] = "нет"

    merged["POS"] = merged["POS"].fillna(0).astype(int)
    merged["KSO"] = merged["KSO"].fillna(0).astype(int)
    merged["cluster"] = merged["cluster"].fillna("нет")

    merged["type_val"] = merged["cluster"].apply(_get_type_value)
    merged["kso_avail"] = merged["KSO"].apply(_kso_having)

    merged["capacity_count_day"] = merged["POS"] + merged["kso_avail"] + merged["type_val"]
    merged["capacity_count_night"] = merged["POS"]

    res = merged.copy()
    res[[
        "day_staff", "day_hours",
        "night_staff", "night_hours",
        "achieved_intensity", "deviation",
    ]] = res.apply(_optimize_staffing, axis=1, result_type="expand")

    # night_active
    night_flags = nights_df.rename(columns={"КССС": "KSSS"})[["KSSS"]].copy()
    night_flags["KSSS"] = pd.to_numeric(night_flags["KSSS"], errors="coerce").astype("Int64")
    night_flags = night_flags.dropna().drop_duplicates()
    night_flags["night_active"] = 1

    res = res.merge(night_flags, on="KSSS", how="left")
    res["night_active"] = res["night_active"].fillna(0).astype(int)

    # корректировка ночного штата
    res[["night_staff", "night_hours", "achieved_intensity", "deviation"]] = res.apply(
        _adjust_night_staff, axis=1, result_type="expand"
    )

    # опционально исключить автоматизированные объекты
    if drop_automated and "automated" in res.columns:
        mask_auto = (res["automated"] == 1) & ((res["day_checks"] + res["night_checks"]) > 0)
        res = res.loc[~mask_auto].copy()

    # схлопываем обратно строго к 1 строке на __row_id
    hours_map = (
        res[["__row_id", "day_hours", "night_hours"]]
        .groupby("__row_id", as_index=True)
        .first()
        .sort_index()
    )

    # формируем детальный выход
    if drop_automated:
        keep_ids = hours_map.index.to_list()
        out = df_in.loc[keep_ids].copy().reset_index(drop=True)
        out[OUT_DAY_COL] = [int(hours_map.loc[i, "day_hours"]) for i in keep_ids]
        out[OUT_NIGHT_COL] = [int(hours_map.loc[i, "night_hours"]) for i in keep_ids]
    else:
        out = df_in.copy()
        out[OUT_DAY_COL] = out.index.map(hours_map["day_hours"]).fillna(0).astype(int)
        out[OUT_NIGHT_COL] = out.index.map(hours_map["night_hours"]).fillna(0).astype(int)

    if add_total_column:
        out[OUT_TOTAL_COL] = (out[OUT_DAY_COL] + out[OUT_NIGHT_COL]).astype(int)

    # пути сохранения
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_hours" + input_path.suffix)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_path is None:
        summary_path = output_path.with_name(output_path.stem + "_summary" + output_path.suffix)
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # сохраняем детальный файл
    _save_table(out, output_path, in_delim=in_delim)

    # строим свод
    ksss_original_col = col_map["KSSS"]
    totals_df, by_ksss_df = _build_summary(out, ksss_original_col, add_total_column=add_total_column)

    # сохраняем свод
    _save_summary(totals_df, by_ksss_df, summary_path, in_delim=in_delim)

    return output_path, summary_path


def simple_calculate_hours(
    input_path: str | Path,
    output_path: str | Path | None = None,
    **kwargs,
) -> Path:
    """
    Backward-compatible: возвращает ТОЛЬКО путь к детальному файлу.
    Сводный файл всё равно создаётся рядом (или по summary_path, если передан).
    """
    detail_path, _summary_path = calculate_hours_and_summary(
        input_path=input_path,
        output_path=output_path,
        **kwargs,
    )
    return detail_path


class HoursCalculator:
    def __init__(
        self,
        pos_kso_path: str | Path = DEFAULT_POS_KSO_PATH,
        cluster_path: str | Path = DEFAULT_CLUSTER_PATH,
        auto_path: str | Path = DEFAULT_AUTO_PATH,
        nights_path: str | Path = DEFAULT_INTENSIVE_NIGHTS_PATH,
        drop_automated: bool = False,
    ) -> None:
        self.drop_automated = drop_automated
        self.pos_kso_path = _require_file(pos_kso_path, "POS_KSO_PATH (POS&KSO.xlsx)")
        self.cluster_path = _require_file(cluster_path, "CLUSTER_PATH (Тип объекта.xlsx)")
        self.auto_path = _require_file(auto_path, "AUTO_PATH (Автоматы.xlsx)")
        self.nights_path = _require_file(nights_path, "INTENSIVE_NIGHTS (АЗС активные ночью.xlsx)")

        pos_kso_df = pd.read_excel(self.pos_kso_path)
        cluster_df = pd.read_excel(self.cluster_path)
        auto_df = pd.read_excel(self.auto_path)
        nights_df = pd.read_excel(self.nights_path)

        if "КССС" in pos_kso_df.columns:
            pos_kso_df = pos_kso_df.drop_duplicates(subset=["КССС"], keep="first")
        if "КССС_union" in cluster_df.columns:
            cluster_df = cluster_df.drop_duplicates(subset=["КССС_union"], keep="first")
        if "КССС" in auto_df.columns:
            auto_df = auto_df.drop_duplicates(subset=["КССС"], keep="first")
        if "КССС" in nights_df.columns:
            nights_df = nights_df.drop_duplicates(subset=["КССС"], keep="first")

        self.pos_kso_df = pos_kso_df
        self.cluster_df = cluster_df
        self.auto_df = auto_df

        night_flags = nights_df.rename(columns={"КССС": "KSSS"})[["KSSS"]].copy()
        night_flags["KSSS"] = pd.to_numeric(night_flags["KSSS"], errors="coerce").astype("Int64")
        night_flags = night_flags.dropna().drop_duplicates()
        night_flags["night_active"] = 1
        self.night_flags = night_flags

    def add_hours(
        self,
        df_in: pd.DataFrame,
        output_day_col: str = "Часы дневные",
        output_night_col: str = "Часы ночные",
    ) -> pd.DataFrame:
        if df_in.empty:
            out = df_in.copy()
            out[output_day_col] = 0
            out[output_night_col] = 0
            return out

        df_in = df_in.reset_index(drop=True)
        col_map = _detect_columns(df_in)

        base = pd.DataFrame(index=df_in.index)
        base["__row_id"] = df_in.index.astype(int)
        base["KSSS"] = pd.to_numeric(df_in[col_map["KSSS"]], errors="coerce").astype("Int64")
        base["Date"] = df_in[col_map["Date"]]
        base["day_checks"] = pd.to_numeric(df_in[col_map["day_checks"]], errors="coerce").fillna(0)
        base["night_checks"] = pd.to_numeric(df_in[col_map["night_checks"]], errors="coerce").fillna(0)
        base["total_checks"] = base["day_checks"] + base["night_checks"]

        merged = base.merge(self.pos_kso_df, left_on="KSSS", right_on="КССС", how="left")
        if "КССС" in merged.columns:
            merged = merged.drop(columns=["КССС"])

        merged = merged.merge(self.cluster_df, left_on="KSSS", right_on="КССС_union", how="left")
        merged = merged.merge(self.auto_df, left_on="KSSS", right_on="КССС", how="left")
        if "КССС" in merged.columns:
            merged = merged.drop(columns=["КССС"])

        merged = merged.rename(columns={
            "POS": "POS",
            "КСО": "KSO",
            "Кластер_по_сервису": "cluster",
            "Автомат": "automated",
        })

        if "POS" not in merged.columns:
            merged["POS"] = 0
        if "KSO" not in merged.columns:
            merged["KSO"] = 0
        if "cluster" not in merged.columns:
            merged["cluster"] = "нет"

        merged["POS"] = merged["POS"].fillna(0).astype(int)
        merged["KSO"] = merged["KSO"].fillna(0).astype(int)
        merged["cluster"] = merged["cluster"].fillna("нет")

        merged["type_val"] = merged["cluster"].apply(_get_type_value)
        merged["kso_avail"] = merged["KSO"].apply(_kso_having)
        merged["capacity_count_day"] = merged["POS"] + merged["kso_avail"] + merged["type_val"]
        merged["capacity_count_night"] = merged["POS"]

        res = merged.copy()
        res[[
            "day_staff", "day_hours",
            "night_staff", "night_hours",
            "achieved_intensity", "deviation",
        ]] = res.apply(_optimize_staffing, axis=1, result_type="expand")

        res = res.merge(self.night_flags, on="KSSS", how="left")
        res["night_active"] = res["night_active"].fillna(0).astype(int)

        res[["night_staff", "night_hours", "achieved_intensity", "deviation"]] = res.apply(
            _adjust_night_staff, axis=1, result_type="expand"
        )

        if self.drop_automated and "automated" in res.columns:
            mask_auto = (res["automated"] == 1) & ((res["day_checks"] + res["night_checks"]) > 0)
            res = res.loc[~mask_auto].copy()

        hours_map = (
            res[["__row_id", "day_hours", "night_hours"]]
            .groupby("__row_id", as_index=True)
            .first()
            .sort_index()
        )

        out = df_in.copy()
        out[output_day_col] = out.index.map(hours_map["day_hours"]).fillna(0).astype(int)
        out[output_night_col] = out.index.map(hours_map["night_hours"]).fillna(0).astype(int)
        return out


# ---------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------
def _save_table(df: pd.DataFrame, path: Path, in_delim: Optional[str]) -> None:
    if path.suffix.lower() == ".csv":
        sep = in_delim or ";"
        df.to_csv(path, index=False, sep=sep, encoding="utf-8-sig")
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Hours", index=False)


def _build_summary(
    out: pd.DataFrame,
    ksss_col: str,
    add_total_column: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # totals (одна строка)
    totals = {
        "Показатель": "ИТОГО",
        OUT_DAY_COL: int(out[OUT_DAY_COL].sum()),
        OUT_NIGHT_COL: int(out[OUT_NIGHT_COL].sum()),
        "Количество строк": int(len(out)),
        "Количество объектов (уник. КССС)": int(out[ksss_col].nunique(dropna=True)),
    }
    if add_total_column and OUT_TOTAL_COL in out.columns:
        totals[OUT_TOTAL_COL] = int(out[OUT_TOTAL_COL].sum())

    totals_df = pd.DataFrame([totals])

    # by_ksss
    cols = [OUT_DAY_COL, OUT_NIGHT_COL] + ([OUT_TOTAL_COL] if add_total_column and OUT_TOTAL_COL in out.columns else [])
    by_ksss = (
        out.groupby(ksss_col, dropna=False)[cols]
        .sum()
        .reset_index()
        .rename(columns={ksss_col: "КССС"})
        .sort_values("КССС", kind="stable")
    )

    return totals_df, by_ksss


def _save_summary(
    totals_df: pd.DataFrame,
    by_ksss_df: pd.DataFrame,
    path: Path,
    in_delim: Optional[str],
) -> None:
    if path.suffix.lower() == ".csv":
        # для CSV: одним файлом "by_ksss" + внизу строка ИТОГО
        sep = in_delim or ";"
        out = by_ksss_df.copy()
        # добавим строку итогов (с пустым КССС)
        total_row = {c: "" for c in out.columns}
        total_row["КССС"] = "ИТОГО"
        for c in out.columns:
            if c in (OUT_DAY_COL, OUT_NIGHT_COL, OUT_TOTAL_COL):
                try:
                    total_row[c] = int(by_ksss_df[c].sum())
                except Exception:
                    pass
        out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)
        out.to_csv(path, index=False, sep=sep, encoding="utf-8-sig")
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            totals_df.to_excel(writer, sheet_name="Totals", index=False)
            by_ksss_df.to_excel(writer, sheet_name="By_KSSS", index=False)


# ---------------------------------------------------------------------
# CLI (опционально)
# ---------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Расчёт часов (день/ночь) из прогноза чеков + свод сумм")
    p.add_argument("--input", "-i", required=True, help="Путь к входному файлу (.xlsx/.csv)")
    p.add_argument("--output", "-o", default=None, help="Путь к детальному выходному файлу. По умолчанию *_hours рядом со входным")
    p.add_argument("--summary", "-s", default=None, help="Путь к сводному файлу. По умолчанию *_hours_summary рядом с детальным")
    p.add_argument("--pos-kso", default=DEFAULT_POS_KSO_PATH, help="Путь к POS&KSO.xlsx")
    p.add_argument("--cluster", default=DEFAULT_CLUSTER_PATH, help="Путь к Тип объекта.xlsx")
    p.add_argument("--auto", default=DEFAULT_AUTO_PATH, help="Путь к Автоматы.xlsx")
    p.add_argument("--nights", default=DEFAULT_INTENSIVE_NIGHTS_PATH, help="Путь к АЗС активные ночью.xlsx")
    p.add_argument("--drop-automated", action="store_true", help="Исключать автоматизированные объекты")
    p.add_argument("--no-total", action="store_true", help="Не добавлять колонку 'кол-во часов всего'")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    detail_path, summary_path = calculate_hours_and_summary(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        pos_kso_path=args.pos_kso,
        cluster_path=args.cluster,
        auto_path=args.auto,
        nights_path=args.nights,
        drop_automated=bool(args.drop_automated),
        add_total_column=not bool(args.no_total),
    )
    print(f"Готово.\nДетальный файл: {detail_path}\nСводный файл: {summary_path}")


if __name__ == "__main__":
    main()
