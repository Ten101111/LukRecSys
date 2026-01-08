import datetime as dt
import traceback
from pathlib import Path
import runpy
import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage, ttk

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union

import Check_check_functions as ccf
from Hours_from_checks_simple import simple_calculate_hours
import constants

# --- ВЕТКА 2: упрощённый расчёт часов (встроено в platform_gui.py) ---
# Вход: csv/xlsx с колонками:
#   ORG_KSSS, date, Прогноз, Чеки дневные, Чеки ночные
# Выход: исходные колонки + "кол-во часов днем", "кол-во часов ночью", "кол-во часов всего" (= день + ночь)

import csv

# --- Параметры алгоритма (как в Hours_check.py) ---
_SIMPLE_TARGET_DAY_INTENSITY = 22.0
_SIMPLE_TARGET_NIGHT_INTENSITY = 15.0
_SIMPLE_SHIFT_HOURS = 12.0
_SIMPLE_SU_TARGET = 18.0


def _simple_norm(s: str) -> str:
    return str(s).strip().lower()


def _simple_sniff_csv_delimiter(path: Path) -> str:
    """Пытаемся определить разделитель CSV. Если не вышло — используем ';'."""
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "	", "|"])
        return dialect.delimiter
    except Exception:
        # На практике у вас чаще ';'
        return ";"


def _simple_read_table(path: Path):
    """Читает .xlsx/.xls/.xlsm или .csv, возвращает (df, csv_delimiter_or_None)."""
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path), None
    if suf == ".csv":
        delim = _simple_sniff_csv_delimiter(path)
        # engine="python" более терпим к различным CSV
        df = pd.read_csv(path, sep=delim, engine="python")
        return df, delim
    raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}. Ожидаю .csv или .xlsx")


def _simple_detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Находит нужные колонки во входном файле и возвращает mapping internal->original."""
    cols = {_simple_norm(c): c for c in df.columns}

    def pick(*variants: str) -> str | None:
        for v in variants:
            v_norm = _simple_norm(v)
            if v_norm in cols:
                return cols[v_norm]
        return None

    ksss = pick("ORG_KSSS", "КССС", "KSSS", "org_ksss", "ksss")
    date = pick("date", "Дата", "дата")
    total = pick("Прогноз", "чеки совокупные", "чеки итого", "Итого прогноз чеков на сутки", "total_checks")
    day = pick("Чеки дневные", "чеки дневные", "Дневные чеки", "чеки день", "day_checks")
    night = pick("Чеки ночные", "чеки ночные", "Ночные чеки", "чеки ночь", "night_checks")

    missing = [
        name
        for name, col in (
            ("ORG_KSSS/КССС", ksss),
            ("date/Дата", date),
            ("Прогноз (совокупные чеки)", total),
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
        "total_checks": total,
        "day_checks": day,
        "night_checks": night,
    }


def _simple_get_type_value(cluster_name: str) -> int:
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


def _simple_kso_having(kso_num) -> int:
    try:
        kso_num = int(kso_num)
    except Exception:
        return 0
    return 1 if kso_num > 0 else 0


def _simple_choose_staff_for_shift(checks: float, cap: int, target_intensity: float) -> int:
    """Подбирает целое число staff (1..cap) с минимальным |checks/(12*staff) - target|."""
    if checks <= 0:
        return 0
    cap = max(int(cap or 0), 1)
    best_staff = 1
    best_diff = float("inf")
    for s in range(1, cap + 1):
        diff = abs((checks / (_SIMPLE_SHIFT_HOURS * s)) - target_intensity)
        if diff < best_diff - 1e-12 or (abs(diff - best_diff) <= 1e-12 and s < best_staff):
            best_diff = diff
            best_staff = s
    return best_staff


def _simple_optimize_staffing(row: pd.Series) -> Tuple[int, int, int, int, float, float]:
    day_cap = int(row.get("capacity_count_day", 0) or 0)
    night_cap = int(row.get("capacity_count_night", 0) or 0)

    day_checks = float(row.get("day_checks", 0) or 0)
    night_checks = float(row.get("night_checks", 0) or 0)
    total_checks = day_checks + night_checks

    if total_checks <= 0:
        return 0, 0, 0, 0, 0.0, abs(0.0 - _SIMPLE_SU_TARGET)

    # если чеки есть, а мощность 0 — поднимаем до 1
    if day_checks > 0 and day_cap == 0:
        day_cap = 1
    if night_checks > 0 and night_cap == 0:
        night_cap = 1

    day_staff = _simple_choose_staff_for_shift(day_checks, day_cap, _SIMPLE_TARGET_DAY_INTENSITY) if day_checks > 0 else 0
    night_staff = _simple_choose_staff_for_shift(night_checks, night_cap, _SIMPLE_TARGET_NIGHT_INTENSITY) if night_checks > 0 else 0

    day_hours = int(day_staff * _SIMPLE_SHIFT_HOURS)
    night_hours = int(night_staff * _SIMPLE_SHIFT_HOURS)
    total_hours = day_hours + night_hours

    achieved_intensity = (total_checks / total_hours) if total_hours > 0 else 0.0
    deviation = abs(achieved_intensity - _SIMPLE_SU_TARGET)

    return day_staff, day_hours, night_staff, night_hours, achieved_intensity, deviation


def _simple_adjust_night_staff(row: pd.Series) -> Tuple[int, int, float, float]:
    night_active = int(row.get("night_active", 0) or 0)
    night_checks = float(row.get("night_checks", 0) or 0)
    night_staff = int(row.get("night_staff", 0) or 0)
    day_staff = int(row.get("day_staff", 0) or 0)
    day_checks = float(row.get("day_checks", 0) or 0)

    if night_active != 1 or night_checks <= 0 or night_staff <= 0:
        return night_staff, int(night_staff * _SIMPLE_SHIFT_HOURS), float(row.get("achieved_intensity", 0.0) or 0.0), float(row.get("deviation", 0.0) or 0.0)

    def night_diff(s: int) -> float:
        return abs((night_checks / (_SIMPLE_SHIFT_HOURS * s)) - _SIMPLE_SU_TARGET)

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
        new_night_hours = int(new_night_staff * _SIMPLE_SHIFT_HOURS)
        total_hours = int(day_staff * _SIMPLE_SHIFT_HOURS + new_night_hours)
        total_checks = day_checks + night_checks
        new_achieved = total_checks / total_hours if total_hours > 0 else 0.0
        new_dev = abs(new_achieved - _SIMPLE_SU_TARGET)
        return new_night_staff, new_night_hours, new_achieved, new_dev

    return night_staff, int(night_staff * _SIMPLE_SHIFT_HOURS), float(row.get("achieved_intensity", 0.0) or 0.0), float(row.get("deviation", 0.0) or 0.0)


def simple_calculate_hours(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Считает часы по упрощённому файлу (csv/xlsx) и сохраняет результат.
    Возвращает путь к выходному файлу.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    df_in, in_delim = _simple_read_table(input_path)
    df_in = df_in.reset_index(drop=True)
    col_map = _simple_detect_columns(df_in)

    # Базовые поля
    base = pd.DataFrame(index=df_in.index)
    base["__row_id"] = df_in.index.astype(int)  # <-- ДОБАВИТЬ
    base["KSSS"] = pd.to_numeric(df_in[col_map["KSSS"]], errors="coerce").astype("Int64")
    base["Date"] = df_in[col_map["Date"]]
    base["total_checks"] = pd.to_numeric(df_in[col_map["total_checks"]], errors="coerce").fillna(0)
    base["day_checks"] = pd.to_numeric(df_in[col_map["day_checks"]], errors="coerce").fillna(0)
    base["night_checks"] = pd.to_numeric(df_in[col_map["night_checks"]], errors="coerce").fillna(0)

    # Справочники — как в проекте
    pos_kso_df = pd.read_excel(constants.POS_KSO_PATH)
    cluster_df = pd.read_excel(constants.CLUSTER_PATH)
    auto_df = pd.read_excel(constants.AUTO_PATH)
    nights_df = pd.read_excel(constants.INTENSIVE_NIGHTS)

    merged = base.merge(pos_kso_df, left_on="KSSS", right_on="КССС", how="left")
    if "КССС" in pos_kso_df.columns:
        pos_kso_df = pos_kso_df.drop_duplicates(subset=["КССС"], keep="first")
    if "КССС_union" in cluster_df.columns:
        cluster_df = cluster_df.drop_duplicates(subset=["КССС_union"], keep="first")
    if "КССС" in auto_df.columns:
        auto_df = auto_df.drop_duplicates(subset=["КССС"], keep="first")
    if "КССС" in nights_df.columns:
        nights_df = nights_df.drop_duplicates(subset=["КССС"], keep="first")

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

    merged["type_val"] = merged["cluster"].apply(_simple_get_type_value)
    merged["kso_avail"] = merged["KSO"].apply(_simple_kso_having)

    merged["capacity_count_day"] = merged["POS"] + merged["kso_avail"] + merged["type_val"]
    merged["capacity_count_night"] = merged["POS"]

    res = merged.copy()
    res[[
        "day_staff", "day_hours",
        "night_staff", "night_hours",
        "achieved_intensity", "deviation",
    ]] = res.apply(_simple_optimize_staffing, axis=1, result_type="expand")

    # night_active флаг
    night_flags = nights_df.rename(columns={"КССС": "KSSS"})[["KSSS"]].copy()
    night_flags["KSSS"] = pd.to_numeric(night_flags["KSSS"], errors="coerce").astype("Int64")
    night_flags = night_flags.dropna().drop_duplicates()
    night_flags["night_active"] = 1

    res = res.merge(night_flags, left_on="KSSS", right_on="KSSS", how="left")
    res["night_active"] = res["night_active"].fillna(0).astype(int)

    # корректировка ночного штата для активных ночью
    res[["night_staff", "night_hours", "achieved_intensity", "deviation"]] = res.apply(
        _simple_adjust_night_staff, axis=1, result_type="expand"
    )

    # итог — исходные колонки + 3 новых
    hours_map = (
        res[["__row_id", "day_hours", "night_hours"]]
        .groupby("__row_id", as_index=True)
        .first()
        .sort_index()
    )

    out = df_in.copy()
    out["кол-во часов днем"] = out.index.map(hours_map["day_hours"]).fillna(0).astype(int)
    out["кол-во часов ночью"] = out.index.map(hours_map["night_hours"]).fillna(0).astype(int)
    out["кол-во часов всего"] = (out["кол-во часов днем"] + out["кол-во часов ночью"]).astype(int)

    # дефолтный output
    if output_path is None:
        if input_path.suffix.lower() == ".csv":
            output_path = input_path.with_name(input_path.stem + "_hours.csv")
        else:
            output_path = input_path.with_name(input_path.stem + "_hours.xlsx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        # пишем CSV (разделитель — как у входа, если удалось определить)
        sep = in_delim or ";"
        out.to_csv(output_path, index=False, sep=sep, encoding="utf-8-sig")
    else:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name="Hours", index=False)

    return output_path


# --- НАСТРОЙКИ ОФОРМЛЕНИЯ ---
COLOR_RED = "#ed1b34"   # основной красный
COLOR_BG = "#ffffff"    # фоновый белый
COLOR_TEXT = "#000000"  # чёрный текст

LOGO_PATH = Path(__file__).resolve().parent / "lukoil_logo_white.png"  # имя файла логотипа


def _ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def run_pipeline(source_file: str):
    """
    ОСНОВНАЯ ВЕТКА (НЕ ТРОГАЕМ ЛОГИКУ):
    1) prediction_analys из Check_check_functions.py
    2) Hours_check.py
    Возвращает пути к основным результатам.
    """
    source_path = Path(source_file)
    if not source_path.exists():
        raise FileNotFoundError(f"Файл не найден: {source_path}")

    new_name = dt.datetime.now().strftime("%Y%m%d_%H%M")

    save_dir = Path(constants.PATH_TO_SAVED)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Шаг 1. Подготовка чеков ---
    ccf.prediction_analys(
        file_name=str(source_path),
        path_to_save=str(save_dir),
        new_name=new_name,
    )

    month_name = ccf.month
    if not month_name:
        raise RuntimeError("Не удалось определить месяц из файла (ccf.month пустой).")

    checks_path = save_dir / f"{month_name}_result_{new_name}.xlsx"

    # --- Шаг 2. Обновляем константы для Hours_check ---
    constants.CHECKS_PATH = str(checks_path)

    mnth = month_name.lower()
    if mnth not in constants.MONTHS_NUMS:
        raise ValueError(f"Месяц {mnth!r} не найден в MONTHS_NUMS.")
    constants.MONTH = constants.MONTHS_NUMS[mnth]
    year = dt.datetime.now().year
    if dt.datetime.now().month > constants.MONTH:
        year += 1
    constants.YEAR = year

    output_full_dir = Path(constants.OUTPUT_PATH).parent
    full_path = output_full_dir / f"{month_name.title()}.xlsx"
    constants.OUTPUT_PATH = str(_ensure_unique_path(full_path))

    output_tpl_dir = Path(constants.OUTPUT_PATH_CHANGES).parent
    tpl_path = output_tpl_dir / f"Шаблон с заменами ({month_name.title()}).xlsx"
    constants.OUTPUT_PATH_CHANGES = str(_ensure_unique_path(tpl_path))

    # --- Шаг 3. Запускаем Hours_check ---
    runpy.run_path(str(Path(__file__).resolve().parent / "Hours_check.py"),
                   run_name="__main__")

    return str(checks_path), constants.OUTPUT_PATH, constants.OUTPUT_PATH_CHANGES


def run_simple_hours(input_file: str) -> str:
    """
    ВЕТКА 2:
    Расчёт часов из файла csv/xlsx с колонками:
      ORG_KSSS, date, Прогноз, Чеки дневные, Чеки ночные
    Возвращает путь к выходному файлу.
    """
    in_path = Path(input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Файл не найден: {in_path}")
    out_path = simple_calculate_hours(input_path=in_path)
    return str(out_path)


def main():
    root = tk.Tk()
    root.title("Платформа анализа чеков ЛУКОЙЛ")
    root.configure(bg=COLOR_BG)

    icon_candidates = [
        Path(__file__).with_name("lukoil.ico"),
        Path(__file__).with_name("lukoil.ico.ico"),
    ]
    for icon_path in icon_candidates:
        if icon_path.exists():
            root.iconbitmap(icon_path)
            break
    root.state('zoomed')

    # --- АДАПТАЦИЯ ПОД ЭКРАН ---
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    win_width = int(screen_width * 0.7)
    win_height = int(screen_height * 0.45)

    pos_x = (screen_width - win_width) // 2
    pos_y = (screen_height - win_height) // 3

    root.geometry(f"{win_width}x{win_height}+{pos_x}+{pos_y}")
    root.minsize(860, 360)

    status_var = tk.StringVar(value="Ожидание действий")

    # ---------- ШАПКА С ЦЕНТРОВАННЫМ ЛОГОТИПОМ ----------
    header = tk.Frame(root, bg=COLOR_RED)
    header.pack(fill="x", side="top")

    logo_img = None
    if LOGO_PATH.exists():
        try:
            logo_img = PhotoImage(file=str(LOGO_PATH))
        except Exception:
            logo_img = None

    if logo_img is not None:
        logo_label = tk.Label(header, image=logo_img, bg=COLOR_RED)
        logo_label.image = logo_img
        logo_label.pack(pady=8)

    # ---------- НАЗВАНИЕ ПЛАТФОРМЫ В РАМКЕ ----------
    title_border = tk.Frame(root, bg=COLOR_RED)
    title_border.pack(fill="x", padx=20, pady=(8, 4))

    title_bg = tk.Frame(title_border, bg=COLOR_BG)
    title_bg.pack(fill="x", padx=2, pady=2)

    title_label = tk.Label(
        title_bg,
        text="Платформа анализа чеков",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 14, "bold"),
    )
    title_label.pack(padx=10, pady=6)

    # ---------- ОСНОВНОЙ БЛОК ----------
    main_frame = tk.Frame(root, bg=COLOR_BG, padx=15, pady=15)
    main_frame.pack(fill="both", expand=True)

    # Вкладки: 1) основной пайплайн 2) упрощённый расчёт
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill="both", expand=True)

    tab_full = tk.Frame(notebook, bg=COLOR_BG)
    tab_simple = tk.Frame(notebook, bg=COLOR_BG)

    notebook.add(tab_full, text="Полный анализ (рассылка → отчёты)")
    notebook.add(tab_simple, text="Часы из упрощённого файла")

    # ======================================================================
    # TAB 1: ОСНОВНАЯ ВЕТКА (как было)
    # ======================================================================
    selected_file = tk.StringVar()

    lbl_intro = tk.Label(
        tab_full,
        text="1. Выберите файл с рассылкой (.xlsx)",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 11),
    )
    lbl_intro.pack(anchor="w")

    file_frame = tk.Frame(tab_full, bg=COLOR_BG)
    file_frame.pack(fill="x", pady=(5, 12))

    entry = tk.Entry(
        file_frame,
        textvariable=selected_file,
        font=("Segoe UI", 10),
        relief="solid",
        bd=1,
    )
    entry.pack(side="left", fill="x", expand=True, ipadx=4, ipady=3)

    def choose_file():
        file_path = filedialog.askopenfilename(
            title="Выберите файл с рассылкой",
            filetypes=[("Excel файлы", "*.xlsx *.xls"), ("Все файлы", "*.*")]
        )
        if file_path:
            selected_file.set(file_path)
            status_var.set("Файл рассылки выбран, можно запускать анализ")

    btn_browse = tk.Button(
        file_frame,
        text="Обзор…",
        command=choose_file,
        bg=COLOR_RED,
        fg="white",
        activebackground=COLOR_RED,
        activeforeground="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=10,
        pady=3,
    )
    btn_browse.pack(side="left", padx=(8, 0))

    def start_analysis():
        file_path = selected_file.get()
        if not file_path:
            messagebox.showwarning("Нет файла", "Сначала выберите файл с рассылкой (.xlsx).")
            return

        try:
            btn_run.config(state="disabled")
            status_var.set("Идёт расчёт... окно может временно не отвечать.")
            root.update_idletasks()

            checks_path, output_full, output_tpl = run_pipeline(file_path)

            msg = (
                "Анализ завершён.\n\n"
                f"Промежуточный файл с чеками:\n{checks_path}\n\n"
                f"Полный отчёт по часам:\n{output_full}\n\n"
                f"Шаблон с заменами:\n{output_tpl}"
            )
            messagebox.showinfo("Готово", msg)
            status_var.set("Готово (полный анализ)")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Во время выполнения произошла ошибка:\n{e}")
            status_var.set("Ошибка (полный анализ) — подробности в консоли")
        finally:
            btn_run.config(state="normal")

    btn_run = tk.Button(
        tab_full,
        text="2. Запустить анализ",
        command=start_analysis,
        bg=COLOR_RED,
        fg="white",
        activebackground=COLOR_RED,
        activeforeground="white",
        relief="flat",
        font=("Segoe UI", 11, "bold"),
        padx=15,
        pady=6,
    )
    btn_run.pack(anchor="w", pady=(0, 10))

    info_full = tk.Label(
        tab_full,
        text=(
            "Результат:\n"
            "• Промежуточный файл (транспонирование)\n"
            "• Полный отчёт по часам\n"
            "• Шаблон с заменами\n"
        ),
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 10),
        justify="left",
    )
    info_full.pack(anchor="w")

    # ======================================================================
    # TAB 2: НОВАЯ ВЕТКА (Hours_from_checks_simple)
    # ======================================================================
    selected_simple = tk.StringVar()

    lbl_simple = tk.Label(
        tab_simple,
        text="1. Выберите файл (.xlsx или .csv) с колонками: ORG_KSSS, date, Прогноз, Чеки дневные, Чеки ночные",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 11),
        wraplength=1100,
        justify="left",
    )
    lbl_simple.pack(anchor="w")

    file_frame2 = tk.Frame(tab_simple, bg=COLOR_BG)
    file_frame2.pack(fill="x", pady=(5, 8))

    entry2 = tk.Entry(
        file_frame2,
        textvariable=selected_simple,
        font=("Segoe UI", 10),
        relief="solid",
        bd=1,
    )
    entry2.pack(side="left", fill="x", expand=True, ipadx=4, ipady=3)

    def choose_simple_file():
        file_path = filedialog.askopenfilename(
            title="Выберите упрощённый файл",
            filetypes=[
                ("CSV файлы", "*.csv"),
                ("Excel файлы", "*.xlsx;*.xls;*.xlsm"),
                ("Все файлы", "*.*"),
            ],
        )
        if file_path:
            selected_simple.set(file_path)
            status_var.set("Упрощённый файл выбран, можно запускать расчёт часов")

    btn_browse2 = tk.Button(
        file_frame2,
        text="Обзор…",
        command=choose_simple_file,
        bg=COLOR_RED,
        fg="white",
        activebackground=COLOR_RED,
        activeforeground="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=10,
        pady=3,
    )
    btn_browse2.pack(side="left", padx=(8, 0))

    def start_simple():
        file_path = selected_simple.get()
        if not file_path:
            messagebox.showwarning("Нет файла", "Сначала выберите файл (.xlsx или .csv).")
            return

        try:
            btn_run2.config(state="disabled")
            status_var.set("Идёт расчёт часов (упрощённый файл)...")
            root.update_idletasks()

            out_path = run_simple_hours(file_path)

            msg = (
                "Расчёт часов завершён.\n\n"
                f"Выходной файл:\n{out_path}"
            )
            messagebox.showinfo("Готово", msg)
            status_var.set("Готово (упрощённый расчёт)")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Во время выполнения произошла ошибка:\n{e}")
            status_var.set("Ошибка (упрощённый расчёт) — подробности в консоли")
        finally:
            btn_run2.config(state="normal")

    btn_run2 = tk.Button(
        tab_simple,
        text="2. Рассчитать часы",
        command=start_simple,
        bg=COLOR_RED,
        fg="white",
        activebackground=COLOR_RED,
        activeforeground="white",
        relief="flat",
        font=("Segoe UI", 11, "bold"),
        padx=15,
        pady=6,
    )
    btn_run2.pack(anchor="w", pady=(6, 10))

    info_simple = tk.Label(
        tab_simple,
        text=(
            "Результат:\n"
            "• Файл с исходными колонками +\n"
            "  - кол-во часов днем\n"
            "  - кол-во часов ночью\n"
            "  - кол-во часов всего (строго = день + ночь)\n"
            "\n"
            "Выход сохраняется рядом со входным файлом с суффиксом *_hours.xlsx (или *_hours.csv для CSV)"
        ),
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 10),
        justify="left",
    )
    info_simple.pack(anchor="w")

    # ---------- СТАТУС ВНИЗУ ----------
    status_frame = tk.Frame(root, bg=COLOR_BG)
    status_frame.pack(fill="x", side="bottom", pady=(0, 8), padx=10)

    lbl_status_title = tk.Label(
        status_frame,
        text="Статус:",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 9, "bold"),
    )
    lbl_status_title.pack(side="left")

    lbl_status = tk.Label(
        status_frame,
        textvariable=status_var,
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 9),
    )
    lbl_status.pack(side="left", padx=(5, 0))

    root.mainloop()


if __name__ == "__main__":
    main()
