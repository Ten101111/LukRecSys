# -*- coding: utf-8 -*-
import calendar
import datetime as dt
import os
import runpy
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage, ttk
import subprocess
import sys

import pandas as pd

from constant_functions import pick_latest_file, pick_latest_folder

COLOR_RED = "#ed1b34"
COLOR_BG = "#ffffff"
COLOR_TEXT = "#000000"
COLOR_ENTRY_BG = "#ffffff"
COLOR_ENTRY_FG = "#000000"
COLOR_TODAY_BG = "#ffe4e8"
COLOR_TODAY_FG = "#000000"
BUTTON_BG = "#ffffff"
BUTTON_FG = "#000000"
BUTTON_BORDER = "#000000"
COLOR_PROGRESS = "#28a745"
COLOR_PROGRESS_BORDER = "#9c9c9c"

BASE_DIR = Path(__file__).resolve().parent
FORECAST_SCRIPT = BASE_DIR / "Прогнозирование на основе МЛ и ВЕСА.py"
DEFAULT_CHECKS_ROOT = BASE_DIR.parent / "Обработка чеков" / "Объединение чеков"
DEFAULT_AZS_FILE = BASE_DIR.parent / "АЗС для прогноза.xlsx"

def _suppress_statsmodels_warnings() -> None:
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
    except Exception:
        return
    import warnings

    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _try_pick_latest_checks() -> Path | None:
    try:
        latest_folder = pick_latest_folder(DEFAULT_CHECKS_ROOT)
        return pick_latest_file(latest_folder)
    except Exception:
        return None


def _normalize_ksss(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _load_ksss_list() -> list[str]:
    if not DEFAULT_AZS_FILE.exists():
        return []
    try:
        df = pd.read_excel(DEFAULT_AZS_FILE)
    except Exception:
        return []
    if "КССС" not in df.columns:
        return []
    ksss_values = [_normalize_ksss(v) for v in df["КССС"].dropna().tolist()]
    return ksss_values


def _sort_ksss(values: list[str]) -> list[str]:
    def key(v: str):
        try:
            return (0, int(v))
        except Exception:
            return (1, str(v))
    return sorted(values, key=key)


def _detect_dark_mode() -> bool:
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                stderr=subprocess.STDOUT,
            ).decode("utf-8", errors="ignore")
            return "Dark" in out
        except Exception:
            return False
    if sys.platform.startswith("win"):
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                return value == 0
        except Exception:
            return False
    return False


def _apply_theme(dark: bool) -> None:
    global COLOR_BG, COLOR_TEXT, COLOR_ENTRY_BG, COLOR_ENTRY_FG, COLOR_TODAY_BG, COLOR_TODAY_FG
    if dark:
        COLOR_BG = "#1f1f1f"
        COLOR_TEXT = "#f2f2f2"
        COLOR_ENTRY_BG = "#2b2b2b"
        COLOR_ENTRY_FG = "#f2f2f2"
        COLOR_TODAY_BG = "#3a2a2d"
        COLOR_TODAY_FG = "#f2f2f2"
    else:
        COLOR_BG = "#ffffff"
        COLOR_TEXT = "#000000"
        COLOR_ENTRY_BG = "#ffffff"
        COLOR_ENTRY_FG = "#000000"
        COLOR_TODAY_BG = "#ffe4e8"
        COLOR_TODAY_FG = "#000000"


def _load_logo() -> PhotoImage | None:
    candidates = [
        BASE_DIR / "lukoil_logo_white.png",
        BASE_DIR.parent / "Проверка чеков" / "lukoil_logo_white.png",
    ]
    for path in candidates:
        if path.exists():
            try:
                return PhotoImage(file=str(path))
            except Exception:
                continue
    return None


def _set_icon(root: tk.Tk) -> None:
    candidates = [
        BASE_DIR / "lukoil.ico",
        BASE_DIR / "lukoil.ico.ico",
        BASE_DIR.parent / "Проверка чеков" / "lukoil.ico",
        BASE_DIR.parent / "Проверка чеков" / "lukoil.ico.ico",
    ]
    for path in candidates:
        if path.exists():
            try:
                root.iconbitmap(path)
                break
            except Exception:
                continue


def _parse_date(value: str) -> dt.date | None:
    value = (value or "").strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except Exception:
            continue
    try:
        return dt.datetime.fromisoformat(value).date()
    except Exception:
        return None


def _format_date(value: dt.date) -> str:
    return value.strftime("%Y-%m-%d")


def _add_month(base_date: dt.date, months: int = 1) -> dt.date:
    year = base_date.year
    month = base_date.month + months
    while month > 12:
        month -= 12
        year += 1
    while month < 1:
        month += 12
        year -= 1
    day = min(base_date.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


class DatePickerPopup(tk.Toplevel):
    def __init__(self, master: tk.Tk, initial_date: dt.date, on_select: callable, selected_date: dt.date | None = None) -> None:
        super().__init__(master)
        self.title("Выберите дату")
        self.resizable(False, False)
        self.configure(bg=COLOR_BG)
        self.transient(master)
        self.grab_set()

        self._on_select = on_select
        self._selected = selected_date or initial_date
        self._today = dt.date.today()

        info_frame = tk.Frame(self, bg=COLOR_BG)
        info_frame.pack(fill="x", padx=10, pady=(8, 4))

        self._today_label = tk.Label(
            info_frame,
            text=f"Сегодня: {self._today.strftime('%d.%m.%Y')}",
            bg=COLOR_TODAY_BG,
            fg=COLOR_TODAY_FG,
            font=("Segoe UI", 9, "bold"),
            padx=6,
            pady=4,
        )
        self._today_label.pack(side="left")

        self._selected_label = tk.Label(
            info_frame,
            text="",
            bg=COLOR_RED,
            fg="white",
            font=("Segoe UI", 9, "bold"),
            padx=6,
            pady=4,
        )
        self._selected_label.pack(side="left", padx=(8, 0))

        selectors = tk.Frame(self, bg=COLOR_BG)
        selectors.pack(fill="x", padx=10, pady=(4, 8))

        self._day_var = tk.IntVar(value=self._selected.day)
        self._month_var = tk.IntVar(value=self._selected.month)
        self._year_var = tk.IntVar(value=self._selected.year)

        tk.Label(
            selectors,
            text="День",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            selectors,
            text="Месяц",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10),
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        tk.Label(
            selectors,
            text="Год",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10),
        ).grid(row=0, column=2, sticky="w", padx=(10, 0))

        self._day_spin = tk.Spinbox(
            selectors,
            from_=1,
            to=31,
            textvariable=self._day_var,
            width=6,
            bg=COLOR_ENTRY_BG,
            fg=COLOR_ENTRY_FG,
            insertbackground=COLOR_ENTRY_FG,
        )
        self._day_spin.grid(row=1, column=0, sticky="w")

        self._month_spin = tk.Spinbox(
            selectors,
            from_=1,
            to=12,
            textvariable=self._month_var,
            width=6,
            bg=COLOR_ENTRY_BG,
            fg=COLOR_ENTRY_FG,
            insertbackground=COLOR_ENTRY_FG,
        )
        self._month_spin.grid(row=1, column=1, sticky="w", padx=(10, 0))

        self._year_spin = tk.Spinbox(
            selectors,
            from_=2020,
            to=2030,
            textvariable=self._year_var,
            width=8,
            bg=COLOR_ENTRY_BG,
            fg=COLOR_ENTRY_FG,
            insertbackground=COLOR_ENTRY_FG,
        )
        self._year_spin.grid(row=1, column=2, sticky="w", padx=(10, 0))

        self._day_var.trace_add("write", self._on_change)
        self._month_var.trace_add("write", self._on_change)
        self._year_var.trace_add("write", self._on_change)

        self._update_day_range()
        self._update_selected_label()

        btn_frame = tk.Frame(self, bg=COLOR_BG)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        btn_ok = tk.Button(
            btn_frame,
            text="Выбрать",
            command=self._apply,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_BG,
            activeforeground=BUTTON_FG,
            relief="solid",
            bd=1,
            highlightbackground=BUTTON_BORDER,
            highlightcolor=BUTTON_BORDER,
            highlightthickness=1,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=4,
        )
        btn_ok.pack(side="left")

        btn_cancel = tk.Button(
            btn_frame,
            text="Отмена",
            command=self.destroy,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_BG,
            activeforeground=BUTTON_FG,
            relief="solid",
            bd=1,
            highlightbackground=BUTTON_BORDER,
            highlightcolor=BUTTON_BORDER,
            highlightthickness=1,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=4,
        )
        btn_cancel.pack(side="left", padx=(8, 0))

    def _update_day_range(self) -> None:
        try:
            year = int(self._year_var.get())
            month = int(self._month_var.get())
        except Exception:
            return
        if month < 1:
            month = 1
            self._month_var.set(month)
        elif month > 12:
            month = 12
            self._month_var.set(month)
        if year < 2020:
            year = 2020
            self._year_var.set(year)
        elif year > 2030:
            year = 2030
            self._year_var.set(year)
        max_day = calendar.monthrange(year, month)[1]
        self._day_spin.config(to=max_day)
        try:
            day = int(self._day_var.get())
        except Exception:
            day = 1
            self._day_var.set(day)
        if day < 1:
            day = 1
            self._day_var.set(day)
        if day > max_day:
            self._day_var.set(max_day)

    def _update_selected_label(self) -> None:
        try:
            day = int(self._day_var.get())
            month = int(self._month_var.get())
            year = int(self._year_var.get())
            selected = dt.date(year, month, day)
        except Exception:
            return
        self._selected = selected
        self._selected_label.config(text=f"Выбрано: {selected.strftime('%d.%m.%Y')}")

    def _on_change(self, *_args) -> None:
        self._update_day_range()
        self._update_selected_label()

    def _apply(self) -> None:
        self._update_selected_label()
        if self._selected:
            self._on_select(self._selected)
        self.destroy()


def main(root: tk.Tk | None = None, parent: tk.Frame | None = None, *, embedded: bool = False) -> tk.Tk:
    calendar.setfirstweekday(calendar.MONDAY)
    _apply_theme(_detect_dark_mode())

    own_root = root is None
    if root is None:
        root = tk.Tk()

    if not embedded:
        root.title("Платформа прогнозирования чеков")
        root.configure(bg=COLOR_BG)
        _set_icon(root)

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        win_width = int(screen_width * 0.65)
        win_height = int(screen_height * 0.55)
        pos_x = (screen_width - win_width) // 2
        pos_y = (screen_height - win_height) // 3
        root.geometry(f"{win_width}x{win_height}+{pos_x}+{pos_y}")
        root.minsize(860, 420)

    container = parent if parent is not None else root

    status_var = tk.StringVar(value="Ожидание действий")
    progress_var = tk.StringVar(value="Прогресс: 0%")

    header = tk.Frame(container, bg=COLOR_RED)
    header.pack(fill="x", side="top")

    logo_img = _load_logo()
    if logo_img is not None:
        logo_label = tk.Label(header, image=logo_img, bg=COLOR_RED)
        logo_label.image = logo_img
        logo_label.pack(pady=8)

    title_border = tk.Frame(container, bg=COLOR_RED)
    title_border.pack(fill="x", padx=20, pady=(8, 4))

    title_bg = tk.Frame(title_border, bg=COLOR_BG)
    title_bg.pack(fill="x", padx=2, pady=2)

    title_label = tk.Label(
        title_bg,
        text="Платформа прогнозирования чеков",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 14, "bold"),
    )
    title_label.pack(padx=10, pady=6)

    main_frame = tk.Frame(container, bg=COLOR_BG, padx=18, pady=16)
    main_frame.pack(fill="both", expand=True)

    file_var = tk.StringVar()
    auto_checks = _try_pick_latest_checks()
    if auto_checks:
        file_var.set(str(auto_checks))
    ksss_options = _sort_ksss(_load_ksss_list())

    delimiter_var = tk.StringVar(value="|")
    selection_mode = tk.StringVar(value="all")
    range_start_var = tk.StringVar()
    range_end_var = tk.StringVar()
    selected_ksss: list[str] = []
    selected_ksss_var = tk.StringVar(value="Не выбраны")
    settings_summary_var = tk.StringVar()
    progress_file = BASE_DIR / ".forecast_progress.txt"
    progress_state = {"percent": 0.0, "active": False}
    progress_ui = {
        "window": None,
        "label": None,
        "canvas": None,
        "border": None,
        "bg": None,
        "fill": None,
        "btn_stop": None,
    }
    stop_file = BASE_DIR / ".forecast_stop.txt"
    result_dir_file = BASE_DIR / ".forecast_result_dir.txt"
    result_state = {"path": None}

    def _selection_summary(items: list[str]) -> str:
        if not items:
            return "Не выбраны"
        if len(items) <= 3:
            return ", ".join(items)
        return f"{len(items)} выбрано: {items[0]}, {items[1]}, {items[2]}..."

    def _build_settings_summary() -> str:
        parts = [f"Разделитель: {delimiter_var.get().strip() or '|'}"]
        mode = selection_mode.get()
        if mode == "range":
            parts.append(f"КССС: {range_start_var.get().strip()}–{range_end_var.get().strip()}")
        elif mode == "list":
            parts.append(f"КССС: {selected_ksss_var.get()}")
        else:
            parts.append("КССС: все")
        return " | ".join(parts)

    def refresh_settings_summary() -> None:
        settings_summary_var.set(_build_settings_summary())

    def _close_progress_window() -> None:
        window = progress_ui["window"]
        if window is not None and window.winfo_exists():
            try:
                window.destroy()
            except Exception:
                pass
        progress_ui["window"] = None
        progress_ui["label"] = None
        progress_ui["canvas"] = None
        progress_ui["border"] = None
        progress_ui["bg"] = None
        progress_ui["fill"] = None
        progress_ui["btn_stop"] = None

    def _ensure_progress_window() -> None:
        window = progress_ui["window"]
        if window is not None and window.winfo_exists():
            return

        window = tk.Toplevel(root)
        window.title("Прогресс прогноза")
        window.configure(bg=COLOR_BG)
        win_w = 460
        win_h = 180
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        pos_x = max(0, int((screen_w - win_w) / 2))
        pos_y = max(0, int((screen_h - win_h) / 2))
        window.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        window.resizable(False, False)
        window.transient(root)

        frame = tk.Frame(window, bg=COLOR_BG, padx=16, pady=14)
        frame.pack(fill="both", expand=True)

        label = tk.Label(
            frame,
            textvariable=progress_var,
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 11, "bold"),
        )
        label.pack(anchor="w")

        canvas = tk.Canvas(
            frame,
            height=18,
            bg=COLOR_BG,
            highlightthickness=0,
            bd=0,
        )
        canvas.pack(anchor="w", fill="x", pady=(8, 16))

        border_id = canvas.create_rectangle(0, 0, 1, 1, outline=COLOR_PROGRESS_BORDER, width=1)
        bg_id = canvas.create_rectangle(1, 1, 1, 1, outline="", fill=COLOR_BG)
        fill_id = canvas.create_rectangle(1, 1, 1, 1, outline="", fill=COLOR_PROGRESS)

        button_frame = tk.Frame(frame, bg=COLOR_BG)
        button_frame.pack(fill="x")

        btn_stop_local = tk.Button(
            button_frame,
            text="Остановить прогноз",
            command=_request_stop,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_BG,
            activeforeground=BUTTON_FG,
            relief="solid",
            bd=1,
            highlightbackground=BUTTON_BORDER,
            highlightcolor=BUTTON_BORDER,
            highlightthickness=1,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=4,
        )
        btn_stop_local.pack()

        def _on_close() -> None:
            _request_stop()
            _close_progress_window()

        window.protocol("WM_DELETE_WINDOW", _on_close)
        canvas.bind("<Configure>", lambda _event: _render_progress())

        progress_ui["window"] = window
        progress_ui["label"] = label
        progress_ui["canvas"] = canvas
        progress_ui["border"] = border_id
        progress_ui["bg"] = bg_id
        progress_ui["fill"] = fill_id
        progress_ui["btn_stop"] = btn_stop_local

    def _render_progress() -> None:
        percent = max(0.0, min(100.0, progress_state["percent"]))
        canvas = progress_ui["canvas"]
        if canvas is None or not canvas.winfo_exists():
            return
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 2 or h <= 2:
            return
        border_id = progress_ui["border"]
        bg_id = progress_ui["bg"]
        fill_id = progress_ui["fill"]
        if border_id is None or bg_id is None or fill_id is None:
            return
        canvas.coords(border_id, 0, 0, w, h)
        canvas.coords(bg_id, 1, 1, w - 1, h - 1)
        fill_w = int((w - 2) * (percent / 100.0))
        canvas.coords(fill_id, 1, 1, 1 + max(0, fill_w), h - 1)
        canvas.itemconfig(bg_id, fill=COLOR_BG)
        canvas.itemconfig(fill_id, fill=COLOR_PROGRESS)

    def _set_progress(done: int, total: int) -> None:
        if total <= 0:
            percent = 0.0
        else:
            percent = round((done / total) * 100)
        progress_state["percent"] = percent
        progress_var.set(f"Прогресс: {percent:.0f}% ({done} из {total})")
        _render_progress()

    def _poll_progress() -> None:
        if not progress_state["active"]:
            return
        if progress_file.exists():
            try:
                content = progress_file.read_text(encoding="utf-8").strip()
                if "/" in content:
                    done_str, total_str = content.split("/", 1)
                    done = int(done_str.strip())
                    total = int(total_str.strip())
                    _set_progress(done, total)
            except Exception:
                pass
        root.after(300, _poll_progress)

    def _start_progress() -> None:
        _ensure_progress_window()
        window = progress_ui["window"]
        if window is not None and window.winfo_exists():
            try:
                window.deiconify()
                window.lift()
            except Exception:
                pass
        btn_stop_local = progress_ui["btn_stop"]
        if btn_stop_local is not None and btn_stop_local.winfo_exists():
            btn_stop_local.config(state="normal")
        progress_state["active"] = True
        _set_progress(0, 0)
        _poll_progress()

    def _stop_progress() -> None:
        progress_state["active"] = False

    def _finish_progress(reason: str | None = None, error_msg: str | None = None) -> None:
        if progress_file.exists():
            try:
                content = progress_file.read_text(encoding="utf-8").strip()
                if "/" in content:
                    done_str, total_str = content.split("/", 1)
                    _set_progress(int(done_str.strip()), int(total_str.strip()))
            except Exception:
                pass
        btn_stop_local = progress_ui["btn_stop"]
        if btn_stop_local is not None and btn_stop_local.winfo_exists():
            btn_stop_local.config(state="disabled")
        _stop_progress()
        _close_progress_window()
        if reason == "stopped":
            messagebox.showwarning("Прогноз остановлен", "Прогнозирование было остановлено пользователем.")
        elif reason == "error":
            details = f"\n{error_msg}" if error_msg else ""
            messagebox.showerror("Ошибка", f"Прогнозирование завершилось с ошибкой.{details}")
        elif reason == "success":
            messagebox.showinfo(
                "Готово",
                "Прогнозирование завершено.\nФайлы результата сохранены в папках выгрузки.",
            )

    def _request_stop() -> None:
        if not progress_state["active"]:
            return
        try:
            stop_file.write_text("stop", encoding="utf-8")
        except Exception:
            pass
        status_var.set("Запрошена остановка прогноза...")
        btn_stop_local = progress_ui["btn_stop"]
        if btn_stop_local is not None and btn_stop_local.winfo_exists():
            btn_stop_local.config(state="disabled")

    def _open_results_folder() -> None:
        target = result_state["path"]
        if target is None and result_dir_file.exists():
            try:
                content = result_dir_file.read_text(encoding="utf-8").strip()
                if content:
                    target = Path(content)
            except Exception:
                target = None
        if target is None or not Path(target).exists():
            messagebox.showwarning("Папка не найдена", "Не удалось найти папку результата.")
            return
        target = Path(target)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            elif os.name == "nt":
                os.startfile(str(target))
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку:\n{exc}")

    def _refresh_result_button() -> None:
        target = None
        if result_dir_file.exists():
            try:
                content = result_dir_file.read_text(encoding="utf-8").strip()
                if content:
                    target = Path(content)
            except Exception:
                target = None
        if target is not None and target.exists():
            result_state["path"] = target
            btn_open_results.config(state="normal")
        else:
            result_state["path"] = None
            btn_open_results.config(state="disabled")

    def open_settings_popup() -> None:
        popup = tk.Toplevel(root)
        popup.title("Доп параметры выгрузки")
        popup.configure(bg=COLOR_BG)
        popup.transient(root)
        popup.grab_set()

        content = tk.Frame(popup, bg=COLOR_BG, padx=12, pady=12)
        content.pack(fill="both", expand=True)

        delimiter_label = tk.Label(
            content,
            text="Разделитель CSV:",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10),
        )
        delimiter_label.pack(anchor="w")

        delimiter_row = tk.Frame(content, bg=COLOR_BG)
        delimiter_row.pack(fill="x", pady=(4, 8))

        delimiter_entry = tk.Entry(
            delimiter_row,
            textvariable=delimiter_var,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            width=6,
            bg=COLOR_ENTRY_BG,
            fg=COLOR_ENTRY_FG,
            insertbackground=COLOR_ENTRY_FG,
        )
        delimiter_entry.pack(side="left", padx=(0, 6), ipadx=2, ipady=2)

        def set_delimiter(value: str) -> None:
            delimiter_var.set(value)

        for value in ("|", ";", ",", "\\t"):
            btn = tk.Button(
                delimiter_row,
                text=value,
                command=lambda v=value: set_delimiter(v),
                bg=BUTTON_BG,
                fg=BUTTON_FG,
                activebackground=BUTTON_BG,
                activeforeground=BUTTON_FG,
                relief="solid",
                bd=1,
                highlightbackground=BUTTON_BORDER,
                highlightcolor=BUTTON_BORDER,
                highlightthickness=1,
                font=("Segoe UI", 9, "bold"),
                padx=6,
                pady=2,
            )
            btn.pack(side="left", padx=(4, 0))

        delimiter_hint = tk.Label(
            content,
            text="Можно указать любой разделитель. Для табуляции используйте \\t.",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 9),
        )
        delimiter_hint.pack(anchor="w", pady=(0, 10))

        ksss_label = tk.Label(
            content,
            text="Ограничение АЗС для прогноза (опционально)",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        ksss_label.pack(anchor="w")


        selection_frame = tk.Frame(content, bg=COLOR_BG)
        selection_frame.pack(fill="x", pady=(6, 8))

        radio_all = tk.Radiobutton(
            selection_frame,
            text="Все объекты",
            variable=selection_mode,
            value="all",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10),
        )
        radio_all.pack(anchor="w")

        range_row = tk.Frame(selection_frame, bg=COLOR_BG)
        range_row.pack(fill="x", pady=(4, 0))
        radio_range = tk.Radiobutton(
            range_row,
            text="Диапазон КССС:",
            variable=selection_mode,
            value="range",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10),
        )
        radio_range.pack(side="left")

        range_start_entry = tk.Entry(
            range_row,
            textvariable=range_start_var,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            width=10,
            bg=COLOR_ENTRY_BG,
            fg=COLOR_ENTRY_FG,
            insertbackground=COLOR_ENTRY_FG,
        )
        range_start_entry.pack(side="left", padx=(8, 4), ipadx=2, ipady=2)

        range_sep = tk.Label(
            range_row,
            text="до",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10),
        )
        range_sep.pack(side="left")

        range_end_entry = tk.Entry(
            range_row,
            textvariable=range_end_var,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            width=10,
            bg=COLOR_ENTRY_BG,
            fg=COLOR_ENTRY_FG,
            insertbackground=COLOR_ENTRY_FG,
        )
        range_end_entry.pack(side="left", padx=(6, 0), ipadx=2, ipady=2)

        list_row = tk.Frame(selection_frame, bg=COLOR_BG)
        list_row.pack(fill="x", pady=(6, 0))
        radio_list = tk.Radiobutton(
            list_row,
            text="Выбор из списка КССС:",
            variable=selection_mode,
            value="list",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10),
        )
        radio_list.pack(side="left")

        def open_ksss_picker() -> None:
            if not ksss_options:
                messagebox.showwarning("Нет данных", "Не удалось загрузить список КССС из файла.")
                return

            list_popup = tk.Toplevel(popup)
            list_popup.title("Выбор КССС")
            list_popup.configure(bg=COLOR_BG)
            list_popup.transient(popup)
            list_popup.grab_set()

            frame = tk.Frame(list_popup, bg=COLOR_BG, padx=10, pady=10)
            frame.pack(fill="both", expand=True)

            search_row = tk.Frame(frame, bg=COLOR_BG)
            search_row.pack(fill="x", pady=(0, 6))

            search_label = tk.Label(
                search_row,
                text="Поиск:",
                bg=COLOR_BG,
                fg=COLOR_TEXT,
                font=("Segoe UI", 10),
            )
            search_label.pack(side="left")

            search_var = tk.StringVar()
            search_entry = tk.Entry(
                search_row,
                textvariable=search_var,
                font=("Segoe UI", 10),
                relief="solid",
                bd=1,
                bg=COLOR_ENTRY_BG,
                fg=COLOR_ENTRY_FG,
                insertbackground=COLOR_ENTRY_FG,
                width=18,
            )
            search_entry.pack(side="left", padx=(8, 0), ipadx=2, ipady=2)

            def clear_search() -> None:
                search_var.set("")

            clear_btn = tk.Button(
                search_row,
                text="Очистить",
                command=clear_search,
                bg=BUTTON_BG,
                fg=BUTTON_FG,
                activebackground=BUTTON_BG,
                activeforeground=BUTTON_FG,
                relief="solid",
                bd=1,
                highlightbackground=BUTTON_BORDER,
                highlightcolor=BUTTON_BORDER,
                highlightthickness=1,
                font=("Segoe UI", 9, "bold"),
                padx=8,
                pady=2,
            )
            clear_btn.pack(side="left", padx=(8, 0))

            listbox = tk.Listbox(
                frame,
                selectmode="extended",
                height=12,
                width=30,
                bg=COLOR_ENTRY_BG,
                fg=COLOR_ENTRY_FG,
                selectbackground=COLOR_RED,
                selectforeground="white",
            )
            listbox.pack(side="left", fill="both", expand=True)

            scrollbar = tk.Scrollbar(frame, orient="vertical", command=listbox.yview)
            scrollbar.pack(side="left", fill="y")
            listbox.config(yscrollcommand=scrollbar.set)

            selected_set = set(selected_ksss)
            displayed_items: list[str] = []

            def refresh_list() -> None:
                query = search_var.get().strip().lower()
                listbox.delete(0, "end")
                displayed_items.clear()
                for item in ksss_options:
                    if query and query not in item.lower():
                        continue
                    displayed_items.append(item)
                    listbox.insert("end", item)
                for idx, item in enumerate(displayed_items):
                    if item in selected_set:
                        listbox.selection_set(idx)

            def on_select(_event=None) -> None:
                chosen = {displayed_items[i] for i in listbox.curselection()}
                for item in displayed_items:
                    if item in selected_set and item not in chosen:
                        selected_set.remove(item)
                for item in chosen:
                    selected_set.add(item)

            listbox.bind("<<ListboxSelect>>", on_select)
            search_var.trace_add("write", lambda *_: refresh_list())
            refresh_list()

            btn_frame = tk.Frame(list_popup, bg=COLOR_BG)
            btn_frame.pack(fill="x", padx=10, pady=(0, 10))

            def select_all() -> None:
                for idx in range(len(displayed_items)):
                    listbox.selection_set(idx)
                selected_set.update(displayed_items)

            def select_none() -> None:
                listbox.selection_clear(0, "end")
                for item in displayed_items:
                    selected_set.discard(item)

            btn_all = tk.Button(
                btn_frame,
                text="Выбрать все",
                command=select_all,
                bg=BUTTON_BG,
                fg=BUTTON_FG,
                activebackground=BUTTON_BG,
                activeforeground=BUTTON_FG,
                relief="solid",
                bd=1,
                highlightbackground=BUTTON_BORDER,
                highlightcolor=BUTTON_BORDER,
                highlightthickness=1,
                font=("Segoe UI", 10, "bold"),
                padx=10,
                pady=4,
            )
            btn_all.pack(side="left")

            btn_none = tk.Button(
                btn_frame,
                text="Снять все",
                command=select_none,
                bg=BUTTON_BG,
                fg=BUTTON_FG,
                activebackground=BUTTON_BG,
                activeforeground=BUTTON_FG,
                relief="solid",
                bd=1,
                highlightbackground=BUTTON_BORDER,
                highlightcolor=BUTTON_BORDER,
                highlightthickness=1,
                font=("Segoe UI", 10, "bold"),
                padx=10,
                pady=4,
            )
            btn_none.pack(side="left", padx=(8, 0))

            def apply_selection() -> None:
                on_select()
                chosen = [item for item in ksss_options if item in selected_set]
                selected_ksss.clear()
                selected_ksss.extend(chosen)
                selected_ksss_var.set(_selection_summary(selected_ksss))
                list_popup.destroy()

            btn_ok = tk.Button(
                btn_frame,
                text="Применить",
                command=apply_selection,
                bg=BUTTON_BG,
                fg=BUTTON_FG,
                activebackground=BUTTON_BG,
                activeforeground=BUTTON_FG,
                relief="solid",
                bd=1,
                highlightbackground=BUTTON_BORDER,
                highlightcolor=BUTTON_BORDER,
                highlightthickness=1,
                font=("Segoe UI", 10, "bold"),
                padx=10,
                pady=4,
            )
            btn_ok.pack(side="left", padx=(8, 0))

            btn_cancel = tk.Button(
                btn_frame,
                text="Отмена",
                command=list_popup.destroy,
                bg=BUTTON_BG,
                fg=BUTTON_FG,
                activebackground=BUTTON_BG,
                activeforeground=BUTTON_FG,
                relief="solid",
                bd=1,
                highlightbackground=BUTTON_BORDER,
                highlightcolor=BUTTON_BORDER,
                highlightthickness=1,
                font=("Segoe UI", 10, "bold"),
                padx=10,
                pady=4,
            )
            btn_cancel.pack(side="left", padx=(8, 0))

        pick_list_btn = tk.Button(
            list_row,
            text="Выбрать...",
            command=open_ksss_picker,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_BG,
            activeforeground=BUTTON_FG,
            relief="solid",
            bd=1,
            highlightbackground=BUTTON_BORDER,
            highlightcolor=BUTTON_BORDER,
            highlightthickness=1,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=3,
        )
        pick_list_btn.pack(side="left", padx=(8, 0))

        selected_label = tk.Label(
            selection_frame,
            textvariable=selected_ksss_var,
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 9),
        )
        selected_label.pack(anchor="w", pady=(4, 0))

        def update_selection_mode(*_args) -> None:
            state_range = "normal" if selection_mode.get() == "range" else "disabled"
            state_list = "normal" if selection_mode.get() == "list" else "disabled"
            range_start_entry.config(state=state_range)
            range_end_entry.config(state=state_range)
            pick_list_btn.config(state=state_list)

        selection_mode.trace_add("write", update_selection_mode)
        update_selection_mode()

        btn_frame = tk.Frame(content, bg=COLOR_BG)
        btn_frame.pack(fill="x", pady=(8, 0))

        def close_popup() -> None:
            refresh_settings_summary()
            popup.destroy()

        btn_close = tk.Button(
            btn_frame,
            text="Готово",
            command=close_popup,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_BG,
            activeforeground=BUTTON_FG,
            relief="solid",
            bd=1,
            highlightbackground=BUTTON_BORDER,
            highlightcolor=BUTTON_BORDER,
            highlightthickness=1,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=4,
        )
        btn_close.pack(side="left")

        popup.protocol("WM_DELETE_WINDOW", close_popup)

    refresh_settings_summary()

    file_label = tk.Label(
        main_frame,
        text="1. Выберите файл с чеками (CSV)",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 11),
    )
    file_label.pack(anchor="w")

    file_frame = tk.Frame(main_frame, bg=COLOR_BG)
    file_frame.pack(fill="x", pady=(5, 12))

    file_entry = tk.Entry(
        file_frame,
        textvariable=file_var,
        font=("Segoe UI", 10),
        relief="solid",
        bd=1,
        bg=COLOR_ENTRY_BG,
        fg=COLOR_ENTRY_FG,
        insertbackground=COLOR_ENTRY_FG,
    )
    file_entry.pack(side="left", fill="x", expand=True, ipadx=4, ipady=3)

    def choose_file() -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл с чеками",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
            initialdir=str(DEFAULT_CHECKS_ROOT) if DEFAULT_CHECKS_ROOT.exists() else str(BASE_DIR),
        )
        if path:
            file_var.set(path)
            status_var.set("Файл выбран, можно задать период")

    btn_file = tk.Button(
        file_frame,
        text="Обзор...",
        command=choose_file,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        activebackground=BUTTON_BG,
        activeforeground=BUTTON_FG,
        relief="solid",
        bd=1,
        highlightbackground=BUTTON_BORDER,
        highlightcolor=BUTTON_BORDER,
        highlightthickness=1,
        font=("Segoe UI", 10, "bold"),
        padx=10,
        pady=3,
    )
    btn_file.pack(side="left", padx=(8, 0))

    params_label = tk.Label(
        main_frame,
        text="2. Доп параметры выгрузки",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 11),
    )
    params_label.pack(anchor="w", pady=(8, 0))

    params_btn = tk.Button(
        main_frame,
        text="Открыть параметры",
        command=open_settings_popup,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        activebackground=BUTTON_BG,
        activeforeground=BUTTON_FG,
        relief="solid",
        bd=1,
        highlightbackground=BUTTON_BORDER,
        highlightcolor=BUTTON_BORDER,
        highlightthickness=1,
        font=("Segoe UI", 10, "bold"),
        padx=10,
        pady=4,
    )
    params_btn.pack(anchor="w", pady=(4, 4))

    params_summary = tk.Label(
        main_frame,
        textvariable=settings_summary_var,
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 9),
    )
    params_summary.pack(anchor="w", pady=(0, 10))

    period_label = tk.Label(
        main_frame,
        text="3. Период прогноза (включительно)",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 11),
    )
    period_label.pack(anchor="w")

    period_frame = tk.Frame(main_frame, bg=COLOR_BG)
    period_frame.pack(fill="x", pady=(5, 10))

    start_var = tk.StringVar()
    end_var = tk.StringVar()
    today = dt.date.today()
    start_var.set(_format_date(today))
    end_var.set(_format_date(_add_month(today, 1)))

    start_entry = tk.Entry(
        period_frame,
        textvariable=start_var,
        font=("Segoe UI", 10),
        relief="solid",
        bd=1,
        width=16,
        bg=COLOR_ENTRY_BG,
        fg=COLOR_ENTRY_FG,
        insertbackground=COLOR_ENTRY_FG,
    )
    start_entry.pack(side="left", padx=(0, 6), ipadx=4, ipady=3)

    def pick_start() -> None:
        initial = _parse_date(start_var.get()) or dt.date.today()
        selected = _parse_date(start_var.get())
        DatePickerPopup(root, initial, lambda d: start_var.set(_format_date(d)), selected_date=selected)

    btn_start = tk.Button(
        period_frame,
        text="Календарь",
        command=pick_start,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        activebackground=BUTTON_BG,
        activeforeground=BUTTON_FG,
        relief="solid",
        bd=1,
        highlightbackground=BUTTON_BORDER,
        highlightcolor=BUTTON_BORDER,
        highlightthickness=1,
        font=("Segoe UI", 10, "bold"),
        padx=8,
        pady=3,
    )
    btn_start.pack(side="left", padx=(0, 12))

    end_entry = tk.Entry(
        period_frame,
        textvariable=end_var,
        font=("Segoe UI", 10),
        relief="solid",
        bd=1,
        width=16,
        bg=COLOR_ENTRY_BG,
        fg=COLOR_ENTRY_FG,
        insertbackground=COLOR_ENTRY_FG,
    )
    end_entry.pack(side="left", padx=(0, 6), ipadx=4, ipady=3)

    def pick_end() -> None:
        initial = _parse_date(end_var.get()) or dt.date.today()
        selected = _parse_date(end_var.get())
        DatePickerPopup(root, initial, lambda d: end_var.set(_format_date(d)), selected_date=selected)

    btn_end = tk.Button(
        period_frame,
        text="Календарь",
        command=pick_end,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        activebackground=BUTTON_BG,
        activeforeground=BUTTON_FG,
        relief="solid",
        bd=1,
        highlightbackground=BUTTON_BORDER,
        highlightcolor=BUTTON_BORDER,
        highlightthickness=1,
        font=("Segoe UI", 10, "bold"),
        padx=8,
        pady=3,
    )
    btn_end.pack(side="left")

    hint_label = tk.Label(
        main_frame,
        text="Формат даты: YYYY-MM-DD или ДД.ММ.ГГГГ",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 9),
    )
    hint_label.pack(anchor="w", pady=(0, 12))

    def run_forecast() -> None:
        path = file_var.get().strip()
        if not path:
            messagebox.showwarning("Нет файла", "Выберите файл с чеками.")
            return
        if not Path(path).exists():
            messagebox.showwarning("Файл не найден", "Проверьте путь к файлу.")
            return

        start_date = _parse_date(start_var.get())
        end_date = _parse_date(end_var.get())
        if start_date is None or end_date is None:
            messagebox.showwarning("Неверная дата", "Введите корректный период прогноза.")
            return
        if end_date < start_date:
            messagebox.showwarning("Неверный период", "Дата окончания меньше даты начала.")
            return
        if (start_date.year, start_date.month) != (end_date.year, end_date.month):
            ok = messagebox.askyesno(
                "Период больше месяца",
                "Алгоритм рассчитан на прогноз в пределах одного месяца.\n"
                "Продолжить с выбранным периодом?",
            )
            if not ok:
                return

        delimiter_value = delimiter_var.get().strip()
        if not delimiter_value:
            delimiter_value = "|"


        mode = selection_mode.get()
        range_start = range_start_var.get().strip()
        range_end = range_end_var.get().strip()
        if mode == "range" and (not range_start or not range_end):
            messagebox.showwarning("Диапазон не задан", "Укажите оба значения диапазона КССС.")
            return
        if mode == "list" and not selected_ksss:
            messagebox.showwarning("Нет объектов", "Выберите КССС из списка или переключитесь на другой режим.")
            return

        try:
            if progress_file.exists():
                progress_file.unlink()
        except Exception:
            pass
        try:
            if stop_file.exists():
                stop_file.unlink()
        except Exception:
            pass
        try:
            if result_dir_file.exists():
                result_dir_file.unlink()
        except Exception:
            pass
        _start_progress()
        btn_run.config(state="disabled")
        btn_open_results.config(state="disabled")

        def task() -> None:
            finish_reason = "success"
            error_text = None
            try:
                root.after(0, lambda: status_var.set("Идёт прогнозирование..."))
                _suppress_statsmodels_warnings()

                os.environ["CHECKS_PATH_OVERRIDE"] = str(Path(path))
                os.environ["FORECAST_START"] = _format_date(start_date)
                os.environ["FORECAST_END"] = _format_date(end_date)
                os.environ["CHECKS_DELIMITER"] = delimiter_value
                os.environ["PROGRESS_FILE"] = str(progress_file)
                os.environ["STOP_FILE"] = str(stop_file)
                os.environ["RESULT_DIR_FILE"] = str(result_dir_file)
                if mode == "range":
                    os.environ["KSSS_RANGE_START"] = range_start
                    os.environ["KSSS_RANGE_END"] = range_end
                elif mode == "list":
                    os.environ["KSSS_LIST"] = ",".join(selected_ksss)

                runpy.run_path(str(FORECAST_SCRIPT), run_name="__main__")

                finish_reason = "stopped" if stop_file.exists() else "success"
                if finish_reason == "stopped":
                    root.after(0, lambda: status_var.set("Прогноз остановлен"))
                else:
                    root.after(0, lambda: status_var.set("Готово"))
            except Exception as exc:
                traceback.print_exc()
                finish_reason = "error"
                error_text = str(exc)
                root.after(0, lambda: status_var.set("Ошибка при выполнении"))
            finally:
                root.after(0, lambda: _finish_progress(finish_reason, error_text))
                root.after(0, _refresh_result_button)
                root.after(0, lambda: btn_run.config(state="normal"))
                os.environ.pop("CHECKS_PATH_OVERRIDE", None)
                os.environ.pop("FORECAST_START", None)
                os.environ.pop("FORECAST_END", None)
                os.environ.pop("CHECKS_DELIMITER", None)
                os.environ.pop("PROGRESS_FILE", None)
                os.environ.pop("STOP_FILE", None)
                os.environ.pop("RESULT_DIR_FILE", None)
                os.environ.pop("KSSS_RANGE_START", None)
                os.environ.pop("KSSS_RANGE_END", None)
                os.environ.pop("KSSS_LIST", None)
                try:
                    if stop_file.exists():
                        stop_file.unlink()
                except Exception:
                    pass

        threading.Thread(target=task, daemon=True).start()

    btn_run = tk.Button(
        main_frame,
        text="4. Запустить прогноз",
        command=run_forecast,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        activebackground=BUTTON_BG,
        activeforeground=BUTTON_FG,
        relief="solid",
        bd=1,
        highlightbackground=BUTTON_BORDER,
        highlightcolor=BUTTON_BORDER,
        highlightthickness=1,
        font=("Segoe UI", 11, "bold"),
        padx=14,
        pady=6,
    )
    btn_run.pack(anchor="w", pady=(0, 6))

    status_label = tk.Label(
        main_frame,
        textvariable=status_var,
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 10, "italic"),
    )
    status_label.pack(anchor="w", pady=(0, 6))

    btn_open_results = tk.Button(
        main_frame,
        text="Открыть папку результата",
        command=_open_results_folder,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        activebackground=BUTTON_BG,
        activeforeground=BUTTON_FG,
        relief="solid",
        bd=1,
        highlightbackground=BUTTON_BORDER,
        highlightcolor=BUTTON_BORDER,
        highlightthickness=1,
        font=("Segoe UI", 10, "bold"),
        padx=12,
        pady=4,
        state="disabled",
    )
    results_label = tk.Label(
        main_frame,
        text="Результаты",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 11),
    )
    results_label.pack(anchor="w", pady=(10, 2))

    btn_open_results.pack(anchor="w", pady=(0, 0))

    if own_root:
        root.mainloop()
    return root


if __name__ == "__main__":
    main()
