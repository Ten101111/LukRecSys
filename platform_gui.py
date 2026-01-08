# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk

BASE_DIR = Path(__file__).resolve().parent
PROCESS_DIR = BASE_DIR / "Обработка чеков"
FORECAST_DIR = BASE_DIR / "ИИ прогнозирование Чеков"
CHECKS_DIR = BASE_DIR / "Проверка чеков"


def _add_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить модуль: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _set_icon(root: tk.Tk) -> None:
    candidates = [
        BASE_DIR / "lukoil.ico",
        BASE_DIR / "lukoil.ico.ico",
        PROCESS_DIR / "lukoil.ico",
        PROCESS_DIR / "lukoil.ico.ico",
        FORECAST_DIR / "lukoil.ico",
        FORECAST_DIR / "lukoil.ico.ico",
        CHECKS_DIR / "lukoil.ico",
        CHECKS_DIR / "lukoil.ico.ico",
    ]
    for path in candidates:
        if path.exists():
            try:
                root.iconbitmap(path)
                break
            except Exception:
                continue


def main() -> None:
    _add_sys_path(PROCESS_DIR)
    _add_sys_path(FORECAST_DIR)

    process_gui = _load_module("process_platform_gui", PROCESS_DIR / "platform_gui.py")
    forecast_gui = _load_module("forecast_platform_gui", FORECAST_DIR / "platform_gui.py")

    root = tk.Tk()
    root.title("Платформа обработки и прогнозирования чеков")
    root.configure(bg="#ffffff")
    _set_icon(root)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    win_width = int(screen_width * 0.8)
    win_height = int(screen_height * 0.75)
    pos_x = (screen_width - win_width) // 2
    pos_y = (screen_height - win_height) // 3
    root.geometry(f"{win_width}x{win_height}+{pos_x}+{pos_y}")
    root.minsize(1100, 650)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    tab_processing = tk.Frame(notebook, bg="#ffffff")
    tab_forecast = tk.Frame(notebook, bg="#ffffff")
    notebook.add(tab_processing, text="Обработка чеков")
    notebook.add(tab_forecast, text="Прогнозирование")

    process_gui.main(root=root, parent=tab_processing, embedded=True)
    tab_processing.configure(bg=getattr(process_gui, "COLOR_BG", "#ffffff"))

    forecast_gui.main(root=root, parent=tab_forecast, embedded=True)
    tab_forecast.configure(bg=getattr(forecast_gui, "COLOR_BG", "#ffffff"))

    root.mainloop()


if __name__ == "__main__":
    main()
