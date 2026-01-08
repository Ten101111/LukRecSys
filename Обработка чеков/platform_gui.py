import datetime as dt
import subprocess
import sys
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage, ttk

from constants_functions import pick_latest_file, pick_latest_folder
from day_night_checks_function import update_by_day_night
from new_checks_add import update_by_day

COLOR_RED = "#ed1b34"
COLOR_BG = "#ffffff"
COLOR_TEXT = "#000000"
COLOR_ENTRY_BG = "#ffffff"
COLOR_ENTRY_FG = "#000000"
BUTTON_BG = "#ffffff"
BUTTON_FG = "#000000"
BUTTON_BORDER = "#000000"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "Чеки"
DEFAULT_DAY_ROOT = BASE_DIR / "Объединение чеков"
DEFAULT_DN_ROOT = BASE_DIR / "Доля чеков дневных и ночных"

DEFAULT_MASTER_DAY_NAME = "Объединение чеков по дням_master.csv"
DEFAULT_MASTER_DN_NAME = "День и Ночь_master.csv"


def _date_stamp(now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now()
    return f"{now.day}.{now.month}.{now.year}"


def _default_out_dirs() -> tuple[Path, Path]:
    stamp = _date_stamp()
    day_out = DEFAULT_DAY_ROOT / f"Объединенные чеки от {stamp}"
    dn_out = DEFAULT_DN_ROOT / f"Абсолютные значения от {stamp}"
    return day_out, dn_out


def _try_pick_latest(base_root: Path) -> Path | None:
    try:
        latest_folder = pick_latest_folder(base_root)
        return pick_latest_file(latest_folder)
    except Exception:
        return None


def _files_summary(files: list[str]) -> str:
    if not files:
        return "Файлы не выбраны"
    if len(files) == 1:
        return files[0]
    first = Path(files[0]).name
    return f"{len(files)} файла(ов): {first} ..."


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
    global COLOR_BG, COLOR_TEXT, COLOR_ENTRY_BG, COLOR_ENTRY_FG
    if dark:
        COLOR_BG = "#1f1f1f"
        COLOR_TEXT = "#f2f2f2"
        COLOR_ENTRY_BG = "#2b2b2b"
        COLOR_ENTRY_FG = "#f2f2f2"
    else:
        COLOR_BG = "#ffffff"
        COLOR_TEXT = "#000000"
        COLOR_ENTRY_BG = "#ffffff"
        COLOR_ENTRY_FG = "#000000"


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


def main(root: tk.Tk | None = None, parent: tk.Frame | None = None, *, embedded: bool = False) -> tk.Tk:
    _apply_theme(_detect_dark_mode())

    own_root = root is None
    if root is None:
        root = tk.Tk()

    if not embedded:
        root.title("Платформа обработки чеков")
        root.configure(bg=COLOR_BG)
        _set_icon(root)

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        win_width = int(screen_width * 0.7)
        win_height = int(screen_height * 0.5)
        pos_x = (screen_width - win_width) // 2
        pos_y = (screen_height - win_height) // 3
        root.geometry(f"{win_width}x{win_height}+{pos_x}+{pos_y}")
        root.minsize(900, 450)

    container = parent if parent is not None else root
    container.configure(bg=COLOR_BG)

    status_var = tk.StringVar(value="Ожидание действий")

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
        text="Платформа обработки чеков",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 14, "bold"),
    )
    title_label.pack(padx=10, pady=6)

    main_frame = tk.Frame(container, bg=COLOR_BG, padx=15, pady=15)
    main_frame.pack(fill="both", expand=True)

    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill="both", expand=True)

    tab_update = tk.Frame(notebook, bg=COLOR_BG)
    tab_create = tk.Frame(notebook, bg=COLOR_BG)
    notebook.add(tab_update, text="Обновить мастер")
    notebook.add(tab_create, text="Создать мастер с нуля")

    def build_files_block(parent: tk.Frame, title: str) -> tuple[callable, callable]:
        files: list[str] = []
        files_var = tk.StringVar(value="Файлы не выбраны")

        lbl = tk.Label(
            parent,
            text=title,
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 11),
            wraplength=1100,
            justify="left",
        )
        lbl.pack(anchor="w")

        frame = tk.Frame(parent, bg=COLOR_BG)
        frame.pack(fill="x", pady=(5, 10))

        entry = tk.Entry(
            frame,
            textvariable=files_var,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            bg=COLOR_ENTRY_BG,
            fg=COLOR_ENTRY_FG,
            insertbackground=COLOR_ENTRY_FG,
        )
        entry.pack(side="left", fill="x", expand=True, ipadx=4, ipady=3)

        def set_files(new_files: list[str]) -> None:
            files.clear()
            files.extend(new_files)
            files_var.set(_files_summary(files))

        def choose_files() -> None:
            paths = filedialog.askopenfilenames(
                title="Выберите CSV файлы с чеками",
                filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
                initialdir=str(DEFAULT_INPUT_DIR) if DEFAULT_INPUT_DIR.exists() else str(BASE_DIR),
            )
            if paths:
                set_files(list(paths))
                status_var.set("Файлы выбраны, можно запускать обработку")

        def choose_folder() -> None:
            folder = filedialog.askdirectory(
                title="Выберите папку с CSV файлами",
                initialdir=str(DEFAULT_INPUT_DIR) if DEFAULT_INPUT_DIR.exists() else str(BASE_DIR),
            )
            if folder:
                csvs = sorted(str(p) for p in Path(folder).glob("*.csv"))
                set_files(csvs)
                status_var.set("Папка выбрана, можно запускать обработку")

        def clear_files() -> None:
            set_files([])
            status_var.set("Список файлов очищен")

        btn_browse = tk.Button(
            frame,
            text="Обзор...",
            command=choose_files,
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
        btn_browse.pack(side="left", padx=(8, 0))

        btn_folder = tk.Button(
            frame,
            text="Папка...",
            command=choose_folder,
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
        btn_folder.pack(side="left", padx=(8, 0))

        btn_clear = tk.Button(
            frame,
            text="Очистить",
            command=clear_files,
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
        btn_clear.pack(side="left", padx=(8, 0))

        def get_files() -> list[str]:
            return list(files)

        return get_files, set_files

    def build_path_block(
        parent: tk.Frame,
        title: str,
        initial_value: str,
        file_mode: bool = True,
        auto_value: Path | None = None,
        save_mode: bool = False,
    ) -> tk.StringVar:
        var = tk.StringVar(value=initial_value)

        lbl = tk.Label(
            parent,
            text=title,
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Segoe UI", 11),
        )
        lbl.pack(anchor="w")

        frame = tk.Frame(parent, bg=COLOR_BG)
        frame.pack(fill="x", pady=(5, 10))

        entry = tk.Entry(
            frame,
            textvariable=var,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            bg=COLOR_ENTRY_BG,
            fg=COLOR_ENTRY_FG,
            insertbackground=COLOR_ENTRY_FG,
        )
        entry.pack(side="left", fill="x", expand=True, ipadx=4, ipady=3)

        def choose_file() -> None:
            dialog = filedialog.asksaveasfilename if save_mode else filedialog.askopenfilename
            path = dialog(
                title=title,
                filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
            )
            if path:
                var.set(path)

        def choose_folder() -> None:
            path = filedialog.askdirectory(title=title)
            if path:
                var.set(path)

        def set_auto() -> None:
            if auto_value is None:
                messagebox.showwarning("Автовыбор недоступен", "Не удалось найти подходящий файл.")
                return
            var.set(str(auto_value))

        btn_browse = tk.Button(
            frame,
            text="Обзор...",
            command=choose_file if file_mode else choose_folder,
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
        btn_browse.pack(side="left", padx=(8, 0))

        if file_mode and auto_value is not None:
            btn_auto = tk.Button(
                frame,
                text="Авто",
                command=set_auto,
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
            btn_auto.pack(side="left", padx=(8, 0))

        return var

    day_out_default, dn_out_default = _default_out_dirs()

    update_files_get, _ = build_files_block(
        tab_update,
        "1. Выберите один или несколько CSV файлов с чеками",
    )

    auto_master_day = _try_pick_latest(DEFAULT_DAY_ROOT)
    auto_master_dn = _try_pick_latest(DEFAULT_DN_ROOT)

    master_day_var = build_path_block(
        tab_update,
        "2. Мастер-файл по дням",
        str(auto_master_day) if auto_master_day else "",
        file_mode=True,
        auto_value=auto_master_day,
    )

    master_dn_var = build_path_block(
        tab_update,
        "3. Мастер-файл по дням/ночам",
        str(auto_master_dn) if auto_master_dn else "",
        file_mode=True,
        auto_value=auto_master_dn,
    )

    update_options = tk.Frame(tab_update, bg=COLOR_BG)
    update_options.pack(fill="x", pady=(0, 8))

    write_master_day = tk.BooleanVar(value=True)
    write_master_dn = tk.BooleanVar(value=True)

    chk_day = tk.Checkbutton(
        update_options,
        text="Обновлять мастер по дням",
        variable=write_master_day,
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        activebackground=COLOR_BG,
        activeforeground=COLOR_TEXT,
        selectcolor=COLOR_BG,
        font=("Segoe UI", 10),
    )
    chk_day.pack(anchor="w")

    chk_dn = tk.Checkbutton(
        update_options,
        text="Обновлять мастер по дням/ночам",
        variable=write_master_dn,
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        activebackground=COLOR_BG,
        activeforeground=COLOR_TEXT,
        selectcolor=COLOR_BG,
        font=("Segoe UI", 10),
    )
    chk_dn.pack(anchor="w")

    def run_update() -> None:
        files = update_files_get()
        if not files:
            messagebox.showwarning("Нет файлов", "Сначала выберите CSV файлы с чеками.")
            return

        master_day = master_day_var.get().strip()
        master_dn = master_dn_var.get().strip()
        if not master_day or not master_dn:
            messagebox.showwarning(
                "Нет мастер-файла",
                "Укажите мастер-файлы или используйте вкладку «Создать мастер с нуля».",
            )
            return

        if not Path(master_day).exists():
            ok = messagebox.askyesno(
                "Мастер по дням не найден",
                "Файл не найден. Создать новый мастер по дням?",
            )
            if not ok:
                return

        if not Path(master_dn).exists():
            ok = messagebox.askyesno(
                "Мастер по дням/ночам не найден",
                "Файл не найден. Создать новый мастер по дням/ночам?",
            )
            if not ok:
                return

        out_day, out_dn = _default_out_dirs()

        try:
            status_var.set("Идёт обработка чеков...")
            root.update_idletasks()

            update_by_day(
                input_dir=str(DEFAULT_INPUT_DIR),
                out_dir=str(out_day),
                master_filename=master_day,
                unique=True,
                append_only_missing=True,
                mode="manual",
                files=files,
                skip_if_in_state=False,
                write_master=write_master_day.get(),
            )

            update_by_day_night(
                input_dir=str(DEFAULT_INPUT_DIR),
                out_dir=str(out_dn),
                master_filename=master_dn,
                unique=True,
                append_only_missing=True,
                mode="manual",
                files=files,
                skip_if_in_state=False,
                write_master=write_master_dn.get(),
            )

            messagebox.showinfo(
                "Готово",
                "Обновление завершено.\n\n"
                f"Объединенные чеки по дням:\n{out_day}\n\n"
                f"Чеки день/ночь:\n{out_dn}",
            )
            status_var.set("Готово (обновление мастера)")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Во время выполнения произошла ошибка:\n{exc}")
            status_var.set("Ошибка при обновлении мастера")

    btn_update = tk.Button(
        tab_update,
        text="4. Запустить обновление",
        command=run_update,
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
        padx=15,
        pady=6,
    )
    btn_update.pack(anchor="w", pady=(0, 10))

    info_update = tk.Label(
        tab_update,
        text=(
            "Результат:\n"
            "• Файлы с объединенными чеками по дням\n"
            "• Файлы с чеками в разрезе дня/ночи\n"
            "• По желанию обновляются мастер-файлы\n"
            "• Сохранение идёт по установленному правилу"
        ),
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 10),
        justify="left",
    )
    info_update.pack(anchor="w")

    create_files_get, _ = build_files_block(
        tab_create,
        "1. Выберите CSV файлы, из которых нужно собрать новый мастер",
    )

    create_master_day_var = build_path_block(
        tab_create,
        "2. Новый мастер по дням",
        str((day_out_default / DEFAULT_MASTER_DAY_NAME)),
        file_mode=True,
        auto_value=None,
        save_mode=True,
    )

    create_master_dn_var = build_path_block(
        tab_create,
        "3. Новый мастер по дням/ночам",
        str((dn_out_default / DEFAULT_MASTER_DN_NAME)),
        file_mode=True,
        auto_value=None,
        save_mode=True,
    )

    def set_create_defaults() -> None:
        day_dir, dn_dir = _default_out_dirs()
        create_master_day_var.set(str(day_dir / DEFAULT_MASTER_DAY_NAME))
        create_master_dn_var.set(str(dn_dir / DEFAULT_MASTER_DN_NAME))

    btn_defaults = tk.Button(
        tab_create,
        text="Заполнить мастер-файлы по умолчанию",
        command=set_create_defaults,
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
        padx=15,
        pady=6,
    )
    btn_defaults.pack(anchor="w", pady=(0, 10))

    def run_create() -> None:
        files = create_files_get()
        if not files:
            messagebox.showwarning("Нет файлов", "Сначала выберите CSV файлы с чеками.")
            return

        master_day = create_master_day_var.get().strip()
        master_dn = create_master_dn_var.get().strip()

        if not master_day or not master_dn:
            messagebox.showwarning("Заполните поля", "Укажите пути мастер-файлов.")
            return

        if Path(master_day).exists():
            ok = messagebox.askyesno(
                "Мастер по дням уже существует",
                "Файл будет перезаписан. Продолжить?",
            )
            if not ok:
                return

        if Path(master_dn).exists():
            ok = messagebox.askyesno(
                "Мастер по дням/ночам уже существует",
                "Файл будет перезаписан. Продолжить?",
            )
            if not ok:
                return

        try:
            status_var.set("Идёт формирование мастеров...")
            root.update_idletasks()

            out_day, out_dn = _default_out_dirs()

            update_by_day(
                input_dir=str(DEFAULT_INPUT_DIR),
                out_dir=str(out_day),
                master_filename=master_day,
                unique=True,
                append_only_missing=True,
                mode="manual",
                files=files,
                skip_if_in_state=False,
                write_master=True,
                reset_master=True,
            )

            update_by_day_night(
                input_dir=str(DEFAULT_INPUT_DIR),
                out_dir=str(out_dn),
                master_filename=master_dn,
                unique=True,
                append_only_missing=True,
                mode="manual",
                files=files,
                skip_if_in_state=False,
                write_master=True,
                reset_master=True,
            )

            messagebox.showinfo(
                "Готово",
                "Мастер-файлы созданы.\n\n"
                f"Мастер по дням:\n{master_day}\n\n"
                f"Мастер по дням/ночам:\n{master_dn}",
            )
            status_var.set("Готово (создание мастера)")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Во время выполнения произошла ошибка:\n{exc}")
            status_var.set("Ошибка при создании мастера")

    btn_create = tk.Button(
        tab_create,
        text="4. Сформировать мастер",
        command=run_create,
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
        padx=15,
        pady=6,
    )
    btn_create.pack(anchor="w", pady=(0, 10))

    info_create = tk.Label(
        tab_create,
        text=(
            "Результат:\n"
            "• Создаются новые мастер-файлы по дням и по дням/ночам\n"
            "• Файлы сохраняются по установленному правилу"
        ),
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Segoe UI", 10),
        justify="left",
    )
    info_create.pack(anchor="w")

    status_frame = tk.Frame(container, bg=COLOR_BG)
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

    if own_root:
        root.mainloop()
    return root


if __name__ == "__main__":
    main()
