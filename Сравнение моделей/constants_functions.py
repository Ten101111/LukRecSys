from pathlib import Path
from datetime import datetime
import re
from typing import Optional

# Дата в конце имени папки (например, "Объединенные чеки от 1.11.2025")
DATE_RE = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})\s*$")

# Имя файла вида: "Объединение чеков по дням_1.11.2025 12.34"
# Допустимы разделители времени '.' или ':' (на всякий случай).
FILE_RE = re.compile(
    r"\s*(?P<date>\d{1,2}\.\d{1,2}\.\d{4})\s+(?P<h>\d{1,2})[.:](?P<m>\d{2})\b"
)


def parse_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%d.%m.%Y")
    except ValueError:
        return None

def parse_file_dt(name: str) -> Optional[datetime]:
    m = FILE_RE.match(name)
    if not m:
        return None
    d = m.group("date")
    h = int(m.group("h"))
    mnt = int(m.group("m"))
    try:
        return datetime.strptime(f"{d} {h:02d}:{mnt:02d}", "%d.%m.%Y %H:%M")
    except ValueError:
        return None

def pick_latest_folder(base: Path) -> Path:
    if not base.is_dir():
        raise FileNotFoundError(f"Путь не найден или не папка: {base}")

    candidates = []
    for p in base.iterdir():
        if p.is_dir():
            m = DATE_RE.search(p.name)
            if m:
                dt = parse_date(m.group(1))
                if dt:
                    candidates.append((dt, p))

    if not candidates:
        raise RuntimeError("Не найдено ни одной папки с датой в конце имени.")

    # Берём папку с максимальной датой
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def pick_latest_file(folder: Path) -> Path:
    files = [f for f in folder.iterdir() if f.is_file()]
    if not files:
        raise RuntimeError(f"В папке нет файлов: {folder}")

    parsed = []
    fallback = []
    for f in files:
        dt = parse_file_dt(f.name)
        if dt:
            parsed.append((dt, f))
        else:
            # на всякий случай сохраняем для возможного fallback по mtime
            fallback.append(f)

    if parsed:
        parsed.sort(key=lambda x: x[0])
        return parsed[-1][1]

    # Если ничего не распарсилось по шаблону — берём самый свежий по mtime
    fallback.sort(key=lambda f: f.stat().st_mtime)
    return fallback[-1]

def pretty_display_name(p: Path) -> str:
    # Заменяем ТОЛЬКО тот "_" который стоит прямо перед "дд.мм.гггг HH.MM"
    return re.sub(
        r"_(?=\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}[.:]\d{2}\b)",
        " ",
        p.name
    )