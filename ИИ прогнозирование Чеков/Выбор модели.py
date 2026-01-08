import os
import ast
import pandas as pd
import numpy as np
import datetime as dt

from constants import (
    OUTPUT_MODELS_COMPARISON,
    LAST_FILE_FROM_PATH_WEIGHTS,
    LAST_FILE_FROM_PATH_MODELS,
    LAST_FILE_FROM_PATH_CURRENT,
)

now = dt.datetime.now()


# ------------------------- утилиты чтения ------------------------- #

def read_weights_csv(path: str) -> pd.DataFrame:
    """Чтение файла с весами (w1, w2, w3 и МАРЕ)."""
    df = None
    last_err = None
    # пробуем utf-8-sig, потом cp1251, разделитель авто
    for enc in ("utf-8-sig", "cp1251"):
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise last_err

    # чистим имена колонок
    df.columns = (
        df.columns.str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def read_best_models_csv(path: str) -> pd.DataFrame:
    """Чтение файла с лучшими ML-моделями (АЗС, Лучшая модель, MAPE)."""
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def read_current_mape_csv(path: str) -> pd.DataFrame:
    """
    Чтение файла current (КССС|MAPE, %).
    Формат строк: '1059|10.994'
    На выходе: KSSS, current_mape_pct (float).
    """
    df = None
    last_err = None
    for enc in ("utf-8-sig", "cp1251"):
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise last_err

    df.columns = (
        df.columns.str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # ищем колонку с КССС|MAPE
    key_col_candidates = [
        c for c in df.columns
        if "КССС|MAPE" in c or "KCCC|MAPE" in c or "KSSS|MAPE" in c
    ]
    if not key_col_candidates:
        raise ValueError("Не найден столбец вида 'КССС|MAPE' в файле current.")
    key_col = key_col_candidates[0]

    split = df[key_col].astype(str).str.split("|", n=1, expand=True)
    df["KSSS"] = split[0].str.strip()
    df["current_mape_pct"] = split[1].apply(coerce_numeric)

    return df[["KSSS", "current_mape_pct"]]


# ------------------------- утилиты обработки ------------------------- #

def coerce_numeric(x):
    """Переводит значение в float, учитывая запятые и проценты."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("%", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_weight_tuple(s):
    """
    Разбор столбца 'Веса' в (w1, w2, w3).
    Поддерживает строковый вид, tuple, а также попытку вытащить числа через regex.
    """
    if pd.isna(s):
        return (np.nan, np.nan, np.nan)
    if isinstance(s, (tuple, list)) and len(s) == 3:
        return tuple(float(v) for v in s)

    s = str(s).strip()
    # нормализуем запятые внутри tuple
    s_norm = s.replace(",", ", ")
    try:
        tup = ast.literal_eval(s_norm)
        if isinstance(tup, tuple) and len(tup) == 3:
            return (float(tup[0]), float(tup[1]), float(tup[2]))
    except Exception:
        # fallback: собственный парсер цифр
        nums = []
        cur = ""
        for ch in s:
            if ch in "0123456789.-":
                cur += ch
            else:
                if cur:
                    nums.append(cur)
                    cur = ""
        if cur:
            nums.append(cur)
        if len(nums) >= 3:
            try:
                return (float(nums[0]), float(nums[1]), float(nums[2]))
            except Exception:
                pass
    return (np.nan, np.nan, np.nan)


# ------------------------- загрузка данных ------------------------- #

wdf = read_weights_csv(LAST_FILE_FROM_PATH_WEIGHTS)
bdf = read_best_models_csv(LAST_FILE_FROM_PATH_MODELS)
cdf = read_current_mape_csv(LAST_FILE_FROM_PATH_CURRENT)

# копии, чтобы не портить оригиналы
w = wdf.copy()
b = bdf.copy()
c = cdf.copy()

# стандартизируем ключ
# в w ожидается колонка 'КССС'
w["KSSS"] = w["КССС"].astype(str).str.strip()
# в b ожидается колонка 'АЗС'
b["KSSS"] = b["АЗС"].astype(str).str.strip()
# в c уже есть 'KSSS' после чтения

# --- парсим веса и MAPE из файла weights --- #

# разбираем 'Веса' -> w1, w2, w3
w[["w1", "w2", "w3"]] = w["Веса"].apply(parse_weight_tuple).apply(pd.Series)

# переводим 'МАРЕ' в float-проценты
w["weights_mape_pct"] = w["МАРЕ"].apply(coerce_numeric)

# --- готовим данные по ML лучшим моделям --- #

b["ml_best_model"] = b["Лучшая модель"].astype(str).str.strip()
b["ml_best_mape_pct"] = b["MAPE"].apply(coerce_numeric)

# ------------------------- объединение трёх источников ------------------------- #

all_keys = pd.DataFrame(
    {
        "KSSS": pd.Index(
            sorted(
                set(w["KSSS"]).union(set(b["KSSS"])).union(set(c["KSSS"]))
            )
        )
    }
)

merged = (
    all_keys
    .merge(
        w[["KSSS", "w1", "w2", "w3", "weights_mape_pct"]],
        on="KSSS",
        how="left",
    )
    .merge(
        b[["KSSS", "ml_best_model", "ml_best_mape_pct"]],
        on="KSSS",
        how="left",
    )
    .merge(
        c[["KSSS", "current_mape_pct"]],
        on="KSSS",
        how="left",
    )
)


# ------------------------- логика выбора лучшего источника ------------------------- #

def choose_row(row: pd.Series) -> pd.Series:
    wm = row["weights_mape_pct"]
    mm = row["ml_best_mape_pct"]
    cm = row["current_mape_pct"]

    # кандидаты: источник -> MAPE
    mape_candidates = {
        "weights": wm,
        "ml": mm,
        "current": cm,
    }

    # оставляем только не-NaN
    available = {k: v for k, v in mape_candidates.items() if pd.notna(v)}

    # если вообще нет MAPE ни из одного источника
    if not available:
        return pd.Series(
            {
                "chosen_source": np.nan,
                "chosen_model": np.nan,
                "chosen_w1": np.nan,
                "chosen_w2": np.nan,
                "chosen_w3": np.nan,
                "chosen_mape_pct": np.nan,
            }
        )

    # выбор источника с минимальным MAPE
    # при равенстве приоритет: weights -> ml -> current (из-за порядка keys)
    best_source = min(available, key=lambda k: available[k])
    best_mape = available[best_source]

    if best_source == "weights":
        return pd.Series(
            {
                "chosen_source": "weights",
                "chosen_model": "Весовая комбинация",
                "chosen_w1": row["w1"],
                "chosen_w2": row["w2"],
                "chosen_w3": row["w3"],
                "chosen_mape_pct": best_mape,
            }
        )
    elif best_source == "ml":
        return pd.Series(
            {
                "chosen_source": "ml",
                "chosen_model": row["ml_best_model"],
                "chosen_w1": np.nan,
                "chosen_w2": np.nan,
                "chosen_w3": np.nan,
                "chosen_mape_pct": best_mape,
            }
        )
    else:  # best_source == "current"
        # как ты просил: модель записываем как 'current'
        return pd.Series(
            {
                "chosen_source": "current",
                "chosen_model": "current",
                "chosen_w1": np.nan,
                "chosen_w2": np.nan,
                "chosen_w3": np.nan,
                "chosen_mape_pct": best_mape,
            }
        )


choice = merged.apply(choose_row, axis=1)
final = pd.concat([merged, choice], axis=1)

# ------------------------- финальный датафрейм ------------------------- #

final = final[
    [
        "KSSS",
        "w1",
        "w2",
        "w3",
        "weights_mape_pct",
        "ml_best_model",
        "ml_best_mape_pct",
        "current_mape_pct",
        "chosen_source",
        "chosen_model",
        "chosen_w1",
        "chosen_w2",
        "chosen_w3",
        "chosen_mape_pct",
    ]
].sort_values(
    ["chosen_source", "chosen_mape_pct", "KSSS"],
    na_position="last",
)

# ------------------------- сохранение результата ------------------------- #

os.makedirs(OUTPUT_MODELS_COMPARISON, exist_ok=True)

file_name = f"Сравнение {now.hour:02d}.{now.minute:02d}.csv"
out_csv = os.path.join(OUTPUT_MODELS_COMPARISON, file_name)

final.to_csv(out_csv, index=False, encoding="utf-8")
