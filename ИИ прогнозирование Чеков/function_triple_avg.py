import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =====================
# Utility data classes
# =====================

@dataclass
class ForecastResult:
    """
    Результат глобальной оптимизации весов для прогноза чеков.

    Attributes
    ----------
    weights : Tuple[float, float, float]
        Лучшие веса (w1, w2, w3) для линейной комбинации компонент comp1/comp2/comp3.
        Если данных для оптимизации нет, будет (nan, nan, nan).
    mape : float
        Общий MAPE (%) по всем дням и всем месяцам. Если данных нет — NaN.
    per_month_mape : Dict[pd.Timestamp, float]
        Карта "первое число месяца -> MAPE(%)" для каждого месяца.
    details : pd.DataFrame
        Подробная посуточная таблица:
        ["date","month","actual","fc","comp1","comp2","comp3","abs_pct_err"].
        Может быть пустой.
    """
    weights: Tuple[float, float, float]
    mape: float
    per_month_mape: Dict[pd.Timestamp, float]
    details: pd.DataFrame  # columns: [date, month, actual, fc, comp1, comp2, comp3, abs_pct_err]


# =====================
# Core helpers
# =====================

def _to_month_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Вернуть первое число месяца для переданного Timestamp."""
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def month_range(start_month: str, end_month: str, inclusive: bool = True) -> List[pd.Timestamp]:
    """
    Построить список первых чисел месяцев между двумя месяцами.
    Формат входа: 'YYYY-MM-01'
    """
    start = pd.Timestamp(start_month)
    end = pd.Timestamp(end_month)
    out = []
    cur = start
    while cur <= end if inclusive else cur < end:
        out.append(cur)
        year = cur.year + (1 if cur.month == 12 else 0)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = pd.Timestamp(year=year, month=month, day=1)
    return out


def robust_remove_outliers_by_weekday(df: pd.DataFrame, value_col: str = "Чеки") -> pd.DataFrame:
    """
    Удалить явные выбросы отдельно в каждом дне недели по правилу IQR.
    Работает безопасно: приводит типы, нормализует даты до полуночи.
    """
    if "Дата" not in df.columns or value_col not in df.columns:
        raise ValueError(f"robust_remove_outliers_by_weekday: нет колонок 'Дата' и/или '{value_col}'.")

    d = df.copy()

    # Приведение типов (страховка)
    d["Дата"] = pd.to_datetime(d["Дата"], errors="coerce", dayfirst=True).dt.normalize()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["Дата", value_col])

    d["weekday"] = d["Дата"].dt.weekday
    keep_idx = []
    for _, grp in d.groupby("weekday"):
        q1 = grp[value_col].quantile(0.25)
        q3 = grp[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        keep_idx.extend(grp[(grp[value_col] >= lower) & (grp[value_col] <= upper)].index.tolist())
    return d.loc[sorted(set(keep_idx)), ["Дата", value_col]].sort_values("Дата").reset_index(drop=True)


def compute_weekday_means(window_df: pd.DataFrame, value_col: str = "Чеки") -> Dict[int, float]:
    """Посчитать средние по weekday (0..6) в заданном окне, безопасно приводя типы."""
    if window_df.empty:
        return {i: np.nan for i in range(7)}
    wdf = window_df.copy()
    wdf["Дата"] = pd.to_datetime(wdf["Дата"], errors="coerce", dayfirst=True).dt.normalize()
    wdf[value_col] = pd.to_numeric(wdf[value_col], errors="coerce")
    wdf = wdf.dropna(subset=["Дата", value_col])
    wdf["weekday"] = wdf["Дата"].dt.weekday
    return wdf.groupby("weekday")[value_col].mean().to_dict()


def safe_get_value(daily_df: pd.DataFrame, date: pd.Timestamp, value_col: str = "Чеки") -> Optional[float]:
    """Вернуть значение для точной даты, если такая запись существует."""
    row = daily_df.loc[daily_df["Дата"] == date]
    if not row.empty:
        return float(row.iloc[0][value_col])
    return None


def mean_ignore_none(values: List[Optional[float]]) -> Optional[float]:
    """Среднее по списку, игнорируя None/NaN."""
    arr = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not arr:
        return None
    return float(np.mean(arr))


def build_components_for_month(
    daily_df: pd.DataFrame,
    month_start: pd.Timestamp,
    value_col: str,
    cutoff_date: pd.Timestamp,
    use_window_year: bool = True
) -> pd.DataFrame:
    """
    Построить три компонентные серии (comp1/comp2/comp3) на каждый день целевого месяца.
    - Используется только история с датами <= cutoff_date.
    - При use_window_year=True ограничиваемся последним годом до cutoff_date.
    """
    # 1) ограничиваем историю
    hist = daily_df[daily_df["Дата"] <= cutoff_date].copy()
    if use_window_year:
        window_start = cutoff_date - pd.DateOffset(years=1) + pd.Timedelta(days=1)
        hist = hist[hist["Дата"] >= window_start]
    hist = hist.sort_values("Дата")

    # 2) средние по weekday + общее среднее
    wd_means = compute_weekday_means(hist, value_col=value_col)
    overall_mean = float(hist[value_col].mean()) if not hist.empty else np.nan

    # 3) календарь целевого месяца
    next_month = (month_start + pd.offsets.MonthBegin(1))
    days = pd.date_range(month_start, next_month - pd.Timedelta(days=1), freq="D")

    rows = []
    for d in days:
        # comp1: тот же день прошлым годом
        same_day_last_year = d - pd.DateOffset(years=1)
        c1 = safe_get_value(hist, same_day_last_year, value_col=value_col)

        # comp2: среднее из d-1м и d-2м
        d_prev1 = (d - pd.DateOffset(months=1))
        d_prev2 = (d - pd.DateOffset(months=2))
        c2_list = []
        for cand in (d_prev1, d_prev2):
            if cand <= cutoff_date and (not use_window_year or cand >= (cutoff_date - pd.DateOffset(years=1) + pd.Timedelta(days=1))):
                val = safe_get_value(hist, cand, value_col=value_col)
            else:
                val = None
            c2_list.append(val)
        c2 = mean_ignore_none(c2_list)

        # comp3: среднее по weekday
        wd = int(d.weekday())
        c3 = wd_means.get(wd, np.nan)

        # фолбэки
        if c1 is None or (isinstance(c1, float) and math.isnan(c1)):
            c1 = c3 if not (isinstance(c3, float) and math.isnan(c3)) else overall_mean
        if c2 is None or (isinstance(c2, float) and math.isnan(c2)):
            c2 = c3 if not (isinstance(c3, float) and math.isnan(c3)) else overall_mean
        if isinstance(c3, float) and math.isnan(c3):
            c3 = overall_mean

        # финальный фолбэк
        c1 = 0.0 if (c1 is None or (isinstance(c1, float) and math.isnan(c1))) else float(c1)
        c2 = 0.0 if (c2 is None or (isinstance(c2, float) and math.isnan(c2))) else float(c2)
        c3 = 0.0 if (c3 is None or (isinstance(c3, float) and math.isnan(c3))) else float(c3)

        rows.append({"date": d, "comp1": c1, "comp2": c2, "comp3": c3})

    return pd.DataFrame(rows)


def combine_forecast(components_df: pd.DataFrame, w: Tuple[float, float, float]) -> np.ndarray:
    """Получить прогноз как линейную комбинацию компонент."""
    w1, w2, w3 = w
    return (w1 * components_df["comp1"].values +
            w2 * components_df["comp2"].values +
            w3 * components_df["comp3"].values)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE (%) с игнорированием невалидных фактов (<= 0)."""
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def weight_grid(step: float = 0.01) -> List[Tuple[float, float, float]]:
    """
    Сгенерировать все неотрицательные тройки весов (w1,w2,w3) на решётке,
    удовлетворяющие сумме 1.0.
    """
    # округляем до 2 знаков, чтобы оставаться на сетке
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    combos = []
    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            w3 = round(w3, 2)
            if w3 < 0:
                continue
            if abs(w1 + w2 + w3 - 1.0) <= 1e-9:
                combos.append((w1, w2, w3))
    return combos


def optimize_global_weights(
    daily_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    org_ksss: Optional[int],
    target_months: List[pd.Timestamp],
    step: float = 0.01,
    value_col: str = "Чеки",
    comparison_org_col: str = "ORG_KSSS",
    comparison_day_col: str = "Day",
    on_empty: str = "skip",  # "skip" -> вернуть NaN-результат; "raise" -> бросить ошибку
) -> ForecastResult:
    """
    Подобрать единые веса (w1,w2,w3) для всех целевых месяцев, минимизируя общий MAPE.

    ВАЖНО: встроены «защёлки»:
    - приведение дат к datetime + .dt.normalize()
    - приведение значений к numeric
    - выравнивание типов ORG_KSSS
    - если нет данных для оптимизации: по умолчанию НЕ падаем, а возвращаем NaN-результат
    """
    # --- нормализация типов и дат ---
    daily_df = daily_df.copy()
    actuals_df = actuals_df.copy()

    # Если внезапно несколько колонок "Дата" — оставим первую
    if (daily_df.columns == "Дата").sum() > 1:
        cols = []
        seen_date = False
        for c in daily_df.columns:
            if c == "Дата":
                if not seen_date:
                    cols.append(c)
                    seen_date = True
            else:
                cols.append(c)
        daily_df = daily_df.loc[:, cols]

    # Приведение типов в daily_df
    daily_df["Дата"] = pd.to_datetime(daily_df["Дата"], errors="coerce", dayfirst=True).dt.normalize()
    daily_df[value_col] = pd.to_numeric(daily_df[value_col], errors="coerce")
    daily_df = daily_df.dropna(subset=["Дата", value_col]).sort_values("Дата")

    # Приведение типов в фактах
    comp = actuals_df.copy()
    comp[comparison_day_col] = pd.to_datetime(comp[comparison_day_col], errors="coerce", dayfirst=True).dt.normalize()
    comp[value_col] = pd.to_numeric(comp[value_col], errors="coerce")

    # Фильтр по объекту
    if comparison_org_col in comp.columns and org_ksss is not None:
        comp = comp[comp[comparison_org_col].astype(str) == str(org_ksss)]

    if comp.empty:
        if on_empty == "skip":
            empty = pd.DataFrame(columns=["date","month","actual","fc","comp1","comp2","comp3","abs_pct_err"])
            return ForecastResult(weights=(np.nan, np.nan, np.nan), mape=np.nan, per_month_mape={}, details=empty)
        raise ValueError("После приведения типов и фильтра по объекту фактов не осталось. Проверьте ORG_KSSS и даты.")

    # 2) очистка выбросов
    daily_df_clean = robust_remove_outliers_by_weekday(daily_df, value_col=value_col)

    # 3) компоненты + факты по каждому месяцу
    month_to_data = {}
    for m_start in target_months:
        cutoff = m_start - pd.Timedelta(days=45)
        comps = build_components_for_month(
            daily_df=daily_df_clean,
            month_start=m_start,
            value_col=value_col,
            cutoff_date=cutoff,
            use_window_year=True
        )
        # факты месяца
        month_mask = (comp[comparison_day_col] >= m_start) & (comp[comparison_day_col] < (m_start + pd.offsets.MonthBegin(1)))
        month_act = comp.loc[month_mask, [comparison_day_col, value_col]].copy().rename(
            columns={comparison_day_col: "date", value_col: "actual"}
        )
        month_act["date"] = month_act["date"].dt.normalize()
        merged = comps.merge(month_act, on="date", how="left")
        if merged["actual"].notna().sum() == 0:
            continue
        month_to_data[m_start] = merged

    if not month_to_data:
        # по умолчанию — пропускаем, НЕ валимся
        if on_empty == "skip":
            empty = pd.DataFrame(columns=["date","month","actual","fc","comp1","comp2","comp3","abs_pct_err"])
            return ForecastResult(weights=(np.nan, np.nan, np.nan), mape=np.nan, per_month_mape={}, details=empty)
        else:
            min_fact = comp[comparison_day_col].min()
            max_fact = comp[comparison_day_col].max()
            months_txt = ", ".join([str(m.date()) for m in target_months])
            raise ValueError(
                "Нет месяцев с фактическими данными для оптимизации. "
                f"Диапазон фактов: {min_fact.date() if pd.notna(min_fact) else '---'} — {max_fact.date() if pd.notna(max_fact) else '---'}. "
                f"Целевые месяцы: {months_txt}. Проверьте совпадение дат и ORG_KSSS."
            )

    # 4) конкатенация
    all_rows = []
    for m_start, dfm in month_to_data.items():
        dfm = dfm.copy()
        dfm["month"] = m_start
        all_rows.append(dfm)
    big = pd.concat(all_rows, ignore_index=True)

    # 5) перебор весов
    comp_mat = big[["comp1", "comp2", "comp3"]].values
    actual_vec = big["actual"].values
    dates_vec = big["date"].values
    months_vec = big["month"].values

    best_w = None
    best_mape = np.inf

    for w in weight_grid(step=step):
        pred = comp_mat @ np.array(w)
        score = mape(actual_vec, pred)
        if not (score is None or np.isnan(score)) and score < best_mape:
            best_mape = score
            best_w = w

    # 6) детали по лучшим весам
    best_pred = comp_mat @ np.array(best_w)
    abs_pct_err = np.where(actual_vec > 0, np.abs((actual_vec - best_pred) / actual_vec) * 100.0, np.nan)

    details = pd.DataFrame({
        "date": dates_vec,
        "month": months_vec,
        "actual": actual_vec,
        "fc": best_pred,
        "comp1": big["comp1"].values,
        "comp2": big["comp2"].values,
        "comp3": big["comp3"].values,
        "abs_pct_err": abs_pct_err
    })

    # помесячный MAPE
    per_month_mapes = {}
    for m in sorted(month_to_data.keys()):
        dsub = details[details["month"] == m]
        per_month_mapes[m] = mape(dsub["actual"].values, dsub["fc"].values)

    return ForecastResult(weights=best_w, mape=float(best_mape), per_month_mape=per_month_mapes, details=details)

# =====================
# High-level API (опционально)
# =====================

def run_optimization(
    checks_path: str,
    compare_path: str,
    org_ksss: Optional[int] = 134142,
    start_month: str = "2024-08-01",
    end_month: str = "2025-08-01",
    step: float = 0.01,
    value_col: str = "Чеки",
    checks_date_col: str = "Дата",
    comparison_org_col: str = "ORG_KSSS",
    comparison_day_col: str = "Day",
    save_details_csv: Optional[str] = None
) -> ForecastResult:
    """
    Полный конвейер: загрузка, подготовка, подбор весов, (опционально) сохранение деталей.
    """
    # Загрузка ежедневных чеков
    df = pd.read_excel(checks_path)
    # чистим заголовки
    df.columns = df.columns.str.replace('\ufeff', '', regex=True).str.strip()
    # дубликаты имён — оставляем первые
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    if checks_date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"В файле {checks_path} должны быть колонки: {checks_date_col}, {value_col}")
    df = df[[checks_date_col, value_col]].copy().rename(columns={checks_date_col: "Дата"})
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", dayfirst=True).dt.normalize()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["Дата", value_col]).sort_values("Дата").reset_index(drop=True)

    # Загрузка фактов для сравнения
    compare_df = pd.read_excel(compare_path)
    compare_df.columns = compare_df.columns.str.replace('\ufeff', '', regex=True).str.strip()
    if compare_df.columns.duplicated().any():
        compare_df = compare_df.loc[:, ~compare_df.columns.duplicated(keep='first')]

    # Формируем целевые месяцы
    months = month_range(start_month=start_month, end_month=end_month, inclusive=True)

    # Подбор весов
    result = optimize_global_weights(
        daily_df=df,
        actuals_df=compare_df,
        org_ksss=org_ksss,
        target_months=months,
        step=step,
        value_col=value_col,
        comparison_org_col=comparison_org_col,
        comparison_day_col=comparison_day_col,
    )

    # Сохранение деталей (если задано)
    if save_details_csv:
        result.details.to_csv(save_details_csv, index=False, encoding="utf-8")

    return result
