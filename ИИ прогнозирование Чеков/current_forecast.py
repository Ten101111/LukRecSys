import calendar
import os
import time

from constants import SRC_PATH
from constants import HORIZON, WEEKDAY, PATH_TO_DAY_NIGHT_FILE, DAYS_UNTIL_SENDINGS, PATH_TO_SAVE_CURRENT_FORCAST

import pandas as pd
import datetime as dt
from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')
# берем массив всех данных по чекам
df = pd.read_csv(SRC_PATH, sep='|')

####### ПОТОМ НАДО УБРАТЬ #########
df_unique_ksss = df['ORG_KSSS'].unique()
df_unique_ksss.sort()
df_unique_ksss = df_unique_ksss[:100]
df = df[df['ORG_KSSS'].isin(df_unique_ksss)]
####### ПОТОМ НАДО УБРАТЬ #########

def periods_maker(df, now = None):
    # Добавляем дату в массив
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    if now is None:
        now = df["date"].max()
    now = pd.to_datetime(now).normalize()
    # Берем крайние 30 дней имеющихся чеков
    dates_trend_now = [i for i in df['date'].unique() if i <= now][-31:]
    # Обозначаем последнюю дату текущего трендового месяца
    trend_day_stop = dates_trend_now[-1]
    # Создаем список дат аналогичного периода трендового месяца в прошлом году
    dates_trend_last = list(pd.date_range(
        start=trend_day_stop - timedelta(days=365+30),
        end=trend_day_stop - timedelta(days=365),
        freq='D'  # по дням
    ))
    # Берем месяц, который мы прогнозируем и создаем список дат прогноза
    y = (now+timedelta(days=HORIZON-15)).year
    m = (now + timedelta(days=HORIZON - 15)).month
    MONTH_LEN = calendar.monthrange(y, m)[-1]
    dates_season_now = list(pd.date_range(
        start=trend_day_stop + timedelta(days=HORIZON+DAYS_UNTIL_SENDINGS-int(MONTH_LEN)),
        end=trend_day_stop + timedelta(days=HORIZON+DAYS_UNTIL_SENDINGS+(30-MONTH_LEN)),
        freq='D'  # по дням
    ))
    # Берем аналогичный месяц прогнозируемого периода прошлого года
    dates_season_last = list(pd.date_range(
        start=dates_season_now[0] - timedelta(days=365) ,
        end=dates_season_now[-1] - timedelta(days=365),
        freq='D'  # по дням
    ))
    return dates_trend_now, dates_trend_last, dates_season_now, dates_season_last

def indicator_adder(df_ksss, sum_month):
    if "unique_cheques" in df_ksss.columns:
        df_ksss['Доля, %'] = round(df_ksss['unique_cheques'] / sum_month, 5)
    df_ksss['ДенНед_числ'] = pd.to_datetime(df_ksss['date'], format="%Y-%m-%d", errors='coerce').dt.isocalendar().day.astype(
        'Int64')
    df_ksss['ДенНед'] = df_ksss['ДенНед_числ'].map(WEEKDAY)
    df_ksss['НомерДняНедВМес'] = df_ksss.groupby('ДенНед').cumcount() + 1
    df_ksss['Сцепка'] = df_ksss['ДенНед'] + "_" + df_ksss['НомерДняНедВМес'].astype(str)
    return df_ksss

def proportion_of_day_night(PATH_TO_DAY_NIGHT_FILE, ksss, dates_trend_last):
    df_day_night = pd.read_csv(PATH_TO_DAY_NIGHT_FILE, sep='|')
    df_day_night = df_day_night[df_day_night['ORG_KSSS'] == ksss]
    df_day_night['date'] = pd.to_datetime(df_day_night[['year', 'month', 'day']], errors='coerce')
    df_day_night = df_day_night[df_day_night['date'].isin(dates_trend_last)]

    dayNnight_checks = (
        df_day_night
        .groupby('time_segment')['unique_cheques']
        .sum()
    )

    value_day = dayNnight_checks.get('08_20', 0)
    value_night = dayNnight_checks.get('other', 0)

    # На всякий случай, если оба 0 — чтобы не делить потом на 0
    if value_day + value_night == 0:
        return 0, 0
    elif value_night == 0:
        return value_day, 0

    return value_day, value_night


def current_forecast(df):
    print('----- Начинаем прогнозировать чеки -----')
    global_start = time.time()
    spisok_datframes = []
    dates_trend_now, dates_trend_last, dates_season_now, dates_season_last = periods_maker(df)

    dataset_for_trend_last = df[df['date'].isin(dates_trend_last)]
    dataset_for_trend_now = df[df['date'].isin(dates_trend_now)]
    dataset_for_season_last = df[df['date'].isin(dates_season_last)]

    for ksss in dataset_for_season_last['ORG_KSSS'].unique():
        start = time.time()
        print(f"Обрабатываем файл для АЗС {ksss}")

        # Подвыборки по конкретной АЗС
        df_t0 = dataset_for_trend_now[dataset_for_trend_now["ORG_KSSS"] == ksss]
        df_t1 = dataset_for_trend_last[dataset_for_trend_last["ORG_KSSS"] == ksss]
        df_s1 = dataset_for_season_last[dataset_for_season_last["ORG_KSSS"] == ksss]

        # 0. Проверка на пустые выборки
        if df_t0.empty or df_t1.empty or df_s1.empty:
            print(
                f"Пропускаю АЗС {ksss}: пустые выборки "
                f"(trend_now={len(df_t0)}, trend_last={len(df_t1)}, season_last={len(df_s1)})\n"
            )
            continue

        # Вспомогательная функция: считаем 0 и NaN в unique_cheques
        def bad_count(df):
            s = df['unique_cheques']
            return s.isna().sum() + (s == 0).sum()

        bad_t0 = bad_count(df_t0)
        bad_t1 = bad_count(df_t1)
        bad_s1 = bad_count(df_s1)

        # 1. Если в какой-то выборке больше 1 нулевого/пустого значения — пропускаем АЗС
        if bad_t0 > 1 or bad_t1 > 1 or bad_s1 > 1:
            print(
                f"Пропускаю АЗС {ksss}: слишком много нулевых/пустых unique_cheques "
                f"(trend_now: {bad_t0}, trend_last: {bad_t1}, season_last: {bad_s1})\n"
            )
            continue

        # 2. Считаем суммы
        sum_t0 = df_t0['unique_cheques'].sum()
        sum_t1 = df_t1['unique_cheques'].sum()
        sum_s1 = df_s1['unique_cheques'].sum()

        # 3. Если хотя бы одна сумма 0 или NaN — тоже пропускаем
        if any((x == 0) or pd.isna(x) for x in (sum_t0, sum_t1, sum_s1)):
            print(
                f"Пропускаю АЗС {ksss}: суммарные чеки нулевые/NaN "
                f"(sum_t0={sum_t0}, sum_t1={sum_t1}, sum_s1={sum_s1})\n"
            )
            continue

        # 4. Всё ок — считаем sum_s0
        sum_s0 = round(sum_s1 * sum_t1 / sum_t0, 0)

        # дальше идёт твоя логика:
        df_ksss_forc_prev = df_s1.copy()
        df_ksss_forc_prev = indicator_adder(df_ksss_forc_prev, sum_s1)

        idx = pd.date_range(start=dates_season_now[0], end=dates_season_now[-1], freq="D")
        df_ksss_forc = pd.DataFrame(index=idx)
        df_ksss_forc['ORG_KSSS'] = ksss
        df_ksss_forc["year"] = df_ksss_forc.index.year
        df_ksss_forc["month"] = df_ksss_forc.index.month
        df_ksss_forc["day"] = df_ksss_forc.index.day
        df_ksss_forc["date"] = df_ksss_forc.index
        df_ksss_forc = indicator_adder(df_ksss_forc, sum_s0)

        # 1) Подтягиваем «Доля, %» по сцепке
        df_ksss_forc = df_ksss_forc.merge(
            df_ksss_forc_prev[['Сцепка', 'Доля, %']],
            on='Сцепка',  # короче, то же самое что left_on=... right_on=...
            how='left'
        )

        # 2) Среднее «Доля, %» по дням недели из prev
        df_weekday_avg = (
            df_ksss_forc_prev
            .groupby('ДенНед', sort=False)['Доля, %']
            .mean()
            .rename('Доля, %_weekday_mean')  # важно: понятное имя колонки после merge
        )

        # 3) Мёрджим среднее по дню недели
        df_ksss_forc = df_ksss_forc.merge(
            df_weekday_avg,
            left_on='ДенНед',
            right_index=True,  # <= ключевой фикс
            how='left'
        )

        # 4) Заполняем NaN в «Доля, %» средним по соответствующему «ДенНед»
        df_ksss_forc['Доля, %'] = df_ksss_forc['Доля, %'].fillna(df_ksss_forc['Доля, %_weekday_mean'])

        # (опционально) Фоллбэк, если «ДенНед» был NaN и среднее не подтянулось:
        df_ksss_forc['Доля, %'] = df_ksss_forc['Доля, %'].fillna(
            df_ksss_forc_prev['Доля, %'].mean()
        )

        # 5) Чистим служебную колонку
        df_ksss_forc.drop(columns=['Доля, %_weekday_mean'], inplace=True)

        df_ksss_forc['Чеки сутки (шт.)'] = round(df_ksss_forc['Доля, %']*sum_s0, 0)

        value_day, value_night = proportion_of_day_night(PATH_TO_DAY_NIGHT_FILE, ksss, periods_maker(df)[1])
        if value_day+value_night is None:
            print(f'Пропускаю {ksss}: нет данных по day/night')
            continue
        df_ksss_forc['Чеки день (шт.)'] = round(df_ksss_forc['Чеки сутки (шт.)']  * (value_day/(value_day+value_night)), 0)
        df_ksss_forc['Чеки ночь (шт.)'] = round(df_ksss_forc['Чеки сутки (шт.)'] * (value_night/(value_day + value_night)), 0)
        df_ksss_forc = df_ksss_forc[['ORG_KSSS', 'date', 'Чеки сутки (шт.)', 'Чеки день (шт.)', 'Чеки ночь (шт.)']]
        spisok_datframes.append(df_ksss_forc)
        print(f'Выполнил для {ksss} за {round(time.time()-start, 3)} сек.\n')
    reslt_df = pd.concat(spisok_datframes, ignore_index=True)
    print()
    print("------ Все объекты спрогнозированы ------")
    print(f'Время выполнения: {round(time.time()-global_start, 2)}')
    now = dt.datetime.now()
    os.mkdir(PATH_TO_SAVE_CURRENT_FORCAST)
    file_name = f'Файл с прогнозом на {now.day}.{now.month}.{now.day}.csv'
    full_path = os.path.join(PATH_TO_SAVE_CURRENT_FORCAST, file_name)
    reslt_df.to_csv(full_path, index=False, sep='|')
    return reslt_df

# df = df[df['ORG_KSSS'] == 1059]
# print(current_forecast(df))