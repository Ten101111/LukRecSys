import pandas as pd
import os
import time

from function_triple_avg import optimize_global_weights, month_range
from constants import SRC_PATH, OUT_CSV, MONTH_RANGE_START, MONTH_RANGE_END, PATH_NEW_FOLDER, AZS_PRED_FILE, NAME_OF_COLUMN_OF_KSSS

import warnings
warnings.filterwarnings("ignore")

beginning_time = time.time()

# ------------------------------ 1 ЭТАП ------------------------------
# ----- ОТКРЫВАЕМ ФАЙЛ С ДАННЫМИ ПО ЧЕКАМ И ПРИВОДИМ К ЗНАЧЕНИЯМ -----

# Для ограничения АЗС
AZS_for_pred = pd.read_excel(AZS_PRED_FILE)['КССС'].tolist()

# читаем данные
df = pd.read_csv(SRC_PATH, sep='|')
df = df[df[NAME_OF_COLUMN_OF_KSSS].isin(AZS_for_pred)]
# чистим заголовки и возможный BOM
df.columns = df.columns.str.replace('\ufeff', '', regex=True).str.strip()
# переименуем столбец количества чеков под ожидаемое имя
if 'unique_cheques' in df.columns and 'Чеки' not in df.columns:
    df = df.rename(columns={'unique_cheques': 'Чеки'})

df['Дата'] = pd.to_datetime(df[['year','month','day']], errors='coerce')
df['Чеки'] = pd.to_numeric(df['Чеки'], errors='coerce')
# выбрасываем мусорные строки
df = df.dropna(subset=['Дата', 'Чеки']).reset_index(drop=True)

# ---------------------------------- 2 ЭТАП ---------------------------------
# ------ ПРОХОДИМСЯ ПО ОБЪЕКТАМ И ВЫПОЛНЯЕМ ФУНКЦИЮ ДЛЯ КАЖДОГО ИЗ НИХ ------

# Создаем список уникальных значений по КССС
ksss_list = df['ORG_KSSS'].astype(str).unique()
# Словарь для дальнейшго добавления
dic_weights = dict()

print('Иду взвешивать АЗСки:\n')

for ksss in ksss_list:
    print(f'Начинаю работу с данными по АЗС {ksss}...')
    start_time = time.time()
    df_analizing = df[df['ORG_KSSS'].astype(str) == str(ksss)].copy()
    # ограничим целевые месяцы фактическим диапазоном этого объекта, чтобы не падать
    if df_analizing['Дата'].empty:
        print(f'АЗС {ksss}: нет данных — пропускаю.')
        continue
    # Выбираем период
    min_month = df_analizing['Дата'].min().to_period('M').to_timestamp()
    max_month = df_analizing['Дата'].max().to_period('M').to_timestamp()
    base_months = month_range(MONTH_RANGE_START, MONTH_RANGE_END)
    target_months = [m for m in base_months if (m >= min_month) and (m <= max_month)]
    if not target_months:
        print(f'АЗС {ksss}: ни один из целевых месяцев не попадает в диапазон фактов '
              f'({min_month.date()} — {max_month.date()}). Пропускаю.')
        continue
    try:
        data_expected = optimize_global_weights(
            daily_df=df_analizing,
            actuals_df=df_analizing,
            org_ksss=ksss,
            target_months=target_months,
            step=0.01,
            value_col="Чеки",
            comparison_org_col="ORG_KSSS",
            comparison_day_col="Дата"
        )
    except ValueError as e:
        # Просто пропускаем объект и идём дальше
        print(f'АЗС {ksss}: пропущено из-за данных -> {e}')
        continue

    w1, w2, w3 = map(float, data_expected.weights)
    mape_val = float(data_expected.mape)

    # сохраняем в словарь — дальше красиво отформатируем
    dic_weights[str(ksss)] = {"w1": w1, "w2": w2, "w3": w3, "MAPE": mape_val}

    print(f'Обработали файл с АЗС {ksss} за {round(time.time()-start_time, 0)} сек.')

# -------------------- 3 ЭТАП -------------------
# ------ ФОРМИРУЕМ МАССИВ ВЕСОВ ПООБЪЕКТНО ------

print('\nВывожу полученный словарь в виде датафрейма:')
# собираем датафрейм
df_save = (pd.DataFrame.from_dict(dic_weights, orient='index')
           .rename(columns={'MAPE': 'МАРЕ'})
           .rename_axis('КССС')
           .reset_index())
# формируем человеческую строку весов "(x.xx, y.yy, z.zz)"
df_save['Веса'] = df_save.apply(
    lambda r: f"({r['w1']:.2f}, {r['w2']:.2f}, {r['w3']:.2f})", axis=1
)
df_save.drop(columns=['w1', 'w2', 'w3'], inplace=True)
# округляем МАРЕ до 3 знаков и добавляем знак процента
df_save['МАРЕ'] = pd.to_numeric(df_save['МАРЕ'], errors='coerce').round(3).map(lambda x: f"{x:.2f}")
# финальный порядок колонок и печать
df_save = df_save[['КССС', 'Веса', 'МАРЕ']]
print(df_save.head())

# -------------------- 4 ЭТАП -------------------
# ------------- СОХРАНЯЕМ РЕЗУЛЬТАТ -------------
os.makedirs(PATH_NEW_FOLDER, exist_ok=True)
print(f'\nЗакончил, сохраняю результат по пути {OUT_CSV}')
# сохраняем как CSV с BOM, чтобы Excel корректно показал русские заголовки
df_save.to_csv(OUT_CSV, sep='|', index=False, encoding='utf-8-sig')

print('---------ГОТОВО! СОХРАНИЛ!---------\n')
print(f'Выполнено за {round(time.time()-beginning_time, 0)} сек.')
