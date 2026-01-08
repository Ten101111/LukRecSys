import pandas as pd
from time import time
import datetime as dt
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

from functions_ml  import path_saver, build_best_models_dataset
from constants import SRC_PATH, OUTPUT_PATH_DIR, AZS_PRED_FILE, NAME_OF_COLUMN_OF_KSSS

AZS_for_pred = pd.read_excel(AZS_PRED_FILE)['КССС'].tolist()

# Загружаем общий датасет
df = pd.read_csv(SRC_PATH, sep='|')
df['Дата'] = pd.to_datetime(df[['year','month','day']], errors='coerce')
df = df[df[NAME_OF_COLUMN_OF_KSSS].isin(AZS_for_pred)]
# Явно приводим дату (на всякий случай)
if "Дата" in df.columns:
    df["Дата"] = pd.to_datetime(df["Дата"])

# Изменяем тип названия чеков
if 'unique_cheques' in df.columns and 'Чеки' not in df.columns:
    df = df.rename(columns={'unique_cheques': 'Чеки'})


####### ПОТОМ НАДО УБРАТЬ #########
# df_unique_ksss = df['ORG_KSSS'].unique()
# df_unique_ksss.sort()
# df_unique_ksss = df_unique_ksss[300:375]
# df = df[df['ORG_KSSS'].isin(df_unique_ksss)]
# df = df[df['Дата'] < pd.Timestamp("2025-08-01")]
####### ПОТОМ НАДО УБРАТЬ #########

now = dt.datetime.now()
print('------------------------------------------------------')
print(f'-------------------- \033[1mДобрый день!\033[0m --------------------')
print(f'----------------- \033[1mСегодня: {now.day}.{now.month}.{now.year}\033[0m -----------------')
print('---- \033[1mНачинаем прогнозирование с использованием ML\033[0m ----')
print('------------------------------------------------------\n')
# === ШАГ 1: строим сводную таблицу по всем АЗС ===
summary_df = build_best_models_dataset(df)
print(summary_df)

# Сохраняем сводную таблицу
os.makedirs(OUTPUT_PATH_DIR, exist_ok=True)
path_saver(summary_df, OUTPUT_PATH_DIR, f"best_models_by_azs_all {now.year}.{now.month}.{now.day} {now.hour}.{now.minute}")













# === ШАГ 2: прогнозы временно отключаем ===
# sp_of_ksss = df[NAME_OF_COLUMN_OF_KSSS].unique()
# df_osnova = ksss_choice(df, sp_of_ksss[0])
# df_resulting = analysing_of_data(df_osnova[:len(df_osnova)-75])
#
# for ksss in sp_of_ksss[1:]:
#     time_start = time()
#     df_analyse = ksss_choice(df, ksss)
#     df_merged = analysing_of_data(df_analyse[:len(df_analyse)-75])
#     df_resulting = df_resulting.merge(
#         df_merged,
#         left_on='Дата',
#         right_on='Дата',
#         how='left'
#     )
#     print(f'Время выполнения расчета для КССС {ksss}:', time()-time_start)
#
# print(df_resulting)
# path_saver(df_resulting, OUTPUT_PATH_DIR, 'АЗС тестовые')
