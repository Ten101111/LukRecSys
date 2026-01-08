import os

from constants import SRC_PATH, CHECKS_LICARD, PATH_TO_CURRENT_DATA_COMP

import pandas as pd
import datetime as dt



def fact_prediction_compare(df_licard_checks, df_real_checks):
    df_real_checks['date'] = pd.to_datetime(df_real_checks[['year', 'month', 'day']], errors='coerce')
    df_licard_checks['Дата'] = pd.to_datetime(df_licard_checks['Дата'], errors='coerce')

    df_licard_checks = pd.merge(
        df_licard_checks, df_real_checks,
        left_on=['КССС', 'Дата'],
        right_on=['ORG_KSSS', 'date'],
        how="left"
    )

    df_licard_checks = df_licard_checks.dropna(subset=['unique_cheques'])
    df_licard_checks = df_licard_checks[['КССС', 'date', 'Совокупное кол-во суточных чеков', 'unique_cheques']]
    today = pd.Timestamp.today().normalize()
    df_licard_checks = df_licard_checks[df_licard_checks['date'] <= today]
    df_licard_checks['MAPE, %'] = abs(round(df_licard_checks['Совокупное кол-во суточных чеков']/df_licard_checks['unique_cheques'] - 1, 4))
    return df_licard_checks


def month_MAPE(dr_result_comp):
    dr_result_comp['month'] = dr_result_comp['date'].dt.month
    df_means_all_period = dr_result_comp.groupby(['КССС', 'month'], as_index=False)['MAPE, %'].mean()
    return df_means_all_period


def year_month_MAPE(dr_result_comp):
    df_means_all_period = dr_result_comp.groupby(['КССС'], as_index=False)['MAPE, %'].mean()
    df_means_all_period['MAPE, %'] = round(df_means_all_period['MAPE, %']*100, 3)
    return df_means_all_period



def saving_data_MAPE_current(df_licard_checks, df_real_checks):
    now = dt.datetime.now()
    os.makedirs(PATH_TO_CURRENT_DATA_COMP, exist_ok=True)
    file_name = f'Файл с сравнением на {now.day}.{now.month}.{now.year}.csv'
    full_path = os.path.join(PATH_TO_CURRENT_DATA_COMP, file_name)

    dr_result_comp = fact_prediction_compare(df_licard_checks, df_real_checks)
    df_means_all_period = month_MAPE(dr_result_comp)
    df_mean_on_month = year_month_MAPE(df_means_all_period)

    df_mean_on_month.to_csv(full_path, index=False, sep='|')
    return df_mean_on_month


df_real_checks = pd.read_csv(SRC_PATH, sep="|")
df_licard_checks = pd.read_csv(CHECKS_LICARD, sep='|')

saving_data_MAPE_current(
    df_licard_checks,
    df_real_checks
)


