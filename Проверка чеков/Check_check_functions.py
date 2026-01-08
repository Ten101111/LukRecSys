import time
from pathlib import Path

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import re
import calendar
from constants import MONTHS_NUMS

month = None


def month_indication(df):
    '''
    С учетом того, что у нас разное количество дней в месяце и зависит также оно от года
    Необходимо уточнить это кол-во. Поэтому данная функция с учтом календарного года вычленяет
    кол-во дней в месяце и передает в результате в функции
    '''
    name_of_recomendation = df.columns[0]
    global month
    month = name_of_recomendation.split()[-2].lower()

    num_of_month = MONTHS_NUMS[month]
    year = dt.datetime.now().year
    if dt.datetime.now().month > num_of_month:
        year += 1

    days = calendar.monthrange(year, num_of_month)[1]

    return days


def df_into_normal_view(df):
    """
    Так как у нас сам массив очень кривой и с пропусками, необходимо его привести в нормальный вид и заполнить значения
    """

    days = month_indication(df)

    new_row = df.iloc[1].astype(str) + "_" + df.iloc[2].astype(str)
    naming = list(df.iloc[1].astype(str)[:9]) + list(new_row[9:(days+11)]) + list(df.iloc[1].astype(str)[(days+11):])
    df.columns = naming

    columns_to_drop = [0, 3, 11, 15]
    df = df.drop(df.columns[columns_to_drop], axis=1)
    length = len(df)
    df = df.drop([0, 1, 2, length-2, length-1])

    col_tofill = ['НПО', 'РУ', 'ТМ', 'Менеджер АЗС', 'Номер АЗС', 'КССС']
    df[col_tofill] = df[col_tofill].ffill()

    df_personal_num = df[df['Рекомендуемая численность'].notna()][['КССС', 'Рекомендуемая численность']]
    df = df.drop('Рекомендуемая численность', axis=1)

    df = pd.merge(
        left=df,
        right=df_personal_num,
        left_on='КССС',
        right_on='КССС',
        how='left'
        )
    return df


def intensivity_counter(df):
    # Копируем DataFrame
    result_df = df.copy()

    # Получаем колонки с датами (формат: "1_ПТ", "2_СБ", ..., "31_ВС")
    date_columns = [col for col in df.columns if col.split('_')[0].isdigit()]

    # Создаем пустой список для новых строк
    intensity_rows = []

    # Группируем данные по АЗС и КССС
    grouped = df.groupby(['Номер АЗС', 'КССС'])

    for (azs_num, ksss), group in grouped:
        # Создаем словарь для хранения данных по сменам
        shift_data = {
            'day_checks': group[group['Показатель'] == 'Прогноз дневных чеков (дневная смена)'],
            'night_checks': group[group['Показатель'] == 'Прогноз ночных чеков (ночная смена)'],
            'day_hours': group[group['Показатель'] == 'Совокупная сумма часов для дневной смены'],
            'night_hours': group[group['Показатель'] == 'Совокупная сумма часов для ночной смены'],
            'all_checks': group[group['Показатель'] == 'Итого прогноз чеков на сутки'],
            'all_hours': group[group['Показатель'] == 'Итого целевая сумма часов на сутки']
        }

        # Обработка дневной смены
        if not shift_data['day_checks'].empty and not shift_data['day_hours'].empty:
            day_intensity = shift_data['day_checks'].iloc[0].copy()
            day_intensity['Показатель'] = 'Интенсивность труда (дневная смена)'
            for col in date_columns:
                try:
                    day_intensity[col] = round(
                        shift_data['day_checks'][col].iloc[0] / shift_data['day_hours'][col].iloc[0], 1)
                except (ZeroDivisionError, IndexError):
                    day_intensity[col] = 0
            intensity_rows.append(day_intensity)

        # Обработка ночной смены
        if not shift_data['night_checks'].empty and not shift_data['night_hours'].empty:
            night_intensity = shift_data['night_checks'].iloc[0].copy()
            night_intensity['Показатель'] = 'Интенсивность труда (ночная смена)'
            for col in date_columns:
                try:
                    night_intensity[col] = round(
                        shift_data['night_checks'][col].iloc[0] / shift_data['night_hours'][col].iloc[0], 1)
                except (ZeroDivisionError, IndexError):
                    night_intensity[col] = 0
            intensity_rows.append(night_intensity)

        # Обработка суточной смены
        if not shift_data['all_checks'].empty and not shift_data['all_hours'].empty:
            all_intensity = shift_data['all_checks'].iloc[0].copy()
            all_intensity['Показатель'] = 'Интенсивность труда суточная'
            for col in date_columns:
                try:
                    all_intensity[col] = round(
                        shift_data['all_checks'][col].iloc[0] / shift_data['all_hours'][col].iloc[0], 1)
                except (ZeroDivisionError, IndexError):
                    all_intensity[col] = 0
            intensity_rows.append(all_intensity)

    # Создаем DataFrame из новых строк и объединяем с исходным
    if intensity_rows:
        intensity_df = pd.DataFrame(intensity_rows)
        result_df = pd.concat([result_df, intensity_df], ignore_index=True)

    # Фильтруем только строки с интенсивностью (с обработкой NaN)
    result_df = result_df[result_df['Показатель'].str.contains('Интенсивность', na=False)]

    # Удаляем столбцы, если они существуют
    for col in ["Всего", "Рекомендуемая численность"]:
        if col in result_df.columns:
            result_df = result_df.drop(col, axis=1)

    # Сортируем для удобства
    result_df = result_df.sort_values(['Номер АЗС', 'КССС', 'Показатель'])

    return result_df


def df_melt_for_indicator(df):
    """
    Свертываем основной массив для того, чтобы все было в одном столбце.
    """

    date_columns = [col for col in df.columns if col.split('_')[0].isdigit()]
    id_vars = [col for col in df.columns if col not in date_columns]
    df = pd.melt(
        df,
        id_vars=id_vars,          # Колонки, которые остаются без изменений
        value_vars=date_columns,  # Колонки, которые "расплавляются" в строки
        var_name="Дата",          # Название новой колонки для дат
        value_name="Значение"     # Название новой колонки для значений
    )
    df[["День", "День недели"]] = df["Дата"].str.split("_", expand=True)
    df["День"] = df["День"].astype(int)  # Делаем день числовым
    return df


def for_concat_maker(df, name):
    df = df.rename(columns={'Значение': name})
    df = df.drop("Показатель", axis = 1)
    return df


def dfs_all_indicators_by_one(df):
    """
    Создаем список из датафреймов по одному - каждому признаку
    """
    df_list = []
    for i in list(df['Показатель'].unique()):
        df_add = df[df['Показатель'] == i]
        df_list.append(for_concat_maker(df_add, i))
    return df_list


def intensivity_df_changer(result_df):
    """
    Изменение (свертывание) в рамках массива с данными по интенсивности, которые были выше получены.
    """
    # 1. Определение колонок для транспонирования и оставления
    date_columns = [col for col in result_df.columns if col.split('_')[0].isdigit()]
    id_vars = [col for col in result_df.columns if col not in date_columns]

    # 2. Преобразуем в длинный формат
    long_df = pd.melt(
        result_df,
        id_vars=id_vars,          # Колонки, которые остаются без изменений
        value_vars=date_columns,  # Колонки, которые "расплавляются" в строки
        var_name="Дата",          # Название новой колонки для дат
        value_name="Значение"     # Название новой колонки для значений
    )

    # 3. Разделяем "Дата" на "День" и "День недели"
    long_df[["День", "День недели"]] = long_df["Дата"].str.split("_", expand=True)
    long_df["День"] = long_df["День"].astype(int)  # Делаем день числовым

    # 4. Сортируем по АЗС, показателю и дате (опционально)
    long_df = long_df.sort_values(["КССС", "Показатель", "День"])

    # 5. Переупорядочиваем колонки (опционально)
    cols_order = [
        "НПО", "РУ", "ТМ", "Менеджер АЗС", "Номер АЗС", "КССС", "Показатель",
        "Дата", "День", "День недели", "Значение"
    ]
    long_df = long_df[cols_order]

    # Выводим результат
    return long_df


def bigdata_former(dff):
    """
    Учитывая, что мы хотим посмотреть на больших числах информацию по АЗС
    Мы берем для каждой и смотрим итоги месяца по ряду показателей
    """
    target_columns = [
        'Прогноз дневных чеков (дневная смена)',
        'Совокупная сумма часов для дневной смены',
        'Прогноз ночных чеков (ночная смена)',
        'Совокупная сумма часов для ночной смены',
        'Итого прогноз чеков на сутки',
        'Итого целевая сумма часов на сутки',
    ]

    # Создаем копию DataFrame чтобы избежать предупреждений
    dff = dff.copy()

    # Преобразуем все целевые колонки в числовой формат
    for col in target_columns:
        dff[col] = pd.to_numeric(dff[col], errors='coerce')

    # Группировка по столбцу 'КССС' и суммирование данных
    df_bigdata = dff.groupby('КССС', as_index=False)[target_columns].sum()

    # Безопасный расчет интенсивности для дневной смены
    df_bigdata['Месячная среднаяя интенсивность (день)'] = np.round(
        df_bigdata['Прогноз дневных чеков (дневная смена)'] /
        df_bigdata['Совокупная сумма часов для дневной смены'].replace(0, np.nan), 1
    )

    # Безопасный расчет интенсивности для ночной смены
    mask_night = df_bigdata['Совокупная сумма часов для ночной смены'] != 0
    df_bigdata['Месячная среднаяя интенсивность (ночь)'] = '-'
    df_bigdata.loc[mask_night, 'Месячная среднаяя интенсивность (ночь)'] = np.round(
        df_bigdata.loc[mask_night, 'Прогноз ночных чеков (ночная смена)'] /
        df_bigdata.loc[mask_night, 'Совокупная сумма часов для ночной смены'], 1
    )

    # Безопасный расчет интенсивности для суток
    df_bigdata['Месячная среднаяя интенсивность (сутки)'] = np.round(
        df_bigdata['Итого прогноз чеков на сутки'] /
        df_bigdata['Итого целевая сумма часов на сутки'].replace(0, np.nan), 1
    )

    # Упорядочивание столбцов
    col_order = [
        'КССС',
        'Прогноз дневных чеков (дневная смена)',
        'Совокупная сумма часов для дневной смены',
        'Месячная среднаяя интенсивность (день)',
        'Прогноз ночных чеков (ночная смена)',
        'Совокупная сумма часов для ночной смены',
        'Месячная среднаяя интенсивность (ночь)',
        'Итого прогноз чеков на сутки',
        'Итого целевая сумма часов на сутки',
        'Месячная среднаяя интенсивность (сутки)'
    ]
    df_bigdata = df_bigdata[col_order]

    col_output = [
        'КССС',
        'Месячный прогноз дневных чеков',
        'Месячная сумма часов для дневной смены',
        'Месячная среднаяя интенсивность (день)',
        'Месячный прогноз ночных чеков',
        'Месячная сумма часов для ночной смены',
        'Месячная среднаяя интенсивность (ночь)',
        'Месячный прогноз чеков итого',
        'Месячная целевая сумма часов итого',
        'Месячная среднаяя интенсивность (сутки)'
    ]

    df_bigdata.columns = col_output

    # Заменяем NaN значения на прочерки
    intensity_cols = ['Месячная среднаяя интенсивность (день)',
                      'Месячная среднаяя интенсивность (ночь)',
                      'Месячная среднаяя интенсивность (сутки)']

    for col in intensity_cols:
        df_bigdata[col] = df_bigdata[col].replace(np.nan, '-')
        # Если значение float, округляем до 1 знака
        df_bigdata[col] = df_bigdata[col].apply(
            lambda x: round(x, 1) if isinstance(x, float) else x
        )

    return df_bigdata

def prediction_analys(file_name, path_to_save, new_name):
    begin_time = time.time()
    print('Функция запущен...')
    # Путь до файла с рассылкой
    df = pd.read_excel(file_name)
    # Преобразуем данные в нормальный датафрейм
    df = df_into_normal_view(df)
    # Расчет интенсивности в новом датафрейме
    df_intens = intensivity_counter(df)
    # Свернем до одного столбца массив с данными исходными, которые были преобразованы
    df_svertivaniye = df_melt_for_indicator(df)
    # Для каждого показателя создаем свой датафрейм и добавляем в список
    sp_of_df_by_indicator = dfs_all_indicators_by_one(df_svertivaniye)
    # Аналогичное делаем с массивом с интенсивностью
    df_intens_svertivaniye = intensivity_df_changer(df_intens)
    sp_of_df_intens_by_indicator = dfs_all_indicators_by_one(df_intens_svertivaniye)
    # Создаем общий список из несольких датафреймов для дальнейшего оббъединения
    sp_all_indicators = sp_of_df_by_indicator + sp_of_df_intens_by_indicator

    # Ищем имена, которые нужно использовать как колонки для будущего объединения
    names_col = []
    for i in sp_all_indicators[1:]:
        for j in i.columns:
            names_col.append(j)

    dict_names = {}
    for i in list(set(names_col)):
        dict_names[i] = names_col.count(i)

    sp_col_for_adding = [i for i in dict_names.keys() if dict_names[i] == 1]

    # Подготавливаем и объединяем файлы через сцепку КССС_День месяца_День недели
    df_itogoviy = sp_all_indicators[0]
    df_itogoviy['Сцепка'] = df_itogoviy['КССС'].astype(str) + "_" + df_itogoviy['Дата'].astype(str)

    for i in sp_all_indicators[1:]:
        i['Сцепка'] = i['КССС'].astype(str) + "_" + i['Дата'].astype(str)
        sp_col_for_concat = ['Сцепка']
        sp_col_for_concat += [j for j in i.columns if j in sp_col_for_adding]
        i = i[sp_col_for_concat]
        df_itogoviy = pd.merge(
                left=df_itogoviy,
                right=i,
                left_on='Сцепка',
                right_on='Сцепка',
                how='left'
        )
        # Меняем порядок колонок в датафрейме для лучшей визуализации
    cols_order = [
            'НПО', 'РУ', 'ТМ', 'Менеджер АЗС', 'Номер АЗС', 'КССС', 'Дата', 'День', 'День недели', 'Сцепка',
            'Прогноз дневных чеков (дневная смена)', 'Совокупная сумма часов для дневной смены',
            'Интенсивность труда (дневная смена)',
            'Прогноз ночных чеков (ночная смена)', 'Совокупная сумма часов для ночной смены',
            'Интенсивность труда (ночная смена)',
            'Итого прогноз чеков на сутки', 'Итого целевая сумма часов на сутки', 'Интенсивность труда суточная'
        ]
    df_itogoviy = df_itogoviy[cols_order]

    # Получаем информацию на больших цифрах в рамках месяца
    df_overall = bigdata_former(df_itogoviy)

    excel_file_name = f"{month}_result_{new_name}.xlsx"
    output_path = Path(path_to_save) / excel_file_name
    with pd.ExcelWriter(output_path) as writer:
        df_itogoviy.to_excel(writer, sheet_name='По дням')
        df_overall.to_excel(writer, sheet_name='Месяц')
    print(f'Время выполнения функции: {round(time.time() - begin_time, 2)} сек.')

    return df_itogoviy, df_overall
