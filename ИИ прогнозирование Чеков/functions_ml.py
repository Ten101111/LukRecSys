import os
import time

import numpy as np
from math import sqrt

import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor

from constants import NAME_OF_COLUMN_OF_KSSS, HORIZON, SEASONAL_PERIOD

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)

# ===========================
# N-BEATS (локальная модель)
# ===========================
NBEATS_CONFIG = {
    "epochs": 40,
    "batch_size": 64,
    "hidden_size": 128,
    "num_blocks": 3,
    "num_layers": 4,
    "lr": 1e-3,
    "seed": 42
}

if TORCH_AVAILABLE:
    class NBeatsBlock(nn.Module):
        def __init__(self, input_size, horizon, hidden_size, num_layers):
            super().__init__()
            layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
            self.backbone = nn.Sequential(*layers)
            self.theta = nn.Linear(hidden_size, input_size + horizon)
            self.input_size = input_size
            self.horizon = horizon

        def forward(self, x):
            h = self.backbone(x)
            theta = self.theta(h)
            backcast = theta[:, :self.input_size]
            forecast = theta[:, self.input_size:]
            return backcast, forecast

    class NBeats(nn.Module):
        def __init__(self, input_size, horizon, hidden_size, num_layers, num_blocks):
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    NBeatsBlock(
                        input_size=input_size,
                        horizon=horizon,
                        hidden_size=hidden_size,
                        num_layers=num_layers
                    )
                    for _ in range(num_blocks)
                ]
            )

        def forward(self, x):
            residual = x
            forecast = torch.zeros(x.size(0), self.blocks[0].horizon, device=x.device)
            for block in self.blocks:
                backcast, f = block(residual)
                residual = residual - backcast
                forecast = forecast + f
            return forecast


def _choose_nbeats_input_size(train_len, horizon, seasonal_period):
    min_size = max(seasonal_period * 4, 30)
    max_size = train_len - horizon - 1
    if max_size < min_size:
        return None
    target = max(horizon * 2, min_size)
    return min(target, max_size)


def _make_nbeats_windows(series, input_size, horizon):
    total = len(series)
    last_start = total - input_size - horizon
    if last_start < 0:
        return None, None
    xs, ys = [], []
    for i in range(last_start + 1):
        xs.append(series[i:i + input_size])
        ys.append(series[i + input_size:i + input_size + horizon])
    return np.stack(xs), np.stack(ys)


def _fit_nbeats(series, input_size, horizon, device, config):
    series = np.asarray(series, dtype=np.float32)
    mean = float(series.mean())
    std = float(series.std())
    if std == 0.0:
        std = 1.0
    series_scaled = (series - mean) / std

    x_np, y_np = _make_nbeats_windows(series_scaled, input_size, horizon)
    if x_np is None:
        raise ValueError("Недостаточно данных для обучения N-BEATS.")

    dataset = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])

    model = NBeats(
        input_size=input_size,
        horizon=horizon,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_blocks=config["num_blocks"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()
    model.train()

    for _ in range(config["epochs"]):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model, mean, std


def _forecast_nbeats(model, series, input_size, horizon, mean, std, device):
    series = np.asarray(series, dtype=np.float32)
    last_window = series[-input_size:]
    last_scaled = (last_window - mean) / std
    x = torch.from_numpy(last_scaled).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy().ravel()
    return pred_scaled * std + mean


def train_nbeats_forecast(series_train, horizon, seasonal_period, config=None):
    if not TORCH_AVAILABLE:
        raise RuntimeError(f"PyTorch недоступен: {TORCH_IMPORT_ERROR}")
    if config is None:
        config = NBEATS_CONFIG

    input_size = _choose_nbeats_input_size(len(series_train), horizon, seasonal_period)
    if input_size is None:
        raise ValueError("Недостаточно данных для окна N-BEATS.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = _fit_nbeats(series_train, input_size, horizon, device, config)
    forecast = _forecast_nbeats(model, series_train, input_size, horizon, mean, std, device)
    return {
        "forecast": forecast,
        "model": model,
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "device": device.type
    }

# ===========================
# 1 ЭТАП. ПОКАЗАТЕЛИ ТОЧНОСТИ
# ===========================

# Mean Absolut Persantage Error
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nz = y_true != 0
    if nz.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0

# Root Mean Square Error
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

# Mean Absolut Error
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def _sarimax_converged(model) -> bool:
    if getattr(model, "converged", None) is False:
        return False
    mle_retvals = getattr(model, "mle_retvals", None)
    if isinstance(mle_retvals, dict) and mle_retvals.get("converged") is False:
        return False
    return True

# ===========================
# 2 Этап. Очистка выбросов (Hampel)
# ===========================
def hampel_filter(series, window_size=15, n_sigmas=3):
    x = series.astype(float).copy()
    med = x.rolling(window=window_size, center=True, min_periods=1).median()
    diff = (x - med).abs()
    mad = diff.rolling(window=window_size, center=True, min_periods=1).median() * 1.4826
    thr = n_sigmas * mad.replace(0, np.nan)
    mask = diff > thr
    x[mask] = med[mask]
    return x, mask.fillna(False)

# ===========================
# 3 Этап. Календарные признаки
# ===========================
def add_calendar_features(df, date_col="Дата"):
    dt = df[date_col]
    df["day_of_week"] = dt.dt.weekday + 1
    df["is_weekend"] = (df["day_of_week"] >= 6).astype(int)
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_year"] = dt.dt.dayofyear
    df["dow_occurrence_in_month"] = ((df["day_of_month"] - 1) // 7) + 1
    # циклические кодировки
    df["sin_dow"] = np.sin(2 * np.pi * (df["day_of_week"] - 1) / 7)
    df["cos_dow"] = np.cos(2 * np.pi * (df["day_of_week"] - 1) / 7)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df

# ===========================
# 4 ЭТАП. Преобразование в формат для ML
# ===========================
def make_supervised(df, target_col="Чеки", lags=(1,7,14), roll_windows=(7,14)):
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[target_col].shift(L)
    for w in roll_windows:
        out[f"roll_mean_{w}"] = out[target_col].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out[target_col].shift(1).rolling(w).std()
    out = out.dropna()
    # исключаем целевую и дату из признаков
    feat_cols = [c for c in out.columns if c not in [target_col, "Дата"]]
    return out, feat_cols

# ===========================
# 5 ЭТАП. Пообъектный анализ
# ===========================
def ksss_choice(df, ksss):
    df = df[df[NAME_OF_COLUMN_OF_KSSS] == ksss]
    return df

def analysing_of_data(df_analyse):
    KSSS = df_analyse[NAME_OF_COLUMN_OF_KSSS].iloc[0]
    df_analyse = df_analyse.rename(columns={"Дата":"Дата", "Чеки":"Чеки"}).copy()
    df_analyse["Дата"] = pd.to_datetime(df_analyse["Дата"])
    df = df_analyse.sort_values("Дата").reset_index(drop=True)

    # гарантия сплошной дневной сетки
    full = pd.date_range(df["Дата"].min(), df["Дата"].max(), freq="D")
    df = df.set_index("Дата").reindex(full).rename_axis("Дата").reset_index()
    if df["Чеки"].isna().any():
        df["Чеки"] = df["Чеки"].interpolate(method="linear").ffill().bfill()
    # очистка выбросов
    df["Чеки"], outlier_mask = hampel_filter(df["Чеки"], window_size=15, n_sigmas=3)
    print(f"Исправлено выбросов (Hampel): {int(outlier_mask.sum())}")

    # признаки
    df = add_calendar_features(df, date_col="Дата")

    # сплит (последние 45 дней под валидацию)
    train_df = df.iloc[:-HORIZON].copy()
    test_df  = df.iloc[-HORIZON:].copy()
    y_train = train_df["Чеки"].values
    y_test  = test_df["Чеки"].values
    train_series = train_df.set_index("Дата")["Чеки"].astype(float)
    try:
        train_series = train_series.asfreq("D")
    except ValueError:
        pass

    # ===========================
    # 1) Holt–Winters
    # ===========================
    results = {}
    try:
        hw = ExponentialSmoothing(
            train_df["Чеки"], trend="add", seasonal="mul",
            seasonal_periods=SEASONAL_PERIOD, initialization_method="estimated"
        ).fit(optimized=True, use_brute=True)
        y_pred_hw = hw.forecast(HORIZON).values
        results["HoltWinters"] = {
            "pred": y_pred_hw,
            "MAPE": mape(y_test, y_pred_hw),
            "RMSE": rmse(y_test, y_pred_hw),
            "MAE" : mae(y_test, y_pred_hw),
            "model": hw
        }
    except Exception as e:
        results["HoltWinters"] = {"error": str(e)}

    # ===========================
    # 2) SARIMAX (небольшой перебор)
    # ===========================
    best_model, best_score, best_aic, best_cfg, best_pred = None, np.inf, np.inf, None, None
    p_vals, d_vals, q_vals = [0,1,2], [0,1], [0,1,2]
    P_vals, D_vals, Q_vals = [0,1], [0,1], [0,1]

    for p in p_vals:
        for d in d_vals:
            for q in q_vals:
                for P in P_vals:
                    for D in D_vals:
                        for Q in Q_vals:
                            try:
                                sar = SARIMAX(
                                    train_series,
                                    order=(p,d,q),
                                    seasonal_order=(P,D,Q,SEASONAL_PERIOD),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                ).fit(disp=False)
                                if not _sarimax_converged(sar):
                                    continue
                                y_pred = sar.forecast(HORIZON).values
                                score = mape(y_test, y_pred)
                                if np.isfinite(score):
                                    if score < best_score:
                                        best_score = score
                                        best_aic = sar.aic
                                        best_model = sar
                                        best_cfg = ((p,d,q),(P,D,Q,SEASONAL_PERIOD))
                                        best_pred = y_pred
                                elif best_model is None and sar.aic < best_aic:
                                    best_aic = sar.aic
                                    best_model = sar
                                    best_cfg = ((p,d,q),(P,D,Q,SEASONAL_PERIOD))
                                    best_pred = y_pred
                            except Exception:
                                pass

    if best_model is not None and best_pred is not None:
        y_pred_sar = best_pred
        results["SARIMAX"] = {
            "pred": y_pred_sar,
            "MAPE": mape(y_test, y_pred_sar),
            "RMSE": rmse(y_test, y_pred_sar),
            "MAE" : mae(y_test, y_pred_sar),
            "AIC" : best_aic if np.isfinite(best_aic) else np.nan,
            "order": best_cfg[0],
            "seasonal_order": best_cfg[1],
            "model": best_model
        }
    else:
        results["SARIMAX"] = {"error": "Не удалось обучить SARIMAX"}

    # ===========================
    # 3) RandomForest (с лагами, рекурсивный прогноз)
    # ===========================
    base_cols = ["Дата","Чеки","day_of_week","is_weekend","day_of_month","month",
                 "week_of_year","day_of_year","dow_occurrence_in_month",
                 "sin_dow","cos_dow","sin_month","cos_month"]

    sup, feat_cols = make_supervised(df[base_cols].copy(), target_col="Чеки")

    cutoff_date = test_df["Дата"].min()
    sup_train = sup[sup["Дата"] < cutoff_date].copy()

    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(sup_train[feat_cols], sup_train["Чеки"])

    # рекурсивный прогноз на тестовый горизонт
    history = sup[sup["Дата"] < cutoff_date][["Дата","Чеки"]].copy()
    rf_preds = []

    for i in range(HORIZON):
        current_date = cutoff_date + pd.Timedelta(days=i)
        row = df[df["Дата"] == current_date][["Дата","day_of_week","is_weekend","day_of_month","month",
                                               "week_of_year","day_of_year","dow_occurrence_in_month",
                                               "sin_dow","cos_dow","sin_month","cos_month"]].copy()
        if row.empty:
            continue
        hist_series = history.set_index("Дата")["Чеки"]

        for L in (1,7,14):
            row[f"lag_{L}"] = hist_series.reindex([current_date - pd.Timedelta(days=L)]).values
        for w in (7,14):
            roll = hist_series.loc[:current_date - pd.Timedelta(days=1)].tail(w)
            row[f"roll_mean_{w}"] = roll.mean() if len(roll) else np.nan
            row[f"roll_std_{w}"]  = roll.std(ddof=0) if len(roll) > 1 else 0.0

        for c in feat_cols:
            if c not in row.columns:
                row[c] = 0.0
        row = row[feat_cols].fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        y_hat = rf.predict(row)[0]
        rf_preds.append(y_hat)
        history = pd.concat([history, pd.DataFrame({"Дата":[current_date], "Чеки":[y_hat]})], ignore_index=True)

    results["RandomForest"] = {
        "pred": np.array(rf_preds),
        "MAPE": mape(y_test, rf_preds),
        "RMSE": rmse(y_test, rf_preds),
        "MAE" : mae(y_test, rf_preds),
        "model": rf,
        "features": feat_cols
    }

    # ===========================
    # 4) N-BEATS (локальная нейросеть)
    # ===========================
    if TORCH_AVAILABLE:
        try:
            nb_res = train_nbeats_forecast(
                series_train=train_df["Чеки"].values,
                horizon=HORIZON,
                seasonal_period=SEASONAL_PERIOD
            )
            y_pred_nb = nb_res["forecast"]
            results["NBEATS"] = {
                "pred": y_pred_nb,
                "MAPE": mape(y_test, y_pred_nb),
                "RMSE": rmse(y_test, y_pred_nb),
                "MAE": mae(y_test, y_pred_nb),
                "model": nb_res["model"],
                "input_size": nb_res["input_size"],
                "device": nb_res["device"]
            }
        except Exception as e:
            results["NBEATS"] = {"error": str(e)}
    else:
        results["NBEATS"] = {"error": f"PyTorch недоступен: {TORCH_IMPORT_ERROR}"}
    # ===========================
    # Сравнение моделей
    # ===========================
    rows = []
    for name, res in results.items():
        if all(k in res for k in ["MAPE","RMSE","MAE"]):
            rows.append([name, round(res["MAPE"],3), round(res["RMSE"],3), round(res["MAE"],3)])
        else:
            rows.append([name, None, None, None])
    metrics_df = pd.DataFrame(rows, columns=["Model","MAPE","RMSE","MAE"]).sort_values("RMSE", na_position="last")
    print(f"\nМетрики на отложенных {HORIZON} днях (чем меньше, тем лучше):")
    print(metrics_df.to_string(index=False))

    best_model_name = metrics_df.iloc[0]["Model"]
    print(f"\nЛучшая модель для {KSSS} по RMSE: \033[1m{best_model_name}\033[0m")

    best_mape_ind = results[best_model_name]["MAPE"]
    return (
        best_model_name,  # название лучшей модели (по RMSE)
        best_mape_ind,  # её MAPE
        df,  # анализированный DataFrame
        results,  # все результаты
        KSSS,  # название АЗС
        base_cols  # базовые признаки
    )

def build_best_models_dataset(df_all):
    """
    Формирует DataFrame с колонками:
    [АЗС, Лучшая модель, MAPE]
    перебирая все значения NAME_OF_COLUMN_OF_KSSS в df_all.
    """
    ksss_all = list(df_all[NAME_OF_COLUMN_OF_KSSS].unique())
    len_ksss_all = len(ksss_all)+1
    rows = []
    num = 1
    for ksss, df_k in df_all.groupby(NAME_OF_COLUMN_OF_KSSS):
        print(f'▶ \033[1;4mНачали обрабатывать АЗС с КССС {ksss}\033[0m - {num}/{len_ksss_all}\n')
        print('Обработка...\n')
        start = time.time()
        try:
            best_model_name, best_mape, *_ = analysing_of_data(df_k)
            rows.append({
                "АЗС": ksss,
                "Лучшая модель": best_model_name,
                "MAPE": round(float(best_mape), 3) if pd.notnull(best_mape) else np.nan
            })
        except Exception as e:
            # Если по какой-то причине не удалось посчитать
            rows.append({
                "АЗС": ksss,
                "Лучшая модель": None,
                "MAPE": np.nan,
                "error": str(e)
            })
        print(f'✅\033[3mВыполнено для АЗС {ksss} за {int(round(time.time()-start, 0))} сек.\033[0m\n')
        num+=1
    summary_df = pd.DataFrame(rows)
    # Если есть колонка error, выведите сначала успешные строки
    if "error" in summary_df.columns:
        summary_df = summary_df.sort_values(by=["Лучшая модель","MAPE"], na_position="last")
    else:
        summary_df = summary_df.sort_values(by="MAPE", na_position="last")
    return summary_df


def prediction_by_mlearning(best_model_name, df, results, KSSS, base_cols):
    # ===========================
    # Финальная подгонка и прогноз на 45 дней вперёд
    # ===========================
    last_date = df["Дата"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON, freq="D")

    if best_model_name == "HoltWinters":
        final = ExponentialSmoothing(
            df["Чеки"], trend="add", seasonal="mul",
            seasonal_periods=SEASONAL_PERIOD, initialization_method="estimated"
        ).fit(optimized=True, use_brute=True)
        future_pred = final.forecast(HORIZON).values

    elif best_model_name == "SARIMAX":
        order = results["SARIMAX"]["order"]
        seas  = results["SARIMAX"]["seasonal_order"]
        series = df.set_index("Дата")["Чеки"].astype(float)
        try:
            series = series.asfreq("D")
        except ValueError:
            pass
        final = SARIMAX(
            series, order=order, seasonal_order=seas,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        future_pred = final.forecast(HORIZON).values

    elif best_model_name == "NBEATS":
        nb_res = train_nbeats_forecast(
            series_train=df["Чеки"].values,
            horizon=HORIZON,
            seasonal_period=SEASONAL_PERIOD
        )
        future_pred = nb_res["forecast"]

    else:  # RandomForest
        sup_full, feat_cols_full = make_supervised(df[base_cols].copy(), target_col="Чеки")
        rf_final = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        rf_final.fit(sup_full[feat_cols_full], sup_full["Чеки"])

        history = df[["Дата","Чеки"]].copy()
        preds = []
        for i in range(HORIZON):
            cur = future_dates[i]
            row = pd.DataFrame({"Дата":[cur]})
            row = add_calendar_features(row, date_col="Дата")
            hist = history.set_index("Дата")["Чеки"]
            for L in (1,7,14):
                row[f"lag_{L}"] = hist.reindex([cur - pd.Timedelta(days=L)]).values
            for w in (7,14):
                roll = hist.loc[:cur - pd.Timedelta(days=1)].tail(w)
                row[f"roll_mean_{w}"] = roll.mean() if len(roll) else np.nan
                row[f"roll_std_{w}"]  = roll.std(ddof=0) if len(roll) > 1 else 0.0
            for c in feat_cols_full:
                if c not in row.columns:
                    row[c] = 0.0
            row = row[feat_cols_full].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
            y_hat = rf_final.predict(row)[0]
            preds.append(y_hat)
            history = pd.concat([history, pd.DataFrame({"Дата":[cur], "Чеки":[y_hat]})], ignore_index=True)
        future_pred = np.array(preds)

    forecast_df = pd.DataFrame({
        "Дата": future_dates,
        f"Прогноз_чеков {KSSS}": np.round(future_pred, 0).astype(int)
    })
    # forecast_df = forecast_df.drop(NAME_OF_COLUMN_OF_KSSS, axis=1)
    return forecast_df


def path_saver(df, OUTPUT_PATH_DIR, name_for_saving):
    # ===========================
    # Сохранение результатов
    # ===========================
    output_dir  = Path(OUTPUT_PATH_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(OUTPUT_PATH_DIR, f"{name_for_saving}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    # df.to_excel(OUTPUT_PATH_DIR+f"{name_for_saving}.xlsx", index=False)
    print('-'*20, 'СОХРАНЕНО!', '-'*20)
