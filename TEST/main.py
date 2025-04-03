import os
import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from tinkoff.invest import CandleInterval, AsyncClient, Client
from tinkoff.invest.utils import now, quotation_to_decimal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

TOKEN = "t.5dSsLD4wC3kwvuyy3gKcX0FBuZDBFje4ZuE_d77OgvTZ-YxPkwXCAnRTo_6ed_CxF2z6hqJ1crS0zOH8n-e5Cg"

def get_figi_by_ticker(ticker: str) -> str:
    with Client(TOKEN) as client:
        instruments = []
        for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
            instruments += getattr(client.instruments, method)().instruments
        filtered = [inst for inst in instruments if inst.ticker.upper() == ticker.upper()]
        if filtered:
            return filtered[0].figi
        else:
            raise ValueError("Инструмент не найден")

async def fetch_candles(figi: str, days: int = 5, interval=CandleInterval.CANDLE_INTERVAL_1_MIN):
    candles = []
    async with AsyncClient(TOKEN) as client:
        async for candle in client.get_all_candles(
            figi=figi,
            from_=now() - timedelta(days=days),
            to=now(),
            interval=interval
        ):
            candles.append({
                "time": candle.time,
                "open": float(quotation_to_decimal(candle.open)),
                "high": float(quotation_to_decimal(candle.high)),
                "low": float(quotation_to_decimal(candle.low)),
                "close": float(quotation_to_decimal(candle.close)),
                "volume": candle.volume
            })
    df = pd.DataFrame(candles)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)
        df.set_index("time", inplace=True)
    return df

def compute_indicators(df):
    df["SMA_10"] = df["close"].rolling(window=10).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["STD_20"] = df["close"].rolling(window=20).std()
    df["BB_upper"] = df["SMA_20"] + 2 * df["STD_20"]
    df["BB_lower"] = df["SMA_20"] - 2 * df["STD_20"]
    df.drop(columns=["SMA_20", "STD_20"], inplace=True)
    df.dropna(inplace=True)
    return df

def create_sequences(X, Y, window_size, horizons):
    X_seq = []
    y_seq = []
    n = len(X)
    max_h = max(horizons)
    for i in range(window_size, n - max_h + 1):
        X_seq.append(X[i-window_size:i])
        y_seq.append([Y[i-1+h][0] for h in horizons])
    return np.array(X_seq), np.array(y_seq)

def create_model(units, dropout_rate, l2_factor, input_shape, output_size):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, kernel_regularizer=l2(l2_factor), input_shape=input_shape))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, kernel_regularizer=l2(l2_factor)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(output_size, kernel_regularizer=l2(l2_factor)))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def grid_search(X, y, param_grid, n_splits=3, epochs=5, batch_size=32):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params = None
    best_score = np.inf
    for units in param_grid["units"]:
        for dropout_rate in param_grid["dropout_rate"]:
            for l2_factor in param_grid["l2_factor"]:
                scores = []
                for train_index, val_index in tscv.split(X):
                    X_train_fold, X_val_fold = X[train_index], X[val_index]
                    y_train_fold, y_val_fold = y[train_index], y[val_index]
                    model = create_model(units, dropout_rate, l2_factor, (X.shape[1], X.shape[2]), y.shape[1])
                    model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0)
                    preds = model.predict(X_val_fold, verbose=0)
                    actual = y_val_fold[:, 0]
                    pred = preds[:, 0]
                    mape = np.mean(np.abs((actual - pred) / actual)) * 100
                    scores.append(mape)
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {"units": units, "dropout_rate": dropout_rate, "l2_factor": l2_factor}
    return best_params, best_score

def get_account_balance():
    with Client(TOKEN) as client:
        accounts = client.users.get_accounts().accounts
        if not accounts:
            return 0.0
        account_id = accounts[0].id
        positions = client.operations.get_positions(account_id=account_id)
        if positions.money and len(positions.money) > 0:
            return float(quotation_to_decimal(positions.money[0]))
        return 0.0

async def train_and_predict(figi: str, window_size: int = 10):
    df = await fetch_candles(figi, days=5, interval=CandleInterval.CANDLE_INTERVAL_1_MIN)
    if df.empty:
        print("Нет данных свечей для данного тикера")
        return None
    df = compute_indicators(df)
    features = df[["open", "high", "low", "close", "volume", "SMA_10", "RSI_14", "MACD", "BB_upper", "BB_lower"]].values
    target = df[["close"]].values
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    scaled_X = scaler_X.fit_transform(features)
    scaled_Y = scaler_Y.fit_transform(target)
    horizons = [5, 15, 30]
    X_all, y_all = create_sequences(scaled_X, scaled_Y, window_size, horizons)
    total = len(X_all)
    split_idx = int(total * 0.8)
    X_cv = X_all[:split_idx]
    y_cv = y_all[:split_idx]
    X_test = X_all[split_idx:]
    y_test = y_all[split_idx:]
    param_grid = {"units": [50, 100], "dropout_rate": [0.0, 0.2], "l2_factor": [0.0, 0.001]}
    best_params, cv_score = grid_search(X_cv, y_cv, param_grid, n_splits=3, epochs=5, batch_size=32)
    model = create_model(best_params["units"], best_params["dropout_rate"], best_params["l2_factor"], (X_cv.shape[1], X_cv.shape[2]), y_cv.shape[1])
    model.fit(X_cv, y_cv, epochs=5, batch_size=32, verbose=0)
    test_preds = model.predict(X_test, verbose=0)
    actual_test = scaler_Y.inverse_transform(y_test[:, 0].reshape(-1, 1)).flatten()
    pred_test = scaler_Y.inverse_transform(test_preds[:, 0].reshape(-1, 1)).flatten()
    mape = np.mean(np.abs((actual_test - pred_test) / actual_test)) * 100
    last_seq = scaled_X[-window_size:]
    last_seq = last_seq.reshape(1, window_size, scaled_X.shape[1])
    future_scaled = model.predict(last_seq, verbose=0)[0]
    future = scaler_Y.inverse_transform(future_scaled.reshape(-1, 1)).flatten()
    last_close = df["close"].iloc[-1]
    return {"predictions": {"5 минут": future[0], "15 минут": future[1], "30 минут": future[2]}, "last_close": last_close, "mape": mape, "best_params": best_params}

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

def print_report(ticker, figi, last_close, predictions, current_time, mape, best_params, balance):
    trend = lambda p: "↑" if p > last_close else "↓" if p < last_close else "-"
    report = "============================================================\n"
    report += f"Ticker: {ticker.upper()}    FIGI: {figi}\n"
    report += f"Дата: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"Счет: {balance:.2f} RUB\n"
    report += f"Последняя цена закрытия: {last_close:.2f}\n"
    report += "Прогноз на будущие интервалы:\n"
    for horizon in ["5 минут", "15 минут", "30 минут"]:
        if horizon in predictions:
            report += f"  {horizon}: {predictions[horizon]:.2f} {trend(predictions[horizon])}\n"
    report += f"\nОценка точности прогноза (MAPE): {mape:.2f}%\n"
    report += f"Лучшие параметры: units={best_params['units']}, dropout_rate={best_params['dropout_rate']}, l2_factor={best_params['l2_factor']}\n"
    report += "============================================================\n"
    print(report)

async def main():
    ticker = input("Введите тикер: ")
    figi = get_figi_by_ticker(ticker)
    while True:
        result = await train_and_predict(figi, window_size=10)
        if result is not None:
            predictions = result["predictions"]
            last_close = result["last_close"]
            mape = result["mape"]
            best_params = result["best_params"]
            balance = get_account_balance()
            current_time = datetime.now()
            clear_terminal()
            print_report(ticker, figi, last_close, predictions, current_time, mape, best_params, balance)
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
