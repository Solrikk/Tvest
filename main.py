import os
import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from tinkoff.invest import CandleInterval, AsyncClient, Client
from tinkoff.invest.utils import now, quotation_to_decimal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

TOKEN = "t.5dSsLD4wC3kwvuyy3gKcX0o_6ed_CxF2z6hqJ1crS0zOH8n-e5Cg"

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

async def fetch_candles(figi: str, days: int = 90):
    candles = []
    async with AsyncClient(TOKEN) as client:
        async for candle in client.get_all_candles(
            figi=figi,
            from_=now() - timedelta(days=days),
            to=now(),
            interval=CandleInterval.CANDLE_INTERVAL_HOUR,
        ):
            candles.append({
                'time': candle.time,
                'open': float(quotation_to_decimal(candle.open)),
                'high': float(quotation_to_decimal(candle.high)),
                'low': float(quotation_to_decimal(candle.low)),
                'close': float(quotation_to_decimal(candle.close)),
                'volume': candle.volume
            })
    df = pd.DataFrame(candles)
    if not df.empty:
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values('time', inplace=True)
        df.set_index('time', inplace=True)
    return df

def calculate_indicators(df: pd.DataFrame):
    df["sma20"] = df["close"].rolling(window=20).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["std20"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["sma20"] + 2 * df["std20"]
    df["bb_lower"] = df["sma20"] - 2 * df["std20"]
    low_min = df["low"].rolling(window=14).min()
    high_max = df["high"].rolling(window=14).max()
    df["stoch"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()
    return df

def create_features(df: pd.DataFrame, lags: int = 3):
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['close'].shift(i)
    df = df.dropna()
    feature_cols = [f'lag_{i}' for i in range(1, lags + 1)] + ["sma20", "ema20", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "stoch", "ema50", "vol_ma20"]
    return df, feature_cols

def train_model_enhanced(df: pd.DataFrame, lags: int = 3):
    df_feat, feature_cols = create_features(df.copy(), lags)
    X = df_feat[feature_cols].values
    y = df_feat['close'].values.reshape(-1, 1)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).ravel()
    model = LinearRegression().fit(X_scaled, y_scaled)
    return model, scaler_X, scaler_y, feature_cols

def predict_multiple_steps_enhanced(df: pd.DataFrame, steps: int = 5, lags: int = 3):
    model, scaler_X, scaler_y, feature_cols = train_model_enhanced(df, lags)
    last_row = df.iloc[-1]
    last_lags = df['close'].values[-lags:].tolist()
    static_features = [last_row[col] for col in feature_cols if col not in [f'lag_{i}' for i in range(1, lags + 1)]]
    predictions = []
    current_lags = last_lags.copy()
    for _ in range(steps):
        feature_vector = current_lags + static_features
        feature_vector_np = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = scaler_X.transform(feature_vector_np)
        pred_scaled = model.predict(feature_vector_scaled)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        predictions.append(pred)
        current_lags.append(pred)
        current_lags.pop(0)
    return predictions

def analyze_market(ind, last_close):
    score_bull = 0
    score_bear = 0
    if ind["rsi"] < 30:
        score_bull += 1
    elif ind["rsi"] > 70:
        score_bear += 1
    if ind["macd"] > ind["macd_signal"]:
        score_bull += 1
    elif ind["macd"] < ind["macd_signal"]:
        score_bear += 1
    if last_close < ind["bb_lower"]:
        score_bull += 1
    elif last_close > ind["bb_upper"]:
        score_bear += 1
    if score_bull > score_bear:
        return "Сигнал к ЛОНГУ"
    elif score_bear > score_bull:
        return "Сигнал к ШОРТУ"
    else:
        return "Рынок нейтрален, рекомендуется выжидать"

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_report(ticker, figi, last_close, predictions, current_time, indicators, analysis):
    predicted_price = predictions[0]
    trend_symbol = "↑" if predicted_price > last_close else "↓" if predicted_price < last_close else "-"
    color_pred = "\033[1;32m" if predicted_price > last_close else "\033[1;31m" if predicted_price < last_close else "\033[0m"
    reset = "\033[0m"
    multi_preds_str = "\n".join([f"Шаг {i+1}: {p:.2f}" for i, p in enumerate(predictions)])
    rep_ind = f"EMA20: {indicators['ema20']:.2f} | SMA20: {indicators['sma20']:.2f} | RSI: {indicators['rsi']:.2f} | MACD: {indicators['macd']:.2f} | MACD Signal: {indicators['macd_signal']:.2f} | BB Upper: {indicators['bb_upper']:.2f} | BB Lower: {indicators['bb_lower']:.2f} | Stoch: {indicators['stoch']:.2f} | EMA50: {indicators['ema50']:.2f} | Vol_MA20: {indicators['vol_ma20']:.2f}"
    report = f"============================================================\nTicker: {ticker.upper()}    FIGI: {figi}\nДата: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\nПоследняя цена закрытия: {last_close:.2f}\nБлижайший прогноз: {color_pred}{predicted_price:.2f} {trend_symbol}{reset}\nДальнейшие {len(predictions)} прогнозируемых цен:\n{multi_preds_str}\n============================================================\nИндикаторы: {rep_ind}\nАнализ рынка: {analysis}\n============================================================"
    print(report)

async def main():
    ticker = input("Введите тикер: ")
    steps = int(input("Сколько шагов вперед предсказывать? (например, 5): "))
    figi = get_figi_by_ticker(ticker)
    while True:
        df = await fetch_candles(figi, days=90)
        if df.empty:
            print("Нет данных свечей для данного тикера")
        else:
            df = calculate_indicators(df)
            predictions = predict_multiple_steps_enhanced(df, steps=steps, lags=3)
            last_close = df["close"].iloc[-1]
            current_time = datetime.now()
            indicators = {
                "ema20": df["ema20"].iloc[-1],
                "sma20": df["sma20"].iloc[-1],
                "rsi": df["rsi"].iloc[-1],
                "macd": df["macd"].iloc[-1],
                "macd_signal": df["macd_signal"].iloc[-1],
                "bb_upper": df["bb_upper"].iloc[-1],
                "bb_lower": df["bb_lower"].iloc[-1],
                "stoch": df["stoch"].iloc[-1],
                "ema50": df["ema50"].iloc[-1],
                "vol_ma20": df["vol_ma20"].iloc[-1]
            }
            analysis = analyze_market(indicators, last_close)
            clear_terminal()
            print_report(ticker, figi, last_close, predictions, current_time, indicators, analysis)
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
