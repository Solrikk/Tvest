import os
import asyncio
import pandas as pd
from datetime import timedelta, datetime
from tinkoff.invest import CandleInterval, AsyncClient, Client
from tinkoff.invest.utils import now, quotation_to_decimal
from sklearn.linear_model import LinearRegression

TOKEN = "t.5dSsLD4wC3kwvuyy3gKcX0FBuZDBFje4ZuE_d77OgvTZ-YxPkwXCAnRTo"

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

def create_lag_features(df: pd.DataFrame, lags: int = 3):
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['close'].shift(i)
    return df.dropna()

def predict_price_movement(df: pd.DataFrame):
    df_feat = create_lag_features(df.copy(), lags=3)
    X = df_feat[[f'lag_{i}' for i in range(1, 4)]].values
    y = df_feat['close'].values
    model = LinearRegression().fit(X, y)
    last_values = df['close'].values[-3:]
    return model.predict([last_values])[0]

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_report(ticker, figi, last_close, predicted_price, current_time):
    trend = "↑" if predicted_price > last_close else "↓" if predicted_price < last_close else "-"
    color_pred = "\033[1;32m" if predicted_price > last_close else "\033[1;31m" if predicted_price < last_close else "\033[0m"
    reset = "\033[0m"
    report = f"""
============================================================
Ticker: {ticker.upper()}    FIGI: {figi}
Дата: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
Последняя цена закрытия: {last_close:.2f}
Прогнозированная цена: {color_pred}{predicted_price:.2f} {trend}{reset}
============================================================
"""
    print(report)

async def main():
    ticker = input("Введите тикер: ")
    figi = get_figi_by_ticker(ticker)
    while True:
        df = await fetch_candles(figi, days=90)
        if df.empty:
            print("Нет данных свечей для данного тикера")
        else:
            predicted_price = predict_price_movement(df)
            last_close = df['close'].iloc[-1]
            current_time = datetime.now()
            clear_terminal()
            print_report(ticker, figi, last_close, predicted_price, current_time)
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
