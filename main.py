import os
import asyncio
import pandas as pd
import numpy as np
import json
import pickle
from datetime import timedelta, datetime
from tinkoff.invest import CandleInterval, AsyncClient, Client
from tinkoff.invest.utils import now, quotation_to_decimal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

TOKEN = "t.5dSsLgvTZ-YxPkwXCAnRTo_6ed_CxF2z6hqJ1crS0zOH8n-e5Cg"


def get_figi_by_ticker(ticker: str) -> str:
    with Client(TOKEN) as client:
        instruments = []
        for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
            instruments += getattr(client.instruments, method)().instruments
        filtered = [
            inst for inst in instruments
            if inst.ticker.upper() == ticker.upper()
        ]
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
                to=now() - timedelta(minutes=60),
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

        async for candle in client.get_all_candles(
                figi=figi,
                from_=now() - timedelta(minutes=60),
                to=now(),
                interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
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


def preprocess_data(df: pd.DataFrame):
    df["last_raw_close"] = df["close"].iloc[-1] if not df.empty else None

    df["close_smooth"] = df["close"].rolling(window=2, min_periods=1).mean()
    roll_med = df["close_smooth"].rolling(window=10, min_periods=1).median()
    roll_std = df["close_smooth"].rolling(window=10, min_periods=1).std().fillna(0)

    diff = np.abs(df["close_smooth"] - roll_med)
    threshold = 2.5 * roll_std

    df["close_clean"] = np.where(diff > threshold, roll_med, df["close_smooth"])

    last_hour_mask = df.index >= (df.index.max() - pd.Timedelta(hours=1))
    if any(last_hour_mask):
        df.loc[last_hour_mask, "close_clean"] = df.loc[last_hour_mask, "close"]

    return df


def calculate_indicators(df: pd.DataFrame):
    df["sma20"] = df["close_clean"].rolling(window=20).mean()
    df["ema20"] = df["close_clean"].ewm(span=20, adjust=False).mean()
    delta = df["close_clean"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = df["close_clean"].ewm(span=12, adjust=False).mean()
    ema26 = df["close_clean"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["std20"] = df["close_clean"].rolling(window=20).std()
    df["bb_upper"] = df["sma20"] + 2 * df["std20"]
    df["bb_lower"] = df["sma20"] - 2 * df["std20"]
    low_min = df["low"].rolling(window=14).min()
    high_max = df["high"].rolling(window=14).max()
    df["stoch"] = 100 * (df["close_clean"] - low_min) / (high_max - low_min)
    df["ema50"] = df["close_clean"].ewm(span=50, adjust=False).mean()
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()
    return df


def create_fourier_features(df: pd.DataFrame, periods=[24, 168], order=3):
    df = df.copy()
    df["time_int"] = df.index.astype(np.int64) // 10**9
    for period in periods:
        for i in range(1, order + 1):
            df[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * df["time_int"] /
                                             (period * 3600))
            df[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * df["time_int"] /
                                             (period * 3600))
    df.drop(columns=["time_int"], inplace=True)
    return df


def create_features(df: pd.DataFrame, lags: int = 3):
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['close_clean'].shift(i)

    for i in range(1, lags + 1):
        lag_col = f'lag_{i}'
        if lag_col in df.columns:
            df[lag_col] = df[lag_col].fillna(method='bfill').fillna(method='ffill')

    for col in ["sma20", "ema20", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
                "stoch", "ema50", "vol_ma20"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

    df = create_fourier_features(df)

    df = df.dropna()

    feature_cols = [f'lag_{i}' for i in range(1, lags + 1)] + [
        "sma20", "ema20", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
        "stoch", "ema50", "vol_ma20"
    ]
    fourier_cols = [
        col for col in df.columns
        if col.startswith("sin_") or col.startswith("cos_")
    ]
    feature_cols += fourier_cols

    feature_cols = [col for col in feature_cols if col in df.columns]

    return df, feature_cols


def train_model_enhanced(df: pd.DataFrame, lags: int = 3):
    df_feat, feature_cols = create_features(df.copy(), lags)

    if len(df_feat) < 30:
        print("Предупреждение: Недостаточно данных для надежного обучения модели.")

    train_size = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:train_size]
    valid_df = df_feat.iloc[train_size:]

    X = df_feat[feature_cols].values
    y = df_feat['close_clean'].values.reshape(-1, 1)

    for i in range(X.shape[1]):
        col_data = X[:, i]
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        X[:, i] = np.clip(col_data, lower_bound, upper_bound)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).ravel()

    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.1).fit(X_scaled, y_scaled)

    if len(valid_df) > 0:
        X_valid = valid_df[feature_cols].values
        y_valid = valid_df['close_clean'].values

        for i in range(X_valid.shape[1]):
            col_data = X_valid[:, i]
            X_valid[:, i] = np.clip(col_data, lower_bound, upper_bound)

        X_valid_scaled = scaler_X.transform(X_valid)
        y_valid_scaled = scaler_y.transform(y_valid.reshape(-1, 1)).ravel()

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        pred_valid_scaled = model.predict(X_valid_scaled)
        pred_valid = scaler_y.inverse_transform(pred_valid_scaled.reshape(-1, 1)).ravel()

        mae = mean_absolute_error(y_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
        r2 = r2_score(y_valid, pred_valid)

        print(f"Валидация модели: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        if r2 < 0.5:
            print("Предупреждение: Низкое качество прогнозов. Рекомендуется увеличить объем данных.")

    return model, scaler_X, scaler_y, feature_cols, df_feat


def predict_multiple_steps_enhanced(df: pd.DataFrame,
                                    steps: int = 5,
                                    lags: int = 3):
    model, scaler_X, scaler_y, feature_cols, df_feat = train_model_enhanced(
        df, lags)

    pred_df = df_feat.tail(1).copy()

    current_features = []
    for col in feature_cols:
        value = pred_df[col].iloc[-1]
        if pd.isna(value):
            value = df_feat[col].mean()
            if pd.isna(value):
                value = 0
        current_features.append(value)

    feature_vector_np = np.array(current_features).reshape(1, -1)
    if np.isnan(feature_vector_np).any():
        feature_vector_np = np.nan_to_num(feature_vector_np, nan=0.0)

    feature_vector_scaled = scaler_X.transform(feature_vector_np)

    pred_scaled = model.predict(feature_vector_scaled)
    pred_1min = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

    time_intervals = [1, 5, 15, 30]
    predictions = []

    last_n_values = df['close_clean'].tail(60).values
    last_values_diff = np.diff(last_n_values)

    mean_change = np.mean(last_values_diff)
    median_change = np.median(last_values_diff)
    std_change = np.std(last_values_diff)

    trend_strength = mean_change / (std_change if std_change > 0 else 0.0001)

    consecutive_up = 0
    consecutive_down = 0
    for i in range(len(last_values_diff) - 1, 0, -1):
        if last_values_diff[i] > 0:
            consecutive_up += 1
            consecutive_down = 0
        elif last_values_diff[i] < 0:
            consecutive_down += 1
            consecutive_up = 0
        else:
            consecutive_up = 0
            consecutive_down = 0

    tech_indicators_bullish = 0
    tech_indicators_bearish = 0

    if df['rsi'].iloc[-1] < 30:
        tech_indicators_bullish += 1
    elif df['rsi'].iloc[-1] > 70:
        tech_indicators_bearish += 1

    if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
        tech_indicators_bullish += 1
    else:
        tech_indicators_bearish += 1

    bb_position = (df['close_clean'].iloc[-1] - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
    if bb_position < 0.2:
        tech_indicators_bullish += 1
    elif bb_position > 0.8:
        tech_indicators_bearish += 1

    if df['close_clean'].iloc[-1] > df['ema20'].iloc[-1] and df['ema20'].iloc[-1] > df['ema50'].iloc[-1]:
        tech_indicators_bullish += 1
    elif df['close_clean'].iloc[-1] < df['ema20'].iloc[-1] and df['ema20'].iloc[-1] < df['ema50'].iloc[-1]:
        tech_indicators_bearish += 1

    indicator_consensus = tech_indicators_bullish - tech_indicators_bearish

    for interval in time_intervals:
        if interval == 1:
            base_prediction = pred_1min

            adjustment_factor = 0.001 * indicator_consensus * df['close_clean'].iloc[-1]
            predictions.append(base_prediction + adjustment_factor)
            continue

        base_forecast = pred_1min + (mean_change * interval)

        trend_adjustment = trend_strength * std_change * np.sqrt(interval) * 0.5

        tech_adjustment = 0.002 * indicator_consensus * df['close_clean'].iloc[-1] * (interval / 5)

        momentum_adjustment = 0
        if consecutive_up > 3:
            momentum_adjustment = 0.001 * consecutive_up * df['close_clean'].iloc[-1] * (interval / 10)
        elif consecutive_down > 3:
            momentum_adjustment = -0.001 * consecutive_down * df['close_clean'].iloc[-1] * (interval / 10)

        volatility_adjustment = std_change * np.sqrt(interval) * 0.3

        contrarian_adjustment = 0
        if df['rsi'].iloc[-1] > 75:
            contrarian_adjustment = -0.003 * (df['rsi'].iloc[-1] - 75) * df['close_clean'].iloc[-1] * (interval / 15)
        elif df['rsi'].iloc[-1] < 25:
            contrarian_adjustment = 0.003 * (25 - df['rsi'].iloc[-1]) * df['close_clean'].iloc[-1] * (interval / 15)

        if interval <= 5:
            final_prediction = (
                base_forecast + 
                trend_adjustment * 0.6 +
                tech_adjustment * 0.2 +
                momentum_adjustment * 0.15 +
                volatility_adjustment * 0.05 +
                contrarian_adjustment * 0.0
            )
        elif interval <= 15:
            final_prediction = (
                base_forecast +
                trend_adjustment * 0.4 +
                tech_adjustment * 0.25 +
                momentum_adjustment * 0.15 +
                volatility_adjustment * 0.1 +
                contrarian_adjustment * 0.1
            )
        else:
            final_prediction = (
                base_forecast +
                trend_adjustment * 0.25 +
                tech_adjustment * 0.2 +
                momentum_adjustment * 0.1 +
                volatility_adjustment * 0.2 +
                contrarian_adjustment * 0.25
            )

        predictions.append(final_prediction)

    return predictions


def detect_regime(df: pd.DataFrame):
    try:
        mr = MarkovRegression(df["close_clean"],
                              k_regimes=2,
                              trend='c',
                              switching_variance=True)
        res = mr.fit(disp=False)
        regimes = res.smoothed_marginal_probabilities.iloc[-1].to_dict()
        return regimes
    except Exception as e:
        return {}


def analyze_market(ind, current_price, predictions, df=None):
    score_bull = 0
    score_bear = 0
    reason_bull = []
    reason_bear = []

    if ind["rsi"] < 30:
        score_bull += 2
        reason_bull.append("RSI перепродан (<30)")
    elif ind["rsi"] < 40:
        score_bull += 1
        reason_bull.append("RSI приближается к перепроданности (<40)")
    elif ind["rsi"] > 70:
        score_bear += 2
        reason_bear.append("RSI перекуплен (>70)")
    elif ind["rsi"] > 60:
        score_bear += 1
        reason_bear.append("RSI приближается к перекупленности (>60)")

    macd_diff = ind["macd"] - ind["macd_signal"]
    if macd_diff > 0:
        if macd_diff > 3:
            score_bull += 2
            reason_bull.append("Сильный бычий сигнал MACD")
        else:
            score_bull += 1
            reason_bull.append("Бычий сигнал MACD")
    elif macd_diff < 0:
        if macd_diff < -3:
            score_bear += 2
            reason_bear.append("Сильный медвежий сигнал MACD")
        else:
            score_bear += 1
            reason_bear.append("Медвежий сигнал MACD")

    bb_percent = (current_price - ind["bb_lower"]) / (ind["bb_upper"] - ind["bb_lower"]) * 100
    if bb_percent < 20:
        score_bull += 2
        reason_bull.append(f"Цена близка к нижней границе BB ({bb_percent:.1f}%)")
    elif bb_percent > 80:
        score_bear += 2
        reason_bear.append(f"Цена близка к верхней границе BB ({bb_percent:.1f}%)")

    if current_price > ind["ema50"] and current_price > ind["ema20"] and ind["ema20"] > ind["ema50"]:
        score_bull += 2
        reason_bull.append("Восходящий тренд по EMA (цена > EMA20 > EMA50)")
    elif current_price < ind["ema50"] and current_price < ind["ema20"] and ind["ema20"] < ind["ema50"]:
        score_bear += 2
        reason_bear.append("Нисходящий тренд по EMA (цена < EMA20 < EMA50)")

    if ind["stoch"] < 20:
        score_bull += 1
        reason_bull.append("Стохастик перепродан (<20)")
    elif ind["stoch"] > 80:
        score_bear += 1
        reason_bear.append("Стохастик перекуплен (>80)")

    pred_trend = sum(1 if p > current_price else -1 if p < current_price else 0 for p in predictions)
    if pred_trend > 0:
        score_bull += pred_trend
        reason_bull.append(f"Прогноз указывает на рост ({pred_trend})")
    elif pred_trend < 0:
        score_bear += abs(pred_trend)
        reason_bear.append(f"Прогноз указывает на снижение ({pred_trend})")

    if df is not None and len(df) > 5:
        last_volume = df['volume'].iloc[-1]
        avg_volume = df['vol_ma20'].iloc[-1]
        volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1.0

        if volume_ratio > 3.0:
            last_close = df['close_clean'].iloc[-1]
            prev_close = df['close_clean'].iloc[-2] if len(df) > 2 else last_close
            price_direction = 1 if last_close > prev_close else -1

            if price_direction > 0:
                score_bull += 3
                reason_bull.append(f"Крупная сделка на покупку (объем x{volume_ratio:.1f})")
            else:
                score_bear += 3
                reason_bear.append(f"Крупная сделка на продажу (объем x{volume_ratio:.1f})")
        elif volume_ratio > 2.0:
            last_close = df['close_clean'].iloc[-1]
            prev_close = df['close_clean'].iloc[-2] if len(df) > 2 else last_close
            price_direction = 1 if last_close > prev_close else -1

            if price_direction > 0:
                score_bull += 2
                reason_bull.append(f"Повышенные покупки (объем x{volume_ratio:.1f})")
            else:
                score_bear += 2
                reason_bear.append(f"Повышенные продажи (объем x{volume_ratio:.1f})")

        if len(df) > 10:
            volume_trend = df['volume'].iloc[-5:].mean() - df['volume'].iloc[-10:-5].mean()
            if volume_trend > 0:
                price_trend = df['close_clean'].iloc[-5:].mean() - df['close_clean'].iloc[-10:-5].mean()
                if price_trend > 0:
                    score_bull += 1
                    reason_bull.append("Растущий объем при росте цены")
                else:
                    score_bear += 1
                    reason_bear.append("Растущий объем при падении цены")
            else:
                price_trend = df['close_clean'].iloc[-5:].mean() - df['close_clean'].iloc[-10:-5].mean()
                if abs(price_trend) > 0 and volume_trend < -avg_volume * 0.2:
                    if price_trend > 0:
                        score_bear += 1
                        reason_bear.append("Падающий объем при росте цены (слабость)")
                    else:
                        score_bull += 1
                        reason_bull.append("Падающий объем при снижении цены (слабость)")

        if len(df) > 20:
            large_buys = 0
            large_sells = 0
            for i in range(-1, -11, -1):
                if i < -len(df):
                    break
                vol = df['volume'].iloc[i]
                if vol > avg_volume * 1.5:
                    price_chg = df['close_clean'].iloc[i] - df['close_clean'].iloc[i-1] if i > -len(df) else 0
                    if price_chg > 0:
                        large_buys += 1
                    elif price_chg < 0:
                        large_sells += 1

            if large_buys >= 3:
                score_bull += 2
                reason_bull.append(f"Серия крупных покупок ({large_buys})")
            if large_sells >= 3:
                score_bear += 2
                reason_bear.append(f"Серия крупных продаж ({large_sells})")

    bull_strength = score_bull / (score_bull + score_bear) * 100 if (score_bull + score_bear) > 0 else 50
    reasons = []

    if bull_strength >= 65:
        signal = "СИГНАЛ к ЛОНГУ"
        reasons = reason_bull
    elif bull_strength <= 35:
        signal = "СИГНАЛ к ШОРТУ"
        reasons = reason_bear
    else:
        signal = "Рынок нейтрален, рекомендуется выжидать"
        reasons = reason_bull + reason_bear

    reason_text = ", ".join(reasons[:3]) if reasons else "Нет явных сигналов"
    return f"{signal} ({bull_strength:.1f}%) - {reason_text}"


class PredictionTracker:
    def __init__(self, ticker):
        self.ticker = ticker
        self.predictions = {}
        self.filename = f"{ticker}_predictions.pkl"
        self.load_history()

    def load_history(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'rb') as f:
                    self.predictions = pickle.load(f)
        except Exception as e:
            print(f"Ошибка при загрузке истории прогнозов: {e}")
            self.predictions = {}

    def save_history(self):
        try:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.predictions, f)
        except Exception as e:
            print(f"Ошибка при сохранении истории прогнозов: {e}")

    def add_predictions(self, timestamp, predictions, intervals):
        ts = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.predictions[ts] = {}

        for i, pred in enumerate(predictions):
            if i < len(intervals):
                interval = intervals[i]
                self.predictions[ts][interval] = {
                    "prediction": pred,
                    "actual": None,
                    "timestamp": timestamp,
                    "target_time": timestamp + timedelta(minutes=interval)
                }
        self.save_history()

    def update_actuals(self, current_time, price):
        updated = False
        for ts, intervals in list(self.predictions.items()):
            for interval, data in list(intervals.items()):
                if data["actual"] is None and current_time >= data["target_time"]:
                    self.predictions[ts][interval]["actual"] = price
                    updated = True

        one_day_ago = datetime.now() - timedelta(days=1)
        for ts in list(self.predictions.keys()):
            timestamp = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if timestamp < one_day_ago:
                del self.predictions[ts]
                updated = True

        if updated:
            self.save_history()

    def get_accuracy_stats(self):
        if not self.predictions:
            return {}

        completed_predictions = []
        for ts, intervals in self.predictions.items():
            for interval, data in intervals.items():
                if data["actual"] is not None:
                    error = abs(data["prediction"] - data["actual"]) / data["actual"] * 100
                    pred_change = data["prediction"] - data["actual"]
                    actual_price_at_prediction_time = data["actual"]

                    pred_direction = 1 if data["prediction"] > actual_price_at_prediction_time else -1 if data["prediction"] < actual_price_at_prediction_time else 0

                    actual_direction = 1 if data["actual"] > actual_price_at_prediction_time else -1 if data["actual"] < actual_price_at_prediction_time else 0

                    direction_correct = pred_direction == actual_direction

                    days_old = (datetime.now() - data["timestamp"]).total_seconds() / (24 * 3600)
                    time_weight = max(0.5, 1.0 - (days_old / 3.0))

                    completed_predictions.append({
                        "interval": interval,
                        "error": error,
                        "direction_correct": direction_correct,
                        "time_diff": (data["target_time"] - data["timestamp"]).total_seconds() / 60,
                        "time_weight": time_weight
                    })

        if not completed_predictions:
            return {}

        stats_by_interval = {}
        for pred in completed_predictions:
            interval = pred["interval"]
            if interval not in stats_by_interval:
                stats_by_interval[interval] = {
                    "count": 0,
                    "error_sum": 0,
                    "correct_directions": 0,
                    "weighted_count": 0,
                    "weighted_error_sum": 0,
                    "weighted_correct_directions": 0
                }
            stats_by_interval[interval]["count"] += 1
            stats_by_interval[interval]["error_sum"] += pred["error"]
            stats_by_interval[interval]["correct_directions"] += 1 if pred["direction_correct"] else 0

            weight = pred["time_weight"]
            stats_by_interval[interval]["weighted_count"] += weight
            stats_by_interval[interval]["weighted_error_sum"] += pred["error"] * weight
            stats_by_interval[interval]["weighted_correct_directions"] += weight if pred["direction_correct"] else 0

        for interval, stats in stats_by_interval.items():
            if stats["count"] > 0:
                stats["avg_error"] = stats["error_sum"] / stats["count"]
                stats["direction_accuracy"] = stats["correct_directions"] / stats["count"] * 100

                if stats["weighted_count"] > 0:
                    stats["weighted_avg_error"] = stats["weighted_error_sum"] / stats["weighted_count"]
                    stats["weighted_direction_accuracy"] = stats["weighted_correct_directions"] / stats["weighted_count"] * 100
                else:
                    stats["weighted_avg_error"] = 0
                    stats["weighted_direction_accuracy"] = 0
            else:
                stats["avg_error"] = 0
                stats["direction_accuracy"] = 0
                stats["weighted_avg_error"] = 0
                stats["weighted_direction_accuracy"] = 0

        total_count = sum(stats["count"] for stats in stats_by_interval.values())
        total_error = sum(stats["error_sum"] for stats in stats_by_interval.values())
        total_correct = sum(stats["correct_directions"] for stats in stats_by_interval.values())

        total_weighted_count = sum(stats["weighted_count"] for stats in stats_by_interval.values())
        total_weighted_error = sum(stats["weighted_error_sum"] for stats in stats_by_interval.values())
        total_weighted_correct = sum(stats["weighted_correct_directions"] for stats in stats_by_interval.values())

        overall_stats = {
            "total_count": total_count,
            "avg_error": total_error / total_count if total_count > 0 else 0,
            "direction_accuracy": total_correct / total_count * 100 if total_count > 0 else 0,
            "weighted_avg_error": total_weighted_error / total_weighted_count if total_weighted_count > 0 else 0,
            "weighted_direction_accuracy": total_weighted_correct / total_weighted_count * 100 if total_weighted_count > 0 else 0,
            "by_interval": stats_by_interval
        }

        return overall_stats


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_report(ticker, figi, last_close, raw_close, predictions, current_time,
                 indicators, regimes, analysis, accuracy_stats=None):
    predicted_price = predictions[0]
    trend_symbol = "↑" if predicted_price > last_close else "↓" if predicted_price < last_close else "-"
    color_pred = "\033[1;32m" if predicted_price > last_close else "\033[1;31m" if predicted_price < last_close else "\033[0m"
    reset = "\033[0m"

    time_intervals = [1, 5, 15, 30]

    interval_preds = []
    for i, pred in enumerate(predictions):
        if i < len(time_intervals):
            interval = time_intervals[i]
            trend = "↑" if pred > raw_close else "↓" if pred < raw_close else "-"
            color = "\033[1;32m" if pred > raw_close else "\033[1;31m" if pred < raw_close else "\033[0m"
            diff_percent = ((pred / raw_close) - 1) * 100
            interval_preds.append(f"  {interval} мин: {color}{pred:.2f} {trend} ({diff_percent:.2f}%){reset}")

    multi_preds_str = "\n".join(interval_preds)

    regime_str = " | ".join(
        [f"Режим {k}: {v*100:.1f}%"
         for k, v in regimes.items()]) if regimes else "Нет данных о режимах"

    vol_color = ""
    if 'vol_ma20' in indicators:
        last_vol = indicators.get('volume', 0)
        avg_vol = indicators['vol_ma20']
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        if vol_ratio > 2.0:
            vol_color = "\033[1;32m"
        elif vol_ratio < 0.5:
            vol_color = "\033[1;31m"

    rep_ind = (
        f"EMA20: {indicators['ema20']:.2f} | SMA20: {indicators['sma20']:.2f} | RSI: {indicators['rsi']:.2f} | "
        f"MACD: {indicators['macd']:.2f} | MACD Signal: {indicators['macd_signal']:.2f} | BB Upper: {indicators['bb_upper']:.2f} | "
        f"BB Lower: {indicators['bb_lower']:.2f} | Stoch: {indicators['stoch']:.2f} | EMA50: {indicators['ema50']:.2f} | "
        f"Vol_MA20: {indicators['vol_ma20']:.2f}")

    volume_info = ""
    if 'volume' in indicators and 'vol_ma20' in indicators:
        last_vol = indicators['volume']
        avg_vol = indicators['vol_ma20']
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        vol_status = "нормальный"
        if vol_ratio > 3.0:
            vol_status = f"{vol_color}ОЧЕНЬ ВЫСОКИЙ{reset}"
        elif vol_ratio > 2.0:
            vol_status = f"{vol_color}высокий{reset}"
        elif vol_ratio > 1.5:
            vol_status = f"{vol_color}повышенный{reset}"
        elif vol_ratio < 0.5:
            vol_status = f"\033[1;31mнизкий{reset}"

        volume_info = f" Текущий объем: {last_vol} ({vol_ratio:.2f}x от среднего) - {vol_status}"

    raw_trend_symbol = "↑" if predicted_price > raw_close else "↓" if predicted_price < raw_close else "-"
    raw_color_pred = "\033[1;32m" if predicted_price > raw_close else "\033[1;31m" if predicted_price < raw_close else "\033[0m"

    accuracy_info = ""
    if accuracy_stats and "total_count" in accuracy_stats and accuracy_stats["total_count"] > 0:
        accuracy_info = (
            "------------------------------------------------------------\n"
            f" ТОЧНОСТЬ ПРОГНОЗОВ (всего {accuracy_stats['total_count']} замеров):\n"
            f"  Общая точность направления: {accuracy_stats['direction_accuracy']:.1f}%\n"
            f"  Средняя ошибка в процентах: {accuracy_stats['avg_error']:.2f}%\n\n"
            " По интервалам:\n"
        )

        for interval, stats in accuracy_stats.get("by_interval", {}).items():
            if stats["count"] > 0:
                accuracy_info += (
                    f"  {interval} мин (проверено {stats['count']}): "
                    f"точность направления {stats['direction_accuracy']:.1f}%, "
                    f"ошибка {stats['avg_error']:.2f}%\n"
                )

        overall_accuracy = accuracy_stats["direction_accuracy"]
        accuracy_desc = ""
        if overall_accuracy >= 75:
            accuracy_desc = f"\033[1;32mВысокая точность ({overall_accuracy:.1f}%)\033[0m"
        elif overall_accuracy >= 60:
            accuracy_desc = f"\033[1;33mСредняя точность ({overall_accuracy:.1f}%)\033[0m"
        else:
            accuracy_desc = f"\033[1;31mНизкая точность ({overall_accuracy:.1f}%)\033[0m"

        accuracy_info += f"\n ОБЩАЯ ОЦЕНКА МОДЕЛИ: {accuracy_desc}\n"

    report = (
        "============================================================\n"
        f" ТИКЕР: {ticker.upper()}    FIGI: {figi}\n"
        f" ДАТА: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "------------------------------------------------------------\n"
        f" Последняя рыночная цена: {raw_close:.2f}\n"
        f" Последняя цена закрытия (очищенная): {last_close:.2f}\n"
        f" Ближайший прогноз от очищенной цены: {color_pred}{predicted_price:.2f} {trend_symbol}{reset}\n"
        f" Ближайший прогноз от рыночной цены: {raw_color_pred}{predicted_price:.2f} {raw_trend_symbol}{reset}\n"
        "------------------------------------------------------------\n"
        " Прогнозируемые цены по интервалам:\n" + multi_preds_str + "\n"
        + accuracy_info +
        "------------------------------------------------------------\n"
        f" Режим рынка: {regime_str}\n"
        f" Индикаторы: {rep_ind}\n"
        + (f" Объемы: {volume_info}\n" if volume_info else "") +
        f" Анализ рынка: {analysis}\n"
        "============================================================")
    print(report)


async def main():
    ticker = input("Введите тикер: ")
    figi = get_figi_by_ticker(ticker)

    prediction_tracker = PredictionTracker(ticker)

    async def get_current_price():
        try:
            async with AsyncClient(TOKEN) as client:
                response = await client.market_data.get_last_prices(figi=[figi])
                if response and response.last_prices:
                    return float(quotation_to_decimal(response.last_prices[0].price))
                return None
        except Exception as e:
            print(f"Ошибка при получении текущей цены: {e}")
            return None

    while True:
        df = await fetch_candles(figi, days=90)
        current_market_price = await get_current_price()
        current_time = datetime.now()

        if current_market_price is not None:
            prediction_tracker.update_actuals(current_time, current_market_price)

        if df.empty:
            print("Нет данных свечей для данного тикера")
        else:
            df = preprocess_data(df)
            df = calculate_indicators(df)

            time_intervals = [1, 5, 15, 30]
            predictions = predict_multiple_steps_enhanced(df, lags=3)

            prediction_tracker.add_predictions(current_time, predictions, time_intervals)

            accuracy_stats = prediction_tracker.get_accuracy_stats()

            last_close = df["close_clean"].iloc[-1]
            raw_close = current_market_price if current_market_price else df["last_raw_close"]
            regimes = detect_regime(df)
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
                "vol_ma20": df["vol_ma20"].iloc[-1],
                "volume": df["volume"].iloc[-1]
            }
            analysis = analyze_market(indicators, raw_close, predictions, df)
            clear_terminal()
            print_report(ticker, figi, last_close, raw_close, predictions, current_time,
                         indicators, regimes, analysis, accuracy_stats)
        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
