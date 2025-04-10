import os
import asyncio
import pandas as pd
import numpy as np
import pickle
import itertools
import time
import logging
from datetime import timedelta, datetime
from typing import Any, Dict, List, Tuple
from tinkoff.invest import CandleInterval, AsyncClient, Client
from tinkoff.invest.utils import now, quotation_to_decimal
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from colorama import init, Fore, Style
import asciichartpy as ascii_chart
from tqdm import tqdm

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

TOKEN: str = os.environ.get("TINKOFF_TOKEN")
if not TOKEN:
    import getpass
    TOKEN = getpass.getpass("Введите ваш Tinkoff API токен: ")

def get_figi_by_ticker(ticker: str) -> str:
    try:
        with Client(TOKEN) as client:
            instruments = []
            for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
                instruments += getattr(client.instruments, method)().instruments
            filtered = [inst for inst in instruments if inst.ticker.upper() == ticker.upper()]
            if filtered:
                return filtered[0].figi
            else:
                raise ValueError("Инструмент не найден")
    except Exception as e:
        logging.error(f"Ошибка получения FIGI: {e}")
        raise

async def fetch_candles(figi: str, days: int = 90) -> pd.DataFrame:
    hourly_candles: List[Dict[str, Any]] = []
    minute_candles: List[Dict[str, Any]] = []
    retries: int = 0
    max_retries: int = 5
    delay: int = 1
    while retries < max_retries:
        try:
            async with AsyncClient(TOKEN) as client:
                async for candle in client.get_all_candles(
                    figi=figi,
                    from_=now() - timedelta(days=days),
                    to=now() - timedelta(minutes=60),
                    interval=CandleInterval.CANDLE_INTERVAL_HOUR
                ):
                    hourly_candles.append({
                        "time": candle.time,
                        "open": float(quotation_to_decimal(candle.open)),
                        "high": float(quotation_to_decimal(candle.high)),
                        "low": float(quotation_to_decimal(candle.low)),
                        "close": float(quotation_to_decimal(candle.close)),
                        "volume": candle.volume,
                        "timeframe": "hour"
                    })
                async for candle in client.get_all_candles(
                    figi=figi,
                    from_=now() - timedelta(minutes=60),
                    to=now(),
                    interval=CandleInterval.CANDLE_INTERVAL_1_MIN
                ):
                    minute_candles.append({
                        "time": candle.time,
                        "open": float(quotation_to_decimal(candle.open)),
                        "high": float(quotation_to_decimal(candle.high)),
                        "low": float(quotation_to_decimal(candle.low)),
                        "close": float(quotation_to_decimal(candle.close)),
                        "volume": candle.volume,
                        "timeframe": "minute"
                    })
            break
        except Exception as e:
            logging.warning(f"Ошибка получения свечей: {e}. Повтор через {delay} сек.")
            retries += 1
            await asyncio.sleep(delay)
            delay *= 2
    if retries == max_retries:
        logging.error("Не удалось получить данные свечей.")
        return pd.DataFrame()
    df_hour: pd.DataFrame = pd.DataFrame(hourly_candles)
    df_minute: pd.DataFrame = pd.DataFrame(minute_candles)
    if not df_hour.empty:
        df = df_hour.copy()
    elif not df_minute.empty:
        df_minute["time"] = pd.to_datetime(df_minute["time"])
        df = df_minute.resample("1h", on="time").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        df["timeframe"] = "hour_from_minute"
    else:
        return pd.DataFrame()
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df["last_raw_close"] = df["close"].iloc[-1]
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["last_raw_close"] = df["close"].iloc[-1]
    df = df.sort_index()
    hourly_data: pd.DataFrame = df[df["timeframe"] == "hour"].copy()
    if not hourly_data.empty:
        expected_hours = pd.date_range(start=hourly_data.index.min(), end=hourly_data.index.max(), freq="h")
        missing_hours = expected_hours.difference(hourly_data.index)
        if len(missing_hours) > 0:
            for missing_hour in missing_hours:
                before = hourly_data[hourly_data.index < missing_hour]
                if before.empty:
                    continue
                last_candle = before.iloc[-1]
                fill = last_candle.copy()
                fill["open"] = last_candle["close"]
                fill["high"] = last_candle["close"]
                fill["low"] = last_candle["close"]
                fill["close"] = last_candle["close"]
                fill["volume"] = 0
                fill.name = missing_hour
                df.loc[missing_hour] = fill
    df = df.sort_index()
    df["close_smooth"] = df["close"].rolling(window=2, min_periods=2).mean()
    window_size: int = min(10, len(df))
    if window_size < 4:
        df["close_clean"] = df["close_smooth"]
    else:
        roll_med = df["close_smooth"].rolling(window=window_size, min_periods=2).median()
        roll_std = df["close_smooth"].rolling(window=window_size, min_periods=2).std().fillna(0)
        diff = np.abs(df["close_smooth"] - roll_med)
        threshold_factor: float = 3.0 if len(df) > 30 else 4.0
        threshold = threshold_factor * roll_std
        df["close_clean"] = np.where(diff > threshold, roll_med, df["close_smooth"])
    last_hour_mask = df.index >= (df.index.max() - pd.Timedelta(hours=1))
    if any(last_hour_mask):
        df.loc[last_hour_mask, "close_clean"] = df.loc[last_hour_mask, "close"]
    df["close_clean"] = df["close_clean"].fillna(df["close"])
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

def create_fourier_features(df: pd.DataFrame, periods: List[int] = [24, 168], order: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["time_int"] = df.index.astype(np.int64) // 10**9
    for period in periods:
        for i in range(1, order + 1):
            df[f"sin_{period}_{i}"] = np.sin(2 * np.pi * i * df["time_int"] / (period * 3600))
            df[f"cos_{period}_{i}"] = np.cos(2 * np.pi * i * df["time_int"] / (period * 3600))
    df.drop(columns=["time_int"], inplace=True)
    return df

def create_features(df: pd.DataFrame, lags: int = 3) -> Tuple[pd.DataFrame, List[str]]:
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["close_clean"].shift(i)
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df[f"lag_{i}"].bfill().ffill()
    for col in ["sma20", "ema20", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "stoch", "ema50", "vol_ma20"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].ffill().bfill().fillna(0)
    df = create_fourier_features(df)
    df = df.dropna()
    feature_cols: List[str] = [f"lag_{i}" for i in range(1, lags + 1)] + [
        "sma20", "ema20", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "stoch", "ema50", "vol_ma20"
    ]
    fourier_cols = [col for col in df.columns if col.startswith("sin_") or col.startswith("cos_")]
    feature_cols += fourier_cols
    feature_cols = [col for col in feature_cols if col in df.columns]
    return df, feature_cols

def train_model_enhanced(df: pd.DataFrame, lags: int = 3) -> Tuple[Any, StandardScaler, StandardScaler, List[str], pd.DataFrame]:
    if len(df) < 30:
        dummy_model = DummyRegressor(strategy="mean")
        dummy_model.fit(np.array([[0]]), np.array([df["close_clean"].mean()]))
        scaler_X = StandardScaler()
        scaler_X.fit(np.array([[0]]))
        scaler_y = StandardScaler()
        scaler_y.fit(np.array([[0]]))
        return dummy_model, scaler_X, scaler_y, ["dummy_feature"], df
    df_feat, feature_cols = create_features(df.copy(), lags)
    adf_result = adfuller(df_feat["close_clean"].dropna())
    if adf_result[1] > 0.05:
        df_feat["price_diff"] = df_feat["close_clean"].diff().fillna(0)
        if "price_diff" not in feature_cols:
            feature_cols.append("price_diff")
    test_size = int(len(df_feat) * 0.2)
    train_df = df_feat.iloc[:-test_size] if test_size > 0 else df_feat.copy()
    test_df = df_feat.iloc[-test_size:] if test_size > 0 else pd.DataFrame()
    tscv = TimeSeriesSplit(n_splits=3)
    X_train = train_df[feature_cols].values
    y_train = train_df["close_clean"].values.reshape(-1, 1)
    lower_bounds: Dict[int, float] = {}
    upper_bounds: Dict[int, float] = {}
    for i in range(X_train.shape[1]):
        col_data = X_train[:, i]
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        lower_bounds[i] = q1 - 1.5 * iqr
        upper_bounds[i] = q3 + 1.5 * iqr
        X_train[:, i] = np.clip(col_data, lower_bounds[i], upper_bounds[i])
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train).ravel()
    if len(X_train) < 50:
        models = {"ridge": Ridge(), "elasticnet": ElasticNet(random_state=42, alpha=1.0)}
    else:
        models = {"ridge": Ridge(), "elasticnet": ElasticNet(random_state=42), "gbr": GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)}
    best_model_name = None
    best_model = None
    best_params = None
    best_score = -float("inf")
    for model_name, model in models.items():
        if model_name == "ridge":
            param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0]}
        elif model_name == "elasticnet":
            param_grid = {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]}
        elif model_name == "gbr":
            param_grid = {"learning_rate": [0.01, 0.1]}
        best_param_score = -float("inf")
        best_model_params = None
        for params in itertools.product(*param_grid.values()):
            param_dict = {name: value for name, value in zip(list(param_grid.keys()), params)}
            model.set_params(**param_dict)
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train_scaled):
                X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_cv_train, y_cv_val = y_train_scaled[train_idx], y_train_scaled[val_idx]
                model.fit(X_cv_train, y_cv_train)
                pred_val = model.predict(X_cv_val)
                r2 = r2_score(y_cv_val, pred_val)
                cv_scores.append(r2)
            mean_score = np.mean(cv_scores)
            if mean_score > best_param_score:
                best_param_score = mean_score
                best_model_params = param_dict
        if best_param_score > best_score:
            best_score = best_param_score
            best_model_name = model_name
            best_params = best_model_params
    if best_model_name == "ridge":
        best_model = Ridge(**best_params)
    elif best_model_name == "elasticnet":
        best_model = ElasticNet(**best_params, random_state=42)
    elif best_model_name == "gbr":
        best_model = GradientBoostingRegressor(**best_params, n_estimators=50, max_depth=3, random_state=42)
    best_model.fit(X_train_scaled, y_train_scaled)
    if len(test_df) > 0:
        X_test = test_df[feature_cols].values
        y_test = test_df["close_clean"].values
        for i in range(X_test.shape[1]):
            if i in lower_bounds:
                X_test[:, i] = np.clip(X_test[:, i], lower_bounds[i], upper_bounds[i])
        X_test_scaled = scaler_X.transform(X_test)
        pred_test_scaled = best_model.predict(X_test_scaled)
        pred_test = scaler_y.inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()
        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        r2 = r2_score(y_test, pred_test)
    return best_model, scaler_X, scaler_y, feature_cols, df_feat

def predict_multiple_steps_enhanced(model: Any, scaler_X: StandardScaler, scaler_y: StandardScaler, feature_cols: List[str], df: pd.DataFrame, steps: int = 5, lags: int = 3) -> List[float]:
    predictions: List[float] = []
    df_pred: pd.DataFrame = df.copy()
    alpha: float = 0.7
    for _ in range(steps):
        last_row = df_pred.iloc[-1]
        input_vector = last_row[feature_cols].values.reshape(1, -1)
        input_scaled = scaler_X.transform(input_vector)
        model_pred = scaler_y.inverse_transform(model.predict(input_scaled).reshape(-1, 1))[0, 0]
        pred = alpha * model_pred + (1 - alpha) * last_row["close_clean"]
        predictions.append(pred)
        new_row = last_row.copy()
        new_row["close_clean"] = pred
        for i in range(lags, 1, -1):
            new_row[f"lag_{i}"] = last_row.get(f"lag_{i-1}", pred)
        new_row["lag_1"] = pred
        new_index = df_pred.index[-1] + pd.Timedelta(minutes=1)
        new_row.name = new_index
        df_pred = pd.concat([df_pred, new_row.to_frame().T])
    return predictions

def detect_regime(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) < 30:
        return {"0": 0.5, "1": 0.5}
    returns = df["close_clean"].pct_change().dropna()
    if len(returns) < 20:
        return {"0": 0.5, "1": 0.5}
    lower_q, upper_q = returns.quantile([0.01, 0.99])
    filtered_returns = returns[(returns >= lower_q) & (returns <= upper_q)]
    if len(filtered_returns) < 20:
        filtered_returns = returns
    try:
        mr = MarkovRegression(filtered_returns, k_regimes=2, trend="c", switching_variance=True, switching_trend=True)
        best_result = None
        best_aic = float("inf")
        for _ in range(3):
            try:
                res = mr.fit(disp=False, maxiter=100)
                if hasattr(res, "aic") and res.aic < best_aic:
                    best_aic = res.aic
                    best_result = res
            except Exception:
                continue
        if best_result is None:
            mr_simple = MarkovRegression(filtered_returns, k_regimes=2, trend="c", switching_variance=False)
            best_result = mr_simple.fit(disp=False)
    except Exception:
        return {"0": 0.5, "1": 0.5}
    regimes = best_result.smoothed_marginal_probabilities.iloc[-1].to_dict()
    try:
        regime_states = best_result.smoothed_marginal_probabilities.idxmax(axis=1)
        regime_0_mean = filtered_returns[regime_states == 0].mean()
        regime_1_mean = filtered_returns[regime_states == 1].mean()
        if regime_0_mean < regime_1_mean:
            regimes = {"0": regimes.get(1, 0.5), "1": regimes.get(0, 0.5)}
        else:
            regimes = {"0": regimes.get(0, 0.5), "1": regimes.get(1, 0.5)}
    except Exception:
        regimes = {"0": 0.5, "1": 0.5}
    return regimes

def analyze_market(ind: Dict[str, float], current_price: float, predictions: List[float], df: pd.DataFrame = None) -> str:
    weights: Dict[str, float] = {"rsi": 0.3, "macd": 0.3, "ema": 0.2, "stoch": 0.2}
    score: float = 0
    factors: List[str] = []
    if df is not None and len(df) > 20:
        recent_vol = df["close_clean"].pct_change().rolling(window=5).std().iloc[-1]
        hist_vol = df["close_clean"].pct_change().rolling(window=20).std().iloc[-1]
        if hist_vol > 0 and recent_vol > 2 * hist_vol:
            score -= 3
            factors.append("Высокая волатильность")
    if df is not None and len(df) > 1:
        last_close = df["close_clean"].iloc[-2]
        current_open = df["open"].iloc[-1]
        if abs(current_open - last_close) / last_close > 0.02:
            if current_open > last_close:
                score += 3
                factors.append("Гэп вверх")
            else:
                score -= 3
                factors.append("Гэп вниз")
    rsi = ind["rsi"]
    score += weights["rsi"] * ((50 - rsi) / 10)
    macd_diff = ind["macd"] - ind["macd_signal"]
    score += weights["macd"] * macd_diff
    ema_trend = 3 if (current_price > ind["ema20"] and ind["ema20"] > ind["ema50"]) else -3 if (current_price < ind["ema20"] and ind["ema20"] < ind["ema50"]) else 0
    score += weights["ema"] * ema_trend
    stoch_signal = 2 if ind["stoch"] < 20 else -2 if ind["stoch"] > 80 else 0
    score += weights["stoch"] * stoch_signal
    regimes = detect_regime(df) if df is not None else {"0": 0.5, "1": 0.5}
    bullish_regime = regimes.get("0", 0.5)
    score += (bullish_regime - 0.5) * 10
    if score >= 3:
        signal = "СИГНАЛ к ЛОНГУ"
    elif score <= -3:
        signal = "СИГНАЛ к ШОРТУ"
    else:
        signal = "РЫНОК НЕЙТРАЛЕН"
    return f"{signal} (score: {score:.1f}) - {', '.join(factors[:3])}"

class PredictionTracker:
    def __init__(self, ticker: str) -> None:
        self.ticker: str = ticker
        self.predictions: Dict[str, Dict[str, Any]] = {}
        self.filename: str = f"{ticker}_predictions.pkl"
        self.load_history()
    def load_history(self) -> None:
        try:
            if os.path.exists(self.filename):
                with open(self.filename, "rb") as f:
                    self.predictions = pickle.load(f)
        except Exception:
            self.predictions = {}
    def save_history(self) -> None:
        try:
            with open(self.filename, "wb") as f:
                pickle.dump(self.predictions, f)
        except Exception:
            pass
    def add_predictions(self, timestamp: datetime, predictions: List[float], intervals: List[int], base_price: float) -> None:
        ts = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.predictions[ts] = {}
        for i, pred in enumerate(predictions):
            if i < len(intervals):
                interval = intervals[i]
                self.predictions[ts][str(interval)] = {
                    "prediction": pred,
                    "actual": None,
                    "timestamp": timestamp,
                    "target_time": timestamp + timedelta(minutes=interval),
                    "base_price": base_price
                }
        self.save_history()
    def update_actuals(self, current_time: datetime, price: float) -> None:
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
    def get_accuracy_stats(self) -> Dict[str, Any]:
        if not self.predictions:
            return {}
        completed = []
        for ts, intervals in self.predictions.items():
            for interval, data in intervals.items():
                if data["actual"] is not None:
                    base_price = data.get("base_price", data["prediction"])
                    pred_direction = 1 if data["prediction"] > base_price else -1 if data["prediction"] < base_price else 0
                    actual_direction = 1 if data["actual"] > base_price else -1 if data["actual"] < base_price else 0
                    direction_correct = pred_direction == actual_direction
                    error = abs(data["prediction"] - data["actual"]) / data["actual"] * 100
                    completed.append({
                        "interval": interval,
                        "error": error,
                        "direction_correct": direction_correct
                    })
        if not completed:
            return {}
        stats: Dict[str, Dict[str, float]] = {}
        for pred in completed:
            interval = pred["interval"]
            if interval not in stats:
                stats[interval] = {"count": 0, "error_sum": 0, "correct": 0}
            stats[interval]["count"] += 1
            stats[interval]["error_sum"] += pred["error"]
            stats[interval]["correct"] += 1 if pred["direction_correct"] else 0
        overall_count = sum(s["count"] for s in stats.values())
        overall_error = sum(s["error_sum"] for s in stats.values())
        overall_correct = sum(s["correct"] for s in stats.values())
        return {
            "total_count": overall_count,
            "avg_error": overall_error / overall_count if overall_count > 0 else 0,
            "direction_accuracy": overall_correct / overall_count * 100 if overall_count > 0 else 0,
            "by_interval": stats
        }

def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")

def show_loading_animation(message: str = "Загрузка данных") -> None:
    clear_terminal()
    print(f"\033[1;36m{message}...\033[0m")
    for _ in tqdm(range(100), desc="Прогресс", ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        time.sleep(0.01)

def calculate_risk_metrics(df: pd.DataFrame, current_price: float, predictions: List[float]) -> Dict[str, Any]:
    if df is None or len(df) < 30:
        return {"var_95": 0, "cvar": 0, "max_loss_pct": 0, "risk_level": "Неизвестно"}
    returns = df["close_clean"].pct_change().dropna()
    if len(returns) < 20:
        return {"var_95": 0, "cvar": 0, "max_loss_pct": 0, "risk_level": "Недостаточно данных"}
    var_95_pct = abs(returns.quantile(0.05) * 100)
    var_95 = abs(returns.quantile(0.05) * current_price)
    cvar = abs(returns[returns <= returns.quantile(0.05)].mean() * current_price)
    rolling_max = df["close_clean"].rolling(window=30).max()
    daily_drawdown = df["close_clean"] / rolling_max - 1.0
    max_loss_pct = abs(daily_drawdown.min() * 100)
    risk_level = "Высокий" if var_95_pct > 5 else "Средний" if var_95_pct > 2 else "Низкий"
    return {"var_95": var_95, "cvar": cvar, "var_95_pct": var_95_pct, "max_loss_pct": max_loss_pct, "risk_level": risk_level}

def generate_ascii_chart(prices: List[float], width: int = 40, height: int = 10) -> str:
    if not prices or len(prices) < 2:
        return "Недостаточно данных для построения графика"
    return ascii_chart.plot(prices, {"height": height, "width": width, "format": "{:,.2f}"})

def print_report(ticker: str, figi: str, last_close: float, raw_close: float, predictions: List[float], current_time: datetime, indicators: Dict[str, float], regimes: Dict[str, float], analysis: str, accuracy_stats: Dict[str, Any], risk_metrics: Dict[str, Any], df: pd.DataFrame) -> None:
    GREEN = Fore.GREEN + Style.BRIGHT
    RED = Fore.RED + Style.BRIGHT
    YELLOW = Fore.YELLOW + Style.BRIGHT
    CYAN = Fore.CYAN + Style.BRIGHT
    MAGENTA = Fore.MAGENTA + Style.BRIGHT
    WHITE = Fore.WHITE + Style.BRIGHT
    historical_chart: str = ""
    if df is not None and len(df) > 20:
        historical_values = df["close_clean"].iloc[-20:].tolist()
        historical_chart = generate_ascii_chart(historical_values, width=50, height=10)
    if regimes:
        regime_items = []
        for k, v in regimes.items():
            percentage = v * 100
            color = GREEN if k == "0" and percentage > 60 else RED if k == "1" and percentage > 60 else YELLOW
            regime_items.append(f"{color}Режим {k}: {percentage:.1f}%{Style.RESET_ALL}")
        regime_str = " | ".join(regime_items)
    else:
        regime_str = f"{YELLOW}Нет данных о режимах{Style.RESET_ALL}"
    vol_info = ""
    if "volume" in indicators and "vol_ma20" in indicators:
        last_vol = indicators.get("volume", 0)
        avg_vol = indicators["vol_ma20"]
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
        if vol_ratio > 3.0:
            vol_color = GREEN
            vol_symbol = "▲▲▲"
        elif vol_ratio > 2.0:
            vol_color = GREEN
            vol_symbol = "▲▲"
        elif vol_ratio > 1.5:
            vol_color = GREEN
            vol_symbol = "▲"
        elif vol_ratio < 0.5:
            vol_color = RED
            vol_symbol = "▼▼"
        else:
            vol_color = WHITE
            vol_symbol = "●"
        vol_info = f"{CYAN}Объем:{Style.RESET_ALL} {vol_color}{last_vol} ({vol_ratio:.2f}x) {vol_symbol}{Style.RESET_ALL}"
    rsi_color = GREEN if indicators["rsi"] < 30 else RED if indicators["rsi"] > 70 else WHITE
    macd_color = GREEN if indicators["macd"] > indicators["macd_signal"] else RED
    bb_position = (raw_close - indicators["bb_lower"]) / (indicators["bb_upper"] - indicators["bb_lower"]) * 100
    bb_color = RED if bb_position > 80 else GREEN if bb_position < 20 else WHITE
    rsi_bars = "▁▂▃▄▅▆▇█"
    rsi_index = min(int(indicators["rsi"] / 12.5), 7)
    rsi_visual = rsi_bars[rsi_index]
    indicators_display = [
        f"{CYAN}RSI:{Style.RESET_ALL} {rsi_color}{indicators['rsi']:.1f} {rsi_visual}{Style.RESET_ALL}",
        f"{CYAN}MACD:{Style.RESET_ALL} {macd_color}{indicators['macd']:.2f}/{indicators['macd_signal']:.2f}{Style.RESET_ALL}",
        f"{CYAN}BB:{Style.RESET_ALL} {bb_color}{bb_position:.1f}%{Style.RESET_ALL}",
        f"{CYAN}EMA:{Style.RESET_ALL} {indicators['ema20']:.2f}/{indicators['ema50']:.2f}",
        vol_info
    ]
    detailed_indicators = (
        f"  EMA20: {indicators['ema20']:.2f} | SMA20: {indicators['sma20']:.2f} | RSI: {indicators['rsi']:.2f}\n"
        f"  MACD: {indicators['macd']:.2f} | MACD Signal: {indicators['macd_signal']:.2f}\n"
        f"  BB: Upper={indicators['bb_upper']:.2f} | Lower={indicators['bb_lower']:.2f}\n"
        f"  Stoch: {indicators['stoch']:.2f} | EMA50: {indicators['ema50']:.2f} | Vol_MA20: {indicators['vol_ma20']:.2f}"
    )
    risk_info = ""
    if risk_metrics:
        if risk_metrics["risk_level"] == "Высокий":
            risk_color = RED
            risk_symbol = "⚠⚠⚠"
        elif risk_metrics["risk_level"] == "Средний":
            risk_color = YELLOW
            risk_symbol = "⚠⚠"
        elif risk_metrics["risk_level"] == "Низкий":
            risk_color = GREEN
            risk_symbol = "⚠"
        else:
            risk_color = WHITE
            risk_symbol = "•"
        var_pct = risk_metrics.get("var_95_pct", 0)
        var_bars = "▁▂▃▄▅▆▇█"
        var_index = min(int(var_pct / 1.25), 7)
        var_visual = var_bars[var_index]
        risk_info = (
            f"\n{CYAN}┌─ ОЦЕНКА РИСКА ────────────────────────────────────────────┐{Style.RESET_ALL}\n"
            f"{CYAN}│{Style.RESET_ALL} Уровень: {risk_color}{risk_metrics['risk_level']} {risk_symbol}{Style.RESET_ALL}\n"
            f"{CYAN}│{Style.RESET_ALL} VaR 95%: {risk_color}{risk_metrics['var_95']:.2f} ({risk_metrics.get('var_95_pct', 0):.2f}%) {var_visual}{Style.RESET_ALL}\n"
            f"{CYAN}│{Style.RESET_ALL} CVaR: {risk_color}{risk_metrics.get('cvar', 0):.2f}{Style.RESET_ALL}\n"
            f"{CYAN}│{Style.RESET_ALL} Макс. просадка: {risk_metrics['max_loss_pct']:.2f}%\n"
            f"{CYAN}└────────────────────────────────────────────────────────────┘{Style.RESET_ALL}"
        )
    if "СИГНАЛ к ЛОНГУ" in analysis:
        signal_part = f"{GREEN}СИГНАЛ к ЛОНГУ{Style.RESET_ALL}"
    elif "СИГНАЛ к ШОРТУ" in analysis:
        signal_part = f"{RED}СИГНАЛ к ШОРТУ{Style.RESET_ALL}"
    else:
        signal_part = f"{YELLOW}РЫНОК НЕЙТРАЛЕН{Style.RESET_ALL}"
    if "(" in analysis and ")" in analysis:
        try:
            score_value = float(analysis.split("score:")[1].split(")")[0])
        except Exception:
            score_value = 0
        strength_color = GREEN if score_value > 3 else RED if score_value < -3 else YELLOW
        strength_bars = "▁▂▃▄▅▆▇█"
        strength_index = min(int((score_value + 5) / 10 * 7), 7)
        strength_visual = strength_bars[strength_index]
        strength_part = f"{strength_color}{score_value:.1f} {strength_visual}{Style.RESET_ALL}"
        reason_part = analysis.split("-")[1].strip() if "-" in analysis else ""
    else:
        strength_part = ""
        reason_part = ""
    analysis_formatted = f"{signal_part} ({strength_part}) - {reason_part}"
    header = (
        f"\n{MAGENTA}{'=' * 70}{Style.RESET_ALL}\n"
        f"{MAGENTA}║{Style.RESET_ALL} {WHITE}{ticker.upper()} - Торговый анализ{Style.RESET_ALL}{' ' * 40}{MAGENTA}║{Style.RESET_ALL}\n"
        f"{MAGENTA}{'=' * 70}{Style.RESET_ALL}\n"
    )
    timestamp = (
        f"{CYAN}┌─ ИНФОРМАЦИЯ ─────────────────────────────────────────────┐{Style.RESET_ALL}\n"
        f"{CYAN}│{Style.RESET_ALL} Дата и время: {WHITE}{current_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}\n"
        f"{CYAN}│{Style.RESET_ALL} Тикер: {WHITE}{ticker.upper()}{Style.RESET_ALL}  FIGI: {figi}\n"
        f"{CYAN}└────────────────────────────────────────────────────────────┘{Style.RESET_ALL}"
    )
    prices = (
        f"{CYAN}┌─ ТЕКУЩИЕ ЦЕНЫ ───────────────────────────────────────────┐{Style.RESET_ALL}\n"
        f"{CYAN}│{Style.RESET_ALL} Рыночная цена: {WHITE}{raw_close:.2f}{Style.RESET_ALL}\n"
        f"{CYAN}│{Style.RESET_ALL} Цена закрытия: {WHITE}{last_close:.2f}{Style.RESET_ALL}\n"
        f"{CYAN}└────────────────────────────────────────────────────────────┘{Style.RESET_ALL}"
    )
    market_analysis = (
        f"{CYAN}┌─ АНАЛИЗ РЫНКА ───────────────────────────────────────────┐{Style.RESET_ALL}\n"
        f"{CYAN}│{Style.RESET_ALL} {analysis_formatted}\n"
        f"{CYAN}│{Style.RESET_ALL} Режим: {regime_str}\n"
        f"{CYAN}└────────────────────────────────────────────────────────────┘{Style.RESET_ALL}"
    )
    indicators_section = (
        f"{CYAN}┌─ ИНДИКАТОРЫ ────────────────────────────────────────────┐{Style.RESET_ALL}\n"
        f"{CYAN}│{Style.RESET_ALL} {' | '.join(indicators_display)}\n"
        f"{CYAN}│{Style.RESET_ALL}\n"
        f"{CYAN}│{Style.RESET_ALL} {detailed_indicators}\n"
        f"{CYAN}└────────────────────────────────────────────────────────────┘{Style.RESET_ALL}"
    )
    history_section = ""
    if historical_chart:
        history_section = (
            f"{CYAN}┌─ ИСТОРИЯ ЦЕН (последние 20 точек) ─────────────────────┐{Style.RESET_ALL}\n"
            f"{historical_chart}\n"
            f"{CYAN}└────────────────────────────────────────────────────────────┘{Style.RESET_ALL}"
        )
    report = (
        f"{header}\n"
        f"{timestamp}\n\n"
        f"{prices}\n\n"
        f"{market_analysis}\n\n"
        f"{indicators_section}\n"
        f"{history_section}\n"
        f"{risk_info}\n\n"
        f"{MAGENTA}{'=' * 70}{Style.RESET_ALL}\n"
    )
    print(report)

async def main() -> None:
    ticker: str = input("Введите тикер: ")
    try:
        figi: str = get_figi_by_ticker(ticker)
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        return
    last_update_time: datetime = datetime.now() - timedelta(minutes=10)
    model_update_interval: timedelta = timedelta(minutes=30)
    data_update_interval: timedelta = timedelta(seconds=10)
    async def get_current_price() -> Any:
        retry_count: int = 0
        max_retries: int = 3
        retry_delay: int = 2
        while retry_count < max_retries:
            try:
                async with AsyncClient(TOKEN) as client:
                    response = await client.market_data.get_last_prices(figi=[figi])
                    if response and response.last_prices:
                        return float(quotation_to_decimal(response.last_prices[0].price))
                    retry_count += 1
            except Exception:
                retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
        if df_cache is not None and "last_raw_close" in df_cache:
            return df_cache["last_raw_close"]
        return None
    df_cache: Any = None
    df_features_cache: Any = None
    predictions_cache: List[float] = []
    regimes_cache: Dict[str, float] = {}
    model_cache: Any = None
    scaler_X_cache: Any = None
    scaler_y_cache: Any = None
    feature_cols_cache: List[str] = []
    try:
        while True:
            current_time = datetime.now()
            current_market_price = await get_current_price()
            need_model_update = (current_time - last_update_time) > model_update_interval
            if need_model_update or df_cache is None:
                show_loading_animation("Обновление данных")
                df: pd.DataFrame = await fetch_candles(figi, days=90)
                last_update_time = current_time
                if df.empty:
                    await asyncio.sleep(30)
                    continue
                df = preprocess_data(df)
                df = calculate_indicators(df)
                model_cache, scaler_X_cache, scaler_y_cache, feature_cols_cache, df_features = train_model_enhanced(df, lags=3)
                predictions = predict_multiple_steps_enhanced(model_cache, scaler_X_cache, scaler_y_cache, feature_cols_cache, df_features, steps=5, lags=3)
                df_cache = df
                df_features_cache = df_features
                predictions_cache = predictions
                regimes_cache = detect_regime(df)
            last_close: float = df_cache["close_clean"].iloc[-1]
            raw_close: float = current_market_price if current_market_price else df_cache["last_raw_close"]
            indicators: Dict[str, float] = {
                "ema20": df_cache["ema20"].iloc[-1],
                "sma20": df_cache["sma20"].iloc[-1],
                "rsi": df_cache["rsi"].iloc[-1],
                "macd": df_cache["macd"].iloc[-1],
                "macd_signal": df_cache["macd_signal"].iloc[-1],
                "bb_upper": df_cache["bb_upper"].iloc[-1],
                "bb_lower": df_cache["bb_lower"].iloc[-1],
                "stoch": df_cache["stoch"].iloc[-1],
                "ema50": df_cache["ema50"].iloc[-1],
                "vol_ma20": df_cache["vol_ma20"].iloc[-1],
                "volume": df_cache["volume"].iloc[-1]
            }
            analysis: str = analyze_market(indicators, raw_close, predictions_cache, df_cache)
            risk_metrics: Dict[str, Any] = calculate_risk_metrics(df_cache, raw_close, predictions_cache)
            clear_terminal()
            print_report(ticker, figi, last_close, raw_close, predictions_cache, current_time, indicators, regimes_cache, analysis, {}, risk_metrics, df_cache)
            await asyncio.sleep(data_update_interval.total_seconds())
    except KeyboardInterrupt:
        logging.info("Завершение работы...")
    except Exception as e:
        logging.error(f"Ошибка основного цикла: {e}")

if __name__ == "__main__":
    asyncio.run(main())
