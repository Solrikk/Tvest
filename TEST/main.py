
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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from colorama import init, Fore, Style
import asciichartpy as ascii_chart
from tqdm import tqdm

init(autoreset=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
TOKEN: str = os.environ.get("TINKOFF_TOKEN")
if not TOKEN:
    import getpass
    TOKEN = getpass.getpass("Введите ваш Tinkoff API токен: ")


def get_figi_by_ticker(ticker: str) -> str:
    try:
        with Client(TOKEN) as client:
            instruments = []
            for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
                instruments += getattr(client.instruments,
                                       method)().instruments
            filtered = [
                inst for inst in instruments
                if inst.ticker.upper() == ticker.upper()
            ]
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
                        interval=CandleInterval.CANDLE_INTERVAL_HOUR):
                    hourly_candles.append({
                        "time":
                        candle.time,
                        "open":
                        float(quotation_to_decimal(candle.open)),
                        "high":
                        float(quotation_to_decimal(candle.high)),
                        "low":
                        float(quotation_to_decimal(candle.low)),
                        "close":
                        float(quotation_to_decimal(candle.close)),
                        "volume":
                        candle.volume,
                        "timeframe":
                        "hour"
                    })
                async for candle in client.get_all_candles(
                        figi=figi,
                        from_=now() - timedelta(minutes=60),
                        to=now(),
                        interval=CandleInterval.CANDLE_INTERVAL_1_MIN):
                    minute_candles.append({
                        "time":
                        candle.time,
                        "open":
                        float(quotation_to_decimal(candle.open)),
                        "high":
                        float(quotation_to_decimal(candle.high)),
                        "low":
                        float(quotation_to_decimal(candle.low)),
                        "close":
                        float(quotation_to_decimal(candle.close)),
                        "volume":
                        candle.volume,
                        "timeframe":
                        "minute"
                    })
            break
        except Exception as e:
            logging.warning(
                f"Ошибка получения свечей: {e}. Повтор через {delay} сек.")
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
        expected_hours = pd.date_range(start=hourly_data.index.min(),
                                       end=hourly_data.index.max(),
                                       freq="h")
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
        roll_med = df["close_smooth"].rolling(window=window_size,
                                              min_periods=2).median()
        roll_std = df["close_smooth"].rolling(window=window_size,
                                              min_periods=2).std().fillna(0)
        diff = np.abs(df["close_smooth"] - roll_med)
        threshold_factor: float = 3.0 if len(df) > 30 else 4.0
        threshold = threshold_factor * roll_std
        df["close_clean"] = np.where(diff > threshold, roll_med,
                                     df["close_smooth"])
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


def create_fourier_features(df: pd.DataFrame,
                            periods: List[int] = [24, 168],
                            order: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["time_int"] = df.index.astype(np.int64) // 10**9
    for period in periods:
        for i in range(1, order + 1):
            df[f"sin_{period}_{i}"] = np.sin(2 * np.pi * i * df["time_int"] /
                                             (period * 3600))
            df[f"cos_{period}_{i}"] = np.cos(2 * np.pi * i * df["time_int"] /
                                             (period * 3600))
    df.drop(columns=["time_int"], inplace=True)
    return df


def create_features(df: pd.DataFrame,
                    lags: int = 3) -> Tuple[pd.DataFrame, List[str]]:
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["close_clean"].shift(i)
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df[f"lag_{i}"].bfill().ffill()
    for col in [
            "sma20", "ema20", "rsi", "macd", "macd_signal", "bb_upper",
            "bb_lower", "stoch", "ema50", "vol_ma20"
    ]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].ffill().bfill().fillna(0)
    df = create_fourier_features(df)
    df = df.dropna()
    feature_cols: List[str] = [f"lag_{i}" for i in range(1, lags + 1)] + [
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


def train_model_enhanced(
    df: pd.DataFrame,
    lags: int = 3
) -> Tuple[Any, StandardScaler, StandardScaler, List[str], pd.DataFrame]:
    if len(df) < 50:
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
    target = df_feat["close_clean"]
    correlations = {}
    for feature in feature_cols:
        corr = np.abs(df_feat[feature].corr(target))
        correlations[feature] = corr
    sorted_features = sorted(correlations.items(),
                             key=lambda x: abs(x[1]),
                             reverse=True)
    top_features = [
        f for f, _ in sorted_features[:min(20, len(sorted_features))]
    ]
    feature_cols = top_features
    test_size = max(int(len(df_feat) * 0.2), 10)
    train_df = df_feat.iloc[:-test_size] if test_size > 0 else df_feat.copy()
    test_df = df_feat.iloc[-test_size:] if test_size > 0 else pd.DataFrame()
    n_samples = len(train_df)
    n_splits = min(5, n_samples // 20)
    if n_splits < 2:
        n_splits = 2
    tscv = TimeSeriesSplit(n_splits=n_splits,
                           test_size=max(5, n_samples // 10))
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
    models = {
        "ridge": Ridge(),
        "elasticnet": ElasticNet(random_state=42),
        "gbr": GradientBoostingRegressor(random_state=42),
        "svr": make_pipeline(StandardScaler(), SVR())
    }
    if len(X_train) < 100:
        models.pop("gbr", None)
        models.pop("svr", None)
    param_grids = {
        "ridge": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "fit_intercept": [True, False],
            "solver":
            ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            "max_iter": [1000]
        },
        "elasticnet": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            "fit_intercept": [True, False],
            "selection": ["random", "cyclic"]
        },
        "gbr": {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
            "max_depth": [2, 3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "subsample": [0.8, 0.9, 1.0]
        },
        "svr": {
            "svr__C": [0.1, 1.0, 10.0],
            "svr__epsilon": [0.1, 0.2, 0.5],
            "svr__kernel": ["linear", "rbf"]
        }
    }
    best_model_name = None
    best_model = None
    best_params = None
    best_score = -float("inf")
    best_std = float("inf")
    for model_name, model in models.items():
        logging.info(f"Оптимизация модели: {model_name}")
        param_grid = param_grids[model_name]
        if model_name == "gbr" and len(train_df) < 200:
            param_grid = {
                "learning_rate": [0.01, 0.1],
                "n_estimators": [50, 100],
                "max_depth": [2, 3],
                "subsample": [0.8, 1.0]
            }
        total_combinations = len(list(itertools.product(*param_grid.values())))
        if total_combinations > 20:
            n_iter = min(20, total_combinations)
            search = RandomizedSearchCV(model,
                                        param_grid,
                                        n_iter=n_iter,
                                        cv=tscv,
                                        scoring="neg_mean_squared_error",
                                        random_state=42)
            search.fit(X_train_scaled, y_train_scaled)
            best_run_score = -search.best_score_
            best_model_params = search.best_params_
            best_run_std = search.cv_results_["std_test_score"][
                search.best_index_]
        else:
            best_param_score = float("inf")
            best_param_std = float("inf")
            best_model_params = None
            for params in itertools.product(*param_grid.values()):
                param_dict = {
                    name: value
                    for name, value in zip(list(param_grid.keys()), params)
                }
                model.set_params(**param_dict)
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_train_scaled):
                    X_cv_train, X_cv_val = X_train_scaled[
                        train_idx], X_train_scaled[val_idx]
                    y_cv_train, y_cv_val = y_train_scaled[
                        train_idx], y_train_scaled[val_idx]
                    model.fit(X_cv_train, y_cv_train)
                    pred_val = model.predict(X_cv_val)
                    mse = mean_squared_error(y_cv_val, pred_val)
                    cv_scores.append(mse)
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                if mean_score < best_param_score or (
                        mean_score == best_param_score
                        and std_score < best_param_std):
                    best_param_score = mean_score
                    best_param_std = std_score
                    best_model_params = param_dict
            best_run_score = best_param_score
            best_run_std = best_param_std
        y_std = np.std(y_train_scaled)
        r2_like_score = 1 - (best_run_score /
                             (y_std**2)) if y_std > 0 else -best_run_score
        logging.info(
            f"{model_name}: best score = {r2_like_score:.4f}, std = {best_run_std:.4f}"
        )
        if r2_like_score > best_score or (r2_like_score >= best_score * 0.95
                                          and best_run_std < best_std * 0.8):
            best_score = r2_like_score
            best_std = best_run_std
            best_model_name = model_name
            best_params = best_model_params
    logging.info(
        f"Выбрана модель: {best_model_name} с параметрами: {best_params}")
    if best_model_name == "ridge":
        best_model = Ridge(**best_params)
    elif best_model_name == "elasticnet":
        best_model = ElasticNet(**best_params, random_state=42)
    elif best_model_name == "gbr":
        best_model = GradientBoostingRegressor(**best_params, random_state=42)
    elif best_model_name == "svr":
        best_model = make_pipeline(
            StandardScaler(),
            SVR(**{
                k.replace('svr__', ''): v
                for k, v in best_params.items()
            }))
    else:
        best_model = Ridge(alpha=1.0)
    best_model.fit(X_train_scaled, y_train_scaled)
    if len(test_df) > 0:
        X_test = test_df[feature_cols].values
        y_test = test_df["close_clean"].values
        for i in range(X_test.shape[1]):
            if i in lower_bounds:
                X_test[:, i] = np.clip(X_test[:, i], lower_bounds[i],
                                       upper_bounds[i])
        X_test_scaled = scaler_X.transform(X_test)
        pred_test_scaled = best_model.predict(X_test_scaled)
        pred_test = scaler_y.inverse_transform(pred_test_scaled.reshape(
            -1, 1)).ravel()
        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        r2 = r2_score(y_test, pred_test)
        logging.info(
            f"Оценка на тестовых данных: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}"
        )
        pred_train_scaled = best_model.predict(X_train_scaled)
        pred_train = scaler_y.inverse_transform(
            pred_train_scaled.reshape(-1, 1)).ravel()
        train_rmse = np.sqrt(
            mean_squared_error(train_df["close_clean"].values, pred_train))
        overfit_ratio = rmse / train_rmse if train_rmse > 0 else float('inf')
        if overfit_ratio > 1.5:
            logging.warning(
                f"Возможное переобучение модели! Отношение RMSE(тест)/RMSE(обучение) = {overfit_ratio:.2f}"
            )
    if hasattr(best_model, 'feature_importances_') or (
            hasattr(best_model, 'steps')
            and hasattr(best_model.steps[-1][1], 'feature_importances_')):
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        else:
            importances = best_model.steps[-1][1].feature_importances_
        feature_importance = [
            (feature, importance)
            for feature, importance in zip(feature_cols, importances)
        ]
        sorted_importance = sorted(feature_importance,
                                   key=lambda x: x[1],
                                   reverse=True)
        logging.info("Важность признаков:")
        for feature, importance in sorted_importance[:10]:
            logging.info(f"  - {feature}: {importance:.4f}")
    return best_model, scaler_X, scaler_y, feature_cols, df_feat


def adaptive_alpha(model_confidence, market_volatility, horizon):
    base_alpha = max(0.2, 0.9 - 0.1 * horizon)
    vol_factor = 1.0 - min(1.0, market_volatility / 0.05)
    final_alpha = base_alpha * model_confidence * vol_factor
    return final_alpha


def improved_multivariate_prediction(model,
                                     scaler_X,
                                     scaler_y,
                                     feature_cols,
                                     df,
                                     steps=5,
                                     lags=3):
    horizon_models = {}
    horizon_scalers_y = {}
    try:
        for h in range(1, min(steps + 1, 4)):
            X_train = []
            y_train = []
            for i in range(lags, len(df) - h):
                features = df.iloc[i][feature_cols].values
                target = df.iloc[i + h]["close_clean"]
                X_train.append(features)
                y_train.append(target)
            if X_train and y_train:
                X_train = np.array(X_train)
                y_train = np.array(y_train).reshape(-1, 1)
                h_scaler_y = StandardScaler().fit(y_train)
                y_train_scaled = h_scaler_y.transform(y_train).ravel()
                h_model = Ridge(alpha=1.0).fit(scaler_X.transform(X_train),
                                               y_train_scaled)
                horizon_models[h] = h_model
                horizon_scalers_y[h] = h_scaler_y
    except Exception as e:
        logging.warning(f"Ошибка при создании многогоризонтных моделей: {e}")
    predictions = []
    last_row = df.iloc[-1]
    current_price = last_row["close_clean"]
    model_conf = 0.8
    volatility = df["close_clean"].pct_change().rolling(5).std().iloc[-1]
    for step in range(1, steps + 1):
        try:
            if step in horizon_models:
                input_vector = last_row[feature_cols].values.reshape(1, -1)
                input_scaled = scaler_X.transform(input_vector)
                h_model = horizon_models[step]
                h_scaler_y = horizon_scalers_y[step]
                pred_scaled = h_model.predict(input_scaled)
                model_pred = h_scaler_y.inverse_transform(
                    pred_scaled.reshape(-1, 1))[0, 0]
                confidence = max(0.6, 1.0 - 0.1 * step)
            else:
                input_vector = last_row[feature_cols].values.reshape(1, -1)
                input_scaled = scaler_X.transform(input_vector)
                pred_scaled = model.predict(input_scaled)
                model_pred = scaler_y.inverse_transform(
                    pred_scaled.reshape(-1, 1))[0, 0]
                confidence = max(0.5, 0.9 - 0.1 * step)
            alpha = adaptive_alpha(model_conf, volatility, step)
            pred = alpha * model_pred + (1 - alpha) * current_price
            predictions.append(pred)
        except Exception as e:
            logging.warning(f"Ошибка прогнозирования на шаге {step}: {e}")
            if predictions:
                last_pred = predictions[-1]
                if len(predictions) > 1:
                    trend = (predictions[-1] - predictions[-2])
                    predictions.append(last_pred + trend)
                else:
                    predictions.append(last_pred)
            else:
                predictions.append(current_price)
    return predictions


def analyze_volatility(df):
    recent_vol = df["close_clean"].pct_change().rolling(
        window=5).std().iloc[-1]
    hist_vol = df["close_clean"].pct_change().rolling(window=20).std().iloc[-1]
    recent_returns = df["close_clean"].pct_change().iloc[-5:]
    recent_mean = recent_returns.mean()
    abs_mean = abs(recent_mean)
    directional_ratio = abs_mean / recent_vol if recent_vol > 0 else 0
    vol_factor = 0
    vol_note = ""
    if hist_vol > 0 and recent_vol > 1.8 * hist_vol:
        if directional_ratio > 0.7:
            if recent_mean > 0:
                vol_factor = 2
                vol_note = "Направленная волатильность (бычья)"
            else:
                vol_factor = -2
                vol_note = "Направленная волатильность (медвежья)"
        else:
            vol_factor = -2
            vol_note = "Высокая хаотичная волатильность"
    elif hist_vol > 0 and recent_vol < 0.5 * hist_vol:
        vol_factor = -1
        vol_note = "Сжатие волатильности (затишье перед бурей)"
    return vol_factor, vol_note


def is_trending_market(df):
    returns = df["close_clean"].pct_change().dropna()
    return returns.rolling(
        window=10).mean().iloc[-1] > 0 if len(returns) >= 10 else False


def dynamic_rsi_thresholds(df):
    if df is None or "rsi" not in df.columns:
        return 30, 70
    
    rsi_history = df["rsi"].dropna()
    if len(rsi_history) < 30:
        return 30, 70
    
    low_threshold = max(10, rsi_history.quantile(0.1))
    high_threshold = min(90, rsi_history.quantile(0.9))
    
    recent_volatility = df["close"].pct_change().rolling(5).std().iloc[-5:].mean() * 100
    historical_volatility = df["close"].pct_change().rolling(20).std().iloc[-20:].mean() * 100
    vol_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1.0
    
    recent_price_change = df["close"].iloc[-5:].pct_change(5).iloc[-1] * 100
    
    is_trend = is_trending_market(df)
    
    trend_strength = 0
    if abs(recent_price_change) > 3:
        trend_strength = 1 if recent_price_change > 0 else -1
    
    recent_rsi = rsi_history.iloc[-10:]
    rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
    
    if is_trend:
        if trend_strength > 0:
            low_threshold = max(20, low_threshold)
            high_threshold = min(85, high_threshold + 5)
        elif trend_strength < 0:
            low_threshold = max(15, low_threshold - 5)
            high_threshold = min(80, high_threshold)
    else:
        mid_point = (low_threshold + high_threshold) / 2
        range_factor = 0.8 if vol_ratio < 1.5 else 1.0
        low_threshold = max(25, mid_point - (mid_point - low_threshold) * range_factor)
        high_threshold = min(75, mid_point + (high_threshold - mid_point) * range_factor)
    
    if vol_ratio > 2.0:
        low_threshold = max(15, low_threshold - 5)
        high_threshold = min(85, high_threshold + 5)
    
    if abs(rsi_trend) > 3:
        if rsi_trend > 0:
            high_threshold = max(70, min(90, high_threshold + 3))
        else:
            low_threshold = max(10, min(30, low_threshold - 3))
    
    low_threshold = round(low_threshold, 1)
    high_threshold = round(high_threshold, 1)
    
    return low_threshold, high_threshold


def analyze_volume_with_price(df, current_price):
    if "volume" not in df.columns or len(df) < 20:
        return 0, "Нет данных по объему"
    recent = df.iloc[-5:]
    avg_vol = df["volume"].iloc[-20:].mean()
    if avg_vol == 0:
        return 0, "Нет объема"
    up_candles = recent[recent["close"] > recent["open"]]
    down_candles = recent[recent["close"] < recent["open"]]
    up_vol = up_candles["volume"].sum() / max(1, len(up_candles))
    down_vol = down_candles["volume"].sum() / max(1, len(down_candles))
    last_vol = df["volume"].iloc[-1]
    last_price_change = df["close"].iloc[-1] - df["open"].iloc[-1]
    vol_score = 0
    vol_message = ""
    if last_vol > 2 * avg_vol:
        if last_price_change > 0:
            if up_vol > down_vol * 1.5:
                vol_score = 3
                vol_message = "Высокий объем покупок"
            else:
                vol_score = 1
                vol_message = "Повышенный объем при росте цены"
        else:
            if down_vol > up_vol * 1.5:
                vol_score = -3
                vol_message = "Высокий объем продаж"
            else:
                vol_score = -1
                vol_message = "Повышенный объем при падении цены"
    elif last_vol < 0.5 * avg_vol:
        vol_score = -0.5
        vol_message = "Низкий торговый объем (низкая ликвидность)"
    price_trend = 1 if df["close"].iloc[-5:].corr(pd.Series(
        range(5))) > 0.7 else -1 if df["close"].iloc[-5:].corr(
            pd.Series(range(5))) < -0.7 else 0
    volume_trend = 1 if df["volume"].iloc[-5:].corr(pd.Series(
        range(5))) > 0.7 else -1 if df["volume"].iloc[-5:].corr(
            pd.Series(range(5))) < -0.7 else 0
    if price_trend == 1 and volume_trend == -1:
        vol_score -= 2
        vol_message += " | Дивергенция: рост цены на падающем объеме"
    elif price_trend == -1 and volume_trend == -1:
        vol_score += 1
        vol_message += " | Ослабление нисходящего тренда (падение объема)"
    return vol_score, vol_message


def calculate_market_health_index(df: pd.DataFrame) -> Tuple[float, str]:
    if df is None or len(df) < 30:
        return 50.0, "Недостаточно данных"
    
    rsi = df["rsi"].iloc[-1]
    close = df["close_clean"].iloc[-1]
    vol_ratio = df["volume"].iloc[-1] / df["vol_ma20"].iloc[-1] if df["vol_ma20"].iloc[-1] > 0 else 1.0
    
    asset_class = detect_asset_class(df)
    
    price_trend = 0
    sma20 = df["sma20"].iloc[-1]
    ema50 = df["ema50"].iloc[-1]
    if close > sma20 and sma20 > ema50:
        price_trend = 10
    elif close < sma20 and sma20 < ema50:
        price_trend = -10
    else:
        ema_slope = df["ema20"].iloc[-5:].diff().mean()
        slope_factor = 100
        if asset_class == "forex":
            slope_factor = 5000
        elif asset_class == "crypto":
            slope_factor = 50
        
        price_trend = min(10, max(-10, ema_slope * slope_factor))
    
    momentum = 0
    low_rsi, high_rsi = 30, 70
    
    if asset_class == "crypto":
        low_rsi, high_rsi = 40, 60
    elif asset_class == "forex":
        low_rsi, high_rsi = 25, 75
    
    if asset_class == "stock":
        regimes = detect_regime(df)
        if regimes.get("2", 0) > 0.6:
            low_rsi -= 5
            high_rsi -= 5
        elif regimes.get("0", 0) > 0.6:
            low_rsi += 5
            high_rsi += 5
    
    if low_rsi <= rsi <= high_rsi:
        momentum = 10 * (rsi - ((high_rsi + low_rsi) / 2)) / ((high_rsi - low_rsi) / 2)
    elif rsi < low_rsi:
        momentum = -10 + (rsi / (low_rsi / 10))
    else:
        momentum = 10 - ((rsi - high_rsi) / ((100 - high_rsi) / 10))
    
    volatility_score = 0
    volatility_5d = df["close_clean"].pct_change().rolling(5).std().iloc[-1] * 100
    
    low_vol, normal_vol, high_vol = 0.5, 2.0, 4.0
    
    if asset_class == "crypto":
        low_vol, normal_vol, high_vol = 2.0, 5.0, 10.0
    elif asset_class == "forex":
        low_vol, normal_vol, high_vol = 0.2, 0.8, 1.5
    
    if low_vol <= volatility_5d <= normal_vol:
        volatility_score = 10
    elif volatility_5d < low_vol:
        volatility_score = max(0, 10 * (volatility_5d / low_vol))
    else:
        decay_factor = 1.0 if asset_class != "crypto" else 0.5
        volatility_score = max(0, 10 - min(10, (volatility_5d - normal_vol) * decay_factor))
    
    volume_score = 0
    
    low_vol_ratio, high_vol_ratio = 0.8, 1.5
    
    if asset_class == "crypto":
        low_vol_ratio, high_vol_ratio = 0.6, 2.0
    elif asset_class == "forex":
        low_vol_ratio, high_vol_ratio = 0.7, 1.3
    
    if low_vol_ratio <= vol_ratio <= high_vol_ratio:
        volume_score = 10
    elif vol_ratio < low_vol_ratio:
        volume_score = max(0, 10 * (vol_ratio / low_vol_ratio))
    else:
        price_up = df["close_clean"].iloc[-1] > df["open"].iloc[-1]
        if (price_up and vol_ratio > 2.0) or (not price_up and vol_ratio < 2.0):
            volume_score = 8
        else:
            volume_score = 5
    
    bb_score = 0
    bb_upper = df["bb_upper"].iloc[-1]
    bb_lower = df["bb_lower"].iloc[-1]
    bb_width = (bb_upper - bb_lower) / df["sma20"].iloc[-1]
    
    bb_position = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    
    narrow_bb = 0.02
    if asset_class == "crypto":
        narrow_bb = 0.04
    elif asset_class == "forex":
        narrow_bb = 0.01
    
    if bb_width < narrow_bb:
        bb_score = 0
    elif 0.1 <= bb_position <= 0.9:
        bb_score = 10
    elif bb_position < 0.1:
        bb_score = 5
    else:
        bb_score = 5
    
    weights = get_adaptive_weights(asset_class, df)
    
    normalized_trend = (price_trend + 10) * 5
    normalized_momentum = (momentum + 10) * 5
    normalized_volatility = volatility_score * 10
    normalized_volume = volume_score * 10
    normalized_bb = bb_score * 10
    
    health_index = (weights["trend"] * normalized_trend +
                   weights["momentum"] * normalized_momentum +
                   weights["volatility"] * normalized_volatility +
                   weights["volume"] * normalized_volume +
                   weights["bollinger"] * normalized_bb)
    
    if health_index >= 70:
        status = "Сильный рынок"
    elif health_index >= 55:
        status = "Здоровый рынок"
    elif health_index >= 45:
        status = "Нейтральный рынок"
    elif health_index >= 30:
        status = "Слабый рынок"
    else:
        status = "Нездоровый рынок"
    
    return health_index, status


def detect_asset_class(df: pd.DataFrame) -> str:
    if df is None or len(df) < 30:
        return "stock"
    
    avg_price = df["close"].mean()
    volatility = df["close"].pct_change().std() * 100
    
    if volatility > 5.0:
        return "crypto"
    
    if volatility < 0.5 and (0.5 < avg_price < 2.0 or 100 < avg_price < 200):
        return "forex"
    
    return "stock"


def get_adaptive_weights(asset_class: str, df: pd.DataFrame) -> Dict[str, float]:
    weights = {
        "trend": 0.25,
        "momentum": 0.2,
        "volatility": 0.15,
        "volume": 0.2,
        "bollinger": 0.2
    }
    
    if asset_class == "crypto":
        weights = {
            "trend": 0.2,
            "momentum": 0.25,
            "volatility": 0.15,
            "volume": 0.25,
            "bollinger": 0.15
        }
    elif asset_class == "forex":
        weights = {
            "trend": 0.3,
            "momentum": 0.2,
            "volatility": 0.1,
            "volume": 0.15,
            "bollinger": 0.25
        }
    
    if df is not None and len(df) > 30:
        volatility = df["close"].pct_change().rolling(20).std().iloc[-1] * 100
        
        if volatility > 3.0:
            factor = min(0.1, (volatility - 3.0) / 10.0)
            weights["volatility"] += factor
            weights["volume"] += factor
            weights["trend"] -= factor
            weights["bollinger"] -= factor
        
        elif volatility < 1.0:
            factor = min(0.1, (1.0 - volatility) / 5.0)
            weights["trend"] += factor
            weights["bollinger"] += factor
            weights["volatility"] -= factor / 2
            weights["volume"] -= factor / 2
    
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total
    
    return weights

def analyze_complex_volume_patterns(df: pd.DataFrame) -> Tuple[float, str]:
    if df is None or len(df) < 20:
        return 0, "Недостаточно данных для анализа объема"
    
    recent = df.iloc[-10:]
    
    avg_vol_20 = df["volume"].iloc[-20:].mean()
    if avg_vol_20 == 0:
        return 0, "Нет объема"
    
    up_candles = recent[recent["close"] > recent["open"]]
    down_candles = recent[recent["close"] < recent["open"]]
    up_vol = up_candles["volume"].sum() / max(1, len(up_candles))
    down_vol = down_candles["volume"].sum() / max(1, len(down_candles))
    
    last_5 = df.iloc[-5:]
    volumes = last_5["volume"].values
    closes = last_5["close"].values
    opens = last_5["open"].values
    
    vol_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
    price_trend = np.polyfit(range(len(closes)), closes, 1)[0]
    
    max_vol_idx = np.argmax(volumes)
    max_vol_ratio = volumes[max_vol_idx] / avg_vol_20
    
    price_range = (max(closes) - min(closes)) / min(closes) * 100
    
    vol_v_shape = False
    price_v_shape = False
    
    if len(volumes) >= 5:
        if (volumes[0] > volumes[1] > volumes[2] and volumes[2] < volumes[3] < volumes[4]):
            vol_v_shape = True
        
        if (closes[0] > closes[1] > closes[2] and closes[2] < closes[3] < closes[4]):
            price_v_shape = True
    
    liquidity_score = 0
    liquidity_message = ""
    
    vol_stability = 1.0 - min(1.0, df["volume"].iloc[-20:].std() / (avg_vol_20 + 1e-10))
    
    daily_volume = avg_vol_20 * 24
    
    avg_spread_pct = np.mean((df["high"].iloc[-20:] - df["low"].iloc[-20:]) / df["close"].iloc[-20:]) * 100
    
    vol_price_ratio = avg_vol_20 / df["close"].iloc[-1]
    
    if daily_volume < 1000 or vol_stability < 0.3 or avg_spread_pct > 3.0:
        liquidity_score = -3
        liquidity_message = "Низкая ликвидность: возможны ложные сигналы"
    elif daily_volume < 10000 or vol_stability < 0.5 or avg_spread_pct > 1.0:
        liquidity_score = -1
        liquidity_message = "Умеренная ликвидность: требуется осторожность"
    else:
        liquidity_score = 1
        liquidity_message = "Хорошая ликвидность инструмента"
    
    recent_vol_ratio = df["volume"].iloc[-5:].mean() / avg_vol_20
    if recent_vol_ratio < 0.5:
        liquidity_score -= 1
        liquidity_message = "Снижение ликвидности в последних сессиях"
    
    volume_score = 0
    messages = []
    
    messages.append(liquidity_message)
    volume_score += liquidity_score
    
    if vol_trend < 0 and price_trend > 0:
        volume_score -= 3
        messages.append("Дивергенция: рост цены при падающем объеме")
    elif vol_trend > 0 and price_trend > 0:
        volume_score += 3
        messages.append("Подтверждение: рост цены при растущем объеме")
    elif vol_trend < 0 and price_trend < 0:
        volume_score += 1
        messages.append("Ослабление медвежьего тренда (падение объема)")
    elif vol_trend > 0 and price_trend < 0:
        volume_score -= 2
        messages.append("Усиление медвежьего тренда (рост объема)")
    
    if max_vol_ratio > 3:
        candle_bullish = closes[max_vol_idx] > opens[max_vol_idx]
        if candle_bullish:
            volume_score += 2
            messages.append(f"Сильный объемный всплеск на бычьей свече ({max_vol_ratio:.1f}x)")
        else:
            volume_score -= 2
            messages.append(f"Сильный объемный всплеск на медвежьей свече ({max_vol_ratio:.1f}x)")
    
    if vol_v_shape and price_v_shape:
        volume_score += 3
        messages.append("V-образный разворот с объемным подтверждением")
    
    if price_range > 5 and vol_trend < 0:
        volume_score -= 1
        messages.append(f"Истощение тренда: движение {price_range:.1f}% с падающим объемом")
    
    if len(up_candles) > 0 and len(down_candles) > 0:
        up_down_ratio = up_vol / down_vol if down_vol > 0 else 999
        if up_down_ratio > 2:
            volume_score += 2
            messages.append(f"Бычий объемный перевес: в {up_down_ratio:.1f}x больше объема на росте")
        elif up_down_ratio < 0.5:
            volume_score -= 2
            messages.append(f"Медвежий объемный перевес: в {1/up_down_ratio:.1f}x больше объема на падении")
    
    if len(messages) <= 1:
        messages.append("Нет явных объемных паттернов")
    
    return volume_score, " | ".join(messages[:2])

def detect_potential_bounce(df: pd.DataFrame, indicators: Dict[str, float], is_short_signal: bool) -> Tuple[bool, str, float]:
    if df is None or len(df) < 10:
        return False, "", 0
    
    recent = df.iloc[-5:]
    bounce_probability = 0
    reasons = []
    
    rsi = indicators["rsi"]
    if is_short_signal and rsi < 30:
        bounce_probability += 3
        reasons.append(f"RSI перепродан ({rsi:.1f})")
    elif not is_short_signal and rsi > 70:
        bounce_probability += 3
        reasons.append(f"RSI перекуплен ({rsi:.1f})")
    
    price = recent["close"].iloc[-1]
    bb_upper = indicators["bb_upper"]
    bb_lower = indicators["bb_lower"]
    
    bb_position = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    
    if is_short_signal and bb_position < 0.1:
        bounce_probability += 3
        reasons.append("Цена у нижней полосы Боллинджера")
    elif not is_short_signal and bb_position > 0.9:
        bounce_probability += 3
        reasons.append("Цена у верхней полосы Боллинджера")
    
    closes = recent["close"].values
    opens = recent["open"].values
    highs = recent["high"].values
    lows = recent["low"].values
    
    if is_short_signal and len(lows) > 0 and len(closes) > 0 and len(opens) > 0:
        last_candle_body = abs(closes[-1] - opens[-1])
        last_candle_shadow = min(opens[-1], closes[-1]) - lows[-1]
        
        if last_candle_shadow > 2 * last_candle_body and last_candle_body > 0:
            bounce_probability += 2
            reasons.append("Свечной паттерн 'молот'")
    
    if not is_short_signal and len(highs) > 0 and len(closes) > 0 and len(opens) > 0:
        last_candle_body = abs(closes[-1] - opens[-1])
        last_candle_shadow = highs[-1] - max(opens[-1], closes[-1])
        
        if last_candle_shadow > 2 * last_candle_body and last_candle_body > 0:
            bounce_probability += 2
            reasons.append("Свечной паттерн 'падающая звезда'")
    
    if len(df) >= 10:
        price_change = df["close"].iloc[-5:].pct_change(5).iloc[-1]
        rsi_change = df["rsi"].iloc[-5:].diff(5).iloc[-1] / 50
        
        if is_short_signal and price_change < 0 and rsi_change > 0:
            bounce_probability += 3
            reasons.append("Бычья дивергенция RSI")
        elif not is_short_signal and price_change > 0 and rsi_change < 0:
            bounce_probability += 3
            reasons.append("Медвежья дивергенция RSI")
    
    if len(df) >= 10:
        price_deviation = (df["close"].iloc[-1] - df["sma20"].iloc[-1]) / df["sma20"].iloc[-1] * 100
        
        if is_short_signal and price_deviation < -10:
            bounce_probability += 2
            reasons.append(f"Сильное отклонение от SMA20 ({price_deviation:.1f}%)")
        elif not is_short_signal and price_deviation > 10:
            bounce_probability += 2
            reasons.append(f"Сильное отклонение от SMA20 ({price_deviation:.1f}%)")
    
    bounce_probability = min(10, bounce_probability)
    
    is_bounce_likely = bounce_probability >= 5
    reason_text = ", ".join(reasons[:2]) if reasons else "Нет явных признаков отскока"
    
    return is_bounce_likely, reason_text, bounce_probability

def analyze_market(ind: Dict[str, float],
                   current_price: float,
                   predictions: List[float],
                   df: pd.DataFrame = None) -> str:
    weights: Dict[str, float] = {
        "rsi": 0.25,
        "macd": 0.25,
        "ema": 0.2,
        "stoch": 0.15,
        "health": 0.15
    }
    score: float = 0
    factors: List[str] = []
    market_description: List[str] = []
    
    health_index, health_status = calculate_market_health_index(df) if df is not None else (50.0, "Нет данных")
    
    health_contribution = (health_index - 50) / 10
    score += weights["health"] * health_contribution
    factors.append(f"Индекс здоровья: {health_index:.1f} ({health_status})")
    
    if health_index >= 70:
        market_description.append("Рынок в сильном состоянии с преобладающими бычьими тенденциями. Наблюдается высокая вероятность продолжения роста.")
    elif health_index >= 60:
        market_description.append("Рынок здоровый, с позитивным настроем. Покупатели контролируют ситуацию, падения могут быть хорошими точками для входа в длинные позиции.")
    elif health_index >= 50:
        market_description.append("Рынок в нейтрально-позитивном состоянии. Возможны движения в обоих направлениях, но с небольшим преимуществом покупателей.")
    elif health_index >= 40:
        market_description.append("Рынок в нейтрально-негативном состоянии. Покупатели и продавцы находятся в противостоянии, с небольшим преимуществом продавцов.")
    elif health_index >= 30:
        market_description.append("Рынок ослаблен, преобладают медвежьи настроения. Рискованно открывать длинные позиции без явных признаков разворота.")
    else:
        market_description.append("Рынок в слабом состоянии, сильное медвежье давление. Возможны краткосрочные отскоки на перепроданности, но основной тренд остается нисходящим.")
    
    if df is not None and len(df) > 20:
        vol_factor, vol_note = analyze_volatility(df)
        score += vol_factor
        factors.append(vol_note)
        
        recent_vol = df["close_clean"].pct_change().rolling(window=5).std().iloc[-1] * 100
        hist_vol = df["close_clean"].pct_change().rolling(window=20).std().iloc[-1] * 100
        
        if recent_vol > 1.8 * hist_vol:
            market_description.append(f"Волатильность резко повысилась (текущая: {recent_vol:.2f}%, историческая: {hist_vol:.2f}%). Это может указывать на смену тренда или начало импульсного движения.")
        elif recent_vol < 0.5 * hist_vol:
            market_description.append(f"Волатильность необычно низкая (текущая: {recent_vol:.2f}%, историческая: {hist_vol:.2f}%). Возможно, рынок готовится к резкому движению ('затишье перед бурей').")
        else:
            market_description.append(f"Волатильность в пределах нормы (текущая: {recent_vol:.2f}%, историческая: {hist_vol:.2f}%).")
    
    if df is not None and len(df) > 1:
        last_close = df["close_clean"].iloc[-2]
        current_open = df["open"].iloc[-1]
        gap_percent = (current_open - last_close) / last_close * 100
        
        if abs(gap_percent) > 2:
            if current_open > last_close:
                score += 3
                factors.append("Гэп вверх")
                market_description.append(f"Обнаружен значительный гэп вверх ({gap_percent:.2f}%). Это может указывать на сильное бычье давление или реакцию на позитивные новости.")
            else:
                score -= 3
                factors.append("Гэп вниз")
                market_description.append(f"Обнаружен значительный гэп вниз ({gap_percent:.2f}%). Это может указывать на сильное медвежье давление или реакцию на негативные новости.")
    
    low_rsi, high_rsi = dynamic_rsi_thresholds(
        df) if df is not None and "rsi" in df.columns else (30, 70)
    rsi = ind["rsi"]
    if rsi < low_rsi:
        rsi_score = weights["rsi"] * (2 * (low_rsi - rsi) / low_rsi)
        rsi_msg = f"RSI перепродан ({rsi:.1f} < {low_rsi:.1f})"
        market_description.append(f"RSI в зоне перепроданности ({rsi:.1f}), что может указывать на истощение продавцов и возможный отскок цены вверх.")
    elif rsi > high_rsi:
        rsi_score = -weights["rsi"] * (2 * (rsi - high_rsi) / (100 - high_rsi))
        rsi_msg = f"RSI перекуплен ({rsi:.1f} > {high_rsi:.1f})"
        market_description.append(f"RSI в зоне перекупленности ({rsi:.1f}), что может указывать на истощение покупателей и возможную коррекцию цены вниз.")
    else:
        norm_rsi = (rsi - 50) / (high_rsi - low_rsi) * 30
        rsi_score = -weights["rsi"] * norm_rsi
        rsi_msg = f"RSI нейтрален ({rsi:.1f})"
        
        if df is not None and len(df) > 5:
            rsi_slope = df["rsi"].iloc[-5:].diff().mean()
            if rsi_slope > 2:
                market_description.append(f"RSI в нейтральной зоне ({rsi:.1f}), но наблюдается уверенный рост индикатора, что указывает на усиление покупателей.")
            elif rsi_slope < -2:
                market_description.append(f"RSI в нейтральной зоне ({rsi:.1f}), но наблюдается уверенное снижение индикатора, что указывает на усиление продавцов.")
            else:
                market_description.append(f"RSI в нейтральной зоне ({rsi:.1f}) без выраженной динамики.")
    
    score += rsi_score
    factors.append(rsi_msg)
    
    macd_diff = ind["macd"] - ind["macd_signal"]
    score += weights["macd"] * macd_diff
    
    if macd_diff > 0.05:
        market_description.append(f"MACD выше сигнальной линии (разница: {macd_diff:.3f}). Индикатор подтверждает бычий импульс.")
    elif macd_diff < -0.05:
        market_description.append(f"MACD ниже сигнальной линии (разница: {macd_diff:.3f}). Индикатор подтверждает медвежий импульс.")
    else:
        market_description.append(f"MACD около сигнальной линии (разница: {macd_diff:.3f}). Возможно формирование нового тренда или продолжение флэта.")
    
    ema_trend = 3 if (current_price > ind["ema20"] and ind["ema20"] > ind["ema50"]) else -3 if (current_price < ind["ema20"] and ind["ema20"] < ind["ema50"]) else 0
    score += weights["ema"] * ema_trend
    
    if ema_trend > 0:
        market_description.append("Цена выше EMA20 и EMA50, что подтверждает восходящий тренд. Скользящие средние выстроены в бычьем порядке.")
    elif ema_trend < 0:
        market_description.append("Цена ниже EMA20 и EMA50, что подтверждает нисходящий тренд. Скользящие средние выстроены в медвежьем порядке.")
    else:
        if abs(ind["ema20"] - ind["ema50"]) / ind["ema50"] < 0.005:
            market_description.append("EMA20 и EMA50 находятся близко друг к другу. Возможно скорое формирование нового тренда.")
        else:
            market_description.append("Неоднозначное положение цены относительно EMA20 и EMA50. Тренд может быть в процессе смены направления.")
    
    stoch_signal = 2 if ind["stoch"] < 20 else -2 if ind["stoch"] > 80 else 0
    score += weights["stoch"] * stoch_signal
    
    if ind["stoch"] < 20:
        market_description.append(f"Стохастик в зоне перепроданности ({ind['stoch']:.1f}), что может сигнализировать о возможном отскоке цены вверх.")
    elif ind["stoch"] > 80:
        market_description.append(f"Стохастик в зоне перекупленности ({ind['stoch']:.1f}), что может сигнализировать о возможной коррекции цены вниз.")
    else:
        market_description.append(f"Стохастик в нейтральной зоне ({ind['stoch']:.1f}).")
    
    vol_score, vol_message = analyze_complex_volume_patterns(df) if df is not None else (0, "")
    score += vol_score
    if vol_message:
        factors.append(vol_message)
        market_description.append(f"Анализ объемов: {vol_message}")
    
    regimes = detect_regime(df) if df is not None else {"0": 0.5, "1": 0.5}
    bullish_regime = regimes.get("0", 0.5)
    bearish_regime = regimes.get("2", 0.0)
    neutral_regime = regimes.get("1", 0.0)
    score += (bullish_regime - 0.5) * 10
    
    if bullish_regime > 0.6:
        market_description.append(f"Рынок находится преимущественно в бычьем режиме (вероятность: {bullish_regime*100:.1f}%). Фаза роста, благоприятная для длинных позиций.")
    elif bearish_regime > 0.6:
        market_description.append(f"Рынок находится преимущественно в медвежьем режиме (вероятность: {bearish_regime*100:.1f}%). Фаза снижения, благоприятная для коротких позиций.")
    elif neutral_regime > 0.6:
        market_description.append(f"Рынок находится преимущественно в нейтральном режиме (вероятность: {neutral_regime*100:.1f}%). Боковое движение в диапазоне.")
    else:
        market_description.append(f"Смешанный режим рынка с элементами бычьего ({bullish_regime*100:.1f}%), медвежьего ({bearish_regime*100:.1f}%) и нейтрального ({neutral_regime*100:.1f}%) поведения.")
    
    if predictions and len(predictions) > 1:
        pred_change = (predictions[-1] - current_price) / current_price * 100
        pred_direction = "рост" if pred_change > 0 else "снижение" if pred_change < 0 else "стабильность"
        market_description.append(f"Прогноз модели указывает на {pred_direction} цены на {abs(pred_change):.2f}% в ближайшее время.")
    
    is_long_signal = score >= 3
    is_short_signal = score <= -3
    
    if is_long_signal or is_short_signal:
        bounce_likely, bounce_reason, bounce_strength = detect_potential_bounce(
            df, ind, is_short_signal)
        
        if bounce_likely:
            if is_short_signal:
                score = max(-2.9, score + bounce_strength * 0.3)
                factors.insert(0, f"⚠️ Возможен отскок вверх ({bounce_reason})")
                market_description.append(f"⚠️ Несмотря на общий медвежий сигнал, обнаружены признаки возможного отскока вверх: {bounce_reason}")
            elif is_long_signal:
                score = min(2.9, score - bounce_strength * 0.3)
                factors.insert(0, f"⚠️ Возможен отскок вниз ({bounce_reason})")
                market_description.append(f"⚠️ Несмотря на общий бычий сигнал, обнаружены признаки возможного отскока вниз: {bounce_reason}")
    
    if score >= 3:
        signal = "СИГНАЛ к ЛОНГУ"
    elif score <= -3:
        signal = "СИГНАЛ к ШОРТУ"
    else:
        signal = "РЫНОК НЕЙТРАЛЕН"
    
    signal_line = f"{signal} (score: {score:.1f}) - {', '.join(factors[:3])}"
    market_state = "\n".join([f"• {desc}" for desc in market_description])
    
    return f"{signal_line}\n\nДЕТАЛЬНЫЙ АНАЛИЗ РЫНКА:\n{market_state}"


def detect_regime(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) < 40:
        return {"0": 0.5, "1": 0.5, "2": 0.0}
    returns = df["close_clean"].pct_change().dropna()
    if len(returns) < 30:
        return {"0": 0.5, "1": 0.5, "2": 0.0}
    adf_result = adfuller(returns)
    if adf_result[1] > 0.1:
        logging.warning(
            "Временной ряд нестационарен, использование модифицированных возвратов"
        )
        returns = returns.diff().dropna()
    lower_q, upper_q = returns.quantile([0.01, 0.99])
    filtered_returns = returns[(returns >= lower_q) & (returns <= upper_q)]
    if len(filtered_returns) < 30:
        filtered_returns = returns
    best_k = 2
    best_bic = float('inf')
    best_result = None
    for k in range(2, 5):
        try:
            mr = MarkovRegression(filtered_returns,
                                  k_regimes=k,
                                  trend='c',
                                  switching_variance=True,
                                  switching_trend=True)
            results = []
            for _ in range(3):
                try:
                    res = mr.fit(disp=False, maxiter=200)
                    results.append((res.bic, res))
                except Exception as e:
                    logging.warning(
                        f"Ошибка подгонки модели с {k} режимами: {e}")
                    continue
            if results:
                best_run_bic, best_run_result = min(results,
                                                    key=lambda x: x[0])
                if best_run_bic < best_bic:
                    best_bic = best_run_bic
                    best_result = best_run_result
                    best_k = k
        except Exception as e:
            logging.warning(f"Невозможно создать модель с {k} режимами: {e}")
    if best_result is None:
        logging.warning(
            "Не удалось подобрать оптимальную модель, использую базовую модель с 2 режимами"
        )
        try:
            mr_simple = MarkovRegression(filtered_returns,
                                         k_regimes=2,
                                         trend='c',
                                         switching_variance=False)
            best_result = mr_simple.fit(disp=False)
            best_k = 2
        except Exception:
            return {"0": 0.5, "1": 0.5, "2": 0.0}
    regimes = best_result.smoothed_marginal_probabilities.iloc[-1].to_dict()
    try:
        regime_states = best_result.smoothed_marginal_probabilities.idxmax(
            axis=1)
        regime_means = {}
        for i in range(best_k):
            regime_data = filtered_returns[regime_states == i]
            if not regime_data.empty:
                regime_means[i] = regime_data.mean()
            else:
                regime_means[i] = 0
        sorted_regimes = sorted(regime_means.items(), key=lambda x: x[1])
        labeled_regimes = {}
        if best_k == 2:
            bear_idx, bull_idx = sorted_regimes[0][0], sorted_regimes[-1][0]
            labeled_regimes["2"] = regimes.get(bear_idx, 0.0)
            labeled_regimes["0"] = regimes.get(bull_idx, 0.0)
            labeled_regimes["1"] = 0.0
        else:
            bear_idx = sorted_regimes[0][0]
            bull_idx = sorted_regimes[-1][0]
            labeled_regimes["2"] = regimes.get(bear_idx, 0.0)
            labeled_regimes["0"] = regimes.get(bull_idx, 0.0)
            neutral_prob = 0.0
            for i in range(1, len(sorted_regimes) - 1):
                neutral_idx = sorted_regimes[i][0]
                neutral_prob += regimes.get(neutral_idx, 0.0)
            labeled_regimes["1"] = neutral_prob
        sum_probs = sum(labeled_regimes.values())
        if sum_probs > 0:
            for k in labeled_regimes:
                labeled_regimes[k] /= sum_probs
    except Exception as e:
        logging.warning(f"Ошибка при определении характеристик режимов: {e}")
        labeled_regimes = {"0": 0.34, "1": 0.33, "2": 0.33}
    logging.info(
        f"Определены режимы рынка: k={best_k}, BIC={best_bic:.2f}, Вероятности: Бычий={labeled_regimes['0']:.2f}, Нейтральный={labeled_regimes['1']:.2f}, Медвежий={labeled_regimes['2']:.2f}"
    )
    return labeled_regimes


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def show_loading_animation(message: str = "Загрузка данных") -> None:
    clear_terminal()
    print(f"\033[1;36m{message}...\033[0m")
    for _ in tqdm(range(100),
                  desc="Прогресс",
                  ncols=80,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        time.sleep(0.01)


def calculate_risk_metrics(df: pd.DataFrame, current_price: float,
                           predictions: List[float]) -> Dict[str, Any]:
    if df is None or len(df) < 30:
        return {
            "var_95": 0,
            "cvar": 0,
            "max_loss_pct": 0,
            "risk_level": "Неизвестно"
        }
    returns = df["close_clean"].pct_change().dropna()
    if len(returns) < 20:
        return {
            "var_95": 0,
            "cvar": 0,
            "max_loss_pct": 0,
            "risk_level": "Недостаточно данных"
        }
    var_95_pct = abs(returns.quantile(0.05) * 100)
    var_95 = abs(returns.quantile(0.05) * current_price)
    cvar = abs(returns[returns <= returns.quantile(0.05)].mean() *
               current_price)
    rolling_max = df["close_clean"].rolling(window=30).max()
    daily_drawdown = df["close_clean"] / rolling_max - 1.0
    max_loss_pct = abs(daily_drawdown.min() * 100)
    risk_level = "Высокий" if var_95_pct > 5 else "Средний" if var_95_pct > 2 else "Низкий"
    return {
        "var_95": var_95,
        "cvar": cvar,
        "var_95_pct": var_95_pct,
        "max_loss_pct": max_loss_pct,
        "risk_level": risk_level
    }


def generate_ascii_chart(prices: List[float],
                         width: int = 40,
                         height: int = 10) -> str:
    if not prices or len(prices) < 2:
        return "Недостаточно данных для построения графика"
    return ascii_chart.plot(prices, {
        "height": height,
        "width": width,
        "format": "{:,.2f}"
    })


def print_report(ticker: str, figi: str, last_close: float, raw_close: float,
                 predictions: List[float], current_time: datetime,
                 indicators: Dict[str, float], regimes: Dict[str, float],
                 analysis: str, accuracy_stats: Dict[str, Any],
                 risk_metrics: Dict[str, Any], df: pd.DataFrame) -> None:
    GREEN = Fore.GREEN + Style.BRIGHT
    RED = Fore.RED + Style.BRIGHT
    YELLOW = Fore.YELLOW + Style.BRIGHT
    CYAN = Fore.CYAN + Style.BRIGHT
    MAGENTA = Fore.MAGENTA + Style.BRIGHT
    WHITE = Fore.WHITE + Style.BRIGHT
    BLUE = Fore.BLUE + Style.BRIGHT
    
    health_index, health_status = calculate_market_health_index(df) if df is not None else (50.0, "Нет данных")
    indicators["market_health"] = health_index
    indicators["health_status"] = health_status
    
    signal_line = analysis.split("\n\nДЕТАЛЬНЫЙ АНАЛИЗ РЫНКА:")[0] if "\n\nДЕТАЛЬНЫЙ АНАЛИЗ РЫНКА:" in analysis else analysis
    detailed_market_analysis = analysis.split("\n\nДЕТАЛЬНЫЙ АНАЛИЗ РЫНКА:")[1] if "\n\nДЕТАЛЬНЫЙ АНАЛИЗ РЫНКА:" in analysis else ""
    
    has_bounce_warning = "⚠️ Возможен отскок" in signal_line
    bounce_info = ""
    if has_bounce_warning:
        bounce_up = "вверх" in signal_line[:signal_line.find("-")]
        bounce_down = "вниз" in signal_line[:signal_line.find("-")]
        
        if "СИГНАЛ к ЛОНГУ" in signal_line or "СИГНАЛ к ШОРТУ" in signal_line:
            bounce_color = YELLOW
            bounce_direction = "↑" if bounce_up else "↓" if bounce_down else "↔"
            bounce_reason = signal_line[signal_line.find("(")+1:signal_line.find(")")]
            
            bounce_info = (
                f"{CYAN}┌─ ПРЕДУПРЕЖДЕНИЕ ОБ ОТСКОКЕ ─────────────────────────────┐{Style.RESET_ALL}\n"
                f"{CYAN}│{Style.RESET_ALL} {bounce_color}⚠️ Обнаружены признаки возможного отскока {bounce_direction}{Style.RESET_ALL}\n"
                f"{CYAN}│{Style.RESET_ALL} Причина: {bounce_reason}\n"
                f"{CYAN}│{Style.RESET_ALL} Рекомендация: {bounce_color}Соблюдайте осторожность при входе в позицию{Style.RESET_ALL}\n"
                f"{CYAN}└────────────────────────────────────────────────────────────┘{Style.RESET_ALL}"
            )
    
    historical_chart: str = ""
    if df is not None and len(df) > 20:
        historical_values = df["close_clean"].iloc[-20:].tolist()
        historical_chart = generate_ascii_chart(historical_values,
                                                width=50,
                                                height=10)
    if regimes:
        regime_items = []
        for k, v in regimes.items():
            percentage = v * 100
            color = GREEN if k == "0" and percentage > 60 else RED if k == "2" and percentage > 60 else YELLOW
            regime_name = "Бычий" if k == "0" else "Медвежий" if k == "2" else "Нейтральный"
            regime_items.append(
                f"{color}{regime_name}: {percentage:.1f}%{Style.RESET_ALL}")
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
    rsi_color = GREEN if indicators[
        "rsi"] < 30 else RED if indicators["rsi"] > 70 else WHITE
    macd_color = GREEN if indicators["macd"] > indicators[
        "macd_signal"] else RED
    bb_position = (raw_close - indicators["bb_lower"]) / (
        indicators["bb_upper"] - indicators["bb_lower"]) * 100
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
    if "СИГНАЛ к ЛОНГУ" in signal_line:
        signal_part = f"{GREEN}СИГНАЛ к ЛОНГУ{Style.RESET_ALL}"
    elif "СИГНАЛ к ШОРТУ" in signal_line:
        signal_part = f"{RED}СИГНАЛ к ШОРТУ{Style.RESET_ALL}"
    else:
        signal_part = f"{YELLOW}РЫНОК НЕЙТРАЛЕН{Style.RESET_ALL}"
    if "(" in signal_line and ")" in signal_line:
        try:
            score_value = float(signal_line.split("score:")[1].split(")")[0])
        except Exception:
            score_value = 0
        strength_color = GREEN if score_value > 3 else RED if score_value < -3 else YELLOW
        strength_bars = "▁▂▃▄▅▆▇█"
        strength_index = min(int((score_value + 5) / 10 * 7), 7)
        strength_visual = strength_bars[strength_index]
        strength_part = f"{strength_color}{score_value:.1f} {strength_visual}{Style.RESET_ALL}"
        reason_part = signal_line.split("-")[1].strip() if "-" in signal_line else ""
    else:
        strength_part = ""
        reason_part = ""
    analysis_formatted = f"{signal_part} ({strength_part}) - {reason_part}"
    header = (
        f"\n{MAGENTA}{'=' * 70}{Style.RESET_ALL}\n"
        f"{MAGENTA}║{Style.RESET_ALL} {WHITE}{ticker.upper()} - Торговый анализ{Style.RESET_ALL}{' ' * 40}{MAGENTA}║{Style.RESET_ALL}\n"
        f"{MAGENTA}{'=' * 70}{Style.RESET_ALL}\n")
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
    health_idx = indicators.get("market_health", 50)
    health_status = indicators.get("health_status", "Нет данных")
    if health_idx >= 70:
        health_color = GREEN
        health_symbol = "●●●●●"
    elif health_idx >= 60:
        health_color = GREEN
        health_symbol = "●●●●○"
    elif health_idx >= 50:
        health_color = YELLOW
        health_symbol = "●●●○○"
    elif health_idx >= 40:
        health_color = YELLOW
        health_symbol = "●●○○○"
    elif health_idx >= 30:
        health_color = RED
        health_symbol = "●○○○○"
    else:
        health_color = RED
        health_symbol = "○○○○○"
    
    health_display = f"{CYAN}Здоровье рынка:{Style.RESET_ALL} {health_color}{health_idx:.1f} {health_symbol} ({health_status}){Style.RESET_ALL}"
    
    indicators_section = (
        f"{CYAN}┌─ ИНДИКАТОРЫ ────────────────────────────────────────────┐{Style.RESET_ALL}\n"
        f"{CYAN}│{Style.RESET_ALL} {' | '.join(indicators_display)}\n"
        f"{CYAN}│{Style.RESET_ALL}\n"
        f"{CYAN}│{Style.RESET_ALL} {health_display}\n"
        f"{CYAN}│{Style.RESET_ALL}\n"
        f"{CYAN}│{Style.RESET_ALL} {detailed_indicators}\n"
        f"{CYAN}└────────────────────────────────────────────────────────────┘{Style.RESET_ALL}"
    )
    
    market_state_section = ""
    if detailed_market_analysis:
        market_points = detailed_market_analysis.strip().split("\n")
        formatted_points = []
        
        for point in market_points:
            point = point.strip()
            if not point:
                continue
                
            point = point.replace("• ", f"• {BLUE}")
            point = point.replace("Бычий", f"{GREEN}Бычий{BLUE}")
            point = point.replace("бычий", f"{GREEN}бычий{BLUE}")
            point = point.replace("Медвежий", f"{RED}Медвежий{BLUE}")
            point = point.replace("медвежий", f"{RED}медвежий{BLUE}")
            point = point.replace("рост", f"{GREEN}рост{BLUE}")
            point = point.replace("снижение", f"{RED}снижение{BLUE}")
            point = point.replace("падение", f"{RED}падение{BLUE}")
            point = point.replace("перепродан", f"{GREEN}перепродан{BLUE}")
            point = point.replace("перекуплен", f"{RED}перекуплен{BLUE}")
            point = point.replace("⚠️", f"{YELLOW}⚠️{BLUE}")
            
            formatted_points.append(f"{point}{Style.RESET_ALL}")
        
        market_state_text = "\n".join(formatted_points)
        
        market_state_section = (
            f"{CYAN}┌─ ДЕТАЛЬНЫЙ АНАЛИЗ РЫНКА ─────────────────────────────────┐{Style.RESET_ALL}\n"
            f"{market_state_text}\n"
            f"{CYAN}└────────────────────────────────────────────────────────────┘{Style.RESET_ALL}"
        )
    
    history_section = ""
    if historical_chart:
        history_section = (
            f"{CYAN}┌─ ИСТОРИЯ ЦЕН (последние 20 точек) ─────────────────────┐{Style.RESET_ALL}\n"
            f"{historical_chart}\n"
            f"{CYAN}└────────────────────────────────────────────────────────────┘{Style.RESET_ALL}"
        )
    
    report = (f"{header}\n"
              f"{timestamp}\n\n"
              f"{prices}\n\n"
              f"{market_analysis}\n\n"
              f"{bounce_info}\n\n" if has_bounce_warning else f"{header}\n"
              f"{timestamp}\n\n"
              f"{prices}\n\n"
              f"{market_analysis}\n\n"
              )
    
    if market_state_section:
        report += f"{market_state_section}\n\n"
    
    report += (f"{indicators_section}\n"
              f"{history_section}\n"
              f"{risk_info}\n\n"
              f"{MAGENTA}{'=' * 70}{Style.RESET_ALL}\n")
    print(report)


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

    def add_predictions(self, timestamp: datetime, predictions: List[float],
                        intervals: List[int], base_price: float) -> None:
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
                if data["actual"] is None and current_time >= data[
                        "target_time"]:
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
                    pred_direction = 1 if data[
                        "prediction"] > base_price else -1 if data[
                            "prediction"] < base_price else 0
                    actual_direction = 1 if data[
                        "actual"] > base_price else -1 if data[
                            "actual"] < base_price else 0
                    direction_correct = pred_direction == actual_direction
                    error = abs(data["prediction"] -
                                data["actual"]) / data["actual"] * 100
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
            "total_count":
            overall_count,
            "avg_error":
            overall_error / overall_count if overall_count > 0 else 0,
            "direction_accuracy":
            overall_correct / overall_count * 100 if overall_count > 0 else 0,
            "by_interval":
            stats
        }


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
                    response = await client.market_data.get_last_prices(
                        figi=[figi])
                    if response and response.last_prices:
                        return float(
                            quotation_to_decimal(
                                response.last_prices[0].price))
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
            need_model_update = (current_time -
                                 last_update_time) > model_update_interval
            if need_model_update or df_cache is None:
                show_loading_animation("Обновление данных")
                df: pd.DataFrame = await fetch_candles(figi, days=90)
                last_update_time = current_time
                if df.empty:
                    await asyncio.sleep(30)
                    continue
                df = preprocess_data(df)
                df = calculate_indicators(df)
                model_cache, scaler_X_cache, scaler_y_cache, feature_cols_cache, df_features = train_model_enhanced(
                    df, lags=3)
                predictions = improved_multivariate_prediction(
                    model_cache,
                    scaler_X_cache,
                    scaler_y_cache,
                    feature_cols_cache,
                    df_features,
                    steps=5,
                    lags=3)
                df_cache = df
                df_features_cache = df_features
                predictions_cache = predictions
                regimes_cache = detect_regime(df)
            last_close: float = df_cache["close_clean"].iloc[-1]
            raw_close: float = current_market_price if current_market_price else df_cache[
                "last_raw_close"]
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
            analysis: str = analyze_market(indicators, raw_close,
                                           predictions_cache, df_cache)
            risk_metrics: Dict[str, Any] = calculate_risk_metrics(
                df_cache, raw_close, predictions_cache)
            clear_terminal()
            print_report(ticker, figi, last_close, raw_close,
                         predictions_cache, current_time, indicators,
                         regimes_cache, analysis, {}, risk_metrics, df_cache)
            await asyncio.sleep(data_update_interval.total_seconds())
    except KeyboardInterrupt:
        logging.info("Завершение работы...")
    except Exception as e:
        logging.error(f"Ошибка основного цикла: {e}")


if __name__ == "__main__":
    asyncio.run(main())
