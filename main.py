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

TOKEN = os.environ.get('TINKOFF_TOKEN')
if not TOKEN:
    print("ВНИМАНИЕ: токен Tinkoff API не найден в переменных окружения!")
    print(
        "Рекомендуется настроить переменную окружения TINKOFF_TOKEN для безопасной работы"
    )
    import getpass
    TOKEN = getpass.getpass("Введите ваш Tinkoff API токен: ")


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
    hourly_candles = []
    minute_candles = []

    async with AsyncClient(TOKEN) as client:
        async for candle in client.get_all_candles(
                figi=figi,
                from_=now() - timedelta(days=days),
                to=now() - timedelta(minutes=60),
                interval=CandleInterval.CANDLE_INTERVAL_HOUR,
        ):
            hourly_candles.append({
                'time':
                candle.time,
                'open':
                float(quotation_to_decimal(candle.open)),
                'high':
                float(quotation_to_decimal(candle.high)),
                'low':
                float(quotation_to_decimal(candle.low)),
                'close':
                float(quotation_to_decimal(candle.close)),
                'volume':
                candle.volume,
                'timeframe':
                'hour'
            })

        async for candle in client.get_all_candles(
                figi=figi,
                from_=now() - timedelta(minutes=60),
                to=now(),
                interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
        ):
            minute_candles.append({
                'time':
                candle.time,
                'open':
                float(quotation_to_decimal(candle.open)),
                'high':
                float(quotation_to_decimal(candle.high)),
                'low':
                float(quotation_to_decimal(candle.low)),
                'close':
                float(quotation_to_decimal(candle.close)),
                'volume':
                candle.volume,
                'timeframe':
                'minute'
            })

    df_hour = pd.DataFrame(hourly_candles)
    df_minute = pd.DataFrame(minute_candles)

    if not df_hour.empty and not df_minute.empty:
        if len(df_minute) >= 30:
            df_minute['time'] = pd.to_datetime(df_minute['time'])
            df_minute.set_index('time', inplace=True)

            minute_to_hour = df_minute.resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()

            minute_to_hour['timeframe'] = 'hour_from_minute'

            # Объединяем с часовыми данными
            all_candles = pd.concat([df_hour, minute_to_hour])
        else:
            # Просто объединяем данные
            all_candles = pd.concat([df_hour, df_minute])
    elif not df_hour.empty:
        all_candles = df_hour
    elif not df_minute.empty:
        all_candles = df_minute
    else:
        return pd.DataFrame()

    if not all_candles.empty:
        all_candles['time'] = pd.to_datetime(all_candles['time'])
        all_candles.sort_values('time', inplace=True)
        all_candles.set_index('time', inplace=True)

        # Сохраняем информацию о последней рыночной цене
        all_candles["last_raw_close"] = all_candles["close"].iloc[
            -1] if not all_candles.empty else None

    return all_candles


def preprocess_data(df: pd.DataFrame):
    if df.empty:
        return df

    df["last_raw_close"] = df["close"].iloc[-1] if not df.empty else None

    df = df.sort_index()

    hourly_data = df[df['timeframe'] == 'hour'].copy()
    if not hourly_data.empty:
        expected_hours = pd.date_range(start=hourly_data.index.min(),
                                       end=hourly_data.index.max(),
                                       freq='h')

        missing_hours = expected_hours.difference(hourly_data.index)
        if len(missing_hours) > 0:
            print(
                f"Обнаружено {len(missing_hours)} пропущенных часовых интервалов (возможно нерыночные часы)"
            )

            missing_percent = len(missing_hours) / len(expected_hours) * 100
            if missing_percent < 30:
                print(
                    f"Заполняем {len(missing_hours)} пропущенных интервалов (возможно аномалии)"
                )
                for missing_hour in missing_hours:
                    closest_before = hourly_data[
                        hourly_data.index <
                        missing_hour].iloc[-1] if not hourly_data[
                            hourly_data.index < missing_hour].empty else None
                    closest_after = hourly_data[
                        hourly_data.index >
                        missing_hour].iloc[0] if not hourly_data[
                            hourly_data.index > missing_hour].empty else None

                    if closest_before is not None and closest_after is not None:
                        time_delta = (closest_after.name -
                                      closest_before.name).total_seconds()
                        time_ratio = (missing_hour - closest_before.name
                                      ).total_seconds() / time_delta

                        new_row = closest_before.copy()
                        for col in ['open', 'high', 'low', 'close']:
                            new_row[col] = closest_before[col] + time_ratio * (
                                closest_after[col] - closest_before[col])

                        new_row['volume'] = int((closest_before['volume'] +
                                                 closest_after['volume']) / 2)
                        new_row.name = missing_hour
                        df.loc[missing_hour] = new_row

    min_periods = min(2, len(df))
    df["close_smooth"] = df["close"].rolling(window=2,
                                             min_periods=min_periods).mean()

    window_size = min(10, len(df))
    if window_size < 4:
        print(
            "Предупреждение: Недостаточно данных для надежной обработки выбросов"
        )
        df["close_clean"] = df[
            "close_smooth"]  # Без обработки выбросов при малом количестве данных
    else:
        roll_med = df["close_smooth"].rolling(
            window=window_size, min_periods=min_periods).median()
        roll_std = df["close_smooth"].rolling(
            window=window_size, min_periods=min_periods).std().fillna(0)

        # Определяем выбросы с адаптивным порогом
        diff = np.abs(df["close_smooth"] - roll_med)
        threshold_factor = 3.0 if len(df) > 30 else 4.0
        threshold = threshold_factor * roll_std

        df["close_clean"] = np.where(diff > threshold, roll_med,
                                     df["close_smooth"])

    # Последний час всегда оставляем как есть, поскольку это текущие рыночные данные
    last_hour_mask = df.index >= (df.index.max() - pd.Timedelta(hours=1))
    if any(last_hour_mask):
        df.loc[last_hour_mask, "close_clean"] = df.loc[last_hour_mask, "close"]

    # Заполняем возможные NaN после всех операций
    df["close_clean"] = df["close_clean"].fillna(df["close"])

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
            df[lag_col] = df[lag_col].bfill().ffill()

    for col in [
            "sma20", "ema20", "rsi", "macd", "macd_signal", "bb_upper",
            "bb_lower", "stoch", "ema50", "vol_ma20"
    ]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].ffill().bfill().fillna(0)

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
    # Импортируем StandardScaler в начале функции
    from sklearn.preprocessing import StandardScaler

    if len(df) < 30:
        print(
            "Предупреждение: Недостаточно данных для надежного обучения модели."
        )
        # Возвращаем простую наивную модель вместо сложной
        from sklearn.dummy import DummyRegressor
        dummy_model = DummyRegressor(strategy="mean")
        dummy_model.fit(np.array([[0]]), np.array([df["close_clean"].mean()]))

        # Создаем пустые масштабировщики
        scaler_X = StandardScaler()
        scaler_X.fit(np.array([[0]]))
        scaler_y = StandardScaler()
        scaler_y.fit(np.array([[0]]))

        return dummy_model, scaler_X, scaler_y, ["dummy_feature"], df

    df_feat, feature_cols = create_features(df.copy(), lags)

    # Проверка стационарности временного ряда
    from statsmodels.tsa.stattools import adfuller

    # Проверяем ряд цен на стационарность
    adf_result = adfuller(df_feat['close_clean'].dropna())

    if adf_result[1] > 0.05:
        print(
            f"Предупреждение: Ряд цен нестационарен (p-value: {adf_result[1]:.4f}). Рассмотрите использование разностей."
        )
        # Создаем признак разности цен для улучшения стационарности
        df_feat['price_diff'] = df_feat['close_clean'].diff().fillna(0)
        if 'price_diff' not in feature_cols:
            feature_cols.append('price_diff')
    else:
        print(f"Временной ряд стационарен (p-value: {adf_result[1]:.4f})")

    # Правильное разделение на обучающую и тестовую выборки для временных рядов
    # Используем строго последовательный подход - последние 20% данных для тестирования
    test_size = int(len(df_feat) * 0.2)
    train_df = df_feat.iloc[:-test_size] if test_size > 0 else df_feat.copy()
    test_df = df_feat.iloc[-test_size:] if test_size > 0 else pd.DataFrame()

    # Создаем кросс-валидацию внутри тренировочного набора
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(
        n_splits=3)  # Уменьшено для более стабильных результатов

    X_train = train_df[feature_cols].values
    y_train = train_df['close_clean'].values.reshape(-1, 1)

    # Удаление выбросов только из тренировочного набора
    lower_bounds = {}
    upper_bounds = {}
    for i in range(X_train.shape[1]):
        col_data = X_train[:, i]
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        lower_bounds[i] = q1 - 1.5 * iqr
        upper_bounds[i] = q3 + 1.5 * iqr
        X_train[:, i] = np.clip(col_data, lower_bounds[i], upper_bounds[i])

    # Масштабируем данные после удаления выбросов
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train).ravel()

    # Оцениваем различные типы моделей для выбора наилучшей
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Тестируем несколько типов моделей, адаптируем сложность к размеру данных
    if len(X_train) < 50:
        # Для маленьких выборок используем более простые модели
        models = {
            'ridge': Ridge(),
            'elasticnet': ElasticNet(random_state=42, alpha=1.0)
        }
    else:
        # Для больших выборок можем позволить более сложные модели
        models = {
            'ridge':
            Ridge(),
            'elasticnet':
            ElasticNet(random_state=42),
            'gbr':
            GradientBoostingRegressor(n_estimators=50,
                                      max_depth=3,
                                      random_state=42)
        }

    best_model_name = None
    best_model = None
    best_params = None
    best_score = -float('inf')

    # Простая оптимизация гиперпараметров
    for model_name, model in models.items():
        if model_name == 'ridge':
            param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
        elif model_name == 'elasticnet':
            param_grid = {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            }
        elif model_name == 'gbr':
            param_grid = {'learning_rate': [0.01, 0.1]}

        best_param_score = -float('inf')
        best_model_params = None

        # Перебор параметров
        import itertools
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        for params in param_combinations:
            param_dict = {
                name: value
                for name, value in zip(param_names, params)
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
                r2 = r2_score(y_cv_val, pred_val)
                cv_scores.append(r2)

            mean_score = np.mean(cv_scores)
            if mean_score > best_param_score:
                best_param_score = mean_score
                best_model_params = param_dict

        # Если эта модель лучше предыдущих, сохраняем её
        if best_param_score > best_score:
            best_score = best_param_score
            best_model_name = model_name
            best_params = best_model_params

    # Создаем и обучаем лучшую модель
    if best_model_name == 'ridge':
        best_model = Ridge(**best_params)
    elif best_model_name == 'elasticnet':
        best_model = ElasticNet(**best_params, random_state=42)
    elif best_model_name == 'gbr':
        best_model = GradientBoostingRegressor(**best_params,
                                               n_estimators=50,
                                               max_depth=3,
                                               random_state=42)

    print(
        f"Выбрана модель: {best_model_name} с параметрами: {best_params}, средний R²: {best_score:.4f}"
    )

    # Обучаем окончательную модель на всех тренировочных данных
    best_model.fit(X_train_scaled, y_train_scaled)

    # Проверка на тестовых данных
    if len(test_df) > 0:
        X_test = test_df[feature_cols].values
        y_test = test_df['close_clean'].values

        # Применяем те же ограничения к тестовым данным, чтобы избежать утечки
        for i in range(X_test.shape[1]):
            if i in lower_bounds:  # Проверка на случай изменения признаков
                X_test[:, i] = np.clip(X_test[:, i], lower_bounds[i],
                                       upper_bounds[i])

        X_test_scaled = scaler_X.transform(X_test)

        # Получаем предсказания
        pred_test_scaled = best_model.predict(X_test_scaled)
        pred_test = scaler_y.inverse_transform(pred_test_scaled.reshape(
            -1, 1)).ravel()

        # Оцениваем качество модели
        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        r2 = r2_score(y_test, pred_test)

        # Вычисляем точность направления
        direction_pred = np.sign(np.diff(np.append([y_test[0]], pred_test)))
        direction_actual = np.sign(np.diff(y_test))
        direction_accuracy = np.mean(direction_pred[:-1] == direction_actual
                                     ) * 100  # последний элемент отбрасываем

        print(
            f"Тестирование модели: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}"
        )
        print(f"Точность направления движения: {direction_accuracy:.2f}%")

        # Наивная модель для сравнения (предыдущее значение)
        naive_pred = np.roll(y_test, 1)
        naive_pred[0] = y_test[0]
        naive_mae = mean_absolute_error(y_test[1:], naive_pred[1:])
        naive_rmse = np.sqrt(mean_squared_error(y_test[1:], naive_pred[1:]))

        print(f"Наивная модель: MAE={naive_mae:.4f}, RMSE={naive_rmse:.4f}")

        # Сравнение с наивной моделью
        if rmse >= naive_rmse:
            print(
                "Предупреждение: Модель не превосходит наивный прогноз. Требуется улучшение."
            )
        elif r2 < 0.3:
            print(
                "Предупреждение: Низкое качество прогнозов. Рекомендуется пересмотреть признаки или увеличить объем данных."
            )

    return best_model, scaler_X, scaler_y, feature_cols, df_feat


def predict_multiple_steps_enhanced(df: pd.DataFrame,
                                    steps: int = 5,
                                    lags: int = 3):
    if df.empty:
        print("Ошибка: нет данных для прогнозирования")
        return [df["close"].iloc[-1] if not df.empty else 0] * steps

    model, scaler_X, scaler_y, feature_cols, df_feat = train_model_enhanced(
        df, lags)

    # Для итеративного прогнозирования
    time_intervals = [1, 5, 15, 30]
    predictions = []

    # Копируем последние данные для обновления
    last_data = df_feat.tail(1).copy()
    current_close = df['close_clean'].iloc[-1]

    # Получаем все необходимые признаки для начальной точки прогноза
    for interval in time_intervals:
        if interval == 1:
            # Получаем прогноз для первого шага
            current_features = []
            for col in feature_cols:
                value = last_data[col].iloc[-1]
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
            pred_1min = scaler_y.inverse_transform(pred_scaled.reshape(-1,
                                                                       1))[0,
                                                                           0]

            # Собираем технические индикаторы для корректировки
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

            bb_position = (df['close_clean'].iloc[-1] - df['bb_lower'].iloc[-1]
                           ) / (df['bb_upper'].iloc[-1] -
                                df['bb_lower'].iloc[-1])
            if bb_position < 0.2:
                tech_indicators_bullish += 1
            elif bb_position > 0.8:
                tech_indicators_bearish += 1

            if df['close_clean'].iloc[-1] > df['ema20'].iloc[-1] and df[
                    'ema20'].iloc[-1] > df['ema50'].iloc[-1]:
                tech_indicators_bullish += 1
            elif df['close_clean'].iloc[-1] < df['ema20'].iloc[-1] and df[
                    'ema20'].iloc[-1] < df['ema50'].iloc[-1]:
                tech_indicators_bearish += 1

            indicator_consensus = tech_indicators_bullish - tech_indicators_bearish

            # Небольшая корректировка на основе индикаторов
            adjustment_factor = 0.001 * indicator_consensus * current_close
            predictions.append(pred_1min + adjustment_factor)

            # Сохраняем прогноз для следующих шагов
            predicted_value = pred_1min
        else:
            # Для каждого следующего интервала создаем новое состояние данных
            # с учетом предыдущих прогнозов
            prev_predictions = predictions.copy()

            # Выполняем несколько прогнозов с интервалом между ними
            # для более точного предсказания на больший период
            step_size = 1
            steps_num = interval // step_size

            # Инициализируем с последнего известного состояния
            temp_df = last_data.copy()
            last_predicted = current_close
            next_predicted = None

            for step in range(steps_num):
                # Обновляем лаги на основе предыдущих прогнозов
                for i in range(1, lags + 1):
                    lag_idx = lags - i
                    if lag_idx < len(prev_predictions):
                        temp_df[f'lag_{i}'] = prev_predictions[lag_idx]
                    else:
                        temp_df[f'lag_{i}'] = last_predicted

                # Обновляем текущее значение для технических индикаторов
                # В реальном сценарии нужно было бы пересчитать все технические индикаторы,
                # здесь мы делаем упрощенную версию
                if step > 0:
                    # Правильно пересчитываем индикаторы для более точного прогноза
                    # Создаем копию последних данных с новым предсказанным значением
                    last_prices = list(df['close_clean'].tail(
                        50).values)  # Берем последние 50 значений для расчета

                    # Проверяем, что next_predicted не None
                    if next_predicted is not None:
                        last_prices.append(
                            next_predicted
                        )  # Добавляем новое предсказанное значение

                    # Убеждаемся, что у нас достаточно данных
                    if len(last_prices) >= 20:
                        # Пересчитываем основные индикаторы
                        ema20 = np.average(
                            last_prices[-20:],
                            weights=[1 + (i / 20) for i in range(20)])
                        sma20 = np.mean(last_prices[-20:])

                        # Проверяем достаточно ли данных для RSI
                        if len(last_prices) >= 15:
                            # Простое обновление RSI
                            deltas = np.diff(last_prices[-15:])
                            gains = np.where(deltas > 0, deltas, 0)
                            losses = np.where(deltas < 0, -deltas, 0)
                            avg_gain = np.mean(gains)
                            avg_loss = np.mean(losses) if np.mean(
                                losses) != 0 else 0.0001
                            rs = avg_gain / avg_loss
                            rsi = 100 - (100 / (1 + rs))
                        else:
                            # Если недостаточно данных, используем текущее значение RSI
                            rsi = temp_df['rsi'].iloc[-1]
                    else:
                        # Если недостаточно данных, используем текущие значения
                        ema20 = temp_df['ema20'].iloc[-1]
                        sma20 = temp_df['sma20'].iloc[-1]
                        rsi = temp_df['rsi'].iloc[-1]

                    update_features = []
                    for col in feature_cols:
                        if col.startswith('lag_'):
                            value = temp_df[col].iloc[-1]
                        elif col == 'ema20':
                            value = ema20
                        elif col == 'sma20':
                            value = sma20
                        elif col == 'rsi':
                            value = rsi
                        elif col == 'price_diff' and 'price_diff' in feature_cols:
                            # Проверяем, что оба значения не None
                            if next_predicted is not None and last_predicted is not None:
                                value = next_predicted - last_predicted
                            else:
                                value = 0  # Безопасное значение по умолчанию
                        else:
                            # Остальные индикаторы пока оставляем прежними из-за сложности расчета
                            value = temp_df[col].iloc[-1]

                        # Проверяем, что значение не None
                        if value is None:
                            value = 0  # Безопасное значение по умолчанию

                        update_features.append(value)

                    update_vector = np.array(update_features).reshape(1, -1)
                    if np.isnan(update_vector).any():
                        update_vector = np.nan_to_num(update_vector, nan=0.0)

                    # Получаем прогноз для следующего шага
                    update_scaled = scaler_X.transform(update_vector)
                    new_pred_scaled = model.predict(update_scaled)
                    next_predicted = scaler_y.inverse_transform(
                        new_pred_scaled.reshape(-1, 1))[0, 0]

                    # Сохраняем для следующего шага
                    last_predicted = next_predicted
                    prev_predictions.append(next_predicted)

            # Финальный прогноз для данного интервала
            if next_predicted is not None:
                predictions.append(next_predicted)
            else:
                # Если по какой-то причине не получилось сделать итеративный прогноз,
                # используем упрощенную версию как запасной вариант
                last_n_values = df['close_clean'].tail(60).values
                last_values_diff = np.diff(last_n_values)
                mean_change = np.mean(last_values_diff)
                std_change = np.std(last_values_diff)

                base_forecast = predictions[0] + (mean_change * interval)
                predictions.append(base_forecast)

    return predictions


def detect_regime(df: pd.DataFrame):
    try:
        if len(df) < 30:  # Проверка наличия достаточного количества данных
            print("Недостаточно данных для определения режима рынка")
            return {}

        mr = MarkovRegression(df["close_clean"],
                              k_regimes=2,
                              trend='c',
                              switching_variance=True)
        res = mr.fit(disp=False)
        regimes = res.smoothed_marginal_probabilities.iloc[-1].to_dict()
        return regimes
    except Exception as e:
        print(f"Ошибка при определении режима рынка: {e}")
        return {}


def analyze_market(ind, current_price, predictions, df=None):
    score_bull = 0
    score_bear = 0
    reason_bull = []
    reason_bear = []

    # Проверка наличия экстремальных рыночных условий
    try:
        # Проверка повышенной волатильности
        if df is not None and len(df) > 20:
            recent_volatility = df['close_clean'].pct_change().rolling(
                window=5).std().iloc[-1]
            historical_volatility = df['close_clean'].pct_change().rolling(
                window=20).std().iloc[-1]

            if recent_volatility > historical_volatility * 2:
                score_bear += 2
                reason_bear.append(
                    f"Экстремальная волатильность (x{recent_volatility/historical_volatility:.1f})"
                )

        # Проверка на гэпы в цене (если данные обновлялись после закрытия и открытия рынка)
        if df is not None and len(df) > 1:
            last_close = df['close_clean'].iloc[-2]
            current_open = df['open'].iloc[-1]
            if abs(current_open -
                   last_close) / last_close > 0.02:  # гэп более 2%
                gap_direction = "верх" if current_open > last_close else "вниз"
                gap_size = abs(current_open - last_close) / last_close * 100
                if gap_direction == "верх":
                    score_bull += 2
                    reason_bull.append(f"Гэп вверх {gap_size:.1f}%")
                else:
                    score_bear += 2
                    reason_bear.append(f"Гэп вниз {gap_size:.1f}%")
    except Exception as e:
        print(f"Предупреждение: ошибка при анализе рыночных условий: {e}")

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

    bb_percent = (current_price - ind["bb_lower"]) / (ind["bb_upper"] -
                                                      ind["bb_lower"]) * 100
    if bb_percent < 20:
        score_bull += 2
        reason_bull.append(
            f"Цена близка к нижней границе BB ({bb_percent:.1f}%)")
    elif bb_percent > 80:
        score_bear += 2
        reason_bear.append(
            f"Цена близка к верхней границе BB ({bb_percent:.1f}%)")

    if current_price > ind["ema50"] and current_price > ind["ema20"] and ind[
            "ema20"] > ind["ema50"]:
        score_bull += 2
        reason_bull.append("Восходящий тренд по EMA (цена > EMA20 > EMA50)")
    elif current_price < ind["ema50"] and current_price < ind["ema20"] and ind[
            "ema20"] < ind["ema50"]:
        score_bear += 2
        reason_bear.append("Нисходящий тренд по EMA (цена < EMA20 < EMA50)")

    if ind["stoch"] < 20:
        score_bull += 1
        reason_bull.append("Стохастик перепродан (<20)")
    elif ind["stoch"] > 80:
        score_bear += 1
        reason_bear.append("Стохастик перекуплен (>80)")

    pred_trend = sum(1 if p > current_price else -1 if p < current_price else 0
                     for p in predictions)
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
            prev_close = df['close_clean'].iloc[-2] if len(
                df) > 2 else last_close
            price_direction = 1 if last_close > prev_close else -1

            if price_direction > 0:
                score_bull += 3
                reason_bull.append(
                    f"Крупная сделка на покупку (объем x{volume_ratio:.1f})")
            else:
                score_bear += 3
                reason_bear.append(
                    f"Крупная сделка на продажу (объем x{volume_ratio:.1f})")
        elif volume_ratio > 2.0:
            last_close = df['close_clean'].iloc[-1]
            prev_close = df['close_clean'].iloc[-2] if len(
                df) > 2 else last_close
            price_direction = 1 if last_close > prev_close else -1

            if price_direction > 0:
                score_bull += 2
                reason_bull.append(
                    f"Повышенные покупки (объем x{volume_ratio:.1f})")
            else:
                score_bear += 2
                reason_bear.append(
                    f"Повышенные продажи (объем x{volume_ratio:.1f})")

        if len(df) > 10:
            volume_trend = df['volume'].iloc[-5:].mean(
            ) - df['volume'].iloc[-10:-5].mean()
            if volume_trend > 0:
                price_trend = df['close_clean'].iloc[-5:].mean(
                ) - df['close_clean'].iloc[-10:-5].mean()
                if price_trend > 0:
                    score_bull += 1
                    reason_bull.append("Растущий объем при росте цены")
                else:
                    score_bear += 1
                    reason_bear.append("Растущий объем при падении цены")
            else:
                price_trend = df['close_clean'].iloc[-5:].mean(
                ) - df['close_clean'].iloc[-10:-5].mean()
                if abs(price_trend) > 0 and volume_trend < -avg_volume * 0.2:
                    if price_trend > 0:
                        score_bear += 1
                        reason_bear.append(
                            "Падающий объем при росте цены (слабость)")
                    else:
                        score_bull += 1
                        reason_bull.append(
                            "Падающий объем при снижении цены (слабость)")

        if len(df) > 20:
            large_buys = 0
            large_sells = 0
            for i in range(-1, -11, -1):
                if i < -len(df):
                    break
                vol = df['volume'].iloc[i]
                if vol > avg_volume * 1.5:
                    price_chg = df['close_clean'].iloc[i] - df[
                        'close_clean'].iloc[i - 1] if i > -len(df) else 0
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

    bull_strength = score_bull / (score_bull + score_bear) * 100 if (
        score_bull + score_bear) > 0 else 50
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

    def get_accuracy_stats(self):
        if not self.predictions:
            return {}

        completed_predictions = []
        for ts, intervals in self.predictions.items():
            for interval, data in intervals.items():
                if data["actual"] is not None:
                    error = abs(data["prediction"] -
                                data["actual"]) / data["actual"] * 100
                    pred_change = data["prediction"] - data["actual"]
                    actual_price_at_prediction_time = data["actual"]

                    pred_direction = 1 if data[
                        "prediction"] > actual_price_at_prediction_time else -1 if data[
                            "prediction"] < actual_price_at_prediction_time else 0

                    actual_direction = 1 if data[
                        "actual"] > actual_price_at_prediction_time else -1 if data[
                            "actual"] < actual_price_at_prediction_time else 0

                    direction_correct = pred_direction == actual_direction

                    days_old = (datetime.now() - data["timestamp"]
                                ).total_seconds() / (24 * 3600)
                    time_weight = max(0.5, 1.0 - (days_old / 3.0))

                    completed_predictions.append({
                        "interval":
                        interval,
                        "error":
                        error,
                        "direction_correct":
                        direction_correct,
                        "time_diff": (data["target_time"] -
                                      data["timestamp"]).total_seconds() / 60,
                        "time_weight":
                        time_weight
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
            stats_by_interval[interval][
                "correct_directions"] += 1 if pred["direction_correct"] else 0

            weight = pred["time_weight"]
            stats_by_interval[interval]["weighted_count"] += weight
            stats_by_interval[interval][
                "weighted_error_sum"] += pred["error"] * weight
            stats_by_interval[interval][
                "weighted_correct_directions"] += weight if pred[
                    "direction_correct"] else 0

        for interval, stats in stats_by_interval.items():
            if stats["count"] > 0:
                stats["avg_error"] = stats["error_sum"] / stats["count"]
                stats["direction_accuracy"] = stats[
                    "correct_directions"] / stats["count"] * 100

                if stats["weighted_count"] > 0:
                    stats["weighted_avg_error"] = stats[
                        "weighted_error_sum"] / stats["weighted_count"]
                    stats["weighted_direction_accuracy"] = stats[
                        "weighted_correct_directions"] / stats[
                            "weighted_count"] * 100
                else:
                    stats["weighted_avg_error"] = 0
                    stats["weighted_direction_accuracy"] = 0
            else:
                stats["avg_error"] = 0
                stats["direction_accuracy"] = 0
                stats["weighted_avg_error"] = 0
                stats["weighted_direction_accuracy"] = 0

        total_count = sum(stats["count"]
                          for stats in stats_by_interval.values())
        total_error = sum(stats["error_sum"]
                          for stats in stats_by_interval.values())
        total_correct = sum(stats["correct_directions"]
                            for stats in stats_by_interval.values())

        total_weighted_count = sum(stats["weighted_count"]
                                   for stats in stats_by_interval.values())
        total_weighted_error = sum(stats["weighted_error_sum"]
                                   for stats in stats_by_interval.values())
        total_weighted_correct = sum(stats["weighted_correct_directions"]
                                     for stats in stats_by_interval.values())

        overall_stats = {
            "total_count":
            total_count,
            "avg_error":
            total_error / total_count if total_count > 0 else 0,
            "direction_accuracy":
            total_correct / total_count * 100 if total_count > 0 else 0,
            "weighted_avg_error":
            total_weighted_error /
            total_weighted_count if total_weighted_count > 0 else 0,
            "weighted_direction_accuracy":
            total_weighted_correct / total_weighted_count *
            100 if total_weighted_count > 0 else 0,
            "by_interval":
            stats_by_interval
        }

        return overall_stats


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_loading_animation(message="Загрузка данных"):
    """Показывает анимацию загрузки с использованием tqdm"""
    from tqdm import tqdm
    import time
    
    clear_terminal()
    print(f"\033[1;36m{message}...\033[0m")
    for _ in tqdm(range(100), desc="Прогресс", ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        time.sleep(0.01)  # Имитация работы


def calculate_risk_metrics(df, current_price, predictions):
    # Расчет простого Value at Risk (VaR) на основе исторической волатильности
    if df is None or len(df) < 30:
        return {"var_95": 0, "max_loss_pct": 0, "risk_level": "Неизвестно"}

    returns = df['close_clean'].pct_change().dropna()
    if len(returns) < 20:
        return {
            "var_95": 0,
            "max_loss_pct": 0,
            "risk_level": "Недостаточно данных"
        }

    # Расчет 95% VaR (потенциальная потеря с 95% уверенностью)
    var_95 = abs(returns.quantile(0.05) * current_price)
    var_95_pct = abs(returns.quantile(0.05) * 100)

    # Максимальная историческая просадка за последние 30 дней
    rolling_max = df['close_clean'].rolling(window=30).max()
    daily_drawdown = df['close_clean'] / rolling_max - 1.0
    max_loss_pct = abs(daily_drawdown.min() * 100)

    # Определение уровня риска
    if var_95_pct > 5:
        risk_level = "Высокий"
    elif var_95_pct > 2:
        risk_level = "Средний"
    else:
        risk_level = "Низкий"

    return {
        "var_95": var_95,
        "var_95_pct": var_95_pct,
        "max_loss_pct": max_loss_pct,
        "risk_level": risk_level
    }


from colorama import init, Fore, Back, Style
from termcolor import colored
import asciichartpy as ascii_chart
import numpy as np

# Initialize colorama for Windows support
init()

def generate_ascii_chart(prices, width=40, height=10):
    if not prices or len(prices) < 2:
        return "Недостаточно данных для построения графика"
    
    return ascii_chart.plot(prices, {'height': height, 'width': width, 'format': '{:,.2f}'})

def print_report(ticker,
                 figi,
                 last_close,
                 raw_close,
                 predictions,
                 current_time,
                 indicators,
                 regimes,
                 analysis,
                 accuracy_stats=None,
                 risk_metrics=None,
                 df=None):
    predicted_price = predictions[0]
    trend_symbol = "↑" if predicted_price > last_close else "↓" if predicted_price < last_close else "→"
    
    # Цветовое кодирование
    GREEN = Fore.GREEN + Style.BRIGHT
    RED = Fore.RED + Style.BRIGHT
    YELLOW = Fore.YELLOW + Style.BRIGHT
    BLUE = Fore.BLUE + Style.BRIGHT
    CYAN = Fore.CYAN + Style.BRIGHT
    MAGENTA = Fore.MAGENTA + Style.BRIGHT
    WHITE = Fore.WHITE + Style.BRIGHT
    RESET = Style.RESET_ALL
    
    # Определяем цвета для трендов
    color_pred = GREEN if predicted_price > last_close else RED if predicted_price < last_close else WHITE

    time_intervals = [1, 5, 15, 30]
    
    # Генерируем график прогнозов
    chart_values = [raw_close] + predictions
    prediction_chart = generate_ascii_chart(chart_values, width=50, height=10)
    
    # Генерируем историю если достаточно данных
    historical_chart = ""
    if df is not None and len(df) > 20:
        historical_values = df['close_clean'].iloc[-20:].tolist()
        historical_chart = generate_ascii_chart(historical_values, width=50, height=10)
    
    # Форматирование прогнозов
    interval_preds = []
    for i, pred in enumerate(predictions):
        if i < len(time_intervals):
            interval = time_intervals[i]
            trend = "↑" if pred > raw_close else "↓" if pred < raw_close else "→"
            color = GREEN if pred > raw_close else RED if pred < raw_close else WHITE
            diff_percent = ((pred / raw_close) - 1) * 100
            interval_preds.append(
                f"  {CYAN}■ {interval:2d} мин:{RESET} {color}{pred:.2f} {trend} ({diff_percent:+.2f}%){RESET}"
            )

    multi_preds_str = "\n".join(interval_preds)

    # Режимы рынка
    if regimes:
        regime_items = []
        for k, v in regimes.items():
            percentage = v*100
            color = GREEN if k == "0" and percentage > 60 else RED if k == "1" and percentage > 60 else YELLOW
            regime_items.append(f"{color}Режим {k}: {percentage:.1f}%{RESET}")
        regime_str = " | ".join(regime_items)
    else:
        regime_str = f"{YELLOW}Нет данных о режимах{RESET}"

    # Объемы торгов
    vol_info = ""
    if 'volume' in indicators and 'vol_ma20' in indicators:
        last_vol = indicators.get('volume', 0)
        avg_vol = indicators['vol_ma20']
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

        vol_info = f"{CYAN}Объем:{RESET} {vol_color}{last_vol} ({vol_ratio:.2f}x) {vol_symbol}{RESET}"

    # Форматирование индикаторов
    rsi_color = GREEN if indicators['rsi'] < 30 else RED if indicators['rsi'] > 70 else WHITE
    macd_color = GREEN if indicators['macd'] > indicators['macd_signal'] else RED
    bb_position = (raw_close - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower']) * 100
    bb_color = RED if bb_position > 80 else GREEN if bb_position < 20 else WHITE
    
    # Создаем визуальное представление для RSI
    rsi_bars = "▁▂▃▄▅▆▇█"
    rsi_index = min(int(indicators['rsi'] / 12.5), 7)
    rsi_visual = rsi_bars[rsi_index]
    
    # Упрощенная визуализация индикаторов
    indicators_display = [
        f"{CYAN}RSI:{RESET} {rsi_color}{indicators['rsi']:.1f} {rsi_visual}{RESET}",
        f"{CYAN}MACD:{RESET} {macd_color}{indicators['macd']:.2f}/{indicators['macd_signal']:.2f}{RESET}",
        f"{CYAN}BB:{RESET} {bb_color}{bb_position:.1f}%{RESET}",
        f"{CYAN}EMA:{RESET} {indicators['ema20']:.2f}/{indicators['ema50']:.2f}",
        vol_info
    ]
    
    # Более подробные индикаторы можно скрыть в дополнительном разделе
    detailed_indicators = (
        f"  EMA20: {indicators['ema20']:.2f} | SMA20: {indicators['sma20']:.2f} | RSI: {indicators['rsi']:.2f}\n"
        f"  MACD: {indicators['macd']:.2f} | MACD Signal: {indicators['macd_signal']:.2f}\n"
        f"  BB: Upper={indicators['bb_upper']:.2f} | Lower={indicators['bb_lower']:.2f}\n"
        f"  Stoch: {indicators['stoch']:.2f} | EMA50: {indicators['ema50']:.2f} | Vol_MA20: {indicators['vol_ma20']:.2f}"
    )

    # Форматирование метрик точности
    accuracy_info = ""
    if accuracy_stats and "total_count" in accuracy_stats and accuracy_stats["total_count"] > 0:
        overall_accuracy = accuracy_stats["direction_accuracy"]
        
        # Визуальный индикатор точности
        accuracy_bars = "▁▂▃▄▅▆▇█"
        accuracy_index = min(int(overall_accuracy / 12.5), 7)
        accuracy_visual = accuracy_bars[accuracy_index]
        
        if overall_accuracy >= 75:
            accuracy_color = GREEN
            accuracy_rating = "ВЫСОКАЯ"
        elif overall_accuracy >= 60:
            accuracy_color = YELLOW
            accuracy_rating = "СРЕДНЯЯ"
        else:
            accuracy_color = RED
            accuracy_rating = "НИЗКАЯ"
            
        accuracy_info = (
            f"\n{CYAN}┌─ ТОЧНОСТЬ МОДЕЛИ ─────────────────────────────────────────┐{RESET}\n"
            f"{CYAN}│{RESET} Общая: {accuracy_color}{overall_accuracy:.1f}% {accuracy_visual} {accuracy_rating}{RESET}\n"
            f"{CYAN}│{RESET} Ошибка: {accuracy_stats['avg_error']:.2f}%\n"
            f"{CYAN}│{RESET} Всего замеров: {accuracy_stats['total_count']}\n"
        )

        for interval, stats in accuracy_stats.get("by_interval", {}).items():
            if stats["count"] > 0:
                interval_accuracy = stats["direction_accuracy"]
                interval_color = GREEN if interval_accuracy >= 75 else YELLOW if interval_accuracy >= 60 else RED
                accuracy_info += (
                    f"{CYAN}│{RESET} {interval:2d} мин: {interval_color}{interval_accuracy:.1f}%{RESET} "
                    f"(ошибка: {stats['avg_error']:.2f}%, замеров: {stats['count']})\n"
                )
                
        accuracy_info += f"{CYAN}└────────────────────────────────────────────────────────────┘{RESET}"

    # Информация о риске
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

        # Создаем визуальный индикатор риска
        var_pct = risk_metrics.get('var_95_pct', 0)
        var_bars = "▁▂▃▄▅▆▇█"
        var_index = min(int(var_pct / 1.25), 7)  # до 10% риска
        var_visual = var_bars[var_index]
        
        risk_info = (
            f"\n{CYAN}┌─ ОЦЕНКА РИСКА ────────────────────────────────────────────┐{RESET}\n"
            f"{CYAN}│{RESET} Уровень: {risk_color}{risk_metrics['risk_level']} {risk_symbol}{RESET}\n"
            f"{CYAN}│{RESET} VaR 95%: {risk_color}{risk_metrics['var_95']:.2f} ({risk_metrics.get('var_95_pct', 0):.2f}%) {var_visual}{RESET}\n"
            f"{CYAN}│{RESET} Макс. просадка: {risk_metrics['max_loss_pct']:.2f}%\n"
            f"{CYAN}└────────────────────────────────────────────────────────────┘{RESET}"
        )

    # Разделитель подсекций
    section_divider = f"{CYAN}─────────────────────────────────────────────────────────────{RESET}"
    
    # Оформляем анализ рынка
    signal_part = ""
    reason_part = ""
    
    if "СИГНАЛ к ЛОНГУ" in analysis:
        signal_part = f"{GREEN}СИГНАЛ к ЛОНГУ{RESET}"
    elif "СИГНАЛ к ШОРТУ" in analysis:
        signal_part = f"{RED}СИГНАЛ к ШОРТУ{RESET}"
    else:
        signal_part = f"{YELLOW}РЫНОК НЕЙТРАЛЕН{RESET}"
    
    if "(" in analysis and ")" in analysis:
        bull_strength = float(analysis.split("(")[1].split("%")[0])
        strength_color = GREEN if bull_strength > 65 else RED if bull_strength < 35 else YELLOW
        
        # Визуализация бычьей силы
        strength_bars = "▁▂▃▄▅▆▇█"
        strength_index = min(int(bull_strength / 12.5), 7)
        strength_visual = strength_bars[strength_index]
        
        strength_part = f"{strength_color}{bull_strength:.1f}% {strength_visual}{RESET}"
        
        if "-" in analysis:
            reason_part = analysis.split("-")[1].strip()
    
    analysis_formatted = f"{signal_part} ({strength_part}) - {reason_part}"
    
    # Формируем основной отчет
    header = (
        f"\n{MAGENTA}{'='*70}{RESET}\n"
        f"{MAGENTA}║{RESET} {WHITE}{ticker.upper()} - Торговый анализ{RESET}{' '*40}{MAGENTA}║{RESET}\n"
        f"{MAGENTA}{'='*70}{RESET}\n"
    )
    
    timestamp = (
        f"{CYAN}┌─ ИНФОРМАЦИЯ ─────────────────────────────────────────────┐{RESET}\n"
        f"{CYAN}│{RESET} Дата и время: {WHITE}{current_time.strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n"
        f"{CYAN}│{RESET} Тикер: {WHITE}{ticker.upper()}{RESET}  FIGI: {figi}\n"
        f"{CYAN}└────────────────────────────────────────────────────────────┘{RESET}"
    )
    
    prices = (
        f"{CYAN}┌─ ТЕКУЩИЕ ЦЕНЫ ───────────────────────────────────────────┐{RESET}\n"
        f"{CYAN}│{RESET} Рыночная цена: {WHITE}{raw_close:.2f}{RESET}\n"
        f"{CYAN}│{RESET} Цена закрытия: {WHITE}{last_close:.2f}{RESET}\n"
        f"{CYAN}│{RESET} Прогноз (1 мин): {color_pred}{predicted_price:.2f} {trend_symbol} ({(predicted_price/raw_close-1)*100:+.2f}%){RESET}\n"
        f"{CYAN}└────────────────────────────────────────────────────────────┘{RESET}"
    )
    
    predictions_section = (
        f"{CYAN}┌─ ПРОГНОЗЫ ──────────────────────────────────────────────┐{RESET}\n"
        f"{multi_preds_str}\n"
        f"{CYAN}│{RESET}\n"
        f"{prediction_chart}\n"
        f"{CYAN}└────────────────────────────────────────────────────────────┘{RESET}"
    )
    
    market_analysis = (
        f"{CYAN}┌─ АНАЛИЗ РЫНКА ───────────────────────────────────────────┐{RESET}\n"
        f"{CYAN}│{RESET} {analysis_formatted}\n"
        f"{CYAN}│{RESET} Режим: {regime_str}\n"
        f"{CYAN}└────────────────────────────────────────────────────────────┘{RESET}"
    )
    
    indicators_section = (
        f"{CYAN}┌─ ИНДИКАТОРЫ ────────────────────────────────────────────┐{RESET}\n"
        f"{CYAN}│{RESET} {' | '.join(indicators_display)}\n"
        f"{CYAN}│{RESET}\n"
        f"{CYAN}│{RESET} {detailed_indicators}\n"
        f"{CYAN}└────────────────────────────────────────────────────────────┘{RESET}"
    )
    
    # Исторический график (если доступен)
    history_section = ""
    if historical_chart:
        history_section = (
            f"{CYAN}┌─ ИСТОРИЯ ЦЕН (последние 20 точек) ─────────────────────┐{RESET}\n"
            f"{historical_chart}\n"
            f"{CYAN}└────────────────────────────────────────────────────────────┘{RESET}"
        )
    
    # Собираем все вместе
    report = (
        f"{header}\n"
        f"{timestamp}\n\n"
        f"{prices}\n\n"
        f"{predictions_section}\n\n"
        f"{market_analysis}\n\n"
        f"{indicators_section}\n"
        f"{history_section}\n"
        f"{accuracy_info}\n"
        f"{risk_info}\n\n"
        f"{MAGENTA}{'='*70}{RESET}\n"
    )
    
    print(report)


async def main():
    ticker = input("Введите тикер: ")
    try:
        figi = get_figi_by_ticker(ticker)
    except ValueError as e:
        print(f"Ошибка: {e}")
        return

    prediction_tracker = PredictionTracker(ticker)
    last_update_time = datetime.now() - timedelta(
        minutes=10)  # Инициализация для первого прогноза
    model_update_interval = timedelta(
        minutes=30)  # Обновляем модель каждые 30 минут
    data_update_interval = timedelta(
        seconds=10)  # Обновляем данные каждые 10 секунд

    async def get_current_price():
        retry_count = 0
        max_retries = 3
        retry_delay = 2  # секунды

        while retry_count < max_retries:
            try:
                async with AsyncClient(TOKEN) as client:
                    response = await client.market_data.get_last_prices(
                        figi=[figi])
                    if response and response.last_prices:
                        return float(
                            quotation_to_decimal(
                                response.last_prices[0].price))

                    print(
                        f"Предупреждение: API вернул пустой результат (попытка {retry_count+1}/{max_retries})"
                    )
                    retry_count += 1

            except Exception as e:
                print(
                    f"Ошибка при получении текущей цены: {e} (попытка {retry_count+1}/{max_retries})"
                )
                retry_count += 1

            if retry_count < max_retries:
                print(f"Повторная попытка через {retry_delay} сек...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Увеличиваем задержку при каждой повторной попытке

        print("Не удалось получить текущую цену после нескольких попыток")
        # Возвращаем кешированное значение, если оно есть
        if df_cache is not None and "last_raw_close" in df_cache:
            print(
                f"Используем последнюю известную цену: {df_cache['last_raw_close']}"
            )
            return df_cache["last_raw_close"]
        return None

    # Кэшируем последние данные
    df_cache = None
    predictions_cache = None
    regimes_cache = {}

    while True:
        current_time = datetime.now()
        current_market_price = await get_current_price()

        if current_market_price is not None:
            prediction_tracker.update_actuals(current_time,
                                              current_market_price)

        # Обновляем полную модель только с определенной периодичностью
        need_model_update = (current_time -
                             last_update_time) > model_update_interval

        if need_model_update or df_cache is None:
            show_loading_animation("Обновление модели и данных")
            df = await fetch_candles(figi, days=90)
            last_update_time = current_time

            if df.empty:
                print("Нет данных свечей для данного тикера")
                await asyncio.sleep(30)  # Увеличиваем паузу при ошибке
                continue

            df = preprocess_data(df)
            df = calculate_indicators(df)

            time_intervals = [1, 5, 15, 30]
            predictions = predict_multiple_steps_enhanced(df, lags=3)

            # Сохраняем кэш
            df_cache = df
            predictions_cache = predictions

            # Обновляем режимы рынка и добавляем прогнозы только при полном обновлении
            regimes_cache = detect_regime(df)
            prediction_tracker.add_predictions(current_time, predictions,
                                               time_intervals)

        # Используем кэшированные данные
        last_close = df_cache["close_clean"].iloc[-1]
        raw_close = current_market_price if current_market_price else df_cache[
            "last_raw_close"]

        indicators = {
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

        analysis = analyze_market(indicators, raw_close, predictions_cache,
                                  df_cache)
        accuracy_stats = prediction_tracker.get_accuracy_stats()

        # Рассчитываем метрики риска
        risk_metrics = calculate_risk_metrics(df_cache, raw_close,
                                              predictions_cache)

        clear_terminal()
        print_report(ticker, figi, last_close, raw_close, predictions_cache,
                     current_time, indicators, regimes_cache, analysis,
                     accuracy_stats, risk_metrics, df_cache)

        await asyncio.sleep(data_update_interval.total_seconds())


if __name__ == "__main__":
    asyncio.run(main())
