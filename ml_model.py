import pandas as pd
import numpy as np
import talib as ta
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


class MLTradingModel:
    def __init__(self, model_path="ml_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.scaler = None

        # V11.2: Final Refinement (Optimized Speed & Stability)
        self.forward_window = 12  # ลดลงเล็กน้อยเพื่อความไว
        self.min_threshold = 0.0035  # ปรับเกณฑ์กำไรขั้นต่ำให้สอดคล้องกับพฤติกรรมทองคำปัจจุบัน

    def create_features(self, df):
        """สร้าง Features เวอร์ชัน 11.2: Focus on Volatility-Adjusted Momentum"""
        df = df.copy()
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values.astype(float)

        # 1. Market Regime & Trend Force
        df['slope_20'] = ta.LINEARREG_SLOPE(close, timeperiod=20) / (close + 1e-9)
        df['adx'] = ta.ADX(high, low, close, timeperiod=14)

        # 2. Squeeze & Volatility Change
        atr = ta.ATR(high, low, close, timeperiod=14)
        upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['sqz_idx'] = (upper - lower) / (atr + 1e-9)
        df['atr_ratio'] = atr / ta.SMA(atr, timeperiod=50)  # ดูว่าปัจจุบันผันผวนกว่าค่าเฉลี่ยไหม

        # 3. Momentum & Force Index Proxy
        df['rsi_14'] = ta.RSI(close, timeperiod=14)
        df['mfi'] = ta.MFI(high, low, close, volume, timeperiod=14)
        _, _, df['macd_hist'] = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_hist'] /= (close + 1e-9)

        # 4. Price Contextual Location
        df['price_in_bb'] = (close - lower) / (upper - lower + 1e-9)
        df['dist_sma_200'] = (close - ta.SMA(close, timeperiod=200)) / (ta.SMA(close, timeperiod=200) + 1e-9)
        df['dist_sma_20'] = (close - ta.SMA(close, timeperiod=20)) / (ta.SMA(close, timeperiod=20) + 1e-9)

        # 5. Volatility-Adjusted Returns
        df['natr'] = ta.NATR(high, low, close, timeperiod=14)

        # 6. Candlestick Patterns (Price Action)
        df['body_size'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
        df['upper_wick'] = (df['high'] - np.maximum(df['close'], df['open'])) / (df['high'] - df['low'] + 1e-9)

        # 7. Time Context
        dt = pd.to_datetime(df['time'], unit='s')
        df['hour'] = dt.dt.hour
        df['market_active'] = dt.dt.hour.map(lambda x: 1 if 14 <= x <= 20 else 0)

        df = df.dropna()
        return df

    def create_labels(self, df):
        """สร้าง Label เวอร์ชัน 11.2: Dynamic Triple Barrier"""
        close = df['close'].values
        atr_val = ta.ATR(df['high'].values, df['low'].values, close, 14)

        labels = np.zeros(len(df))
        # ปรับเกณฑ์ตามความผันผวนจริง แต่ไม่ต่ำกว่าที่กำหนด
        dynamic_threshold = np.maximum(self.min_threshold, (atr_val / close) * 1.25)

        for i in range(len(df) - self.forward_window):
            future_slice = close[i + 1: i + 1 + self.forward_window]
            max_f = np.max(future_slice)
            min_f = np.min(future_slice)

            ret_up = (max_f - close[i]) / close[i]
            ret_down = (close[i] - min_f) / close[i]

            # เน้น Risk-Reward 1:2
            if ret_up > dynamic_threshold[i] and ret_down < (ret_up * 0.5):
                labels[i] = 1
            elif ret_down > dynamic_threshold[i] and ret_up < (ret_down * 0.5):
                labels[i] = -1

        df['label'] = labels
        df = df.iloc[:-self.forward_window].copy()
        return df

    def prepare_training_data(self, symbol, timeframe=None, num_bars=7000):
        if timeframe is None: timeframe = mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None: return None, None, None, None

        df = pd.DataFrame(rates)
        df = self.create_features(df)
        df = self.create_labels(df)

        self.feature_names = [
            'slope_20', 'adx', 'sqz_idx', 'atr_ratio', 'rsi_14', 'mfi', 'macd_hist',
            'price_in_bb', 'dist_sma_200', 'dist_sma_20', 'natr', 'body_size', 'upper_wick', 'hour', 'market_active'
        ]

        X = df[self.feature_names].values
        y = df['label'].values
        close_prices = df['close'].values
        dates = pd.to_datetime(df['time'], unit='s')

        return X, y, close_prices, dates

    def train(self, symbol, timeframe=None, num_bars=7000):
        X, y, close_prices, _ = self.prepare_training_data(symbol, timeframe, num_bars)
        if X is None or len(X) < 100: return False

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        gbm = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.8, random_state=42
        )

        rf = RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=12, class_weight='balanced', random_state=42
        )

        self.model = VotingClassifier(
            estimators=[('gbm', gbm), ('rf', rf)],
            voting='soft',
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        print(f"\n✅ Ensemble Model V11.2 (Final) Trained: {symbol}")
        print(f"   🔥 Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, self.model_path)
        self.is_trained = True
        return True

    def simulate(self, symbol, timeframe=None, num_bars=7000):
        X, y, close_prices, dates = self.prepare_training_data(symbol, timeframe, num_bars)
        split_idx = int(len(X) * 0.8)
        X_test_scaled = self.scaler.transform(X[split_idx:])
        close_test = close_prices[split_idx:]
        dates_test = dates[split_idx:]

        probs = self.model.predict_proba(X_test_scaled)
        classes = self.model.classes_

        predictions = []
        CONF_THRESHOLD = 0.54

        for p in probs:
            if np.max(p) < CONF_THRESHOLD:
                predictions.append(0)
            else:
                predictions.append(classes[np.argmax(p)])

        trading_cost = 0.0004
        returns_bh = np.diff(close_test) / close_test[:-1]

        strat_returns = []
        pos = 0
        for i in range(len(predictions) - 1):
            sig = predictions[i]

            # Sentinel Logic V11.2
            current_slope = X[split_idx + i][0]  # slope_20
            current_adx = X[split_idx + i][1]
            slope_limit = 0.00012 if current_adx > 25 else 0.00006

            if (sig == 1 and current_slope < -slope_limit) or (sig == -1 and current_slope > slope_limit):
                sig = 0

            cost = trading_cost if sig != pos and sig != 0 else 0
            pos = sig
            if pos == 1:
                strat_returns.append(returns_bh[i] - cost)
            elif pos == -1:
                strat_returns.append(-returns_bh[i] - cost)
            else:
                strat_returns.append(0)

        cum_bh = np.cumprod(1 + returns_bh)
        cum_strat = np.cumprod(1 + np.array(strat_returns))

        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plt.plot(dates_test[1:], cum_bh, label='Buy & Hold', color='#7f8c8d', alpha=0.5)
        plt.plot(dates_test[1:], cum_strat, label='Ensemble V11.2', color='#2980b9', linewidth=2.5)
        plt.title(f'Final Performance: {symbol} - V11.2 Optimized Ensemble')
        plt.legend()
        plt.grid(alpha=0.2)

        plt.subplot(2, 1, 2)
        counts = Counter(predictions)
        plt.bar(['Sell', 'Hold', 'Buy'], [counts[-1], counts[0], counts[1]], color=['#c0392b', '#bdc3c7', '#27ae60'])
        plt.tight_layout()
        plt.show()

        print(f"\n📊 Result: Strategy {cum_strat[-1] - 1:.2%} vs B&H {cum_bh[-1] - 1:.2%}")
        print(f"   Signals - Buy: {counts[1]}, Hold: {counts[0]}, Sell: {counts[-1]}")


if __name__ == '__main__':
    import MetaTrader5 as mt5

    if mt5.initialize():
        ml = MLTradingModel()
        symbol = "XAUUSDm"
        if ml.train(symbol):
            ml.simulate(symbol)
        mt5.shutdown()