import pandas as pd
import numpy as np
import talib as ta
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter


class MLTradingModel:
    def __init__(self, model_path="models/ml_model_no_RandomForest.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.scaler = None

        # V10.1: Optimized Sentinel
        self.forward_window = 14
        self.min_threshold = 0.0035

    def create_features(self, df):
        """สร้าง Features เวอร์ชัน 10.1: Market Regime & Force Analysis"""
        df = df.copy()
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values.astype(float)

        # 1. Market Regime
        df['slope_30'] = ta.LINEARREG_SLOPE(close, timeperiod=30) / close
        df['adx'] = ta.ADX(high, low, close, timeperiod=14)

        # 2. Volatility Squeeze
        atr = ta.ATR(high, low, close, timeperiod=14)
        upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['sqz_idx'] = (upper - lower) / (atr + 1e-9)

        # 3. Momentum & Divergence
        df['rsi_14'] = ta.RSI(close, timeperiod=14)
        df['mfi'] = ta.MFI(high, low, close, volume, timeperiod=14)
        _, _, df['macd_hist'] = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_hist'] /= (close + 1e-9)

        # 4. Price Location
        df['price_in_bb'] = (close - lower) / (upper - lower + 1e-9)
        df['dist_sma_200'] = (close - ta.SMA(close, timeperiod=200)) / (ta.SMA(close, timeperiod=200) + 1e-9)

        # 5. Volatility-Adjusted Returns
        df['natr'] = ta.NATR(high, low, close, timeperiod=14)

        # 6. Candlestick Context
        df['body_size'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-9)

        # 7. Time Context
        dt = pd.to_datetime(df['time'], unit='s')
        df['hour'] = dt.dt.hour
        df['market_active'] = dt.dt.hour.map(lambda x: 1 if 14 <= x <= 20 else 0)

        df = df.dropna()
        return df

    def create_labels(self, df):
        """สร้าง Label เวอร์ชัน 10.1: Fix Warning & Risk-Adjusted"""
        close = df['close'].values
        atr_val = ta.ATR(df['high'].values, df['low'].values, close, 14)

        # ป้องกัน All-NaN warning โดยการลดช่วงการคำนวณลงตาม forward_window
        labels = np.zeros(len(df))
        dynamic_threshold = np.maximum(self.min_threshold, (atr_val / close) * 1.3)

        # คำนวณถึงแค่จุดที่ยังมีข้อมูลอนาคตครบเท่านั้น
        for i in range(len(df) - self.forward_window):
            future_slice = close[i + 1: i + 1 + self.forward_window]
            max_f = np.max(future_slice)
            min_f = np.min(future_slice)

            ret_up = (max_f - close[i]) / close[i]
            ret_down = (close[i] - min_f) / close[i]

            # Risk-Reward 1:2 (ชนะต้องได้มากกว่าแพ้ 2 เท่า)
            if ret_up > dynamic_threshold[i] and ret_down < (ret_up * 0.5):
                labels[i] = 1
            elif ret_down > dynamic_threshold[i] and ret_up < (ret_down * 0.5):
                labels[i] = -1

        df['label'] = labels
        # ตัดข้อมูลส่วนท้ายที่ไม่มี label ออก
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
            'slope_30', 'adx', 'sqz_idx', 'rsi_14', 'mfi', 'macd_hist',
            'price_in_bb', 'dist_sma_200', 'natr', 'body_size', 'hour', 'market_active'
        ]

        X = df[self.feature_names].values
        y = df['label'].values
        close_prices = df['close'].values
        dates = pd.to_datetime(df['time'], unit='s')

        return X, y, close_prices, dates

    def train(self, symbol, timeframe=None, num_bars=7000):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.preprocessing import StandardScaler

        X, y, close_prices, _ = self.prepare_training_data(symbol, timeframe, num_bars)
        if X is None or len(X) < 100: return False

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        print(f"\n✅ Model V10.1 Trained: {symbol}")
        print(f"   🔥 Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        print(f"\n📋 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

        joblib.dump({'model': self.model, 'scaler': self.scaler, 'feature_names': self.feature_names}, self.model_path)
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
        for p in probs:
            if np.max(p) < 0.52:  # ปรับ Confidence เล็กน้อยเพื่อความสมดุล
                predictions.append(0)
            else:
                predictions.append(classes[np.argmax(p)])

        trading_cost = 0.0004
        returns_bh = np.diff(close_test) / close_test[:-1]

        strat_returns = []
        pos = 0
        for i in range(len(predictions) - 1):
            sig = predictions[i]
            # Sentinel Rule: กรองสวนเทรนด์
            current_slope = X[split_idx + i][0]
            if (sig == 1 and current_slope < -0.0001) or (sig == -1 and current_slope > 0.0001):
                sig = 0

            cost = trading_cost if sig != pos and sig != 0 else 0
            pos = sig
            if pos == 1:
                strat_returns.append(ret := returns_bh[i] - cost)
            elif pos == -1:
                strat_returns.append(ret := -returns_bh[i] - cost)
            else:
                strat_returns.append(0)

        cum_bh = np.cumprod(1 + returns_bh)
        cum_strat = np.cumprod(1 + np.array(strat_returns))

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(dates_test[1:], cum_bh, label='Buy & Hold', color='#7f8c8d', alpha=0.5)
        plt.plot(dates_test[1:], cum_strat, label='V10.1 Sentinel', color='#1abc9c', linewidth=2.5)
        plt.title(f'Performance: {symbol} (Alpha Strategy)')
        plt.legend()
        plt.grid(alpha=0.2)

        plt.subplot(2, 1, 2)
        counts = Counter(predictions)
        plt.bar(['Sell', 'Hold', 'Buy'], [counts[-1], counts[0], counts[1]], color=['#e74c3c', '#bdc3c7', '#2ecc71'])
        plt.tight_layout()
        # plt.show()

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