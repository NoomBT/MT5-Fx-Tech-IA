import pandas as pd
import numpy as np
import talib as ta
import joblib
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import io
import MetaTrader5 as mt5

if not mt5.initialize():
    print("❌ ไม่สามารถเชื่อมต่อ MT5 ได้")
    exit()
# Set UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class MLTradingModel:
    def __init__(self, model_path="ml_model_gpu.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.scaler = None
        self.label_encoder = None  # เพิ่ม LabelEncoder สำหรับ XGBoost

        # 3-Class Classification: 1=Buy, 0=Neutral, -1=Sell
        self.forward_window = 5
        self.threshold = 0.002
        self.prob_threshold = 0.30

    def create_features(self, df):
        """สร้าง Features เชิงลึก (EMA/MACD/RSI Strategy: Fast=12, Slow=26)"""
        df = df.copy()
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values

        # ===== EMA Features =====
        ema_12 = ta.EMA(close, timeperiod=12)
        ema_26 = ta.EMA(close, timeperiod=26)
        df['ema_cross'] = (ema_12 - ema_26) / ema_26
        df['price_to_ema12'] = (close - ema_12) / ema_12
        df['price_to_ema26'] = (close - ema_26) / ema_26
        ema_12_series = pd.Series(ema_12, index=df.index)
        ema_26_series = pd.Series(ema_26, index=df.index)
        df['ema12_slope'] = ema_12_series.diff(1).rolling(window=3).mean()
        df['ema26_slope'] = ema_26_series.diff(1).rolling(window=3).mean()

        # ===== MACD Features =====
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_line'] = macd / close
        df['macd_signal'] = macd_signal / close
        df['macd_hist'] = macd_hist / close
        macd_series = pd.Series(macd, index=df.index)
        macd_signal_series = pd.Series(macd_signal, index=df.index)
        df['macd_cross'] = (macd_series - macd_signal_series) / close
        macd_hist_series = pd.Series(macd_hist, index=df.index)
        df['macd_hist_slope'] = macd_hist_series.diff(1).rolling(window=3).mean()

        # ===== RSI Features =====
        rsi = ta.RSI(close, timeperiod=14)
        rsi_series = pd.Series(rsi, index=df.index)
        df['rsi_14'] = rsi
        df['rsi_slope'] = rsi_series.diff(1).rolling(window=3).mean()
        df['rsi_buy_signal'] = (rsi - 60) / 40
        df['rsi_sell_signal'] = (rsi - 40) / 40
        df['rsi_trend_strength'] = np.abs(rsi - 50) / 50

        # ===== Price Action Features =====
        df['price_change'] = df['close'].pct_change()
        df['price_momentum'] = df['close'].pct_change(3)
        df['volatility'] = df['close'].rolling(window=10).std() / df['close']
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=20).mean()

        # ===== Candlestick Pattern Features =====
        body = abs(close - open_price)
        body = pd.Series(body, index=df.index)
        body_ratio = body / (high - low + 1e-10)
        body_ratio = pd.Series(body_ratio, index=df.index)
        upper_shadow = high - np.maximum(open_price, close)
        upper_shadow_ratio = upper_shadow / (high - low + 1e-10)
        upper_shadow_ratio = pd.Series(upper_shadow_ratio, index=df.index)
        lower_shadow = np.minimum(open_price, close) - low
        lower_shadow_ratio = lower_shadow / (high - low + 1e-10)
        lower_shadow_ratio = pd.Series(lower_shadow_ratio, index=df.index)
        candle_direction = pd.Series(np.where(close > open_price, 1, -1), index=df.index)

        df['body_ratio'] = body_ratio
        df['body_momentum'] = body_ratio - body_ratio.rolling(5).mean()
        df['upper_shadow_ratio'] = upper_shadow_ratio
        df['lower_shadow_ratio'] = lower_shadow_ratio
        df['doji'] = (body_ratio < 0.1).astype(int)

        prev_body = body.shift(1)
        df['engulfing'] = np.where(
            (candle_direction == 1) & (candle_direction.shift(1) == -1) & (body > prev_body * 1.5),
            1,
            np.where(
                (candle_direction == -1) & (candle_direction.shift(1) == 1) & (body > prev_body * 1.5),
                -1,
                0
            )
        )

        df = df.dropna()
        return df

    def create_labels(self, df):
        """3-Class Classification: 1=Buy, 0=Neutral, -1=Sell"""
        future_price = df['close'].shift(-self.forward_window)
        price_change_pct = (future_price - df['close']) / df['close']

        df['label'] = 0  # Neutral
        df.loc[price_change_pct > self.threshold, 'label'] = 1  # Buy
        df.loc[price_change_pct < -self.threshold, 'label'] = -1  # Sell

        df = df.dropna(subset=['label'])
        return df

    def prepare_training_data(self, symbol, timeframe=None, num_bars=2000):
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M5

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None:
            return None, None, None

        df = pd.DataFrame(rates)
        df = self.create_features(df)
        df = self.create_labels(df)

        self.feature_names = [
            'ema_cross', 'price_to_ema12', 'price_to_ema26', 'ema12_slope', 'ema26_slope',
            'macd_line', 'macd_signal', 'macd_hist', 'macd_cross', 'macd_hist_slope',
            'rsi_14', 'rsi_slope', 'rsi_buy_signal', 'rsi_sell_signal', 'rsi_trend_strength',
            'price_change', 'price_momentum', 'volatility', 'volatility_ratio',
            'body_ratio', 'body_momentum', 'upper_shadow_ratio', 'lower_shadow_ratio',
            'doji', 'engulfing'
        ]

        X = df[self.feature_names].values
        y = df['label'].values
        close_prices = df['close'].values

        return X, y, close_prices

    def train(self, symbol, timeframe=None, num_bars=3000):
        """ฝึกโมเดล XGBoost โดยใช้ GPU (3-Class: Buy, Neutral, Sell)"""
        from xgboost import XGBClassifier
        from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from collections import Counter

        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1

        X, y, close_prices = self.prepare_training_data(symbol, timeframe, num_bars)

        if X is None or len(X) < 100:
            print(f"❌ ไม่พบข้อมูลเพียงพอสำหรับ {symbol}")
            return False

        class_dist = Counter(y)
        print(f"\n📊 Class Distribution: {dict(class_dist)}")

        # XGBoost ต้องการให้ Label เริ่มตั้งแต่ 0, 1, 2 จึงต้องใช้ LabelEncoder แปลง -1, 0, 1
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

        print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"\n🚀 เริ่มทำการเทรนโมเดลด้วย XGBoost (GPU Accelerated)...")

        # ปรับพารามิเตอร์สำหรับ XGBoost
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        tscv = TimeSeriesSplit(n_splits=5)

        # ตั้งค่า XGBoost ให้ใช้ GPU
        # หมายเหตุ: สำหรับ XGBoost เวอร์ชันใหม่ใช้ device='cuda' (เวอร์ชันเก่าใช้ tree_method='gpu_hist')
        base_model = XGBClassifier(
            tree_method='hist',
            device='cuda',  # <--- คำสั่งเรียกใช้ GPU
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss'
        )

        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            scoring='f1_macro',
            random_state=42,
            n_jobs=1,  # ระวังการตั้งค่า n_jobs=-1 พร้อม GPU อาจเกิดคอขวด แนะนำให้รันทีละรอบบน GPU
            verbose=1
        )

        # เริ่มการจูนและเทรน
        random_search.fit(X_train, y_train)

        print(f"\n✅ Best Parameters: {random_search.best_params_}")
        print(f"   Best CV F1-Macro: {random_search.best_score_:.2%}")

        self.model = random_search.best_estimator_

        # ทำนายผล
        y_pred_train_encoded = self.model.predict(X_train)
        y_pred_test_encoded = self.model.predict(X_test)

        # แปลงกลับเป็น -1, 0, 1 เพื่อวัดผล
        y_test_original = self.label_encoder.inverse_transform(y_test)
        y_pred_test = self.label_encoder.inverse_transform(y_pred_test_encoded)

        accuracy = accuracy_score(y_test_original, y_pred_test)
        f1 = f1_score(y_test_original, y_pred_test, average='macro')

        print(f"\n✅ Model trained for {symbol}")
        print(f"   Test Accuracy: {accuracy:.2%}")
        print(f"   Test F1-Macro: {f1:.2%}")

        cm = confusion_matrix(y_test_original, y_pred_test)
        print(f"\n📊 Confusion Matrix:")
        print(f"        Pred: -1   0   +1")
        print(f"True -1:  {cm[0][0]:3d} {cm[0][1]:3d} {cm[0][2]:3d}")
        print(f"True  0:  {cm[1][0]:3d} {cm[1][1]:3d} {cm[1][2]:3d}")
        print(f"True +1:  {cm[2][0]:3d} {cm[2][1]:3d} {cm[2][2]:3d}")

        print(f"\n📋 Classification Report:")
        print(classification_report(y_test_original, y_pred_test, target_names=['Sell (-1)', 'Neutral (0)', 'Buy (1)']))

        # บันทึกโมเดล (รวม LabelEncoder ไปด้วย)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder
        }, self.model_path)
        self.is_trained = True

        return True

    def load_model(self):
        """โหลดโมเดล"""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']

            # รองรับโค้ดเก่าที่อาจจะยังไม่มี label_encoder
            if 'label_encoder' in data:
                self.label_encoder = data['label_encoder']

            if 'prob_threshold' in data:
                self.prob_threshold = data['prob_threshold']
            self.is_trained = True
            return True
        return False

    def predict(self, symbol, timeframe=None):
        """ทำนายสัญญาณ 3-Class: BULLISH, BEARISH, NEUTRAL"""
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M5

        if not self.is_trained or self.model is None:
            if not self.load_model():
                return "NEUTRAL"

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 60)
        if rates is None or len(rates) < 30:
            return "NEUTRAL"

        df = pd.DataFrame(rates)
        df = self.create_features(df)
        df = df.dropna()

        if df.empty or df.isnull().any().any():
            return "NEUTRAL"

        X = df[self.feature_names].values[-1:]
        X_scaled = self.scaler.transform(X)

        probas = self.model.predict_proba(X_scaled)[0]

        max_prob_idx = np.argmax(probas)
        max_prob = probas[max_prob_idx]

        # ดึง class ออกมา (จะเป็น 0, 1, 2) แล้วแปลงกลับเป็น -1, 0, 1
        encoded_prediction = self.model.classes_[max_prob_idx]
        prediction = self.label_encoder.inverse_transform([encoded_prediction])[0]

        if max_prob < self.prob_threshold:
            return "NEUTRAL"

        if prediction == 1:
            return "BULLISH"
        elif prediction == -1:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def get_feature_importance(self):
        """ดู Feature Importance"""
        if self.model is not None:
            importance = self.model.feature_importances_
            print("\n📈 Feature Importance:")
            sorted_idx = np.argsort(importance)[::-1]
            for idx in sorted_idx:
                bar = '█' * int(importance[idx] * 50)
                print(f"   {self.feature_names[idx]:20s}: {importance[idx]:.4f} {bar}")

    def simulate(self, symbol, timeframe=None, num_bars=2000):
        """Simulation: เปรียบเทียบ Strategy vs Buy & Hold"""
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1

        X, y, close_prices = self.prepare_training_data(symbol, timeframe, num_bars)

        if X is None:
            print("❌ ไม่สามารถดึงข้อมูลสำหรับ Simulation ได้")
            return

        X_scaled = self.scaler.transform(X)

        split_idx = int(len(X_scaled) * 0.8)
        X_test_scaled = X_scaled[split_idx:]
        y_test = y[split_idx:]
        close_prices_test = close_prices[split_idx:]

        # ทำนายผล (ได้ออกมาเป็น 0, 1, 2 ต้องแปลงกลับ)
        encoded_predictions = self.model.predict(X_test_scaled)
        predictions = self.label_encoder.inverse_transform(encoded_predictions)

        max_len = len(predictions) - self.forward_window

        bh_returns = np.zeros(max_len)
        strategy_returns = np.zeros(max_len)

        for i in range(max_len):
            future_idx = i + self.forward_window
            future_price = close_prices_test[future_idx]
            current_price = close_prices_test[i]
            period_return = (future_price - current_price) / current_price

            bh_returns[i] = period_return

            if predictions[i] == 1:  # Buy -> Long
                strategy_returns[i] = period_return
            elif predictions[i] == -1:  # Sell -> Short
                strategy_returns[i] = -period_return

        cumulative_bh = np.cumprod(1 + np.array(bh_returns))
        cumulative_strategy = np.cumprod(1 + np.array(strategy_returns))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        ax1.plot(cumulative_bh, label='Buy & Hold', color='blue', linewidth=2)
        ax1.plot(cumulative_strategy, label='ML Strategy', color='green', linewidth=2)
        ax1.set_title(f'{symbol} - Cumulative Returns (Test Set Only)', fontsize=14)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        signal_counts = {
            'Buy': np.sum(predictions == 1),
            'Neutral': np.sum(predictions == 0),
            'Sell': np.sum(predictions == -1)
        }
        colors = ['green', 'gray', 'red']
        ax2.bar(signal_counts.keys(), signal_counts.values(), color=colors)
        ax2.set_title(f'{symbol} - Signal Distribution (Test Set)', fontsize=14)
        ax2.set_ylabel('Count')
        for i, (k, v) in enumerate(signal_counts.items()):
            ax2.text(i, v + 5, str(v), ha='center', fontsize=12)

        plt.tight_layout()
        plt.savefig('simulation_result.png', dpi=150)
        plt.show()

        total_return_bh = cumulative_bh[-1] - 1
        total_return_strategy = cumulative_strategy[-1] - 1

        print(f"\n📊 Simulation Results for {symbol} (Test Set Only):")
        print(f"   Buy & Hold Return: {total_return_bh:.2%}")
        print(f"   ML Strategy Return: {total_return_strategy:.2%}")
        print(f"   Test Size: {len(X_test_scaled)}")
        print(
            f"   Signals - Buy: {signal_counts['Buy']}, Neutral: {signal_counts['Neutral']}, Sell: {signal_counts['Sell']}")


if __name__ == '__main__':
    if not mt5.initialize():
        print("❌ ไม่สามารถเชื่อมต่อ MT5 ได้")
        exit()

    symbol = "XAUUSDm"
    ml = MLTradingModel()

    print("=" * 60)
    print(f"📊 กำลังฝึกโมเดลสำหรับ {symbol}")
    print("=" * 60)

    # รันบน GPU จะเร็วกว่ามาก สามารถเพิ่ม num_bars ไปที่ 10000+ ได้ถ้าต้องการ
    ml.train(symbol, mt5.TIMEFRAME_H1, num_bars=5000)

    print("\n📈 Feature Importance:")
    ml.get_feature_importance()

    print("\n🔮 ทดสอบการทำนาย:")
    signal = ml.predict(symbol, mt5.TIMEFRAME_M1)
    print(f"   Signal: {signal}")

    print("\n🎮 Running Simulation...")
    ml.simulate(symbol, mt5.TIMEFRAME_H1, num_bars=2000)

    mt5.shutdown()