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
    def __init__(self, model_path="ml_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.scaler = None
        
        # 3-Class Classification: 1=Buy, 0=Neutral, -1=Sell
        self.forward_window = 5
        self.threshold = 0.002
        self.prob_threshold = 0.25  # ลดลงเพื่อให้ predict Buy/Sell บ่อยขึ้น

    def create_features(self, df):
        """สร้าง Features เชิงลึก (EMA/MACD/RSI Strategy: Fast=12, Slow=26)"""
        df = df.copy()
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        
        # ===== EMA Features (Fast=12, Slow=26) =====
        ema_12 = ta.EMA(close, timeperiod=12)
        ema_26 = ta.EMA(close, timeperiod=26)
        
        # EMA Cross (EMA12 - EMA26) / EMA26
        df['ema_cross'] = (ema_12 - ema_26) / ema_26
        
        # Price to EMA12
        df['price_to_ema12'] = (close - ema_12) / ema_12
        
        # Price to EMA26
        df['price_to_ema26'] = (close - ema_26) / ema_26
        
        # EMA Slope (3-period)
        ema_12_series = pd.Series(ema_12, index=df.index)
        ema_26_series = pd.Series(ema_26, index=df.index)
        df['ema12_slope'] = ema_12_series.diff(1).rolling(window=3).mean()
        df['ema26_slope'] = ema_26_series.diff(1).rolling(window=3).mean()
        
        # ===== MACD Features (Fast=12, Slow=26, Signal=9) =====
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # MACD Line (as ratio to price)
        df['macd_line'] = macd / close
        
        # MACD Signal Line
        df['macd_signal'] = macd_signal / close
        
        # MACD Histogram (oscillator)
        df['macd_hist'] = macd_hist / close
        
        # MACD Cross (MACD above/below signal)
        macd_series = pd.Series(macd, index=df.index)
        macd_signal_series = pd.Series(macd_signal, index=df.index)
        df['macd_cross'] = (macd_series - macd_signal_series) / close
        
        # MACD Histogram Slope
        macd_hist_series = pd.Series(macd_hist, index=df.index)
        df['macd_hist_slope'] = macd_hist_series.diff(1).rolling(window=3).mean()
        
        # ===== RSI Features (Period=14, Buy=60, Sell=40) =====
        rsi = ta.RSI(close, timeperiod=14)
        rsi_series = pd.Series(rsi, index=df.index)
        df['rsi_14'] = rsi
        df['rsi_slope'] = rsi_series.diff(1).rolling(window=3).mean()
        
        # RSI Buy/Sell Signal (normalized: >60 = buy, <40 = sell)
        df['rsi_buy_signal'] = (rsi - 60) / 40   # >60 = positive (strong buy), <60 = negative
        df['rsi_sell_signal'] = (rsi - 40) / 40 # <40 = positive (strong sell), >40 = negative
        
        # RSI Trend Confirmation (60-40 zone = strong trend)
        df['rsi_trend_strength'] = np.abs(rsi - 50) / 50  # 0-1 scale (higher = stronger trend)
        
        # ===== Price Action Features =====
        # Price change (momentum)
        df['price_change'] = df['close'].pct_change()
        
        # Price momentum (3-period)
        df['price_momentum'] = df['close'].pct_change(3)
        
        # Volatility: Standard Deviation ของราคา 10 แท่ง (เป็น % ของราคา)
        df['volatility'] = df['close'].rolling(window=10).std() / df['close']
        
        # Volatility ratio (recent vs long-term)
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=20).mean()
        
        # ===== เพิ่ม Candlestick Pattern Features =====
        # Body size
        body = abs(close - open_price)
        body = pd.Series(body, index=df.index)  # แปลงเป็น Series
        body_ratio = body / (high - low + 1e-10)  # หลีกเลี่ยง divide by zero
        body_ratio = pd.Series(body_ratio, index=df.index)  # แปลงเป็น Series
        
        # Upper shadow
        upper_shadow = high - np.maximum(open_price, close)
        upper_shadow_ratio = upper_shadow / (high - low + 1e-10)
        upper_shadow_ratio = pd.Series(upper_shadow_ratio, index=df.index)
        
        # Lower shadow
        lower_shadow = np.minimum(open_price, close) - low
        lower_shadow_ratio = lower_shadow / (high - low + 1e-10)
        lower_shadow_ratio = pd.Series(lower_shadow_ratio, index=df.index)
        
        # Candle direction (1=bullish, -1=bearish)
        candle_direction = pd.Series(np.where(close > open_price, 1, -1), index=df.index)
        
        # Moving average of body (momentum)
        df['body_ratio'] = body_ratio
        df['body_momentum'] = body_ratio - body_ratio.rolling(5).mean()
        
        # Upper/Lower shadow ratio
        df['upper_shadow_ratio'] = upper_shadow_ratio
        df['lower_shadow_ratio'] = lower_shadow_ratio
        
        # Doji indicator (body เล็กมาก)
        df['doji'] = (body_ratio < 0.1).astype(int)
        
        # Engulfing pattern (simplified)
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
        """3-Class Classification: 1=Buy, 0=Neutral, -1=Sell
        ใช้ close ของ t+forward_window เทียบกับ close ปัจจุบัน"""
        
        future_price = df['close'].shift(-self.forward_window)
        price_change_pct = (future_price - df['close']) / df['close']
        
        df['label'] = 0  # Neutral
        df.loc[price_change_pct > self.threshold, 'label'] = 1   # Buy
        df.loc[price_change_pct < -self.threshold, 'label'] = -1  # Sell
        
        df = df.dropna(subset=['label'])
        
        return df

    def prepare_training_data(self, symbol, timeframe=None, num_bars=2000):
        """ดึงข้อมูลและเตรียม training data"""
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
        """ฝึกโมเดล Gradient Boosting (3-Class: Buy, Neutral, Sell)"""
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
        from sklearn.preprocessing import StandardScaler
        from collections import Counter
        
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1
        
        X, y, close_prices = self.prepare_training_data(symbol, timeframe, num_bars)
        
        if X is None or len(X) < 100:
            print(f"❌ ไม่พบข้อมูลเพียงพอสำหรับ {symbol}")
            return False
        
        class_dist = Counter(y)
        print(f"\n📊 Class Distribution: {dict(class_dist)}")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # ===== Gradient Boosting with Tuning =====
        print(f"\n🔍 Hyperparameter Tuning (Gradient Boosting)...")
        
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        base_model = GradientBoostingClassifier(random_state=42)
        
        # ใช้ class weights เพื่อให้ model predict Buy/Sell มากขึ้น
        # ปรับ weight เป็น {-1: 2.5, 0: 1, 1: 2.5}
        class_weights = {-1: 2.5, 0: 1, 1: 2.5}
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weights, y_train)
        
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            scoring='f1_macro',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"\n✅ Best Parameters: {random_search.best_params_}")
        print(f"   Best CV F1-Macro: {random_search.best_score_:.2%}")
        
        self.model = random_search.best_estimator_
        self.model.fit(X_train, y_train)
        
        # 3-Class: ใช้ predict โดยตรง
        y_pred = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='macro')
        
        print(f"\n✅ Model trained for {symbol}")
        print(f"   Test Accuracy: {accuracy:.2%}")
        print(f"   Test F1-Macro: {f1:.2%}")
        
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\n📊 Confusion Matrix:")
        print(f"        Pred: -1   0   +1")
        print(f"True -1:  {cm[0][0]:3d} {cm[0][1]:3d} {cm[0][2]:3d}")
        print(f"True  0:  {cm[1][0]:3d} {cm[1][1]:3d} {cm[1][2]:3d}")
        print(f"True +1:  {cm[2][0]:3d} {cm[2][1]:3d} {cm[2][2]:3d}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['Sell (-1)', 'Neutral (0)', 'Buy (1)']))
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
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
            if 'prob_threshold' in data:
                self.prob_threshold = data['prob_threshold']
            self.is_trained = True
            # print(f"✅ โหลดโมเดลสำเร็จ: {self.model_path}")
            # print(f"   Using threshold: {self.prob_threshold:.3f}")
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
        classes = self.model.classes_
        
        max_prob_idx = np.argmax(probas)
        max_prob = probas[max_prob_idx]
        prediction = classes[max_prob_idx]
        
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
        """Simulation: เปรียบเทียบ Strategy vs Buy & Hold (3-Class: Buy, Sell, Neutral)"""
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
        
        # 3-Class predictions
        probas = self.model.predict_proba(X_test_scaled)
        predictions = self.model.predict(X_test_scaled)
        
        max_len = len(predictions) - self.forward_window
        
        bh_returns = np.zeros(max_len)
        strategy_returns = np.zeros(max_len)
        
        for i in range(max_len):
            future_idx = i + self.forward_window
            future_price = close_prices_test[future_idx]
            current_price = close_prices_test[i]
            period_return = (future_price - current_price) / current_price
            
            bh_returns[i] = period_return
            
            if predictions[i] == 1:       # Buy -> Long
                strategy_returns[i] = period_return
            elif predictions[i] == -1:     # Sell -> Short
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
        print(f"   Signals - Buy: {signal_counts['Buy']}, Neutral: {signal_counts['Neutral']}, Sell: {signal_counts['Sell']}")


if __name__ == '__main__':
    import MetaTrader5 as mt5
    
    if not mt5.initialize():
        print("❌ ไม่สามารถเชื่อมต่อ MT5 ได้")
        exit()
    
    symbol = "XAUUSDm"
    ml = MLTradingModel()
    
    print("="*60)
    print(f"📊 กำลังฝึกโมเดลสำหรับ {symbol}")
    print("="*60)
    
    ml.train(symbol, mt5.TIMEFRAME_M15, num_bars=3000)
    
    print("\n📈 Feature Importance:")
    ml.get_feature_importance()
    
    print("\n🔮 ทดสอบการทำนาย:")
    signal = ml.predict(symbol, mt5.TIMEFRAME_M15)
    print(f"   Signal: {signal}")
    
    print("\n🎮 Running Simulation...")
    ml.simulate(symbol, mt5.TIMEFRAME_M15, num_bars=2000)
    
    mt5.shutdown()