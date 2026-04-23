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
        self.threshold = 0.002  # เพิ่มขึ้นเพื่อให้ Neutral มากขึ้น
        self.prob_threshold = 0.45  # ลดลง - กล้าเสี่ยงขึ้น

    def create_features(self, df, num_bars=2000):
        """สร้าง Features เชิงลึก (EMA/MACD/RSI Strategy: Fast=12, Slow=26)"""
        df = df.copy()
        df = df.reset_index(drop=True)  # Reset index for proper matching
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        
        # ===== Get H1 data for each M15 time =====
        # Get enough H1 bars to cover all M15 bars (M15 / 4 + buffer)
        h1_tf = mt5.TIMEFRAME_H1
        rates_h1 = mt5.copy_rates_from_pos("XAUUSDm", h1_tf, 0, num_bars)
        print(f"📊 rates_h1 count: {len(rates_h1)}")
        df['price_to_ema200_h1'] = 0.0
        df['h1_macd'] = 0.0
        df['h1_rsi'] = 50.0
        df['h1_adx'] = 25.0
        df['trend_alignment'] = 0.0
        df['h1_trend'] = 0
        df['m15_trend'] = 0
        df['h1_slope'] = 0.0
        df['m15_slope'] = 0.0
        df['h1_rsi_slope'] = 0.0
        df['h1_adx_slope'] = 0.0
        df['london_session'] = 0.0
        df['ny_session'] = 0.0
        df['ny_volatility'] = 0.0
        
        # Cast to float to avoid dtype warnings
        df['price_to_ema200_h1'] = df['price_to_ema200_h1'].astype(float)
        df['h1_macd'] = df['h1_macd'].astype(float)
        df['h1_rsi'] = df['h1_rsi'].astype(float)
        df['h1_adx'] = df['h1_adx'].astype(float)
        df['trend_alignment'] = df['trend_alignment'].astype(float)
        df['h1_trend'] = df['h1_trend'].astype(float)
        df['m15_trend'] = df['m15_trend'].astype(float)
        df['h1_slope'] = df['h1_slope'].astype(float)
        df['m15_slope'] = df['m15_slope'].astype(float)
        df['h1_rsi_slope'] = df['h1_rsi_slope'].astype(float)
        df['h1_adx_slope'] = df['h1_adx_slope'].astype(float)
        df['london_session'] = df['london_session'].astype(float)
        df['ny_session'] = df['ny_session'].astype(float)
        df['ny_volatility'] = df['ny_volatility'].astype(float)
        
        # ===== EMA Features (Fast=12, Slow=26) - needed before H1 merge =====
        ema_12 = ta.EMA(close, timeperiod=12)
        ema_26 = ta.EMA(close, timeperiod=26)
        
        if rates_h1 is not None and len(rates_h1) >= 26:
            try:
                df_h1 = pd.DataFrame(rates_h1)
                
                # Ensure time column exists for both dataframes before merge
                if 'time' in df_h1.columns:
                    df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                
                close_h1 = df_h1['close'].values
                high_h1 = df_h1['high'].values
                low_h1 = df_h1['low'].values
                
                # EMA200 on H1
                ema_200_h1 = ta.EMA(close_h1, timeperiod=200)
                
                # MACD on H1
                h1_macd, h1_signal, _ = ta.MACD(close_h1, fastperiod=12, slowperiod=26, signalperiod=9)
                
                # RSI on H1
                h1_rsi = ta.RSI(close_h1, timeperiod=14)
                
                # ADX on H1 (trend strength)
                h1_adx = ta.ADX(high_h1, low_h1, close_h1, timeperiod=14)
                
                # Add H1 features to df_h1 and compute SLOPES BEFORE merge
                df_h1['h1_ema200'] = ema_200_h1
                df_h1['h1_macd'] = h1_macd
                df_h1['h1_rsi'] = h1_rsi
                df_h1['h1_adx'] = h1_adx
                
                # Compute slopes BEFORE shifting (as % change)
                df_h1['h1_slope'] = df_h1['h1_ema200'].diff(1) / df_h1['h1_ema200'].shift(1) * 100
                df_h1['h1_rsi_slope'] = df_h1['h1_rsi'].diff(1)
                df_h1['h1_adx_slope'] = df_h1['h1_adx'].diff(1)
                
                # Shift by 1 to use PAST H1 data (prevent leakage)
                df_h1['h1_ema200'] = df_h1['h1_ema200'].shift(1)
                df_h1['h1_macd'] = df_h1['h1_macd'].shift(1)
                df_h1['h1_rsi'] = df_h1['h1_rsi'].shift(1)
                df_h1['h1_adx'] = df_h1['h1_adx'].shift(1)
                df_h1['h1_slope'] = df_h1['h1_slope'].shift(1)
                df_h1['h1_rsi_slope'] = df_h1['h1_rsi_slope'].shift(1)
                df_h1['h1_adx_slope'] = df_h1['h1_adx_slope'].shift(1)
                
                # Merge using time (backward = get H1 data that happened BEFORE this M15 bar)
                if 'time' in df.columns and 'time' in df_h1.columns:
                    df = df.sort_values('time')
                    df_h1 = df_h1.sort_values('time')
                    merge_cols = ['time', 'h1_ema200', 'h1_macd', 'h1_rsi', 'h1_adx', 
                                'h1_slope', 'h1_rsi_slope', 'h1_adx_slope']
                    merged = pd.merge_asof(df, df_h1[merge_cols], 
                                      on='time', direction='backward', tolerance=pd.Timedelta('15min'))
                    
                    # Calculate H1 features - use merged df directly instead of .values
                    if 'h1_ema200' in merged.columns and not merged['h1_ema200'].isna().all():
                        # Trend Alignment: use SLOPE of EMA instead of position
                        # Use pre-computed slopes from merge, don't recompute!
                        h1_ema200_series = pd.Series(merged['h1_ema200'].values, index=merged.index)
                        # Shift ema_12 by 1 to avoid leakage
                        m15_ema12_series = pd.Series(ema_12, index=df.index).shift(1)
                        m15_slope = m15_ema12_series.diff(1) / m15_ema12_series * 100  # as percentage
                        
                        df['price_to_ema200_h1'] = ((merged['close'] - merged['h1_ema200']) / merged['close'] * 100).fillna(0)
                        df['h1_slope'] = merged['h1_slope'].fillna(0)  # Already percentage from df_h1
                        df['m15_slope'] = m15_slope.fillna(0)  # Already percentage now
                        df['h1_macd'] = (merged['h1_macd'] / merged['close'] * 1000).fillna(0)
                        df['h1_rsi'] = merged['h1_rsi'].fillna(50)
                        df['h1_rsi_slope'] = merged['h1_rsi_slope'].fillna(0)  # Use pre-computed slope
                        df['h1_adx'] = merged['h1_adx'].fillna(25)
                        df['h1_adx_slope'] = merged['h1_adx_slope'].fillna(0)  # Use pre-computed slope
                        
                        # Trend Alignment: compare slopes
                        df['h1_trend'] = np.where(df['h1_slope'] > 0, 1, -1)  # H1 EMA slope
                        df['m15_trend'] = np.where(df['m15_slope'] > 0, 1, -1)  # M15 EMA slope
                        df['trend_alignment'] = np.where(df['h1_trend'] == df['m15_trend'], 1.0, -1.0)
            except Exception as e:
                pass
        
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
        df['h1_slope'] = df['ema26_slope']  # Same as ema26_slope for H1
        df['h1_slope_accel'] = df['h1_slope'].diff(1)  # Acceleration of slope
        
        # ===== Session Features (London: 8-16, NY: 13-21 UTC) =====
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
            hour = df['time'].dt.hour
            
            df['london_session'] = ((hour >= 8) & (hour < 16)).astype(float)
            df['ny_session'] = ((hour >= 13) & (hour < 21)).astype(float)
        
        # Volatility must be created before ny_volatility
        df['volatility'] = df['close'].rolling(window=10).std() / df['close']
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=20).mean()
        
        # Now create ny_volatility
        if 'time' in df.columns:
            df['ny_volatility'] = df['ny_session'] * df['volatility']
        
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
        # Already created above at line 155
        
        # Volatility ratio (recent vs long-term)
        # Already created above at line 156
        
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
        ใช้ Triple Barrier Method - พิจารณา TP/SL และเวลา"""
        
        tp_pct = 0.010  # TP 1.5% for H1
        sl_pct = 0.008  # SL 0.8% for H1
        max_bars = self.forward_window * 2  # Max holding period
        
        labels = []
        for i in range(len(df)):
            if i >= len(df) - max_bars:
                labels.append(0)
                continue
            
            entry_price = df['close'].iloc[i]
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
            
            hit_tp = False
            hit_sl = False
            
            for j in range(1, max_bars + 1):
                if i + j >= len(df):
                    break
                
                high = df['high'].iloc[i + j]
                low = df['low'].iloc[i + j]
                
                if high >= tp_price:
                    hit_tp = True
                    break
                if low <= sl_price:
                    hit_sl = True
                    break
            
            if hit_tp and not hit_sl:
                labels.append(1)  # Buy (TP hit first)
            elif hit_sl and not hit_tp:
                labels.append(-1)  # Sell (SL hit first)
            else:
                labels.append(0)  # Neutral (neither hit within max_bars)
        
        df['label'] = labels
        df = df.dropna(subset=['label'])
        
        return df

    def prepare_training_data(self, symbol, timeframe=None, num_bars=2000):
        """ดึงข้อมูลและเตรียม training data"""
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None:
            return None, None, None
        
        df = pd.DataFrame(rates)
        df = self.create_features(df, num_bars)
        df = self.create_labels(df)
        
        self.feature_names = [
            'h1_slope', 'h1_slope_accel',  # EMA slope and acceleration
            'london_session', 'ny_session', 'ny_volatility',
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
        """ฝึกโมเดล Random Forest (3-Class: Buy, Neutral, Sell)"""
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
        
        # ===== Random Forest with Tuning =====
        print(f"\n🔍 Hyperparameter Tuning (Random Forest)...")
        
        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [5, 7, 10],
            'min_samples_split': [20, 30],
            'min_samples_leaf': [10, 15],
            'max_features': ['sqrt', 'log2'],
            # 'class_weight': [{0: 0.8, 1: 3.0, -1: 2.0}]  # Favor Buy more, moderate Sell
            # 'class_weight': [{0: 1.2, 1: 4.0, -1: 1.0}]  # 15%
            'class_weight': [{0: 0.5, 1: 4.0, -1: 2.0}]  # 15%
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=15,  # ลดลง
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
        file_name, file_extension = os.path.splitext(self.model_path)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'prob_threshold': self.prob_threshold
        }, f"models/{file_name}-{symbol}{file_extension}")
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
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1
            
        if not self.is_trained or self.model is None:
            if not self.load_model():
                return "NEUTRAL"
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 60)
        if rates is None or len(rates) < 30:
            return "NEUTRAL"
        
        df = pd.DataFrame(rates)
        df = self.create_features(df, 60)
        df = df.dropna()
        
        if df.empty or df.isnull().any().any():
            return "NEUTRAL"
        
        # Volatility Filter: ถ้า volatility สูงเกินไป ให้เลือก Neutral
        volatility_now = df['volatility'].values[-1]
        volatility_ma = df['volatility'].rolling(20).mean().values[-1]
        if volatility_now > volatility_ma * 2.0:
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
                print(f"{self.feature_names[idx]:20s}: {importance[idx]:.4f} {bar}")

    def simulate(self, symbol, timeframe=None, num_bars=2000):
        """Simulation: เปรียบเทียบ Strategy vs Buy & Hold (3-Class: Buy, Sell, Neutral)"""
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1
        
        # ถ้ายังไม่มี model ให้โหลดก่อน
        if not self.is_trained or self.model is None:
            if not self.load_model():
                print("❌ ไม่พบโมเดล กรุณาฝึกโมเดลก่อน")
                return
        
        if self.scaler is None:
            print("❌ ไม่พบ Scaler กรุณาฝึกโมเดลก่อน")
            return
        
        X, y, close_prices = self.prepare_training_data(symbol, timeframe, num_bars)
        
        if X is None:
            print("❌ ไม่สามารถดึงข้อมูลสำหรับ Simulation ได้")
            return
        
        if self.scaler is None:
            print("❌ ไม่พบ Scaler กรุณาฝึกโมเดลก่อน")
            return
        
        X_scaled = self.scaler.transform(X)
        
        split_idx = int(len(X_scaled) * 0.8)
        X_train_scaled = X_scaled[:split_idx]
        X_test_scaled = X_scaled[split_idx:]
        y_test = y[split_idx:]
        X_test = X[split_idx:]
        close_prices_test = close_prices[split_idx:]
        
        # Get test features for ADX filter
        test_features = X_test
        
        # 3-Class predictions with confidence threshold
        probas = self.model.predict_proba(X_test_scaled)
        classes = self.model.classes_
        
        # Apply confidence threshold - trade only when confident
        confidence_threshold = 0.40  # lower - more trades
        predictions = np.zeros(len(probas), dtype=int)
        
        for i in range(len(probas)):
            max_prob_idx = np.argmax(probas[i])
            max_prob = probas[i, max_prob_idx]
            
            if max_prob >= confidence_threshold:
                predictions[i] = classes[max_prob_idx]
            else:
                predictions[i] = 0  # Neutral if not confident
        
        max_len = len(predictions) - self.forward_window
        
        # Simulation parameters
        tp_pct = 0.015  # TP 1.5% for H1
        sl_pct = 0.008  # SL 0.8% for H1
        max_bars = 10  # Max holding bars
        
        bh_returns = np.zeros(max_len)
        strategy_returns = np.zeros(max_len)
        
        for i in range(max_len):
            current_price = close_prices_test[i]
            entry_price = current_price
            
            # Buy & Hold return (forward_window bars)
            future_idx = min(i + self.forward_window, len(close_prices_test) - 1)
            future_price = close_prices_test[future_idx]
            bh_returns[i] = (future_price - current_price) / current_price
            
            # ML Strategy with real TP/SL
            signal = predictions[i]
            if signal == 0:  # Neutral
                strategy_returns[i] = 0
                continue
            
            # Find TP/SL hit
            tp_price = entry_price * (1 + tp_pct) if signal == 1 else entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 - sl_pct) if signal == 1 else entry_price * (1 + sl_pct)
            
            hit_tp = False
            hit_sl = False
            
            for j in range(1, max_bars + 1):
                if i + j >= len(close_prices_test):
                    break
                
                high = close_prices_test[i + j]
                low = close_prices_test[i + j]
                
                if signal == 1:  # Buy
                    if high >= tp_price:
                        hit_tp = True
                        break
                    if low <= sl_price:
                        hit_sl = True
                        break
                else:  # Sell
                    if low <= tp_price:
                        hit_tp = True
                        break
                    if high >= sl_price:
                        hit_sl = True
                        break
            
            if hit_tp and not hit_sl:
                strategy_returns[i] = tp_pct
            elif hit_sl:
                strategy_returns[i] = -sl_pct
            else:
                # No TP/SL hit within max_bars, use price change
                exit_idx = min(i + max_bars, len(close_prices_test) - 1)
                exit_price = close_prices_test[exit_idx]
                if signal == 1:
                    strategy_returns[i] = (exit_price - entry_price) / entry_price
                else:
                    strategy_returns[i] = (entry_price - exit_price) / entry_price
        
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
        
        # plt.tight_layout()
        # plt.savefig('simulation_result.png', dpi=150)
        # plt.show()
        
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
    
    symbols = ["XAUUSDm","BTCUSDm"]
    for symbol in symbols:
        ml = MLTradingModel()

        print("="*60)
        print(f"📊 กำลังฝึกโมเดลสำหรับ {symbol}")
        print("="*60)

        ml.train(symbol, mt5.TIMEFRAME_H1, num_bars=30000)

        print("\n📈 Feature Importance:")
        ml.get_feature_importance()

        print("\n🔮 ทดสอบการทำนาย:")
        signal = ml.predict(symbol, mt5.TIMEFRAME_H1)
        print(f"   Signal: {signal}")

        print("\n🎮 Running Simulation...")
        ml.simulate(symbol, mt5.TIMEFRAME_H1, num_bars=2000)
    
    mt5.shutdown()