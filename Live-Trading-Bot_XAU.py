import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import time
import talib as ta
import os
from datetime import datetime
from ml_model_XAU import MLTradingModel  # นำเข้า Class จากไฟล์เดิม

# ==========================================
# CONFIGURATION (V11.2 Final Implementation)
# ==========================================
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M15
MODEL_PATH = "models/ml_model_XAU.pkl"

# Risk Management
LOT_SIZE = 0.01
CONFIDENCE_LEVEL = 0.35
STOP_LOSS_ATR = 2.0  # SL เป็น 2 เท่าของ ATR เพื่อลดการโดนสะบัด
TAKE_PROFIT_ATR = 2.0  # TP เป็น 4 เท่าของ ATR (RR 1:2)
USE_TRAILING = True  # เปิดใช้ Trailing Stop เพื่อล็อคกำไรแทนการปิดมั่ว
TRAILING_STEP_ATR = 1.0  # เลื่อน SL เมื่อกำไรไปแล้ว 1 เท่าของ ATR


class SentinelLiveBot:
    def __init__(self, model_path):
        # โหลดโมเดลและเครื่องมือที่เกี่ยวข้อง
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")

        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.ml_logic = MLTradingModel()

        print(f"📦 Loaded Model Features: {self.feature_names}")
        self.magic_number = 20241122

    def get_latest_data(self, symbol, timeframe, n_bars=300):
        """ดึงข้อมูลราคาล่าสุดจาก MT5"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None:
            print(f"❌ Cannot get rates for {symbol}")
            return None
        df = pd.DataFrame(rates)
        return df

    def check_existing_positions(self, symbol):
        """ตรวจสอบสถานะออเดอร์ปัจจุบัน"""
        positions = mt5.positions_get(symbol=symbol, magic=self.magic_number)
        if positions is None or len(positions) == 0:
            return 0, None

        # คืนค่าทิศทางและข้อมูลออเดอร์
        pos = positions[0]
        direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
        return direction, pos

    def manage_trailing_stop(self, position, current_price, atr):
        """ระบบ Trailing Stop ตามค่าความผันผวน (ATR)"""
        if not USE_TRAILING:
            return

        ticket = position.ticket
        current_sl = position.sl
        pos_type = position.type

        # คำนวณระยะ Trailing
        trailing_dist = atr * 1.5

        if pos_type == mt5.ORDER_TYPE_BUY:
            new_sl = round(current_price - trailing_dist, 3)
            # ขยับ SL ขึ้นเท่านั้น
            if new_sl > current_sl + (atr * 0.2):
                self.modify_sl(ticket, new_sl, position.tp)

        elif pos_type == mt5.ORDER_TYPE_SELL:
            new_sl = round(current_price + trailing_dist, 3)
            # ขยับ SL ลงเท่านั้น
            if current_sl == 0 or new_sl < current_sl - (atr * 0.2):
                self.modify_sl(ticket, new_sl, position.tp)

    def modify_sl(self, ticket, new_sl, tp):
        """แก้ไขราคา Stop Loss"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "operation": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "tp": tp
        }
        # result = mt5.order_send(request)
        # if result.retcode == mt5.TRADE_RETCODE_DONE:
        #     print(f"\n🔄 Trailing Stop Updated for #{ticket} to {new_sl}")

    def send_order(self, symbol, order_type, lot, sl_price=0, tp_price=0):
        """ส่งคำสั่งซื้อขายที่แม่นยำ"""
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": round(sl_price, 3),
            "tp": round(tp_price, 3),
            "magic": self.magic_number,
            "comment": "V11.2 AI Sentinel",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"\n❌ Order Failed: {result.retcode} - {result.comment}")
        else:
            print(f"\n✅ AI SIGNAL EXECUTED: {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'} at {price}")

    def run(self):
        print(f"🚀 Sentinel Bot V11.2 (AI Managed) is running for {SYMBOL}...")

        while True:
            # 1. ดึงข้อมูลล่าสุด
            df = self.get_latest_data(SYMBOL, TIMEFRAME)
            if df is not None:
                # 2. สร้าง Features และ Predict
                df_features = self.ml_logic.create_features(df)
                last_row = df_features[self.feature_names].tail(1)

                X_scaled = self.scaler.transform(last_row.values)
                probs = self.model.predict_proba(X_scaled)
                max_prob = np.max(probs)
                prediction = self.model.classes_[np.argmax(probs)]

                # 3. Sentinel Live Filters (กรองด้วย Slope และ ADX)
                current_slope = last_row['slope_20'].iloc[0]
                current_adx = last_row['adx'].iloc[0]
                slope_limit = 0.00012 if current_adx > 25 else 0.00006

                signal = 0
                if prediction == 1 and current_slope > -slope_limit:
                    signal = 1
                elif prediction == -1 and current_slope < slope_limit:
                    signal = -1

                # 4. Position & Trailing Management
                current_dir, active_pos = self.check_existing_positions(SYMBOL)
                atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)[-1]

                # ถ้ามีออเดอร์ค้างอยู่ ให้จัดการ Trailing Stop
                if active_pos:
                    self.manage_trailing_stop(active_pos, df['close'].iloc[-1], atr)

                # 5. Execution Logic (เข้าออเดอร์เมื่อมั่นใจ และไม่มีออเดอร์ฝั่งเดิม)
                if max_prob >= CONFIDENCE_LEVEL:
                    if signal != 0 and signal != current_dir:
                        # ปิดออเดอร์ฝั่งตรงข้ามก่อนถ้ามี (เพื่อทำตาม AI 100%)
                        if current_dir != 0:
                            print(f"\n🔄 AI Reversing Signal: Closing #{active_pos.ticket}")
                            mt5.order_send({
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": SYMBOL,
                                "volume": active_pos.volume,
                                "type": mt5.ORDER_TYPE_SELL if active_pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                "position": active_pos.ticket,
                                "price": mt5.symbol_info_tick(
                                    SYMBOL).bid if active_pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(
                                    SYMBOL).ask,
                                "magic": self.magic_number,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            })

                        current_price = df['close'].iloc[-1]
                        if signal == 1:
                            sl = current_price - (atr * STOP_LOSS_ATR)
                            tp = current_price + (atr * TAKE_PROFIT_ATR)
                            self.send_order(SYMBOL, mt5.ORDER_TYPE_BUY, LOT_SIZE, sl, tp)
                        elif signal == -1:
                            sl = current_price + (atr * STOP_LOSS_ATR)
                            tp = current_price - (atr * TAKE_PROFIT_ATR)
                            self.send_order(SYMBOL, mt5.ORDER_TYPE_SELL, LOT_SIZE, sl, tp)

                status_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Price: {df['close'].iloc[-1]:.2f} | Conf: {max_prob:.2%} | Sig: {signal} | Pos: {current_dir}"
                print(f"\r{status_msg}", end='', flush=True)

            time.sleep(1)  # เช็คทุก 30 วินาทีเพื่อความไวในการทำ Trailing


if __name__ == '__main__':
    if mt5.initialize():
        try:
            bot = SentinelLiveBot(MODEL_PATH)
            bot.run()
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            mt5.shutdown()
    else:
        print("❌ Failed to initialize MT5")