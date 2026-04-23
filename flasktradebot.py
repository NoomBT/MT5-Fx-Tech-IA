from flask import Flask, jsonify, request, render_template
import MetaTrader5 as mt5
import threading
import time
import pandas as pd
import talib as ta
import random
from datetime import datetime
import joblib
import os
import numpy as np

# ==========================================
# 1. การตั้งค่าตัวแปร Global & Flask App
# ==========================================
app = Flask(__name__)

# สถานะของบอทและคอนฟิก
bot_status = {
    "is_running": False,
    "symbols": ["XAUUSDm"],
    "initial_lot": 0.01,      # Lot เริ่มต้น
    "max_lot": 1.0,         # Lot สูงสุด
    "martingale_mult": 1.5,   # Martingale Multiplier
    "max_risk_percent": 10,     # ความเสี่ยงสูงสุด 10% ของ Balance
    
    # 📌 TP/SL Settings
    "sl_points": 0,          # Stop Loss (0 = ไม่ใช้)
    "tp_points": 0,           # Take Profit (0 = ไม่ใช้)
    "use_trailing": False,    # ใช้ Trailing Stop
    "trailing_distance": 0,   # Trailing Stop Distance (points)
    "trailing_step": 10,      # Trailing Step (points)
    
    # 💎 Close All Profit Settings
    "close_all_percent": 0.1,   # ปิดเมื่อกำไร >= 0.1% ของ Balance
    "close_all_dollar": 0,      # ปิดเมื่อกำไร >= $ (0 = ไม่ใช้)
    "close_all_pullback": 0,    # ปิดเมื่อกำไรลดลง % จากสูงสุด (0 = ไม่ใช้)
    
    "max_per_side": 1,
    "use_sl_tp": True,
    "close_profit": 10,
    "close_loss": 20,
    "use_ai": True,
    "use_bb_ma_macd": False,  # ใช้ BB+MA+MACD Strategy
    "ai_model_path": "ml_model.pkl",
    "use_gpu_model": False,     # ใช้ GPU Model (ml_model_gpu.pkl)
    "timeframe": mt5.TIMEFRAME_M5,
    "use_trend_filter": True,
    "trend_timeframe": mt5.TIMEFRAME_H1,
    
    # 📊 Grid Trading Settings
    "use_grid": False,        # เปิด/ปิด Grid Trading
    "grid_distance": 300,    # ระยะห่างระหว่าง Grid (points)
    "max_grid_orders": 10,   # จำนวน Grid สูงสุด
    "grid_buy_start": 0,     # ราคาเริ่มต้น BUY grid
    "grid_sell_start": 0     # ราคาเริ่มต้น SELL grid
}

# ติดตามจำนวนออเดอร์ที่ขาดทุนติดต่อกัน (สำหรับ Martingale)
consecutive_losses = {}

# ติดตามราคาเปิดออเดอร์ล่าสุดของแต่ละ symbol
last_order_prices = {}

# ติดตาม Max Profit และ Trailing SL
max_profit_seen = {}  # {symbol: max_profit}
trailing_sl_levels = {}  # {ticket: sl_price}

# เก็บประวัติการทำงาน (Logs)
logs = []

# ML Model
ml_model = None

# Timeframe mapping
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}


def get_timeframe(tf_str):
    return TIMEFRAME_MAP.get(tf_str, mt5.TIMEFRAME_M5)


# Log file handler
log_file = None

def setup_log_file():
    """สร้างไฟล์ log ใหม่"""
    global log_file
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # สร้างชื่อไฟล์ตามวันที่
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"trade_{date_str}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    log_file = open(log_path, 'w', encoding='utf-8')
    log_file.write(f"=== Trading Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_file.flush()


def write_log_to_file(msg):
    """เขียน log ลงไฟล์"""
    global log_file
    if log_file:
        try:
            log_file.write(msg + "\n")
            log_file.flush()
        except:
            pass


def close_log_file():
    """ปิดไฟล์ log"""
    global log_file
    if log_file:
        log_file.write(f"=== Trading Log Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.close()
        log_file = None


def add_log(msg):
    time_str = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{time_str}] {msg}"
    logs.insert(0, log_msg)
    if len(logs) > 50:
        logs.pop()
    
    # เขียนลงไฟล์ด้วย
    write_log_to_file(log_msg)


def load_ml_model():
    """โหลด ML Model (GPU หรือ CPU)"""
    global ml_model
    
    if bot_status.get("use_gpu_model", False):
        model_path = bot_status.get("ai_model_path", "ml_model_gpu.pkl")
        model_module = "ml_model_gpu"
        model_type = "GPU"
    else:
        model_path = bot_status.get("ai_model_path", "ml_model.pkl")
        model_module = "ml_model"
        model_type = "CPU"
    
    if os.path.exists(model_path):
        try:
            data = joblib.load(model_path)
            ml_model = data
            add_log(f"✅ โหลด AI Model ({model_type}) สำเร็จ: {model_path}")
            if 'feature_names' in data:
                add_log(f"📊 Features: {len(data['feature_names'])} features")
            if 'prob_threshold' in data:
                add_log(f"📊 Threshold: {data['prob_threshold']:.3f}")
            return True
        except Exception as e:
            add_log(f"❌ โหลด AI Model ล้มเหลว: {e}")
    else:
        add_log(f"⚠️ ไม่พบไฟล์โมเดล: {model_path}")
    return False


# ==========================================
# 💰 ระบบบริหารเงิน (Money Management)
# ==========================================
def calculate_lot_size(symbol, strategy="martingale", consecutive_loss=0):
    initial_lot = bot_status.get("initial_lot", 0.01)
    max_lot = bot_status.get("max_lot", 1.0)
    martingale_mult = bot_status.get("martingale_mult", 1.5)
    
    if strategy == "martingale":
        lot_size = initial_lot * (martingale_mult ** consecutive_loss)
    else:
        lot_size = initial_lot
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        lot_size = min(lot_size, max_lot)
        lot_size = max(lot_size, symbol_info.volume_min)
        lot_size = min(lot_size, symbol_info.volume_max)
    
    return round(lot_size, 2)


def check_max_risk(symbol):
    max_risk_percent = bot_status.get("max_risk_percent", 10)
    account_info = mt5.account_info()
    if account_info is None:
        return True
    
    balance = account_info.balance
    equity = account_info.equity
    max_risk_amount = balance * (max_risk_percent / 100)
    floating_loss = balance - equity
    
    if floating_loss >= max_risk_amount:
        add_log(f"🛑 หยุดเทรด: ขาดทุนลอย ${floating_loss:.2f} >= ${max_risk_amount:.2f}")
        return True
    return False


def get_consecutive_loss_count(symbol):
    global consecutive_losses
    return consecutive_losses.get(symbol, 0)


def increment_consecutive_loss(symbol):
    global consecutive_losses
    if symbol not in consecutive_losses:
        consecutive_losses[symbol] = 0
    consecutive_losses[symbol] += 1


def reset_consecutive_loss(symbol):
    global consecutive_losses
    consecutive_losses[symbol] = 0


last_order_prices = {}
max_profit_seen = {}
trailing_sl_levels = {}


# ==========================================
# 📌 Trailing Stop & Close All Profit
# ==========================================
def update_trailing_stop(symbol):
    if not bot_status.get("use_trailing", False):
        return
    
    trailing_dist = bot_status.get("trailing_distance", 0)
    if trailing_dist <= 0:
        return
    
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return
    
    point = mt5.symbol_info(symbol).point
    
    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        current_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        
        if pos.type == mt5.ORDER_TYPE_BUY:
            price_diff = current_price - pos.price_open
        else:
            price_diff = pos.price_open - current_price
        
        points_diff = int(price_diff / point)
        
        if points_diff >= trailing_dist:
            if pos.type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - (trailing_dist * point)
            else:
                new_sl = current_price + (trailing_dist * point)
            
            if pos.sl == 0 or ((pos.type == mt5.ORDER_TYPE_BUY and new_sl > pos.sl) or (pos.type == mt5.ORDER_TYPE_SELL and new_sl < pos.sl)):
                digits = mt5.symbol_info(symbol).digits
                new_tp = round(pos.tp, digits) if pos.tp > 0 else 0
                request = {"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": pos.ticket, "sl": round(new_sl, digits), "tp": new_tp}
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    add_log(f"📈 {symbol} Trailing SL: {new_sl}")


def check_close_all_profit(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return False, ""
    
    account_info = mt5.account_info()
    if account_info is None:
        return False, ""
    
    balance = account_info.balance
    equity = account_info.equity
    floating_profit = equity - balance
    
    close_percent = bot_status.get("close_all_percent", 0.1)
    if close_percent > 0:
        threshold = balance * (close_percent / 100)
        if floating_profit >= threshold:
            return True, f"กำไร {floating_profit:.2f} >= {threshold:.2f}"
    
    close_dollar = bot_status.get("close_all_dollar", 0)
    if close_dollar > 0 and floating_profit >= close_dollar:
        return True, f"กำไร {floating_profit:.2f} >= ${close_dollar}"
    
    global max_profit_seen
    if symbol not in max_profit_seen:
        max_profit_seen[symbol] = floating_profit
    if floating_profit > max_profit_seen[symbol]:
        max_profit_seen[symbol] = floating_profit
    
    close_pullback = bot_status.get("close_all_pullback", 0)
    if close_pullback > 0 and max_profit_seen[symbol] > 0:
        pullback = ((max_profit_seen[symbol] - floating_profit) / max_profit_seen[symbol]) * 100
        if pullback >= close_pullback:
            return True, f"Pullback {pullback:.1f}%"
    
    return False, ""


def close_all_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return 0
    closed = 0
    for pos in positions:
        if close_position(pos.ticket, pos.symbol, pos.volume):
            closed += 1
    global max_profit_seen
    max_profit_seen[symbol] = 0
    return closed


def create_features(df):
    """สร้าง Features จากข้อมูลราคา"""
    df = df.copy()
    df['sma_10'] = ta.SMA(df['close'].values, timeperiod=10)
    df['sma_30'] = ta.SMA(df['close'].values, timeperiod=30)
    df['rsi_14'] = ta.RSI(df['close'].values, timeperiod=14)
    df['price_change'] = df['close'].pct_change()
    df = df.dropna()
    return df


def get_ai_signal(symbol, timeframe=None):
    """ทำนายสัญญาณซื้อ/ขายด้วย ML Model (3-Class: Buy, Neutral, Sell)"""
    global ml_model
    
    if timeframe is None:
        timeframe = bot_status.get("timeframe", mt5.TIMEFRAME_M5)
    
    if ml_model is None:
        if not load_ml_model():
            return "NEUTRAL"
    
    try:
        model = ml_model.get('model') if isinstance(ml_model, dict) else ml_model
        scaler = ml_model.get('scaler') if isinstance(ml_model, dict) else None
        feature_names = ml_model.get('feature_names') if isinstance(ml_model, dict) else None
        label_encoder = ml_model.get('label_encoder') if isinstance(ml_model, dict) else None
        prob_threshold = ml_model.get('prob_threshold', 0.30) if isinstance(ml_model, dict) else 0.30
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 60)
        if rates is None or len(rates) < 50:
            return "NEUTRAL"
        
        df = pd.DataFrame(rates)
        
        from ml_model_gpu import MLTradingModel
        ml = MLTradingModel()
        ml.model = model
        ml.scaler = scaler
        ml.feature_names = feature_names
        ml.label_encoder = label_encoder
        ml.prob_threshold = prob_threshold
        
        signal = ml.predict(symbol, timeframe)
        return signal
    except Exception as e:
        add_log(f"⚠️ AI Signal Error: {e}")
        return "NEUTRAL"


# ==========================================
# 2. ฟังก์ชันหลักสำหรับเทรด (Trading Logic) + Indicator
# ==========================================
def is_sideway_market(symbol, timeframe=None):
    """ตรวจสอบว่าตลาดอยู่ในช่วง Sideway (ไม่ควรเทรด)"""
    if timeframe is None:
        timeframe = bot_status.get("timeframe", mt5.TIMEFRAME_M5)
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    if rates is None or len(rates) < 100:
        return False
    
    df = pd.DataFrame(rates)
    close_prices = df['close'].values
    
    # EMA 9 และ 21
    ema_fast = ta.EMA(close_prices, timeperiod=9)
    ema_slow = ta.EMA(close_prices, timeperiod=21)
    
    # หาค่า EMA ที่ห่างกัน (percentage)
    ema_diff = abs(ema_fast[-1] - ema_slow[-1]) / ema_slow[-1] * 100
    
    # MACD
    macd, signal, macdhist = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_value = macd[-1]
    signal_value = signal[-1]
    
    # ตรวจสอบ Sideway:
    # 1. EMA สองเส้นใกล้กันมาก (น้อยกว่า 0.1%)
    # 2. MACD อยู่ใกล้ Zero Line (น้อยกว่า 0.0005)
    is_sideway = ema_diff < 0.1 and abs(macd_value) < 0.0005
    
    return is_sideway


def get_macd_signal(symbol, timeframe=None, fastperiod=12, slowperiod=26, signalperiod=9):
    """ฟังก์ชันคำนวณ MACD และส่งสัญญาณ"""
    if timeframe is None:
        timeframe = bot_status.get("timeframe", mt5.TIMEFRAME_M5)
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    if rates is None or len(rates) < 100:
        return None
    
    df = pd.DataFrame(rates)
    close_prices = df['close'].values
    
    macd, signal, macdhist = ta.MACD(close_prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    
    current_macd = macd[-1]
    current_signal = signal[-1]
    prev_macd = macd[-2]
    prev_signal = signal[-2]
    
    if prev_macd <= prev_signal and current_macd > current_signal:
        return "BUY"
    elif prev_macd >= prev_signal and current_macd < current_signal:
        return "SELL"
    elif current_macd > current_signal:
        return "BULLISH"
    elif current_macd < current_signal:
        return "BEARISH"
    
    return "NEUTRAL"


def get_trend_ema200(symbol, timeframe=mt5.TIMEFRAME_H1):
    """ตรวจสอบแนวโน้มใหญ่จาก EMA 200 ใน timeframe ที่ใหญ่กว่า"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
    if rates is None or len(rates) < 200:
        return "NEUTRAL"
    
    df = pd.DataFrame(rates)
    close_prices = df['close'].values
    ema200 = ta.EMA(close_prices, timeperiod=200)
    
    current_price = close_prices[-1]
    current_ema200 = ema200[-1]
    
    tf_name = "H1"
    if timeframe == mt5.TIMEFRAME_H4:
        tf_name = "H4"
    elif timeframe == mt5.TIMEFRAME_D1:
        tf_name = "D1"
    
    if current_price > current_ema200:
        return "BULLISH", f"({tf_name} price > EMA200)"
    elif current_price < current_ema200:
        return "BEARISH", f"({tf_name} price < EMA200)"
    
    return "NEUTRAL", f"({tf_name} price ≈ EMA200)"


# ==========================================
# 📊 ระบบ Grid Trading
# ==========================================
def get_last_order_price(symbol, order_type="BUY"):
    """ดึงราคาเปิดออเดอร์ล่าสุด"""
    global last_order_prices
    key = f"{symbol}_{order_type}"
    return last_order_prices.get(key, 0)


def set_last_order_price(symbol, order_type, price):
    """บันทึกราคาเปิดออเดอร์ล่าสุด"""
    global last_order_prices
    key = f"{symbol}_{order_type}"
    last_order_prices[key] = price


grid_count = 0


def check_grid_condition(symbol, current_price, order_type):
    global grid_count
    """ตรวจสอบว่าถึงเวลาเปิด Grid ใหม่หรือยัง"""
    if not bot_status.get("use_grid", False):
        grid_count = 0
        return False, 0
    last_price = 0
    grid_distance = bot_status.get("grid_distance", 300)
    max_grid = bot_status.get("max_grid_orders", 10)
    
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        positions = []

    
    type_enum = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
    
    for pos in positions:
        grid_count += 1
        if pos.type == type_enum:
            if order_type == "BUY":
                if last_price == 0 or pos.price_open < last_price:
                    last_price = pos.price_open
            else:
                if last_price == 0 or pos.price_open > last_price:
                    last_price = pos.price_open
    
    if grid_count >= max_grid:
        return False, 0
    
    if last_price == 0:
        return True, current_price
    
    point = mt5.symbol_info(symbol).point
    price_distance = abs(current_price - last_price)
    points_distance = price_distance / point
    
    should_grid = points_distance >= grid_distance
    
    if should_grid:
        add_log(f"📊 {symbol} Grid Check: Price={current_price}, Last={last_price}, Dist={points_distance:.0f}pts")
    
    return should_grid, last_price


def open_grid_order(symbol, action, price_reference):
    """เปิด Grid Order ด้วย Martingale"""
    if check_max_risk(symbol):
        add_log(f"🛑 {symbol} ถึงขีดจำกัดความเสี่ยง หยุดเปิด Grid")
        return False
    
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        type_enum = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
        grid_count = sum(1 for p in positions if p.type == type_enum)
    else:
        grid_count = 0
    
    lot_size = calculate_lot_size(symbol, "martingale", grid_count)
    if grid_count: grid_count=0
    add_log(f"📊 {symbol} เปิด Grid #{grid_count + 1}: {action.upper()}, Lot: {lot_size}")
    
    if bot_status.get("use_sl_tp", True):
        open_trade(
            symbol=symbol,
            action=action,
            lot=lot_size,
            sl_points=bot_status["sl_points"],
            tp_points=bot_status["tp_points"]
        )
    else:
        open_trade_no_sl_tp(
            symbol=symbol,
            action=action,
            lot=lot_size
        )
    
    return True


def get_ema_signal(symbol, timeframe=None, fast_period=12, slow_period=26):
    """ฟังก์ชันคำนวณ EMA และส่งสัญญาณ
    Fast EMA: 12 periods
    Slow EMA: 26 periods
    """
    if timeframe is None:
        timeframe = bot_status.get("timeframe", mt5.TIMEFRAME_M5)
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    if rates is None or len(rates) < 100:
        return "NEUTRAL"
    
    df = pd.DataFrame(rates)
    close_prices = df['close'].values
    
    ema_fast = ta.EMA(close_prices, timeperiod=fast_period)
    ema_slow = ta.EMA(close_prices, timeperiod=slow_period)
    
    current_fast = ema_fast[-1]
    current_slow = ema_slow[-1]
    prev_fast = ema_fast[-2]
    prev_slow = ema_slow[-2]
    
    # Crossover signals
    if prev_fast <= prev_slow and current_fast > current_slow:
        return "BULLISH"  # EMA fast ตัดขึ้น
    elif prev_fast >= prev_slow and current_fast < current_slow:
        return "BEARISH"  # EMA fast ตัดลง
    # Trend signals
    elif current_fast > current_slow:
        return "BULLISH"
    elif current_fast < current_slow:
        return "BEARISH"
    
    return "NEUTRAL"


def get_bb_ma_macd_signal(symbol, timeframe=None):
    """Bollinger Bands + Smoothed MA + MACD Strategy
    Bollinger: Period 20, Deviation 2
    MA: Smoothed Period 2
    MACD: Fast EMA 11, Slow EMA 27, MACD SMA 4
    
    Buy: MA crosses above BB middle from below + MACD below or below histogram
    Sell: MA crosses below BB middle from above + MACD above or above histogram
    """
    if timeframe is None:
        timeframe = bot_status.get("timeframe", mt5.TIMEFRAME_M5)
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    if rates is None or len(rates) < 50:
        return "NEUTRAL", ""
    
    df = pd.DataFrame(rates)
    close = df['close'].values
    
    bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    
    ma = ta.MA(close, timeperiod=2, matype=1)
    
    macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=11, slowperiod=27, signalperiod=4)
    
    ma_prev = ma[-2]
    ma_curr = ma[-1]
    bb_mid_prev = bb_middle[-2]
    bb_mid_curr = bb_middle[-1]
    macd_curr = macd[-1]
    macd_hist_curr = macd_hist[-1]
    macd_hist_prev = macd_hist[-2]
    
    ma_above_bb = ma_curr > bb_mid_curr
    ma_was_below_bb = ma_prev <= bb_mid_prev
    
    ma_below_bb = ma_curr < bb_mid_curr
    ma_was_above_bb = ma_prev >= bb_mid_prev
    
    macd_below_hist = macd_curr < macd_hist_curr
    macd_above_hist = macd_curr > macd_hist_curr
    
    buy_condition = (ma_was_below_bb and ma_above_bb) and (macd_below_hist or macd_curr < macd_hist_prev)
    sell_condition = (ma_was_above_bb and ma_below_bb) and (macd_above_hist or macd_curr > macd_hist_prev)
    
    if buy_condition:
        return "BULLISH", f"MA>{BB}>BB_mid, MACD<Hist"
    elif sell_condition:
        return "BEARISH", f"MA<BB_mid, MACD>Hist"
    
    return "NEUTRAL", f"MA:{ma_curr:.5f} BB:{bb_mid_curr:.5f}"


def get_rsi_signal(symbol, timeframe=None, period=14, buy_level=60, sell_level=40):
    """ฟังก์ชันคำนวณ RSI และส่งสัญญาณ
    Period: 14 periods
    Buy Level: 60 (เหนือระดับนี้ = แรงซื้อ)
    Sell Level: 40 (ต่ำกว่าระดับนี้ = แรงขาย)
    """
    if timeframe is None:
        timeframe = bot_status.get("timeframe", mt5.TIMEFRAME_M5)
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    if rates is None or len(rates) < 100:
        return "NEUTRAL"
    
    df = pd.DataFrame(rates)
    close_prices = df['close'].values
    
    rsi = ta.RSI(close_prices, timeperiod=period)
    current_rsi = rsi[-1]
    
    if current_rsi > buy_level:
        return "BULLISH"  # แรงซื้อ (RSI > 60)
    elif current_rsi < sell_level:
        return "BEARISH"  # แรงขาย (RSI < 40)
    
    return "NEUTRAL"


def open_trade(symbol, action, lot, sl_points, tp_points):
    """ฟังก์ชันสำหรับส่งคำสั่งซื้อขายไปยัง MT5 (มี SL/TP)"""
    # ตรวจสอบการเชื่อมต่อ MT5
    terminal = mt5.terminal_info()
    if terminal is None or not terminal.connected:
        add_log(f"⚠️ ไม่มีการเชื่อมต่อ MT5 (connected={terminal.connected if terminal else 'None'})")
        return
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        add_log(f"❌ ไม่พบ symbol {symbol}")
        return
    
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)
        time.sleep(1)
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        add_log(f"❌ ไม่พบ tick สำหรับ {symbol}")
        return
    
    point = symbol_info.point
    price = tick.ask if action == 'buy' else tick.bid
    
    # ตรวจสอบ Balance และคำนวณ lot size ที่เป็นไปได้
    account_info = mt5.account_info()
    if account_info is None:
        add_log(f"❌ ไม่พบข้อมูลบัญชี")
        return
    
    balance = account_info.balance
    free_margin = account_info.margin_free
    
    add_log(f"💰 Balance: {balance:.2f}, Free Margin: {free_margin:.2f}")
    
    # ใช้ lot ตามที่กำหนด ถ้าน้อยกว่า max ของ symbol
    valid_lot = min(lot, symbol_info.volume_max)
    valid_lot = max(valid_lot, symbol_info.volume_min)
    
    if valid_lot < symbol_info.volume_min:
        add_log(f"❌ Lot size น้อยเกินไป (min: {symbol_info.volume_min}, ได้: {valid_lot:.2f})")
        return
    
    add_log(f"📝 เตรียมเปิด {action.upper()} {symbol} lot={valid_lot}")
    
    if action == 'buy':
        order_type = mt5.ORDER_TYPE_BUY
        sl = price - (sl_points * point * 100)
        tp = price + (tp_points * point * 100)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        sl = price + (sl_points * point * 100)
        tp = price - (tp_points * point * 100)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(valid_lot),
        "type": order_type,
        "price": price,
        "sl": round(sl, symbol_info.digits) if sl_points > 0 else 0,
        "tp": round(tp, symbol_info.digits) if tp_points > 0 else 0,
        "deviation": 50,
        "magic": 123456,
        "comment": "AutoWebBot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
       }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        add_log(f"❌ ส่งคำสั่ง {action.upper()} {symbol} ล้มเหลว: {result.comment} (retcode: {result.retcode})")
    else:
        add_log(f"✅ เปิดออเดอร์ {action.upper()} {symbol} สำเร็จที่ราคา {price}")


def open_trade_no_sl_tp(symbol, action, lot):
    """ฟังก์ชันส่งคำสั่งซื้อขายโดยไม่มี SL และ TP"""
    # ตรวจสอบการเชื่อมต่อ MT5
    terminal = mt5.terminal_info()
    if terminal is None or not terminal.connected:
        add_log(f"⚠️ ไม่มีการเชื่อมต่อ MT5")
        return None
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        add_log(f"❌ ไม่พบ symbol {symbol}")
        return None
    
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)
        time.sleep(1)
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        add_log(f"❌ ไม่พบ tick สำหรับ {symbol}")
        return None
    
    # ตรวจสอบ Balance และคำนวณ lot size
    account_info = mt5.account_info()
    if account_info is None:
        add_log(f"❌ ไม่พบข้อมูลบัญชี")
        return None
    
    balance = account_info.balance
    free_margin = account_info.margin_free
    
    # ใช้ lot ตามที่กำหนด
    valid_lot = min(lot, symbol_info.volume_max)
    valid_lot = max(valid_lot, symbol_info.volume_min)
    
    if valid_lot < symbol_info.volume_min:
        add_log(f"❌ Lot size น้อยเกินไป (min: {symbol_info.volume_min}, ได้: {valid_lot:.2f})")
        return None
    
    price = tick.ask if action == 'buy' else tick.bid
    add_log(f"📝 เตรียมเปิด {action.upper()} {symbol} lot={valid_lot}")
    
    order_type = mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(valid_lot),
        "type": order_type,
        "price": price,
        "deviation": 50,
        "magic": 123456,
        "comment": "AutoWebBot_NoSLTP",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        add_log(f"❌ ส่งคำสั่ง {action.upper()} {symbol} (No SL/TP) ล้มเหลว: {result.comment} (retcode: {result.retcode})")
        return None
    else:
        add_log(f"✅ เปิดออเดอร์ {action.upper()} {symbol} (No SL/TP) สำเร็จที่ราคา {price}")
        return result.order


def close_position(ticket, symbol, volume):
    """ฟังก์ชันสำหรับปิด position"""
    positions = mt5.positions_get(ticket=ticket)
    if positions is None or len(positions) == 0:
        add_log(f"⚠️ ไม่พบ position {ticket}")
        return False
    
    position = positions[0]
    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    # pos_type = position.type
    #
    # # เตรียมราคาและฝั่งที่จะปิด
    # if pos_type == mt5.POSITION_TYPE_BUY:
    #     order_type = mt5.ORDER_TYPE_SELL
    #     price = mt5.symbol_info_tick(symbol).bid
    # else:
    #     order_type = mt5.ORDER_TYPE_BUY
    #     price = mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "position": ticket,  # สำคัญมาก: ต้องมีอันนี้เพื่อบอกว่าปิดออเดอร์ไหน
        "price": price,
        "deviation": 50,
        "magic": 123456,
        "comment": f"Close_{ticket}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, # แนะนำใช้ IOC เผื่อ Volume ไม่พอในราคานั้น
    }

    result = mt5.order_send(request)


    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        add_log(f"❌ ปิด position {ticket} ล้มเหลว! Code: {result.retcode}")
        return False
    else:
        profit = position.profit
        add_log(f"✅ ปิด position {ticket}-{symbol} สำเร็จ (กำไร/ขาดทุน: {profit:.2f})")
        
        # 💰 อัปเดต consecutive loss counter
        if profit < 0:
            increment_consecutive_loss(symbol)
            add_log(f"📉 {symbol} ขาดทุนติดต่อกัน: {get_consecutive_loss_count(symbol)} ครั้ง")
        else:
            if get_consecutive_loss_count(symbol) > 0:
                add_log(f"📈 {symbol} ได้กำไร รีเซ็ต Loss Streak")
            reset_consecutive_loss(symbol)
        
        return True


def check_and_close_positions():
    """ตรวจสอบ positions และปิดอัตโนมัติเมื่อได้กำไรหรือขาดทุน หรือ Signal เปลี่ยนเทรน"""
    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        return
    
    close_profit = bot_status.get("close_profit", 10)
    close_loss = bot_status.get("close_loss", 20)
    add_log(f"📋 ตรวจสอบการปิดออเดอร์: profit={close_profit}, loss={close_loss}")
    
    for pos in positions:
        profit = pos.profit
        should_close = False
        close_reason = ""
        pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        
        add_log(f"📊 Position - #{pos.ticket}-{pos.symbol} {pos_type} กำไร/ขาดทุน: {profit:.2f}")

        if profit >= close_profit:
            should_close = True
            close_reason = f"ได้กำไร {profit:.2f} >= {close_profit}"
        elif profit <= -close_loss:
            should_close = True
            close_reason = f"ขาดทุน {profit:.2f} <= -{close_loss}"
        
        # ปิดเมื่อ Signal เปลี่ยนเทรน (แต่ต้องไม่ขาดทุน)
        if not should_close and profit > 0:
            if bot_status.get("use_ai", True):
                signal = get_ai_signal(pos.symbol)
            else:
                # ใช้ EMA + MACD + RSI
                ema_signal = get_ema_signal(pos.symbol)
                macd_signal = get_macd_signal(pos.symbol)
                rsi_signal = get_rsi_signal(pos.symbol)
                
                # ตรวจสอบ MACD > 0 หรือ < 0
                rates = mt5.copy_rates_from_pos(pos.symbol, bot_status.get("timeframe", mt5.TIMEFRAME_M5), 0, 50)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    close_prices = df['close'].values
                    macd, signal_line, macdhist = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                    macd_above_zero = macd[-1] > 0
                    macd_below_zero = macd[-1] < 0
                else:
                    macd_above_zero = False
                    macd_below_zero = False
                
                # BUY: EMA + MACD > 0 + RSI > 60
                # SELL: EMA + MACD < 0 + RSI < 40
                is_bearish = (ema_signal == "BEARISH" and 
                              macd_below_zero and 
                              rsi_signal == "BEARISH")
                is_bullish = (ema_signal == "BULLISH" and 
                              macd_above_zero and 
                              rsi_signal == "BULLISH")
                
                if is_bearish:
                    signal = "BEARISH"
                elif is_bullish:
                    signal = "BULLISH"
                else:
                    signal = "NEUTRAL"
                add_log(f"📊 {pos.symbol} Close Check - EMA:{ema_signal}, MACD:{macd_signal}, RSI:{rsi_signal} -> {signal}")
            
            # 3-Class: BUY, NEUTRAL, SELL
            # BUY position ปิดเมื่อ signal = BEARISH (เทรนลง)
            # SELL position ปิดเมื่อ signal = BULLISH (เทรนขึ้น)
            if pos.type == mt5.ORDER_TYPE_BUY and signal == "BEARISH":
                should_close = True
                close_reason = f"AI Signal เปลี่ยน (Signal:{signal})"
            elif pos.type == mt5.ORDER_TYPE_SELL and signal == "BULLISH":
                should_close = True
                close_reason = f"AI Signal เปลี่ยน (Signal:{signal})"

        if should_close:
            add_log(f"📌 Position #{pos.ticket} {close_reason} ปิดออเดอร์...")
            close_position(pos.ticket, pos.symbol, pos.volume)


def auto_trading_loop():
    """ลูปการทำงานหลักของ Bot (ทำงานเบื้องหลัง)"""
    add_log("🚀 เริ่มต้นระบบเทรดอัตโนมัติ...")
    while bot_status["is_running"]:
        # ตรวจสอบการเชื่อมต่อ MT5
        if not mt5.terminal_info():
            add_log("⚠️ ขาดการเชื่อมต่อกับ MT5 กำลังพยายามเชื่อมต่อใหม่...")
            mt5.initialize()
            time.sleep(5)
            continue

        symbols = bot_status["symbols"]

        for symbol in symbols:
            # ดึงข้อมูลคู่เงินเพื่อตรวจสอบว่าพร้อมเทรดหรือไม่
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                add_log(f"⚠️ ไม่พบข้อมูลคู่เงิน {symbol} ข้ามไปก่อน")
                continue
            if not symbol_info.visible:
                mt5.symbol_select(symbol, True)

            # ดึงออเดอร์ที่เปิดอยู่ทั้งหมดสำหรับคู่เงินนี้
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                positions = []

            # แยกนับจำนวนออเดอร์ Buy และ Sell ปัจจุบัน
            buy_count = sum(1 for p in positions if p.type == mt5.ORDER_TYPE_BUY)
            sell_count = sum(1 for p in positions if p.type == mt5.ORDER_TYPE_SELL)

            # กลยุทธ์: ใช้ AI (Random Forest) หรือ MACD หาจังหวะเข้า order
            actions_to_open = []
            
            # ตรวจสอบแนวโน้มใหญ่ (EMA 200 ใน timeframe ที่ใหญ่กว่า)
            if bot_status.get("use_trend_filter", True):
                trend_signal, trend_desc = get_trend_ema200(symbol, bot_status.get("trend_timeframe", mt5.TIMEFRAME_H1))
                add_log(f"📈 {symbol} Trend: {trend_signal} {trend_desc}")
            
            if bot_status.get("use_ai", True):
                ai_signal = get_ai_signal(symbol)
                add_log(f"🤖 {symbol} AI Signal: {ai_signal}")
                
                max_per_side = bot_status.get("max_per_side", 1)
                if not bot_status.get("use_grid", False):
                    max_per_side = 1
                
                if ai_signal == "BULLISH" and buy_count < max_per_side:
                    actions_to_open.append('buy')
                elif ai_signal == "BEARISH" and sell_count < max_per_side:
                    actions_to_open.append('sell')
            elif bot_status.get("use_bb_ma_macd", False):
                signal, desc = get_bb_ma_macd_signal(symbol)
                add_log(f"📊 {symbol} BB+MA+MACD: {signal} ({desc})")
                
                max_per_side = bot_status.get("max_per_side", 1)
                if not bot_status.get("use_grid", False):
                    max_per_side = 1
                
                if signal == "BULLISH" and buy_count < max_per_side:
                    actions_to_open.append('buy')
                elif signal == "BEARISH" and sell_count < max_per_side:
                    actions_to_open.append('sell')
            else:
                # ตรวจสอบ Sideway Market ก่อนเทรด
                if is_sideway_market(symbol):
                    add_log(f"⏸️ {symbol} ตลาด Sideway - ไม่เทรด")
                else:
                    ema_signal = get_ema_signal(symbol)
                    macd_signal = get_macd_signal(symbol)
                    rsi_signal = get_rsi_signal(symbol)
                    add_log(f"📊 {symbol} EMA: {ema_signal}, MACD: {macd_signal}, RSI: {rsi_signal}")
                    
                    # เงื่อนไขการเทรดใหม่:
                    # BUY: Fast EMA ตัดขึ้นเหนือ Slow EMA + MACD > 0 + RSI > 60
                    # SELL: Fast EMA ตัดลงต่ำกว่า Slow EMA + MACD < 0 + RSI < 40
                    
                    # ตรวจสอบ MACD > 0 หรือ < 0
                    rates = mt5.copy_rates_from_pos(symbol, bot_status.get("timeframe", mt5.TIMEFRAME_M5), 0, 50)
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        close_prices = df['close'].values
                        macd, signal, macdhist = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                        macd_above_zero = macd[-1] > 0
                        macd_below_zero = macd[-1] < 0
                    else:
                        macd_above_zero = False
                        macd_below_zero = False
                    
                    # BUY: EMA Cross Up + MACD > 0 + RSI > 60
                    is_bullish = (ema_signal == "BULLISH" and 
                                  macd_above_zero and 
                                  rsi_signal == "BULLISH")
                    
                    # SELL: EMA Cross Down + MACD < 0 + RSI < 40
                    is_bearish = (ema_signal == "BEARISH" and 
                                  macd_below_zero and 
                                  rsi_signal == "BEARISH")
                    
                    max_per_side = bot_status.get("max_per_side", 1)
                    if not bot_status.get("use_grid", False):
                        max_per_side = 1
                    
                    # ถ้าใช้ trend filter ต้องตรงกับแนวโน้มด้วย
                    if bot_status.get("use_trend_filter", True):
                        if trend_signal == "BULLISH" and is_bullish and buy_count < max_per_side:
                            actions_to_open.append('buy')
                        elif trend_signal == "BEARISH" and is_bearish and sell_count < max_per_side:
                            actions_to_open.append('sell')
                    else:
                        if is_bullish and buy_count < max_per_side:
                            actions_to_open.append('buy')
                        elif is_bearish and sell_count < max_per_side:
                            actions_to_open.append('sell')
            
            # 📊 Grid Trading Logic
            if bot_status.get("use_grid", False):
                # ดึงราคาปัจจุบัน
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    current_price = tick.ask if bot_status.get("last_action", "buy") == "buy" else tick.bid
                    
                    # ตรวจสอบ Grid BUY
                    should_buy_grid, last_buy_price = check_grid_condition(symbol, current_price, "BUY")
                    if should_buy_grid:
                        add_log(f"📊 {symbol} Grid BUY Trigger! Price: {current_price}")
                        open_grid_order(symbol, "buy", current_price)
                    
                    # ตรวจสอบ Grid SELL  
                    should_sell_grid, last_sell_price = check_grid_condition(symbol, current_price, "SELL")
                    if should_sell_grid:
                        add_log(f"📊 {symbol} Grid SELL Trigger! Price: {current_price}")
                        open_grid_order(symbol, "sell", current_price)
            
            if actions_to_open:
                # 💰 ตรวจสอบ Max Risk ก่อนเปิดออเดอร์
                if check_max_risk(symbol):
                    add_log(f"🛑 {symbol} ถึงขีดจำกัดความเสี่ยง หยุดเทรด")
                    close_count = close_all_positions(symbol)
                    add_log(f"✅ ปิดออเดอร์ทั้งหมด {close_count} รายการ")
                else:
                    action = actions_to_open[0] if len(actions_to_open) == 1 else random.choice(actions_to_open)
                    
                    # 💰 คำนวณ Lot ด้วย Martingale
                    consecutive_loss = get_consecutive_loss_count(symbol)
                    lot_size = calculate_lot_size(symbol, "martingale", consecutive_loss)
                    
                    add_log(f"🎯 จะเปิด: {action.upper()} {symbol}, Lot: {lot_size} (Loss streak: {consecutive_loss})")
                    
                    if bot_status.get("use_sl_tp", True):
                        open_trade(
                            symbol=symbol,
                            action=action,
                            lot=lot_size,
                            sl_points=bot_status["sl_points"],
                            tp_points=bot_status["tp_points"]
                        )
                    else:
                        open_trade_no_sl_tp(
                            symbol=symbol,
                            action=action,
                            lot=lot_size
                        )

        check_and_close_positions()
        
        # 📌 Trailing Stop & Close All Profit
        for symbol in symbols:
            # อัปเดต Trailing Stop
            update_trailing_stop(symbol)
            
            # ตรวจสอบ Close All Profit
            should_close, reason = check_close_all_profit(symbol)
            if should_close:
                add_log(f"💎 Close All Profit: {reason}")
                close_count = close_all_positions(symbol)
                add_log(f"✅ ปิดออเดอร์ทั้งหมด {close_count} รายการ")
        
        # รอ 5 วินาทีก่อนวนลูปใหม่เพื่อไม่ให้กินทรัพยากรเครื่อง
        time.sleep(5)

    add_log("🛑 ระบบเทรดอัตโนมัติหยุดการทำงาน")


# ==========================================
# 3. Web API Endpoints
# ==========================================
@app.route('/')
def index():
    """แสดงหน้า Web UI"""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """ส่งข้อมูลสถานะบัญชีและออเดอร์กลับไปที่หน้าเว็บ"""
    if not mt5.initialize():
        return jsonify({"connected": False, "error": "MT5 Not Initialized"})

    account_info = mt5.account_info()
    if account_info is None:
        return jsonify({"connected": False, "error": "Cannot get account info"})

    positions = mt5.positions_get()
    pos_list = []
    if positions:
        for p in positions:
            pos_list.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price_open": f"{p.price_open:.5f}" if p.price_open else "",
                "price_current": f"{p.price_current:.5f}" if p.price_current else "",
                "sl": f"{p.sl:.5f}" if p.sl else "",
                "tp": f"{p.tp:.5f}" if p.tp else "",
                "profit": round(p.profit, 2)
            })

    return jsonify({
        "connected": True,
        "bot_running": bot_status["is_running"],
        "balance": round(account_info.balance, 2),
        "equity": round(account_info.equity, 2),
        "profit": round(account_info.profit, 2),
        "positions": pos_list,
        "logs": logs,
        "settings": {
            "use_ai": bot_status.get("use_ai", True),
            "use_gpu_model": bot_status.get("use_gpu_model", False),
            "use_bb_ma_macd": bot_status.get("use_bb_ma_macd", False),
            "ai_model_path": bot_status.get("ai_model_path", "ml_model.pkl")
        }
    })


@app.route('/api/control', methods=['POST'])
def control_bot():
    """รับคำสั่งเริ่ม/หยุด บอท จากหน้าเว็บ"""
    data = request.json
    action = data.get('action')

    if action == 'start':
        if not bot_status["is_running"]:
            raw_symbols = data.get('symbols', 'EURUSDm')
            bot_status["symbols"] = [s.strip() for s in raw_symbols.split(',')]
            
            # 💰 Money Management Settings
            bot_status["initial_lot"] = float(data.get('initial_lot', 0.01))
            bot_status["max_lot"] = float(data.get('max_lot', 1.0))
            bot_status["martingale_mult"] = float(data.get('martingale_mult', 1.5))
            bot_status["max_risk_percent"] = float(data.get('max_risk_percent', 10))
            
            # 📊 Grid Trading Settings
            bot_status["use_grid"] = data.get('use_grid', False)
            bot_status["grid_distance"] = int(data.get('grid_distance', 300))
            bot_status["max_grid_orders"] = int(data.get('max_grid_orders', 10))
            
            # 📌 TP/SL Settings
            bot_status["sl_points"] = int(data.get('sl_points', 0))
            bot_status["tp_points"] = int(data.get('tp_points', 0))
            bot_status["use_trailing"] = data.get('use_trailing', False)
            bot_status["trailing_distance"] = int(data.get('trailing_distance', 0))
            bot_status["trailing_step"] = int(data.get('trailing_step', 10))
            
            # 💎 Close All Profit Settings
            bot_status["close_all_percent"] = float(data.get('close_all_percent', 0.1))
            bot_status["close_all_dollar"] = float(data.get('close_all_dollar', 0))
            bot_status["close_all_pullback"] = float(data.get('close_all_pullback', 0))
            
            bot_status["use_sl_tp"] = data.get('use_sl_tp', True)
            bot_status["close_profit"] = float(data.get('close_profit', 10))
            bot_status["close_loss"] = float(data.get('close_loss', 20))
            bot_status["use_ai"] = data.get('use_ai', True)
            bot_status["use_bb_ma_macd"] = data.get('use_bb_ma_macd', False)
            bot_status["ai_model_path"] = data.get('ai_model_path', 'ml_model.pkl')
            bot_status["use_gpu_model"] = data.get('use_gpu_model', False)
            
            tf_str = data.get('timeframe', 'M5')
            bot_status["timeframe"] = get_timeframe(tf_str)
            bot_status["timeframe_str"] = tf_str
            
            # Trend filter settings
            bot_status["use_trend_filter"] = data.get('use_trend_filter', True)
            trend_tf = data.get('trend_timeframe', 'H1')
            bot_status["trend_timeframe"] = get_timeframe(trend_tf)
            
            # 💰 Reset consecutive losses when starting
            global consecutive_losses, last_order_prices
            consecutive_losses = {}
            last_order_prices = {}
            
            # 📝 เริ่มบันทึก log ลงไฟล์
            setup_log_file()
            
            bot_status["is_running"] = True
            
            if bot_status["use_ai"]:
                load_ml_model()

            t = threading.Thread(target=auto_trading_loop)
            t.daemon = True
            t.start()
            return jsonify({"status": "started"})

    elif action == 'stop':
        bot_status["is_running"] = False
        
        # 📝 ปิดไฟล์ log
        close_log_file()
        
        return jsonify({"status": "stopped"})
    
    elif action == 'close_all':
        positions = mt5.positions_get()
        closed_count = 0
        if positions:
            for pos in positions:
                if close_position(pos.ticket, pos.symbol, pos.volume):
                    closed_count += 1
        return jsonify({"status": "closed", "count": closed_count})

    return jsonify({"error": "invalid action"}), 400


# ==========================================
# 5. จุดเริ่มต้นการทำงานของโปรแกรม
# ==========================================
if __name__ == '__main__':
    # Set UTF-8 encoding for Windows
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    if not mt5.initialize():
        print("Failed to initialize MT5")
    else:
        print("MT5 Initialized Successfully")

    print("Web UI is running on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)