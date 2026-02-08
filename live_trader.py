import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time
import time as sleep
from model_utils import predict_signal

# ----------------------------------
# CONFIG
# ----------------------------------
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
LOT = 0.01
MAX_TRADES_PER_DAY = 3
PROB_THRESHOLD = 0.50
ATR_MULTIPLIER_SL = 1.0
ATR_MULTIPLIER_TP = 1.5

# ----------------------------------
# MANUAL NEWS FILTER
# ----------------------------------
HIGH_IMPACT_HOURS = [
    (time(13,30), time(14,30)),
    (time(12,0), time(12,30)),
]

def is_news_time(dt):
    for start, end in HIGH_IMPACT_HOURS:
        if start <= dt.time() <= end:
            return True
    return False

# ----------------------------------
# SESSION FILTER
# ----------------------------------
def is_trading_session(dt):
    return time(13,0) <= dt.time() <= time(21,0)

# ----------------------------------
# FEATURES
# ----------------------------------
FEATURES = [
    'return_1','return_3','return_6',
    'ema_slope','rsi','macd','atr'
]

# ----------------------------------
# DATA + FEATURES
# ----------------------------------
def get_data(symbol, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def add_features(df):
    df['return_1'] = df['close'].pct_change(1)
    df['return_3'] = df['close'].pct_change(3)
    df['return_6'] = df['close'].pct_change(6)
    df['ema_slope'] = df['close'].ewm(span=10).mean().diff()

    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['macd'] = (
        df['close'].ewm(span=12).mean()
        - df['close'].ewm(span=26).mean()
    )

    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df.fillna(0, inplace=True)
    return df

# ----------------------------------
# ORDER
# ----------------------------------
def place_trade(action, atr):
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if action == 1 else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL

    sl = price - ATR_MULTIPLIER_SL * atr if action == 1 else price + ATR_MULTIPLIER_SL * atr
    tp = price + ATR_MULTIPLIER_TP * atr if action == 1 else price - ATR_MULTIPLIER_TP * atr

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 5,
        "magic": 777,
        "comment": "ML_M15",
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    return mt5.order_send(request)

# ----------------------------------
# MAIN
# ----------------------------------
if not mt5.initialize():
    print("MT5 init failed")
    quit()

model = joblib.load("model.pkl")
daily_trades = {}
today = datetime.utcnow().date()

print("LIVE TRADER STARTED")

while True:
    now = datetime.utcnow()

    if now.date() != today:
        daily_trades = {}
        today = now.date()

    if not is_trading_session(now) or is_news_time(now):
        sleep.sleep(60)
        continue

    if daily_trades.get(today, 0) >= MAX_TRADES_PER_DAY:
        sleep.sleep(60)
        continue

    df = add_features(get_data(SYMBOL))
    last = df.iloc[-1]

    action, confidence = predict_signal(model, last, PROB_THRESHOLD)
    if action == 0:
        sleep.sleep(60)
        continue

    result = place_trade(action, last['atr'])
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        daily_trades[today] = daily_trades.get(today, 0) + 1
        print(f"{now} | {'BUY' if action == 1 else 'SELL'} executed")

    sleep.sleep(900)  # wait next M15 candle
