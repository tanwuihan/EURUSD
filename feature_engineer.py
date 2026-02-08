import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

def add_features(df):
    """
    Add technical features to DataFrame
    """
    df['return_1'] = df['close'].pct_change(1)
    df['return_3'] = df['close'].pct_change(3)
    df['return_6'] = df['close'].pct_change(6)

    # EMA slope
    ema = EMAIndicator(df['close'], window=14)
    df['ema_slope'] = ema.ema_indicator().diff()

    # RSI
    rsi = RSIIndicator(df['close'], window=14)
    df['rsi'] = rsi.rsi()

    # MACD
    macd = MACD(df['close'])
    df['macd'] = macd.macd_diff()

    # ATR
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()

    df.dropna(inplace=True)
    return df
