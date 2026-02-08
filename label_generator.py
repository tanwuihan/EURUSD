import pandas as pd
import numpy as np

def create_labels(df, lookahead=4):
    """
    Generate labels for M15 bars (no H1 lookahead assumption).
    1 = BUY, 2 = SELL, 0 = NO TRADE

    lookahead: number of future M15 bars to consider for target movement
    """
    df = df.copy()
    labels = []

    for i in range(len(df) - lookahead):
        entry = df['close'].iloc[i]
        atr = df['atr'].iloc[i]

        tp_buy = entry + atr
        sl_buy = entry - atr
        tp_sell = entry - atr
        sl_sell = entry + atr

        # lookahead future highs/lows
        future_high = df['high'].iloc[i+1:i+lookahead+1].max()
        future_low = df['low'].iloc[i+1:i+lookahead+1].min()

        if future_high >= tp_buy:
            labels.append(1)  # BUY
        elif future_low <= tp_sell:
            labels.append(2)  # SELL
        else:
            labels.append(0)  # NO TRADE

    # truncate last lookahead rows
    df = df.iloc[:-lookahead]
    df['label'] = labels
    return df
