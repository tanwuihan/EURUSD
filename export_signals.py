import os
import joblib
import pandas as pd

from data_loader import get_data
from feature_engineer import add_features

# ================= CONFIG =================
SYMBOL = "EURUSD"
TIMEFRAME = "M15"
N_BARS = 5000

FEATURES = [
    'return_1',
    'return_3',
    'return_6',
    'ema_slope',
    'rsi',
    'macd',
    'atr'
]

# MT5 COMMON FILES PATH (Windows)
MT5_COMMON_PATH = os.path.join(
    os.environ["APPDATA"],
    "MetaQuotes",
    "Terminal",
    "Common",
    "Files",
    "signals.csv"
)

# ==========================================


def load_model():
    return joblib.load("model.pkl")


def predict_signal(model, row):
    X = row[FEATURES].values.reshape(1, -1)
    pred = model.predict(X)[0]

    if pred == 1:
        return "BUY"
    elif pred == -1:
        return "SELL"
    else:
        return None


def export_signals():
    print("Loading model...")
    model = load_model()

    print("Loading market data...")
    df = get_data(symbol=SYMBOL, timeframe=TIMEFRAME, n_bars=N_BARS)
    df = add_features(df)
    df = df.dropna().reset_index(drop=True)

    signals = []

    for i in range(len(df)):
        row = df.iloc[i]
        signal = predict_signal(model, row)

        if signal is None:
            continue

        price = row["close"]
        atr = row["atr"]

        # RR stays 1 : 1.5 (as you requested)
        if signal == "BUY":
            sl = price - atr
            tp = price + atr * 1.5
        else:
            sl = price + atr
            tp = price - atr * 1.5

        signals.append({
            "time": row["time"],
            "signal": signal,
            "sl": round(sl, 6),
            "tp": round(tp, 6)
        })

    signals_df = pd.DataFrame(signals)

    # 🚨 CRITICAL: MT5 datetime format
    signals_df["time"] = pd.to_datetime(signals_df["time"]) \
        .dt.strftime("%Y.%m.%d %H:%M")

    # Save DIRECTLY to MT5 Common Files
    signals_df.to_csv(MT5_COMMON_PATH, index=False)

    print("====================================")
    print(f"Signals exported successfully!")
    print(f"File location:\n{MT5_COMMON_PATH}")
    print(f"Total signals: {len(signals_df)}")
    print("====================================")


if __name__ == "__main__":
    export_signals()
