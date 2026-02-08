import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import time
from model_utils import predict_signal

# ----------------------------------
# MANUAL NEWS FILTER (NO API LAG)
# ----------------------------------
HIGH_IMPACT_HOURS = [
    (time(13,30), time(14,30)),  # NFP
    (time(12,0), time(12,30)),   # CPI
]

def is_high_impact_news(dt):
    for start, end in HIGH_IMPACT_HOURS:
        if start <= dt.time() <= end:
            return True
    return False

# ----------------------------------
# SESSION FILTER (LONDON + NY)
# ----------------------------------
def is_trading_hours(dt):
    return time(13,0) <= dt.time() <= time(21,0)

# ----------------------------------
# WALK-FORWARD BACKTEST
# ----------------------------------
def walk_forward_backtest(
    df,
    model,
    starting_equity=500,
    prob_threshold=0.50,
    risk_per_trade=0.01,
    max_trades_per_day=3,
    holding_window=16
):

    equity = starting_equity
    peak = equity
    max_dd = 0

    wins = losses = trades = 0
    equity_curve = []
    trade_log = []

    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    daily_trades = {}

    for i in range(len(df) - holding_window):

        row = df.iloc[i]
        today = row['date']

        # ---- filters ----
        if daily_trades.get(today, 0) >= max_trades_per_day:
            equity_curve.append(equity)
            continue

        if not is_trading_hours(row['time']):
            equity_curve.append(equity)
            continue

        if is_high_impact_news(row['time']):
            equity_curve.append(equity)
            continue

        # ---- model ----
        action, confidence = predict_signal(model, row, prob_threshold)
        if action == 0:
            equity_curve.append(equity)
            continue

        trades += 1
        daily_trades[today] = daily_trades.get(today, 0) + 1

        entry = row['close']
        atr = row['atr']
        risk_amount = equity * risk_per_trade

        sl = entry - atr if action == 1 else entry + atr
        tp = entry + 1.5 * atr if action == 1 else entry - 1.5 * atr

        outcome = "HIT NEITHER"

        # ---- forward simulation ----
        for j in range(1, holding_window + 1):
            future = df.iloc[i + j]

            if action == 1:  # BUY
                if future['low'] <= sl:
                    equity -= risk_amount
                    losses += 1
                    outcome = "LOSS"
                    break
                if future['high'] >= tp:
                    equity += risk_amount * 1.5
                    wins += 1
                    outcome = "WIN"
                    break
            else:  # SELL
                if future['high'] >= sl:
                    equity -= risk_amount
                    losses += 1
                    outcome = "LOSS"
                    break
                if future['low'] <= tp:
                    equity += risk_amount * 1.5
                    wins += 1
                    outcome = "WIN"
                    break

        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        equity_curve.append(equity)

        trade_log.append({
            "Trade #": trades,
            "Type": "BUY" if action == 1 else "SELL",
            "Outcome": outcome,
            "Equity": equity
        })

    # ---- REPORT ----
    print("\n================ WALK-FORWARD BACKTEST RESULTS ================")
    print(f"Starting Equity : {starting_equity:.2f}")
    print(f"Final Equity    : {equity:.2f}")
    print(f"Total Trades    : {trades}")
    print(f"Wins            : {wins}")
    print(f"Losses          : {losses}")
    print(f"Win Rate        : {wins / max(1, trades):.2%}")
    print(f"Max Drawdown    : {max_dd:.2%}")
    print("===============================================================")

    trade_df = pd.DataFrame(trade_log)
    print("\nSample Trades:")
    print(trade_df.head(20))

    plt.figure(figsize=(12,6))
    plt.plot(equity_curve)
    plt.title("Equity Curve")
    plt.grid(True)
    plt.show()


# ----------------------------------
# RUN
# ----------------------------------
if __name__ == "__main__":
    model = joblib.load("model.pkl")
    df = pd.read_csv("features_labels.csv")
    walk_forward_backtest(df, model)
