import joblib
import numpy as np

MODEL_PATH = "model.pkl"

FEATURES = [
    'return_1', 'return_3', 'return_6',
    'ema_slope', 'rsi', 'macd', 'atr'
]

_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_signal(row, prob_threshold=0.50):
    model = load_model()

    X = row[FEATURES].values.reshape(1, -1)
    probs = model.predict_proba(X)[0]
    action = np.argmax(probs)

    if action == 0 or probs[action] < prob_threshold:
        return 0, None, None

    atr = row['atr']
    price = row['close']

    if action == 1:  # BUY
        sl = price - atr
        tp = price + 1.5 * atr
    else:            # SELL
        sl = price + atr
        tp = price - 1.5 * atr

    return action, sl, tp
