import joblib
from xgboost import XGBClassifier
from data_loader import get_data
from feature_engineer import add_features
from label_generator import create_labels


def retrain_model(symbol="EURUSD"):
    # Fetch 30,000 bars of M15 data (~6 months)
    df = get_data(symbol=symbol, timeframe="M15", n_bars=30000)

    # Add technical features
    df = add_features(df)

    # Create labels for M15 using proper lookahead
    df = create_labels(df, lookahead=4)  # 4 bars = 1 hour into future

    FEATURES = ['return_1', 'return_3', 'return_6', 'ema_slope', 'rsi', 'macd', 'atr']
    X = df[FEATURES]
    y = df['label']

    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric='mlogloss',
    )
    model.fit(X, y)

    # Save model & features/labels
    joblib.dump(model, "model.pkl")
    df.to_csv("features_labels.csv", index=False)

    print("Model retrained and saved as model.pkl")
    print("Features + labels saved as features_labels.csv")


if __name__ == "__main__":
    retrain_model()
