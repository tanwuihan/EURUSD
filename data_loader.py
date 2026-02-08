import MetaTrader5 as mt5
import pandas as pd


def get_data(symbol="EURUSD", timeframe="M15", n_bars=15000):
    """
    Fetch historical M15 data from MT5
    """
    if not mt5.initialize():
        raise ConnectionError("MT5 initialization failed")

    tf_dict = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
               "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
               "H1": mt5.TIMEFRAME_H1}

    rates = mt5.copy_rates_from_pos(symbol, tf_dict[timeframe], 0, n_bars)
    mt5.shutdown()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    return df
