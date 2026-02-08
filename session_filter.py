from datetime import time

def is_trading_session(now):
    return time(7,0) <= now.time() <= time(17,0)
