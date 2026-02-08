import MetaTrader5 as mt5

if not mt5.initialize():
    print("MT5 initialize() failed")
    mt5.shutdown()
else:
    info = mt5.account_info()
    print("Connected to MT5")
    print("Account:", info.login)
    print("Balance:", info.balance)
    mt5.shutdown()
