import MetaTrader5 as mt5
import pandas as pd
import numpy as np

def init_mt5():
    """Ensures MT5 terminal connection is active."""
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return False
    return True

def fetch_rates(symbol: str, timeframe, count: int = 300) -> pd.DataFrame:
    """
    Fetches OHLCV rate data safely without freezing the socket thread.
    """
    if not init_mt5():
        return None

    # Ensure symbol is selected in Market Watch to avoid request hanging
    if not mt5.symbol_select(symbol, True):
        print(f"Error: Symbol {symbol} not found or inactive in Market Watch.")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        print(f"Error fetching rates for {symbol}: {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)

    return df
