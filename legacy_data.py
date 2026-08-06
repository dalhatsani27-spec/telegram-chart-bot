"""
legacy_data.py
================
Fallback data path (Twelve Data / yfinance) -- used only when MT5 isn't
running/reachable, or a symbol isn't available in the terminal's Market
Watch. Under normal operation with MT5 open, mt5_data.py handles everything
and this file is rarely touched, which is the point: it keeps the free-tier
external API usage to near zero.
"""

import os
import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")

_TF_TWELVE_INTERVAL = {"1min": "1min", "3min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "1h": "1h", "4h": "4h"}
_TF_YF_INTERVAL = {"1min": "1m", "3min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "60m", "4h": "60m"}
_TF_RESAMPLE = {"3min": "3min"}
_TF_YF_PERIOD = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "60m": "60d"}


def normalize_ticker_twelve_data(symbol):
    symbol = symbol.strip().upper().replace("[", "").replace("]", "").replace("'", "")
    mapping = {
        "GOLD": "XAU/USD", "XAUUSD": "XAU/USD", "SILVER": "XAG/USD", "XAGUSD": "XAG/USD",
        "OIL": "WTI/USD", "USOIL": "WTI/USD", "WTI": "WTI/USD", "BRENT": "BRENT/USD",
        "BTC": "BTC/USD", "BTCUSD": "BTC/USD", "ETH": "ETH/USD", "ETHUSD": "ETH/USD",
        "US30": "DJI", "NAS100": "IXIC", "SPX500": "SPX",
    }
    if symbol in mapping:
        return mapping[symbol]
    if len(symbol) == 6 and "/" not in symbol and symbol not in ["US30", "NAS100", "SPX500"]:
        return f"{symbol[:3]}/{symbol[3:]}"
    return symbol


def normalize_ticker_yfinance(symbol):
    symbol = symbol.strip().upper().replace("[", "").replace("]", "").replace("'", "")
    alias_map = {
        "GOLD": "GC=F", "XAUUSD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
        "BTC": "BTC-USD", "BTCUSD": "BTC-USD", "ETH": "ETH-USD", "ETHUSD": "ETH-USD",
        "US30": "^DJI", "NAS100": "^IXIC", "SPX500": "^GSPC",
    }
    if symbol in alias_map:
        return alias_map[symbol]
    if len(symbol) == 6 and not symbol.endswith("=X") and symbol not in ["US30", "NAS100", "SPX500"]:
        return f"{symbol}=X"
    return symbol


def fetch_twelve_data(symbol, interval="30min", outputsize=150):
    if not TWELVE_DATA_API_KEY:
        return pd.DataFrame()
    clean_symbol = normalize_ticker_twelve_data(symbol)
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": clean_symbol, "interval": interval, "outputsize": outputsize, "apikey": TWELVE_DATA_API_KEY}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        if res.status_code == 200 and "values" in data:
            df = pd.DataFrame(data["values"])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.sort_index()
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
            return df[['Open', 'High', 'Low', 'Close']].dropna()
    except Exception:
        pass
    return pd.DataFrame()


def clean_and_normalize_data(df):
    if df.empty or len(df) < 30:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.dropna(inplace=True)
    df['EMA20'] = df['Close'].ewm(span=min(20, len(df)), adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=min(50, len(df)), adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=min(200, len(df)), adjust=False).mean()

    # Approximate VWAP using typical price * volume (tick_volume when available).
    # Resets conceptually over the full loaded window — useful as dynamic S/R.
    typical = (df['High'] + df['Low'] + df['Close']) / 3.0
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        vol = df['Volume'].replace(0, np.nan).fillna(1.0)
    else:
        vol = pd.Series(1.0, index=df.index)
    cum_tp_vol = (typical * vol).cumsum()
    cum_vol = vol.cumsum()
    df['VWAP'] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return df


def _resample_ohlc(df, rule):
    if df.empty:
        return df
    return df.resample(rule).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()


def _yf_download_with_timeout(ticker, period, interval, timeout_sec=18):
    """yfinance has no built-in timeout; wrap so Render free tier cannot hang forever."""
    import concurrent.futures
    def _job():
        data = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=True, threads=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_job)
            return fut.result(timeout=timeout_sec)
    except Exception:
        return pd.DataFrame()


def fetch_timeframe_data_legacy(symbol, tf_code, outputsize=250):
    td_interval = _TF_TWELVE_INTERVAL.get(tf_code, "30min")
    fetch_size = outputsize * 3 if tf_code == "3min" else outputsize
    raw = fetch_twelve_data(symbol, interval=td_interval, outputsize=min(fetch_size, 5000))

    if raw.empty and YF_AVAILABLE:
        ticker = normalize_ticker_yfinance(symbol)
        yf_interval = _TF_YF_INTERVAL.get(tf_code, "30m")
        period = _TF_YF_PERIOD.get(yf_interval, "30d")
        raw = _yf_download_with_timeout(ticker, period, yf_interval, timeout_sec=18)

    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return pd.DataFrame()

    if tf_code in _TF_RESAMPLE:
        raw = raw.copy()
        raw.columns = [c.capitalize() for c in raw.columns]
        cols = [c for c in ["Open", "High", "Low", "Close"] if c in raw.columns]
        if len(cols) < 4:
            return pd.DataFrame()
        raw = _resample_ohlc(raw[cols], _TF_RESAMPLE[tf_code])

    return clean_and_normalize_data(raw)

