"""
data.py — market data, volume profile, legacy feed helpers.
"""
"""
mt5_data.py
================
All market data, account info, broker info, and trade history come straight
from the locally-running MT5 terminal via the official MetaTrader5 package --
no external API, no rate limits, no "exhaustion" risk, and it's exactly the
feed the EA itself trades on.

Twelve Data / yfinance are kept ONLY as a fallback for symbols MT5 doesn't
carry, or for running/testing this file when MT5 isn't open (e.g. designing
away from the trading PC). That fallback lives in legacy_data.py.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# [merged] was: import legacy_data

_TF_MAP = {
    "1min": "M1", "3min": "M3", "5min": "M5", "15min": "M15",
    "30min": "M30", "1h": "H1", "4h": "H4",
}


def _mt5_timeframe(tf_code):
    if not MT5_AVAILABLE:
        return None
    name = _TF_MAP.get(tf_code)
    if name is None:
        return None
    return getattr(mt5, f"TIMEFRAME_{name}", None)


def ensure_connected():
    """Call before any MT5 operation. Returns True if the terminal is reachable."""
    if not MT5_AVAILABLE:
        return False
    if not mt5.terminal_info():
        return mt5.initialize()
    return True


def is_mt5_ready():
    return MT5_AVAILABLE and ensure_connected()


# ----------------------------------------------------------------------------
# Market data
# ----------------------------------------------------------------------------
def fetch_candles(symbol, tf_code, count=250):
    """
    Returns a cleaned OHLC dataframe (Open/High/Low/Close columns, datetime
    index, chronological order) sourced from MT5 if available, otherwise from
    the Twelve Data / yfinance fallback.
    """
    if is_mt5_ready():
        mt5_tf = _mt5_timeframe(tf_code)
        if mt5_tf is not None:
            if not mt5.symbol_select(symbol, True):
                pass  # symbol not in Market Watch under this exact name -- fall through to legacy
            else:
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 1, count)  # shift 1 = skip current forming bar
                if rates is not None and len(rates) >= 30:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                                       'tick_volume': 'Volume'}, inplace=True)
                    cols = ['Open', 'High', 'Low', 'Close'] + (['Volume'] if 'Volume' in df.columns else [])
                    return legacy_data.clean_and_normalize_data(df[cols])

    # Fallback: legacy Twelve Data / yfinance path
    return legacy_data.fetch_timeframe_data_legacy(symbol, tf_code, count)


def fetch_tick(symbol):
    """Returns (bid, ask, spread_points) or (None, None, None) if unavailable."""
    if is_mt5_ready() and mt5.symbol_select(symbol, True):
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick and info:
            spread_points = (tick.ask - tick.bid) / info.point if info.point else None
            return tick.bid, tick.ask, spread_points
    return None, None, None


# ----------------------------------------------------------------------------
# Account / broker info
# ----------------------------------------------------------------------------
def get_account_summary():
    """Returns a dict of account + broker info, or None if MT5 isn't reachable."""
    if not is_mt5_ready():
        return None
    acc = mt5.account_info()
    term = mt5.terminal_info()
    if acc is None:
        return None
    return {
        "broker": acc.company,
        "server": acc.server,
        "login": acc.login,
        "currency": acc.currency,
        "leverage": acc.leverage,
        "balance": acc.balance,
        "equity": acc.equity,
        "margin": acc.margin,
        "margin_free": acc.margin_free,
        "trade_allowed": bool(term.trade_allowed) if term else None,
        "connected": bool(term.connected) if term else None,
    }


def get_pnl_summary(magic_number, days=1):
    """
    Aggregates closed-deal profit and win-rate for this EA's magic number
    over the trailing `days` days. Returns dict or None if unavailable.
    """
    if not is_mt5_ready():
        return None
    date_to = datetime.now()
    date_from = date_to - timedelta(days=days)
    deals = mt5.history_deals_get(date_from, date_to)
    if deals is None:
        return {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0, "win_rate": None}

    relevant = [d for d in deals if d.magic == magic_number and d.entry == 1]  # entry==1 -> DEAL_ENTRY_OUT (closing deal)
    pnl = sum(d.profit + d.swap + d.commission for d in relevant)
    wins = sum(1 for d in relevant if d.profit > 0)
    losses = sum(1 for d in relevant if d.profit <= 0)
    total = len(relevant)
    win_rate = (wins / total * 100.0) if total > 0 else None
    return {"trades": total, "pnl": pnl, "wins": wins, "losses": losses, "win_rate": win_rate}


def get_open_positions(magic_number=None):
    """Returns a list of open position dicts, optionally filtered by magic number."""
    if not is_mt5_ready():
        return []
    positions = mt5.positions_get()
    if positions is None:
        return []
    out = []
    for p in positions:
        if magic_number is not None and p.magic != magic_number:
            continue
        out.append({
            "ticket": p.ticket, "symbol": p.symbol,
            "type": "BUY" if p.type == 0 else "SELL",
            "volume": p.volume, "price_open": p.price_open,
            "sl": p.sl, "tp": p.tp, "profit": p.profit,
        })
    return out


def get_pattern_win_rates(magic_number, days=30):
    """
    Groups closed trades by pattern name (parsed from each deal's comment,
    which the EA tags as "PCP:<PatternName>" when it fires). Returns a dict:
    {pattern_name: {"trades": N, "wins": N, "pnl": X, "win_rate": pct}},
    sorted by trade count descending. Only meaningful once there's real
    trade history -- returns {} if none yet or MT5 isn't reachable.
    """
    if not is_mt5_ready():
        return {}
    date_to = datetime.now()
    date_from = date_to - timedelta(days=days)
    deals = mt5.history_deals_get(date_from, date_to)
    if deals is None:
        return {}

    stats = {}
    for d in deals:
        if d.magic != magic_number or d.entry != 1:  # entry==1 -> closing deal
            continue
        comment = d.comment or ""
        if comment.startswith("PCP:"):
            pattern_name = comment[4:]
        else:
            pattern_name = "Unknown"
        s = stats.setdefault(pattern_name, {"trades": 0, "wins": 0, "pnl": 0.0})
        s["trades"] += 1
        s["pnl"] += (d.profit + d.swap + d.commission)
        if d.profit > 0:
            s["wins"] += 1

    for name, s in stats.items():
        s["win_rate"] = (s["wins"] / s["trades"] * 100.0) if s["trades"] > 0 else None

    return dict(sorted(stats.items(), key=lambda kv: kv[1]["trades"], reverse=True))


def shutdown():
    if MT5_AVAILABLE:
        try:
            mt5.shutdown()
        except Exception:
            pass
"""
volume_profile.py
================
Fixed Range Volume Profile, computed from MT5 tick_volume (number of price
updates per bar -- a real, commonly-used proxy in forex/CFD trading, NOT
true traded volume, since that doesn't exist in retail forex. If tick
volume isn't available (e.g. Twelve Data fallback with no Volume column),
degrades to a "time spent at price" profile -- still useful, just weaker.

Produces:
  - Point of Control (POC): the price level with the most activity.
  - Value Area: the price band containing ~68% of total activity around the POC.

Used two ways:
  1. Visual overlay on the chart (a horizontal histogram beside price).
  2. A confidence signal: a pattern's trigger/neckline sitting at the POC or
     inside the Value Area gets a confidence boost (that level has proven
     significance); one sitting in a low-volume gap gets a penalty (thin,
     less-respected price, more prone to a violent, unreliable move).
"""

import numpy as np


def compute_volume_profile(df, bins=24, value_area_pct=0.68):
    """
    df: OHLC dataframe, optionally with a 'Volume' column (tick_volume).
    Returns dict with bin_edges, bin_volumes, poc_price, value_area_low,
    value_area_high -- or None if there's not enough range to profile.
    """
    if df is None or df.empty:
        return None
    price_min = float(df['Low'].min())
    price_max = float(df['High'].max())
    if price_max <= price_min:
        return None

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_volumes = np.zeros(bins)
    has_volume = 'Volume' in df.columns

    highs = df['High'].values
    lows = df['Low'].values
    vols = df['Volume'].values if has_volume else np.ones(len(df))

    for lo, hi, vol in zip(lows, highs, vols):
        if hi <= lo:
            continue
        vol = float(vol) if vol and vol > 0 else 1.0
        start_idx = max(0, int(np.searchsorted(bin_edges, lo, side='right')) - 1)
        end_idx = min(bins, int(np.searchsorted(bin_edges, hi, side='left')))
        if end_idx <= start_idx:
            idx = min(bins - 1, max(0, start_idx))
            bin_volumes[idx] += vol
            continue
        span = hi - lo
        for b in range(start_idx, end_idx):
            b_lo = max(lo, bin_edges[b])
            b_hi = min(hi, bin_edges[b + 1])
            frac = max(0.0, (b_hi - b_lo)) / span
            bin_volumes[b] += vol * frac

    total = bin_volumes.sum()
    if total <= 0:
        return None

    poc_idx = int(np.argmax(bin_volumes))
    poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

    lo_i = hi_i = poc_idx
    included_vol = bin_volumes[poc_idx]
    while included_vol / total < value_area_pct and (lo_i > 0 or hi_i < bins - 1):
        left_vol = bin_volumes[lo_i - 1] if lo_i > 0 else -1
        right_vol = bin_volumes[hi_i + 1] if hi_i < bins - 1 else -1
        if right_vol >= left_vol and hi_i < bins - 1:
            hi_i += 1
            included_vol += bin_volumes[hi_i]
        elif lo_i > 0:
            lo_i -= 1
            included_vol += bin_volumes[lo_i]
        else:
            break

    return {
        "bin_edges": bin_edges, "bin_volumes": bin_volumes, "poc_price": poc_price,
        "value_area_low": float(bin_edges[lo_i]), "value_area_high": float(bin_edges[hi_i + 1]),
    }


def level_volume_bonus(profile, price_level, tolerance_frac=0.0015):
    """
    Confidence delta for a pattern's trigger/neckline sitting at a
    volume-significant level (+) or in a thin, low-activity gap (-).
    Range: roughly -5 to +8. Returns 0.0 if no profile is available.
    """
    if profile is None or price_level is None:
        return 0.0

    bin_edges = profile["bin_edges"]
    avg_price = float((bin_edges[0] + bin_edges[-1]) / 2) or 1.0
    tol = avg_price * tolerance_frac

    poc = profile["poc_price"]
    if abs(price_level - poc) <= tol:
        return 8.0

    va_lo, va_hi = profile["value_area_low"], profile["value_area_high"]
    if va_lo - tol <= price_level <= va_hi + tol:
        return 4.0

    bin_volumes = profile["bin_volumes"]
    idx = int(np.searchsorted(bin_edges, price_level)) - 1
    idx = min(max(idx, 0), len(bin_volumes) - 1)
    max_vol = bin_volumes.max() or 1.0
    if bin_volumes[idx] < 0.15 * max_vol:
        return -5.0
    return 0.0
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


def fetch_timeframe_data_legacy(symbol, tf_code, outputsize=250):
    td_interval = _TF_TWELVE_INTERVAL.get(tf_code, "30min")
    fetch_size = outputsize * 3 if tf_code == "3min" else outputsize
    raw = fetch_twelve_data(symbol, interval=td_interval, outputsize=min(fetch_size, 5000))

    if raw.empty and YF_AVAILABLE:
        ticker = normalize_ticker_yfinance(symbol)
        yf_interval = _TF_YF_INTERVAL.get(tf_code, "30m")
        period = _TF_YF_PERIOD.get(yf_interval, "30d")
        try:
            data = yf.download(ticker, period=period, interval=yf_interval, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            raw = data
        except Exception:
            raw = pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if tf_code in _TF_RESAMPLE:
        raw.columns = [c.capitalize() for c in raw.columns]
        raw = _resample_ohlc(raw[['Open', 'High', 'Low', 'Close']], _TF_RESAMPLE[tf_code])

    return clean_and_normalize_data(raw)
