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

import legacy_data

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
