"""Compatibility API for the modern Trendline and OTE engines."""
from strategy_upgrade import trendline_analysis, ote_analysis, format_trendline_report, format_ote_report


def run_trendline_analysis(symbol: str, tf_code: str = "30min", topdown=None):
    return trendline_analysis(symbol, tf_code=tf_code, topdown=topdown)


def run_ote_analysis(symbol: str, tf_code: str = "30min", topdown=None):
    return ote_analysis(symbol, tf_code=tf_code, topdown=topdown)


def build_position_container(a):
    if not a or a.get("direction") not in ("BUY", "SELL") or not a.get("valid"):
        return None
    d = a.get("df")
    atr = float(d["ATR"].iloc[-1]) if d is not None and "ATR" in d.columns else 0.0
    close = float(d["Close"].iloc[-1]) if d is not None else None
    line = a.get("trendline") or {}
    x = len(d) - 1 if d is not None else 0
    entry = float(line.get("slope", 0.0)) * x + float(line.get("intercept", close or 0.0)) if line else close
    entry = close if entry is None else entry
    risk = max(0.8 * atr, abs(entry) * 0.0005, 1e-9)
    sl = entry - risk if a["direction"] == "BUY" else entry + risk
    tp1 = entry + 1.5 * risk if a["direction"] == "BUY" else entry - 1.5 * risk
    tp2 = entry + 2.5 * risk if a["direction"] == "BUY" else entry - 2.5 * risk
    tp3 = entry + 3.5 * risk if a["direction"] == "BUY" else entry - 3.5 * risk
    return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "order_type": "MARKET", "tp3_basis": "3.5R"}
