"""
topdown_engine.py
==================
Plain multi-timeframe top-down bias engine: 4H -> 1H, feeding into a 30M
entry (the 30M part is left to the caller -- Trendline or OTE -- since
each has its own entry mechanics).

This REPLACES the old institutional_analysis.py "SMC Top-Down" stack.
There is deliberately no FVG / Order Block / Inducement zone detection
here -- just a normal top-down read:

  1. 4H  -- macro regime via 200 EMA + swing structure bias (context only)
  2. 1H  -- swing structure + structure-based trade permission, gated
            against the 4H macro bias (this is the actual confirmation
            direction -- see market_analysis.structure_trade_permission)

Both remaining strategies (strategies.py) call get_topdown_bias() so
Trendline and OTE agree on overall market direction instead of each
reading the market independently.
"""

from __future__ import annotations

from typing import Any, Dict

import market_data
from market_analysis import analyse_structure, structure_trade_permission


def _ema200_bias(df):
    if df is None or df.empty or "EMA200" not in df.columns:
        return "NEUTRAL", "EMA200 n/a", 0.0
    close = float(df["Close"].iloc[-1])
    ema200 = float(df["EMA200"].iloc[-1])
    if ema200 <= 0:
        return "NEUTRAL", "EMA200 n/a", 0.0
    dist = (close - ema200) / ema200 * 100.0
    if close > ema200 * 1.001:
        return "BUY", f"Above 200 EMA (+{dist:.2f}%)", dist
    if close < ema200 * 0.999:
        return "SELL", f"Below 200 EMA ({dist:.2f}%)", dist
    return "NEUTRAL", f"At 200 EMA ({dist:+.2f}%)", dist


def get_topdown_bias(symbol: str, count_4h: int = 200, count_1h: int = 200) -> Dict[str, Any]:
    """
    Full 4H -> 1H top-down read for a symbol.

    Returns dict:
      direction     : 'BUY' | 'SELL' | 'NEUTRAL'  -- the 1H confirmation direction
      allowed       : bool  -- structure permission granted on the 1H
      reasons       : list[str], human-readable trail
      bias_4h       : 'BUY' | 'SELL' | 'NEUTRAL'  -- 4H macro regime
      structure_4h  : dict from market_analysis.analyse_structure()
      structure_1h  : dict from market_analysis.analyse_structure()
      df_4h, df_1h  : the fetched dataframes (or None on insufficient data)
      error         : str, only present if data was insufficient
    """
    df_4h = market_data.fetch_candles(symbol, "4h", count=count_4h)
    df_1h = market_data.fetch_candles(symbol, "1h", count=count_1h)

    if df_4h is None or df_4h.empty or len(df_4h) < 40:
        return {
            "direction": "NEUTRAL", "allowed": False,
            "reasons": ["Insufficient 4H data for top-down bias"],
            "bias_4h": "NEUTRAL", "structure_4h": {}, "structure_1h": {},
            "df_4h": df_4h, "df_1h": df_1h,
            "error": "insufficient_4h_data",
        }
    if df_1h is None or df_1h.empty or len(df_1h) < 40:
        return {
            "direction": "NEUTRAL", "allowed": False,
            "reasons": ["Insufficient 1H data for top-down bias"],
            "bias_4h": "NEUTRAL", "structure_4h": {}, "structure_1h": {},
            "df_4h": df_4h, "df_1h": df_1h,
            "error": "insufficient_1h_data",
        }

    reasons = []

    # --- 4H: macro regime (EMA200) + swing structure ---
    ema_bias_4h, ema_note_4h, _ = _ema200_bias(df_4h)
    structure_4h = analyse_structure(df_4h, left=3, right=3, lookback=80)
    reasons.append(f"4H regime: {ema_note_4h}")
    reasons.append(f"4H structure: {structure_4h.get('note', structure_4h.get('bias'))}")

    macro_bias = ema_bias_4h
    struct_bias_4h = structure_4h.get("bias", "NEUTRAL")
    if struct_bias_4h == "BULLISH" and macro_bias != "SELL":
        macro_bias = "BUY"
    elif struct_bias_4h == "BEARISH" and macro_bias != "BUY":
        macro_bias = "SELL"

    # --- 1H: swing structure + trade permission, gated against 4H bias ---
    structure_1h = analyse_structure(df_1h, left=3, right=3, lookback=80)
    allowed, permission_reason, direction = structure_trade_permission(macro_bias, structure_1h)
    reasons.append(f"1H structure: {structure_1h.get('note', structure_1h.get('bias'))}")
    reasons.append(permission_reason)

    # If the 1H confirmation direction fights the 4H macro regime outright
    # (not just "EMA neutral"), flag it -- the caller (Trendline/OTE)
    # decides how much weight to give a counter-trend 1H confirmation.
    if macro_bias in ("BUY", "SELL") and direction in ("BUY", "SELL") and direction != macro_bias:
        reasons.append(
            f"⚠️ 1H confirmation ({direction}) is counter to the 4H regime "
            f"({macro_bias}) -- treat with caution"
        )

    return {
        "direction": direction,
        "allowed": bool(allowed),
        "reasons": reasons,
        "bias_4h": macro_bias,
        "structure_4h": structure_4h,
        "structure_1h": structure_1h,
        "df_4h": df_4h,
        "df_1h": df_1h,
    }


def format_topdown_summary(bias: Dict[str, Any]) -> str:
    """Short human-readable summary of the 4H/1H top-down read."""
    lines = [
        f"Top-down bias: {bias.get('bias_4h', 'NEUTRAL')} (4H)  →  "
        f"{bias.get('direction', 'NEUTRAL')} (1H confirmation)"
    ]
    for r in bias.get("reasons") or []:
        lines.append(f"  • {r}")
    return "\n".join(lines)
