"""
trendline_family.py
===================
Full trendline family engine for the Trendline strategy (and projection
overlays for all strategies).

Builds:
  - Primary uptrend lines (connect swing lows → higher lows)
  - Primary downtrend lines (connect swing highs → lower highs)
  - Parallel channel (project opposite boundary)
  - Fan lines (secondary shallower lines from same origin)
  - Measured-move / breakout projections (1.0x, 1.618x, 2.618x)
  - Touch count scoring (more touches = stronger line)
  - Optional long/short position container (entry, SL, TP1, TP2)

Uses ZigZag swings from market_structure for clean pivots.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from market_structure import zigzag_swings
from volume_profile import compute_volume_profile


def _line_value(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y0
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)


def _count_touches(df: pd.DataFrame, x0: int, y0: float, x1: int, y1: float,
                   kind: str, tol_atr: float = 0.35) -> int:
    """Count how many bars touch the line within tol_atr * ATR."""
    if df is None or len(df) < 5:
        return 0
    atr = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).values
    highs = df["High"].values
    lows = df["Low"].values
    touches = 0
    for i in range(min(x0, x1), min(max(x0, x1) + 1, len(df))):
        lv = _line_value(x0, y0, x1, y1, i)
        a = float(atr[i]) if i < len(atr) and atr[i] > 0 else abs(y1 - y0) * 0.05
        tol = max(a * tol_atr, 1e-9)
        if kind == "support":
            if abs(lows[i] - lv) <= tol:
                touches += 1
        else:
            if abs(highs[i] - lv) <= tol:
                touches += 1
    return touches


def build_trendline_family(df: pd.DataFrame, max_lines: int = 4) -> Dict[str, Any]:
    """
    Build a complete trendline family from ZigZag swings.

    Returns dict with:
      uptrends, downtrends, channel, projections, volume_profile,
      direction, strength, geometry arrays for charting
    """
    if df is None or len(df) < 30:
        return {"error": "Insufficient data for trendline family", "direction": "NEUTRAL"}

    n = len(df)
    pivots = zigzag_swings(df, depth=4, deviation_atr=0.28)
    if len(pivots) < 4:
        pivots = zigzag_swings(df, depth=3, deviation_atr=0.20)

    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]

    uptrends = []
    # Connect consecutive higher lows
    for i in range(len(lows) - 1):
        for j in range(i + 1, len(lows)):
            a, b = lows[i], lows[j]
            if b["price"] > a["price"] and b["index"] > a["index"]:
                touches = _count_touches(df, a["index"], a["price"], b["index"], b["price"], "support")
                # Project to end of chart
                y_end = _line_value(a["index"], a["price"], b["index"], b["price"], n - 1)
                uptrends.append({
                    "x0": a["index"], "y0": a["price"],
                    "x1": b["index"], "y1": b["price"],
                    "y_end": y_end,
                    "touches": touches,
                    "slope": (b["price"] - a["price"]) / max(b["index"] - a["index"], 1),
                    "kind": "uptrend",
                })
    uptrends.sort(key=lambda t: (-t["touches"], -t["x1"]))
    uptrends = uptrends[:max_lines]

    downtrends = []
    for i in range(len(highs) - 1):
        for j in range(i + 1, len(highs)):
            a, b = highs[i], highs[j]
            if b["price"] < a["price"] and b["index"] > a["index"]:
                touches = _count_touches(df, a["index"], a["price"], b["index"], b["price"], "resistance")
                y_end = _line_value(a["index"], a["price"], b["index"], b["price"], n - 1)
                downtrends.append({
                    "x0": a["index"], "y0": a["price"],
                    "x1": b["index"], "y1": b["price"],
                    "y_end": y_end,
                    "touches": touches,
                    "slope": (b["price"] - a["price"]) / max(b["index"] - a["index"], 1),
                    "kind": "downtrend",
                })
    downtrends.sort(key=lambda t: (-t["touches"], -t["x1"]))
    downtrends = downtrends[:max_lines]

    # Primary channel: best uptrend + parallel through highest high in span
    channel = None
    if uptrends:
        prim = uptrends[0]
        span_highs = [p for p in highs if prim["x0"] <= p["index"] <= prim["x1"]]
        if not span_highs:
            span_highs = highs[-3:] if highs else []
        if span_highs:
            top_pt = max(span_highs, key=lambda p: p["price"])
            # Parallel: same slope, through top_pt
            # y = slope*(x - x0) + y0_parallel
            slope = prim["slope"]
            y0_par = top_pt["price"] - slope * (top_pt["index"] - prim["x0"])
            y_end_par = y0_par + slope * ((n - 1) - prim["x0"])
            channel = {
                "lower": prim,
                "upper": {
                    "x0": prim["x0"], "y0": y0_par,
                    "x1": prim["x1"], "y1": y0_par + slope * (prim["x1"] - prim["x0"]),
                    "y_end": y_end_par,
                    "slope": slope,
                    "kind": "channel_upper",
                },
                "mid_end": (prim["y_end"] + y_end_par) / 2.0,
                "width": abs(y_end_par - prim["y_end"]),
            }
    elif downtrends:
        prim = downtrends[0]
        span_lows = [p for p in lows if prim["x0"] <= p["index"] <= prim["x1"]]
        if not span_lows:
            span_lows = lows[-3:] if lows else []
        if span_lows:
            bot_pt = min(span_lows, key=lambda p: p["price"])
            slope = prim["slope"]
            y0_par = bot_pt["price"] - slope * (bot_pt["index"] - prim["x0"])
            y_end_par = y0_par + slope * ((n - 1) - prim["x0"])
            channel = {
                "upper": prim,
                "lower": {
                    "x0": prim["x0"], "y0": y0_par,
                    "x1": prim["x1"], "y1": y0_par + slope * (prim["x1"] - prim["x0"]),
                    "y_end": y_end_par,
                    "slope": slope,
                    "kind": "channel_lower",
                },
                "mid_end": (prim["y_end"] + y_end_par) / 2.0,
                "width": abs(prim["y_end"] - y_end_par),
            }

    # Direction / strength
    close = float(df["Close"].iloc[-1])
    direction = "NEUTRAL"
    strength = 40
    reasons = []

    if uptrends and (not downtrends or uptrends[0]["touches"] >= downtrends[0]["touches"]):
        direction = "BUY"
        strength = 50 + min(30, uptrends[0]["touches"] * 8)
        reasons.append(f"Primary uptrend ({uptrends[0]['touches']} touches)")
        if close > uptrends[0]["y_end"]:
            reasons.append("Price above primary support trendline")
            strength += 10
        else:
            reasons.append("Price testing/below support trendline")
    elif downtrends:
        direction = "SELL"
        strength = 50 + min(30, downtrends[0]["touches"] * 8)
        reasons.append(f"Primary downtrend ({downtrends[0]['touches']} touches)")
        if close < downtrends[0]["y_end"]:
            reasons.append("Price below primary resistance trendline")
            strength += 10
        else:
            reasons.append("Price testing/above resistance trendline")

    # Measured-move projections from last impulse leg
    projections = _measured_move_projections(df, pivots, direction)

    # Volume profile
    vp = compute_volume_profile(df.iloc[:-1])

    # Series arrays for charting (full length)
    upper_line = np.full(n, np.nan)
    lower_line = np.full(n, np.nan)
    mid_line = np.full(n, np.nan)
    if channel:
        u = channel.get("upper") or {}
        lo = channel.get("lower") or {}
        for i in range(n):
            if u:
                upper_line[i] = _line_value(u["x0"], u["y0"], u.get("x1", u["x0"] + 1), u.get("y1", u["y0"]), i)
            if lo:
                lower_line[i] = _line_value(lo["x0"], lo["y0"], lo.get("x1", lo["x0"] + 1), lo.get("y1", lo["y0"]), i)
            if not np.isnan(upper_line[i]) and not np.isnan(lower_line[i]):
                mid_line[i] = (upper_line[i] + lower_line[i]) / 2.0

    return {
        "direction": direction,
        "strength": min(100, strength),
        "reasons": reasons,
        "uptrends": uptrends,
        "downtrends": downtrends,
        "channel": channel,
        "projections": projections,
        "pivots": pivots[-12:],
        "volume_profile": vp,
        "upper_line": upper_line,
        "lower_line": lower_line,
        "middle_line": mid_line,
        "df": df,
        "mode": "channel" if channel else "lines",
    }


def _measured_move_projections(df, pivots, direction) -> List[Dict[str, Any]]:
    """
    Classic measured move + Fib extensions from last impulse leg.
    Distance D = |last swing extreme - prior swing|.
    Project from break/current: 1.0x, 1.618x, 2.618x.
    """
    if not pivots or len(pivots) < 2:
        return []
    last = pivots[-1]
    prev = pivots[-2]
    d = abs(last["price"] - prev["price"])
    if d <= 0:
        return []
    close = float(df["Close"].iloc[-1])
    projs = []
    mults = [(1.0, "P1 1.0x"), (1.618, "P2 1.618x"), (2.618, "P3 2.618x")]
    if direction == "BUY" or (direction == "NEUTRAL" and last["type"] == "low"):
        base = last["price"] if last["type"] == "low" else close
        for m, label in mults:
            projs.append({"price": base + d * m, "label": label, "mult": m, "side": "BUY"})
    elif direction == "SELL" or last["type"] == "high":
        base = last["price"] if last["type"] == "high" else close
        for m, label in mults:
            projs.append({"price": base - d * m, "label": label, "mult": m, "side": "SELL"})
    return projs


def build_position_container(family: Dict[str, Any], atr_mult_sl: float = 1.2) -> Optional[Dict[str, Any]]:
    """
    Build long/short position box from trendline family + projections.
    """
    if not family or family.get("error"):
        return None
    df = family.get("df")
    if df is None or df.empty:
        return None
    close = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else abs(float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]))
    direction = family.get("direction", "NEUTRAL")
    if direction not in ("BUY", "SELL"):
        return None

    projs = family.get("projections") or []
    channel = family.get("channel")

    if direction == "BUY":
        # Entry near lower channel / support, SL below, TPs at projections
        entry = close
        if channel and channel.get("lower"):
            entry = min(close, float(channel["lower"].get("y_end", close)))
        sl = entry - atr * atr_mult_sl
        tp1 = projs[0]["price"] if projs else entry + atr * 1.5
        tp2 = projs[1]["price"] if len(projs) > 1 else entry + atr * 3.0
        return {
            "side": "LONG",
            "direction": "BUY",
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "order_type": "LIMIT" if entry < close else "MARKET",
        }
    else:
        entry = close
        if channel and channel.get("upper"):
            entry = max(close, float(channel["upper"].get("y_end", close)))
        sl = entry + atr * atr_mult_sl
        tp1 = projs[0]["price"] if projs else entry - atr * 1.5
        tp2 = projs[1]["price"] if len(projs) > 1 else entry - atr * 3.0
        return {
            "side": "SHORT",
            "direction": "SELL",
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "order_type": "LIMIT" if entry > close else "MARKET",
        }


def format_trendline_report(family: Dict[str, Any], symbol: str) -> str:
    if family.get("error"):
        return family["error"]
    lines = [
        f"📐 TRENDLINE FAMILY  |  {symbol}",
        f"Direction: {family.get('direction')}  |  Strength: {family.get('strength', 0)}/100",
    ]
    for r in family.get("reasons") or []:
        lines.append(f"  • {r}")
    up = family.get("uptrends") or []
    dn = family.get("downtrends") or []
    lines.append(f"Uptrends: {len(up)}  |  Downtrends: {len(dn)}")
    if family.get("channel"):
        w = family["channel"].get("width")
        lines.append(f"Channel width: {w:.5f}" if w else "Channel: active")
    projs = family.get("projections") or []
    if projs:
        lines.append("Projections:")
        for p in projs:
            lines.append(f"  {p['label']}: {p['price']:.5f}")
    vp = family.get("volume_profile")
    if vp:
        lines.append(
            f"POC {vp['poc_price']:.5f} | VA {vp['value_area_low']:.5f}–{vp['value_area_high']:.5f}"
        )
    pos = build_position_container(family)
    if pos:
        lines.append(
            f"Position: {pos['side']}  Entry {pos['entry']:.5f}  SL {pos['sl']:.5f}  "
            f"TP1 {pos['tp1']:.5f}  TP2 {pos['tp2']:.5f}"
        )
    return "\n".join(lines)
