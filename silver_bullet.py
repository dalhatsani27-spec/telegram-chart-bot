"""
silver_bullet.py
================
ICT Silver Bullet strategy engine.

Three official 60-minute windows (New York local time):
  London SB   : 03:00 – 04:00
  NY AM SB    : 10:00 – 11:00   (highest probability)
  NY PM SB    : 14:00 – 15:00

Required sequence inside the window:
  1. Liquidity sweep (BSL or SSL)
  2. Displacement + Market Structure Shift
  3. Fair Value Gap formed inside the window
  4. Retracement into the FVG → entry
  5. Target = opposing liquidity
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from market_structure import analyse_structure, find_swings
from smc_zones import detect_fvgs, detect_order_blocks
import mt5_data


# Silver Bullet windows in New York time (hour start inclusive, end exclusive)
SB_WINDOWS_NY = [
    {"name": "London SB", "start": 3, "end": 4, "priority": 2},
    {"name": "NY AM SB", "start": 10, "end": 11, "priority": 1},  # highest
    {"name": "NY PM SB", "start": 14, "end": 15, "priority": 3},
]


def _ny_now() -> datetime:
    """Approximate current New York time (handles EST/EDT roughly via UTC-4/UTC-5)."""
    utc = datetime.now(timezone.utc)
    # Simple DST approximation: Mar–Nov ≈ EDT (UTC-4), else EST (UTC-5)
    month = utc.month
    if 3 <= month <= 11:
        return utc - timedelta(hours=4)
    return utc - timedelta(hours=5)


def current_silver_bullet_window(now_ny: Optional[datetime] = None) -> Optional[Dict]:
    now_ny = now_ny or _ny_now()
    h = now_ny.hour
    for w in SB_WINDOWS_NY:
        if w["start"] <= h < w["end"]:
            return w
    return None


def minutes_to_next_window(now_ny: Optional[datetime] = None) -> Tuple[str, int]:
    now_ny = now_ny or _ny_now()
    h, m = now_ny.hour, now_ny.minute
    current_mins = h * 60 + m
    candidates = []
    for w in SB_WINDOWS_NY:
        start_mins = w["start"] * 60
        delta = start_mins - current_mins
        if delta <= 0:
            delta += 24 * 60
        candidates.append((w["name"], delta))
    candidates.sort(key=lambda x: x[1])
    return candidates[0]


def _detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> Optional[Dict]:
    """Detect recent sweep of a swing high (BSL) or swing low (SSL)."""
    if df is None or len(df) < lookback + 5:
        return None
    swings = find_swings(df.iloc[-(lookback + 10):], left=2, right=2)
    if not swings:
        return None

    recent = df.iloc[-8:]
    highs = recent["High"].values
    lows = recent["Low"].values
    closes = recent["Close"].values

    swing_highs = [s for s in swings if s["type"] == "high"]
    swing_lows = [s for s in swings if s["type"] == "low"]

    # Buy-side liquidity sweep (raid above swing high then close back)
    for sh in reversed(swing_highs[-4:]):
        level = sh["price"]
        for i in range(len(recent)):
            if highs[i] > level * 1.00015 and closes[i] < level:
                return {
                    "side": "BSL",
                    "level": level,
                    "direction_hint": "SELL",
                    "note": f"Buy-side liquidity swept at {level:.5f}",
                }

    # Sell-side liquidity sweep
    for sl in reversed(swing_lows[-4:]):
        level = sl["price"]
        for i in range(len(recent)):
            if lows[i] < level * 0.99985 and closes[i] > level:
                return {
                    "side": "SSL",
                    "level": level,
                    "direction_hint": "BUY",
                    "note": f"Sell-side liquidity swept at {level:.5f}",
                }
    return None


def _fvg_inside_window(fvgs: List[Dict], df: pd.DataFrame, window: Dict) -> List[Dict]:
    """Keep only FVGs whose candle index falls inside the current SB window (approx)."""
    if not fvgs or df is None or df.empty:
        return []
    # We approximate: last ~12 bars are "inside" the 1-hour window on M5/M15
    n = len(df)
    cutoff = max(0, n - 16)
    return [z for z in fvgs if int(z.get("index", 0)) >= cutoff]


def run_silver_bullet_analysis(symbol: str, timeframe: str = "5min") -> Dict[str, Any]:
    """
    Full Silver Bullet package on the given timeframe (default M5).
    """
    symbol = symbol.strip().upper()
    now_ny = _ny_now()
    window = current_silver_bullet_window(now_ny)
    next_name, mins_left = minutes_to_next_window(now_ny)

    df = mt5_data.fetch_candles(symbol, timeframe, count=200)
    if df is None or df.empty or len(df) < 40:
        return {"error": f"Insufficient data for Silver Bullet on {symbol}."}

    structure = analyse_structure(df, left=2, right=2, lookback=50)
    fvgs = detect_fvgs(df, min_gap_atr=0.10, max_zones=8)
    obs = detect_order_blocks(df, structure=structure, max_zones=5)
    sweep = _detect_liquidity_sweep(df)

    inside = window is not None
    window_fvgs = _fvg_inside_window(fvgs, df, window) if inside else []

    # Score the setup
    score = 0
    reasons = []
    direction = "NEUTRAL"

    if inside:
        score += 30
        reasons.append(f"Inside {window['name']} window")
    else:
        reasons.append(f"Outside SB window — next: {next_name} in ~{mins_left} min")

    if sweep:
        score += 25
        reasons.append(sweep["note"])
        direction = sweep["direction_hint"]

    if window_fvgs:
        score += 25
        z = window_fvgs[0]
        reasons.append(f"FVG present inside window ({z.get('bias', '')})")
        if direction == "NEUTRAL":
            direction = "BUY" if str(z.get("bias", "")).upper() in ("BULLISH", "BUY") else "SELL"
    elif fvgs and inside:
        score += 10
        reasons.append("FVG exists but may be outside strict window bars")

    if structure and structure.get("bias"):
        if structure["bias"] == "BULLISH" and direction == "BUY":
            score += 15
            reasons.append("Structure aligned bullish")
        elif structure["bias"] == "BEARISH" and direction == "SELL":
            score += 15
            reasons.append("Structure aligned bearish")
        else:
            reasons.append(f"Structure: {structure.get('note', structure.get('bias'))}")

    valid = inside and score >= 55 and direction in ("BUY", "SELL")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "now_ny": now_ny.strftime("%Y-%m-%d %H:%M"),
        "window": window,
        "inside_window": inside,
        "next_window": next_name,
        "minutes_to_next": mins_left,
        "sweep": sweep,
        "fvgs": fvgs,
        "window_fvgs": window_fvgs,
        "order_blocks": obs,
        "structure": structure,
        "direction": direction,
        "score": score,
        "valid": valid,
        "reasons": reasons,
        "df": df,
    }


def format_silver_bullet_report(analysis: Dict[str, Any]) -> str:
    if "error" in analysis:
        return analysis["error"]

    lines = []
    lines.append(f"⚡ ICT SILVER BULLET  |  {analysis['symbol']}  |  {analysis['timeframe']}")
    lines.append(f"NY Time: {analysis['now_ny']}")

    if analysis["inside_window"]:
        w = analysis["window"]
        lines.append(f"Window: ✅ {w['name']} (active)")
    else:
        lines.append(f"Window: ❌ Outside  |  Next: {analysis['next_window']} in ~{analysis['minutes_to_next']} min")

    lines.append(f"Direction: {analysis['direction']}  |  Score: {analysis['score']}/100  |  Valid: {'YES' if analysis['valid'] else 'NO'}")

    if analysis.get("sweep"):
        lines.append(f"Sweep: {analysis['sweep']['note']}")

    n_fvg = len(analysis.get("window_fvgs") or [])
    lines.append(f"FVGs in window: {n_fvg}")

    for r in analysis.get("reasons") or []:
        lines.append(f"  • {r}")

    if analysis["valid"]:
        lines.append("Setup: READY — look for retrace into FVG for entry")
    else:
        lines.append("Setup: WAIT — conditions not complete")

    return "\n".join(lines)


def build_silver_bullet_ticket(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a trade ticket if the setup is valid."""
    if not analysis.get("valid"):
        return None

    df = analysis.get("df")
    if df is None or df.empty:
        return None

    direction = analysis["direction"]
    close = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else abs(float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]))

    # Prefer first window FVG mid as entry zone
    entry = close
    window_fvgs = analysis.get("window_fvgs") or []
    if window_fvgs:
        z = window_fvgs[0]
        entry = (float(z["top"]) + float(z["bottom"])) / 2.0

    if direction == "BUY":
        sl = entry - atr * 1.2
        tp1 = entry + atr * 1.8
        tp2 = entry + atr * 3.0
    else:
        sl = entry + atr * 1.2
        tp1 = entry - atr * 1.8
        tp2 = entry - atr * 3.0

    return {
        "symbol": analysis["symbol"],
        "direction": direction,
        "strategy": "ICT Silver Bullet",
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "score": analysis["score"],
        "reasons": analysis.get("reasons") or [],
        "window": (analysis.get("window") or {}).get("name", ""),
        "order_type": "LIMIT",
    }
