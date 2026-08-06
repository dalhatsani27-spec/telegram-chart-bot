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
from direction_banner import direction_banner
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


def _detect_displacement(df: pd.DataFrame, sweep: Optional[Dict], lookback: int = 12) -> Optional[Dict]:
    """
    Displacement = a strong, wide-bodied momentum candle moving away from the
    sweep, in the direction of the anticipated reversal. Required step 2 of
    the ICT Silver Bullet sequence (sweep -> displacement -> FVG -> retrace),
    but until now the score never actually checked for it -- alignment alone
    could pass without a real displacement leg ever happening.
    """
    if df is None or len(df) < lookback + 2 or sweep is None:
        return None
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else abs(float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]))
    if atr <= 0:
        return None
    recent = df.iloc[-lookback:]
    want_dir = sweep["direction_hint"]  # BUY after SSL sweep, SELL after BSL sweep
    best = None
    for i in range(len(recent)):
        row = recent.iloc[i]
        body = row["Close"] - row["Open"]
        body_ratio = abs(body) / atr
        candle_dir = "BUY" if body > 0 else "SELL"
        if candle_dir != want_dir:
            continue
        if body_ratio >= 1.3 and (best is None or body_ratio > best["body_ratio_atr"]):
            best = {
                "index": recent.index[i],
                "body_ratio_atr": round(float(body_ratio), 2),
                "direction": candle_dir,
            }
    if best:
        best["note"] = f"Displacement candle ({best['body_ratio_atr']}x ATR body) confirms {best['direction']}"
    return best


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
    displacement = _detect_displacement(df, sweep)

    # Enrich with Institutional Structure Engine when enough bars (dynamic
    # liquidity / manipulation / acceptance — same pipeline as Trendline/AMD)
    ise = None
    if len(df) >= 60:
        try:
            from structure_engine import run_structure_engine
            ise = run_structure_engine(df)
            if ise and not ise.get("error"):
                # Prefer ISE sweep when local detector missed it
                if sweep is None and ise.get("sweep"):
                    sw = ise["sweep"]
                    sweep = {
                        "side": sw.get("side", "BSL"),
                        "level": sw.get("level"),
                        "direction_hint": sw.get("direction_hint", "NEUTRAL"),
                        "note": sw.get("note", "ISE liquidity sweep"),
                    }
                # Use ISE impulse as displacement confirmation when candle scan missed
                if displacement is None and ise.get("impulse") and not ise["impulse"].get("weak"):
                    imp = ise["impulse"]
                    displacement = {
                        "direction": imp["direction"],
                        "body_ratio_atr": imp.get("length_atr", 0),
                        "note": f"ISE impulse displacement ({imp['length_atr']}x ATR / {imp['bars']} bars)",
                    }
        except Exception:
            ise = None

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
        score += 20
        reasons.append(sweep["note"])
        direction = sweep["direction_hint"]
    else:
        reasons.append("No liquidity sweep detected — sequence step 1 missing")

    if displacement:
        score += 20
        reasons.append(displacement["note"])
    elif sweep:
        reasons.append("No displacement candle after the sweep — sequence step 2 missing, likely too early")

    if window_fvgs:
        score += 15
        z = window_fvgs[0]
        reasons.append(f"FVG present inside window ({z.get('bias', '')})")
        if direction == "NEUTRAL":
            direction = "BUY" if str(z.get("bias", "")).upper() in ("BULLISH", "BUY") else "SELL"
    elif fvgs and inside:
        score += 8
        reasons.append("FVG exists but may be outside strict window bars")

    mss_confirmed = False
    if structure and structure.get("bias"):
        event = structure.get("last_event")
        event_bias = structure.get("event_bias")
        if event == "MSS" and event_bias:
            mss_dir = "BUY" if event_bias == "BULLISH" else "SELL"
            if mss_dir == direction:
                mss_confirmed = True
                score += 15
                reasons.append(f"MSS confirms {event_bias} shift — full sequence intact")
        if structure["bias"] == "BULLISH" and direction == "BUY":
            score += 10
            reasons.append("Structure aligned bullish")
        elif structure["bias"] == "BEARISH" and direction == "SELL":
            score += 10
            reasons.append("Structure aligned bearish")
        else:
            reasons.append(f"Structure: {structure.get('note', structure.get('bias'))}")

    # ISE acceptance boost when available
    if ise and not ise.get("error"):
        acc = ise.get("acceptance") or {}
        man = ise.get("manipulation") or {}
        if man.get("confirmed"):
            score += 5
            reasons.append("ISE manipulation confirmed")
        if acc.get("accepted"):
            score += 8
            reasons.append(f"ISE acceptance: {acc.get('note', 'held')}")
            if ise.get("direction") in ("BUY", "SELL") and direction == "NEUTRAL":
                direction = ise["direction"]
        if ise.get("valid") and ise.get("direction") == direction:
            score = min(100, score + 5)
            reasons.append("ISE full path aligned with SB direction")

    # Full ICT sequence requires the sweep -> displacement chain, not just a
    # score threshold reached through alignment alone.
    valid = inside and score >= 55 and direction in ("BUY", "SELL") and sweep is not None and displacement is not None

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "now_ny": now_ny.strftime("%Y-%m-%d %H:%M"),
        "window": window,
        "inside_window": inside,
        "next_window": next_name,
        "minutes_to_next": mins_left,
        "sweep": sweep,
        "displacement": displacement,
        "mss_confirmed": mss_confirmed,
        "fvgs": fvgs,
        "window_fvgs": window_fvgs,
        "order_blocks": obs,
        "structure": structure,
        "direction": direction,
        "score": score,
        "valid": valid,
        "reasons": reasons,
        "df": df,
        "ise": ise,
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

    lines.append(direction_banner(analysis['direction'], extra=analysis['symbol']))
    lines.append(f"Score: {analysis['score']}/100  |  Valid: {'YES' if analysis['valid'] else 'NO'}")

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
