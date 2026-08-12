
"""
market_intelligence.py
======================
V2 deterministic market-intelligence layer for telegram-chart-bot.

Purpose:
- Detect market state: TREND / ACCUMULATION / MANIPULATION / DISTRIBUTION /
  EXPANSION / PULLBACK / RANGE.
- Map liquidity pools and sweeps.
- Confirm displacement.
- Track FVGs and OBs and promote invalidated OBs to breakers.
- Produce a compact, AI-ready structured report.

This module is intentionally deterministic. It does not predict the next move.
It describes what price has already confirmed.

Expected dataframe columns:
Open, High, Low, Close
Optional: Volume, ATR

No external dependencies beyond numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd


EPS = 1e-12


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if "ATR" in df.columns:
        x = pd.to_numeric(df["ATR"], errors="coerce")
        if x.notna().any():
            return x.ffill().bfill()
    h, l, c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _safe(v, default=None):
    try:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return default
        return float(v)
    except Exception:
        return default


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().title() for c in out.columns]
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=list(required)).reset_index(drop=True)
    out["__ATR"] = _atr(out)
    return out


def _local_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[dict], List[dict]]:
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    highs, lows = [], []
    n = len(df)
    for i in range(left, n-right):
        if h[i] >= np.max(h[i-left:i+right+1]) and h[i] > h[i-1] and h[i] >= h[i+1]:
            highs.append({"index": i, "price": float(h[i])})
        if l[i] <= np.min(l[i-left:i+right+1]) and l[i] < l[i-1] and l[i] <= l[i+1]:
            lows.append({"index": i, "price": float(l[i])})
    return highs, lows


def _cluster_levels(points: List[dict], tolerance: float) -> List[dict]:
    if not points:
        return []
    groups: List[List[dict]] = []
    for p in sorted(points, key=lambda x: x["price"]):
        placed = False
        for g in groups:
            center = np.mean([x["price"] for x in g])
            if abs(p["price"] - center) <= tolerance:
                g.append(p)
                placed = True
                break
        if not placed:
            groups.append([p])
    levels = []
    for g in groups:
        levels.append({
            "price": float(np.mean([x["price"] for x in g])),
            "touches": len(g),
            "first_index": int(min(x["index"] for x in g)),
            "last_index": int(max(x["index"] for x in g)),
        })
    return levels


def detect_liquidity(df: pd.DataFrame, lookback: int = 100) -> Dict[str, Any]:
    """Map equal highs/lows and recent swing liquidity, then detect a confirmed sweep."""
    x = df.tail(lookback).reset_index(drop=True)
    atr = float(x["__ATR"].iloc[-1])
    tol = max(atr * 0.18, EPS)
    ph, pl = _local_pivots(x, 2, 2)
    highs = _cluster_levels(ph, tol)
    lows = _cluster_levels(pl, tol)

    equal_highs = [z for z in highs if z["touches"] >= 2]
    equal_lows = [z for z in lows if z["touches"] >= 2]

    close = float(x["Close"].iloc[-1])
    high = float(x["High"].iloc[-1])
    low = float(x["Low"].iloc[-1])

    sweep = None
    candidates = sorted(equal_highs + equal_lows, key=lambda z: z["last_index"], reverse=True)
    for z in candidates[:8]:
        level = z["price"]
        # Sweep = wick through liquidity followed by close back inside.
        if high > level + tol * 0.15 and close < level:
            sweep = {"side": "BUY_SIDE", "level": level, "type": "BEARISH_SWEEP", "index": len(x)-1}
            break
        if low < level - tol * 0.15 and close > level:
            sweep = {"side": "SELL_SIDE", "level": level, "type": "BULLISH_SWEEP", "index": len(x)-1}
            break

    return {
        "equal_highs": equal_highs[-8:],
        "equal_lows": equal_lows[-8:],
        "recent_high_liquidity": sorted(highs, key=lambda z: z["last_index"], reverse=True)[:6],
        "recent_low_liquidity": sorted(lows, key=lambda z: z["last_index"], reverse=True)[:6],
        "sweep": sweep,
    }


def detect_displacement(df: pd.DataFrame, bars: int = 5) -> Dict[str, Any]:
    x = df.tail(max(10, bars + 2)).reset_index(drop=True)
    atr = x["__ATR"].replace(0, np.nan)
    body = (x["Close"] - x["Open"]).abs()
    rng = (x["High"] - x["Low"]).replace(0, np.nan)
    body_ratio = body / rng
    impulse = rng / atr
    close_location = np.where(
        x["Close"] >= x["Open"],
        (x["Close"] - x["Low"]) / rng,
        (x["High"] - x["Close"]) / rng,
    )
    score = (body_ratio.fillna(0) * 45 + impulse.fillna(0).clip(0, 3) / 3 * 40 +
             pd.Series(close_location).fillna(0).clip(0, 1) * 15)
    i = int(score.iloc[-1])
    bull = x["Close"].iloc[-1] > x["Open"].iloc[-1]
    strong = bool(body_ratio.iloc[-1] >= 0.60 and impulse.iloc[-1] >= 1.0 and score.iloc[-1] >= 65)
    return {
        "confirmed": strong,
        "direction": "BULLISH" if bull else "BEARISH",
        "score": max(0, min(100, i)),
        "body_ratio": float(body_ratio.iloc[-1]) if pd.notna(body_ratio.iloc[-1]) else 0.0,
        "range_atr": float(impulse.iloc[-1]) if pd.notna(impulse.iloc[-1]) else 0.0,
    }


def detect_fvgs(df: pd.DataFrame, lookback: int = 80) -> List[dict]:
    x = df.tail(lookback).reset_index(drop=True)
    atr = x["__ATR"].to_numpy(float)
    out = []
    for i in range(2, len(x)):
        # Bullish FVG: current low > high two bars ago.
        if x["Low"].iloc[i] > x["High"].iloc[i-2]:
            lo, hi = float(x["High"].iloc[i-2]), float(x["Low"].iloc[i])
            size = hi - lo
            if size >= max(atr[i] * 0.08, EPS):
                out.append({"type": "BULLISH_FVG", "low": lo, "high": hi, "index": i,
                            "size_atr": float(size / max(atr[i], EPS)),
                            "mitigated": bool(x["Low"].iloc[-1] <= hi)})
        # Bearish FVG: current high < low two bars ago.
        if x["High"].iloc[i] < x["Low"].iloc[i-2]:
            lo, hi = float(x["High"].iloc[i]), float(x["Low"].iloc[i-2])
            size = hi - lo
            if size >= max(atr[i] * 0.08, EPS):
                out.append({"type": "BEARISH_FVG", "low": lo, "high": hi, "index": i,
                            "size_atr": float(size / max(atr[i], EPS)),
                            "mitigated": bool(x["High"].iloc[-1] >= lo)})
    return out[-12:]


def detect_order_blocks_and_breakers(df: pd.DataFrame, lookback: int = 100) -> List[dict]:
    """
    OB = last opposite candle before a decisive displacement.
    Breaker = an OB later violated by a close and followed by directional
    displacement. The old zone is retained as a breaker candidate.
    """
    x = df.tail(lookback).reset_index(drop=True)
    atr = x["__ATR"].to_numpy(float)
    zones = []
    for i in range(1, len(x)):
        o, h, l, c = map(float, [x["Open"].iloc[i], x["High"].iloc[i], x["Low"].iloc[i], x["Close"].iloc[i]])
        po, ph, pl, pc = map(float, [x["Open"].iloc[i-1], x["High"].iloc[i-1], x["Low"].iloc[i-1], x["Close"].iloc[i-1]])
        a = max(float(atr[i]), EPS)
        body = abs(c-o)
        strong_bull = c > o and body/a >= 0.8 and c >= h - 0.25*max(h-l, EPS)
        strong_bear = c < o and body/a >= 0.8 and c <= l + 0.25*max(h-l, EPS)
        if strong_bull and pc < po:
            zones.append({"type": "BULLISH_OB", "low": pl, "high": ph, "index": i-1, "broken": False})
        elif strong_bear and pc > po:
            zones.append({"type": "BEARISH_OB", "low": pl, "high": ph, "index": i-1, "broken": False})

    close_now = float(x["Close"].iloc[-1])
    out = []
    for z in zones[-12:]:
        broken = False
        if z["type"] == "BULLISH_OB" and close_now < z["low"]:
            broken = True
        if z["type"] == "BEARISH_OB" and close_now > z["high"]:
            broken = True
        z = dict(z)
        z["broken"] = broken
        if broken:
            z["type"] = "BEARISH_BREAKER" if z["type"] == "BULLISH_OB" else "BULLISH_BREAKER"
        z["active"] = not broken
        z["distance_atr"] = float(min(abs(close_now-z["low"]), abs(close_now-z["high"])) /
                                  max(float(atr[-1]), EPS))
        out.append(z)
    return out


def _structure(df: pd.DataFrame) -> Dict[str, Any]:
    x = df.tail(120).reset_index(drop=True)
    ph, pl = _local_pivots(x, 3, 3)
    if len(ph) < 2 or len(pl) < 2:
        return {"bias": "NEUTRAL", "event": None, "structure": "UNDEFINED",
                "protected_high": None, "protected_low": None}
    hh = ph[-1]["price"] > ph[-2]["price"]
    hl = pl[-1]["price"] > pl[-2]["price"]
    lh = ph[-1]["price"] < ph[-2]["price"]
    ll = pl[-1]["price"] < pl[-2]["price"]
    if hh and hl:
        bias, struct = "BULLISH", "HH_HL"
    elif lh and ll:
        bias, struct = "BEARISH", "LH_LL"
    else:
        bias, struct = "NEUTRAL", "MIXED"

    close = float(x["Close"].iloc[-1])
    event = None
    if close > ph[-1]["price"]:
        event = "BULLISH_BREAK"
    elif close < pl[-1]["price"]:
        event = "BEARISH_BREAK"

    return {
        "bias": bias,
        "event": event,
        "structure": struct,
        "protected_high": ph[-1]["price"],
        "protected_low": pl[-1]["price"],
        "swing_highs": ph[-6:],
        "swing_lows": pl[-6:],
    }


def _base_metrics(df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    x = df.tail(window)
    atr = float(x["__ATR"].iloc[-1])
    hi, lo = float(x["High"].max()), float(x["Low"].min())
    width_atr = (hi-lo) / max(atr, EPS)
    net = float(x["Close"].iloc[-1] - x["Close"].iloc[0]) / max(atr, EPS)
    return {"range_width_atr": width_atr, "net_move_atr": net, "atr": atr}


def classify_market_state(df: pd.DataFrame, structure: Dict[str, Any],
                           liquidity: Dict[str, Any], displacement: Dict[str, Any]) -> Dict[str, Any]:
    m = _base_metrics(df)
    width, net, atr = m["range_width_atr"], m["net_move_atr"], m["atr"]
    sweep = liquidity.get("sweep")

    # Confirmed state precedence: manipulation -> expansion -> pullback/trend -> range/base.
    if sweep and displacement["confirmed"]:
        state = "MANIPULATION"
        phase = "SWEEP_CONFIRMED"
        bias = "BEARISH" if sweep["type"] == "BEARISH_SWEEP" else "BULLISH"
        reason = f"{sweep['side']} liquidity swept and {displacement['direction'].lower()} displacement confirmed."
    elif displacement["confirmed"] and abs(net) >= 1.5:
        state = "EXPANSION"
        phase = "DISPLACEMENT"
        bias = displacement["direction"]
        reason = "Large directional displacement relative to ATR."
    elif width <= 4.0 and abs(net) <= 1.2:
        state = "RANGE"
        phase = "BASE"
        bias = structure.get("bias", "NEUTRAL")
        reason = "Price is contained in a relatively narrow ATR-normalized range."
    elif structure.get("bias") in ("BULLISH", "BEARISH"):
        bias = structure["bias"]
        state = "TREND"
        phase = "CONTINUATION"
        reason = f"Structure is {structure['structure']} with {bias.lower()} directional bias."
    else:
        state = "TRANSITION"
        phase = "UNCONFIRMED"
        bias = "NEUTRAL"
        reason = "Structure and displacement do not yet agree."

    return {
        "state": state,
        "phase": phase,
        "bias": bias,
        "reason": reason,
        **m,
    }


def score_setup(report: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    reasons = []
    bias = report["market_state"]["bias"]
    sweep = report["liquidity"].get("sweep")
    disp = report["displacement"]
    st = report["structure"]
    poi = report["poi"]

    if bias != "NEUTRAL":
        score += 15; reasons.append("Directional market state confirmed.")
    if sweep:
        score += 25; reasons.append("Liquidity sweep confirmed.")
    if disp["confirmed"]:
        score += 25; reasons.append("Displacement confirmed.")
    if st.get("event"):
        score += 15; reasons.append(f"Structure event: {st['event']}.")
    if poi:
        score += 20; reasons.append("Fresh POI available.")

    score = min(score, 100)
    grade = "A+" if score >= 90 else "A" if score >= 75 else "B" if score >= 55 else "C"
    return {"score": score, "grade": grade, "reasons": reasons}


def analyze_market_intelligence(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main entry point.

    Returns a JSON-serializable dictionary suitable for Telegram, chart
    annotations, execution gating, or an AI interpreter.
    """
    x = _norm(df)
    if len(x) < 40:
        return {"error": "Need at least 40 candles for V2 market intelligence."}

    structure = _structure(x)
    liquidity = detect_liquidity(x)
    displacement = detect_displacement(x)
    fvgs = detect_fvgs(x)
    zones = detect_order_blocks_and_breakers(x)
    market_state = classify_market_state(x, structure, liquidity, displacement)

    current = float(x["Close"].iloc[-1])
    directional_pois = []
    for z in zones:
        if z["type"].endswith("BREAKER"):
            if z["low"] <= current <= z["high"] or z["distance_atr"] <= 1.5:
                directional_pois.append(z)
        elif not z["broken"] and z["distance_atr"] <= 1.5:
            directional_pois.append(z)

    for f in fvgs:
        if not f["mitigated"] and f["low"] <= current + 2*x["__ATR"].iloc[-1] and f["high"] >= current - 2*x["__ATR"].iloc[-1]:
            directional_pois.append(f)

    report = {
        "version": "2.0",
        "current_price": current,
        "market_state": market_state,
        "structure": structure,
        "liquidity": liquidity,
        "displacement": displacement,
        "fvgs": fvgs,
        "zones": zones,
        "poi": directional_pois[-6:],
    }
    report["setup"] = score_setup(report)

    # Explicit execution permission: never turn a mere prediction into a trade.
    report["execution"] = {
        "permission": bool(
            market_state["state"] in ("MANIPULATION", "EXPANSION", "TREND")
            and market_state["bias"] != "NEUTRAL"
            and report["setup"]["score"] >= 55
        ),
        "side": "BUY" if market_state["bias"] == "BULLISH" else
                "SELL" if market_state["bias"] == "BEARISH" else "WAIT",
        "rule": "Wait for confirmed sweep/displacement/structure; no predictive entry.",
    }
    return report


def format_intelligence_report(report: Dict[str, Any], symbol: str = "") -> str:
    if report.get("error"):
        return f"❌ {report['error']}"
    ms, st, liq, disp, setup, ex = (
        report["market_state"], report["structure"], report["liquidity"],
        report["displacement"], report["setup"], report["execution"]
    )
    lines = [
        "══════════════════════════════",
        f"🧠 MARKET INTELLIGENCE V2 | {symbol}".strip(),
        "══════════════════════════════",
        f"STATE       : {ms['state']} / {ms['phase']}",
        f"BIAS        : {ms['bias']}",
        f"STRUCTURE   : {st.get('structure')} | {st.get('event') or 'NO NEW BREAK'}",
        f"LIQUIDITY   : {liq.get('sweep')['type'] if liq.get('sweep') else 'NO CONFIRMED SWEEP'}",
        f"DISPLACEMENT: {'CONFIRMED' if disp['confirmed'] else 'NOT CONFIRMED'} ({disp['direction']})",
        f"SETUP       : {setup['grade']} | {setup['score']}/100",
        f"EXECUTION   : {'🟢 ' + ex['side'] if ex['permission'] else '⏳ WAIT'}",
    ]
    if report.get("poi"):
        p = report["poi"][-1]
        if "low" in p and "high" in p:
            lines.append(f"POI         : {p.get('type')} {p['low']:.5f}–{p['high']:.5f}")
    lines.append("WHY:")
    lines.extend(f"  • {r}" for r in setup["reasons"][:6])
    lines.append("══════════════════════════════")
    return "\n".join(lines)


__all__ = [
    "analyze_market_intelligence",
    "format_intelligence_report",
    "classify_market_state",
    "detect_liquidity",
    "detect_displacement",
    "detect_fvgs",
    "detect_order_blocks_and_breakers",
]
