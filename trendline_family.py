"""
trendline_family.py
===================
Clean parallel-channel trendline family (MT5-style).

Goal: map the chart so direction reveals itself —
  one primary trendline + 2–3 true parallel members of the same family.
  Not a web of crossing independent lines.

Logic:
  1. ZigZag swings → candidate support (HL) and resistance (LH) lines
  2. Pick the strongest primary by touch count + recency
  3. Build the FAMILY = same slope, parallel offsets through other swings
  4. Price position vs family → direction (above = bullish structure, below = bearish)
  5. Measured-move projections from last impulse for targets
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from market_structure import zigzag_swings
from volume_profile import compute_volume_profile


def _line_value(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _count_touches(df: pd.DataFrame, x0: int, y0: float, x1: int, y1: float,
                   kind: str, tol_atr: float = 0.40) -> int:
    if df is None or len(df) < 5:
        return 0
    atr = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).values
    highs = df["High"].values
    lows = df["Low"].values
    touches = 0
    lo, hi = min(x0, x1), max(x0, x1)
    for i in range(lo, min(hi + 1, len(df))):
        lv = _line_value(x0, y0, x1, y1, i)
        a = float(atr[i]) if i < len(atr) and atr[i] > 0 else abs(y1 - y0) * 0.05
        tol = max(a * tol_atr, 1e-9)
        if kind == "support" and abs(lows[i] - lv) <= tol:
            touches += 1
        elif kind == "resistance" and abs(highs[i] - lv) <= tol:
            touches += 1
    return touches


def _fit_primary(pivots: List[Dict], kind: str, n: int, df: pd.DataFrame) -> Optional[Dict]:
    """Best 2-point primary line of given kind (support=lows, resistance=highs)."""
    pts = [p for p in pivots if p["type"] == ("low" if kind == "support" else "high")]
    if len(pts) < 2:
        return None
    best = None
    best_score = -1
    for i in range(len(pts) - 1):
        for j in range(i + 1, len(pts)):
            a, b = pts[i], pts[j]
            if b["index"] <= a["index"]:
                continue
            # Uptrend support needs higher low; downtrend resistance needs lower high
            if kind == "support" and b["price"] <= a["price"]:
                continue
            if kind == "resistance" and b["price"] >= a["price"]:
                continue
            slope = (b["price"] - a["price"]) / max(b["index"] - a["index"], 1)
            touches = _count_touches(df, a["index"], a["price"], b["index"], b["price"], kind)
            # Prefer more touches + more recent + longer span
            score = touches * 10 + (b["index"] / max(n, 1)) * 5 + (b["index"] - a["index"]) * 0.05
            if score > best_score:
                best_score = score
                y_end = _line_value(a["index"], a["price"], b["index"], b["price"], n - 1)
                best = {
                    "x0": a["index"], "y0": a["price"],
                    "x1": b["index"], "y1": b["price"],
                    "y_end": y_end,
                    "slope": slope,
                    "touches": touches,
                    "kind": kind,
                }
    return best


def _build_parallel_family(primary: Dict, pivots: List[Dict], n: int, max_members: int = 4) -> List[Dict]:
    """
    True parallel family: same slope as primary, each member anchored
    through a swing on the opposite side (or further on same side).
    This is what your MT5 screenshots show — one slope, multiple rails.
    """
    slope = primary["slope"]
    kind = primary["kind"]
    members = [primary]

    # Candidate anchors: swings that are not the primary anchors
    anchors = []
    for p in pivots:
        if p["index"] == primary["x0"] or p["index"] == primary["x1"]:
            continue
        anchors.append(p)

    # Offset of each anchor relative to primary line at that x
    seen_offsets = [0.0]  # primary offset = 0
    for p in anchors:
        y_on_primary = _line_value(primary["x0"], primary["y0"], primary["x1"], primary["y1"], p["index"])
        offset = p["price"] - y_on_primary
        # Skip near-duplicates
        if any(abs(offset - o) / max(abs(offset), abs(o), 1e-9) < 0.08 for o in seen_offsets):
            continue
        seen_offsets.append(offset)
        y0 = primary["y0"] + offset
        y1 = primary["y1"] + offset
        y_end = primary["y_end"] + offset
        members.append({
            "x0": primary["x0"], "y0": y0,
            "x1": primary["x1"], "y1": y1,
            "y_end": y_end,
            "slope": slope,
            "offset": offset,
            "kind": "parallel",
            "touches": 0,
        })
        if len(members) >= max_members:
            break

    # Sort by price level at chart end (lowest to highest)
    members.sort(key=lambda m: m["y_end"])
    return members


def build_trendline_family(df: pd.DataFrame, max_lines: int = 4) -> Dict[str, Any]:
    """
    Build one clean parallel family (ascending OR descending), not both mixed.
    Market reveals direction: price relative to the family rails.
    """
    if df is None or len(df) < 30:
        return {"error": "Insufficient data for trendline family", "direction": "NEUTRAL", "pivots": []}

    n = len(df)
    pivots = zigzag_swings(df, depth=4, deviation_atr=0.28)
    if len(pivots) < 4:
        pivots = zigzag_swings(df, depth=3, deviation_atr=0.20)

    support = _fit_primary(pivots, "support", n, df)
    resistance = _fit_primary(pivots, "resistance", n, df)

    # Choose which family dominates (more touches + price respect)
    close = float(df["Close"].iloc[-1])
    primary = None
    family_kind = "none"

    if support and resistance:
        # Prefer the one price is currently interacting with / stronger touches
        s_end = support["y_end"]
        r_end = resistance["y_end"]
        if support["touches"] >= resistance["touches"] and close >= s_end * 0.998:
            primary, family_kind = support, "ascending"
        elif resistance["touches"] > support["touches"] and close <= r_end * 1.002:
            primary, family_kind = resistance, "descending"
        elif close > (s_end + r_end) / 2:
            primary, family_kind = support, "ascending"
        else:
            primary, family_kind = resistance, "descending"
    elif support:
        primary, family_kind = support, "ascending"
    elif resistance:
        primary, family_kind = resistance, "descending"

    family_lines = []
    channel = None
    if primary:
        family_lines = _build_parallel_family(primary, pivots, n, max_members=max_lines)
        if len(family_lines) >= 2:
            channel = {
                "lower": family_lines[0],
                "upper": family_lines[-1],
                "mid_end": (family_lines[0]["y_end"] + family_lines[-1]["y_end"]) / 2.0,
                "width": abs(family_lines[-1]["y_end"] - family_lines[0]["y_end"]),
                "members": family_lines,
            }

    # Direction from family geometry (price reveals it)
    direction = "NEUTRAL"
    strength = 40
    reasons = []

    if primary and family_lines:
        lower = family_lines[0]["y_end"]
        upper = family_lines[-1]["y_end"]
        mid = (lower + upper) / 2.0
        if family_kind == "ascending":
            if close >= lower:
                direction = "BUY"
                strength = 55 + min(25, primary["touches"] * 7)
                reasons.append(f"Ascending family · {primary['touches']} touches on primary")
                if close > mid:
                    reasons.append("Price in upper half of channel — bullish control")
                    strength += 10
                else:
                    reasons.append("Price near support rail — watch bounce / break")
            else:
                direction = "SELL"
                strength = 60
                reasons.append("Price broke below ascending family — structure failure")
        else:  # descending
            if close <= upper:
                direction = "SELL"
                strength = 55 + min(25, primary["touches"] * 7)
                reasons.append(f"Descending family · {primary['touches']} touches on primary")
                if close < mid:
                    reasons.append("Price in lower half of channel — bearish control")
                    strength += 10
                else:
                    reasons.append("Price near resistance rail — watch reject / break")
            else:
                direction = "BUY"
                strength = 60
                reasons.append("Price broke above descending family — structure failure")

    projections = _measured_move_projections(df, pivots, direction)
    vp = compute_volume_profile(df.iloc[:-1])

    # Series for chart (only the parallel family — clean)
    upper_line = np.full(n, np.nan)
    lower_line = np.full(n, np.nan)
    mid_line = np.full(n, np.nan)
    if channel:
        u, lo = channel["upper"], channel["lower"]
        for i in range(n):
            upper_line[i] = _line_value(u["x0"], u["y0"], u["x1"], u["y1"], i)
            lower_line[i] = _line_value(lo["x0"], lo["y0"], lo["x1"], lo["y1"], i)
            mid_line[i] = (upper_line[i] + lower_line[i]) / 2.0

    return {
        "direction": direction,
        "strength": min(100, int(strength)),
        "reasons": reasons,
        "family_kind": family_kind,
        "family_lines": family_lines,  # the clean parallel set
        "uptrends": [primary] if family_kind == "ascending" and primary else [],
        "downtrends": [primary] if family_kind == "descending" and primary else [],
        "channel": channel,
        "projections": projections,
        "pivots": pivots[-20:],
        "volume_profile": vp,
        "upper_line": upper_line,
        "lower_line": lower_line,
        "middle_line": mid_line,
        "df": df,
        "mode": "channel" if channel else "lines",
    }


def _measured_move_projections(df, pivots, direction) -> List[Dict[str, Any]]:
    if not pivots or len(pivots) < 2:
        return []
    last, prev = pivots[-1], pivots[-2]
    d = abs(last["price"] - prev["price"])
    if d <= 0:
        return []
    close = float(df["Close"].iloc[-1])
    projs = []
    mults = [(1.0, "P1 1.0x"), (1.618, "P2 1.618x"), (2.618, "P3 2.618x")]
    if direction == "BUY":
        base = last["price"] if last["type"] == "low" else close
        for m, label in mults:
            projs.append({"price": base + d * m, "label": label, "mult": m, "side": "BUY"})
    elif direction == "SELL":
        base = last["price"] if last["type"] == "high" else close
        for m, label in mults:
            projs.append({"price": base - d * m, "label": label, "mult": m, "side": "SELL"})
    return projs


def build_position_container(family: Dict[str, Any], atr_mult_sl: float = 1.15) -> Optional[Dict[str, Any]]:
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
    lines = family.get("family_lines") or []

    if direction == "BUY":
        entry = close
        if lines:
            # Prefer nearest rail below price as limit entry
            below = [m["y_end"] for m in lines if m["y_end"] <= close]
            if below:
                entry = max(below)
        elif channel and channel.get("lower"):
            entry = min(close, float(channel["lower"].get("y_end", close)))
        sl = entry - atr * atr_mult_sl
        tp1 = projs[0]["price"] if projs else entry + atr * 1.5
        tp2 = projs[1]["price"] if len(projs) > 1 else entry + atr * 2.8
        return {"side": "LONG", "direction": "BUY", "entry": entry, "sl": sl,
                "tp1": tp1, "tp2": tp2, "order_type": "LIMIT" if entry < close - atr * 0.1 else "MARKET"}
    else:
        entry = close
        if lines:
            above = [m["y_end"] for m in lines if m["y_end"] >= close]
            if above:
                entry = min(above)
        elif channel and channel.get("upper"):
            entry = max(close, float(channel["upper"].get("y_end", close)))
        sl = entry + atr * atr_mult_sl
        tp1 = projs[0]["price"] if projs else entry - atr * 1.5
        tp2 = projs[1]["price"] if len(projs) > 1 else entry - atr * 2.8
        return {"side": "SHORT", "direction": "SELL", "entry": entry, "sl": sl,
                "tp1": tp1, "tp2": tp2, "order_type": "LIMIT" if entry > close + atr * 0.1 else "MARKET"}


def format_trendline_report(family: Dict[str, Any], symbol: str) -> str:
    if family.get("error"):
        return family["error"]
    lines = [
        f"📐 TRENDLINE FAMILY  |  {symbol}",
        f"Family: {family.get('family_kind', '—').upper()}  |  "
        f"Direction: {family.get('direction')}  |  Strength: {family.get('strength', 0)}/100",
    ]
    for r in family.get("reasons") or []:
        lines.append(f"  • {r}")
    n_rails = len(family.get("family_lines") or [])
    lines.append(f"Parallel rails: {n_rails}")
    if family.get("channel"):
        w = family["channel"].get("width")
        if w:
            lines.append(f"Channel width: {w:.5f}")
    projs = family.get("projections") or []
    if projs:
        lines.append("Projections: " + " · ".join(f"{p['label']} {p['price']:.5f}" for p in projs))
    vp = family.get("volume_profile")
    if vp:
        lines.append(f"POC {vp['poc_price']:.5f} | VA {vp['value_area_low']:.5f}–{vp['value_area_high']:.5f}")
    pos = build_position_container(family)
    if pos:
        lines.append(
            f"Position: {pos['side']}  Entry {pos['entry']:.5f}  SL {pos['sl']:.5f}  "
            f"TP1 {pos['tp1']:.5f}  TP2 {pos['tp2']:.5f}"
        )
    return "\n".join(lines)
