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
    """Best 2-point primary line of given kind (support=lows, resistance=highs).

    Professional validation standard: a 2-point line is only a *candidate* --
    it takes a 3rd touch for traders to actually respect it as real structure.
    We still return 2-touch lines (better than nothing), but tag them
    "unconfirmed" so downstream scoring/reporting can be honest about it.
    5+ touches is flagged "crowded": the level has been tested so many times
    the order flow defending it is likely used up, and the next test is
    statistically more likely to fail than hold.
    """
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
            # Prefer more touches + more recent + longer span, but back off once
            # touches go past the 3-4 "sweet spot" -- a crowded level scores
            # slightly lower than a freshly-confirmed one at the same touch count.
            touch_score = touches * 10
            if touches >= 5:
                touch_score -= (touches - 4) * 3  # fatigue penalty, doesn't erase the line
            score = touch_score + (b["index"] / max(n, 1)) * 5 + (b["index"] - a["index"]) * 0.05
            if score > best_score:
                best_score = score
                y_end = _line_value(a["index"], a["price"], b["index"], b["price"], n - 1)
                if touches < 3:
                    quality = "unconfirmed"
                elif touches <= 4:
                    quality = "confirmed"
                else:
                    quality = "crowded"
                best = {
                    "x0": a["index"], "y0": a["price"],
                    "x1": b["index"], "y1": b["price"],
                    "y_end": y_end,
                    "slope": slope,
                    "touches": touches,
                    "confirmed": touches >= 3,
                    "quality": quality,
                    "kind": kind,
                }
    return best


def _fit_line_any_slope(pivots: List[Dict], kind: str, n: int, df: pd.DataFrame) -> Optional[Dict]:
    """Best 2-point line through swing highs or lows, ANY slope direction.

    _fit_primary locks support to rising / resistance to falling only, which
    is correct for a parallel channel but can never produce the two
    independent-slope rails a trader draws by hand for a wedge or triangle
    (e.g. an ascending-highs line paired with a steeper ascending-lows line
    = rising wedge). This is the same touch-scored 2-point fit, just without
    that directional constraint.
    """
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
            slope = (b["price"] - a["price"]) / max(b["index"] - a["index"], 1)
            touches = _count_touches(df, a["index"], a["price"], b["index"], b["price"], kind)
            touch_score = touches * 10
            if touches >= 5:
                touch_score -= (touches - 4) * 3
            score = touch_score + (b["index"] / max(n, 1)) * 5 + (b["index"] - a["index"]) * 0.05
            if score > best_score:
                best_score = score
                y_end = _line_value(a["index"], a["price"], b["index"], b["price"], n - 1)
                quality = "unconfirmed" if touches < 3 else ("confirmed" if touches <= 4 else "crowded")
                best = {
                    "x0": a["index"], "y0": a["price"],
                    "x1": b["index"], "y1": b["price"],
                    "y_end": y_end, "slope": slope, "touches": touches,
                    "confirmed": touches >= 3, "quality": quality, "kind": kind,
                }
    return best


def _detect_converging_wedge(pivots: List[Dict], df: pd.DataFrame, n: int) -> Optional[Dict]:
    """
    Detect two independent-slope trendlines (one off the lows, one off the
    highs) that converge toward an apex -- wedges and triangles. This is
    distinct from _build_parallel_family, which only ever produces rails
    that share the primary's slope. A hand-drawn chart very often shows
    exactly this shape (two lines of different steepness meeting), and the
    old single-family logic could never reproduce it.
    """
    lower = _fit_line_any_slope(pivots, "support", n, df)
    upper = _fit_line_any_slope(pivots, "resistance", n, df)
    if not lower or not upper or lower["touches"] < 2 or upper["touches"] < 2:
        return None

    start_x = max(min(lower["x0"], upper["x0"]), 0)
    gap_start = (_line_value(upper["x0"], upper["y0"], upper["x1"], upper["y1"], start_x) -
                 _line_value(lower["x0"], lower["y0"], lower["x1"], lower["y1"], start_x))
    gap_end = upper["y_end"] - lower["y_end"]
    if gap_start <= 0 or gap_end <= 0:
        return None  # lines have already crossed -- not a clean wedge anymore
    if gap_end >= gap_start * 0.92:
        return None  # not meaningfully converging -- let the parallel-channel path handle it

    slope_l, slope_u = lower["slope"], upper["slope"]
    flat = lambda s: abs(s) < 1e-9
    if slope_l > 0 and slope_u > 0 and slope_l > slope_u:
        pattern, bias = "Rising Wedge", "SELL"
    elif slope_l < 0 and slope_u < 0 and slope_u < slope_l:
        pattern, bias = "Falling Wedge", "BUY"
    elif slope_l > 0 and slope_u < 0:
        pattern, bias = "Symmetrical Triangle", "NEUTRAL"
    elif flat(slope_l) and slope_u < 0:
        pattern, bias = "Descending Triangle", "SELL"
    elif slope_l > 0 and flat(slope_u):
        pattern, bias = "Ascending Triangle", "BUY"
    else:
        pattern, bias = "Converging Channel", "NEUTRAL"

    apex_x = None
    if abs(slope_l - slope_u) > 1e-9:
        apex_x = (n - 1) + (upper["y_end"] - lower["y_end"]) / (slope_l - slope_u)

    return {
        "pattern": pattern, "bias": bias,
        "lower": lower, "upper": upper,
        "gap_start": round(gap_start, 5), "gap_end": round(gap_end, 5),
        "apex_index": apex_x,
    }


def _detect_horizontal_levels(df: pd.DataFrame, pivots: List[Dict], n: int,
                               max_levels: int = 4, tol_atr: float = 0.45) -> List[Dict]:
    """
    Cluster ALL swing pivots across the full chart history by price
    proximity, not just the last few swings. A level a trader marks by eye
    usually earned that mark by getting tested repeatedly over the *life*
    of the chart -- a flip zone from weeks ago that's still respected is
    exactly the kind of line the old "last 6 pivots, top 2 highs/lows"
    logic would silently drop once it aged out of that recent window.
    """
    if not pivots or df is None or len(df) < 10:
        return []
    atr = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).values
    avg_atr = float(np.nanmean(atr[-50:])) if len(atr) else 0.0
    tol = max(avg_atr * tol_atr, 1e-9)

    clusters: List[Dict] = []
    for p in pivots:
        price = float(p["price"])
        placed = False
        for c in clusters:
            if abs(price - c["price"]) <= tol:
                c["touches"].append(p)
                c["price"] = float(np.mean([t["price"] for t in c["touches"]]))
                placed = True
                break
        if not placed:
            clusters.append({"price": price, "touches": [p]})

    close = float(df["Close"].iloc[-1])
    levels = []
    for c in clusters:
        n_touch = len(c["touches"])
        if n_touch < 2:
            continue
        first_idx = min(t["index"] for t in c["touches"])
        last_idx = max(t["index"] for t in c["touches"])
        span = last_idx - first_idx
        recency = last_idx / max(n, 1)
        # Durability (span) and touch count matter more than raw recency --
        # this is what lets an older, well-tested zone outscore a level
        # that just happened to form in the last handful of candles.
        score = n_touch * 10 + span * 0.08 + recency * 5
        quality = "unconfirmed" if n_touch < 3 else ("confirmed" if n_touch <= 4 else "crowded")
        levels.append({
            "price": c["price"], "touches": n_touch, "span": span,
            "first_index": first_idx, "last_index": last_idx,
            "side": "resistance" if c["price"] >= close else "support",
            "quality": quality, "score": round(score, 2),
        })
    levels.sort(key=lambda l: l["score"], reverse=True)
    return levels[:max_levels]


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
        # Require meaningful spacing vs primary (avoid cluttered near-duplicates)
        if any(abs(offset - o) / max(abs(y_on_primary) * 0.002, abs(o), 1e-9) < 0.35 for o in seen_offsets):
            continue
        if abs(offset) < abs(y_on_primary) * 0.0015:  # too tight to primary
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



def _detect_mw_pattern(pivots, df):
    """
    Detect simple M (double top) or W (double bottom) and neckline.
    M: two swing highs near same price, neckline = swing low between them.
    W: two swing lows near same price, neckline = swing high between them.
    """
    if not pivots or len(pivots) < 3 or df is None or len(df) < 20:
        return None
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else abs(float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]))
    tol = max(atr * 0.35, 1e-9)
    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]

    # Double top (M) — last two significant highs
    if len(highs) >= 2:
        for i in range(len(highs) - 1, 0, -1):
            h2, h1 = highs[i], highs[i - 1]
            if abs(h2["price"] - h1["price"]) <= tol and h2["index"] > h1["index"]:
                between = [p for p in lows if h1["index"] < p["index"] < h2["index"]]
                if between:
                    neck = min(between, key=lambda p: p["price"])
                    return {
                        "pattern": "M",
                        "name": "Double Top (M)",
                        "left": h1, "right": h2,
                        "neckline": neck["price"],
                        "neck_index": neck["index"],
                        "bias": "SELL",
                        "note": f"M pattern — neckline at {neck['price']:.5f}",
                    }

    # Double bottom (W)
    if len(lows) >= 2:
        for i in range(len(lows) - 1, 0, -1):
            l2, l1 = lows[i], lows[i - 1]
            if abs(l2["price"] - l1["price"]) <= tol and l2["index"] > l1["index"]:
                between = [p for p in highs if l1["index"] < p["index"] < l2["index"]]
                if between:
                    neck = max(between, key=lambda p: p["price"])
                    return {
                        "pattern": "W",
                        "name": "Double Bottom (W)",
                        "left": l1, "right": l2,
                        "neckline": neck["price"],
                        "neck_index": neck["index"],
                        "bias": "BUY",
                        "note": f"W pattern — neckline at {neck['price']:.5f}",
                    }
    return None

def _grade_breakout(df: pd.DataFrame, line: Dict, kind: str, n: int) -> Dict[str, Any]:
    """
    Grade how much to trust a close beyond the family rail, instead of
    treating every cross as an equal, instant signal (the #1 cause of
    trendline whipsaws per standard breakout-trading practice):

      - penetration_atr : how far beyond the line the close is, in ATR --
        a close that's barely beyond (wick-through territory) is graded
        weak even though it technically "broke" the line.
      - consecutive      : how many bars in a row have closed beyond it --
        1 bar is a first break, 2+ is starting to look real.
      - body_ratio       : candle body vs full range on the break bar --
        a small body with long wicks against the break direction is a
        classic fakeout signature.
      - retest_level     : the rail's current price -- where a limit order
        would sit if waiting for the break-and-retest entry instead of
        chasing the break at market.
    """
    close = df["Close"].values
    open_ = df["Open"].values
    high = df["High"].values
    low = df["Low"].values
    atr_col = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).values
    last = n - 1
    line_val = _line_value(line["x0"], line["y0"], line["x1"], line["y1"], last)
    atr = float(atr_col[last]) if atr_col[last] and atr_col[last] > 0 else abs(high[last] - low[last]) or 1e-9

    beyond = (close[last] > line_val) if kind == "resistance_break_up" or kind == "support_break_up" else (close[last] < line_val)
    penetration_atr = abs(close[last] - line_val) / atr

    rng = max(high[last] - low[last], 1e-9)
    body_ratio = abs(close[last] - open_[last]) / rng

    consecutive = 0
    for i in range(last, max(last - 5, -1), -1):
        lv_i = _line_value(line["x0"], line["y0"], line["x1"], line["y1"], i)
        beyond_i = (close[i] > lv_i) if kind.endswith("break_up") else (close[i] < lv_i)
        if beyond_i:
            consecutive += 1
        else:
            break

    if penetration_atr >= 0.35 and body_ratio >= 0.45 and consecutive >= 2:
        strength = "confirmed"
    elif penetration_atr >= 0.15 and consecutive >= 1:
        strength = "developing"
    else:
        strength = "weak"

    return {
        "strength": strength,
        "penetration_atr": round(penetration_atr, 2),
        "consecutive_closes": consecutive,
        "body_ratio": round(body_ratio, 2),
        "retest_level": line_val,
    }


def build_trendline_family(df: pd.DataFrame, max_lines: int = 4, lookback_bars: int = 90) -> Dict[str, Any]:
    """
    Build one clean parallel family (ascending OR descending), not both mixed.
    Market reveals direction: price relative to the family rails.

    lookback_bars: diagonal trendlines (and the wedge detector) only
    consider pivots from the most recent `lookback_bars` candles. Without
    this, a long flat chop zone from days ago can out-score the swing that
    actually shapes the current move -- a flat zone sits near dozens of
    candles' lows just by being flat and old, racking up touch count, while
    a real recent diagonal (the thing a trader actually draws) only grazes
    a handful of candles because it's moving fast. Scoring by raw touch
    count alone then picks the stale flat line and the rails end up
    running almost horizontal across the whole chart -- indistinguishable
    from a moving average -- instead of hugging the live structure.
    Horizontal S/R intentionally stays full-history (older well-tested
    flip zones ARE still relevant); only the diagonal fit is windowed.
    """
    if df is None or len(df) < 30:
        return {"error": "Insufficient data for trendline family", "direction": "NEUTRAL", "pivots": []}

    n = len(df)
    # LOCKED: only non-ranging swings (zigzag_swings now filters ranging legs)
    # Prefer cleaner, larger pivots so lines follow real directional structure
    pivots = zigzag_swings(df, depth=4, deviation_atr=0.30)
    if len(pivots) < 4:
        pivots = zigzag_swings(df, depth=3, deviation_atr=0.25)
    if len(pivots) < 3:
        pivots = zigzag_swings(df, depth=3, deviation_atr=0.18)

    # Recent-only candidate pool for the DIAGONAL fit (see docstring).
    cutoff = max(0, n - lookback_bars)
    recent_pivots = [p for p in pivots if p["index"] >= cutoff]
    if len(recent_pivots) < 3:
        # Not enough recent structure -- widen gradually rather than
        # snapping straight back to the full, stale-prone history.
        recent_pivots = [p for p in pivots if p["index"] >= max(0, n - lookback_bars * 2)]
    if len(recent_pivots) < 2:
        recent_pivots = pivots

    support = _fit_primary(recent_pivots, "support", n, df)
    resistance = _fit_primary(recent_pivots, "resistance", n, df)

    # LOCKED RULE:
    #   Uptrend  → map pivot LOWS  (ascending support)
    #   Downtrend → map pivot HIGHS (descending resistance)
    # Prefer the directional family that matches structure; do not mix.
    close = float(df["Close"].iloc[-1])
    primary = None
    family_kind = "none"

    # Detect simple structure bias from recent non-ranging pivots
    recent = pivots[-6:] if len(pivots) >= 4 else pivots
    highs = [p for p in recent if p["type"] == "high"]
    lows = [p for p in recent if p["type"] == "low"]
    struct_bias = "NEUTRAL"
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1]["price"] > highs[-2]["price"]
        hl = lows[-1]["price"] > lows[-2]["price"]
        lh = highs[-1]["price"] < highs[-2]["price"]
        ll = lows[-1]["price"] < lows[-2]["price"]
        if hh and hl:
            struct_bias = "BULLISH"
        elif lh and ll:
            struct_bias = "BEARISH"

    if struct_bias == "BULLISH" and support:
        # Uptrend: only map pivot lows
        primary, family_kind = support, "ascending"
    elif struct_bias == "BEARISH" and resistance:
        # Downtrend: only map pivot highs
        primary, family_kind = resistance, "descending"
    elif support and resistance:
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
        family_lines = _build_parallel_family(primary, recent_pivots, n, max_members=min(2, max_lines))
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
    breakout_grade = None

    if primary and family_lines:
        lower = family_lines[0]["y_end"]
        upper = family_lines[-1]["y_end"]
        mid = (lower + upper) / 2.0
        touch_note = {
            "unconfirmed": "⚠️ only 2 touches -- unconfirmed, treat as tentative",
            "confirmed": f"{primary['touches']} touches -- validated structure",
            "crowded": f"{primary['touches']} touches -- crowded level, order flow may be depleted",
        }.get(primary.get("quality"), f"{primary['touches']} touches")

        if family_kind == "ascending":
            if close >= lower:
                direction = "BUY"
                strength = 55 + min(25, primary["touches"] * 7)
                reasons.append(f"Ascending family · {touch_note}")
                if primary.get("quality") == "unconfirmed":
                    strength -= 12
                elif primary.get("quality") == "crowded":
                    strength -= 6
                if close > mid:
                    reasons.append("Price in upper half of channel — bullish control")
                    strength += 10
                else:
                    reasons.append("Price near support rail — watch bounce / break")
            else:
                brk = _grade_breakout(df, primary, "support_break_down", n)
                direction = "SELL"
                if brk["strength"] == "confirmed":
                    strength = 68
                    reasons.append(f"Confirmed break below ascending support — "
                                    f"{brk['penetration_atr']} ATR beyond, {brk['consecutive_closes']} closes, "
                                    f"body {brk['body_ratio']}")
                elif brk["strength"] == "developing":
                    strength = 52
                    reasons.append(f"Developing break below support ({brk['consecutive_closes']} close(s), "
                                    f"{brk['penetration_atr']} ATR) — not yet confirmed, watch for retest "
                                    f"at {brk['retest_level']:.5f}")
                else:
                    strength = 38
                    reasons.append(f"Weak/wick break below support ({brk['penetration_atr']} ATR, "
                                    f"body {brk['body_ratio']}) — likely noise, high fakeout risk")
                reasons.append(touch_note)
                breakout_grade = brk
        else:  # descending
            if close <= upper:
                direction = "SELL"
                strength = 55 + min(25, primary["touches"] * 7)
                reasons.append(f"Descending family · {touch_note}")
                if primary.get("quality") == "unconfirmed":
                    strength -= 12
                elif primary.get("quality") == "crowded":
                    strength -= 6
                if close < mid:
                    reasons.append("Price in lower half of channel — bearish control")
                    strength += 10
                else:
                    reasons.append("Price near resistance rail — watch reject / break")
            else:
                brk = _grade_breakout(df, primary, "resistance_break_up", n)
                direction = "BUY"
                if brk["strength"] == "confirmed":
                    strength = 68
                    reasons.append(f"Confirmed break above descending resistance — "
                                    f"{brk['penetration_atr']} ATR beyond, {brk['consecutive_closes']} closes, "
                                    f"body {brk['body_ratio']}")
                elif brk["strength"] == "developing":
                    strength = 52
                    reasons.append(f"Developing break above resistance ({brk['consecutive_closes']} close(s), "
                                    f"{brk['penetration_atr']} ATR) — not yet confirmed, watch for retest "
                                    f"at {brk['retest_level']:.5f}")
                else:
                    strength = 38
                    reasons.append(f"Weak/wick break above resistance ({brk['penetration_atr']} ATR, "
                                    f"body {brk['body_ratio']}) — likely noise, high fakeout risk")
                reasons.append(touch_note)
                breakout_grade = brk

    # Converging wedge/triangle (independent-slope rails) and full-history
    # horizontal S/R -- computed off the FULL pivot list, before it gets
    # truncated to the last 16 below, so an older well-tested level or a
    # wedge anchored further back doesn't silently drop out.
    wedge = _detect_converging_wedge(recent_pivots, df, n)
    horizontal_levels = _detect_horizontal_levels(df, pivots, n)
    if wedge and direction == "NEUTRAL" and wedge["bias"] != "NEUTRAL":
        direction = wedge["bias"]
        strength = max(strength, 58)
        reasons.append(f"{wedge['pattern']} — rails converging toward apex")

    projections = _measured_move_projections(df, pivots, direction)
    vp = compute_volume_profile(df.iloc[:-1])
    mw = _detect_mw_pattern(pivots, df)
    if mw:
        reasons.append(mw["note"])
        if direction == "NEUTRAL":
            direction = mw["bias"]
            strength = max(strength, 65)
        elif direction == mw["bias"]:
            strength = min(100, strength + 12)
    else:
        mw = None

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
        "strength": max(0, min(100, int(strength))),
        "reasons": reasons,
        "family_kind": family_kind,
        "family_lines": family_lines,  # the clean parallel set
        "uptrends": [primary] if family_kind == "ascending" and primary else [],
        "downtrends": [primary] if family_kind == "descending" and primary else [],
        "channel": channel,
        "wedge": wedge,
        "horizontal_levels": horizontal_levels,
        "projections": projections,
        "mw_pattern": mw,
        "pivots": pivots[-16:],
        "volume_profile": vp,
        "upper_line": upper_line,
        "lower_line": lower_line,
        "middle_line": mid_line,
        "df": df,
        "mode": "channel" if channel else "lines",
        "breakout_grade": breakout_grade,
        "primary_quality": primary.get("quality") if primary else None,
        "primary_touches": primary.get("touches") if primary else 0,
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


def _liquidity_targets(pivots, direction: str, entry: float) -> List[float]:
    """
    External liquidity pools price is drawn to:
      BUY  → swing highs above entry (BSL)
      SELL → swing lows below entry (SSL)
    Sorted nearest → farthest.
    """
    if not pivots:
        return []
    targets = []
    for p in pivots:
        px = float(p["price"])
        if direction == "BUY" and p.get("type") == "high" and px > entry:
            targets.append(px)
        elif direction == "SELL" and p.get("type") == "low" and px < entry:
            targets.append(px)
    if direction == "BUY":
        targets.sort()  # nearest high first
    else:
        targets.sort(reverse=True)  # nearest low first
    # unique with small tolerance
    cleaned = []
    for t in targets:
        if not cleaned or abs(t - cleaned[-1]) / max(abs(t), 1e-9) > 0.0003:
            cleaned.append(t)
    return cleaned


def build_position_container(family: Dict[str, Any], atr_mult_sl: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Position box with DYNAMIC R:R from real liquidity distance.

    Entry  : nearest structure rail / close
    SL     : beyond invalidation (last opposing swing or rail break)
    TP1/TP2: nearest / next liquidity pools (swing highs or lows)
    R:R    : |TP1 - Entry| / |Entry - SL|  (not a fixed multiple)
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

    pivots = family.get("pivots") or []
    lines = family.get("family_lines") or []
    channel = family.get("channel")
    mw = family.get("mw_pattern")
    projs = family.get("projections") or []
    vp = family.get("volume_profile") or {}
    brk = family.get("breakout_grade")

    if direction == "BUY":
        # Entry: support rail or close
        entry = close
        if lines:
            below = [m["y_end"] for m in lines if m["y_end"] <= close]
            if below:
                entry = max(below)
        elif channel and channel.get("lower"):
            entry = float(channel["lower"].get("y_end", close))

        # SL: below last swing low under entry (structural invalidation)
        swing_lows = [float(p["price"]) for p in pivots if p.get("type") == "low" and p["price"] < entry]
        if swing_lows:
            sl = min(swing_lows[-2:], default=min(swing_lows))  # recent low
            sl = min(swing_lows) if len(swing_lows) == 1 else sorted(swing_lows)[-1]
            # use nearest swing low below entry
            below_entry = [x for x in swing_lows if x < entry]
            sl = max(below_entry) if below_entry else entry - atr * atr_mult_sl
            sl = sl - atr * 0.15  # buffer beyond liquidity
        else:
            sl = entry - atr * atr_mult_sl

        # Targets = buy-side liquidity (swing highs above) + neckline if W + POC if above
        liq = _liquidity_targets(pivots, "BUY", entry)
        if mw and mw.get("pattern") == "W" and mw.get("neckline", 0) > entry:
            liq = sorted(set(liq + [float(mw["neckline"])]))
        if vp.get("poc_price") and vp["poc_price"] > entry:
            liq = sorted(set(liq + [float(vp["poc_price"])]))
        # fallback measured move only if no swing liquidity
        if not liq and projs:
            liq = [float(p["price"]) for p in projs if p["price"] > entry]

        tp1 = liq[0] if liq else entry + atr * 1.5
        tp2 = liq[1] if len(liq) > 1 else (liq[0] if liq else entry + atr * 2.5)

        order_type = "LIMIT" if entry < close - atr * 0.08 else "MARKET"
        entry_note = None
        # A breakout that isn't confirmed yet (weak/developing) shouldn't be
        # chased at market -- route the entry to the break-and-retest zone
        # instead, per standard breakout-trading practice.
        if brk and brk["strength"] != "confirmed":
            entry = brk["retest_level"]
            order_type = "LIMIT"
            entry_note = (f"Unconfirmed breakout ({brk['strength']}) -- entry routed to retest of the "
                          f"broken rail at {entry:.5f} instead of chasing at market")
            sl = min(sl, entry - atr * 0.5)

        risk = abs(entry - sl)
        reward = abs(tp1 - entry)
        rr = (reward / risk) if risk > 0 else 0.0

        return {
            "side": "LONG",
            "direction": "BUY",
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "rr": round(rr, 2),
            "risk": risk,
            "reward": reward,
            "liquidity_tp1": "swing high / BSL" if liq else "measured move",
            "order_type": order_type,
            "entry_note": entry_note,
            "breakout_grade": brk,
        }

    else:  # SELL
        entry = close
        if lines:
            above = [m["y_end"] for m in lines if m["y_end"] >= close]
            if above:
                entry = min(above)
        elif channel and channel.get("upper"):
            entry = float(channel["upper"].get("y_end", close))

        swing_highs = [float(p["price"]) for p in pivots if p.get("type") == "high" and p["price"] > entry]
        if swing_highs:
            above_entry = [x for x in swing_highs if x > entry]
            sl = min(above_entry) if above_entry else entry + atr * atr_mult_sl
            sl = sl + atr * 0.15
        else:
            sl = entry + atr * atr_mult_sl

        # Sell-side liquidity (swing lows below) + M neckline + POC
        liq = _liquidity_targets(pivots, "SELL", entry)
        if mw and mw.get("pattern") == "M" and mw.get("neckline", 0) < entry:
            liq_set = list(liq) + [float(mw["neckline"])]
            liq = sorted(liq_set, reverse=True)
        if vp.get("poc_price") and vp["poc_price"] < entry:
            liq = sorted(set(list(liq) + [float(vp["poc_price"])]), reverse=True)
        if not liq and projs:
            liq = [float(p["price"]) for p in projs if p["price"] < entry]

        tp1 = liq[0] if liq else entry - atr * 1.5
        tp2 = liq[1] if len(liq) > 1 else (liq[0] if liq else entry - atr * 2.5)

        order_type = "LIMIT" if entry > close + atr * 0.08 else "MARKET"
        entry_note = None
        if brk and brk["strength"] != "confirmed":
            entry = brk["retest_level"]
            order_type = "LIMIT"
            entry_note = (f"Unconfirmed breakout ({brk['strength']}) -- entry routed to retest of the "
                          f"broken rail at {entry:.5f} instead of chasing at market")
            sl = max(sl, entry + atr * 0.5)

        risk = abs(sl - entry)
        reward = abs(entry - tp1)
        rr = (reward / risk) if risk > 0 else 0.0

        return {
            "side": "SHORT",
            "direction": "SELL",
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "rr": round(rr, 2),
            "risk": risk,
            "reward": reward,
            "liquidity_tp1": "swing low / SSL" if liq else "measured move",
            "order_type": order_type,
            "entry_note": entry_note,
            "breakout_grade": brk,
        }


def format_trendline_report(family: Dict[str, Any], symbol: str) -> str:
    if family.get("error"):
        return family["error"]
    quality = family.get("primary_quality")
    quality_tag = {
        "unconfirmed": "⚠️ UNCONFIRMED (2 touches)",
        "confirmed": "✅ CONFIRMED",
        "crowded": "⚠️ CROWDED (5+ touches)",
    }.get(quality, "")
    lines = [
        f"📐 TRENDLINE FAMILY  |  {symbol}",
        f"Family: {family.get('family_kind', '—').upper()}  |  "
        f"Direction: {family.get('direction')}  |  Strength: {family.get('strength', 0)}/100",
    ]
    if quality_tag:
        lines.append(f"Trendline validation: {quality_tag} · {family.get('primary_touches', 0)} touches")
    for r in family.get("reasons") or []:
        lines.append(f"  • {r}")
    n_rails = len(family.get("family_lines") or [])
    lines.append(f"Parallel rails: {n_rails}")
    wedge = family.get("wedge")
    if wedge:
        lines.append(
            f"Structure: {wedge['pattern']} · lower rail {wedge['lower']['touches']} touches, "
            f"upper rail {wedge['upper']['touches']} touches · converging (gap {wedge['gap_end']:.5f})"
        )
    hz = family.get("horizontal_levels") or []
    if hz:
        lines.append("Horizontal levels: " + " · ".join(
            f"{l['side'][0].upper()} {l['price']:.5f} ({l['touches']}x)" for l in hz))
    mw = family.get("mw_pattern")
    if mw:
        lines.append(f"Pattern: {mw['name']} · neckline {mw['neckline']:.5f}")
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
    brk = family.get("breakout_grade")
    if brk:
        grade_tag = {"confirmed": "✅ CONFIRMED", "developing": "🟡 DEVELOPING",
                     "weak": "🔴 WEAK / LIKELY FAKEOUT"}.get(brk["strength"], brk["strength"])
        lines.append(
            f"Breakout grade: {grade_tag} · {brk['penetration_atr']} ATR beyond · "
            f"{brk['consecutive_closes']} consecutive close(s) · body {brk['body_ratio']}"
        )
        lines.append(f"Retest zone: {brk['retest_level']:.5f}")
    pos = build_position_container(family)
    if pos:
        lines.append(
            f"Position: {pos['side']}  Entry {pos['entry']:.5f}  SL {pos['sl']:.5f}  "
            f"TP1 {pos['tp1']:.5f}  TP2 {pos['tp2']:.5f}"
        )
        lines.append(
            f"R:R 1:{pos.get('rr', 0):.2f}  (risk {pos.get('risk', 0):.5f} → reward {pos.get('reward', 0):.5f})"
        )
        if pos.get("liquidity_tp1"):
            lines.append(f"TP1 liquidity: {pos['liquidity_tp1']}")
        if pos.get("entry_note"):
            lines.append(f"⚠️ {pos['entry_note']}")
    return "\n".join(lines)
