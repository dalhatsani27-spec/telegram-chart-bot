"""
strategies.py
=============
The two chart-analysis strategies this bot runs: Trendline and OTE.

Both call topdown_engine.get_topdown_bias() first to establish a 4H/1H
directional read, then do their own timeframe-specific work on the 30M
chart (the geometry/entry engine), and finally gate/score that 30M read
against the top-down bias so Trendline and OTE never disagree with the
bigger picture without saying so.

  - Trendline: parallel-channel trendline family + measured-move /
    liquidity targets, entry on 30M, full 4H -> 1H -> 30M cascade.
  - OTE: Fibonacci Fan + Expansion off the most recent clean impulse leg,
    entry on 30M, gated by the same 4H -> 1H top-down read.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import market_data
from market_analysis import zigzag_swings, find_swings, compute_volume_profile, detect_confirmation_candle, analyse_structure, detect_order_blocks, scan_all_patterns
from topdown_engine import get_topdown_bias, format_topdown_summary


# ============================================================
# TRENDLINE GEOMETRY ENGINE -- parallel-channel family, wedges,
# horizontal S/R clustering, measured-move & liquidity targets,
# breakout grading. Operates on whatever df is passed in (the
# orchestration at the bottom of this file always passes the 30M
# chart).
# ============================================================


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


def _touch_points(df: pd.DataFrame, x0: int, y0: float, x1: int, y1: float,
                   kind: str, tol_atr: float = 0.40) -> List[Dict]:
    """Same tolerance test as _count_touches, but returns the actual
    (index, price) of each touching wick instead of just a count, so the
    chart can mark every bounce along the trendline the way a trader
    circles them by hand -- not just the two pivots that defined the line."""
    if df is None or len(df) < 5:
        return []
    atr = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).values
    highs = df["High"].values
    lows = df["Low"].values
    points = []
    lo, hi = min(x0, x1), max(x0, x1)
    for i in range(lo, min(hi + 1, len(df))):
        lv = _line_value(x0, y0, x1, y1, i)
        a = float(atr[i]) if i < len(atr) and atr[i] > 0 else abs(y1 - y0) * 0.05
        tol = max(a * tol_atr, 1e-9)
        if kind == "support" and abs(lows[i] - lv) <= tol:
            points.append({"index": i, "price": float(lows[i])})
        elif kind == "resistance" and abs(highs[i] - lv) <= tol:
            points.append({"index": i, "price": float(highs[i])})
    return points


def _fit_primary(pivots: List[Dict], kind: str, n: int, df: pd.DataFrame) -> Optional[Dict]:
    """Best 2-point primary line of given kind (support=lows, resistance=highs).

    Professional validation standard: a 2-point line is only a *candidate* --
    it takes a 3rd touch for traders to actually respect it as real structure.
    We still return 2-touch lines (better than nothing), but tag them
    "unconfirmed" so downstream scoring/reporting can be honest about it.
    5+ touches is flagged "crowded": the level has been tested so many times
    the order flow defending it is likely used up, and the next test is
    statistically more likely to fail than hold.

    The line's 2nd point must be recent enough that extrapolating it to the
    current bar is still meaningful -- otherwise you get a line anchored to
    an old pivot, stretched flat across everything that's happened since,
    cutting through unrelated later price action instead of tracking it.
    """
    pts = [p for p in pivots if p["type"] == ("low" if kind == "support" else "high")]
    if len(pts) < 2:
        return None
    # The most recent defining touch has to sit within the last ~40% of
    # the window (min 20 bars) -- a line whose last touch is ancient
    # relative to the current bar has no business being extrapolated
    # across everything since.
    recency_floor = max(0, n - max(20, int((max(p["index"] for p in pts) - min(p["index"] for p in pts) or n) * 0.4)))
    best = None
    best_score = -1
    for i in range(len(pts) - 1):
        for j in range(i + 1, len(pts)):
            a, b = pts[i], pts[j]
            if b["index"] <= a["index"]:
                continue
            if b["index"] < recency_floor:
                continue  # last touch too stale to extrapolate from
            # Uptrend support needs higher low; downtrend resistance needs lower high
            if kind == "support" and b["price"] <= a["price"]:
                continue
            if kind == "resistance" and b["price"] >= a["price"]:
                continue
            slope = (b["price"] - a["price"]) / max(b["index"] - a["index"], 1)
            touches = _count_touches(df, a["index"], a["price"], b["index"], b["price"], kind)
            # Prefer more touches + a recent 2nd touch. NOTE: deliberately no
            # reward for a wide a-b span -- that used to bias the fit toward
            # an old, distant starting pivot just because it made the line
            # "look" more established, which produced lines extrapolated far
            # past anything they were actually still tracking.
            touch_score = touches * 10
            if touches >= 5:
                touch_score -= (touches - 4) * 3  # fatigue penalty, doesn't erase the line
            score = touch_score + (b["index"] / max(n, 1)) * 8
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
                    "bars_since_last_touch": n - 1 - b["index"],
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


def _entry_confirmation(df: pd.DataFrame, direction: str) -> Dict[str, Any]:
    """
    The "Confirmation" checklist from the entry rules -- this is the part
    that was previously missing entirely: direction used to be decided
    purely from price-vs-rail geometry, with nothing checking whether the
    move actually has confirmation behind it (image's Core Rule: "wait
    for confirmation before entry", "never force a trade").

    Checks 4 things:
      1. Candlestick pattern in the trade direction
      2. Break of MINOR market structure (small-swing BOS/MSS) confirming
         momentum has actually shifted, not just "price is near the line"
      3. Volume/momentum (rising volume if the feed has it, else a
         short-term rate-of-change proxy when it doesn't)
      4. RSI(14) above 50 for longs / below 50 for shorts
    """
    checks = {
        "candle": (False, "no candle data"),
        "structure": (False, "no structure data"),
        "momentum": (False, "no momentum data"),
        "rsi": (False, "no RSI data"),
    }
    if df is None or len(df) < 20 or direction not in ("BUY", "SELL"):
        return {"checks": checks, "passed": 0, "required": 2, "confirmed": False}

    found, name = detect_confirmation_candle(df, direction)
    checks["candle"] = (found, name or "no matching pattern in last 3 bars")

    minor = analyse_structure(df, left=2, right=2, lookback=30)
    want_bias = "BULLISH" if direction == "BUY" else "BEARISH"
    struct_ok = minor.get("last_event") in ("BOS", "MSS") and minor.get("event_bias") == want_bias
    checks["structure"] = (struct_ok, minor.get("note") or "no recent minor BOS/MSS")

    if "Volume" in df.columns and df["Volume"].tail(20).sum() > 0:
        recent_vol = float(df["Volume"].iloc[-1])
        avg_vol = float(df["Volume"].iloc[-11:-1].mean()) if len(df) > 11 else recent_vol
        mom_ok = recent_vol > avg_vol * 1.05
        checks["momentum"] = (mom_ok, f"vol {recent_vol:.0f} vs 10-bar avg {avg_vol:.0f}")
    elif len(df) > 6:
        roc = float((df["Close"].iloc[-1] - df["Close"].iloc[-6]) / df["Close"].iloc[-6] * 100)
        mom_ok = roc > 0.05 if direction == "BUY" else roc < -0.05
        checks["momentum"] = (mom_ok, f"5-bar RoC {roc:+.2f}% (no volume feed)")

    if "RSI" in df.columns:
        rsi = float(df["RSI"].iloc[-1])
        rsi_ok = rsi > 50 if direction == "BUY" else rsi < 50
        checks["rsi"] = (rsi_ok, f"RSI {rsi:.1f}")

    passed = sum(1 for ok, _ in checks.values() if ok)
    required = 2  # at least half the checklist -- "confirmation", not "perfection"
    return {"checks": checks, "passed": passed, "required": required, "confirmed": passed >= required}


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

    # Reject a candidate diagonal line whose actual price movement across
    # its own span is too shallow to be a meaningful trend -- e.g. two
    # swing lows that are technically "rising" by a few points over three
    # days. That's a range, not an uptrend, and drawing it as a diagonal
    # "ascending" rail is misleading -- it should be left for the
    # horizontal S/R clustering below (_detect_horizontal_levels) to pick
    # up instead, which is exactly what that layer is for.
    MIN_TREND_MOVE_ATR = 1.8  # total rise/fall across the line's full span, in ATRs
    atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else None

    def _has_meaningful_slope(line):
        if not line:
            return False
        if not atr_now or atr_now <= 0:
            return True  # can't measure -- don't reject on a technicality
        total_move = abs(line["y1"] - line["y0"])
        return total_move >= MIN_TREND_MOVE_ATR * atr_now

    if support and not _has_meaningful_slope(support):
        support = None
        shallow_rejected = True
    else:
        shallow_rejected = False
    if resistance and not _has_meaningful_slope(resistance):
        resistance = None
        shallow_rejected = True

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
    if shallow_rejected and not primary:
        reasons.append(
            "No diagonal trendline met the minimum slope -- price is "
            "ranging rather than trending here; see horizontal S/R levels instead"
        )
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
    # Keep only the 2 strongest horizontal levels (1 support + 1 resistance,
    # nearest to price) for the chart -- the full clustered history is still
    # useful for the text report, but 3-4 stacked S/R lines on top of a
    # diagonal family + wedge/M-W pattern is what caused the label pile-up
    # around the entry zone. The report path can still ask for more via
    # max_levels if it wants the full picture.
    horizontal_levels = _detect_horizontal_levels(df, pivots, n, max_levels=2)
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

    # --- Entry confirmation gate (Core Rule: wait for confirmation, never
    # force a trade). This did not exist before -- direction was decided
    # purely from price-vs-rail geometry above. It still is, but now we
    # separately record whether the move is actually confirmed, and the
    # chart/report show it plainly instead of presenting every geometric
    # bias as a ready-to-fire signal.
    entry_rules = _entry_confirmation(df, direction) if direction in ("BUY", "SELL") else None
    if entry_rules:
        passed_names = [k for k, (ok, _) in entry_rules["checks"].items() if ok]
        if entry_rules["confirmed"]:
            strength = min(100, strength + 6)
            reasons.append(f"Entry confirmed ({entry_rules['passed']}/4 checks: {', '.join(passed_names) or 'none'})")
        else:
            strength = max(0, strength - 15)
            reasons.append(
                f"⚠ Entry NOT confirmed yet ({entry_rules['passed']}/4 checks, need {entry_rules['required']}) "
                f"— trendline bias only, wait for confirmation before entering"
            )

    # --- Order block reaction gate --------------------------------------
    # order_blocks (computed below for the chart) were being detected and
    # drawn but never actually consulted by the direction/strength logic --
    # so the bot could show a BUY signal while price was sitting inside a
    # bearish supply OB, or vice versa, with zero acknowledgement of it.
    # Compute once here, before the return, and let it push back on
    # whatever the trendline geometry decided.
    order_blocks = detect_order_blocks(df, lookback=len(df))
    active_ob = None
    for ob in order_blocks:
        if float(ob["bottom"]) <= close <= float(ob["top"]):
            active_ob = ob
            break  # list is nearest-to-price first
    if active_ob:
        ob_side = active_ob["type"]  # 'bullish' or 'bearish'
        ob_desc = (f"{ob_side.capitalize()} order block ({active_ob['grade']}, "
                   f"{active_ob['confidence']}%, {active_ob['freshness']})")
        if direction == "BUY" and ob_side == "bearish":
            strength = max(0, strength - 20)
            reasons.append(f"⚠️ Price is trading INSIDE a {ob_desc} — supply zone overhead, "
                            f"counter-trend risk, expect rejection before continuation")
        elif direction == "SELL" and ob_side == "bullish":
            strength = max(0, strength - 20)
            reasons.append(f"⚠️ Price is trading INSIDE a {ob_desc} — demand zone below, "
                            f"counter-trend risk, expect rejection before continuation")
        elif direction == "SELL" and ob_side == "bearish":
            strength = min(100, strength + 10)
            reasons.append(f"✅ Price reacting inside the {ob_desc} that's driving this move — "
                            f"aligned with direction")
        elif direction == "BUY" and ob_side == "bullish":
            strength = min(100, strength + 10)
            reasons.append(f"✅ Price reacting inside the {ob_desc} that's driving this move — "
                            f"aligned with direction")
        elif direction == "NEUTRAL":
            direction = "SELL" if ob_side == "bearish" else "BUY"
            strength = max(strength, 50)
            reasons.append(f"Price sitting inside an untested {ob_desc} with no trendline bias otherwise — "
                            f"reacting off the zone")

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
        "entry_rules": entry_rules,
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
        # Single "which pattern wins the chart" decision, so the renderer
        # never has to draw wedge + M/W + channel all at once. Priority:
        # M/W (specific reversal structure, high signal) > wedge/triangle
        # (specific continuation/reversal structure) > plain channel/bias
        # line (just "this is the trend", not a named pattern).
        "active_pattern": (
            "mw" if mw else "wedge" if wedge else "channel" if channel else "none"
        ),
        "pattern_confidence": max(0, min(100, int(strength))),
        # Every wick that actually touches the bias line within tolerance --
        # not just its 2 defining pivots -- so the chart can circle each
        # bounce (the "notice the bounces" behavior from a hand-drawn map).
        "bias_touch_points": (
            _touch_points(df, int(primary["x0"]), primary["y0"], int(primary["x1"]), primary["y1"], primary["kind"])
            if primary else []
        ),
        # Order blocks "likely to be respected" -- graded by displacement
        # strength, freshness, and whether they actually caused a
        # confirmed structure break (see detect_order_blocks docstring).
        # Previously excluded by design; added back deliberately now.
        "order_blocks": order_blocks,
        "active_order_block": active_ob,
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
    entry_rules = family.get("entry_rules")
    confirmed = bool(entry_rules and entry_rules.get("confirmed"))

    # Trendline value at the current bar -- used as the other half of the
    # "SL below recent swing low / trendline" rule, alongside the swing-low
    # based stop already computed below. Only trusted if the line's last
    # defining touch is recent -- an old line stretched a long way to reach
    # the current bar isn't a real invalidation level, and letting it push
    # the stop out is exactly how risk balloons for no real reason.
    primary = (family.get("uptrends") or family.get("downtrends") or [None])[0]
    trendline_val = None
    if primary and df is not None and len(df) and primary.get("bars_since_last_touch", 999) <= 25:
        trendline_val = _line_value(primary["x0"], primary["y0"], primary["x1"], primary["y1"], len(df) - 1)

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
        # "below recent swing low / trendline" -- use whichever is further
        # from entry, but capped: a stop should never balloon past ~3x ATR
        # just because a (possibly stale) trendline sat further away.
        if trendline_val is not None and trendline_val < entry:
            max_reasonable_risk = max(atr * 3.0, (entry - sl) * 1.4)
            candidate_sl = trendline_val - atr * 0.15
            if (entry - candidate_sl) <= max_reasonable_risk:
                sl = min(sl, candidate_sl)

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
        reward_tp1 = abs(tp1 - entry)
        # Headline reward/R:R quoted against TP2 ("next resistance", a real
        # target) rather than TP1 ("previous high" -- often the very next
        # bit of structure, deliberately close). Quoting R:R off the quick
        # partial made every setup look worse than the actual plan is.
        reward = abs(tp2 - entry) if tp2 is not None else reward_tp1
        rr = (reward / risk) if risk > 0 else 0.0
        rr_tp1 = (reward_tp1 / risk) if risk > 0 else 0.0

        # TP3: "use RR 1:2 or 1:3" -- prefer a real liquidity level beyond
        # RR 1:2 if one exists, otherwise the fixed RR 1:2 extension.
        tp3 = entry + risk * 2.0
        tp3_basis = "fixed RR 1:2 target"
        for lvl in liq[2:]:
            cand_rr = abs(lvl - entry) / risk if risk > 0 else 0
            if cand_rr >= 2.0:
                tp3 = lvl
                tp3_basis = f"next liquidity level (RR 1:{cand_rr:.1f})"
                break

        # Full draw target: an untouched bearish OB above, or the volume
        # profile POC if it sits further out than either, is where a BUY
        # is actually being drawn to -- take whichever real magnet is
        # farthest beyond wherever the liquidity/RR math landed.
        obs = family.get("order_blocks") or []
        magnet = [ob for ob in obs if ob["type"] == "bearish" and ob["freshness"] == "untested"
                  and float(ob["bottom"]) > entry]
        if magnet:
            nearest_magnet = min(magnet, key=lambda ob: float(ob["bottom"]))
            magnet_edge = float(nearest_magnet["bottom"])
            if magnet_edge > tp3:
                tp3 = magnet_edge
                tp3_basis = f"unmitigated bearish OB @ {magnet_edge:.5f} ({nearest_magnet['confidence']}%)"
        poc = vp.get("poc_price")
        if poc is not None and float(poc) > entry and float(poc) > tp3:
            tp3 = float(poc)
            tp3_basis = f"POC magnet @ {tp3:.5f}"

        return {
            "side": "LONG",
            "direction": "BUY",
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "tp3_basis": tp3_basis,
            "rr": round(rr, 2),
            "rr_tp1": round(rr_tp1, 2),
            "risk": risk,
            "reward": reward,
            "liquidity_tp1": "swing high / BSL" if liq else "measured move",
            "order_type": order_type,
            "entry_note": entry_note,
            "breakout_grade": brk,
            "confirmed": confirmed,
            "entry_rules": entry_rules,
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
        # "below recent swing low / trendline" mirrored for shorts: SL
        # above recent swing high OR the trendline (capped the same way).
        if trendline_val is not None and trendline_val > entry:
            max_reasonable_risk = max(atr * 3.0, (sl - entry) * 1.4)
            candidate_sl = trendline_val + atr * 0.15
            if (candidate_sl - entry) <= max_reasonable_risk:
                sl = max(sl, candidate_sl)

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
        reward_tp1 = abs(entry - tp1)
        reward = abs(entry - tp2) if tp2 is not None else reward_tp1
        rr = (reward / risk) if risk > 0 else 0.0
        rr_tp1 = (reward_tp1 / risk) if risk > 0 else 0.0

        tp3 = entry - risk * 2.0
        tp3_basis = "fixed RR 1:2 target"
        for lvl in liq[2:]:
            cand_rr = abs(entry - lvl) / risk if risk > 0 else 0
            if cand_rr >= 2.0:
                tp3 = lvl
                tp3_basis = f"next liquidity level (RR 1:{cand_rr:.1f})"
                break

        # Full draw target: an untouched bullish OB below, or the volume
        # profile POC if it sits further out than either, is where a SELL
        # is actually being drawn to -- take whichever real magnet is
        # farthest beyond wherever the liquidity/RR math landed.
        obs = family.get("order_blocks") or []
        magnet = [ob for ob in obs if ob["type"] == "bullish" and ob["freshness"] == "untested"
                  and float(ob["top"]) < entry]
        if magnet:
            nearest_magnet = min(magnet, key=lambda ob: entry - float(ob["top"]))
            magnet_edge = float(nearest_magnet["top"])
            if magnet_edge < tp3:
                tp3 = magnet_edge
                tp3_basis = f"unmitigated bullish OB @ {magnet_edge:.5f} ({nearest_magnet['confidence']}%)"
        poc = vp.get("poc_price")
        if poc is not None and float(poc) < entry and float(poc) < tp3:
            tp3 = float(poc)
            tp3_basis = f"POC magnet @ {tp3:.5f}"

        return {
            "side": "SHORT",
            "direction": "SELL",
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "tp3_basis": tp3_basis,
            "rr": round(rr, 2),
            "rr_tp1": round(rr_tp1, 2),
            "risk": risk,
            "reward": reward,
            "liquidity_tp1": "swing low / SSL" if liq else "measured move",
            "order_type": order_type,
            "entry_note": entry_note,
            "breakout_grade": brk,
            "confirmed": confirmed,
            "entry_rules": entry_rules,
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
        f"📐 TRENDLINE  |  {symbol}  (4H → 1H → 30M top-down)",
    ]
    topdown = family.get("topdown")
    if topdown:
        lines.append(format_topdown_summary(topdown))
        lines.append("—")
    lines.append(
        f"30M Family: {family.get('family_kind', '—').upper()}  |  "
        f"Direction: {family.get('direction')}  |  Strength: {family.get('strength', 0)}/100"
    )
    if quality_tag:
        lines.append(f"Trendline validation: {quality_tag} · {family.get('primary_touches', 0)} touches")
    for r in family.get("gating_notes") or []:
        lines.append(f"  • {r}")
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
    sp = family.get("scanned_pattern")
    if sp:
        lines.append(f"Chart pattern: {sp['name']} ({sp['bias']}, {sp['confidence']:.0f}%) — {sp['note']}")

    hz = family.get("horizontal_levels") or []
    if hz:
        lines.append("Horizontal levels: " + " · ".join(
            f"{l['side'][0].upper()} {l['price']:.5f} ({l['touches']}x)" for l in hz))
    obs = family.get("order_blocks") or []
    if obs:
        ob_lines = []
        for ob in obs:
            tag = "UNMITIGATED" if ob["freshness"] == "untested" else "mitigated"
            ob_lines.append(
                f"{ob['type'][:4].capitalize()} {ob['bottom']:.5f}-{ob['top']:.5f} "
                f"({tag}, {ob['confidence']}%)"
            )
        lines.append("Order blocks: " + " · ".join(ob_lines))
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


# ============================================================
# OTE STRATEGY -- Fibonacci Fan + Fibonacci Expansion
#
#   1. Get 4H -> 1H top-down bias (topdown_engine.get_topdown_bias)
#   2. Detect the most recent clear impulse swing on the 30M chart
#   3. Draw Fibonacci Fan (38.2 / 50 / 61.8) from the impulse origin
#   4. Entry zone = deeper fan lines (50-61.8%) acting as dynamic OTE
#   5. Project Fibonacci Expansion targets (127.2 / 161.8 / 200 / 261.8)
#   6. Gate/score the setup against the 4H/1H top-down bias
#
# Always runs and displays on the 30M timeframe.
# ============================================================

FAN_RATIOS = [0.382, 0.50, 0.618]
EXPANSION_RATIOS = [1.272, 1.618, 2.0, 2.618]


def _ensure_atr(df: pd.DataFrame) -> pd.DataFrame:
    if "ATR" not in df.columns or df["ATR"].isna().all():
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df = df.copy()
        df["ATR"] = tr.rolling(14, min_periods=1).mean()
    return df


def _find_impulse(df: pd.DataFrame, lookback: int = 120) -> Optional[Dict[str, Any]]:
    """
    Find the most recent clean impulse leg suitable for Fan + Expansion.

    Returns:
      {
        "direction": "BUY" | "SELL",
        "start": {index, price, type},
        "end":   {index, price, type},
        "retracement": {index, price, type} | None,   # point C if available
        "leg_size": float,
      }
    """
    if df is None or len(df) < 40:
        return None

    df = _ensure_atr(df)
    n = len(df)
    swings = zigzag_swings(df, depth=5, deviation_atr=0.40)
    if len(swings) < 2:
        swings = find_swings(df, left=3, right=3)
    if len(swings) < 2:
        return None

    # Restrict to recent window
    swings = [s for s in swings if s["index"] >= max(0, n - lookback)]
    if len(swings) < 2:
        return None

    # Walk backwards looking for a strong directional leg
    for i in range(len(swings) - 1, 0, -1):
        a = swings[i - 1]
        b = swings[i]
        if a["type"] == b["type"]:
            continue

        leg = abs(b["price"] - a["price"])
        atr = float(df["ATR"].iloc[min(b["index"], n - 1)])
        if atr <= 0:
            atr = leg * 0.1
        if leg < 1.2 * atr:          # require meaningful impulse
            continue

        # Bullish impulse: low -> high
        if a["type"] == "low" and b["type"] == "high" and b["price"] > a["price"]:
            retrace = None
            for s in swings[i + 1:]:
                if s["type"] == "low" and s["price"] < b["price"]:
                    retrace = s
                    break
            return {"direction": "BUY", "start": a, "end": b, "retracement": retrace, "leg_size": leg}

        # Bearish impulse: high -> low
        if a["type"] == "high" and b["type"] == "low" and b["price"] < a["price"]:
            retrace = None
            for s in swings[i + 1:]:
                if s["type"] == "high" and s["price"] > b["price"]:
                    retrace = s
                    break
            return {"direction": "SELL", "start": a, "end": b, "retracement": retrace, "leg_size": leg}

    return None


def _build_fan(impulse: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """Build Fibonacci Fan rays from impulse start -> end, extendable to any bar index."""
    x0 = impulse["start"]["index"]
    y0 = impulse["start"]["price"]
    x1 = impulse["end"]["index"]
    y1 = impulse["end"]["price"]
    dy = y1 - y0

    fans = []
    for r in FAN_RATIOS:
        y_div = y0 + dy * r
        slope = (y_div - y0) / max(x1 - x0, 1)
        y_end = y0 + slope * (n - 1 - x0)
        fans.append({
            "ratio": r, "label": f"{r*100:.1f}%",
            "x0": x0, "y0": y0, "x1": x1, "y1": y_div,
            "slope": slope, "y_at_end": y_end,
        })
    return fans


def _fan_price_at(fan: Dict, x: float) -> float:
    return fan["y0"] + fan["slope"] * (x - fan["x0"])


def _build_expansion(impulse: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fibonacci Expansion (3-point when a retracement point exists, else simple extension)."""
    start = impulse["start"]["price"]
    end = impulse["end"]["price"]
    leg = impulse["leg_size"]
    direction = impulse["direction"]
    retrace = impulse.get("retracement")

    expansions = []
    if retrace is not None:
        c = retrace["price"]
        for r in EXPANSION_RATIOS:
            price = c + leg * r if direction == "BUY" else c - leg * r
            expansions.append({"ratio": r, "label": f"{r*100:.1f}%", "price": float(price), "from_point": "C"})
    else:
        for r in EXPANSION_RATIOS:
            price = end + leg * (r - 1.0) if direction == "BUY" else end - leg * (r - 1.0)
            expansions.append({"ratio": r, "label": f"{r*100:.1f}%", "price": float(price), "from_point": "B"})
    return expansions


def _evaluate_entry(
    df: pd.DataFrame,
    impulse: Dict[str, Any],
    fans: List[Dict],
    expansions: List[Dict],
    topdown: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Decide if price is currently in a valid OTE entry zone on the Fan,
    gate the score against the 4H/1H top-down bias, and build the ticket.
    """
    n = len(df)
    close = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else abs(impulse["leg_size"]) * 0.1
    direction = impulse["direction"]

    fan_prices = sorted(
        [{"ratio": f["ratio"], "label": f["label"], "price": _fan_price_at(f, n - 1)} for f in fans],
        key=lambda x: x["price"],
    )

    in_zone = False
    nearest_fan = None
    min_dist = 1e18
    for fp in fan_prices:
        dist = abs(close - fp["price"])
        if dist < min_dist:
            min_dist = dist
            nearest_fan = fp
        if dist <= atr * 0.45:
            in_zone = True

    reasons = []
    score = 40

    if impulse["leg_size"] >= 2.0 * atr:
        score += 15
        reasons.append(f"Strong impulse ({impulse['leg_size']/atr:.1f} ATR)")
    else:
        reasons.append(f"Moderate impulse ({impulse['leg_size']/atr:.1f} ATR)")

    if in_zone:
        score += 25
        reasons.append(f"Price interacting with Fan {nearest_fan['label']}")
    else:
        lowest = fan_prices[0]["price"]
        highest = fan_prices[-1]["price"]
        if direction == "BUY" and lowest <= close <= highest + atr * 0.3:
            score += 10
            reasons.append("Price inside Fan channel")
        elif direction == "SELL" and highest >= close >= lowest - atr * 0.3:
            score += 10
            reasons.append("Price inside Fan channel")

    if nearest_fan and nearest_fan["ratio"] >= 0.50:
        score += 12
        reasons.append(f"Deep Fan zone ({nearest_fan['label']}) — OTE quality")

    if expansions:
        score += 8
        reasons.append(f"{len(expansions)} Expansion targets projected")

    # --- gate against the 4H/1H top-down bias ---
    td_dir = (topdown or {}).get("direction", "NEUTRAL")
    td_allowed = bool((topdown or {}).get("allowed"))
    if td_dir in ("BUY", "SELL"):
        if td_dir == direction and td_allowed:
            score += 15
            reasons.append(f"✅ Aligned with 4H/1H top-down bias ({td_dir}) -- structure permission granted")
        elif td_dir == direction and not td_allowed:
            reasons.append(f"Aligned with top-down direction ({td_dir}) but 1H structure permission not yet granted")
        else:
            score -= 25
            reasons.append(f"⚠️ 30M impulse direction ({direction}) conflicts with 4H/1H top-down bias ({td_dir})")
    else:
        reasons.append("4H/1H top-down read is NEUTRAL -- 30M impulse direction stands on its own")

    # Build ticket
    entry = close
    if direction == "BUY":
        sl_candidates = [fp["price"] for fp in fan_prices] + [impulse["start"]["price"]]
        sl = min(sl_candidates) - atr * 0.35
        tps = sorted([e["price"] for e in expansions if e["price"] > entry])
    else:
        sl_candidates = [fp["price"] for fp in fan_prices] + [impulse["start"]["price"]]
        sl = max(sl_candidates) + atr * 0.35
        tps = sorted([e["price"] for e in expansions if e["price"] < entry], reverse=True)

    tp1 = tps[0] if tps else (entry + atr * 1.8 if direction == "BUY" else entry - atr * 1.8)
    tp2 = tps[1] if len(tps) > 1 else (entry + atr * 3.0 if direction == "BUY" else entry - atr * 3.0)

    risk = abs(entry - sl)
    reward = abs(tp1 - entry)
    rr = (reward / risk) if risk > 0 else 0.0

    score = max(0, min(100, score))
    valid = direction in ("BUY", "SELL") and score >= 58 and in_zone and rr >= 1.2 and not (
        td_dir in ("BUY", "SELL") and td_dir != direction
    )

    ticket = {
        "side": "LONG" if direction == "BUY" else "SHORT",
        "direction": direction,
        "entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2),
        "rr": round(rr, 2), "risk": float(risk), "reward": float(reward),
        "order_type": "MARKET",
        "nearest_fan": nearest_fan["label"] if nearest_fan else None,
    }

    return {
        "in_zone": in_zone, "nearest_fan": nearest_fan, "score": score,
        "reasons": reasons, "valid": valid,
        "ticket": ticket if valid else None, "fan_prices": fan_prices,
    }


def run_ote_analysis(symbol: str, df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Full OTE analysis for a symbol: 4H/1H top-down bias, then impulse +
    Fan + Expansion detection and entry evaluation on the 30M chart.
    Always fetches/displays on 30M (falls back to 15M only if 30M truly
    doesn't have enough bars yet).
    """
    topdown = get_topdown_bias(symbol)

    timeframe = "30min"
    if df is None:
        df = market_data.fetch_candles(symbol, "30min", count=220)
        if df is None or df.empty or len(df) < 50:
            df = market_data.fetch_candles(symbol, "15min", count=220)
            timeframe = "15min (30M had insufficient history)"

    if df is None or df.empty or len(df) < 50:
        return {
            "error": "Insufficient 30M data for OTE analysis",
            "direction": "NEUTRAL", "score": 0, "valid": False,
            "symbol": symbol, "topdown": topdown,
        }

    df = _ensure_atr(df)
    n = len(df)

    impulse = _find_impulse(df)
    if impulse is None:
        return {
            "error": "No clear impulse swing found for Fan / Expansion",
            "direction": "NEUTRAL", "score": 0, "valid": False,
            "df": df, "timeframe": timeframe, "symbol": symbol, "topdown": topdown,
        }

    fans = _build_fan(impulse, n)
    expansions = _build_expansion(impulse)
    entry_eval = _evaluate_entry(df, impulse, fans, expansions, topdown=topdown)

    direction = impulse["direction"]
    score = entry_eval["score"]
    valid = entry_eval["valid"]
    reasons = entry_eval["reasons"]

    return {
        "strategy": "OTE",
        "direction": direction if valid else "NEUTRAL",
        "score": score,
        "reasons": reasons,
        "valid": valid,
        "impulse": impulse,
        "fans": fans,
        "expansions": expansions,
        "fan_prices": entry_eval["fan_prices"],
        "nearest_fan": entry_eval["nearest_fan"],
        "in_zone": entry_eval["in_zone"],
        "position": entry_eval["ticket"],
        "ticket": entry_eval["ticket"],
        "df": df,
        "timeframe": timeframe,
        "symbol": symbol,
        "topdown": topdown,
    }


def format_ote_report(analysis: Dict[str, Any]) -> str:
    symbol = analysis.get("symbol", "")
    if analysis.get("error"):
        lines = [f"🎯 OTE  (Fib Fan + Expansion)  |  {symbol}  (4H → 1H → 30M top-down)"]
        topdown = analysis.get("topdown")
        if topdown:
            lines.append(format_topdown_summary(topdown))
            lines.append("—")
        lines.append(analysis["error"])
        return "\n".join(lines)

    direction = analysis.get("direction", "NEUTRAL")
    score = analysis.get("score", 0)
    valid = analysis.get("valid", False)
    impulse = analysis.get("impulse") or {}
    fans = analysis.get("fans") or []
    expansions = analysis.get("expansions") or []
    ticket = analysis.get("ticket")
    nearest = analysis.get("nearest_fan")
    topdown = analysis.get("topdown")

    lines = [f"🎯 OTE  (Fib Fan + Expansion)  |  {symbol}  (4H → 1H → 30M top-down)"]
    if topdown:
        lines.append(format_topdown_summary(topdown))
        lines.append("—")
    lines.append(f"30M Direction: {direction}  |  Score: {score}/100  |  {'✅ VALID' if valid else '⏳ WAIT'}")
    lines.append(
        f"Impulse: {impulse.get('start', {}).get('type', '?')} → {impulse.get('end', {}).get('type', '?')}  "
        f"({impulse.get('leg_size', 0):.5f})"
    )

    if fans:
        lines.append("Fan rays: " + " · ".join(f["label"] for f in fans))
    if nearest:
        lines.append(f"Nearest Fan: {nearest.get('label')} @ {nearest.get('price', 0):.5f}")
    if expansions:
        lines.append("Expansion targets: " + " · ".join(f"{e['label']} {e['price']:.5f}" for e in expansions[:3]))

    for r in analysis.get("reasons") or []:
        lines.append(f"  • {r}")

    if ticket:
        lines.append(
            f"Ticket: {ticket['side']}  Entry {ticket['entry']:.5f}  "
            f"SL {ticket['sl']:.5f}  TP1 {ticket['tp1']:.5f}  TP2 {ticket['tp2']:.5f}"
        )
        lines.append(f"R:R 1:{ticket.get('rr', 0):.2f}")

    return "\n".join(lines)


def build_ote_ticket(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return analysis.get("ticket")


# ============================================================
# TRENDLINE STRATEGY ORCHESTRATION -- full top-down cascade:
#   4H bias (EMA200 + structure) -> 1H structure permission -> 30M entry
# The 30M trendline family supplies the entry/SL/TP geometry; the 4H/1H
# read gates and scores it (see topdown_engine.get_topdown_bias).
# ============================================================

def run_trendline_analysis(symbol: str) -> Dict[str, Any]:
    topdown = get_topdown_bias(symbol)
    df_30m = market_data.fetch_candles(symbol, "30min", count=250)
    if df_30m is None or df_30m.empty or len(df_30m) < 30:
        return {
            "error": "Insufficient 30M data for Trendline analysis",
            "direction": "NEUTRAL", "symbol": symbol, "topdown": topdown,
        }

    family = build_trendline_family(df_30m, max_lines=4, lookback_bars=90)
    family["symbol"] = symbol
    family["timeframe"] = "30min"
    family["topdown"] = topdown
    if family.get("error"):
        return family

    # Classic chart-pattern scan (triangles, wedges, flags/pennants, H&S,
    # double/triple tops, rectangles) -- this already existed for the
    # auto-trade engine but was never surfaced on the Trendline chart/report.
    try:
        best_pattern, all_patterns = scan_all_patterns(df_30m)
        family["scanned_pattern"] = best_pattern.to_dict() if best_pattern else None
        family["scanned_patterns"] = [p.to_dict() for p in all_patterns]
    except Exception as e:
        print(f"[run_trendline_analysis] pattern scan failed for {symbol}: {e!r}")
        family["scanned_pattern"] = None
        family["scanned_patterns"] = []

    direction = family.get("direction", "NEUTRAL")
    strength = family.get("strength", 0)
    td_dir = topdown.get("direction", "NEUTRAL")
    gating_notes = []

    if direction in ("BUY", "SELL"):
        if td_dir == direction and topdown.get("allowed"):
            strength = min(100, strength + 15)
            gating_notes.append(f"✅ Aligned with 4H/1H top-down bias ({td_dir}) -- structure permission granted")
        elif td_dir == direction and not topdown.get("allowed"):
            gating_notes.append(
                f"Aligned with top-down direction ({td_dir}) but 1H structure permission not yet "
                f"granted -- treat as lower conviction"
            )
        elif td_dir == "NEUTRAL":
            gating_notes.append("4H/1H top-down read is NEUTRAL -- 30M trendline direction stands on its own")
        else:
            strength = max(0, strength - 25)
            gating_notes.append(
                f"⚠️ 30M trendline direction ({direction}) conflicts with 4H/1H top-down bias "
                f"({td_dir}) -- high risk of counter-trend trade"
            )

    family["strength"] = strength
    family["gating_notes"] = gating_notes
    return family
