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
from market_analysis import (
    zigzag_swings, find_swings, compute_volume_profile, detect_confirmation_candle,
    analyse_structure, detect_order_blocks, scan_all_patterns, detect_market_sequence,
)
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
                   kind: str, tol_atr: float = 0.35) -> int:
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


def _count_violations(df: pd.DataFrame, x0: int, y0: float, x1: int, y1: float,
                      kind: str, tol_atr: float = 0.25) -> int:
    """
    How many times price CLOSED on the wrong side of the candidate line.
    A clean support should rarely close below it; a clean resistance should
    rarely close above it. Heavy violations = the line is not real structure.
    """
    if df is None or len(df) < 5:
        return 0
    atr = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).values
    closes = df["Close"].values
    violations = 0
    lo, hi = min(x0, x1), max(x0, x1)
    # Only score the segment after the first pivot (structure is being built)
    for i in range(lo, min(hi + 1, len(df))):
        lv = _line_value(x0, y0, x1, y1, i)
        a = float(atr[i]) if i < len(atr) and atr[i] > 0 else abs(y1 - y0) * 0.05
        tol = max(a * tol_atr, 1e-9)
        if kind == "support" and closes[i] < lv - tol:
            violations += 1
        elif kind == "resistance" and closes[i] > lv + tol:
            violations += 1
    return violations


def _touch_points(df: pd.DataFrame, x0: int, y0: float, x1: int, y1: float,
                   kind: str, tol_atr: float = 0.35) -> List[Dict]:
    """Return actual (index, price) of each touching wick for chart markers."""
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


def _significant_swings(pivots: List[Dict], df: pd.DataFrame, kind: str,
                        min_leg_atr: float = 0.85, max_points: int = 6) -> List[Dict]:
    """
    Keep only the most meaningful swings of the requested type.
    Rank by leg size (ATR multiples) so micro-noise is dropped and the
    line is built from the same pivots a careful trader would use.
    """
    want = "low" if kind == "support" else "high"
    pts = [p for p in pivots if p["type"] == want]
    if len(pts) < 2 or df is None or len(df) < 10:
        return pts[:max_points]

    atr = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).rolling(14).mean().values
    scored = []
    for i, p in enumerate(pts):
        # leg size vs previous opposite-type pivot or previous same-type
        prev = pts[i - 1] if i > 0 else None
        a = float(atr[min(p["index"], len(atr) - 1)]) if len(atr) else 0.0
        a = max(a, 1e-9)
        leg = abs(p["price"] - prev["price"]) / a if prev else 2.0
        scored.append((leg, p))
    # Keep the strongest legs, then re-sort by time
    scored.sort(key=lambda t: t[0], reverse=True)
    kept = [p for leg, p in scored if leg >= min_leg_atr][:max_points]
    if len(kept) < 2:
        kept = [p for _, p in scored[:max(2, max_points // 2)]]
    kept.sort(key=lambda p: p["index"])
    return kept


def _theil_sen_line(points: List[Dict], n: int) -> Optional[Dict]:
    """
    Theil-Sen robust line through swing points.
    Median of all pairwise slopes → resistant to outlier pivots.
    Intercept chosen so the line sits on the median residual (passes
    through the 'middle' of the structure, not pulled by one extreme).
    """
    if len(points) < 2:
        return None
    xs = [float(p["index"]) for p in points]
    ys = [float(p["price"]) for p in points]
    slopes = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = xs[j] - xs[i]
            if abs(dx) < 1e-9:
                continue
            slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return None
    slopes.sort()
    slope = slopes[len(slopes) // 2]  # median slope

    intercepts = [ys[i] - slope * xs[i] for i in range(len(points))]
    intercepts.sort()
    intercept = intercepts[len(intercepts) // 2]

    x0, x1 = int(xs[0]), int(xs[-1])
    y0 = slope * x0 + intercept
    y1 = slope * x1 + intercept
    y_end = slope * (n - 1) + intercept
    return {
        "x0": x0, "y0": float(y0),
        "x1": x1, "y1": float(y1),
        "y_end": float(y_end),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def find_fractal_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[Dict]:
    """
    Simple fractal swing highs and lows.
    No ZigZag, no forced alternation — just clean local extremes.
    This is the sole pivot source for the trendline family.
    """
    if df is None or len(df) < left + right + 2:
        return []

    highs = df["High"].values
    lows = df["Low"].values
    pivots = []

    for i in range(left, len(df) - right):
        # Swing High
        if highs[i] == max(highs[i - left: i + right + 1]):
            pivots.append({"index": i, "price": float(highs[i]), "type": "high"})
        # Swing Low
        if lows[i] == min(lows[i - left: i + right + 1]):
            pivots.append({"index": i, "price": float(lows[i]), "type": "low"})

    pivots.sort(key=lambda p: p["index"])
    return pivots



def find_structural_pivots(
    df: pd.DataFrame,
    left: int = 4,
    right: int = 4,
    min_gap: int = 5,
    min_leg_atr: float = 0.80,
) -> List[Dict]:
    """Find structural pivots from a *line-chart* (Close) first, then
    map each accepted pivot to the candle's true High/Low.

    The close series is deliberately used for detection because it removes
    much of the wick noise that makes raw candle fractals overproduce pivots.
    ATR and minimum-bar filters then require a meaningful swing before a
    pivot is allowed into the structural sequence.

    Output alternates HIGH/LOW whenever possible and contains only pivots
    suitable for the primary trendline/OTE engines.
    """
    if df is None or len(df) < left + right + 5:
        return []

    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(float)
    highs = pd.to_numeric(df["High"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(df["Low"], errors="coerce").to_numpy(float)
    if "ATR" in df.columns:
        atr = pd.to_numeric(df["ATR"], errors="coerce").to_numpy(float)
    else:
        tr = np.maximum(highs - lows,
                        np.maximum(np.abs(highs - np.roll(close, 1)),
                                   np.abs(lows - np.roll(close, 1))))
        tr[0] = highs[0] - lows[0]
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy(float)

    candidates = []
    for i in range(left, len(close) - right):
        window = close[i-left:i+right+1]
        if not np.isfinite(close[i]):
            continue
        is_high = close[i] >= np.max(window)
        is_low = close[i] <= np.min(window)
        # A close cannot normally be both, but a flat series can produce both.
        # Skip flat dual-pivots rather than inventing structure.
        if is_high and is_low:
            continue
        if is_high:
            candidates.append({
                "index": i, "price": float(highs[i]), "close_price": float(close[i]),
                "type": "high", "source": "line_close",
                "atr": max(float(atr[i]), 1e-9),
            })
        elif is_low:
            candidates.append({
                "index": i, "price": float(lows[i]), "close_price": float(close[i]),
                "type": "low", "source": "line_close",
                "atr": max(float(atr[i]), 1e-9),
            })

    if not candidates:
        return []

    # Collapse nearby/same-type candidates first.
    compact = []
    for p in candidates:
        if not compact:
            compact.append(p)
            continue
        q = compact[-1]
        gap = p["index"] - q["index"]
        if p["type"] == q["type"] and gap < min_gap:
            more_extreme = (
                p["close_price"] > q["close_price"]
                if p["type"] == "high"
                else p["close_price"] < q["close_price"]
            )
            if more_extreme:
                compact[-1] = p
            continue
        compact.append(p)

    # Enforce alternating structure and meaningful leg size.
    out = []
    for p in compact:
        if not out:
            out.append(p)
            continue
        q = out[-1]
        gap = p["index"] - q["index"]
        if gap < min_gap:
            continue

        if p["type"] == q["type"]:
            more_extreme = (
                p["close_price"] > q["close_price"]
                if p["type"] == "high"
                else p["close_price"] < q["close_price"]
            )
            if more_extreme:
                out[-1] = p
            continue

        leg = abs(p["close_price"] - q["close_price"])
        leg_atr = max(float(p["atr"]), float(q["atr"]), 1e-9)
        if leg >= min_leg_atr * leg_atr:
            p["leg_atr"] = float(leg / leg_atr)
            out.append(p)

    return out

def _get_sequential_pivots(pivots: List[Dict], kind: str, min_bars: int = 5) -> List[Dict]:
    """
    Keep only sequential higher lows (support) or lower highs (resistance).
    Restarts the chain when structure is broken so the most recent
    valid sequence is always used — this is what makes the line hug
    current price action.
    """
    want = "low" if kind == "support" else "high"
    pts = [p for p in pivots if p["type"] == want]
    if len(pts) < 2:
        return pts

    cleaned = [pts[0]]
    for p in pts[1:]:
        gap = p["index"] - cleaned[-1]["index"]
        if gap < min_bars:
            # Too close — keep the more extreme one
            if kind == "support":
                if p["price"] < cleaned[-1]["price"]:
                    cleaned[-1] = p
            else:
                if p["price"] > cleaned[-1]["price"]:
                    cleaned[-1] = p
            continue

        if kind == "support":
            if p["price"] > cleaned[-1]["price"] * 0.998:
                cleaned.append(p)
            else:
                cleaned = [p]  # restart on lower low
        else:
            if p["price"] < cleaned[-1]["price"] * 1.002:
                cleaned.append(p)
            else:
                cleaned = [p]  # restart on higher high

    return cleaned


def _fit_primary(pivots: List[Dict], kind: str, n: int, df: pd.DataFrame) -> Optional[Dict]:
    """
    Classic trendline: connect sequential Higher Lows (support) or
    Lower Highs (resistance). Filters are kept light so the line is
    almost always drawn when structure exists — matching MT5 hand-drawn style.
    """
    pts = _get_sequential_pivots(pivots, kind, min_bars=3)
    if len(pts) < 2:
        # Fallback: any two pivots of the correct type
        want = "low" if kind == "support" else "high"
        pts = [p for p in pivots if p["type"] == want]
        if len(pts) < 2:
            return None
        pts = pts[-2:]

    # Prefer the longest clean span in the current structural sequence.
    # This is deliberately different from the local pattern engine: the
    # primary trendline should describe the broader move, not just the last
    # two pivots. Keep it bounded to the recent structural chain so an old
    # regime does not dominate a newly changed market.
    max_chain = pts[-6:]
    span_candidates = []
    for i in range(len(max_chain) - 1):
        for j in range(i + 1, len(max_chain)):
            a0, b0 = max_chain[i], max_chain[j]
            span = b0["index"] - a0["index"]
            if span < 12:
                continue
            span_candidates.append((span, a0, b0))
    if span_candidates:
        _, a, b = max(span_candidates, key=lambda z: (z[0], z[2]["index"]))
    else:
        a, b = pts[-2], pts[-1]

    if b["index"] <= a["index"]:
        return None
    if b["index"] - a["index"] < 4:
        return None

    slope = (b["price"] - a["price"]) / max(b["index"] - a["index"], 1)
    y_end = _line_value(a["index"], a["price"], b["index"], b["price"], n - 1)

    # Direction must match kind (light check only)
    if kind == "support" and slope < -1e-12:
        return None
    if kind == "resistance" and slope > 1e-12:
        return None

    touches = _count_touches(df, a["index"], a["price"], b["index"], b["price"], kind, tol_atr=0.50)
    violations = _count_violations(df, a["index"], a["price"], b["index"], b["price"], kind, tol_atr=0.35)

    quality = "unconfirmed" if touches < 2 else ("confirmed" if touches <= 4 else "crowded")

    return {
        "x0": a["index"], "y0": float(a["price"]),
        "x1": b["index"], "y1": float(b["price"]),
        "y_end": float(y_end),
        "slope": float(slope),
        "touches": max(touches, 2),
        "violations": violations,
        "confirmed": quality == "confirmed",
        "quality": quality,
        "kind": kind,
        "bars_since_last_touch": n - 1 - b["index"],
        "method": "classic_sequential",
    }


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

    Strict rules (aligned with market_analysis.detect_double_top/bottom):
    - Peaks/troughs must be within ~0.55% of each other (relative).
    - For M: the second high must not be meaningfully higher than the first
      (higher-high = continuation, not Double Top).
    - For W: the second low must not be meaningfully lower than the first.
    - Minimum bar separation and meaningful depth required.
    """
    if not pivots or len(pivots) < 3 or df is None or len(df) < 20:
        return None
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else abs(float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]))
    atr = max(atr, 1e-9)
    # Aligned with the stricter classic Double Top/Bottom detectors
    max_rel_diff = 0.0035  # 0.35%
    min_bars = 12
    min_depth_atr = 1.4
    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]

    # Double top (M) — last two significant highs
    if len(highs) >= 2:
        for i in range(len(highs) - 1, 0, -1):
            h2, h1 = highs[i], highs[i - 1]
            if h2["index"] <= h1["index"] or (h2["index"] - h1["index"]) < min_bars:
                continue
            p1, p2 = float(h1["price"]), float(h2["price"])
            rel = abs(p2 - p1) / max(abs(p1), 1e-9)
            if rel > max_rel_diff:
                continue
            # Reject higher-high continuation
            if p2 > p1 * 1.002:
                continue
            between = [p for p in lows if h1["index"] < p["index"] < h2["index"]]
            if not between:
                continue
            neck = min(between, key=lambda p: p["price"])
            depth = max(p1, p2) - float(neck["price"])
            if depth < min_depth_atr * atr:
                continue
            # Freshness
            if (len(df) - 1 - h2["index"]) > 25:
                continue
            return {
                "pattern": "M",
                "name": "Double Top (M)",
                "left": h1, "right": h2,
                "neckline": neck["price"],
                "neck_index": neck["index"],
                "bias": "SELL",
                "note": f"Clean M pattern — neckline at {neck['price']:.5f} (peaks within {rel*100:.2f}%)",
            }

    # Double bottom (W)
    if len(lows) >= 2:
        for i in range(len(lows) - 1, 0, -1):
            l2, l1 = lows[i], lows[i - 1]
            if l2["index"] <= l1["index"] or (l2["index"] - l1["index"]) < min_bars:
                continue
            p1, p2 = float(l1["price"]), float(l2["price"])
            rel = abs(p2 - p1) / max(abs(p1), 1e-9)
            if rel > max_rel_diff:
                continue
            # Reject lower-low continuation
            if p2 < p1 * 0.998:
                continue
            between = [p for p in highs if l1["index"] < p["index"] < l2["index"]]
            if not between:
                continue
            neck = max(between, key=lambda p: p["price"])
            height = float(neck["price"]) - min(p1, p2)
            if height < min_depth_atr * atr:
                continue
            if (len(df) - 1 - l2["index"]) > 25:
                continue
            return {
                "pattern": "W",
                "name": "Double Bottom (W)",
                "left": l1, "right": l2,
                "neckline": neck["price"],
                "neck_index": neck["index"],
                "bias": "BUY",
                "note": f"Clean W pattern — neckline at {neck['price']:.5f} (bottoms within {rel*100:.2f}%)",
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
        return {"checks": checks, "passed": 0, "required": 3, "confirmed": False}

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
    # Strict mode: require 3 out of 4 for a high-probability setup.
    # Mediocre 2/4 setups are no longer considered confirmed.
    required = 3
    return {"checks": checks, "passed": passed, "required": required, "confirmed": passed >= required}




def _trendline_retest_state(df: pd.DataFrame, line: Optional[Dict[str, Any]],
                            breakout: Optional[Dict[str, Any]],
                            break_kind: Optional[str]) -> Dict[str, Any]:
    """
    Confirm the classic break -> retest sequence without predicting.

    A retest is only marked when:
      1) the line was actually broken by a candle close,
      2) a later candle trades back to the line within 0.35 ATR,
      3) price closes back on the breakout side of the line.

    A wick through the line that closes back across it is classified as a
    likely fakeout, not a confirmed breakout/retest.
    """
    out = {
        "status": "INTACT",
        "break_index": None,
        "retest_index": None,
        "retest_level": None,
        "fakeout": False,
        "note": "No confirmed trendline break.",
    }
    if df is None or df.empty or not line:
        return out

    n = len(df)
    def lv(i):
        return _line_value(line["x0"], line["y0"], line["x1"], line["y1"], i)

    # Locate the first meaningful close beyond the line in the recent window.
    kind = break_kind
    if kind is None:
        if breakout and breakout.get("strength") in ("confirmed", "developing", "weak"):
            # Infer from the line role and current close.
            kind = "support_break_down" if line.get("kind") == "support" else "resistance_break_up"
    if kind not in ("support_break_down", "resistance_break_up"):
        return out

    is_down = kind == "support_break_down"
    closes = df["Close"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    opens = df["Open"].to_numpy(float)
    atrs = df["ATR"].to_numpy(float) if "ATR" in df.columns else (df["High"]-df["Low"]).to_numpy(float)

    start = max(int(line.get("x1", 0)) + 1, n - 30)
    break_i = None
    for i in range(start, n):
        line_i = lv(i)
        if (closes[i] < line_i) if is_down else (closes[i] > line_i):
            body = abs(closes[i] - opens[i])
            rng = max(highs[i] - lows[i], 1e-9)
            pen = abs(closes[i] - line_i) / max(float(atrs[i]), 1e-9)
            if pen >= 0.10 and body / rng >= 0.35:
                break_i = i
                break

    if break_i is None:
        # A wick-only excursion through the line is a fakeout candidate.
        for i in range(start, n):
            line_i = lv(i)
            wick_cross = (highs[i] > line_i and closes[i] <= line_i) if not is_down else (
                lows[i] < line_i and closes[i] >= line_i
            )
            if wick_cross:
                out.update(status="FAKEOUT", break_index=i, fakeout=True,
                           retest_level=float(line_i),
                           note="Wick crossed the trendline but the candle reclaimed it.")
                return out
        return out

    out["break_index"] = break_i
    # If price has not had a chance to retest, keep the break state.
    tol_mult = 0.35
    for i in range(break_i + 1, n):
        line_i = lv(i)
        atr = max(float(atrs[i]), 1e-9)
        touched = (highs[i] >= line_i - atr*tol_mult) if is_down else (
            lows[i] <= line_i + atr*tol_mult
        )
        # The retest candle must close on the new side of the line.
        held = closes[i] < line_i if is_down else closes[i] > line_i
        if touched and held:
            out.update(
                status="BREAK_RETEST_CONFIRMED",
                retest_index=i,
                retest_level=float(line_i),
                note="Confirmed break followed by a held retest."
            )
            return out

        # Reclaim across the line after the break = failed break/fakeout.
        reclaimed = closes[i] > line_i if is_down else closes[i] < line_i
        if reclaimed:
            out.update(
                status="FAKEOUT",
                retest_index=i,
                retest_level=float(line_i),
                fakeout=True,
                note="Break was reclaimed before a valid retest held."
            )
            return out

    out["status"] = "BREAK_CONFIRMED" if breakout and breakout.get("strength") == "confirmed" else "BREAK_DEVELOPING"
    out["retest_level"] = float(lv(n-1))
    out["note"] = "Trendline break detected; waiting for a clean retest."
    return out


def _trendline_swing_annotations(pivots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return compact HH/HL/LH/LL labels for the pivots used by the lines."""
    highs = [p for p in pivots if p.get("type") == "high"]
    lows = [p for p in pivots if p.get("type") == "low"]
    labels = []
    prev_h = prev_l = None
    for p in sorted(pivots, key=lambda z: z.get("index", 0)):
        if p.get("type") == "high":
            label = "HH" if prev_h is not None and p["price"] > prev_h else "LH" if prev_h is not None else "H"
            prev_h = p["price"]
        else:
            label = "HL" if prev_l is not None and p["price"] > prev_l else "LL" if prev_l is not None else "L"
            prev_l = p["price"]
        labels.append({"index": int(p["index"]), "price": float(p["price"]),
                       "label": label, "type": p.get("type")})
    return labels

def build_trendline_family(df: pd.DataFrame, max_lines: int = 4, lookback_bars: int = 60) -> Dict[str, Any]:
    """
    Build one clean parallel family (ascending OR descending), not both mixed.
    Market reveals direction: price relative to the family rails.

    Uses a filtered line-chart pivot engine. Close-based pivots are detected first,
    then mapped to candle extremes. Major structural pivots feed the long trendline;
    local pattern detectors remain separate. This reduces wick noise while preserving
    the actual swing prices used for drawing.
    """
    if df is None or len(df) < 30:
        return {"error": "Insufficient data for trendline family", "direction": "NEUTRAL", "pivots": []}

    n = len(df)

    # --- Simple fractal pivots (NO ZigZag) ---
    pivots = find_structural_pivots(df, left=4, right=4, min_gap=5, min_leg_atr=0.80)
    if len(pivots) < 5:
        pivots = find_structural_pivots(df, left=3, right=3, min_gap=4, min_leg_atr=0.65)

    # Recent window for diagonal fit (tighter for better hugging)
    cutoff = max(0, n - lookback_bars)
    recent_pivots = [p for p in pivots if p["index"] >= cutoff]
    if len(recent_pivots) < 4:
        recent_pivots = [p for p in pivots if p["index"] >= max(0, n - int(lookback_bars * 1.5))]
    if len(recent_pivots) < 3:
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
    MIN_TREND_MOVE_ATR = 0.35
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

    # SIMPLE RULE (matches MT5 hand-drawn style):
    # Always keep BOTH a clean ascending support AND a clean descending
    # resistance when they exist. This is exactly how a trader draws the
    # two green lines on the chart you showed.
    close = float(df["Close"].iloc[-1])
    primary = None
    family_kind = "none"

    if support and resistance:
        # Both valid → treat as a converging structure (triangle / wedge)
        # Pick the primary for scoring based on which side price is closer to,
        # but we will still return BOTH lines for drawing.
        s_end = support["y_end"]
        r_end = resistance["y_end"]
        if abs(close - s_end) <= abs(close - r_end):
            primary, family_kind = support, "ascending"
        else:
            primary, family_kind = resistance, "descending"
    elif support:
        primary, family_kind = support, "ascending"
    elif resistance:
        primary, family_kind = resistance, "descending"

    # KEEP the primary trendline even after a break.
    # Classic breakout + retest analysis (like the reference image) requires
    # the line to stay visible so the break and the retest can be scored.
    # Direction / strength logic below correctly labels the current state.

    family_lines = []
    channel = None
    if primary:
        # Prefer a single clean primary trendline (the black line style in
        # the reference image). Only add a parallel rail when the channel
        # is meaningfully wide.
        family_lines = [primary]
        extras = _build_parallel_family(primary, recent_pivots, n, max_members=2)
        if len(extras) >= 2:
            width = abs(extras[-1]["y_end"] - extras[0]["y_end"])
            atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else None
            if atr_now and width > 0.8 * atr_now:
                family_lines = extras
                channel = {
                    "lower": family_lines[0],
                    "upper": family_lines[-1],
                    "mid_end": (family_lines[0]["y_end"] + family_lines[-1]["y_end"]) / 2.0,
                    "width": width,
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
            # Price clearly above rising support → BUY bias.
            # But if price is only sitting ON / testing the line, stay cautious
            # (WAIT) instead of forcing a buy — the line can still break.
            atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else abs(upper - lower) * 0.1
            near_buffer = max(atr_now * 0.35, 1e-9)
            if close >= lower + near_buffer:
                direction = "BUY"
                strength = 60 + min(25, primary["touches"] * 7)
                reasons.append(f"Ascending family · {touch_note}")
                reasons.append("Short-term Trend: BUY (price clearly above rising support)")
                if primary.get("quality") == "unconfirmed":
                    strength -= 10
                elif primary.get("quality") == "crowded":
                    strength -= 5
                if close > mid:
                    reasons.append("Price in upper half of channel — bullish control")
                    strength += 12
                else:
                    reasons.append("Price above support but not extended — watch continuation")
            elif close >= lower - near_buffer * 0.5:
                # Sitting on / slightly below the line → no forced direction
                direction = "NEUTRAL"
                strength = 40
                reasons.append(f"Ascending family · {touch_note}")
                reasons.append("Price testing rising support — WAIT for bounce or confirmed break")
            else:
                brk = _grade_breakout(df, primary, "support_break_down", n)
                direction = "SELL"
                if brk["strength"] == "confirmed":
                    strength = 68
                    reasons.append(f"Confirmed break below ascending support — "
                                    f"{brk['penetration_atr']} ATR beyond, {brk['consecutive_closes']} closes, "
                                    f"body {brk['body_ratio']}")
                    reasons.append("Short-term Trend: SELL (break of rising support)")
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
            # Price clearly below falling resistance → SELL bias.
            # If only testing the line, stay neutral instead of forcing sell.
            atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else abs(upper - lower) * 0.1
            near_buffer = max(atr_now * 0.35, 1e-9)
            if close <= upper - near_buffer:
                direction = "SELL"
                strength = 60 + min(25, primary["touches"] * 7)
                reasons.append(f"Descending family · {touch_note}")
                reasons.append("Short-term Trend: SELL (price clearly below falling resistance)")
                if primary.get("quality") == "unconfirmed":
                    strength -= 10
                elif primary.get("quality") == "crowded":
                    strength -= 5
                if close < mid:
                    reasons.append("Price in lower half of channel — bearish control")
                    strength += 12
                else:
                    reasons.append("Price below resistance but not extended — watch continuation")
            elif close <= upper + near_buffer * 0.5:
                direction = "NEUTRAL"
                strength = 40
                reasons.append(f"Descending family · {touch_note}")
                reasons.append("Price testing falling resistance — WAIT for reject or confirmed break")
            else:
                brk = _grade_breakout(df, primary, "resistance_break_up", n)
                direction = "BUY"
                if brk["strength"] == "confirmed":
                    strength = 68
                    reasons.append(f"Confirmed break above descending resistance — "
                                    f"{brk['penetration_atr']} ATR beyond, {brk['consecutive_closes']} closes, "
                                    f"body {brk['body_ratio']}")
                    reasons.append("Short-term Trend: BUY (break of falling resistance)")
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

    # Trendline lifecycle: intact -> break -> retest (or fakeout).
    # Keep the line visible after a break so the chart can show the exact
    # break/retest geometry rather than deleting the evidence.
    trendline_retest = {"status": "INTACT", "note": "No confirmed trendline break."}
    if primary and breakout_grade:
        break_kind = "support_break_down" if primary.get("kind") == "support" else "resistance_break_up"
        trendline_retest = _trendline_retest_state(df, primary, breakout_grade, break_kind)
        if trendline_retest.get("status") == "BREAK_RETEST_CONFIRMED":
            reasons.append("✅ Trendline break + retest confirmed — continuation entry can be evaluated.")
        elif trendline_retest.get("status") == "FAKEOUT":
            reasons.append("🚫 Trendline break reclaimed — treat as fakeout, not confirmation.")
        elif trendline_retest.get("status") in ("BREAK_CONFIRMED", "BREAK_DEVELOPING"):
            reasons.append("⏳ Trendline broken — wait for a clean retest before chasing.")

    continuation_state = _classify_trendline_state({"trendline_retest": trendline_retest, "family_kind": family_kind, "trendline_annotations": _trendline_swing_annotations(recent_pivots)})
    topdown_context = None
    # The orchestration adds topdown after this function, but accepting a pre-existing
    # context keeps this builder reusable.
    topdown_context = {}
    
    trendline_annotations = _trendline_swing_annotations(recent_pivots)

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
    # Only let a wedge set direction when it is reasonably strong.
    # Weak converging geometry should not create trade bias.
    if wedge and direction == "NEUTRAL" and wedge["bias"] != "NEUTRAL" and strength >= 55:
        direction = wedge["bias"]
        strength = max(strength, 60)
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

    # --- Market sequence: RBR / DBD / RBD / DBR -------------------------
    # Core context layer. Continuation sequences (RBR/DBD) support flags
    # and trend continuation. Reversal leans (RBD/DBR) warn against
    # chasing the prior impulse and favor M/W/H&S style setups.
    market_seq = None
    try:
        market_seq = detect_market_sequence(df, lookback=min(90, len(df)))
    except Exception as e:
        print(f"[build_trendline_family] market sequence failed: {e!r}")
        market_seq = None
    if market_seq:
        reasons.append(market_seq["note"])
        seq = market_seq["sequence"]
        seq_bias = market_seq["bias"]
        seq_conf = float(market_seq.get("confidence") or 0)
        if seq in ("RBR", "DBD") and seq_conf >= 60:
            # Continuation sequence — reinforce matching direction
            if direction == "NEUTRAL":
                direction = seq_bias
                strength = max(strength, int(seq_conf) - 5)
            elif direction == seq_bias:
                strength = min(100, strength + 10)
            else:
                # Sequence disagrees with geometry — reduce conviction
                strength = max(0, strength - 12)
                reasons.append(f"⚠ {seq} sequence conflicts with current bias — prefer WAIT for confirmation")
        elif seq in ("RBD", "DBR") and seq_conf >= 58:
            # Reversal lean — do not let continuation patterns force entry
            if direction == seq_bias:
                strength = min(100, strength + 6)
            else:
                strength = max(0, strength - 10)
                reasons.append(
                    f"⚠ {seq} (reversal lean) — avoid chasing prior impulse; "
                    f"prefer M/W/H&S or wait for break & retest"
                )

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
    nearest_unmit_primary = None
    atr_ref = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else abs(close) * 0.002
    for ob in order_blocks:
        if float(ob["bottom"]) <= close <= float(ob["top"]):
            active_ob = ob
            break
        if (ob.get("freshness") == "untested" and not ob.get("is_inducement")
                and nearest_unmit_primary is None):
            dist = min(abs(close - float(ob["top"])), abs(close - float(ob["bottom"])))
            if dist < atr_ref * 2.8:
                nearest_unmit_primary = ob

    # Prefer unmitigated primary OB as the entry zone when price is close
    if nearest_unmit_primary is not None and active_ob is None:
        nob = nearest_unmit_primary
        nob_side = nob["type"]
        if (direction == "BUY" and nob_side == "bullish") or (direction == "SELL" and nob_side == "bearish"):
            reasons.append(
                f"📍 Preferred entry: unmitigated {nob_side} OB ({nob['confidence']}%) nearby — "
                f"wait for confirmation at this zone"
            )
            strength = min(100, strength + 6)
        elif direction in ("BUY", "SELL") and nob_side != ("bullish" if direction == "BUY" else "bearish"):
            reasons.append(
                f"⚠ Unmitigated opposite OB nearby — treat with caution / wait for clear reaction"
            )

    if active_ob:
        ob_side = active_ob["type"]  # 'bullish' or 'bearish'
        role_tag = " [INDUCEMENT]" if active_ob.get("is_inducement") else ""
        ob_desc = (f"{ob_side.capitalize()} order block ({active_ob['grade']}, "
                   f"{active_ob['confidence']}%, {active_ob['freshness']}){role_tag}")
        if active_ob.get("is_inducement"):
            reasons.append(
                f"⚠ Price inside INDUCEMENT OB — expect liquidity grab then move toward the "
                f"primary unmitigated zone"
            )
            strength = max(0, strength - 8)
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
        # Always expose both lines when they exist (MT5-style dual trendlines)
        "uptrends": [support] if support else [],
        "downtrends": [resistance] if resistance else [],
        "channel": channel,
        "wedge": wedge,
        "horizontal_levels": horizontal_levels,
        "projections": projections,
        "mw_pattern": mw,
        "market_sequence": market_seq,
        "pivots": pivots[-16:],
        "volume_profile": vp,
        "upper_line": upper_line,
        "lower_line": lower_line,
        "middle_line": mid_line,
        "df": df,
        "mode": "channel" if channel else "lines",
        "breakout_grade": breakout_grade,
        "trendline_retest": trendline_retest,
        "trendline_annotations": trendline_annotations,
        "trendline_status": trendline_retest.get("status", "INTACT"),
        "continuation_state": continuation_state.get("state", "CONTINUATION"),
        "state_reason": continuation_state.get("reason", ""),
        "htf_key_levels_4h": [],
        "htf_swings_4h": [],
        "htf_swings_1h": [],
        "primary_quality": primary.get("quality") if primary else None,
        "primary_touches": primary.get("touches") if primary else 0,
        # Single "which pattern wins the chart" decision, so the renderer
        # never has to draw wedge + M/W + channel all at once. Priority:
        # M/W (specific reversal structure, high signal) > wedge/triangle
        # (specific continuation/reversal structure) > plain channel/bias
        # line (just "this is the trend", not a named pattern).
        # Only promote wedge/channel as the active pattern when strength is
        # meaningful. Low-strength geometry (e.g. 43%) is noise — keep it
        # for context drawing but do not present it as a tradeable pattern.
        "active_pattern": (
            "mw" if mw else
            "wedge" if (wedge and strength >= 60) else
            "channel" if (channel and strength >= 55) else
            "none"
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


# Minimum acceptable reward:risk on the FIRST partial (TP1). A signal
# whose nearest liquidity pool sits closer than this to entry is not
# useful risk management, so TP1 (and TP2) are chosen from liquidity
# levels that actually clear this bar -- falling back to a synthetic
# risk-multiple target (never a real level closer than that) if nothing
# in the liquidity list does. This guarantees every position container
# this function returns has rr_tp1 >= MIN_RR, no exceptions.
MIN_RR = 1.5

# If price has already run more than this many ATRs beyond the rail it
# broke, a "confirmed" breakout still shouldn't be chased at market --
# same rule the classic chart-pattern engine (execution_engine's
# ConfirmationEngine) already applies. Route to a Fibonacci pullback
# zone instead.
# Lowered from 2.0 → 1.3 so we stop chasing extended moves much earlier.
FAR_ATR_MULTIPLE = 1.3
# Beyond this many ATRs we refuse to build an entry container at all
# (trend already ran without us — wait for a real pullback instead of
# catching the falling knife / chasing the top).
TOO_EXTENDED_ATR = 2.4
FIB_ZONE_ENTRY_ANCHOR = 0.618


def _fib_pullback_entry(trigger_price: float, extreme_price: float, direction: str) -> float:
    """61.8% retracement anchor of the trigger->extreme leg, used as the
    LIMIT entry when price is already stretched past FAR_ATR_MULTIPLE."""
    if direction == "BUY":
        leg = extreme_price - trigger_price
        return extreme_price - leg * FIB_ZONE_ENTRY_ANCHOR
    leg = trigger_price - extreme_price
    return extreme_price + leg * FIB_ZONE_ENTRY_ANCHOR


def _select_tp_targets(liq: List[float], entry: float, risk: float, direction: str,
                        min_rr: float = MIN_RR) -> Tuple[float, float, bool]:
    """Pick TP1/TP2 from real liquidity levels, but only ones that clear
    min_rr against the actual risk. Returns (tp1, tp2, tp1_is_synthetic).
    If no real level clears the bar, TP1/TP2 fall back to fixed
    risk-multiples (min_rr and 2*min_rr) so the signal is never handed
    out with a sub-minimum R:R."""
    if risk <= 0:
        sign = 1 if direction == "BUY" else -1
        return entry + sign * risk, entry + sign * risk * 2, True
    sign = 1 if direction == "BUY" else -1
    valid = [lvl for lvl in liq if (abs(lvl - entry) / risk) >= min_rr]
    if valid:
        tp1 = valid[0]
        tp2 = valid[1] if len(valid) > 1 else entry + sign * risk * min_rr * 2
        return tp1, tp2, False
    tp1 = entry + sign * risk * min_rr
    tp2 = entry + sign * risk * min_rr * 2
    return tp1, tp2, True


def build_position_container(family: Dict[str, Any], atr_mult_sl: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Position box with DYNAMIC R:R from real liquidity distance.

    Entry  : nearest structure rail / close (or a Fibonacci pullback zone
             if price is already stretched too far to chase -- see
             FAR_ATR_MULTIPLE below)
    SL     : beyond invalidation (last opposing swing or rail break)
    TP1/TP2: nearest / next liquidity pools that clear MIN_RR; falls back
             to a fixed risk-multiple if no real level does
    R:R    : |TP1 - Entry| / |Entry - SL|  -- guaranteed >= MIN_RR
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
    # Pattern still FORMING or FAKEOUT → never mark confirmed / ACTIVE
    if family.get("force_wait_pattern"):
        confirmed = False

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

        # Hard rule: if price has already run too far without us, do NOT
        # build a chase entry near market. Wait for a real pullback.
        extension_atr = (close - entry) / atr if atr > 0 else 0
        if extension_atr > TOO_EXTENDED_ATR:
            return {
                "side": "LONG",
                "direction": "BUY",
                "entry": None,
                "sl": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
                "rr": 0,
                "confirmed": False,
                "order_type": None,
                "entry_note": (f"Trend already extended {extension_atr:.1f}×ATR — "
                               f"no entry. Wait for pullback toward structure instead of chasing."),
                "too_extended": True,
            }

        # SL: beyond real structural invalidation (not micro-noise).
        # Priority: pattern extreme (W bottom / H&S head) > significant swing low
        # > trendline > ATR fallback. Buffer = 0.35×ATR beyond the level.
        sl_candidates = []
        sp = family.get("scanned_pattern")
        if sp and sp.get("bias") == "BUY":
            for kp in (sp.get("key_points") or []):
                if len(kp) >= 3 and any(t in str(kp[2]).lower() for t in ("bottom", "head", "shoulder")):
                    sl_candidates.append(float(kp[1]))
        if mw and mw.get("pattern") == "W":
            for side in ("left", "right"):
                p = mw.get(side)
                if p and p.get("price") is not None:
                    sl_candidates.append(float(p["price"]))
        # Significant swing lows only (skip tiny noise pivots)
        swing_lows = []
        for p in pivots:
            if p.get("type") != "low":
                continue
            px = float(p["price"])
            if px >= entry:
                continue
            swing_lows.append(px)
        if swing_lows:
            # Prefer the lowest of the last 2 significant lows under entry
            recent = sorted(swing_lows)[:2] if len(swing_lows) >= 2 else swing_lows
            sl_candidates.extend(recent)
            sl_candidates.append(min(swing_lows))  # structural extreme
        if trendline_val is not None and trendline_val < entry:
            sl_candidates.append(trendline_val)

        if sl_candidates:
            # SL must be below entry; pick the level that gives room (not the tightest)
            below = [x for x in sl_candidates if x < entry - atr * 0.15]
            if below:
                # Use the highest level that still has at least ~0.6 ATR risk
                # (avoids ultra-tight stops) but never wider than ~2.8 ATR
                viable = [x for x in below if (entry - x) >= atr * 0.55]
                if viable:
                    sl = max(viable)  # closest viable = tightest still-valid structure
                else:
                    sl = min(below)  # only deep structure available
            else:
                sl = entry - atr * max(atr_mult_sl, 1.0)
            sl = sl - atr * 0.35  # buffer beyond the invalidation level
        else:
            sl = entry - atr * max(atr_mult_sl, 1.2)

        # Cap risk so SL never balloons past ~3 ATR
        if atr > 0 and (entry - sl) > atr * 3.0:
            sl = entry - atr * 3.0

        liq = _liquidity_targets(pivots, "BUY", entry)
        if mw and mw.get("pattern") == "W" and mw.get("neckline", 0) > entry:
            liq = sorted(set(liq + [float(mw["neckline"])]))
        if vp.get("poc_price") and vp["poc_price"] > entry:
            liq = sorted(set(liq + [float(vp["poc_price"])]))
        if not liq and projs:
            liq = [float(p["price"]) for p in projs if p["price"] > entry]

        order_type = "LIMIT" if entry < close - atr * 0.08 else "MARKET"
        entry_note = None
        # Pattern neckline retest preferred over chasing the breakout
        if family.get("prefer_retest_entry") and family.get("retest_level") is not None:
            retest = float(family["retest_level"])
            if retest < close:
                entry = retest
                order_type = "LIMIT"
                entry_note = (
                    f"Pattern {family.get('pattern_stage', 'TRIGGERED')} — "
                    f"entry routed to neckline retest at {entry:.5f} (do not chase breakout)"
                )
                if sl >= entry:
                    sl = entry - atr * 1.0
        elif brk and brk["strength"] != "confirmed":
            entry = brk["retest_level"]
            order_type = "LIMIT"
            entry_note = (f"Unconfirmed breakout ({brk['strength']}) -- entry routed to retest of the "
                          f"broken rail at {entry:.5f} instead of chasing at market")
            # Rebuild SL relative to new entry
            if sl >= entry:
                sl = entry - atr * 1.0
        elif order_type == "MARKET" and atr > 0 and (close - entry) > FAR_ATR_MULTIPLE * atr:
            pullback_entry = _fib_pullback_entry(entry, close, "BUY")
            entry_note = (f"Price extended {(close - entry) / atr:.1f}x ATR beyond the rail -- "
                          f"entry routed to a pullback at {pullback_entry:.5f} instead of chasing")
            entry = pullback_entry
            order_type = "LIMIT"
            if sl >= entry:
                sl = entry - atr * 1.0

        risk = abs(entry - sl)
        # Reject absurdly tight stops (noise, not structure)
        if atr > 0 and risk < atr * 0.45:
            sl = entry - atr * 0.80
            risk = abs(entry - sl)
            entry_note = ((entry_note + " " if entry_note else "") +
                          "SL widened to minimum structure distance (0.8×ATR).")
        tp1, tp2, tp1_synthetic = _select_tp_targets(liq, entry, risk, "BUY", MIN_RR)
        if tp1_synthetic:
            entry_note = ((entry_note + " " if entry_note else "") +
                          f"No liquidity level cleared the {MIN_RR:.1f}R minimum for TP1 -- "
                          f"using a fixed {MIN_RR:.1f}R target instead.")
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

        # Hard rule: do not chase an already-extended down move
        extension_atr = (entry - close) / atr if atr > 0 else 0
        if extension_atr > TOO_EXTENDED_ATR:
            return {
                "side": "SHORT",
                "direction": "SELL",
                "entry": None,
                "sl": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
                "rr": 0,
                "confirmed": False,
                "order_type": None,
                "entry_note": (f"Trend already extended {extension_atr:.1f}×ATR — "
                               f"no entry. Wait for pullback toward structure instead of catching the falling knife."),
                "too_extended": True,
            }

        # SL: beyond real structural invalidation (mirror of BUY logic)
        sl_candidates = []
        sp = family.get("scanned_pattern")
        if sp and sp.get("bias") == "SELL":
            for kp in (sp.get("key_points") or []):
                if len(kp) >= 3 and any(t in str(kp[2]).lower() for t in ("top", "head", "shoulder")):
                    sl_candidates.append(float(kp[1]))
        if mw and mw.get("pattern") == "M":
            for side in ("left", "right"):
                p = mw.get(side)
                if p and p.get("price") is not None:
                    sl_candidates.append(float(p["price"]))
        swing_highs = []
        for p in pivots:
            if p.get("type") != "high":
                continue
            px = float(p["price"])
            if px <= entry:
                continue
            swing_highs.append(px)
        if swing_highs:
            recent = sorted(swing_highs, reverse=True)[:2] if len(swing_highs) >= 2 else swing_highs
            sl_candidates.extend(recent)
            sl_candidates.append(max(swing_highs))
        if trendline_val is not None and trendline_val > entry:
            sl_candidates.append(trendline_val)

        if sl_candidates:
            above = [x for x in sl_candidates if x > entry + atr * 0.15]
            if above:
                viable = [x for x in above if (x - entry) >= atr * 0.55]
                if viable:
                    sl = min(viable)  # closest viable structure above
                else:
                    sl = max(above)
            else:
                sl = entry + atr * max(atr_mult_sl, 1.0)
            sl = sl + atr * 0.35  # buffer beyond invalidation
        else:
            sl = entry + atr * max(atr_mult_sl, 1.2)

        if atr > 0 and (sl - entry) > atr * 3.0:
            sl = entry + atr * 3.0

        liq = _liquidity_targets(pivots, "SELL", entry)
        if mw and mw.get("pattern") == "M" and mw.get("neckline", 0) < entry:
            liq_set = list(liq) + [float(mw["neckline"])]
            liq = sorted(liq_set, reverse=True)
        if vp.get("poc_price") and vp["poc_price"] < entry:
            liq = sorted(set(list(liq) + [float(vp["poc_price"])]), reverse=True)
        if not liq and projs:
            liq = [float(p["price"]) for p in projs if p["price"] < entry]

        order_type = "LIMIT" if entry > close + atr * 0.08 else "MARKET"
        entry_note = None
        if family.get("prefer_retest_entry") and family.get("retest_level") is not None:
            retest = float(family["retest_level"])
            if retest > close:
                entry = retest
                order_type = "LIMIT"
                entry_note = (
                    f"Pattern {family.get('pattern_stage', 'TRIGGERED')} — "
                    f"entry routed to neckline retest at {entry:.5f} (do not chase breakout)"
                )
                if sl <= entry:
                    sl = entry + atr * 1.0
        elif brk and brk["strength"] != "confirmed":
            entry = brk["retest_level"]
            order_type = "LIMIT"
            entry_note = (f"Unconfirmed breakout ({brk['strength']}) -- entry routed to retest of the "
                          f"broken rail at {entry:.5f} instead of chasing at market")
            if sl <= entry:
                sl = entry + atr * 1.0
        elif order_type == "MARKET" and atr > 0 and (entry - close) > FAR_ATR_MULTIPLE * atr:
            pullback_entry = _fib_pullback_entry(entry, close, "SELL")
            entry_note = (f"Price extended {(entry - close) / atr:.1f}x ATR beyond the rail -- "
                          f"entry routed to a pullback at {pullback_entry:.5f} instead of chasing")
            entry = pullback_entry
            order_type = "LIMIT"
            if sl <= entry:
                sl = entry + atr * 1.0

        risk = abs(sl - entry)
        if atr > 0 and risk < atr * 0.45:
            sl = entry + atr * 0.80
            risk = abs(sl - entry)
            entry_note = ((entry_note + " " if entry_note else "") +
                          "SL widened to minimum structure distance (0.8×ATR).")
        tp1, tp2, tp1_synthetic = _select_tp_targets(liq, entry, risk, "SELL", MIN_RR)
        if tp1_synthetic:
            entry_note = ((entry_note + " " if entry_note else "") +
                          f"No liquidity level cleared the {MIN_RR:.1f}R minimum for TP1 -- "
                          f"using a fixed {MIN_RR:.1f}R target instead.")
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


def _trendline_structure_sequence(family: Dict[str, Any], limit: int = 4) -> str:
    """Compact HH/HL/LH/LL sequence for the educational report."""
    anns = [a for a in (family.get("trendline_annotations") or [])
            if str(a.get("label")) in {"HH", "HL", "LH", "LL"}]
    labels = [str(a["label"]) for a in anns[-limit:]]
    return " → ".join(labels) if labels else "—"


def _classify_trendline_state(family: Dict[str, Any]) -> Dict[str, str]:
    """Separate continuation, transition and confirmed reversal.

    A trendline rejection is continuation evidence, not a reversal signal.
    A break is only a transition until the underlying swing structure also
    confirms the opposite side.
    """
    tr=family.get("trendline_retest") or {}
    status=str(tr.get("status") or "INTACT")
    kind=str(family.get("family_kind") or "none").lower()
    anns=[a for a in (family.get("trendline_annotations") or []) if a.get("label") in {"HH","HL","LH","LL"}]
    labels=[str(a.get("label")) for a in anns[-6:]]
    if status in ("INTACT","FAKEOUT"):
        if status=="FAKEOUT":
            return {"state":"CONTINUATION","reason":"Trendline was reclaimed; treat the rejection as continuation, not reversal."}
        return {"state":"CONTINUATION","reason":"Trendline respected and structure remains intact; look for continuation entry."}
    # After a break, require a structural opposite extreme before calling reversal.
    if kind=="ascending":
        confirmed = bool(labels and labels[-1]=="LL") or ("LL" in labels[-2:] and "LH" in labels[-3:])
        if confirmed:
            return {"state":"REVERSAL_CONFIRMED","reason":"Ascending trendline break is supported by bearish swing structure (LH/LL)."}
    elif kind=="descending":
        confirmed = bool(labels and labels[-1]=="HH") or ("HH" in labels[-2:] and "HL" in labels[-3:])
        if confirmed:
            return {"state":"REVERSAL_CONFIRMED","reason":"Descending trendline break is supported by bullish swing structure (HL/HH)."}
    return {"state":"TRANSITION","reason":"Trendline is broken, but reversal is not structurally confirmed; wait for retest and structural confirmation."}


def _trendline_status_text(family: Dict[str, Any]) -> Dict[str, str]:
    """Translate the raw lifecycle state into the short trader-facing state."""
    tr = family.get("trendline_retest") or {}
    status = str(tr.get("status") or "INTACT")
    brk = family.get("breakout_grade") or {}
    kind = str(family.get("family_kind") or "none").lower()

    if status == "BREAK_RETEST_CONFIRMED":
        return {
            "breakout": "BROKEN",
            "close": "✅ CONFIRMED",
            "retest": "CONFIRMED",
            "displacement": "CONFIRMED" if brk.get("strength") == "confirmed" else "DEVELOPING",
            "status": status,
        }
    if status in ("BREAK_CONFIRMED", "BREAK_DEVELOPING"):
        return {
            "breakout": "BROKEN",
            "close": "✅ CONFIRMED" if status == "BREAK_CONFIRMED" else "⏳ DEVELOPING",
            "retest": "NOT CONFIRMED",
            "displacement": "CONFIRMED" if brk.get("strength") == "confirmed" else "DEVELOPING",
            "status": status,
        }
    if status == "FAKEOUT":
        return {
            "breakout": "FAKEOUT",
            "close": "⚠️ RECLAIMED",
            "retest": "INVALID",
            "displacement": "FAILED",
            "status": status,
        }

    return {
        "breakout": "NOT BROKEN",
        "close": "❌",
        "retest": "NOT ACTIVE",
        "displacement": "NOT ACTIVE",
        "status": "INTACT",
    }


def format_trendline_report(family: Dict[str, Any], symbol: str) -> str:
    """Clean educational Trendline report.

    The report intentionally separates BIAS, STRUCTURE, BREAKOUT/RETEST and
    DECISION. Raw detector output stays out of the main report so the trader
    can see the actual market story instead of a pile of competing labels.
    """
    if family.get("error"):
        return family["error"]

    topdown = family.get("topdown") or {}
    bias_4h = str(topdown.get("bias_4h") or topdown.get("bias") or "NEUTRAL").upper()
    bias_1h = str(topdown.get("direction") or "NEUTRAL").upper()
    bias_30 = str(family.get("short_term_signal") or family.get("direction") or "NEUTRAL").upper()

    primary_kind = str(family.get("family_kind") or "NONE").upper()
    touches = int(family.get("primary_touches") or 0)
    validation = str(family.get("primary_quality") or "").upper()
    if validation == "CONFIRMED":
        validation_text = "CONFIRMED"
    elif validation == "CROWDED":
        validation_text = "CROWDED"
    else:
        validation_text = "TENTATIVE"

    lifecycle = _trendline_status_text(family)
    structure = _trendline_structure_sequence(family)
    structure_bias = "BULLISH" if bias_30 == "BUY" else "BEARISH" if bias_30 == "SELL" else "NEUTRAL"

    # A setup is not an entry simply because the trendline points in one
    # direction. Entry requires the actual confirmation engine to pass.
    pos = build_position_container(family)
    confirmed_entry = bool(pos and pos.get("confirmed") and not family.get("force_wait_pattern"))
    if lifecycle["status"] == "BREAK_RETEST_CONFIRMED" and pos:
        confirmed_entry = bool(pos.get("confirmed"))

    if confirmed_entry:
        decision_status = "CONFIRMED"
        decision_entry = "CONFIRMED"
    else:
        decision_status = "WAIT"
        decision_entry = "NOT CONFIRMED"

    # Keep confluence short: only the most useful structural facts.
    obs = family.get("order_blocks") or []
    bullish_ob = any(o.get("type") == "bullish" and o.get("freshness") == "untested" for o in obs)
    bearish_ob = any(o.get("type") == "bearish" and o.get("freshness") == "untested" for o in obs)
    mseq = family.get("market_sequence") or {}
    seq = str(mseq.get("sequence") or "").upper()
    seq_bias = str(mseq.get("bias") or "NEUTRAL").upper()
    sp = family.get("scanned_pattern") or {}
    sp_name = str(sp.get("name") or "")
    sp_stage = str(family.get("pattern_stage") or sp.get("stage") or "").upper()

    lines = [
        f"📐 TRENDLINE ANALYSIS — {symbol} M30",
        "",
        "BIAS",
        f"4H: {bias_4h}",
        f"1H: {bias_1h}",
        f"30M: {bias_30}",
        "",
        "HIGHER-TIMEFRAME STRUCTURE",
        f"4H swings: {str(family.get('htf_structure_4h') or 'NEUTRAL')}",
        f"1H swings: {str(family.get('htf_structure_1h') or 'NEUTRAL')}",
    ]
    htf_levels=family.get("htf_key_levels_4h") or []
    if htf_levels:
        for lvl in htf_levels[:3]:
            lines.append(f"4H {str(lvl.get('side','level')).upper()}: {float(lvl.get('price',0)):.5f} ({int(lvl.get('touches',0))} touches)")
    else:
        lines.append("4H key S/R: —")
    lines += [
        "",
        "TRENDLINE",
        f"Type: {primary_kind if primary_kind != 'NONE' else 'NONE'}",
        f"Touches: {touches}",
        f"Validation: {validation_text}",
        f"Status: {'INTACT' if lifecycle['status'] == 'INTACT' else lifecycle['breakout']}",
        "",
        "STRUCTURE",
        f"{structure}",
        f"Structure: {structure_bias}",
        "",
        "BREAKOUT",
        f"Status: {lifecycle['breakout']}",
        f"Close beyond trendline: {lifecycle['close']}",
        "",
        "RETEST",
        f"Status: {lifecycle['retest']}",
        "",
        "CONFLUENCE",
    ]

    if bullish_ob and bias_30 == "BUY":
        lines.append("Bullish OB: ✅")
    elif bearish_ob and bias_30 == "SELL":
        lines.append("Bearish OB: ✅")
    else:
        lines.append("Order block: —")

    if seq:
        icon = "✅" if (seq_bias == structure_bias) else "⚠️"
        lines.append(f"{seq}: {icon} {seq_bias}")
    if sp_name and sp_stage:
        icon = "⚠️" if sp_stage in ("FORMING", "TRIGGERED") else "✅" if sp_stage == "CONFIRMED" else "🚫"
        lines.append(f"{sp_name}: {icon} {sp_stage}")

    lines += [
        "",
        "━━━━━━━━━━━━━━━━",
        "",
        "🎯 DECISION",
        f"BIAS: {bias_30}",
        f"MARKET STATE: {str((family.get('continuation_state') or {}).get('state') if isinstance(family.get('continuation_state'), dict) else family.get('continuation_state') or 'CONTINUATION')}",
        f"STATUS: {decision_status}",
        f"ENTRY: {decision_entry}",
    ]

    if not confirmed_entry:
        waits = []
        state_obj=family.get("continuation_state") or {}
        state_name=state_obj.get("state") if isinstance(state_obj,dict) else str(state_obj)
        if state_name=="CONTINUATION":
            waits.append("continuation entry confirmation from trendline reaction")
        elif state_name=="TRANSITION":
            waits.append("retest + structural confirmation before reversal")
        elif state_name=="REVERSAL_CONFIRMED":
            waits.append("entry confirmation in the new direction")
        if lifecycle["status"] == "INTACT":
            waits.append("trendline confirmation")
        elif lifecycle["status"] in ("BREAK_CONFIRMED", "BREAK_DEVELOPING"):
            waits.append("confirmed retest")
        if lifecycle["status"] == "BREAK_RETEST_CONFIRMED":
            waits.append("candle confirmation")
        entry_rules = family.get("entry_rules") or {}
        if entry_rules and not entry_rules.get("confirmed"):
            waits.append(f"entry confirmation ({entry_rules.get('passed', 0)}/{entry_rules.get('required', 3)})")
        if not waits:
            waits.append("price-action confirmation")

        lines.append("")
        lines.append("WAIT FOR:")
        for i, item in enumerate(dict.fromkeys(waits), 1):
            lines.append(f"{i}. {item.capitalize()}")
        lines.append("")
        lines.append("No trade yet.")
        return "\n".join(lines)

    # Confirmed setup block — only shown after the confirmation gate passes.
    direction = str(pos.get("direction") or bias_30).upper()
    lines += [
        "",
        f"🔥 {direction} CONFIRMED",
        "",
        f"Trendline: {lifecycle['breakout']}",
        f"Retest: {lifecycle['retest']}",
        f"Structure: {structure_bias}",
        f"Displacement: {lifecycle['displacement']}",
        "",
        f"ENTRY: {pos.get('entry'):.5f}",
        f"SL: {pos.get('sl'):.5f}",
        f"TP1: {pos.get('tp1'):.5f}",
        f"TP2: {pos.get('tp2'):.5f}",
        f"R:R: 1:{float(pos.get('rr') or 0):.1f}",
    ]
    return "\n".join(lines)


# ============================================================
# OTE STRATEGY -- structural Fibonacci OTE
# ============================================================
OTE_RATIOS = [0.62, 0.705, 0.79]
OTE_MIN_IMPULSE_ATR = 1.75
OTE_MIN_PIVOT_GAP = 3
OTE_CONFIRM_BODY_RATIO = 0.45

def _ensure_atr(df: pd.DataFrame) -> pd.DataFrame:
    if "ATR" not in df.columns or df["ATR"].isna().all():
        tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift(1)).abs(), (df["Low"]-df["Close"].shift(1)).abs()], axis=1).max(axis=1)
        df=df.copy(); df["ATR"]=tr.rolling(14,min_periods=1).mean()
    return df

def _refined_ote_pivots(df,left=3,right=3,min_leg_atr=0.85):
    raw=find_structural_pivots(df,left=max(3,left),right=max(3,right),min_gap=OTE_MIN_PIVOT_GAP,min_leg_atr=min_leg_atr)
    out=[]; atr=df["ATR"].values
    for p in raw:
        if not out: out.append(p); continue
        q=out[-1]
        if p["index"]-q["index"] < OTE_MIN_PIVOT_GAP:
            if p["type"]==q["type"] and ((p["type"]=="high" and p["price"]>q["price"]) or (p["type"]=="low" and p["price"]<q["price"])): out[-1]=p
            continue
        if p["type"]==q["type"]:
            if (p["type"]=="high" and p["price"]>q["price"]) or (p["type"]=="low" and p["price"]<q["price"]): out[-1]=p
            continue
        a=max(float(atr[min(p["index"],len(atr)-1)]),1e-9)
        if abs(p["price"]-q["price"])>=min_leg_atr*a: out.append(p)
    return out

def _fib_retrace_price(start,end,r,direction):
    leg=abs(end-start)
    return end-leg*r if direction=="BUY" else end+leg*r

def _find_ote_impulse(df,topdown=None,lookback=140):
    if df is None or len(df)<50:return None
    pivots=[p for p in _refined_ote_pivots(df) if p["index"]>=max(0,len(df)-lookback)]
    if len(pivots)<2:return None
    td=(topdown or {}).get("direction","NEUTRAL")
    structure=analyse_structure(df,left=3,right=3,lookback=min(100,len(df)-1))
    sb=structure.get("bias","NEUTRAL"); atrs=df["ATR"].values; candidates=[]
    for i in range(1,len(pivots)):
        a,b=pivots[i-1],pivots[i]
        if a["type"]==b["type"]:continue
        leg=abs(float(b["price"])-float(a["price"])); atr=max(float(atrs[min(b["index"],len(df)-1)]),1e-9); mult=leg/atr
        if mult<OTE_MIN_IMPULSE_ATR:continue
        if a["type"]=="low" and b["type"]=="high" and b["price"]>a["price"]:d="BUY"
        elif a["type"]=="high" and b["type"]=="low" and b["price"]<a["price"]:d="SELL"
        else:continue
        score=mult*10+(18 if td==d else 0)+(15 if sb==d else 0)+b["index"]/len(df)*10
        candidates.append((score,b["index"],{"direction":d,"start":a,"end":b,"leg_size":leg,"atr_multiple":mult,"structure":structure,"pivots":pivots}))
    if not candidates:return None
    candidates.sort(key=lambda x:(x[0],x[1]),reverse=True); imp=candidates[0][2]
    if sb not in ("NEUTRAL",imp["direction"]) and td!=imp["direction"]:return None
    retrace=None
    for q in pivots:
        if q["index"]<=imp["end"]["index"]:continue
        if imp["direction"]=="BUY" and q["type"]=="low" and q["price"]<imp["end"]["price"]:retrace=q;break
        if imp["direction"]=="SELL" and q["type"]=="high" and q["price"]>imp["end"]["price"]:retrace=q;break
    imp["retracement"]=retrace; imp["structure_bias"]=sb; imp["structure_event"]=structure.get("last_event")
    return imp

def _build_ote_zone(impulse):
    start=float(impulse["start"]["price"]);end=float(impulse["end"]["price"]);d=impulse["direction"]
    vals={"62":_fib_retrace_price(start,end,.62,d),"70.5":_fib_retrace_price(start,end,.705,d),"79":_fib_retrace_price(start,end,.79,d)}
    return {**vals,"low":min(vals.values()),"high":max(vals.values()),"direction":d,"origin":start,"extreme":end,"leg_size":abs(end-start)}

def _zone_state(close,zone,atr,impulse=None,df=None):
    """Direction-aware OTE lifecycle.

    ACTIVE means price is currently inside the 62-79% zone. If price already
    entered the zone and then left it in the expected direction, the setup is
    MITIGATED/PASSED rather than being reported as a fresh WAITING setup.
    """
    tol=max(atr*.10,1e-9)
    low=float(zone["low"]); high=float(zone["high"])
    direction=str(zone.get("direction","BUY")).upper()

    if low-tol <= close <= high+tol:
        return "ACTIVE"

    # Detect whether this impulse's OTE zone was already touched by a later
    # candle. This prevents the bot from presenting an old, already-reacted
    # zone as if price were still waiting for its first entry.
    if impulse is not None and df is not None:
        end_i=int(impulse.get("end",{}).get("index",-1))
        if end_i >= 0 and end_i < len(df)-1:
            highs=df["High"].to_numpy(float)
            lows=df["Low"].to_numpy(float)
            later_hi=highs[end_i+1:]
            later_lo=lows[end_i+1:]
            if len(later_hi):
                touched = bool(np.any((later_lo <= high+tol) & (later_hi >= low-tol)))
                if touched:
                    return "MITIGATED / PASSED"

    if direction=="BUY":
        return "TOO_DEEP / INVALID" if close < low-tol else "WAITING"
    return "TOO_DEEP / INVALID" if close > high+tol else "WAITING"

def _find_ote_poi(df,zone,direction):
    try:obs=detect_order_blocks(df,max_per_side=3,min_confidence=45)
    except Exception:obs=[]
    wanted="bullish" if direction=="BUY" else "bearish"; candidates=[]
    for ob in obs:
        if str(ob.get("type","")).lower()!=wanted:continue
        top=float(ob.get("top",0));bottom=float(ob.get("bottom",0))
        if top<=0 or bottom<=0:continue
        overlap=max(0.0,min(top,zone["high"])-max(bottom,zone["low"]))
        dist=0.0 if overlap>0 else min(abs(top-zone["low"]),abs(bottom-zone["high"]))
        candidates.append((dist,-overlap,{"type":wanted,"top":top,"bottom":bottom,"overlap":overlap,"confidence":ob.get("confidence"),"freshness":ob.get("freshness"),"grade":ob.get("grade")}))
    if not candidates:return None
    candidates.sort(key=lambda x:(x[0],x[1]));return candidates[0][2]

def _ote_confirmation(df,direction,zone):
    if len(df)<2:return {"confirmed":False,"label":None,"displacement":False}
    o=float(df["Open"].iloc[-1]);h=float(df["High"].iloc[-1]);l=float(df["Low"].iloc[-1]);c=float(df["Close"].iloc[-1]);atr=max(float(df["ATR"].iloc[-1]),1e-9);rng=max(h-l,1e-9)
    displacement=rng>=atr and abs(c-o)/rng>=OTE_CONFIRM_BODY_RATIO
    near=zone["low"]-.15*atr<=c<=zone["high"]+.15*atr
    confirmed=near and displacement and ((direction=="BUY" and c>o) or (direction=="SELL" and c<o))
    return {"confirmed":bool(confirmed),"label":"Displacement candle" if confirmed else None,"displacement":bool(displacement)}

def _evaluate_ote(df,impulse,topdown=None):
    close=float(df["Close"].iloc[-1]);atr=max(float(df["ATR"].iloc[-1]),1e-9);d=impulse["direction"];zone=_build_ote_zone(impulse);state=_zone_state(close,zone,atr,impulse,df);poi=_find_ote_poi(df,zone,d);confirm=_ote_confirmation(df,d,zone);reasons=[];score=40
    if impulse["atr_multiple"]>=2:score+=15;reasons.append(f"Valid displacement leg ({impulse['atr_multiple']:.1f} ATR)")
    else:score+=8;reasons.append(f"Moderate impulse ({impulse['atr_multiple']:.1f} ATR)")
    td=(topdown or {}).get("direction","NEUTRAL")
    if td==d:score+=15;reasons.append(f"HTF bias aligned ({td})")
    elif td in ("BUY","SELL"):score-=12;reasons.append(f"HTF bias conflicts ({td})")
    if impulse.get("structure_event") in ("BOS","MSS","CHoCH") and impulse.get("structure_bias")==d:score+=12;reasons.append(f"Structure confirmation: {impulse['structure_event']} ({d})")
    elif impulse.get("structure_bias")==d:score+=7;reasons.append("Directional structure aligned")
    else:reasons.append("Structural confirmation pending")
    if state=="ACTIVE":score+=18;reasons.append("Price is inside the 62%-79% OTE zone")
    elif state=="WAITING":reasons.append("Price has not reached the OTE zone")
    elif state=="MITIGATED / PASSED":
        score-=5
        reasons.append("OTE zone was already touched after the impulse; waiting for a new setup")
    else:score-=15;reasons.append("Retracement has exceeded the 79% boundary")
    if poi and poi.get("overlap",0)>0:score+=10;reasons.append("Directional OB overlaps OTE")
    elif poi:reasons.append("Directional OB is nearby, outside OTE")
    else:reasons.append("No qualifying directional OB inside/near OTE")
    if confirm["confirmed"]:score+=15;reasons.append("Displacement candle confirms the OTE reaction")
    else:reasons.append("Entry confirmation not yet present")
    score=int(max(0,min(100,score)))
    entry=close;sl=impulse["start"]["price"]-atr*.20 if d=="BUY" else impulse["start"]["price"]+atr*.20;tp1=impulse["end"]["price"];tp2=impulse["end"]["price"]+impulse["leg_size"]*.618 if d=="BUY" else impulse["end"]["price"]-impulse["leg_size"]*.618
    risk=abs(entry-sl);rr=abs(tp1-entry)/risk if risk>0 else 0
    valid=d in ("BUY","SELL") and state=="ACTIVE" and confirm["confirmed"] and td in ("NEUTRAL",d) and rr>=MIN_RR
    ticket={"side":"LONG" if d=="BUY" else "SHORT","direction":d,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":round(rr,2),"risk":risk,"reward":abs(tp1-entry),"order_type":"MARKET"} if valid else None
    return {"zone":zone,"zone_state":state,"poi":poi,"confirmation":confirm,"score":score,"reasons":reasons,"valid":valid,"status":"CONFIRMED" if valid else state,"ticket":ticket,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":rr}

def run_ote_analysis(symbol: str, df: pd.DataFrame = None) -> Dict[str, Any]:
    topdown=get_topdown_bias(symbol);df=market_data.fetch_candles(symbol,"30min",count=240) if df is None else df
    if df is None or df.empty or len(df)<60:return {"error":"Insufficient 30M data for OTE analysis","direction":"NEUTRAL","score":0,"valid":False,"symbol":symbol,"topdown":topdown}
    df=_ensure_atr(df);impulse=_find_ote_impulse(df,topdown)
    if impulse is None:return {"error":"No valid structural impulse for OTE (waiting for meaningful BOS/displacement leg).","direction":"NEUTRAL","score":0,"valid":False,"df":df,"timeframe":"30min","symbol":symbol,"topdown":topdown}
    ev=_evaluate_ote(df,impulse,topdown)
    return {"strategy":"OTE","direction":impulse["direction"],"score":ev["score"],"reasons":ev["reasons"],"valid":ev["valid"],"status":ev["status"],"impulse":impulse,"zone":ev["zone"],"zone_state":ev["zone_state"],"poi":ev["poi"],"confirmation":ev["confirmation"],"position":ev["ticket"],"ticket":ev["ticket"],"entry":ev["entry"],"sl":ev["sl"],"tp1":ev["tp1"],"tp2":ev["tp2"],"rr":ev["rr"],"df":df,"timeframe":"30min","symbol":symbol,"topdown":topdown}

def format_ote_report(analysis: Dict[str, Any]) -> str:
    symbol=analysis.get("symbol","")
    if analysis.get("error"):
        return f"🎯 OTE ANALYSIS — {symbol} M30\n\n{analysis['error']}"
    imp=analysis.get("impulse") or {};z=analysis.get("zone") or {};poi=analysis.get("poi");conf=analysis.get("confirmation") or {};td=analysis.get("topdown") or {};d=analysis.get("direction","NEUTRAL");valid=analysis.get("valid",False);state=analysis.get("status","WAIT")
    td_dir=td.get("direction","NEUTRAL")
    bias4=str(td.get("bias_4h") or td_dir).upper(); s4=str(td.get("swing_context_4h",{}).get("structure_bias") or "NEUTRAL").upper(); s1=str(td.get("swing_context_1h",{}).get("structure_bias") or "NEUTRAL").upper()
    lines=[f"🎯 OTE ANALYSIS — {symbol} M30","","BIAS",f"4H: {bias4}",f"1H: {td_dir}",f"30M: {d}","","HIGHER-TIMEFRAME STRUCTURE",f"4H swings: {s4}",f"1H swings: {s1}"]
    for lvl in (td.get("key_levels_4h") or [])[:3]:
        lines.append(f"4H {str(lvl.get('side','level')).upper()}: {float(lvl.get('price',0)):.5f} ({int(lvl.get('touches',0))} touches)")
    lines += ["","IMPULSE",f"{imp.get('start',{}).get('type','?').upper()} → {imp.get('end',{}).get('type','?').upper()}",f"Leg: {imp.get('leg_size',0):.5f} ({imp.get('atr_multiple',0):.1f} ATR)",f"Structure: {imp.get('structure_bias','NEUTRAL')}",f"Event: {imp.get('structure_event') or 'PENDING'}","","OTE ZONE",f"62%: {z.get('62',0):.5f}",f"70.5%: {z.get('70.5',0):.5f}",f"79%: {z.get('79',0):.5f}",f"Status: {analysis.get('zone_state','WAITING')}","","CONFLUENCE",f"Directional OB: {'✅' if poi else '❌'}",f"OB overlaps OTE: {'✅' if poi and poi.get('overlap',0)>0 else '❌'}",f"Displacement: {'✅' if conf.get('confirmed') else '❌'}","","━━━━━━━━━━━━━━━━","","🎯 DECISION",f"BIAS: {d}",f"STATUS: {'CONFIRMED' if valid else ('ACTIVE — WAIT' if state=='ACTIVE' else 'WAIT')}",f"ENTRY: {'CONFIRMED' if valid else 'NOT CONFIRMED'}"]
    if not valid:
        wait1 = {
            "ACTIVE": "1. OTE reaction",
            "MITIGATED / PASSED": "1. A new valid impulse and fresh OTE zone",
            "TOO_DEEP / INVALID": "1. A new valid impulse (current OTE invalidated)",
        }.get(analysis.get("zone_state"), "1. Price to enter the 62–79% OTE zone")
        lines += ["","WAIT FOR:",wait1,"2. Directional POI reaction/alignment","3. Displacement confirmation","4. Structure to remain valid","","No trade yet."]
    else:
        t=analysis.get("ticket") or {};lines += ["","🔥 OTE CONFIRMED","",f"Direction: {d}","Structure: CONFIRMED","Displacement: CONFIRMED",f"Entry: {t.get('entry',0):.5f}",f"SL: {t.get('sl',0):.5f}",f"TP1: {t.get('tp1',0):.5f}",f"TP2: {t.get('tp2',0):.5f}",f"R:R: 1:{t.get('rr',0):.2f}"]
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

    family = build_trendline_family(df_30m, max_lines=4, lookback_bars=120)
    family["symbol"] = symbol
    family["timeframe"] = "30min"
    family["topdown"] = topdown
    family["htf_key_levels_4h"] = topdown.get("key_levels_4h") or []
    family["htf_swings_4h"] = topdown.get("swings_4h") or []
    family["htf_swings_1h"] = topdown.get("swings_1h") or []
    family["htf_structure_4h"] = (topdown.get("swing_context_4h") or {}).get("structure_bias", "NEUTRAL")
    family["htf_structure_1h"] = (topdown.get("swing_context_1h") or {}).get("structure_bias", "NEUTRAL")
    family["continuation_state"] = _classify_trendline_state(family)
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
    mw = family.get("mw_pattern")
    reasons = list(family.get("reasons") or [])

    # ------------------------------------------------------------------
    # Pattern conflict resolution (v2)
    # Priority of truth:
    #   1. Clear trendline geometry (already set)
    #   2. Strong continuation (Bull/Bear Flag, channels) — preferred over forced reversals
    #   3. Strong classic reversal (Inverse H&S etc.)
    #   4. Tightened M/W detector
    #   5. Order-block reaction
    # Also demotes Double Top (M) when ascending structure or Bull Flag is present.
    # ------------------------------------------------------------------
    sp = family.get("scanned_pattern")
    close = float(df_30m["Close"].iloc[-1])
    family_kind = family.get("family_kind", "none")

    # ------------------------------------------------------------------
    # Neckline stage gate (FORMING / TRIGGERED / CONFIRMED / FAKEOUT)
    # Never treat an unbroken pattern as a live entry. Liquidity grabs
    # that spike the neckline and reclaim are marked FAKEOUT.
    # ------------------------------------------------------------------
    reversal_names = {
        "Double Top", "Double Bottom", "Triple Top", "Triple Bottom",
        "Head and Shoulders", "Inverse Head and Shoulders",
        "Double Top (M)", "Double Bottom (W)",
    }
    if sp and sp.get("name") in reversal_names:
        stage = str(sp.get("stage") or "FORMING").upper()
        stage_note = sp.get("stage_note") or ""
        reasons.append(f"Pattern stage: {stage} — {stage_note}")
        family["pattern_stage"] = stage
        if stage == "FORMING":
            # Shape only — force WAIT, do not let pattern override to ACTIVE
            gating_notes.append("⏳ Pattern FORMING — neckline not broken by close. No entry yet.")
            # Soften strength so entry_rules alone cannot flip to confirmed trade
            strength = min(strength, 55)
            family["force_wait_pattern"] = True
        elif stage == "FAKEOUT":
            gating_notes.append("🚫 Pattern FAKEOUT — neckline reclaimed (liquidity grab). Invalidated.")
            strength = max(0, strength - 25)
            family["force_wait_pattern"] = True
            # Do not let a fakeout pattern set direction
            if family.get("active_pattern") == "scanned":
                family["active_pattern"] = "none"
        elif stage == "TRIGGERED":
            gating_notes.append(
                f"⚡ Pattern TRIGGERED — real neckline break. Prefer retest entry near "
                f"{sp.get('retest_level') or sp.get('trigger_price')}"
            )
            family["prefer_retest_entry"] = True
            family["retest_level"] = sp.get("retest_level") or sp.get("trigger_price")
        elif stage == "CONFIRMED":
            gating_notes.append("✅ Pattern CONFIRMED — break + retest held. Highest quality trigger.")
            strength = min(100, strength + 8)
            family["prefer_retest_entry"] = True
            family["retest_level"] = sp.get("retest_level") or sp.get("trigger_price")

    # Demote Double Top (M) when structure is clearly bullish continuation
    if mw and mw.get("pattern") == "M":
        has_bull_cont = False
        for p in (family.get("scanned_patterns") or []):
            if p.get("name") in ("Bull Flag", "Bullish Pennant", "Ascending Triangle", "Ascending Channel") and float(p.get("confidence") or 0) >= 68:
                has_bull_cont = True
                break
        if has_bull_cont or (family_kind == "ascending" and direction == "BUY"):
            reasons.append(
                "⚠ Demoted Double Top (M) — bullish continuation / ascending structure is cleaner"
            )
            family["mw_pattern"] = None
            mw = None
            if has_bull_cont and sp and sp.get("name") in ("Bull Flag", "Bullish Pennant", "Ascending Triangle"):
                family["active_pattern"] = "scanned"
                family["pattern_confidence"] = int(sp.get("confidence") or 70)
            elif family_kind == "ascending":
                family["active_pattern"] = "channel" if family.get("channel") else "none"

    if sp:
        sp_name = sp.get("name", "")
        sp_bias = sp.get("bias", "NEUTRAL")
        sp_conf = float(sp.get("confidence") or 0)
        is_strong_bullish_rev = (
            sp_name in ("Inverse Head and Shoulders", "Double Bottom", "Triple Bottom")
            and sp_bias == "BUY" and sp_conf >= 72
        )
        is_strong_bearish_rev = (
            sp_name in ("Head and Shoulders", "Double Top", "Triple Top")
            and sp_bias == "SELL" and sp_conf >= 72
        )
        is_bullish_cont = (
            sp_name in ("Bull Flag", "Bullish Pennant", "Ascending Triangle", "Ascending Channel")
            and sp_conf >= 68
        )
        is_bearish_cont = (
            sp_name in ("Bear Flag", "Bearish Pennant", "Descending Triangle", "Descending Channel")
            and sp_conf >= 68
        )

        # Continuation patterns own the bias when they are clear
        if is_bullish_cont and direction in ("SELL", "NEUTRAL"):
            direction = "BUY"
            strength = max(strength, int(sp_conf) - 2)
            reasons.append(f"✅ {sp_name} ({sp_conf:.0f}%) — bullish continuation preferred")
            family["active_pattern"] = "scanned"
            family["pattern_confidence"] = int(sp_conf)
            if mw and mw.get("pattern") == "M":
                family["mw_pattern"] = None
                mw = None
        elif is_bearish_cont and direction in ("BUY", "NEUTRAL"):
            direction = "SELL"
            strength = max(strength, int(sp_conf) - 2)
            reasons.append(f"✅ {sp_name} ({sp_conf:.0f}%) — bearish continuation preferred")
            family["active_pattern"] = "scanned"
            family["pattern_confidence"] = int(sp_conf)
            if mw and mw.get("pattern") == "W":
                family["mw_pattern"] = None
                mw = None

        # Strong Inverse H&S / Double Bottom vs weak M or NEUTRAL
        elif is_strong_bullish_rev and direction in ("SELL", "NEUTRAL"):
            head_price = None
            for kp in (sp.get("key_points") or []):
                if len(kp) >= 3 and "Head" in str(kp[2]):
                    head_price = float(kp[1])
                    break
            head_broken = head_price is None or close > head_price * 1.001
            bullish_ob_support = False
            for ob in (family.get("order_blocks") or []):
                if ob.get("type") == "bullish":
                    ob_top = float(ob.get("top", 0))
                    ob_bot = float(ob.get("bottom", 0))
                    if ob_bot <= close <= ob_top * 1.015 or abs(close - ob_top) / max(close, 1e-9) < 0.005:
                        bullish_ob_support = True
                        break
            if head_broken or bullish_ob_support or sp_conf >= 70:
                old_dir = direction
                direction = "BUY"
                strength = max(strength, int(sp_conf) - 4)
                if mw and mw.get("pattern") == "M":
                    reasons.append(
                        f"⚠ Overrode weak Double Top (M) — {sp_name} ({sp_conf:.0f}%) cleaner "
                        f"(head broken / bullish OB). Bias {old_dir} → BUY"
                    )
                    family["mw_pattern"] = None
                else:
                    reasons.append(f"✅ {sp_name} ({sp_conf:.0f}%) confirms bullish reversal")
                family["active_pattern"] = "scanned"
                family["pattern_confidence"] = max(int(family.get("pattern_confidence") or 0), int(sp_conf))

        # Strong H&S / Double Top vs weak W or NEUTRAL
        elif is_strong_bearish_rev and direction in ("BUY", "NEUTRAL"):
            head_price = None
            for kp in (sp.get("key_points") or []):
                if len(kp) >= 3 and "Head" in str(kp[2]):
                    head_price = float(kp[1])
                    break
            head_broken = head_price is None or close < head_price * 0.999
            if head_broken or sp_conf >= 70:
                old_dir = direction
                direction = "SELL"
                strength = max(strength, int(sp_conf) - 4)
                if mw and mw.get("pattern") == "W":
                    reasons.append(
                        f"⚠ Overrode weak Double Bottom (W) — {sp_name} ({sp_conf:.0f}%) cleaner. "
                        f"Bias {old_dir} → SELL"
                    )
                    family["mw_pattern"] = None
                else:
                    reasons.append(f"✅ {sp_name} ({sp_conf:.0f}%) confirms bearish reversal")
                family["active_pattern"] = "scanned"
                family["pattern_confidence"] = max(int(family.get("pattern_confidence") or 0), int(sp_conf))

        elif sp_bias == direction and sp_conf >= 65:
            strength = min(100, strength + 8)
            reasons.append(f"✅ {sp_name} ({sp_conf:.0f}%) aligns with current bias — conviction +")

    family["direction"] = direction
    # Re-evaluate lifecycle state after all pattern/structure processing.
    family["continuation_state"] = _classify_trendline_state(family)
    
    family["strength"] = max(0, min(100, int(strength)))
    family["reasons"] = reasons

    # SHORT-TERM TRENDLINE IS PRIMARY.
    # We adapt to the structure the trendlines show instead of fighting it
    # with lagging higher-timeframe direction. 4H/1H is now advisory only.
    if direction in ("BUY", "SELL"):
        if td_dir == direction and topdown.get("allowed"):
            strength = min(100, strength + 12)
            gating_notes.append(
                f"✅ Short-term trend ({direction}) aligned with 4H/1H top-down ({td_dir})"
            )
        elif td_dir == direction and not topdown.get("allowed"):
            gating_notes.append(
                f"Short-term trend ({direction}) matches top-down direction but 1H permission "
                f"not yet granted — still valid, slightly lower conviction"
            )
        elif td_dir == "NEUTRAL":
            gating_notes.append(
                f"Short-term Trend: {direction} (trendline structure) — higher TF neutral"
            )
        else:
            # Conflict: keep the short-term signal, only mild confidence reduction
            strength = max(0, strength - 8)
            gating_notes.append(
                f"Short-term Trend: {direction} (from trendline) — higher TF still {td_dir} "
                f"(advisory only, not blocking)"
            )

    family["strength"] = max(0, min(100, int(strength)))
    family["gating_notes"] = gating_notes
    family["short_term_signal"] = direction  # explicit short-term read for reports
    return family
