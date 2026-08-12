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


def _select_best_line(pivots: List[Dict], kind: str, n: int, df: pd.DataFrame,
                       min_span: int = 12, recency_limit: int = 55) -> Optional[Dict]:
    """
    Score EVERY valid pair of same-type pivots (not just chain-adjacent
    ones) and keep the single best-fitting line.

    This replaces the old "sequential chain that restarts on any minor
    violation" approach. That approach almost always collapsed to the two
    most recent, tiniest pivots on real (noisy) price action, which then
    failed the min-move filter and produced NO line at all -- the root
    cause of trendlines never appearing.

    A trader draws a trendline by picking the two (or more) swing points
    that best describe the whole visible structure -- exactly what this
    scoring does: reward long span + high touch count + low violations,
    same as the reference chart (one clean diagonal spanning most of the
    visible range, not a 10-bar sliver).
    """
    want = "low" if kind == "support" else "high"
    pts = sorted([p for p in pivots if p["type"] == want], key=lambda p: p["index"])
    if len(pts) < 2:
        return None

    best = None
    best_score = -1e9
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            a, b = pts[i], pts[j]
            span = b["index"] - a["index"]
            if span < min_span:
                continue
            if b["index"] < n - recency_limit:
                continue  # the line must still be relevant to current price
            if kind == "support" and b["price"] <= a["price"]:
                continue
            if kind == "resistance" and b["price"] >= a["price"]:
                continue

            touches = _count_touches(df, a["index"], a["price"], b["index"], b["price"], kind, tol_atr=0.45)
            violations = _count_violations(df, a["index"], a["price"], b["index"], b["price"], kind, tol_atr=0.32)
            if touches < 2:
                continue
            if violations > touches:
                continue

            span_score = min(span / 55.0, 1.6)          # reward structural length
            recency_score = 1.0 - (n - 1 - b["index"]) / max(recency_limit, 1)
            score = touches * 2.0 - violations * 3.2 + span_score * 3.5 + recency_score * 1.2

            if score > best_score:
                best_score = score
                best = (a, b, touches, violations)

    if not best:
        return None
    a, b, touches, violations = best
    slope = (b["price"] - a["price"]) / max(b["index"] - a["index"], 1)
    y_end = _line_value(a["index"], a["price"], b["index"], b["price"], n - 1)
    quality = "unconfirmed" if touches < 3 else "confirmed"

    return {
        "x0": a["index"], "y0": a["price"],
        "x1": b["index"], "y1": b["price"],
        "y_end": y_end,
        "slope": slope,
        "touches": touches,
        "violations": violations,
        "confirmed": quality == "confirmed",
        "quality": quality,
        "kind": kind,
        "bars_since_last_touch": n - 1 - b["index"],
        "method": "structural_best_fit",
    }


def _fit_primary(pivots: List[Dict], kind: str, n: int, df: pd.DataFrame) -> Optional[Dict]:
    """
    Classic dynamic trendline (matches the educational image style).
    Picks the best-scoring structural pivot pair (see _select_best_line)
    instead of always chasing the most recent 2 pivots.
    """
    line = _select_best_line(pivots, kind, n, df)
    if not line:
        return None

    # Soft sanity check only -- no longer a hard kill. A structurally
    # well-touched, long-span line is real even if the raw price delta
    # between its two anchor points is modest (this used to reject good
    # lines that just happened to have a shallow-looking start/end pair).
    atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else None
    if atr_now and atr_now > 0:
        total_move = abs(line["y1"] - line["y0"])
        if total_move < 0.35 * atr_now and line["touches"] < 3:
            return None

    return line


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


def _label_hh_structure(pivots: List[Dict], df: pd.DataFrame, n: int, max_labels: int = 4,
                         exclude_near: Optional[List[int]] = None) -> List[Dict]:
    """
    Label the swing-high sequence as HH (Higher High) / 'HH Failed' the
    way the reference chart does: consecutive higher highs get 'HH', and
    when a high comes in roughly equal to (fails to clear) the prior HH,
    it's tagged 'HH Failed' -- the classic early warning that a trend is
    losing momentum right before a trendline break.

    exclude_near: indices already claimed by a more specific label (e.g.
    the M/W pattern's Top/Bottom markers) -- skip those pivots so we don't
    stamp a duplicate/colliding label on the exact same point.
    """
    highs = sorted([p for p in pivots if p["type"] == "high"], key=lambda p: p["index"])
    if len(highs) < 2:
        return []
    exclude_near = exclude_near or []

    def _too_close(idx):
        return any(abs(idx - e) <= 3 for e in exclude_near)

    highs = [p for p in highs if not _too_close(p["index"])]
    if len(highs) < 2:
        return []
    recent = highs[-max_labels:]
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else None

    labels = []
    for idx, p in enumerate(recent):
        if idx == 0:
            labels.append({"index": p["index"], "price": p["price"], "label": "HH"})
            continue
        prev = recent[idx - 1]
        tol = atr * 0.3 if atr else abs(p["price"]) * 0.0015
        if p["price"] > prev["price"] + tol:
            labels.append({"index": p["index"], "price": p["price"], "label": "HH"})
        elif abs(p["price"] - prev["price"]) <= tol:
            labels.append({"index": p["index"], "price": p["price"], "label": "HH Failed",
                            "pair_index": prev["index"], "pair_price": prev["price"]})
        # a clear lower high isn't labeled -- it's no longer part of the HH story
    return labels


def _detect_breakout_and_retest(df: pd.DataFrame, line: Dict, kind: str, n: int) -> Optional[Dict]:
    """
    Find the breakout candle (confirmed close through the line) and, if
    it happened, the first retest afterward -- matching the two callouts
    in the reference image ('Trendline Breakout' / 'Trendline Retest').

    kind: 'support' looks for a downside break, 'resistance' an upside break.
    """
    if df is None or n < 5:
        return None
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    atr_col = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).values
    x0, y0, x1, y1 = line["x0"], line["y0"], line["x1"], line["y1"]

    break_idx = None
    for i in range(int(x1) + 1, n):
        lv = _line_value(x0, y0, x1, y1, i)
        atr = float(atr_col[i]) if atr_col[i] and atr_col[i] > 0 else abs(y1 - y0) * 0.05 or 1e-9
        if kind == "support" and close[i] < lv - 0.15 * atr:
            break_idx = i
            break
        if kind == "resistance" and close[i] > lv + 0.15 * atr:
            break_idx = i
            break
    if break_idx is None:
        return None

    retest_idx = None
    for i in range(break_idx + 1, n):
        lv = _line_value(x0, y0, x1, y1, i)
        atr = float(atr_col[i]) if atr_col[i] and atr_col[i] > 0 else abs(y1 - y0) * 0.05 or 1e-9
        tol = 0.30 * atr
        if kind == "support" and high[i] >= lv - tol:
            retest_idx = i
            break
        if kind == "resistance" and low[i] <= lv + tol:
            retest_idx = i
            break

    result = {
        "breakout_index": break_idx,
        "breakout_price": float(close[break_idx]),
        "retest_index": retest_idx,
    }
    if retest_idx is not None:
        result["retest_price"] = float(high[retest_idx] if kind == "support" else low[retest_idx])
    return result


def build_trendline_family(df: pd.DataFrame, max_lines: int = 4, lookback_bars: int = 60) -> Dict[str, Any]:
    """
    Build one clean parallel family (ascending OR descending), not both mixed.
    Market reveals direction: price relative to the family rails.

    Uses simple fractal pivots (NO ZigZag). The primary trendline is always
    the two most recent sequential higher-lows (support) or lower-highs
    (resistance). This produces responsive lines that hug current structure
    the way a trader draws them by hand.
    """
    if df is None or len(df) < 30:
        return {"error": "Insufficient data for trendline family", "direction": "NEUTRAL", "pivots": []}

    n = len(df)

    # --- Simple fractal pivots (NO ZigZag) ---
    pivots = find_fractal_pivots(df, left=3, right=3)
    if len(pivots) < 6:
        pivots = find_fractal_pivots(df, left=2, right=2)

    # Recent window for diagonal fit (tighter for better hugging)
    cutoff = max(0, n - lookback_bars)
    recent_pivots = [p for p in pivots if p["index"] >= cutoff]
    if len(recent_pivots) < 4:
        recent_pivots = [p for p in pivots if p["index"] >= max(0, n - int(lookback_bars * 1.5))]
    if len(recent_pivots) < 3:
        recent_pivots = pivots

    support = _fit_primary(recent_pivots, "support", n, df)
    resistance = _fit_primary(recent_pivots, "resistance", n, df)

    # Reject a candidate diagonal line only when it's BOTH shallow AND
    # poorly touched -- e.g. two swing lows that are technically "rising"
    # by a few points with barely any structure behind them. That's a
    # range, not a trend, and belongs to horizontal S/R clustering below.
    #
    # NOTE: this used to reject on shallow endpoint-to-endpoint move alone
    # (>= 1.0 ATR required), which threw out well-touched, long-span,
    # low-violation structural lines just because their two anchor points
    # happened to be close in price (very common for a wide, gently-sloped
    # rail with 20-40 touches). _select_best_line already validates
    # structure via touches/violations/span, so this is now a safety net
    # only, not the primary quality gate.
    MIN_TREND_MOVE_ATR = 0.35
    atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else None

    def _has_meaningful_slope(line):
        if not line:
            return False
        if line.get("touches", 0) >= 4:
            return True  # well-touched structural line -- trust it regardless of raw delta
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

    # HH / HH Failed structure labels (educational-image style) and the
    # breakout + retest callouts for whichever primary line is active.
    # Exclude pivots already claimed by the M/W pattern's Top/Bottom
    # markers so we don't stamp a colliding duplicate label on them.
    exclude_near = []
    if mw:
        for key in ("left", "right"):
            pt = mw.get(key)
            if pt:
                exclude_near.append(int(pt["index"]))
    hh_labels = _label_hh_structure(recent_pivots, df, n, exclude_near=exclude_near)
    breakout_retest = None
    if primary:
        primary_kind_for_break = "support" if family_kind == "ascending" else "resistance"
        breakout_retest = _detect_breakout_and_retest(df, primary, primary_kind_for_break, n)

    return {
        "direction": direction,
        "strength": max(0, min(100, int(strength))),
        "reasons": reasons,
        "entry_rules": entry_rules,
        "family_kind": family_kind,
        "family_lines": family_lines,  # the clean parallel set
        "hh_labels": hh_labels,
        "breakout_retest": breakout_retest,
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


def format_trendline_report(family: Dict[str, Any], symbol: str) -> str:
    if family.get("error"):
        return family["error"]
    quality = family.get("primary_quality")
    quality_tag = {
        "unconfirmed": "⚠️ UNCONFIRMED (2 touches)",
        "confirmed": "✅ CONFIRMED",
        "crowded": "⚠️ CROWDED (5+ touches)",
    }.get(quality, "")
    short_sig = family.get("short_term_signal") or family.get("direction", "NEUTRAL")
    lines = [
        f"📐 TRENDLINE  |  {symbol}  (Short-term structure primary)",
        f"Short-term Trend: {short_sig}",
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

    mseq = family.get("market_sequence")
    if mseq:
        lines.append(
            f"Market sequence: {mseq['sequence']} ({mseq['bias']}, {mseq['confidence']:.0f}%) — {mseq['note']}"
        )

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
        # build_position_container's dict key is "direction" ("BUY"/"SELL"),
        # never "side" -- that mismatch was throwing KeyError: 'side' on
        # every setup that actually reached this point (i.e. most pairs).
        side = pos.get("direction", "?")
        if pos.get("too_extended") or pos.get("entry") is None:
            # Entry/SL/TP are intentionally None here (trend too extended --
            # see build_position_container). Formatting them with :.5f
            # would crash the same way; show the wait note instead.
            lines.append(f"Position: {side}  — no entry (too extended, wait for pullback)")
        else:
            lines.append(
                f"Position: {side}  Entry {pos['entry']:.5f}  SL {pos['sl']:.5f}  "
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

    # --- gate against the 4H/1H top-down bias (advisory only) ---
    # Short-term impulse / trendline structure is PRIMARY. Higher TF no longer blocks.
    td_dir = (topdown or {}).get("direction", "NEUTRAL")
    td_allowed = bool((topdown or {}).get("allowed"))
    if td_dir in ("BUY", "SELL"):
        if td_dir == direction and td_allowed:
            score += 12
            reasons.append(f"✅ Short-term impulse ({direction}) aligned with 4H/1H top-down ({td_dir})")
        elif td_dir == direction and not td_allowed:
            reasons.append(f"Short-term impulse ({direction}) matches top-down but 1H permission pending")
        else:
            score -= 8
            reasons.append(
                f"Short-term impulse: {direction} — higher TF still {td_dir} (advisory only, not blocking)"
            )
    else:
        reasons.append(f"Short-term impulse: {direction} — higher TF neutral")

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
    # No longer invalidate just because higher TF disagrees. RR floor
    # matches MIN_RR used by the Trendline strategy's position container --
    # a signal below this is not worth the risk regardless of how clean
    # the Fan/Expansion read looks.
    valid = direction in ("BUY", "SELL") and score >= 58 and in_zone and rr >= MIN_RR

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
    valid_tag = "VALID (checked)" if valid else "WAIT (pending)"
    lines.append(
   
