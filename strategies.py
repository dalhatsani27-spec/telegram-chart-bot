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
    is_price_ranging_vs_sma,
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


# ============================================================
# 20 SMA (median price) -- direction + confluence with the primary
# trendline. This is what lets the Trendline chart color itself
# green while price/SMA agree bullish, and flip red the moment
# price breaks below the SMA and starts forming a bear trend
# (rather than the diagonal line always being drawn the same
# fixed color regardless of state).
# ============================================================

def _sma20_series(df: pd.DataFrame, period: int = 20, applied_price: str = "median") -> pd.Series:
    if applied_price == "median":
        price = (df["High"] + df["Low"]) / 2.0
    elif applied_price == "close":
        price = df["Close"]
    else:
        price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    return price.rolling(period, min_periods=max(2, period // 2)).mean()


def _sma_direction(sma: pd.Series, slope_lookback: int = 5, atr_now: Optional[float] = None) -> str:
    valid = sma.dropna()
    if len(valid) < slope_lookback + 1:
        return "FLAT"
    now = float(valid.iloc[-1])
    prev = float(valid.iloc[-1 - slope_lookback])
    threshold = (atr_now * 0.08) if atr_now else abs(now) * 0.0004
    if now - prev > threshold:
        return "RISING"
    if prev - now > threshold:
        return "FALLING"
    return "FLAT"


def _sma_trendline_confluence(sma_now: Optional[float], trendline_now: Optional[float],
                               atr_now: Optional[float], sma_dir: str, family_kind: str) -> Dict[str, Any]:
    if sma_now is None or trendline_now is None:
        return {"relationship": "N/A", "distance_atr": None, "status": "NOT AVAILABLE", "strength": "N/A"}

    distance = abs(sma_now - trendline_now)
    distance_atr = round(distance / atr_now, 2) if atr_now else None

    aligned = (sma_dir == "RISING" and family_kind == "ascending") or \
              (sma_dir == "FALLING" and family_kind == "descending")
    conflicting = (sma_dir == "RISING" and family_kind == "descending") or \
                  (sma_dir == "FALLING" and family_kind == "ascending")
    relationship = "ALIGNED" if aligned else ("CONFLICTING" if conflicting else "NEUTRAL")

    if distance_atr is None:
        status = "UNKNOWN"
    elif distance_atr <= 0.10:
        status = "TOUCHING"
    elif distance_atr <= 0.5:
        status = "NEAR"
    else:
        status = "FAR"

    if aligned and status in ("TOUCHING", "NEAR"):
        strength = "STRONG"
    elif aligned:
        strength = "MODERATE"
    elif conflicting:
        strength = "WEAK"
    else:
        strength = "MODERATE"

    return {"relationship": relationship, "distance_atr": distance_atr, "status": status, "strength": strength}


def _trendline_color_state(direction: str, sma_dir: str) -> str:
    """
    Dynamic trend-state color for the drawn trendline (chart_engine reads
    this): GREEN while price/SMA support a bullish read, RED the moment
    price/SMA confirm bearish, WHITE/grey when neutral or conflicting.
    """
    if direction == "BUY" and sma_dir != "FALLING":
        return "BULLISH"
    if direction == "SELL" and sma_dir != "RISING":
        return "BEARISH"
    if direction == "NEUTRAL":
        if sma_dir == "RISING":
            return "BULLISH"
        if sma_dir == "FALLING":
            return "BEARISH"
    return "NEUTRAL"


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


def _find_sma_reclaim_anchor(pivots: List[Dict], sma: Optional[pd.Series], df: pd.DataFrame,
                              kind: str, lookahead: int = 25) -> Optional[Dict]:
    """
    Find the swing low/high that actually STARTED the current trend leg:
    the pivot low that price broke away from and then closed back above
    the 20 SMA (uptrend), or the pivot high it closed back below (downtrend).

    Anchoring the trendline here instead of on an arbitrary older swing
    gives a clean line that begins exactly where the SMA (a lagging
    indicator) finally confirmed the move -- and it means a close back
    through that SAME trendline later is a genuine reason to start
    hunting for a bias flip, not just noise.

    Scans from the most recent pivot backward so we land on the anchor
    that defines the CURRENT leg, not a stale one from an earlier cycle.
    """
    if sma is None or df is None or sma.dropna().empty:
        return None
    want_type = "low" if kind == "support" else "high"
    candidates = sorted([p for p in pivots if p["type"] == want_type], key=lambda p: p["index"])
    if not candidates:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(float)
    sma_vals = sma.to_numpy(float)
    n = len(close)

    for p in reversed(candidates):
        idx = p["index"]
        end = min(idx + lookahead, n - 1)
        for j in range(idx + 1, end + 1):
            if not (np.isfinite(close[j]) and np.isfinite(sma_vals[j])):
                continue
            if kind == "support" and close[j] > sma_vals[j]:
                return p
            if kind == "resistance" and close[j] < sma_vals[j]:
                return p
    return None



def _find_early_directional_anchor(pivots: List[Dict], sma: Optional[pd.Series], df: pd.DataFrame,
                                    kind: str, atr_now: Optional[float],
                                    min_move_atr: float = 0.35, confirm_bars: int = 3,
                                    lookahead: int = 25) -> Optional[Dict]:
    """
    Draw the trendline as soon as direction is PROVEN, without waiting for a
    second confirming pivot (HH/HL sequence) to exist yet.

    "Proven" here means: the SMA-reclaim swing (the pivot where price broke
    away and closed back through the 20 SMA -- the same anchor
    _find_sma_reclaim_anchor locates) has since seen price move a meaningful
    distance beyond it AND stay on the confirmed side of the SMA for several
    bars in a row (filters out a single fakeout candle flipping the SMA
    momentarily).

    This is what lets a fresh reversal leg get a trendline immediately --
    anchored at the reversal point, sloped along the SMA's realized path to
    the current bar -- so a pullback touching it is a genuine continuation
    entry instead of waiting for a second swing to fully form first (by
    which point much of the move is already over).

    Returns a synthetic (anchor, endpoint) pair shaped like the pivot dicts
    _fit_primary expects, with endpoint pinned to the current bar and priced
    off the live SMA value (not a raw candle wick) so the line tracks the
    SMA's slope the way a trader's hand-drawn reversal line does.
    """
    anchor = _find_sma_reclaim_anchor(pivots, sma, df, kind, lookahead=lookahead)
    if anchor is None or sma is None or sma.dropna().empty:
        return None

    # Guard against a genuine range: a local dip/spike can push a few closes
    # to the "wrong" side of an otherwise FLAT SMA without the market
    # actually having reversed. Require the SMA itself to be sloping in the
    # matching direction, not just price sitting on one side of it.
    want_dir = "RISING" if kind == "support" else "FALLING"
    if _sma_direction(sma, atr_now=atr_now) != want_dir:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(float)
    sma_vals = sma.to_numpy(float)
    n = len(close)
    a_idx = anchor["index"]
    last_idx = n - 1
    if last_idx - a_idx < confirm_bars + 1:
        return None  # too fresh -- give it a few bars to prove itself

    # Require the last `confirm_bars` closes to stay on the confirmed side
    # of the SMA (support: above it; resistance: below it) -- one wick back
    # through doesn't count.
    tail = range(last_idx - confirm_bars + 1, last_idx + 1)
    for j in tail:
        if not (np.isfinite(close[j]) and np.isfinite(sma_vals[j])):
            return None
        if kind == "support" and close[j] <= sma_vals[j]:
            return None
        if kind == "resistance" and close[j] >= sma_vals[j]:
            return None

    move = abs(float(close[last_idx]) - float(anchor["price"]))
    if atr_now and atr_now > 0 and move < min_move_atr * atr_now:
        return None  # hasn't moved far enough yet to call it proven

    # Reject a range: a local dip/rally inside an established box can pass
    # every check above (real ATR-sized move, several closes on one side of
    # a momentarily-sloping SMA) without the market actually having broken
    # anything -- it's just another swing inside the same range. Require an
    # actual break of structure: price must have closed beyond the most
    # recent OPPOSITE-type pivot before the anchor (for a resistance/down
    # reversal, that's the last swing low that was propping up the prior
    # uptrend -- breaking it is the real tell, not just retracing partway
    # back into ground the range already covers).
    opp_type = "low" if kind == "resistance" else "high"
    prior_opposite = [p for p in pivots if p["type"] == opp_type and p["index"] < a_idx]
    if prior_opposite:
        last_opp = max(prior_opposite, key=lambda p: p["index"])
        if kind == "resistance" and float(close[last_idx]) >= float(last_opp["price"]):
            return None
        if kind == "support" and float(close[last_idx]) <= float(last_opp["price"]):
            return None

    if not np.isfinite(sma_vals[last_idx]):
        return None
    endpoint = {"index": last_idx, "price": float(sma_vals[last_idx]), "type": anchor["type"], "source": "sma_slope"}
    return {"anchor": anchor, "endpoint": endpoint}


def _find_impulse_anchor_pair(pivots: List[Dict], df: pd.DataFrame, kind: str) -> Optional[Tuple[Dict, Dict]]:
    """Select the hand-drawn structural trendline anchors.

    BUY/uptrend: last confirmed HL before a meaningful bullish impulse
    (HL -> HH), then the latest confirmed HL after that impulse.

    SELL/downtrend: last confirmed HH before a meaningful bearish impulse
    (HH -> LL), then the latest confirmed LH after that impulse.
    """
    if df is None or len(pivots) < 4:
        return None

    ordered = sorted(pivots, key=lambda p: p.get('index', 0))
    atr = pd.to_numeric(df['ATR'], errors='coerce').to_numpy(float) if 'ATR' in df.columns else None

    def _atr_at(i: int) -> float:
        if atr is not None and 0 <= i < len(atr) and np.isfinite(atr[i]) and atr[i] > 0:
            return float(atr[i])
        return max(float(df['High'].iloc[i] - df['Low'].iloc[i]), 1e-9)

    lows = [p for p in ordered if p.get('type') == 'low']
    highs = [p for p in ordered if p.get('type') == 'high']

    if kind == 'support':
        if len(lows) < 3 or len(highs) < 2:
            return None
        candidates = []
        for i in range(1, len(lows) - 1):
            anchor = lows[i]
            if float(anchor['price']) <= float(lows[i - 1]['price']):
                continue
            next_highs = [h for h in highs if h['index'] > anchor['index']]
            if not next_highs:
                continue
            impulse_high = next_highs[0]
            prior_highs = [h for h in highs if h['index'] < impulse_high['index']]
            if not prior_highs or float(impulse_high['price']) <= float(prior_highs[-1]['price']):
                continue
            move = float(impulse_high['price']) - float(anchor['price'])
            impulse_atr = move / max((_atr_at(anchor['index']) + _atr_at(impulse_high['index'])) / 2.0, 1e-9)
            if impulse_atr < 1.25:
                continue
            candidates.append((impulse_high['index'], impulse_atr, anchor))
        if not candidates:
            return None
        _, _, anchor = max(candidates, key=lambda x: (x[0], x[1]))
        endpoints = [p for p in lows if p['index'] > anchor['index'] and float(p['price']) > float(anchor['price'])]
        if not endpoints:
            return None
        endpoint = endpoints[-1]
        return (anchor, endpoint) if endpoint['index'] - anchor['index'] >= 4 else None

    if kind == 'resistance':
        if len(highs) < 3 or len(lows) < 2:
            return None
        candidates = []
        for i in range(1, len(highs) - 1):
            anchor = highs[i]
            if float(anchor['price']) >= float(highs[i - 1]['price']):
                continue
            next_lows = [l for l in lows if l['index'] > anchor['index']]
            if not next_lows:
                continue
            impulse_low = next_lows[0]
            prior_lows = [l for l in lows if l['index'] < impulse_low['index']]
            if not prior_lows or float(impulse_low['price']) >= float(prior_lows[-1]['price']):
                continue
            move = float(anchor['price']) - float(impulse_low['price'])
            impulse_atr = move / max((_atr_at(anchor['index']) + _atr_at(impulse_low['index'])) / 2.0, 1e-9)
            if impulse_atr < 1.25:
                continue
            candidates.append((impulse_low['index'], impulse_atr, anchor))
        if not candidates:
            return None
        _, _, anchor = max(candidates, key=lambda x: (x[0], x[1]))
        endpoints = [p for p in highs if p['index'] > anchor['index'] and float(p['price']) < float(anchor['price'])]
        if not endpoints:
            return None
        endpoint = endpoints[-1]
        return (anchor, endpoint) if endpoint['index'] - anchor['index'] >= 4 else None

    return None

def _fit_primary(pivots: List[Dict], kind: str, n: int, df: pd.DataFrame,
                  sma: Optional[pd.Series] = None, atr_now: Optional[float] = None) -> Optional[Dict]:
    """
    Classic trendline: connect sequential Higher Lows (support) or
    Lower Highs (resistance). Filters are kept light so the line is
    almost always drawn when structure exists — matching MT5 hand-drawn style.
    """
    pts = _get_sequential_pivots(pivots, kind, min_bars=3)
    too_few_pivots = False
    if len(pts) < 2:
        # Fallback: any two pivots of the correct type
        want = "low" if kind == "support" else "high"
        raw_pts = [p for p in pivots if p["type"] == want]
        if len(raw_pts) < 2:
            # Not enough raw pivots of this type to even attempt a classic
            # fit -- don't bail out yet though, the early SMA-slope anchor
            # below only needs ONE pivot (the reversal swing itself) plus
            # the current bar, so it can still fire here.
            too_few_pivots = True
            pts = raw_pts
        else:
            pts = raw_pts[-2:]

    # Hand-drawn structural rule: anchor at the last HL/HH that launched
    # the current impulse, then connect it to the latest same-side pivot.
    # The older SMA-reclaim method remains only as a fallback when the
    # structural sequence is not mature enough.
    impulse_pair = None if too_few_pivots else _find_impulse_anchor_pair(pivots, df, kind)
    if impulse_pair is not None:
        a, b = impulse_pair
        sma_anchor = None
    else:
        sma_anchor = _find_sma_reclaim_anchor(pivots, sma, df, kind)
        if sma_anchor is not None:
            later_pts = [p for p in pts if p["index"] > sma_anchor["index"]]
            b = later_pts[-1] if later_pts else (pts[-1] if pts else None)
            if b is not None and b["index"] > sma_anchor["index"]:
                a = sma_anchor
            else:
                a = None
        else:
            a = None

    early_anchor = None
    if a is None:
        # Direction just reversed and no second confirming pivot exists yet
        # (e.g. price just made a fresh LL but there's only one lower high
        # so far) -- rather than falling straight to a same-old-direction
        # fallback pair that gets rejected by the slope check below, check
        # whether the SMA-reclaim swing has since PROVEN the new direction
        # (meaningful move + several bars holding the new side of the SMA).
        # If so, draw the line now -- anchored at the reversal, sloped along
        # the SMA to the current bar -- instead of waiting for a slower
        # second-pivot confirmation that costs most of the pullback move.
        early = _find_early_directional_anchor(pivots, sma, df, kind, atr_now)
        if early is not None:
            a, b = early["anchor"], early["endpoint"]
            early_anchor = early["anchor"]

    if a is None:
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
        elif len(pts) >= 2:
            a, b = pts[-2], pts[-1]
        else:
            # Too few pivots for any classic fit, and the early SMA-slope
            # anchor above didn't clear its bar either (not proven yet) --
            # genuinely nothing to draw here, let S/R handle it instead.
            return None

    if b["index"] <= a["index"]:
        return None
    if b["index"] - a["index"] < 4:
        return None

    slope = (b["price"] - a["price"]) / max(b["index"] - a["index"], 1)
    y_end = _line_value(a["index"], a["price"], b["index"], b["price"], n - 1)

    # Direction must match kind (light check only) -- skipped for the early
    # SMA-slope anchor since its slope IS the proof of direction already.
    if early_anchor is None:
        if kind == "support" and slope < -1e-12:
            return None
        if kind == "resistance" and slope > 1e-12:
            return None

    if early_anchor is not None:
        touches = 1
        violations = 0
        quality = "unconfirmed"
        method = "sma_slope_early"
    else:
        touches = _count_touches(df, a["index"], a["price"], b["index"], b["price"], kind, tol_atr=0.50)
        violations = _count_violations(df, a["index"], a["price"], b["index"], b["price"], kind, tol_atr=0.35)
        quality = "unconfirmed" if touches < 2 else ("confirmed" if touches <= 4 else "crowded")
        touches = max(touches, 2)
        method = "impulse_structural" if impulse_pair is not None else ("sma_reclaim" if (sma_anchor is not None and a is sma_anchor) else "classic_sequential")

    return {
        "x0": a["index"], "y0": float(a["price"]),
        "x1": b["index"], "y1": float(b["price"]),
        "y_end": float(y_end),
        "slope": float(slope),
        "touches": touches,
        "violations": violations,
        "confirmed": quality == "confirmed",
        "quality": quality,
        "kind": kind,
        "bars_since_last_touch": n - 1 - b["index"],
        "method": method,
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


def _rail_fit_quality(df: pd.DataFrame, line: Dict, kind: str, n: int) -> int:
    """
    0-100 goodness-of-fit for a rail across ALL its actual touches (not just
    the 2 anchor points that defined it) -- how tightly the wicks hug the
    line, in ATR terms. This is what the sample report's "Fit quality: 94%"
    represents.
    """
    pts = _touch_points(df, int(line["x0"]), line["y0"], int(line["x1"]), line["y1"], kind, tol_atr=0.6)
    if len(pts) < 2:
        return 60  # only the 2 defining anchors -- can't score fit, assume moderate
    atr_col = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).values
    devs = []
    for p in pts:
        lv = _line_value(line["x0"], line["y0"], line["x1"], line["y1"], p["index"])
        a = float(atr_col[p["index"]]) if p["index"] < len(atr_col) and atr_col[p["index"]] > 0 else abs(p["price"]) * 0.002
        devs.append(abs(p["price"] - lv) / max(a, 1e-9))
    avg_dev = sum(devs) / len(devs)
    # avg_dev of 0 ATR -> 100%, 0.6+ ATR -> floor near 40%
    return max(40, min(100, int(round(100 - avg_dev * 100))))


def _slope_word(slope: float, atr_now: Optional[float]) -> str:
    threshold = (atr_now * 0.02) if atr_now else 1e-6
    if slope > threshold:
        return "rising"
    if slope < -threshold:
        return "falling"
    return "flat"


def build_pattern_visual_report(wedge: Dict[str, Any], df: pd.DataFrame, n: int) -> Optional[Dict[str, Any]]:
    """
    Assembles the '📐 VISUAL PATTERN' geometry + breakout/retest + confidence
    block for a converging wedge/triangle, matching the sample schema:
    rail directions, touches, fit quality, breakout state, retest state,
    final confidence, entry decision.
    """
    if not wedge:
        return None

    lower, upper = wedge["lower"], wedge["upper"]
    bias = wedge["bias"]
    atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else None

    upper_dir = _slope_word(upper["slope"], atr_now)
    lower_dir = _slope_word(lower["slope"], atr_now)
    upper_fit = _rail_fit_quality(df, upper, "resistance", n)
    lower_fit = _rail_fit_quality(df, lower, "support", n)
    fit_quality = int(round((upper_fit + lower_fit) / 2))

    # Geometry confidence: touches + fit quality + how cleanly the rails
    # converge (gap_end vs gap_start -- tighter apex = more mature pattern).
    touch_score = min(30, (upper["touches"] + lower["touches"]) * 3)
    convergence_ratio = 1 - (wedge["gap_end"] / wedge["gap_start"]) if wedge.get("gap_start") else 0
    convergence_score = min(25, int(convergence_ratio * 25))
    geometry_confidence = max(0, min(100, int(fit_quality * 0.45 + touch_score + convergence_score)))

    # Breakout direction implied by the pattern bias.
    if bias == "BUY":
        break_kind, break_line = "resistance_break_up", upper
    elif bias == "SELL":
        break_kind, break_line = "support_break_down", lower
    else:
        break_kind, break_line = None, None

    breakout_grade = None
    retest = {"status": "INTACT", "note": "No confirmed break yet."}
    if break_line:
        breakout_grade = _grade_breakout(df, break_line, break_kind, n)
        retest = _trendline_retest_state(df, break_line, breakout_grade, break_kind)

    br_status = "NOT YET" if not breakout_grade else \
        ("CONFIRMED" if breakout_grade["strength"] == "confirmed" else
         "DEVELOPING" if breakout_grade["strength"] == "developing" else "WEAK / WICK ONLY")
    rt_status = {
        "BREAK_RETEST_CONFIRMED": "CONFIRMED",
        "FAKEOUT": "FAILED (fakeout)",
        "BREAK_CONFIRMED": "PENDING",
        "BREAK_DEVELOPING": "PENDING",
        "INTACT": "NOT APPLICABLE",
    }.get(retest.get("status"), "PENDING")

    # Final confidence blends geometry with how far the breakout/retest
    # sequence has actually progressed -- a clean-looking wedge that hasn't
    # broken yet scores lower here than in the top-line geometry confidence.
    final_confidence = geometry_confidence
    entry_status = "WAIT"
    if retest.get("status") == "BREAK_RETEST_CONFIRMED":
        final_confidence = min(100, geometry_confidence + 4)
        entry_status = "CONFIRMED"
    elif breakout_grade and breakout_grade["strength"] == "confirmed":
        final_confidence = max(0, geometry_confidence - 8)
        entry_status = "WAIT FOR RETEST"
    elif breakout_grade and breakout_grade["strength"] == "developing":
        final_confidence = max(0, geometry_confidence - 15)
        entry_status = "WAIT"
    elif retest.get("status") == "FAKEOUT":
        final_confidence = max(0, geometry_confidence - 30)
        entry_status = "INVALIDATED"
    else:
        final_confidence = max(0, geometry_confidence - 20)
        entry_status = "WAIT"

    return {
        "pattern_name": wedge["pattern"], "bias": bias,
        "confidence": geometry_confidence,
        "upper_dir": upper_dir, "lower_dir": lower_dir,
        "upper_touches": upper["touches"], "lower_touches": lower["touches"],
        "fit_quality": fit_quality,
        "breakout_status": br_status, "retest_status": rt_status,
        "final_confidence": final_confidence,
        "entry_status": entry_status,
    }


def _sr_setup_confidence(df: pd.DataFrame, horizontal_levels: List[Dict], close: float,
                          atr_now: Optional[float]) -> Optional[Dict[str, Any]]:
    """
    Confidence that the nearest horizontal S/R level is a live, tradeable
    setup right now -- not just a level that exists somewhere on the chart.
    Rewards proximity (price actually at the level, not 5 ATR away),
    touch count, and level quality.
    """
    if not horizontal_levels:
        return None
    best, best_dist = None, None
    for lvl in horizontal_levels:
        dist = abs(close - float(lvl["price"]))
        if best_dist is None or dist < best_dist:
            best, best_dist = lvl, dist
    if best is None:
        return None
    dist_atr = (best_dist / atr_now) if atr_now else 999
    if dist_atr <= 0.15:
        prox_score = 30
    elif dist_atr <= 0.5:
        prox_score = 20
    elif dist_atr <= 1.2:
        prox_score = 8
    else:
        prox_score = 0
    touch_score = min(35, int(best.get("touches", 0)) * 8)
    quality_score = {"confirmed": 15, "crowded": 8, "unconfirmed": 4}.get(best.get("quality"), 4)
    confidence = max(0, min(100, touch_score + quality_score + prox_score + 15))
    bias = "BUY" if best.get("side") == "support" else "SELL"
    return {"confidence": confidence, "level": best, "bias": bias, "distance_atr": round(dist_atr, 2)}


def _pattern_uses_trendline_geometry(sp: Optional[Dict], primary_line: Optional[Dict],
                                      index_tol: int = 2, price_tol_frac: float = 0.01) -> bool:
    """
    A trendline is a plain structural read: a rising support or falling
    resistance connecting swings. A classic chart pattern (Head & Shoulders,
    Double Top, Wedge, Triangle...) is a *named formation* -- and several of
    them are literally built out of trendline rails (an H&S's
    shoulder-to-shoulder/neckline slope, a wedge's converging upper/lower
    rails). Drawing that rail on the chart doesn't make the setup a
    "trendline setup" -- it's the pattern's own skeleton.

    This returns True when the "TRENDLINE" score being computed is really
    just re-describing the same rail the detected pattern already accounts
    for, so the caller can stop letting it compete as an independent setup
    and inflate the report with e.g. "TRENDLINE 63%" while the chart is
    unambiguously showing a Head and Shoulders.

    This is decided live, per-call, off the actual anchor points -- never a
    blanket "any wedge counts" assumption. A scanned pattern's key points
    (for a wedge/triangle these are the same points its own boundary_lines
    were fit through; for a classic pattern they're the labelled Head/
    Shoulder/Top/Bottom points) are checked against the primary trendline's
    own two anchor points. If at least two of them coincide within
    tolerance, the "trendline" being scored IS the pattern's own skeleton --
    otherwise they're genuinely two different rails on the chart and both
    can stand as independent setups.
    """
    if not sp or not primary_line:
        return False
    kps = sp.get("key_points") or []
    if len(kps) < 2:
        return False
    lx0, ly0 = primary_line.get("x0"), primary_line.get("y0")
    lx1, ly1 = primary_line.get("x1"), primary_line.get("y1")
    if None in (lx0, ly0, lx1, ly1):
        return False
    anchors = [(lx0, ly0), (lx1, ly1)]
    matches = 0
    for ax_, ay in anchors:
        for kp in kps:
            try:
                kx, ky = float(kp[0]), float(kp[1])
            except (TypeError, ValueError, IndexError):
                continue
            if abs(kx - ax_) <= index_tol and abs(ky - ay) <= max(abs(ay), 1e-9) * price_tol_frac:
                matches += 1
                break
    return matches >= 2


def select_best_setup(family: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Scans the SAME analysis (trendline geometry, pattern geometry, S/R
    proximity) and scores all three so the report/chart can lead with
    whichever one is actually the strongest, live opportunity right now --
    instead of always framing everything as a trendline setup and leaving
    a trader "waiting endlessly without real reason" when the trendline
    happens to be weak but a pattern or S/R reaction is strong.
    """
    close = float(df["Close"].iloc[-1])
    atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else None

    trendline_conf = family.get("strength", 0) if family.get("family_kind", "none") != "none" else 0

    # Score the properly stage-gated, confirmation-checked pattern
    # (scanned_pattern) as the ONE source of pattern confidence -- this
    # covers triangles/wedges too (market_analysis.detect_triangle_or_wedge),
    # not just classic reversal shapes. The older independent-slope wedge
    # fit (`pattern_visual` / family["wedge"]) is deliberately NOT scored
    # here: it has no stage gate and no confirmation-candle check, so
    # letting it compete would let an unconfirmed shape win BEST SETUP.
    pattern_conf, pattern_label = 0, None
    sp = family.get("scanned_pattern")
    mw = family.get("mw_pattern")
    if sp:
        stage = str(family.get("pattern_stage") or sp.get("stage") or "").upper()
        mult = {"CONFIRMED": 1.0, "TRIGGERED": 0.85, "FORMING": 0.55}.get(stage, 0.5)
        pattern_conf, pattern_label = int(sp.get("confidence", 0) * mult), sp.get("name")
    elif mw:
        pattern_conf, pattern_label = int(family.get("pattern_confidence", 0)), mw.get("name")

    sr_result = _sr_setup_confidence(df, family.get("horizontal_levels") or [], close, atr_now)
    sr_conf = sr_result["confidence"] if sr_result else 0

    # A trendline rail that's actually just the detected pattern's own
    # skeleton (e.g. this H&S's Head -> Right Shoulder slope) is not a
    # second, independent setup -- score it for display, but don't let it
    # outrank the pattern it's part of.
    sp_for_check = family.get("scanned_pattern")
    primary_line = (family.get("downtrends") or [None])[0] or (family.get("uptrends") or [None])[0]
    trendline_is_pattern_rail = _pattern_uses_trendline_geometry(sp_for_check, primary_line)

    scores = {"TRENDLINE": trendline_conf, "PATTERN": pattern_conf, "S/R": sr_conf}
    eligible = dict(scores)
    if trendline_is_pattern_rail and pattern_conf > 0:
        eligible["TRENDLINE"] = -1
    winner = max(eligible, key=eligible.get)
    # Don't dress up a genuinely setup-less chart as "TRENDLINE (0%)" --
    # if nothing cleared a basic bar, say so plainly instead of implying
    # a setup is being tracked when none exists.
    if eligible[winner] < 30:
        winner = "NONE"

    family["setup_scores"] = scores
    family["active_setup"] = winner
    family["active_setup_confidence"] = scores.get(winner, 0)
    family["pattern_label"] = pattern_label
    family["trendline_is_pattern_rail"] = trendline_is_pattern_rail
    if winner == "S/R" and sr_result:
        family["sr_setup"] = sr_result
        family["active_pattern"] = "sr"  # drives chart drawing selection
    return family


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
        touch_prices = [float(t["price"]) for t in c["touches"]]
        levels.append({
            "price": c["price"], "touches": n_touch, "span": span,
            "first_index": first_idx, "last_index": last_idx,
            "side": "resistance" if c["price"] >= close else "support",
            "quality": quality, "score": round(score, 2),
            # Real zone band (not a single infinitely-thin price) so the
            # chart can shade the actual range the touches occurred in,
            # padded a touch by the clustering tolerance itself.
            "zone_low": min(min(touch_prices), c["price"] - tol),
            "zone_high": max(max(touch_prices), c["price"] + tol),
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

    # Computed early (not just for the later confluence/color-state display)
    # so the SMA-reclaim pivot can anchor the trendline itself -- see
    # _find_sma_reclaim_anchor.
    sma20_early = _sma20_series(df, period=20, applied_price="median")
    atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else None

    support = _fit_primary(recent_pivots, "support", n, df, sma=sma20_early, atr_now=atr_now)
    resistance = _fit_primary(recent_pivots, "resistance", n, df, sma=sma20_early, atr_now=atr_now)

    # Reject a candidate diagonal line whose actual price movement across
    # its own span is too shallow to be a meaningful trend -- e.g. two
    # swing lows that are technically "rising" by a few points over three
    # days. That's a range, not an uptrend, and drawing it as a diagonal
    # "ascending" rail is misleading -- it should be left for the
    # horizontal S/R clustering below (_detect_horizontal_levels) to pick
    # up instead, which is exactly what that layer is for.
    MIN_TREND_MOVE_ATR = 0.35

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
        # When BOTH rails exist, do not let the nearest rail decide the
        # whole directional bias. A falling resistance + rising support is
        # a converging structure. Use the combined geometry + 20-SMA state
        # for directional bias; breakout/retest remains the confirmation gate.
        lower = min(
            float((support or {}).get("y_end", family_lines[0]["y_end"])),
            float((resistance or {}).get("y_end", family_lines[-1]["y_end"])),
        )
        upper = max(
            float((support or {}).get("y_end", family_lines[0]["y_end"])),
            float((resistance or {}).get("y_end", family_lines[-1]["y_end"])),
        )
        mid = (lower + upper) / 2.0

        if support and resistance:
            sma_dir_early = _sma_direction(sma20_early, atr_now=atr_now)
            sma_valid = sma20_early.dropna()
            sma_now_early = float(sma_valid.iloc[-1]) if not sma_valid.empty else None
            support_now = float(support["y_end"])
            resistance_now = float(resistance["y_end"])

            bullish_context = (
                support_now < close
                and sma_dir_early == "RISING"
                and (sma_now_early is None or close >= sma_now_early)
            )
            bearish_context = (
                close < resistance_now
                and sma_dir_early == "FALLING"
                and (sma_now_early is None or close <= sma_now_early)
            )

            if bullish_context and not bearish_context:
                direction = "BUY"
                strength = 55
                reasons.append("Converging rails · rising support + falling resistance")
                reasons.append("Bullish structure: price above rising support and 20 SMA is rising")
                if close > mid:
                    strength += 10
                    reasons.append("Price above pattern midpoint — bullish control")
                else:
                    reasons.append("Price above support but below resistance — bullish setup, wait for confirmation")
                primary = support
                family_kind = "ascending"
            elif bearish_context and not bullish_context:
                direction = "SELL"
                strength = 55
                reasons.append("Converging rails · rising support + falling resistance")
                reasons.append("Bearish structure: price below falling resistance and 20 SMA is falling")
                if close < mid:
                    strength += 10
                    reasons.append("Price below pattern midpoint — bearish control")
                else:
                    reasons.append("Price below resistance but above support — bearish setup, wait for confirmation")
                primary = resistance
                family_kind = "descending"
            else:
                reasons.append("Converging rails conflict — wait for directional breakout")
        touch_note = {
            "unconfirmed": "⚠️ only 2 touches -- unconfirmed, treat as tentative",
            "confirmed": f"{primary['touches']} touches -- validated structure",
            "crowded": f"{primary['touches']} touches -- crowded level, order flow may be depleted",
        }.get(primary.get("quality"), f"{primary['touches']} touches")

        if support and resistance:
            # Dual-rail direction was decided above; do not overwrite it
            # using only whichever rail is closest to current price.
            pass
        elif family_kind == "ascending":
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

    # Dual-rail breakout detection: when both rising support and falling
    # resistance exist, inspect BOTH rails before the lifecycle/report block.
    # The previous logic only graded a breakout inside the single-rail branch,
    # so a visible close below rising support could still be reported as
    # NOT BROKEN when the two rails were both present.
    if support and resistance:
        close_now = float(df["Close"].iloc[-1])
        support_now = float(support["y_end"])
        resistance_now = float(resistance["y_end"])

        if close_now < support_now:
            primary = support
            family_kind = "ascending"
            breakout_grade = _grade_breakout(df, support, "support_break_down", n)
            direction = "SELL"
            reasons.append(
                f"BREAK below rising support — {breakout_grade['penetration_atr']} ATR, "
                f"{breakout_grade['consecutive_closes']} close(s), body {breakout_grade['body_ratio']}"
            )
        elif close_now > resistance_now:
            primary = resistance
            family_kind = "descending"
            breakout_grade = _grade_breakout(df, resistance, "resistance_break_up", n)
            direction = "BUY"
            reasons.append(
                f"BREAK above falling resistance — {breakout_grade['penetration_atr']} ATR, "
                f"{breakout_grade['consecutive_closes']} close(s), body {breakout_grade['body_ratio']}"
            )

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
    pattern_visual = build_pattern_visual_report(wedge, df, n) if wedge else None
    # Keep only the 2 strongest horizontal levels (1 support + 1 resistance,
    # nearest to price) for the chart -- the full clustered history is still
    # useful for the text report, but 3-4 stacked S/R lines on top of a
    # diagonal family + wedge/M-W pattern is what caused the label pile-up
    # around the entry zone. The report path can still ask for more via
    # max_levels if it wants the full picture.
    horizontal_levels = _detect_horizontal_levels(df, pivots, n, max_levels=2)
    # NOTE: this legacy independent-slope wedge fit (`wedge`) is kept ONLY
    # for the geometry it hands to `pattern_visual`'s narrative text -- it
    # must never set `direction` itself. It has no neckline-stage gate and
    # no marubozu/engulfing confirmation-candle check, so letting it move
    # direction here would silently bypass the same confirmation rule every
    # other pattern in this bot is held to. The properly-gated equivalent
    # (market_analysis.detect_triangle_or_wedge, via scanned_pattern) is
    # what's allowed to set direction -- see run_trendline_analysis, which
    # only does so once the pattern's stage is TRIGGERED/CONFIRMED.

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

    # --- 20 SMA (median price) direction + confluence with the primary
    # trendline. Computed here so both callers (run_trendline_analysis and
    # execution_engine's auto-fallback) get it for free.
    atr_now_sma = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else None
    sma20 = sma20_early
    sma_dir = _sma_direction(sma20, atr_now=atr_now_sma)
    sma_last = float(sma20.dropna().iloc[-1]) if not sma20.dropna().empty else None
    primary_now = primary.get("y_end") if primary else None
    sma_confluence = _sma_trendline_confluence(sma_last, primary_now, atr_now_sma, sma_dir, family_kind)

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
        "pattern_visual": pattern_visual,
        "sma20_series": sma20,
        "horizontal_levels": horizontal_levels,
        "projections": projections,
        "mw_pattern": mw,
        "market_sequence": market_seq,
        "pivots": pivots[-16:],
        # Full pivot history (not trimmed to 16) so pattern detection has
        # enough structure to check for a dominant rival swing outside the
        # last few points -- see scan_all_patterns(pivots=...).
        "pivots_full": pivots,
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
            # When both rails exist, the converging geometry is the primary
            # pattern. A merely-forming M/W must not hide it or manufacture
            # a high-confidence reversal label.
            "wedge" if (support and resistance and wedge and strength >= 55) else
            "mw" if mw and not (support and resistance) else
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
        "sma_period": 20,
        "sma_applied_price": "median",
        "sma_direction": sma_dir,
        "sma_last": sma_last,
        "sma_confluence": sma_confluence,
        "sma_series": sma20.values,
        "trendline_color_state": _trendline_color_state(direction, sma_dir),
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
        for m, label in mult
