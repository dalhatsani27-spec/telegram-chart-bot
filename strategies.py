"""
strategies.py
=============
The trendline analysis engine this bot runs.

Calls topdown_engine-equivalent HTF context (unified_strategy.get_topdown_bias)
when the caller supplies it, then does its own timeframe-specific work on
the chart: pivot-fit trendline family, an independent 20-SMA median-price
master-line reading, a pullback-entry confirmation adapter, measured-move
and liquidity targets, and classic-pattern detection sharing the same
pivots. SMC and OTE were removed -- trendline is the sole chart-geometry
evidence source now (see unified_strategy.py for how it's combined with
Alligator regime, HTF bias, and fundamentals into one decision).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import market_data
from market_analysis import (
    zigzag_swings, find_swings, compute_volume_profile, detect_confirmation_candle,
    analyse_structure, detect_order_blocks, scan_all_patterns, detect_market_sequence,
    is_price_ranging_vs_sma, is_bullish_marubozu, is_bearish_marubozu,
    is_bullish_engulfing, is_bearish_engulfing,
)


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
    support_raw, resistance_raw = support, resistance

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

    # --- Always have SOMETHING to draw ---------------------------------
    # `support`/`resistance` above stay exactly as before (None whenever
    # the structure doesn't clear the trading bar) so direction/strength/
    # readiness logic below is completely unchanged. `support_line`/
    # `resistance_line` are separate, chart-only variables: when the real
    # ones got rejected (or never existed), fall back to whatever geometry
    # is actually there -- the rejected-for-shallow line, or if there
    # wasn't even that, the loosest possible fractal-swing connection --
    # so a ranging/choppy chart still gets a line instead of a blank
    # space. These NEVER feed direction, strength, or the trading gate.
    def _visual_fallback_line(kind: str) -> Optional[Dict[str, Any]]:
        loose = find_fractal_pivots(df, left=2, right=2)
        want = "low" if kind == "support" else "high"
        pts = [p for p in loose if p["type"] == want]
        if len(pts) >= 2:
            a, b = pts[-2], pts[-1]
            if b["index"] == a["index"]:
                return None
        elif len(pts) == 1:
            a = pts[0]
            b = {"index": n - 1, "price": float(df["Close"].iloc[-1])}
            if b["index"] == a["index"]:
                return None
        else:
            return None
        slope = (b["price"] - a["price"]) / max(b["index"] - a["index"], 1)
        y_end = _line_value(a["index"], a["price"], b["index"], b["price"], n - 1)
        return {
            "x0": a["index"], "y0": float(a["price"]),
            "x1": b["index"], "y1": float(b["price"]),
            "y_end": float(y_end), "slope": float(slope),
            "touches": _count_touches(df, a["index"], a["price"], b["index"], b["price"], kind),
            "violations": 0, "confirmed": False, "quality": "visual_only",
            "tradeable": False, "kind": kind, "method": "visual_fallback",
        }

    support_line = support_raw or _visual_fallback_line("support")
    if support_line is not None and support_line.get("quality") != "visual_only" and support_line is support_raw and support is None:
        support_line = dict(support_line, quality="shallow", tradeable=False)
    resistance_line = resistance_raw or _visual_fallback_line("resistance")
    if resistance_line is not None and resistance_line.get("quality") != "visual_only" and resistance_line is resistance_raw and resistance is None:
        resistance_line = dict(resistance_line, quality="shallow", tradeable=False)

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

    # When nothing cleared the trading bar, `family_lines` above stays
    # empty and nothing gets drawn -- fall back to the visual-only
    # line(s) so the chart still shows a reference trendline. This only
    # affects what's drawn; `direction`/`strength`/`primary` below are
    # untouched and still correctly reflect "no tradeable trendline here".
    used_visual_fallback = False
    if not family_lines:
        visual_candidates = [l for l in (support_line, resistance_line) if l is not None]
        if visual_candidates:
            family_lines = visual_candidates
            used_visual_fallback = True

    # Direction from family geometry (price reveals it)
    direction = "NEUTRAL"
    strength = 40
    reasons = []
    if used_visual_fallback:
        reasons.append(
            "No trendline cleared the trading bar here -- showing the "
            "nearest swing-to-swing reference line(s) for visual context "
            "only; not a trade signal."
        )
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

    return_family = {
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

    return _apply_trendline_upgrades(return_family, df)


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


def _sma20_median(df: pd.DataFrame) -> pd.Series:
    median = (df["High"].astype(float) + df["Low"].astype(float)) / 2.0
    return median.rolling(20, min_periods=20).mean()


def _sma20_state(sma: pd.Series, df: pd.DataFrame) -> Tuple[str, float]:
    v = sma.dropna()
    if len(v) < 12:
        return "FLAT", 0.0
    lb = 8
    slope = (float(v.iloc[-1]) - float(v.iloc[-1 - lb])) / lb
    a = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else abs(v.iloc[-1]) * 0.001
    threshold = max(a * 0.035, abs(float(v.iloc[-1])) * 0.00008)
    if slope > threshold:
        return "RISING", slope
    if slope < -threshold:
        return "FALLING", slope
    return "FLAT", slope


def _sma20_establishing_leg(df: pd.DataFrame, sma: pd.Series, direction: str, lookback: int = 100):
    close = df["Close"].astype(float).to_numpy()
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    sv = sma.to_numpy(float)
    n = len(df)
    start = max(1, n - lookback)
    crosses = []
    for i in range(start, n):
        if not (np.isfinite(sv[i]) and np.isfinite(sv[i - 1])):
            continue
        if direction == "RISING" and close[i - 1] <= sv[i - 1] and close[i] > sv[i]:
            crosses.append(i)
        elif direction == "FALLING" and close[i - 1] >= sv[i - 1] and close[i] < sv[i]:
            crosses.append(i)
    cross = crosses[-1] if crosses else None
    if cross is None:
        return None
    leg_start = cross
    for j in range(cross - 1, max(start, cross - 35), -1):
        if not np.isfinite(sv[j]):
            continue
        opposite = close[j] <= sv[j] if direction == "RISING" else close[j] >= sv[j]
        if opposite:
            leg_start = j
        else:
            break
    lo = max(start, leg_start - 3)
    hi = cross
    if direction == "RISING":
        anchor = lo + int(np.argmin(low[lo:hi + 1]))
        return cross, anchor, "support", low
    anchor = lo + int(np.argmax(high[lo:hi + 1]))
    return cross, anchor, "resistance", high


def _sma20_master_line(df: pd.DataFrame, sma: pd.Series, direction: str, sma_slope: float) -> Optional[Dict[str, Any]]:
    """
    Authoritative 20-SMA (median-price) trendline: rather than fitting a
    line through two swing pivots, the slope comes ONLY from the 20 SMA's
    own realized slope, anchored at the establishing leg that actually
    crossed it. This is a second, independent read on trend direction
    from the pivot-fit primary trendline above -- kept as a distinct
    "master" line (`family["master_trendline"]`) rather than replacing
    the pivot fit, since the two disagreeing is itself useful information.
    """
    leg = _sma20_establishing_leg(df, sma, direction)
    if leg is None:
        return None
    cross, anchor, role, extremes = leg
    n = len(df)
    atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else abs(float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]))
    slope = abs(float(sma_slope)) if role == "support" else -abs(float(sma_slope))
    x0 = int(anchor)
    target = float(extremes[x0])
    sma_anchor = float(sma.iloc[x0])
    y0 = sma_anchor + float(np.clip(target - sma_anchor, -3 * atr_now, 3 * atr_now))
    y1 = y0 + slope * (n - 1 - x0)

    def _atr_i(i):
        if "ATR" in df.columns:
            v = float(df["ATR"].iloc[i])
            if np.isfinite(v) and v > 0:
                return v
        return max(float(df["High"].iloc[i] - df["Low"].iloc[i]), 1e-9)

    touches, violations, points = 0, 0, []
    for i in range(x0, n):
        line = y0 + slope * (i - x0)
        a = _atr_i(i)
        if abs(float(extremes[i]) - line) <= 0.60 * a:
            touches += 1
            points.append({"index": i, "price": float(extremes[i])})
        c = float(df["Close"].iloc[i])
        if role == "support" and c < line - 0.35 * a:
            violations += 1
        if role == "resistance" and c > line + 0.35 * a:
            violations += 1

    return {
        "x0": x0, "y0": y0, "x1": n - 1, "y1": y1, "y_end": y1,
        "slope": slope, "touches": touches, "violations": violations,
        "confirmed": touches >= 2,
        "quality": "confirmed" if touches >= 2 else "developing",
        "kind": role, "method": "20SMA_MEDIAN_PRICE_ESTABLISHING_LEG",
        "establishing_cross": int(cross), "establishing_anchor": int(anchor),
        "touch_points": points,
        "bars_since_last_touch": (n - 1 - points[-1]["index"]) if points else 999,
    }


def _rejection_confirmation(df: pd.DataFrame, index: int, direction: str, line_price: float, atr: float) -> Tuple[bool, Optional[str]]:
    """A clean pullback rejection candle at a trendline: marubozu, engulfing,
    or a plain strong-bodied close back in the trend direction off the line."""
    if index < 0 or index >= len(df):
        return False, None
    r = df.iloc[int(index)]
    o, h, l, c = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    body_ratio = body / rng
    prior = None
    if index > 0:
        p = df.iloc[int(index) - 1]
        prior = (float(p["Open"]), float(p["High"]), float(p["Low"]), float(p["Close"]))

    MIN_BODY_RATIO = 0.45
    MAX_REJECTION_WICK_BODY = 1.25

    if direction == "BUY":
        if is_bullish_marubozu((o, h, l, c), atr):
            return True, "Bullish Marubozu"
        if prior is not None and is_bullish_engulfing(prior, (o, h, l, c)):
            return True, "Bullish Engulfing"
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        if (c > o and body_ratio >= MIN_BODY_RATIO and lower_wick >= body * 0.60
                and upper_wick <= max(body * MAX_REJECTION_WICK_BODY, atr * 0.12)
                and c >= line_price + atr * 0.10):
            return True, "Bullish Trendline Rejection"
    if direction == "SELL":
        if is_bearish_marubozu((o, h, l, c), atr):
            return True, "Bearish Marubozu"
        if prior is not None and is_bearish_engulfing(prior, (o, h, l, c)):
            return True, "Bearish Engulfing"
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        if (c < o and body_ratio >= MIN_BODY_RATIO and upper_wick >= body * 0.60
                and lower_wick <= max(body * MAX_REJECTION_WICK_BODY, atr * 0.12)
                and c <= line_price - atr * 0.10):
            return True, "Bearish Trendline Rejection"
    return False, None


def _pullback_zone(line_price: float, atr: float, direction: str) -> Dict[str, Any]:
    PULLBACK_ZONE_ATR = 0.35
    PULLBACK_INVALIDATION_ATR = 0.15
    if direction == "BUY":
        return {"low": line_price - atr * PULLBACK_INVALIDATION_ATR, "high": line_price + atr * PULLBACK_ZONE_ATR,
                "anchor": line_price, "width_atr": PULLBACK_ZONE_ATR + PULLBACK_INVALIDATION_ATR, "side": "support"}
    return {"low": line_price - atr * PULLBACK_ZONE_ATR, "high": line_price + atr * PULLBACK_INVALIDATION_ATR,
            "anchor": line_price, "width_atr": PULLBACK_ZONE_ATR + PULLBACK_INVALIDATION_ATR, "side": "resistance"}


def _trendline_pullback_state(family: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preferred entry mode while the master trendline is intact: wait for
    price to pull back INTO the rail, then require an actual rejection/
    continuation candle before treating it as a confirmed entry -- rather
    than entering on a raw break, or on geometry alone.
    """
    df = family.get("df") if family else None
    master = family.get("master_trendline") if family else None
    direction = str((family or {}).get("direction") or "NEUTRAL").upper()
    role = str((family or {}).get("master_role") or "none").lower()

    out = {
        "state": "WAIT_PULLBACK", "confirmed": False, "confirmation": None,
        "entry_mode": None, "entry_price": None, "line_price": None,
        "zone": None, "touch": False, "distance_atr": None,
        "reason": "Wait for price to pull back into the master trendline zone.",
    }
    if df is None or df.empty or not master or direction not in ("BUY", "SELL"):
        return out
    if direction == "BUY" and role != "support":
        return out
    if direction == "SELL" and role != "resistance":
        return out

    i = len(df) - 1
    line_price = _line_value(master["x0"], master["y0"], master["x1"], master["y1"], i)
    atr = float(df["ATR"].iloc[i]) if "ATR" in df.columns and df["ATR"].iloc[i] > 0 else max(float(df["High"].iloc[i] - df["Low"].iloc[i]), 1e-9)
    zone = _pullback_zone(line_price, atr, direction)
    close = float(df["Close"].iloc[i])
    high = float(df["High"].iloc[i])
    low = float(df["Low"].iloc[i])
    distance_atr = abs(close - line_price) / atr if atr > 0 else 999.0
    out.update(line_price=line_price, zone=zone, distance_atr=round(distance_atr, 2))

    PULLBACK_INVALIDATION_ATR = 0.15
    if direction == "BUY" and close < line_price - atr * PULLBACK_INVALIDATION_ATR:
        out.update(state="INVALIDATED", reason="Price closed materially below rising support; pullback setup invalidated.")
        return out
    if direction == "SELL" and close > line_price + atr * PULLBACK_INVALIDATION_ATR:
        out.update(state="INVALIDATED", reason="Price closed materially above falling resistance; pullback setup invalidated.")
        return out

    touch = high >= zone["low"] and low <= zone["high"]
    out["touch"] = bool(touch)
    if not touch:
        out["reason"] = (f"Wait for pullback into master {role} {zone['low']:.5f}-{zone['high']:.5f} "
                          f"({zone['anchor']:.5f} trendline).")
        return out

    confirmed, name = _rejection_confirmation(df, i, direction, line_price, atr)
    if confirmed:
        out.update(state="PULLBACK_ENTRY_CONFIRMED", confirmed=True, confirmation=name,
                    entry_mode="MARKET", entry_price=close,
                    reason=f"{name} rejected the master {role} and closed back in the trend direction.")
    else:
        out.update(state="WAIT_PULLBACK_CONFIRMATION",
                    reason=f"Price touched the master {role}; wait for a directional rejection/continuation candle before entry.")
    return out


def _legacy_break_retest_entry(family: Dict[str, Any]) -> Dict[str, Any]:
    """Break/retest continuation entry, used once the master line has broken."""
    df = family.get("df") if family else None
    direction = str((family or {}).get("direction") or "NEUTRAL").upper()
    retest = (family or {}).get("trendline_retest") or {}
    base = {"confirmed": False, "state": "WAIT_BREAK", "confirmation": None, "entry_mode": None, "reason": "Wait for trendline setup."}
    if df is None or len(df) < 3 or direction not in ("BUY", "SELL"):
        return base
    status = str(retest.get("status") or "INTACT")
    if status == "FAKEOUT":
        return {**base, "state": "INVALIDATED", "reason": "Trendline break was reclaimed; wait for a fresh setup."}
    if status != "BREAK_RETEST_CONFIRMED":
        if status in ("BREAK_CONFIRMED", "BREAK_DEVELOPING"):
            return {**base, "state": "WAIT_RETEST", "reason": "Break detected; wait for the broken trendline to retest and hold."}
        return base
    try:
        ri = int(retest.get("retest_index"))
    except Exception:
        ri = None
    if ri is None or ri >= len(df) - 1:
        return {**base, "state": "WAIT_CONTINUATION", "reason": "Retest confirmed; wait for the first directional continuation candle."}
    latest = len(df) - 1
    atr = float(df["ATR"].iloc[latest]) if "ATR" in df.columns and df["ATR"].iloc[latest] > 0 else max(float(df["High"].iloc[latest] - df["Low"].iloc[latest]), 1e-9)
    ok, name = _rejection_confirmation(df, latest, direction, float(retest.get("retest_level") or df["Close"].iloc[ri]), atr)
    if ok:
        return {"confirmed": True, "state": "BREAK_RETEST_ENTRY_CONFIRMED", "confirmation": name,
                "entry_mode": "MARKET", "reason": f"{name} confirmed continuation after trendline break/retest."}
    return {**base, "state": "WAIT_CONTINUATION", "reason": "Retest held; wait for a directional rejection/continuation candle."}


def _apply_trendline_upgrades(family: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Runs at the end of build_trendline_family(): (1) an independent 20-SMA
    median-price "master" trendline reading, anchored at the leg that
    actually crossed the SMA, kept alongside (not replacing) the pivot-fit
    primary line; (2) a pullback-entry adapter that requires price to
    return to the master line AND get a rejection/continuation candle
    before calling anything confirmed, falling back to break+retest
    continuation once the master line itself has broken. This used to be
    two separate files (sitecustomize.py / usercustomize.py) that
    monkey-patched this function at interpreter startup -- folded in here
    as real code instead, with one change: a flat SMA no longer wipes the
    chart's family_lines/uptrends/downtrends to empty (that fought with
    the "always show a reference trendline" behavior above); it just
    skips setting a master line and leaves whatever was already built.
    """
    if not family or family.get("error"):
        return family

    sma = _sma20_median(df)
    direction, slope = _sma20_state(sma, df)
    line = _sma20_master_line(df, sma, direction, slope) if direction != "FLAT" else None

    family["sma20_direction"] = direction
    family["sma20_slope"] = float(slope)

    if line is not None:
        family["master_trendline"] = line
        family["master_role"] = line["kind"]
        if line["kind"] == "support":
            family["direction"] = "BUY"
        else:
            family["direction"] = "SELL"
        family["reasons"] = list(family.get("reasons") or []) + [
            f"20 SMA median-price {direction.lower()} — {line['kind']} master trendline follows SMA slope "
            f"(independent of the pivot-fit line above).",
        ]
    else:
        family["master_trendline"] = None
        family["master_role"] = "none"
        if direction == "FLAT":
            family["reasons"] = list(family.get("reasons") or []) + [
                "20 SMA is flat -- no master-trendline reading; pivot-fit geometry above still applies."
            ]

    pullback = _trendline_pullback_state(family)
    family["pullback_entry"] = pullback
    family["pullback_zone"] = pullback.get("zone")
    family["pullback_entry_price"] = pullback.get("entry_price")
    family["pullback_distance_atr"] = pullback.get("distance_atr")

    if pullback.get("confirmed"):
        family["entry_rules"] = {
            "checks": {"pullback": (True, pullback.get("confirmation"))},
            "passed": 1, "required": 1, "confirmed": True,
            "state": "PULLBACK_ENTRY_CONFIRMED",
            "wait_reason": pullback.get("reason"),
            "confirmation": pullback.get("confirmation"),
            "entry_mode": "MARKET",
            "entry_price": pullback.get("entry_price"),
            "pullback_zone": pullback.get("zone"),
        }
        family["master_entry_ready"] = True
        family["reasons"] = list(family.get("reasons") or []) + [
            f"✅ PULLBACK ENTRY CONFIRMED — {pullback['confirmation']} at master trendline."
        ]
    elif family.get("master_trendline") is not None:
        legacy = _legacy_break_retest_entry(family)
        family["entry_rules"] = {
            "checks": {"trendline": (bool(legacy.get("confirmed")), legacy.get("reason"))},
            "passed": 1 if legacy.get("confirmed") else 0, "required": 1,
            "confirmed": bool(legacy.get("confirmed")),
            "state": legacy.get("state"),
            "wait_reason": legacy.get("reason"),
            "confirmation": legacy.get("confirmation"),
            "entry_mode": legacy.get("entry_mode"),
            "pullback_zone": pullback.get("zone"),
        }
        family["master_entry_ready"] = bool(legacy.get("confirmed"))
        family["reasons"] = list(family.get("reasons") or []) + [
            (("✅ " if legacy.get("confirmed") else "⏳ ") + legacy.get("reason", ""))
        ]
    else:
        family["master_entry_ready"] = False

    return family


def _build_position_container_base(family: Dict[str, Any], atr_mult_sl: float = 1.0) -> Optional[Dict[str, Any]]:
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


def build_position_container(family: Dict[str, Any], atr_mult_sl: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Thin wrapper around _build_position_container_base(): when a master
    20-SMA trendline exists (see _apply_trendline_upgrades), a ticket is
    only valid once entry_rules actually confirms a pullback-rejection or
    break-retest continuation candle -- not on rail/close geometry alone.
    Once confirmed, entry is rebased to the confirmation candle's close
    (rather than the raw geometry entry), and SL/TP1/TP2/TP3 shift by the
    same delta so the risk:reward ratio the geometry computed is preserved
    around the real entry price.
    """
    is_master_trendline_setup = bool(
        family and (family.get("master_trendline") is not None or family.get("trendline_retest") is not None)
    )
    if is_master_trendline_setup and not (family.get("entry_rules") or {}).get("confirmed"):
        return None

    pos = _build_position_container_base(family, atr_mult_sl=atr_mult_sl)
    if not pos or not is_master_trendline_setup:
        return pos

    rules = family.get("entry_rules") or {}
    df = family.get("df")
    if rules.get("confirmed") and df is not None and not df.empty:
        new_entry = rules.get("entry_price")
        if new_entry is None:
            new_entry = float(df["Close"].iloc[-1])
        old_entry = pos.get("entry")
        if old_entry is not None:
            delta = float(new_entry) - float(old_entry)
            for key in ("sl", "tp1", "tp2", "tp3"):
                if pos.get(key) is not None:
                    pos[key] = float(pos[key]) + delta
        pos["entry"] = float(new_entry)
        pos["order_type"] = "MARKET"
        pos["entry_confirmation"] = rules.get("confirmation")
        pos["entry_confirmation_state"] = rules.get("state")
        pos["pullback_zone"] = rules.get("pullback_zone")
        pos["confirmed"] = True
        pos["entry_rules"] = rules
    return pos


def _swing_structure_bias(family: Dict[str, Any], limit: int = 4) -> str:
    """
    Reads the ACTUAL HH/HL/LH/LL sequence and returns what the swing
    structure itself says (BULLISH/BEARISH/MIXED) -- independent of
    whatever the trendline geometry decided the trade direction is.
    This is what the 'Structure:' line should reflect; mirroring the
    trade bias instead (the old bug) let the report show a LH -> LL
    sequence labeled BULLISH, which is self-contradictory.
    """
    anns = [a for a in (family.get("trendline_annotations") or [])
            if str(a.get("label")) in {"HH", "HL", "LH", "LL"}]
    labels = [str(a["label"]) for a in anns[-limit:]]
    if not labels:
        return "NEUTRAL"
    bull_votes = sum(1 for l in labels if l in ("HH", "HL"))
    bear_votes = sum(1 for l in labels if l in ("LH", "LL"))
    # Weight the most recent label more heavily -- it's the current state.
    last = labels[-1]
    if last in ("HH", "HL"):
        bull_votes += 1
    else:
        bear_votes += 1
    if bull_votes > bear_votes:
        return "BULLISH"
    if bear_votes > bull_votes:
        return "BEARISH"
    return "MIXED"


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

    When the pullback-entry adapter (_apply_trendline_upgrades) is still
    waiting on something, its `wait_reason` is a more accurate "what
    happens next" than the generic entry_rules text below, and is spliced
    into the WAIT FOR section when present.
    """
    report = _format_trendline_report_base(family, symbol)
    rules = (family or {}).get("entry_rules") or {}
    reason = rules.get("wait_reason")
    if reason and "WAIT FOR:" in report and not rules.get("confirmed"):
        head, tail = report.split("WAIT FOR:", 1)
        suffix = ""
        marker = "\nNo trade yet."
        if marker in tail:
            _, end = tail.split(marker, 1)
            suffix = marker + end
        return head + "WAIT FOR:\n1. " + reason + suffix
    return report


def _format_trendline_report_base(family: Dict[str, Any], symbol: str) -> str:
    if family.get("error"):
        return family["error"]

    topdown = family.get("topdown") or {}
    tf_label = family.get("timeframe_label") or "M30"
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
    trade_bias = "BULLISH" if bias_30 == "BUY" else "BEARISH" if bias_30 == "SELL" else "NEUTRAL"
    structure_bias = _swing_structure_bias(family)
    structure_conflict = (
        trade_bias != "NEUTRAL" and structure_bias not in ("NEUTRAL", "MIXED") and structure_bias != trade_bias
    )

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

    sma_dir = family.get("sma_direction", "FLAT")
    sma_icon = "🟢" if sma_dir == "RISING" else ("🔴" if sma_dir == "FALLING" else "⚪")
    sma_last = family.get("sma_last")
    sc = family.get("sma_confluence") or {}
    color_state = family.get("trendline_color_state", "NEUTRAL")
    color_icon = "🟢" if color_state == "BULLISH" else ("🔴" if color_state == "BEARISH" else "⚪")

    setup_scores = family.get("setup_scores") or {}
    active_setup = family.get("active_setup", "TRENDLINE")
    active_conf = family.get("active_setup_confidence", 0)

    lines = [
        f"📐 ANALYSIS — {symbol} {tf_label}",
        "",
        "🔎 SETUP SCAN",
    ]
    for name in ("TRENDLINE", "PATTERN", "S/R"):
        marker = "👉" if name == active_setup else "  "
        suffix = ""
        if name == "TRENDLINE" and family.get("trendline_is_pattern_rail"):
            suffix = "  (this rail IS the pattern below, not a separate setup)"
        lines.append(f"{marker} {name}: {setup_scores.get(name, 0)}%{suffix}")
    if active_setup == "NONE":
        lines.append("BEST SETUP: NONE — no setup cleared minimum confidence, market is choppy/ranging")
    else:
        lines.append(f"BEST SETUP: {active_setup} ({active_conf}%)")

    sr = family.get("sr_setup")
    if active_setup == "S/R" and sr:
        lvl = sr["level"]
        lines += [
            "",
            "S/R ZONE",
            f"Level: {float(lvl['price']):.5f} ({str(lvl.get('side','level')).upper()})",
            f"Touches: {int(lvl.get('touches',0))}  |  Quality: {str(lvl.get('quality','')).upper()}",
            f"Distance from price: {sr['distance_atr']} ATR",
            f"Reaction bias: {sr['bias']}",
        ]

    lines += [
        "",
        "BIAS",
    ]
    if topdown:
        lines += [f"4H: {bias_4h}", f"1H: {bias_1h}"]
    lines.append(f"{tf_label}: {bias_30}")
    lines += [
        "",
        "20 SMA",
        f"Applied price: MEDIAN PRICE",
        f"Period: {family.get('sma_period', 20)}",
        f"Direction: {sma_icon} {sma_dir}",
        f"Value: {sma_last:.5f}" if sma_last is not None else "Value: —",
    ]
    if topdown:
        lines += [
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
        "MARKET STRUCTURE",
        f"Trendline: {color_icon} {color_state}",
        f"Type: {primary_kind if primary_kind != 'NONE' else 'NONE'}",
        f"Touches: {touches}",
        f"Validation: {validation_text}",
        f"Status: {'INTACT' if lifecycle['status'] == 'INTACT' else lifecycle['breakout']}",
        "",
        "20 SMA ↔ TRENDLINE",
        f"Relationship: {sc.get('relationship', 'N/A')}",
        f"Distance: {sc.get('distance_atr')} ATR" if sc.get("distance_atr") is not None else "Distance: —",
        f"Status: {sc.get('status', 'UNKNOWN')}",
        f"Directional strength: {sc.get('strength', 'N/A')}",
        "",
        "STRUCTURE",
        f"{structure}",
        f"Structure: {structure_bias}" + (f"  ⚠ conflicts with trade bias ({trade_bias})" if structure_conflict else ""),
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

    # Visual pattern block (wedge/triangle) -- only shown when it's the
    # active, meaningful pattern on this chart, matching the sample
    # '📐 VISUAL PATTERN' geometry/confidence/breakout/retest schema.
    pv = family.get("pattern_visual")
    if pv and family.get("active_pattern") == "wedge":
        lines += [
            "",
            "━━━━━━━━━━━━━━━━",
            "",
            "📐 VISUAL PATTERN",
            pv["pattern_name"].upper(),
            "",
            f"Confidence: {pv['confidence']}%",
            "",
            "Geometry:",
            f"• Upper rail: {pv['upper_dir']}",
            f"• Lower rail: {pv['lower_dir']}",
            "• Rails: converging",
            f"• Upper touches: {pv['upper_touches']}",
            f"• Lower touches: {pv['lower_touches']}",
            f"• Fit quality: {pv['fit_quality']}%",
            "",
            f"BREAKOUT: {pv['breakout_status']}",
            f"RETEST: {pv['retest_status']}",
            "",
            f"🎯 FINAL CONFIDENCE: {pv['final_confidence']}%",
            f"ENTRY: {pv['bias']} {pv['entry_status']}" if pv["bias"] != "NEUTRAL" else f"ENTRY: {pv['entry_status']}",
        ]

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
# TRENDLINE STRATEGY ORCHESTRATION -- full top-down cascade:
#   4H bias (EMA200 + structure) -> 1H structure permission -> 30M entry
# The 30M trendline family supplies the entry/SL/TP geometry; the 4H/1H
# read gates and scores it (see unified_strategy.get_topdown_bias).
# ============================================================

def run_trendline_analysis(symbol: str, tf_code: str = "30min", topdown: Optional[Dict[str, Any]] = None,
                            df=None) -> Dict[str, Any]:
    """Runs entirely on the single selected timeframe (tf_code). Top-down
    HTF context is OPTIONAL -- pass a pre-fetched topdown_engine.get_topdown_bias()
    dict in explicitly (e.g. from a separate "HTF Context" button) if you want
    it folded into the gating notes below; otherwise it's skipped and the
    trendline read stands on its own.

    Pass `df` (e.g. an EA's own freshly-pushed candles) to skip the internal
    fetch and analyze that data directly -- keeps live execution and this
    report reading off the exact same bars instead of two independent pulls.
    """
    df_tf = df if df is not None else market_data.fetch_candles(symbol, tf_code, count=250)
    tf_label = {"1min": "M1", "3min": "M3", "5min": "M5", "15min": "M15",
                "30min": "M30", "1h": "H1", "4h": "H4"}.get(tf_code, tf_code)
    if df_tf is None or df_tf.empty or len(df_tf) < 30:
        return {
            "error": f"Insufficient {tf_label} data for Trendline analysis",
            "direction": "NEUTRAL", "symbol": symbol, "topdown": topdown,
        }

    family = build_trendline_family(df_tf, max_lines=4, lookback_bars=120)
    family["symbol"] = symbol
    family["timeframe"] = tf_code
    family["timeframe_label"] = tf_label
    family["topdown"] = topdown
    if topdown:
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
        # Share the SAME structural pivots the chart's HH/HL/LH/LL and
        # trendlines are built from, instead of letting the pattern scanner
        # run its own independent swing detection. Two different swing
        # engines on one chart is how a Double Top ends up drawn on minor
        # pullback highs while the structure line shows a bigger untested
        # swing high right next to it.
        best_pattern, all_patterns = scan_all_patterns(
            df_tf, pivots=family.get("pivots_full") or family.get("pivots"),
            sma=family.get("sma20_series"),
        )
        family["scanned_pattern"] = best_pattern.to_dict() if best_pattern else None
        family["scanned_patterns"] = [p.to_dict() for p in all_patterns]
    except Exception as e:
        print(f"[run_trendline_analysis] pattern scan failed for {symbol}: {e!r}")
        family["scanned_pattern"] = None
        family["scanned_patterns"] = []

    # Surface WHY no pattern is being shown when price has already left the
    # 20-SMA chop and established a sloping trend -- otherwise "no pattern"
    # reads as the scanner having failed rather than as the expected state
    # ("no pattern, because there's a trend" is itself the read).
    try:
        is_ranging, ranging_reason = is_price_ranging_vs_sma(df_tf, sma=family.get("sma20_series"))
        family["sma_ranging"] = is_ranging
        family["sma_ranging_reason"] = ranging_reason
        if not is_ranging and not family.get("scanned_pattern"):
            family.setdefault("reasons", []).append(
                f"No live pattern — price has left the 20SMA chop and the SMA is trending "
                f"({ranging_reason}). Any earlier shape is already complete/broken; this is a "
                f"trend leg now, not a forming pattern."
            )
    except Exception as e:
        print(f"[run_trendline_analysis] ranging check failed for {symbol}: {e!r}")
        family["sma_ranging"] = None
        family["sma_ranging_reason"] = None

    direction = family.get("direction", "NEUTRAL")
    strength = family.get("strength", 0)
    td_dir = topdown.get("direction", "NEUTRAL") if topdown else "NEUTRAL"
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
    close = float(df_tf["Close"].iloc[-1])
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
    # Triangles/wedges get the exact same neckline-stage treatment as the
    # classic reversal patterns -- a wedge is just as capable of being an
    # unbroken shape as a Double Top is, and must not be allowed to move
    # direction/strength until its own rail has actually broken with a
    # confirmation candle (see classify_pattern_stage).
    stage_gated_names = reversal_names | {
        "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle",
        "Rising Wedge", "Falling Wedge",
    }
    if sp and sp.get("name") in stage_gated_names:
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

    # Triangle/wedge/rectangle names, classified by market_analysis.py's
    # detect_triangle_or_wedge / detect_rectangle to match the standard
    # trendline-pattern taxonomy (ascending/descending triangle, rising/
    # falling wedge, symmetrical triangle). These already carry the correct
    # bias, the FORMING/TRIGGERED/CONFIRMED/FAKEOUT stage gate, and the
    # marubozu/engulfing confirmation-candle check via classify_pattern_stage
    # -- unlike the older independent-slope wedge fit (_detect_converging_
    # wedge / family["wedge"]) which has none of that. Whenever one of these
    # is present, it must be the chart's SINGLE source of truth for the
    # drawn geometry and for direction -- never the older ungated fit.
    WEDGE_TRIANGLE_NAMES = (
        "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle",
        "Rising Wedge", "Falling Wedge",
    )
    if sp and sp.get("name") in WEDGE_TRIANGLE_NAMES:
        # Always let the properly-gated scanned pattern win the chart over
        # the legacy ungated wedge fit, regardless of whether it happens to
        # agree with the direction already on the board.
        family["active_pattern"] = "scanned"
        family["pattern_confidence"] = max(int(family.get("pattern_confidence") or 0),
                                            int(sp.get("confidence") or 0))

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
            sp_name in ("Bull Flag", "Bullish Pennant", "Ascending Triangle", "Ascending Channel",
                        "Falling Wedge")
            and sp_bias == "BUY" and sp_conf >= 68
        )
        is_bearish_cont = (
            sp_name in ("Bear Flag", "Bearish Pennant", "Descending Triangle", "Descending Channel",
                        "Rising Wedge")
            and sp_bias == "SELL" and sp_conf >= 68
        )
        # A triangle/wedge that hasn't broken its own trigger rail yet
        # (stage FORMING) is a shape, not a signal -- same rule as the
        # reversal patterns above. Only let it move direction once there's
        # a real break (TRIGGERED) confirmed by a marubozu/engulfing candle,
        # or a break-and-hold retest (CONFIRMED).
        if sp_name in ("Ascending Triangle", "Descending Triangle", "Symmetrical Triangle",
                       "Rising Wedge", "Falling Wedge"):
            cont_stage = str(family.get("pattern_stage") or "").upper()
            if cont_stage not in ("TRIGGERED", "CONFIRMED"):
                if is_bullish_cont or is_bearish_cont:
                    reasons.append(
                        f"⏳ {sp_name} ({sp_conf:.0f}%) still FORMING — rail not broken with "
                        f"confirmation candle yet, cannot set direction"
                    )
                is_bullish_cont = False
                is_bearish_cont = False

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

        # A reversal pattern that hasn't even broken its neckline (stage
        # FORMING) is a shape, not a signal -- it must never be allowed to
        # flip the trade direction. Only TRIGGERED/CONFIRMED reversals may
        # override below. FORMING stays visible in CONFLUENCE as a watch
        # item (handled further down) but cannot drive BIAS.
        reversal_stage = str(family.get("pattern_stage") or "").upper()
        reversal_confirmed_enough = reversal_stage in ("TRIGGERED", "CONFIRMED")
        if (is_strong_bullish_rev or is_strong_bearish_rev) and not reversal_confirmed_enough:
            reasons.append(
                f"⏳ {sp_name} ({sp_conf:.0f}%) still FORMING — neckline not broken, "
                f"cannot override current bias yet"
            )
            is_strong_bullish_rev = False
            is_strong_bearish_rev = False

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
    # with lagging higher-timeframe direction. HTF context is now OPTIONAL --
    # only applied when the caller explicitly passed a `topdown` dict in
    # (e.g. from a separate "HTF Context" button), otherwise skipped entirely.
    if direction in ("BUY", "SELL") and topdown:
        if td_dir == direction and topdown.get("allowed"):
            strength = min(100, strength + 12)
            gating_notes.append(
                f"✅ Short-term trend ({direction}) aligned with HTF top-down ({td_dir})"
            )
        elif td_dir == direction and not topdown.get("allowed"):
            gating_notes.append(
                f"Short-term trend ({direction}) matches top-down direction but HTF permission "
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

    # Recompute the trendline's dynamic color state with the FINAL resolved
    # direction (pattern/OB/sequence logic above can change it from what
    # build_trendline_family saw initially).
    family["trendline_color_state"] = _trendline_color_state(direction, family.get("sma_direction", "FLAT"))

    # Best-setup scan: score Trendline vs Pattern vs S/R and let whichever
    # is actually strongest right now drive the report/chart, instead of
    # always framing everything through the trendline lens.
    family = select_best_setup(family, df_tf)

    # ------------------------------------------------------------------
    # Reconciliation: BIAS must agree with whichever setup the scan just
    # crowned the winner. Without this, SETUP SCAN could hand TRENDLINE
    # an 80% lead while BIAS still carried whatever a lower-scoring,
    # unconfirmed pattern decided earlier -- a chart/report that
    # contradicts itself. If TRENDLINE wins by a clear margin, is INTACT
    # (not broken), and the pattern that disagrees hasn't even confirmed
    # (no FORMING pattern gets veto power), direction snaps back to what
    # the winning trendline actually implies: ascending = BUY-on-retest,
    # descending = SELL-on-retest.
    # ------------------------------------------------------------------
    scores = family.get("setup_scores") or {}
    active_setup = family.get("active_setup")
    trendline_lifecycle = _trendline_status_text(family)
    pattern_stage_now = str(family.get("pattern_stage") or "").upper()
    if (
        active_setup == "TRENDLINE"
        and trendline_lifecycle.get("status") == "INTACT"
        and scores.get("TRENDLINE", 0) - scores.get("PATTERN", 0) >= 20
        and pattern_stage_now not in ("TRIGGERED", "CONFIRMED")
    ):
        trend_kind = family.get("family_kind", "none")
        implied_dir = "BUY" if trend_kind == "ascending" else "SELL" if trend_kind == "descending" else None
        if implied_dir and direction != implied_dir:
            old_dir = direction
            reasons.append(
                f"↩ Reconciled: TRENDLINE ({scores.get('TRENDLINE',0)}%) dominates unconfirmed "
                f"{family.get('pattern_label') or 'pattern'} ({scores.get('PATTERN',0)}%) — "
                f"bias follows trendline continuation ({old_dir} → {implied_dir})"
            )
            direction = implied_dir
            family["direction"] = direction
            family["trendline_color_state"] = _trendline_color_state(direction, family.get("sma_direction", "FLAT"))
            family["continuation_state"] = _classify_trendline_state(family)

    family["strength"] = max(0, min(100, int(strength)))
    family["gating_notes"] = gating_notes
    family["reasons"] = reasons
    family["short_term_signal"] = direction  # explicit short-term read for reports
    return family
