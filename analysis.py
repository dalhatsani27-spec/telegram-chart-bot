"""
analysis.py — patterns, channels, SMC zones, AMD, Silver Bullet, institutional top-down.
"""
from __future__ import annotations
import data as mt5_data
"""
patterns.py
================
Professional-grade price-action pattern detection engine.

Design goals (per spec):
  - Detect swing highs/lows the way a discretionary trader would (fractal pivots).
  - Scan for ALL common chart patterns, not just one or two:
        Reversal:      Head & Shoulders, Inverse H&S, Double Top/Bottom,
                        Triple Top/Bottom, Rising Wedge, Falling Wedge
        Continuation:  Bull Flag, Bear Flag, Pennant, Ascending/Descending/
                        Symmetrical Triangle, Rectangle/Range
  - Every detected pattern carries a "trigger line" (neckline / trendline /
    flag channel line) so the bot can plot exactly what to watch.
  - Flags and continuation/reversal patterns are weighted higher because they
    give the cleanest directional read, per the user's requirement.
  - If nothing genuinely qualifies, return nothing rather than forcing a label.

This module is self-contained and pandas/numpy only (no plotting here).
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. SWING PIVOT DETECTION
# ----------------------------------------------------------------------------
def find_pivots(df, left=3, right=3):
    """
    Fractal pivot detection: a bar is a pivot HIGH if its High is the max of
    the (left + 1 + right) window centered on it; pivot LOW analogous on Lows.

    Returns two lists of integer indices (positions, not timestamps):
        pivot_highs, pivot_lows
    """
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)
    pivot_highs, pivot_lows = [], []

    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        if highs[i] == window_h.max() and np.argmax(window_h) == left:
            pivot_highs.append(i)
        window_l = lows[i - left:i + right + 1]
        if lows[i] == window_l.min() and np.argmin(window_l) == left:
            pivot_lows.append(i)

    return pivot_highs, pivot_lows


def _dedupe_adjacent(pivots, min_gap):
    """Collapse pivots that are within min_gap bars of each other, keeping the first."""
    if not pivots:
        return pivots
    out = [pivots[0]]
    for p in pivots[1:]:
        if p - out[-1] >= min_gap:
            out.append(p)
    return out


# ----------------------------------------------------------------------------
# 2. SHARED HELPERS
# ----------------------------------------------------------------------------
def _line_through(p1, p2):
    """Return (slope, intercept) of the line through two (x, y) points."""
    (x1, y1), (x2, y2) = p1, p2
    if x2 == x1:
        return 0.0, y1
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def _pct(a, b):
    """Percent difference of a relative to b."""
    if b == 0:
        return 0.0
    return (a - b) / abs(b)


def _atr(df):
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        return float(df['ATR'].iloc[-1])
    return float((df['High'] - df['Low']).tail(14).mean())


class Pattern:
    """Container for a detected pattern, with everything needed to render it."""
    def __init__(self, name, category, bias, trigger_price, trigger_line,
                 key_points, confidence, note):
        self.name = name                 # display name
        self.category = category         # 'reversal' | 'continuation'
        self.bias = bias                 # 'BUY' | 'SELL'
        self.trigger_price = trigger_price   # single price level to watch (breakout/neckline)
        self.trigger_line = trigger_line     # list of (x, y) points describing the line to draw (2+ pts)
        self.key_points = key_points          # list of (x, y, label) marker points to draw
        self.confidence = confidence      # 0-100 raw detector confidence
        self.note = note                  # human-readable rationale

    def to_dict(self):
        return {
            "name": self.name, "category": self.category, "bias": self.bias,
            "trigger_price": self.trigger_price, "trigger_line": self.trigger_line,
            "key_points": self.key_points, "confidence": self.confidence, "note": self.note,
        }


# ----------------------------------------------------------------------------
# 3. INDIVIDUAL PATTERN DETECTORS
#    Each takes (df, pivot_highs, pivot_lows) and returns a Pattern or None.
# ----------------------------------------------------------------------------

def detect_double_top(df, ph, pl, min_bars=8, min_depth_atr=1.0, max_peak_diff=0.006):
    """
    Stricter Double Top for lower noise (especially on 30m).
    - Peaks within max_peak_diff (~0.6%)
    - At least min_bars between the two tops
    - Trough depth >= min_depth_atr * ATR
    """
    if len(ph) < 2:
        return None
    i2, i1 = ph[-1], ph[-2]
    if (i2 - i1) < min_bars:
        return None
    h1, h2 = float(df['High'].iloc[i1]), float(df['High'].iloc[i2])
    if abs(_pct(h2, h1)) > max_peak_diff:
        return None
    between_lows = [p for p in pl if i1 < p < i2]
    if not between_lows:
        return None
    trough_i = min(between_lows, key=lambda p: df['Low'].iloc[p])
    neckline = float(df['Low'].iloc[trough_i])
    atr = _atr(df) or 1e-9
    depth = max(h1, h2) - neckline
    if depth < min_depth_atr * atr:
        return None
    current = float(df['Close'].iloc[-1])
    if current > max(h1, h2):
        return None
    equality_bonus = min(12, (1 - abs(_pct(h2, h1)) * 100) * 8)
    depth_bonus = min(10, (depth / atr - min_depth_atr) * 4)
    conf = 58 + equality_bonus + depth_bonus
    return Pattern(
        "Double Top", "reversal", "SELL",
        trigger_price=neckline,
        trigger_line=[(i1, neckline), (i2, neckline)],
        key_points=[(i1, h1, "Top 1"), (i2, h2, "Top 2"), (trough_i, neckline, "Neckline")],
        confidence=float(np.clip(conf, 55, 88)),
        note=(f"Two near-equal highs ({h1:.5f} / {h2:.5f}) separated by {i2 - i1} bars. "
              f"Neckline {neckline:.5f} (depth {depth / atr:.1f}×ATR). "
              f"Close below neckline confirms breakdown.")
    )


def detect_double_bottom(df, ph, pl, min_bars=8, min_depth_atr=1.0, max_peak_diff=0.006):
    """
    Stricter Double Bottom for lower noise (especially on 30m).
    - Bottoms within max_peak_diff (~0.6%)
    - At least min_bars between the two bottoms
    - Peak height >= min_depth_atr * ATR
    """
    if len(pl) < 2:
        return None
    i2, i1 = pl[-1], pl[-2]
    if (i2 - i1) < min_bars:
        return None
    l1, l2 = float(df['Low'].iloc[i1]), float(df['Low'].iloc[i2])
    if abs(_pct(l2, l1)) > max_peak_diff:
        return None
    between_highs = [p for p in ph if i1 < p < i2]
    if not between_highs:
        return None
    peak_i = max(between_highs, key=lambda p: df['High'].iloc[p])
    neckline = float(df['High'].iloc[peak_i])
    atr = _atr(df) or 1e-9
    height = neckline - min(l1, l2)
    if height < min_depth_atr * atr:
        return None
    current = float(df['Close'].iloc[-1])
    if current < min(l1, l2):
        return None
    equality_bonus = min(12, (1 - abs(_pct(l2, l1)) * 100) * 8)
    depth_bonus = min(10, (height / atr - min_depth_atr) * 4)
    conf = 58 + equality_bonus + depth_bonus
    return Pattern(
        "Double Bottom", "reversal", "BUY",
        trigger_price=neckline,
        trigger_line=[(i1, neckline), (i2, neckline)],
        key_points=[(i1, l1, "Bottom 1"), (i2, l2, "Bottom 2"), (peak_i, neckline, "Neckline")],
        confidence=float(np.clip(conf, 55, 88)),
        note=(f"Two near-equal lows ({l1:.5f} / {l2:.5f}) separated by {i2 - i1} bars. "
              f"Neckline {neckline:.5f} (height {height / atr:.1f}×ATR). "
              f"Close above neckline confirms breakout.")
    )


def detect_triple_top(df, ph, pl):
    if len(ph) < 3:
        return None
    i1, i2, i3 = ph[-3], ph[-2], ph[-1]
    h1, h2, h3 = df['High'].iloc[i1], df['High'].iloc[i2], df['High'].iloc[i3]
    tops = [h1, h2, h3]
    if (max(tops) - min(tops)) / max(tops) > 0.008:
        return None
    between = [p for p in pl if i1 < p < i3]
    if not between:
        return None
    trough_i = min(between, key=lambda p: df['Low'].iloc[p])
    neckline = float(df['Low'].iloc[trough_i])
    current = float(df['Close'].iloc[-1])
    if current > max(tops):
        return None
    return Pattern(
        "Triple Top", "reversal", "SELL",
        trigger_price=neckline,
        trigger_line=[(i1, neckline), (i3, neckline)],
        key_points=[(i1, h1, "Top 1"), (i2, h2, "Top 2"), (i3, h3, "Top 3"), (trough_i, neckline, "Neckline")],
        confidence=68,
        note=f"Three tests of resistance near {max(tops):.5f} rejected. Neckline at {neckline:.5f}."
    )


def detect_triple_bottom(df, ph, pl):
    if len(pl) < 3:
        return None
    i1, i2, i3 = pl[-3], pl[-2], pl[-1]
    l1, l2, l3 = df['Low'].iloc[i1], df['Low'].iloc[i2], df['Low'].iloc[i3]
    bots = [l1, l2, l3]
    if (max(bots) - min(bots)) / max(bots) > 0.008:
        return None
    between = [p for p in ph if i1 < p < i3]
    if not between:
        return None
    peak_i = max(between, key=lambda p: df['High'].iloc[p])
    neckline = float(df['High'].iloc[peak_i])
    current = float(df['Close'].iloc[-1])
    if current < min(bots):
        return None
    return Pattern(
        "Triple Bottom", "reversal", "BUY",
        trigger_price=neckline,
        trigger_line=[(i1, neckline), (i3, neckline)],
        key_points=[(i1, l1, "Bottom 1"), (i2, l2, "Bottom 2"), (i3, l3, "Bottom 3"), (peak_i, neckline, "Neckline")],
        confidence=68,
        note=f"Three tests of support near {min(bots):.5f} held. Neckline at {neckline:.5f}."
    )


def detect_head_shoulders(df, ph, pl):
    if len(ph) < 3:
        return None
    i1, i2, i3 = ph[-3], ph[-2], ph[-1]
    ls, head, rs = df['High'].iloc[i1], df['High'].iloc[i2], df['High'].iloc[i3]
    if not (head > ls and head > rs):
        return None
    if abs(_pct(rs, ls)) > 0.03:  # shoulders should be roughly symmetric
        return None
    between1 = [p for p in pl if i1 < p < i2]
    between2 = [p for p in pl if i2 < p < i3]
    if not between1 or not between2:
        return None
    t1 = min(between1, key=lambda p: df['Low'].iloc[p])
    t2 = min(between2, key=lambda p: df['Low'].iloc[p])
    slope, intercept = _line_through((t1, df['Low'].iloc[t1]), (t2, df['Low'].iloc[t2]))
    neckline_now = slope * (len(df) - 1) + intercept
    current = float(df['Close'].iloc[-1])
    if current < neckline_now * 0.99:
        return None  # already broke down further back, stale
    return Pattern(
        "Head and Shoulders", "reversal", "SELL",
        trigger_price=float(neckline_now),
        trigger_line=[(t1, float(df['Low'].iloc[t1])), (t2, float(df['Low'].iloc[t2]))],
        key_points=[(i1, ls, "L Shoulder"), (i2, head, "Head"), (i3, rs, "R Shoulder")],
        confidence=72,
        note=f"Classic H&S: head {head:.5f} above shoulders {ls:.5f}/{rs:.5f}. "
             f"Neckline (sloped) currently ~{neckline_now:.5f} — close below confirms."
    )


def detect_inverse_head_shoulders(df, ph, pl):
    if len(pl) < 3:
        return None
    i1, i2, i3 = pl[-3], pl[-2], pl[-1]
    ls, head, rs = df['Low'].iloc[i1], df['Low'].iloc[i2], df['Low'].iloc[i3]
    if not (head < ls and head < rs):
        return None
    if abs(_pct(rs, ls)) > 0.03:
        return None
    between1 = [p for p in ph if i1 < p < i2]
    between2 = [p for p in ph if i2 < p < i3]
    if not between1 or not between2:
        return None
    t1 = max(between1, key=lambda p: df['High'].iloc[p])
    t2 = max(between2, key=lambda p: df['High'].iloc[p])
    slope, intercept = _line_through((t1, df['High'].iloc[t1]), (t2, df['High'].iloc[t2]))
    neckline_now = slope * (len(df) - 1) + intercept
    current = float(df['Close'].iloc[-1])
    if current > neckline_now * 1.01:
        return None
    return Pattern(
        "Inverse Head and Shoulders", "reversal", "BUY",
        trigger_price=float(neckline_now),
        trigger_line=[(t1, float(df['High'].iloc[t1])), (t2, float(df['High'].iloc[t2]))],
        key_points=[(i1, ls, "L Shoulder"), (i2, head, "Head"), (i3, rs, "R Shoulder")],
        confidence=72,
        note=f"Inverse H&S: head {head:.5f} below shoulders {ls:.5f}/{rs:.5f}. "
             f"Neckline (sloped) currently ~{neckline_now:.5f} — close above confirms."
    )


def _fit_trend(points):
    """points: list of (x, y). Returns slope, intercept via least squares."""
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if len(xs) < 2 or np.all(xs == xs[0]):
        return 0.0, float(ys[-1]) if len(ys) else 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _touch_quality_score(points, slope, intercept, avg_price):
    """
    How tightly the given points actually hug their fitted line, as a
    confidence bonus (0-8). A triangle/wedge can have the "right" slope
    pattern by the numbers while price barely respects either boundary --
    this distinguishes a real, well-defended structure from a coincidental
    one. Tighter fit (lower deviation relative to price scale) -> higher bonus.
    """
    if len(points) < 2 or avg_price <= 0:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    fitted = slope * xs + intercept
    deviations = np.abs(ys - fitted)
    rms = float(np.sqrt(np.mean(deviations ** 2)))
    normalized_rms = rms / avg_price
    bonus = 8.0 - normalized_rms * 3000.0
    return float(np.clip(bonus, 0.0, 8.0))


def detect_triangle_or_wedge(df, ph, pl, lookback=60):
    """
    Uses the last several pivot highs (upper boundary) and pivot lows (lower
    boundary) within `lookback` bars to fit two trendlines, then classifies:
        - both flat-ish, converging  -> not used here (handled by rectangle)
        - upper flat, lower rising   -> Ascending Triangle (bullish continuation)
        - upper falling, lower flat  -> Descending Triangle (bearish continuation)
        - both converging, opposite slopes -> Symmetrical Triangle
        - both rising, converging    -> Rising Wedge (bearish reversal)
        - both falling, converging   -> Falling Wedge (bullish reversal)
    """
    n = len(df)
    start = max(0, n - lookback)
    recent_ph = [p for p in ph if p >= start][-4:]
    recent_pl = [p for p in pl if p >= start][-4:]
    if len(recent_ph) < 2 or len(recent_pl) < 2:
        return None

    upper_pts = [(p, float(df['High'].iloc[p])) for p in recent_ph]
    lower_pts = [(p, float(df['Low'].iloc[p])) for p in recent_pl]
    up_slope, up_intercept = _fit_trend(upper_pts)
    lo_slope, lo_intercept = _fit_trend(lower_pts)

    avg_price = float(df['Close'].tail(lookback).mean()) or 1.0
    touch_quality_bonus = (_touch_quality_score(upper_pts, up_slope, up_intercept, avg_price) +
                           _touch_quality_score(lower_pts, lo_slope, lo_intercept, avg_price)) / 2.0
    up_norm = (up_slope * lookback) / avg_price
    lo_norm = (lo_slope * lookback) / avg_price
    FLAT = 0.003   # ~0.3% drift over the window counts as "flat"

    x_now = n - 1
    upper_now = up_slope * x_now + up_intercept
    lower_now = lo_slope * x_now + lo_intercept
    if upper_now <= lower_now:
        return None  # lines already crossed, pattern played out

    current = float(df['Close'].iloc[-1])
    line = [(upper_pts[0][0], upper_pts[0][1]), (upper_pts[-1][0], upper_pts[-1][1])]
    lower_line = [(lower_pts[0][0], lower_pts[0][1]), (lower_pts[-1][0], lower_pts[-1][1])]

    # Ascending triangle: flat top, rising bottom -> bullish continuation
    if abs(up_norm) < FLAT and lo_norm > FLAT:
        return Pattern(
            "Ascending Triangle", "continuation", "BUY",
            trigger_price=float(upper_now),
            trigger_line=line,
            key_points=[(p, y, "Resistance") for p, y in upper_pts] + [(p, y, "Higher Low") for p, y in lower_pts],
            confidence=65 + touch_quality_bonus,
            note=f"Flat resistance near {upper_now:.5f} with rising higher-lows underneath — "
                 f"buyers stepping in earlier each time. Breakout above the flat top favors continuation up."
        )
    # Descending triangle: falling top, flat bottom -> bearish continuation
    if up_norm < -FLAT and abs(lo_norm) < FLAT:
        return Pattern(
            "Descending Triangle", "continuation", "SELL",
            trigger_price=float(lower_now),
            trigger_line=lower_line,
            key_points=[(p, y, "Support") for p, y in lower_pts] + [(p, y, "Lower High") for p, y in upper_pts],
            confidence=65 + touch_quality_bonus,
            note=f"Flat support near {lower_now:.5f} with falling lower-highs above — "
                 f"sellers stepping in earlier each time. Breakdown below flat support favors continuation down."
        )
    # Rising wedge: both rising, converging, upper rising slower -> bearish reversal
    if up_norm > FLAT and lo_norm > FLAT and lo_norm > up_norm:
        return Pattern(
            "Rising Wedge", "reversal", "SELL",
            trigger_price=float(lower_now),
            trigger_line=lower_line,
            key_points=[(p, y, "Upper") for p, y in upper_pts] + [(p, y, "Lower") for p, y in lower_pts],
            confidence=63 + touch_quality_bonus,
            note="Both boundaries rising but converging (upper line losing steam faster) — "
                 "classic exhaustion structure. Break of the rising lower trendline signals reversal down."
        )
    # Falling wedge: both falling, converging, lower falling slower -> bullish reversal
    if up_norm < -FLAT and lo_norm < -FLAT and up_norm < lo_norm:
        return Pattern(
            "Falling Wedge", "reversal", "BUY",
            trigger_price=float(upper_now),
            trigger_line=line,
            key_points=[(p, y, "Upper") for p, y in upper_pts] + [(p, y, "Lower") for p, y in lower_pts],
            confidence=63 + touch_quality_bonus,
            note="Both boundaries falling but converging (lower line losing steam faster) — "
                 "selling pressure fading. Break of the falling upper trendline signals reversal up."
        )
    # Symmetrical triangle: opposite slopes converging, roughly equal magnitude
    if up_norm < -FLAT and lo_norm > FLAT:
        bias = "BUY" if current >= (upper_now + lower_now) / 2 else "SELL"
        return Pattern(
            "Symmetrical Triangle", "continuation", bias,
            trigger_price=float(upper_now if bias == "BUY" else lower_now),
            trigger_line=line if bias == "BUY" else lower_line,
            key_points=[(p, y, "Upper") for p, y in upper_pts] + [(p, y, "Lower") for p, y in lower_pts],
            confidence=55 + touch_quality_bonus,
            note="Converging trendlines with contracting range (coiling price action). "
                 "Direction is set by whichever side breaks first — currently leaning "
                 + ("up." if bias == "BUY" else "down.")
        )
    return None


def detect_rectangle(df, ph, pl, lookback=50):
    n = len(df)
    start = max(0, n - lookback)
    recent_ph = [p for p in ph if p >= start]
    recent_pl = [p for p in pl if p >= start]
    if len(recent_ph) < 2 or len(recent_pl) < 2:
        return None
    highs = [df['High'].iloc[p] for p in recent_ph]
    lows = [df['Low'].iloc[p] for p in recent_pl]
    top = float(np.mean(highs))
    bottom = float(np.mean(lows))
    if (max(highs) - min(highs)) / top > 0.01 or (max(lows) - min(lows)) / bottom > 0.01:
        return None  # not flat enough to call a clean range
    if (top - bottom) / top < 0.003:
        return None
    current = float(df['Close'].iloc[-1])
    bias = "BUY" if current <= bottom * 1.003 else ("SELL" if current >= top * 0.997 else None)
    if bias is None:
        return None
    return Pattern(
        "Rectangle / Range", "continuation", bias,
        trigger_price=top if bias == "SELL" else bottom,
        trigger_line=[(recent_ph[0], top), (recent_ph[-1], top)] if bias == "SELL"
                     else [(recent_pl[0], bottom), (recent_pl[-1], bottom)],
        key_points=[(p, df['High'].iloc[p], "Range High") for p in recent_ph] +
                   [(p, df['Low'].iloc[p], "Range Low") for p in recent_pl],
        confidence=52,
        note=f"Price ranging between {bottom:.5f} and {top:.5f}. Currently testing the "
             f"{'top' if bias=='SELL' else 'bottom'} of the range."
    )


def detect_flag_or_pennant(df, lookback_pole=20, lookback_flag=15):
    """
    Bull/Bear flag & pennant detector — weighted highest per user requirement.

    Logic: look for a strong directional "flagpole" move over the last
    ~lookback_pole+lookback_flag bars, then a tight, shallow consolidation
    (the flag/pennant) over the most recent lookback_flag bars that retraces
    only a modest fraction of the pole and slopes counter to (or flat versus)
    the pole direction.
    """
    n = len(df)
    if n < lookback_pole + lookback_flag + 5:
        return None

    flag_start = n - lookback_flag
    pole_start = max(0, flag_start - lookback_pole)

    pole_df = df.iloc[pole_start:flag_start]
    flag_df = df.iloc[flag_start:]

    pole_move = float(pole_df['Close'].iloc[-1] - pole_df['Close'].iloc[0])
    pole_range = float(pole_df['High'].max() - pole_df['Low'].min()) or 1e-9
    atr = _atr(df) or 1e-9

    # Require a genuine impulsive pole: move at least ~3x ATR and directionally clean
    if abs(pole_move) < atr * 3.0:
        return None
    pole_up = pole_move > 0

    # directional cleanliness: fraction of bars closing in the pole's direction
    closes = pole_df['Close'].values
    diffs = np.diff(closes)
    if len(diffs) == 0:
        return None
    clean_frac = np.mean(diffs > 0) if pole_up else np.mean(diffs < 0)
    if clean_frac < 0.55:
        return None

    # Flag: shallow retracement, tight range, and (ideally) sloping against the pole
    flag_x = np.arange(len(flag_df))
    flag_slope, flag_intercept = np.polyfit(flag_x, flag_df['Close'].values, 1) if len(flag_df) >= 2 else (0, flag_df['Close'].iloc[-1])
    flag_range = float(flag_df['High'].max() - flag_df['Low'].min())
    retrace = flag_range / pole_range

    if retrace > 0.65:
        return None  # too deep a pullback to still call it a flag

    flag_norm_slope = (flag_slope * len(flag_df)) / (float(np.mean(flag_df['Close'])) or 1.0)
    counter_slope_ok = (flag_norm_slope < 0.004) if pole_up else (flag_norm_slope > -0.004)
    if not counter_slope_ok:
        return None

    upper = float(flag_df['High'].max())
    lower = float(flag_df['Low'].min())
    is_pennant = retrace < 0.35 and abs(flag_norm_slope) < 0.002  # tight converging = pennant

    name = ("Bull Flag" if pole_up else "Bear Flag") if not is_pennant else ("Bullish Pennant" if pole_up else "Bearish Pennant")
    bias = "BUY" if pole_up else "SELL"
    trigger_price = upper if pole_up else lower
    trigger_line = [(flag_start, upper if pole_up else lower), (n - 1, upper if pole_up else lower)]

    conf = 75 + min(15, clean_frac * 15) - min(10, retrace * 15)
    return Pattern(
        name, "continuation", bias,
        trigger_price=float(trigger_price),
        trigger_line=trigger_line,
        key_points=[(pole_start, float(pole_df['Close'].iloc[0]), "Pole Start"),
                    (flag_start, float(pole_df['Close'].iloc[-1]), "Pole End / Flag Start")],
        confidence=float(np.clip(conf, 55, 92)),
        note=(f"Strong {'bullish' if pole_up else 'bearish'} flagpole ({abs(pole_move):.5f}, "
              f"{clean_frac*100:.0f}% directional bars) followed by a tight "
              f"{retrace*100:.0f}%-retrace consolidation. Watch {trigger_price:.5f} — a break "
              f"in the pole's direction projects continuation roughly equal to the flagpole length.")
    )


# ----------------------------------------------------------------------------
# 4. TOP-LEVEL SCANNER
# ----------------------------------------------------------------------------
# Priority: flags/pennants first (highest-conviction continuation signal per
# user spec), then other continuation patterns, then classic reversals.
_PRIORITY = {
    "Bull Flag": 100, "Bear Flag": 100, "Bullish Pennant": 98, "Bearish Pennant": 98,
    "Ascending Channel": 92, "Descending Channel": 92,
    "Ascending Triangle": 85, "Descending Triangle": 85, "Symmetrical Triangle": 80,
    "Rising Wedge": 75, "Falling Wedge": 75,
    "Head and Shoulders": 72, "Inverse Head and Shoulders": 72,
    "Double Top": 68, "Double Bottom": 68,
    "Triple Top": 66, "Triple Bottom": 66,
    "Rectangle / Range": 50,
}


def scan_all_patterns(df, left=3, right=3, volume_profile=None):
    """
    Runs every detector against the given OHLC dataframe (must have a
    'Close'-indexed reset-friendly integer position order — pass df as-is,
    positions are derived internally).

    volume_profile: optional dict from volume_profile.compute_volume_profile().
    When provided, each detected pattern's confidence is adjusted based on
    whether its trigger/neckline sits at a volume-significant level (Point
    of Control / Value Area) or in a thin, low-activity gap. This is a
    post-detection adjustment layered on top -- it never changes whether a
    pattern is detected, only how much weight its trigger level deserves.

    Returns: (best_pattern_or_None, all_detected_list)
    """
    ph, pl = find_pivots(df, left=left, right=right)
    ph = _dedupe_adjacent(ph, min_gap=left + right)
    pl = _dedupe_adjacent(pl, min_gap=left + right)

    detected = []
    for fn in (detect_flag_or_pennant,):
        try:
            res = fn(df)
        except Exception as e:
            print(f"[patterns] {fn.__name__} raised: {e!r}")
            res = None
        if res:
            detected.append(res)

    for fn in (detect_double_top, detect_double_bottom, detect_triple_top,
               detect_triple_bottom, detect_head_shoulders,
               detect_inverse_head_shoulders):
        try:
            res = fn(df, ph, pl)
        except Exception as e:
            print(f"[patterns] {fn.__name__} raised: {e!r}")
            res = None
        if res:
            detected.append(res)

    for fn in (detect_triangle_or_wedge, detect_rectangle):
        try:
            res = fn(df, ph, pl)
        except Exception as e:
            print(f"[patterns] {fn.__name__} raised: {e!r}")
            res = None
        if res:
            detected.append(res)

    if not detected:
        return None, []

    if volume_profile is not None:
        from data import level_volume_bonus
        for p in detected:
            bonus = level_volume_bonus(volume_profile, p.trigger_price)
            p.confidence = float(np.clip(p.confidence + bonus, 0.0, 100.0))
            if bonus > 0:
                p.note += " Trigger level sits at a high-volume node (POC/Value Area) -- reinforced."
            elif bonus < 0:
                p.note += " Trigger level sits in a thin, low-activity price gap -- treat with extra caution."

    # S/R zone clustering -- self-contained, reuses the pivots already
    # computed above. A trigger sitting on a level touched many times gets
    # weighted higher than one sitting on a level nobody's actually tested.
    # [merged] was: from sr_zones import cluster_sr_zones, zone_strength_bonus
    touch_prices = [df['High'].iloc[i] for i in ph] + [df['Low'].iloc[i] for i in pl]
    zones = cluster_sr_zones(touch_prices)
    for p in detected:
        bonus = zone_strength_bonus(zones, p.trigger_price)
        if bonus > 0:
            p.confidence = float(np.clip(p.confidence + bonus, 0.0, 100.0))
            p.note += f" Trigger aligns with a well-defended S/R zone -- reinforced."

    detected.sort(key=lambda p: (_PRIORITY.get(p.name, 40) + p.confidence), reverse=True)
    return detected[0], detected
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


from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from structure_engine import zigzag_swings
from data import compute_volume_profile


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


def build_trendline_family(df: pd.DataFrame, max_lines: int = 4) -> Dict[str, Any]:
    """
    Build one clean parallel family (ascending OR descending), not both mixed.
    Market reveals direction: price relative to the family rails.
    """
    if df is None or len(df) < 30:
        return {"error": "Insufficient data for trendline family", "direction": "NEUTRAL", "pivots": []}

    n = len(df)
    # Prefer more pivots on lower TFs so channels match hand-drawn structure
    pivots = zigzag_swings(df, depth=3, deviation_atr=0.22)
    if len(pivots) < 5:
        pivots = zigzag_swings(df, depth=3, deviation_atr=0.15)
    if len(pivots) < 4:
        pivots = zigzag_swings(df, depth=2, deviation_atr=0.12)

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
        family_lines = _build_parallel_family(primary, pivots, n, max_members=min(3, max_lines))
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
"""
smc_zones.py
============
Smart Money Concepts zones:

  - Fair Value Gap (FVG) and Inverse FVG (IFVG / mitigated FVG that flips)
  - Order Block (OB) and Breaker Block (mitigated / inverted OB)
  - Inducement (IDM) zones — liquidity that traps traders before the true move

Rules used (ICT-style, practical for algo):

FVG (3-candle imbalance):
  Bullish FVG: candle[i-2].high < candle[i].low   (gap up)
  Bearish FVG: candle[i-2].low  > candle[i].high  (gap down)
  Minimum gap size: fraction of ATR (filters noise)

Order Block:
  Bullish OB: last down-close (or bearish) candle before a strong bullish
              displacement that creates BOS/CHoCH and preferably an FVG
  Bearish OB: last up-close candle before strong bearish displacement
  Body of displacement candle should be meaningful vs ATR

Inducement (IDM):
  Equal highs / equal lows (liquidity pools), or a minor internal swing
  that sits beyond a short-term range and is likely to be raided before
  the real expansion (classic trap for breakout traders).

Mitigation:
  Bullish zone mitigated when price trades down through the zone
  After mitigation, zone can become Breaker / IFVG
"""

import numpy as np
import pandas as pd


def _atr_series(df, period=14):
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        return df['ATR']
    prev_c = df['Close'].shift(1)
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum((df['High'] - prev_c).abs(), (df['Low'] - prev_c).abs()))
    return tr.rolling(period).mean()


def detect_fvgs(df, min_gap_atr=0.15, max_zones=8):
    """
    Detect unfilled and recently mitigated FVGs.
    Returns list of zone dicts (most recent first).
    """
    if df is None or len(df) < 5:
        return []

    atr = _atr_series(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    n = len(df)
    zones = []

    for i in range(2, n):
        a_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
        if a_atr <= 0:
            continue

        # Bullish FVG: gap between candle i-2 high and candle i low
        gap_low = float(highs[i - 2])
        gap_high = float(lows[i])
        if gap_high > gap_low:
            gap_size = gap_high - gap_low
            if gap_size >= min_gap_atr * a_atr:
                zones.append({
                    "type": "FVG",
                    "bias": "BULLISH",
                    "top": gap_high,
                    "bottom": gap_low,
                    "mid": (gap_high + gap_low) / 2.0,
                    "index": i,
                    "mitigated": False,
                    "inverted": False,
                })

        # Bearish FVG: gap between candle i-2 low and candle i high
        gap_high_b = float(lows[i - 2])
        gap_low_b = float(highs[i])
        if gap_high_b > gap_low_b:
            gap_size = gap_high_b - gap_low_b
            if gap_size >= min_gap_atr * a_atr:
                zones.append({
                    "type": "FVG",
                    "bias": "BEARISH",
                    "top": gap_high_b,
                    "bottom": gap_low_b,
                    "mid": (gap_high_b + gap_low_b) / 2.0,
                    "index": i,
                    "mitigated": False,
                    "inverted": False,
                })

    # Mark mitigation / inversion using later price action
    for z in zones:
        start = z["index"] + 1
        if start >= n:
            continue
        if z["bias"] == "BULLISH":
            # Mitigated when a later low trades into/through the gap
            later_lows = lows[start:]
            if len(later_lows) and later_lows.min() <= z["top"]:
                z["mitigated"] = True
                # Fully filled + closed below → inverse FVG potential
                later_closes = closes[start:]
                if len(later_closes) and later_closes.min() < z["bottom"]:
                    z["inverted"] = True
                    z["type"] = "IFVG"
        else:
            later_highs = highs[start:]
            if len(later_highs) and later_highs.max() >= z["bottom"]:
                z["mitigated"] = True
                later_closes = closes[start:]
                if len(later_closes) and later_closes.max() > z["top"]:
                    z["inverted"] = True
                    z["type"] = "IFVG"

    # Prefer active (unmitigated) zones, then recent
    active = [z for z in zones if not z["mitigated"]]
    inverted = [z for z in zones if z.get("inverted")]
    combined = active + [z for z in inverted if z not in active]
    combined.sort(key=lambda z: z["index"], reverse=True)
    return combined[:max_zones]


def detect_order_blocks(df, structure=None, min_body_atr=0.45, max_zones=6):
    """
    ICT-correct Order Blocks tied to ZigZag swings + displacement.

    Valid sequence on every structural leg:
      1. Swing high/low (ZigZag)
      2. Optional liquidity sweep of that swing
      3. Displacement that breaks structure (BOS/CHoCH)
      4. OB = last opposing candle BEFORE the displacement
      5. FVG often forms inside that same displacement (linked by impulse_index)

    Rule: No BOS/CHoCH-quality displacement → no valid OB.
    """
    if df is None or len(df) < 20:
        return []

    from structure_engine import zigzag_swings

    atr = _atr_series(df)
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    n = len(df)

    pivots = zigzag_swings(df, depth=4, deviation_atr=0.30)
    if len(pivots) < 3:
        pivots = []  # fall through will return fewer zones

    zones = []

    # --- Primary path: OB at origin of leg that breaks a prior swing ---
    for k in range(1, len(pivots)):
        cur = pivots[k]
        prev = pivots[k - 1]
        # Displacement leg runs from prev pivot toward cur
        leg_start = prev["index"]
        leg_end = cur["index"]
        if leg_end - leg_start < 2:
            continue

        # Did this leg break a prior swing of the same type? (BOS quality)
        broken = None
        for older in reversed(pivots[: k - 1]):
            if older["type"] == cur["type"]:
                if cur["type"] == "high" and cur["price"] > older["price"]:
                    broken = older
                elif cur["type"] == "low" and cur["price"] < older["price"]:
                    broken = older
                break
        if broken is None:
            # Still allow strong displacement vs ATR even without prior same-type break
            leg_range = abs(cur["price"] - prev["price"])
            a_mid = float(atr.iloc[min(leg_end, n - 1)]) if not np.isnan(atr.iloc[min(leg_end, n - 1)]) else 0
            if a_mid <= 0 or leg_range < 1.2 * a_mid:
                continue

        # Find last opposing candle before the impulsive part of the leg
        # Scan from leg_end backward toward leg_start
        if cur["type"] == "high":
            # Bullish displacement → last bearish candle in the leg = bullish OB
            ob_idx = None
            for j in range(leg_end - 1, leg_start - 1, -1):
                if j < 0:
                    break
                if c[j] < o[j]:  # bearish candle
                    a = float(atr.iloc[j]) if not np.isnan(atr.iloc[j]) else 0
                    body = abs(c[j] - o[j])
                    if a > 0 and body >= min_body_atr * a * 0.5:
                        ob_idx = j
                        break
            if ob_idx is None:
                continue
            zones.append({
                "type": "OB",
                "bias": "BULLISH",
                "top": float(max(o[ob_idx], c[ob_idx])),
                "bottom": float(min(o[ob_idx], c[ob_idx])),
                "wick_top": float(h[ob_idx]),
                "wick_bottom": float(l[ob_idx]),
                "index": ob_idx,
                "impulse_index": leg_end,
                "swing_broken": broken["price"] if broken else None,
                "bos": broken is not None,
                "mitigated": False,
                "inverted": False,
            })
        else:
            # Bearish displacement → last bullish candle = bearish OB
            ob_idx = None
            for j in range(leg_end - 1, leg_start - 1, -1):
                if j < 0:
                    break
                if c[j] > o[j]:
                    a = float(atr.iloc[j]) if not np.isnan(atr.iloc[j]) else 0
                    body = abs(c[j] - o[j])
                    if a > 0 and body >= min_body_atr * a * 0.5:
                        ob_idx = j
                        break
            if ob_idx is None:
                continue
            zones.append({
                "type": "OB",
                "bias": "BEARISH",
                "top": float(max(o[ob_idx], c[ob_idx])),
                "bottom": float(min(o[ob_idx], c[ob_idx])),
                "wick_top": float(h[ob_idx]),
                "wick_bottom": float(l[ob_idx]),
                "index": ob_idx,
                "impulse_index": leg_end,
                "swing_broken": broken["price"] if broken else None,
                "bos": broken is not None,
                "mitigated": False,
                "inverted": False,
            })

    # Prefer OBs that actually broke structure
    zones.sort(key=lambda z: (not z.get("bos", False), -z["index"]))

    # Mitigation → Breaker
    for z in zones:
        start = z["impulse_index"] + 1
        if start >= n:
            continue
        if z["bias"] == "BULLISH":
            if l[start:].min() < z["bottom"]:
                z["mitigated"] = True
                z["inverted"] = True
                z["type"] = "BREAKER"
        else:
            if h[start:].max() > z["top"]:
                z["mitigated"] = True
                z["inverted"] = True
                z["type"] = "BREAKER"

    for z in zones:
        z["mid"] = (z["top"] + z["bottom"]) / 2.0

    # Keep unmitigated first, then recent breakers
    active = [z for z in zones if not z["mitigated"]]
    breakers = [z for z in zones if z.get("inverted") and z not in active]
    combined = active + breakers
    # Deduplicate near-identical zones
    cleaned = []
    for z in combined:
        if any(abs(z["mid"] - c0["mid"]) / max(abs(z["mid"]), 1e-9) < 0.0008 for c0 in cleaned):
            continue
        cleaned.append(z)
    return cleaned[:max_zones]


def build_bos_events(df, max_events=8):
    """
    Build BOS / CHoCH events for chart drawing (dotted lines at broken levels).
    Uses ZigZag pivots.
    Returns list of {index, price, type: 'BOS'|'CHoCH', bias: 'BULLISH'|'BEARISH'}
    """
    if df is None or len(df) < 20:
        return []
    from structure_engine import zigzag_swings

    pivots = zigzag_swings(df, depth=4, deviation_atr=0.30)
    if len(pivots) < 4:
        return []

    events = []
    bias = "NEUTRAL"
    for k in range(1, len(pivots)):
        cur = pivots[k]
        # Find previous same-type pivot
        older = None
        for j in range(k - 1, -1, -1):
            if pivots[j]["type"] == cur["type"]:
                older = pivots[j]
                break
        if older is None:
            continue

        if cur["type"] == "high" and cur["price"] > older["price"]:
            if bias == "BEARISH":
                events.append({
                    "index": cur["index"],
                    "price": older["price"],
                    "type": "CHoCH",
                    "bias": "BULLISH",
                })
                bias = "BULLISH"
            else:
                events.append({
                    "index": cur["index"],
                    "price": older["price"],
                    "type": "BOS",
                    "bias": "BULLISH",
                })
                bias = "BULLISH"
        elif cur["type"] == "low" and cur["price"] < older["price"]:
            if bias == "BULLISH":
                events.append({
                    "index": cur["index"],
                    "price": older["price"],
                    "type": "CHoCH",
                    "bias": "BEARISH",
                })
                bias = "BEARISH"
            else:
                events.append({
                    "index": cur["index"],
                    "price": older["price"],
                    "type": "BOS",
                    "bias": "BEARISH",
                })
                bias = "BEARISH"

    return events[-max_events:]


def detect_inducement_zones(df, equal_tol=0.0008, max_zones=8):
    """
    Inducement (IDM) = internal liquidity that sits BEFORE an extreme zone.

    Matches the OB + IDM + Confirmation model:
      - Equal highs / equal lows (classic liquidity pools)
      - Internal swing highs/lows (minor structure that traps breakout traders)
      - Explicit mitigated (swept) vs unmitigated status

    Buy-side IDM  → often raided before a sell into extreme bearish OB
    Sell-side IDM → often raided before a buy into extreme bullish OB
    """
    if df is None or len(df) < 20:
        return []

    from structure_engine import find_swings

    swings = find_swings(df, left=2, right=2)
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    zones = []
    atr = _atr_series(df)
    last_atr = float(atr.iloc[-1]) if len(atr) and not np.isnan(atr.iloc[-1]) else 0.0
    highs_arr = df["High"].values
    lows_arr = df["Low"].values
    n = len(df)

    def _mitigated_buy_side(top, from_idx):
        """Buy-side IDM mitigated when a later high trades through it."""
        if from_idx + 1 >= n:
            return False
        return float(highs_arr[from_idx + 1:].max()) > top

    def _mitigated_sell_side(bottom, from_idx):
        if from_idx + 1 >= n:
            return False
        return float(lows_arr[from_idx + 1:].min()) < bottom

    # 1) Equal highs → buy-side inducement
    for i in range(1, len(highs)):
        a, b = highs[i - 1], highs[i]
        mid = (a["price"] + b["price"]) / 2.0
        if mid <= 0:
            continue
        if abs(a["price"] - b["price"]) / mid <= equal_tol:
            top = max(a["price"], b["price"])
            bottom = min(a["price"], b["price"])
            pad = last_atr * 0.05 if last_atr > 0 else mid * 0.0002
            mit = _mitigated_buy_side(top, b["index"])
            zones.append({
                "type": "IDM",
                "bias": "BUY_SIDE",
                "side": "buy_side_liquidity",
                "top": top + pad,
                "bottom": bottom - pad,
                "mid": mid,
                "index": b["index"],
                "mitigated": mit,
                "swept": mit,
                "note": "Equal highs IDM — internal buy-side liquidity before extreme",
            })

    # 2) Equal lows → sell-side inducement
    for i in range(1, len(lows)):
        a, b = lows[i - 1], lows[i]
        mid = (a["price"] + b["price"]) / 2.0
        if mid <= 0:
            continue
        if abs(a["price"] - b["price"]) / mid <= equal_tol:
            top = max(a["price"], b["price"])
            bottom = min(a["price"], b["price"])
            pad = last_atr * 0.05 if last_atr > 0 else mid * 0.0002
            mit = _mitigated_sell_side(bottom, b["index"])
            zones.append({
                "type": "IDM",
                "bias": "SELL_SIDE",
                "side": "sell_side_liquidity",
                "top": top + pad,
                "bottom": bottom - pad,
                "mid": mid,
                "index": b["index"],
                "mitigated": mit,
                "swept": mit,
                "note": "Equal lows IDM — internal sell-side liquidity before extreme",
            })

    # 3) Internal swing highs/lows as single-point inducement (not only equals)
    #    Take recent intermediate swings (skip the absolute extreme high/low)
    if len(highs) >= 3:
        sorted_h = sorted(highs, key=lambda s: s["price"], reverse=True)
        extreme_h = sorted_h[0]
        for sw in sorted_h[1:4]:  # next internal highs
            if sw["index"] >= extreme_h["index"]:
                continue  # only IDM that formed BEFORE the extreme
            pad = last_atr * 0.08 if last_atr > 0 else sw["price"] * 0.0003
            mit = _mitigated_buy_side(sw["price"], sw["index"])
            zones.append({
                "type": "IDM",
                "bias": "BUY_SIDE",
                "side": "buy_side_liquidity",
                "top": sw["price"] + pad,
                "bottom": sw["price"] - pad,
                "mid": sw["price"],
                "index": sw["index"],
                "mitigated": mit,
                "swept": mit,
                "note": "Internal swing high IDM — before extreme high/OB",
                "before_extreme": True,
            })

    if len(lows) >= 3:
        sorted_l = sorted(lows, key=lambda s: s["price"])
        extreme_l = sorted_l[0]
        for sw in sorted_l[1:4]:
            if sw["index"] >= extreme_l["index"]:
                continue
            pad = last_atr * 0.08 if last_atr > 0 else sw["price"] * 0.0003
            mit = _mitigated_sell_side(sw["price"], sw["index"])
            zones.append({
                "type": "IDM",
                "bias": "SELL_SIDE",
                "side": "sell_side_liquidity",
                "top": sw["price"] + pad,
                "bottom": sw["price"] - pad,
                "mid": sw["price"],
                "index": sw["index"],
                "mitigated": mit,
                "swept": mit,
                "note": "Internal swing low IDM — before extreme low/OB",
                "before_extreme": True,
            })

    # Prefer unmitigated, then before_extreme, then recent
    zones.sort(key=lambda z: (
        z.get("mitigated", False),
        not z.get("before_extreme", False),
        -z["index"],
    ))
    cleaned = []
    for z in zones:
        if any(abs(z["mid"] - c["mid"]) / max(abs(z["mid"]), 1e-9) < equal_tol for c in cleaned):
            continue
        cleaned.append(z)
    return cleaned[:max_zones]


def pair_idm_with_extreme_ob(inducements, order_blocks):
    """
    Link IDM that sits in front of an extreme unmitigated OB.
    Returns list of setup dicts: {idm, ob, sequence_note}
    """
    setups = []
    active_obs = [o for o in (order_blocks or []) if not o.get("mitigated")]
    for idm in inducements or []:
        for ob in active_obs:
            # Buy-side IDM in front of bearish OB (IDM below the OB)
            if idm["bias"] == "BUY_SIDE" and ob["bias"] == "BEARISH":
                if idm["mid"] < ob["bottom"] and idm["index"] < ob.get("index", 10**9):
                    setups.append({
                        "idm": idm,
                        "ob": ob,
                        "direction": "SELL",
                        "sequence": "IDM (buy-side) → extreme bearish OB — wait sweep of IDM then confirmation into OB",
                    })
            # Sell-side IDM in front of bullish OB (IDM above the OB)
            if idm["bias"] == "SELL_SIDE" and ob["bias"] == "BULLISH":
                if idm["mid"] > ob["top"] and idm["index"] < ob.get("index", 10**9):
                    setups.append({
                        "idm": idm,
                        "ob": ob,
                        "direction": "BUY",
                        "sequence": "IDM (sell-side) → extreme bullish OB — wait sweep of IDM then confirmation into OB",
                    })
    return setups[:4]


def summarise_smc_zones(fvgs, obs, max_show=5, inducements=None):
    """Short text lines — always show MITIGATED vs UNMITIGATED."""
    lines = []
    for z in fvgs[:max_show]:
        tag = z["type"]
        if z.get("mitigated"):
            status = "MITIGATED→IFVG" if z.get("inverted") else "MITIGATED"
        else:
            status = "UNMITIGATED"
        lines.append(
            f"  {tag} {z['bias']} [{status}]: {z['bottom']:.5f} – {z['top']:.5f}"
        )
    for z in obs[:max_show]:
        tag = z["type"]
        if z.get("mitigated"):
            status = "MITIGATED→BREAKER" if z.get("inverted") else "MITIGATED"
        else:
            status = "UNMITIGATED"
        lines.append(
            f"  {tag} {z['bias']} [{status}]: {z['bottom']:.5f} – {z['top']:.5f}"
        )
    if inducements:
        for z in inducements[:max_show]:
            status = "MITIGATED (swept)" if z.get("mitigated") or z.get("swept") else "UNMITIGATED"
            tag_extra = " before-extreme" if z.get("before_extreme") else ""
            lines.append(
                f"  IDM {z['bias']} [{status}]{tag_extra}: {z['bottom']:.5f} – {z['top']:.5f}"
            )
    return lines
"""
sr_zones.py
================
Right now a pattern's "resistance_level"/"support_level" is a single price
taken from a local high/low -- that treats a level touched once the same as
one defended four times. This clusters nearby pivot touches (highs and lows
together) into zones and scores them by touch count, so a trigger sitting
on a heavily-touched zone can be weighted higher than one sitting on a
level nobody's ever really tested.

Self-contained: patterns.py already computes pivot highs/lows internally
for its own detectors, so this just reuses those same arrays -- no new
external data source needed (unlike volume_profile.py, which needs real
tick data).
"""

import numpy as np


def cluster_sr_zones(prices, tolerance_frac=0.0015):
    """
    prices: flat list/array of price levels (pivot highs + pivot lows combined).
    Greedily clusters values within tolerance_frac of each other.
    Returns list of {"level": float, "touch_count": int}, sorted by touch_count desc.
    """
    if prices is None or len(prices) == 0:
        return []
    sorted_prices = sorted(float(p) for p in prices)
    zones = []
    current_cluster = [sorted_prices[0]]

    for p in sorted_prices[1:]:
        cluster_center = sum(current_cluster) / len(current_cluster)
        tol = cluster_center * tolerance_frac
        if abs(p - cluster_center) <= tol:
            current_cluster.append(p)
        else:
            zones.append({"level": sum(current_cluster) / len(current_cluster), "touch_count": len(current_cluster)})
            current_cluster = [p]
    zones.append({"level": sum(current_cluster) / len(current_cluster), "touch_count": len(current_cluster)})

    zones.sort(key=lambda z: z["touch_count"], reverse=True)
    return zones


def zone_strength_bonus(zones, price_level, tolerance_frac=0.0015, max_bonus=10.0):
    """
    Confidence delta for a trigger sitting at/near a well-touched S/R zone.
    +2 per touch beyond the first 2 (i.e. a 4-touch zone -> +4, capped at max_bonus).
    Returns 0.0 if no zone is nearby.
    """
    if not zones or price_level is None:
        return 0.0
    for z in zones:
        tol = z["level"] * tolerance_frac
        if abs(price_level - z["level"]) <= tol:
            bonus = max(0, z["touch_count"] - 2) * 2.0
            return float(min(max_bonus, bonus))
    return 0.0
"""
candlestick_patterns.py
================
Single/double/triple candle patterns, used as an ALTERNATIVE confirmation
trigger alongside the marubozu rule in confirmation_engine.py -- not a
replacement. A clean bullish engulfing or hammer at a breakout is just as
much "conviction" as a marubozu candle; this widens what counts as
confirmation without loosening the underlying standard (each pattern here
still requires a real, well-formed shape -- not just "any candle").

Only high-probability, clearly-directional patterns are included. Doji is
deliberately excluded from confirmation (it signals indecision, not
conviction) -- it's noted separately as a caution flag, not a trigger.
"""


def _body(o, c):
    return abs(c - o)


def _range(h, l):
    return h - l


def is_bullish_engulfing(prior, current):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    prior_bearish = pc < po
    current_bullish = cc > co
    engulfs = (co <= pc) and (cc >= po)
    return prior_bearish and current_bullish and engulfs


def is_bearish_engulfing(prior, current):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    prior_bullish = pc > po
    current_bearish = cc < co
    engulfs = (co >= pc) and (cc <= po)
    return prior_bullish and current_bearish and engulfs


def is_hammer(bar, atr):
    o, h, l, c = bar
    rng = _range(h, l)
    if rng <= 0 or atr is None or atr <= 0 or rng < 0.5 * atr:
        return False
    body = _body(o, c)
    if body / rng < 0.08:  # avoid doji-like near-zero bodies
        return False
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (lower_wick >= 2.0 * body) and (upper_wick <= 0.3 * body if body > 0 else upper_wick <= 0.05 * rng)


def is_shooting_star(bar, atr):
    o, h, l, c = bar
    rng = _range(h, l)
    if rng <= 0 or atr is None or atr <= 0 or rng < 0.5 * atr:
        return False
    body = _body(o, c)
    if body / rng < 0.08:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (upper_wick >= 2.0 * body) and (lower_wick <= 0.3 * body if body > 0 else lower_wick <= 0.05 * rng)


def is_inverted_hammer(bar, atr):
    # same shape as shooting star, but used at the base of a downtrend as a bullish signal
    return is_shooting_star(bar, atr)


def is_piercing_line(prior, current):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    prior_bearish = pc < po
    prior_mid = (po + pc) / 2.0
    current_bullish = cc > co
    opens_below_prior_low_zone = co < pc  # gaps down or opens near/below prior close
    closes_above_midpoint = cc > prior_mid and cc < po
    return prior_bearish and current_bullish and opens_below_prior_low_zone and closes_above_midpoint


def is_dark_cloud_cover(prior, current):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    prior_bullish = pc > po
    prior_mid = (po + pc) / 2.0
    current_bearish = cc < co
    opens_above_prior_high_zone = co > pc
    closes_below_midpoint = cc < prior_mid and cc > po
    return prior_bullish and current_bearish and opens_above_prior_high_zone and closes_below_midpoint


def is_morning_star(bar1, bar2, bar3):
    o1, h1, l1, c1 = bar1
    o2, h2, l2, c2 = bar2
    o3, h3, l3, c3 = bar3
    first_bearish = c1 < o1
    second_small = _body(o2, c2) < 0.4 * _body(o1, c1) if _body(o1, c1) > 0 else True
    gapped_down = max(o2, c2) < c1
    third_bullish = c3 > o3
    closes_into_first_body = c3 > (o1 + c1) / 2.0
    return first_bearish and second_small and gapped_down and third_bullish and closes_into_first_body


def is_evening_star(bar1, bar2, bar3):
    o1, h1, l1, c1 = bar1
    o2, h2, l2, c2 = bar2
    o3, h3, l3, c3 = bar3
    first_bullish = c1 > o1
    second_small = _body(o2, c2) < 0.4 * _body(o1, c1) if _body(o1, c1) > 0 else True
    gapped_up = min(o2, c2) > c1
    third_bearish = c3 < o3
    closes_into_first_body = c3 < (o1 + c1) / 2.0
    return first_bullish and second_small and gapped_up and third_bearish and closes_into_first_body


def is_three_white_soldiers(bar1, bar2, bar3):
    bars = [bar1, bar2, bar3]
    for (o, h, l, c) in bars:
        if c <= o:
            return False
    for i in range(1, 3):
        po, ph, pl, pc = bars[i-1]
        o, h, l, c = bars[i]
        if not (o > po and o < pc):  # opens within prior body
            return False
        if not (c > pc):  # each close higher than the last
            return False
    return True


def is_three_black_crows(bar1, bar2, bar3):
    bars = [bar1, bar2, bar3]
    for (o, h, l, c) in bars:
        if c >= o:
            return False
    for i in range(1, 3):
        po, ph, pl, pc = bars[i-1]
        o, h, l, c = bars[i]
        if not (o < po and o > pc):
            return False
        if not (c < pc):
            return False
    return True


def is_tweezer_bottom(prior, current, tolerance_frac=0.001):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    avg = (pl + cl) / 2.0 or 1.0
    return abs(pl - cl) / abs(avg) <= tolerance_frac and pc < po and cc > co


def is_tweezer_top(prior, current, tolerance_frac=0.001):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    avg = (ph + ch) / 2.0 or 1.0
    return abs(ph - ch) / abs(avg) <= tolerance_frac and pc > po and cc < co


def is_doji(bar, atr):
    """Not used as confirmation -- indecision, not conviction. Exposed for
    optional caution-flagging elsewhere (e.g. 'setup forming but last candle
    was a doji, expect more chop before a real move')."""
    o, h, l, c = bar
    rng = _range(h, l)
    if rng <= 0:
        return False
    return _body(o, c) / rng < 0.08


def detect_confirmation_candle(df, bias):
    """
    Checks the most recent bars for a directionally-matching candlestick
    confirmation pattern. Returns (found: bool, pattern_name: str or None).
    """
    n = len(df)
    if n < 3:
        return False, None
    atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else None

    def bar_at(i):
        row = df.iloc[i]
        return (float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']))

    b1, b2, b3 = bar_at(-3), bar_at(-2), bar_at(-1)

    if bias == "BUY":
        if is_bullish_engulfing(b2, b3): return True, "Bullish Engulfing"
        if is_hammer(b3, atr): return True, "Hammer"
        if is_inverted_hammer(b3, atr): return True, "Inverted Hammer"
        if is_piercing_line(b2, b3): return True, "Piercing Line"
        if is_morning_star(b1, b2, b3): return True, "Morning Star"
        if is_three_white_soldiers(b1, b2, b3): return True, "Three White Soldiers"
        if is_tweezer_bottom(b2, b3): return True, "Tweezer Bottom"
    else:
        if is_bearish_engulfing(b2, b3): return True, "Bearish Engulfing"
        if is_shooting_star(b3, atr): return True, "Shooting Star"
        if is_dark_cloud_cover(b2, b3): return True, "Dark Cloud Cover"
        if is_evening_star(b1, b2, b3): return True, "Evening Star"
        if is_three_black_crows(b1, b2, b3): return True, "Three Black Crows"
        if is_tweezer_top(b2, b3): return True, "Tweezer Top"

    return False, None
"""
htf_context.py
================
Top-down analysis: for a lower-timeframe entry signal, check the higher
timeframe's own structure/trend first. Scalping timeframes (1m/3m/5m/15m)
are for entry TIMING -- the higher timeframe establishes the actual
directional bias. A lower-timeframe pattern that aligns with the higher
timeframe gets reinforced; one that fights it gets flagged as counter-trend.

HTF bias is read two ways, preferring the stronger signal:
  1. If the higher timeframe itself has a valid detected chart pattern,
     that pattern's bias IS the HTF bias (strongest signal).
  2. Otherwise, fall back to a simple EMA50 trend read on the HTF.
"""

# [merged] was: from patterns import scan_all_patterns
import data

LTF_TO_HTF = {
    "1min": "15min",
    "3min": "30min",
    "5min": "1h",
    "15min": "4h",
}

HTF_ALIGN_BONUS = 8.0
HTF_COUNTER_PENALTY = -10.0


def get_htf_bias(symbol, ltf_timeframe):
    """
    Returns (bias, description) or (None, None) if no HTF mapping exists
    for this timeframe (e.g. already a high timeframe) or data is unavailable.
    """
    htf_tf = LTF_TO_HTF.get(ltf_timeframe)
    if htf_tf is None:
        return None, None

    df = mt5_data.fetch_candles(symbol, htf_tf, count=150)
    if df is None or df.empty or len(df) < 40:
        return None, None

    best, _ = scan_all_patterns(df)
    if best is not None:
        return best.bias, f"{best.name} on {htf_tf}"

    if 'EMA50' in df.columns:
        current_close = float(df['Close'].iloc[-1])
        ema50 = float(df['EMA50'].iloc[-1])
        bias = "BUY" if current_close > ema50 else "SELL"
        return bias, f"price {'above' if bias=='BUY' else 'below'} EMA50 trend on {htf_tf}"

    return None, None


def htf_alignment_adjustment(ltf_bias, htf_bias, htf_desc):
    """
    Returns (confidence_delta, note_text). Zero delta and a neutral note
    if no HTF context is available at all -- never silently penalizes
    when we simply don't know.
    """
    if htf_bias is None:
        return 0.0, "No higher-timeframe context available."
    if ltf_bias == htf_bias:
        return HTF_ALIGN_BONUS, f"Aligned with higher-timeframe bias: {htf_desc}."
    return HTF_COUNTER_PENALTY, f"⚠️ COUNTER-TREND: against higher-timeframe bias ({htf_desc}) -- treat with extra caution."
"""
institutional_analysis.py
=========================
True Top-Down Institutional Analysis.

Priority stack:
  1. 200 EMA on HTF          → Who is ruling the market
  2. Trendline Families      → Primary structure + projections
  3. BOS / CHoCH / MSS       → Structure permission
  4. VWAP / Volume Profile   → Dynamic levels
  5. FVG / OB / IDM          → SMC zones (drawn on chart)
  6. Chart Patterns          → Confluence only

Reports are SHORT — the chart carries the visual story.
"""

import numpy as np
import pandas as pd
# [merged] was: from patterns import scan_all_patterns, find_pivots, _atr, Pattern
from data import compute_volume_profile
from structure_engine import analyse_structure, structure_trade_permission
import data

TOPDOWN_LADDER = [
    ("4h", "4 Hour"),
    ("1h", "1 Hour"),
    ("30min", "30 Minute"),
]
ALT_LADDER = [
    ("1h", "1 Hour"),
    ("30min", "30 Minute"),
    ("15min", "15 Minute"),
]


def _ema200_bias(df):
    if df is None or df.empty or "EMA200" not in df.columns:
        return "NEUTRAL", "EMA200 n/a", 0.0
    close = float(df["Close"].iloc[-1])
    ema200 = float(df["EMA200"].iloc[-1])
    if ema200 <= 0:
        return "NEUTRAL", "EMA200 n/a", 0.0
    dist = (close - ema200) / ema200 * 100.0
    if close > ema200 * 1.001:
        return "BUY", f"Above 200 EMA (+{dist:.2f}%)", dist
    if close < ema200 * 0.999:
        return "SELL", f"Below 200 EMA ({dist:.2f}%)", dist
    return "NEUTRAL", f"At 200 EMA ({dist:+.2f}%)", dist


def _vwap_context(df):
    if df is None or df.empty or "VWAP" not in df.columns:
        return None
    close = float(df["Close"].iloc[-1])
    vwap = float(df["VWAP"].iloc[-1])
    if vwap <= 0:
        return None
    dist_pct = (close - vwap) / vwap * 100.0
    if close > vwap * 1.0005:
        pos, note = "ABOVE", f"Above VWAP (+{dist_pct:.2f}%)"
    elif close < vwap * 0.9995:
        pos, note = "BELOW", f"Below VWAP ({dist_pct:.2f}%)"
    else:
        pos, note = "AT", f"At VWAP ({dist_pct:+.2f}%)"
    return {"vwap": vwap, "position": pos, "distance_pct": dist_pct, "note": note}


def _fit_trendline_family(df, lookback=80):
    if df is None or len(df) < 30:
        return None
    ph, pl = find_pivots(df, left=4, right=4)
    n = len(df)
    start = max(0, n - lookback)
    recent_ph = [p for p in ph if p >= start][-4:]
    recent_pl = [p for p in pl if p >= start][-4:]
    if len(recent_ph) < 2 or len(recent_pl) < 2:
        return None

    upper_pts = [(p, float(df["High"].iloc[p])) for p in recent_ph]
    lower_pts = [(p, float(df["Low"].iloc[p])) for p in recent_pl]

    def _fit(pts):
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        if len(xs) < 2 or np.all(xs == xs[0]):
            return 0.0, float(ys[-1])
        slope, intercept = np.polyfit(xs, ys, 1)
        return float(slope), float(intercept)

    up_slope, up_int = _fit(upper_pts)
    lo_slope, lo_int = _fit(lower_pts)
    x_now = n - 1
    upper_now = up_slope * x_now + up_int
    lower_now = lo_slope * x_now + lo_int
    if upper_now <= lower_now:
        return None
    height = upper_now - lower_now
    close = float(df["Close"].iloc[-1])
    pos = (close - lower_now) / height if height > 0 else 0.5
    avg_price = float(df["Close"].tail(lookback).mean()) or 1.0
    up_norm = (up_slope * lookback) / avg_price
    lo_norm = (lo_slope * lookback) / avg_price
    FLAT = 0.004
    if abs(up_norm) < FLAT and lo_norm > FLAT:
        family, bias = "Ascending Channel", "BUY"
    elif up_norm < -FLAT and abs(lo_norm) < FLAT:
        family, bias = "Descending Channel", "SELL"
    elif up_norm > FLAT and lo_norm > FLAT:
        family, bias = "Rising Channel", "BUY"
    elif up_norm < -FLAT and lo_norm < -FLAT:
        family, bias = "Falling Channel", "SELL"
    else:
        family, bias = "Range / Contract", "NEUTRAL"
    return {
        "family": family, "bias": bias,
        "upper": float(upper_now), "lower": float(lower_now),
        "mid": float((upper_now + lower_now) / 2),
        "height": float(height),
        "position": float(np.clip(pos, 0, 1)),
        "proj_up": float(upper_now + height),
        "proj_down": float(lower_now - height),
        "upper_pts": upper_pts, "lower_pts": lower_pts,
    }


def _analyse_single_tf(symbol, tf_code, tf_label):
    df = mt5_data.fetch_candles(symbol, tf_code, count=250)
    if df is None or df.empty or len(df) < 40:
        return None
    ema_bias, ema_note, ema_dist = _ema200_bias(df)
    vwap = _vwap_context(df)
    trend = _fit_trendline_family(df)
    vp = compute_volume_profile(df.iloc[:-1])
    best, all_pats = scan_all_patterns(df.iloc[:-1], volume_profile=vp)
    structure = analyse_structure(df, left=3, right=3, lookback=70)
    fvgs = detect_fvgs(df, min_gap_atr=0.15, max_zones=5)
    obs = detect_order_blocks(df, structure=structure, max_zones=4)
    idms = detect_inducement_zones(df, max_zones=4)
    bos_events = build_bos_events(df, max_events=8)
    return {
        "tf": tf_code, "tf_label": tf_label, "df": df,
        "close": float(df["Close"].iloc[-1]),
        "ema200_bias": ema_bias, "ema200_note": ema_note, "ema200_dist": ema_dist,
        "vwap": vwap, "trendline": trend, "volume_profile": vp,
        "best_pattern": best, "all_patterns": all_pats[:3] if all_pats else [],
        "structure": structure, "fvgs": fvgs, "order_blocks": obs, "inducements": idms,
        "bos_events": bos_events,
    }


def run_topdown_analysis(symbol):
    symbol = symbol.strip().upper()
    frames = []
    for tf_code, tf_label in TOPDOWN_LADDER:
        snap = _analyse_single_tf(symbol, tf_code, tf_label)
        if snap:
            frames.append(snap)
    if len(frames) < 2:
        frames = []
        for tf_code, tf_label in ALT_LADDER:
            snap = _analyse_single_tf(symbol, tf_code, tf_label)
            if snap:
                frames.append(snap)
    if not frames:
        return {"error": f"No data for {symbol}."}

    htf = frames[0]
    overall_bias = htf["ema200_bias"]
    if htf.get("trendline") and htf["trendline"]["bias"] != "NEUTRAL":
        if htf["trendline"]["bias"] == overall_bias or overall_bias == "NEUTRAL":
            overall_bias = htf["trendline"]["bias"]

    biases = [f["ema200_bias"] for f in frames if f["ema200_bias"] != "NEUTRAL"]
    if not biases:
        alignment = "MIXED"
    else:
        buy_c = sum(1 for b in biases if b == "BUY")
        sell_c = sum(1 for b in biases if b == "SELL")
        if buy_c == len(biases):
            alignment = "ALIGNED BULLISH"
        elif sell_c == len(biases):
            alignment = "ALIGNED BEARISH"
        elif buy_c > sell_c:
            alignment = "MOSTLY BULLISH"
        elif sell_c > buy_c:
            alignment = "MOSTLY BEARISH"
        else:
            alignment = "MIXED"

    primary_proj = None
    if htf.get("trendline"):
        t = htf["trendline"]
        if overall_bias == "BUY":
            primary_proj = {"direction": "UP", "target": t["proj_up"], "invalidation": t["lower"]}
        elif overall_bias == "SELL":
            primary_proj = {"direction": "DOWN", "target": t["proj_down"], "invalidation": t["upper"]}

    pairs = pair_idm_with_extreme_ob(htf.get("inducements") or [], htf.get("order_blocks") or [])
    ltf = frames[-1]
    allowed, reason, pref = structure_trade_permission(
        overall_bias, ltf.get("structure") or {}
    )

    return {
        "symbol": symbol,
        "overall_bias": overall_bias,
        "alignment": alignment,
        "htf_regime": htf["ema200_note"],
        "frames": frames,
        "primary_projection": primary_proj,
        "htf_trendline": htf.get("trendline"),
        "htf_vwap": htf.get("vwap"),
        "htf_poc": htf["volume_profile"]["poc_price"] if htf.get("volume_profile") else None,
        "idm_ob_pairs": pairs,
        "structure_allowed": allowed,
        "structure_reason": reason,
        "structure_prefer": pref,
        # Chart payload (HTF preferred for institutional map; LTF for entry view)
        "chart_frame": htf,
    }


def format_institutional_report(analysis):
    """SHORT summary — chart shows the zones."""
    if "error" in analysis:
        return analysis["error"]

    symbol = analysis["symbol"]
    bias = analysis["overall_bias"]
    align = analysis["alignment"]
    htf = analysis["frames"][0]
    lines = []

    lines.append(f"🏛 {symbol}  |  Bias: {bias}  |  {align}")
    lines.append(f"HTF: {analysis['htf_regime']}")

    # One-line structure per TF
    bits = []
    for f in analysis["frames"]:
        ev = (f.get("structure") or {}).get("last_event") or "—"
        bits.append(f"{f['tf_label']}: {ev}")
    lines.append("Struct: " + " · ".join(bits))

    # Key levels only (not every zone price)
    keys = []
    if htf.get("vwap"):
        keys.append(f"VWAP {htf['vwap']['vwap']:.5f}")
    if analysis.get("htf_poc"):
        keys.append(f"POC {analysis['htf_poc']:.5f}")
    if htf.get("trendline"):
        t = htf["trendline"]
        keys.append(f"Ch {t['lower']:.5f}/{t['upper']:.5f}")
    if keys:
        lines.append("Levels: " + " | ".join(keys))

    proj = analysis.get("primary_projection")
    if proj:
        lines.append(f"Proj: {proj['direction']} → {proj['target']:.5f}  (inv {proj['invalidation']:.5f})")

    # Zone counts (details are on the chart)
    n_fvg = len(htf.get("fvgs") or [])
    n_ob = len(htf.get("order_blocks") or [])
    n_idm = len(htf.get("inducements") or [])
    unmit_idm = sum(1 for z in (htf.get("inducements") or []) if not z.get("mitigated"))
    lines.append(f"Zones: {n_fvg} FVG · {n_ob} OB · {n_idm} IDM ({unmit_idm} unmitigated)")

    pairs = analysis.get("idm_ob_pairs") or []
    if pairs:
        p = pairs[0]
        lines.append(f"Setup: {p['direction']} — IDM→OB (see chart)")
    else:
        lines.append("Setup: no clean IDM→OB pair")

    allowed = analysis.get("structure_allowed")
    lines.append(
        f"Permission: {'YES' if allowed else 'WAIT'} — {analysis.get('structure_prefer', 'n/a')}"
    )
    if analysis.get("structure_reason"):
        lines.append(f"  {analysis['structure_reason']}")

    if htf.get("best_pattern"):
        p = htf["best_pattern"]
        lines.append(f"Pattern: {p.name} ({p.bias}) {p.confidence:.0f}%")

    lines.append("📷 Chart = full story (FVG/OB/IDM/EMA/structure)")
    return "\n".join(lines)
"""
amd_analysis.py
===============
AMD = Accumulation → Manipulation → Distribution (ICT / power-of-three style).

NOW DRIVEN BY the Institutional Structure Engine (ISE):

    Price → Structure → Liquidity → Manipulation → Acceptance → Trade

The legacy range/phase segmenter is kept only for chart shading and session
context. Direction, validity, phase identity, and trade permission come from
structure_engine.run_structure_engine() — the same pipeline already used by
the Trendline strategy — so AMD is fully dynamic and consistent with the rest
of the system.

Primary chart: **1 Hour**
Context: 4H bias → 1H ISE/AMD → 30M / 15M for entry refinement.
"""


from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from structure_engine import analyse_structure
from data import compute_volume_profile
from structure_engine import run_structure_engine, format_structure_report
import data


# UTC hour ranges (inclusive start, exclusive end-style checks use hour)
SESSION_WINDOWS = {
    "Asian": (0, 8),
    "London": (7, 16),
    "NewYork": (12, 21),
}


def _session_for_ts(ts):
    """Return session name(s) for a pandas Timestamp (assumes UTC-like index)."""
    try:
        h = ts.hour
    except Exception:
        return "Unknown"
    names = []
    for name, (start, end) in SESSION_WINDOWS.items():
        if start <= h < end:
            names.append(name)
    return "+".join(names) if names else "Off-session"


def _label_sessions(df):
    """Add a Session column for reporting."""
    if df is None or df.empty:
        return df
    out = df.copy()
    sessions = []
    for ts in out.index:
        sessions.append(_session_for_ts(ts))
    out["Session"] = sessions
    return out


def _map_ise_to_amd_phase(ise: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate Institutional Structure Engine stages into AMD phase language.

    Mapping (dynamic, not hardcoded calendar):
      • No impulse / RANGE / COMPRESSION / RECTANGLE  → ACCUMULATION
      • Liquidity sweep present, manip not confirmed   → MANIPULATION
      • Strong impulse / EXPANSION / flag breakout     → DISPLACEMENT
      • Pullback after impulse (flag / channel / retest) → REVERSION
      • Acceptance + continuation path confirmed       → CONTINUATION
    """
    if ise.get("error"):
        return {
            "phase": "ACCUMULATION",
            "bias": "NEUTRAL",
            "note": f"ISE unavailable: {ise['error']}",
            "path": None,
        }

    state = (ise.get("state") or {}).get("state", "RANGE")
    impulse = ise.get("impulse")
    pullback = ise.get("pullback")
    sweep = ise.get("sweep")
    manipulation = ise.get("manipulation") or {}
    acceptance = ise.get("acceptance") or {}
    entry = ise.get("entry") or {}
    direction = ise.get("direction", "NEUTRAL")
    path = entry.get("path")
    pattern = (pullback or {}).get("pattern") or ""

    # Highest priority: confirmed trade path from ISE
    if ise.get("valid") and direction in ("BUY", "SELL"):
        if path == "continuation":
            return {
                "phase": "CONTINUATION",
                "bias": direction,
                "note": "ISE continuation path accepted (flag/expansion break + structure). Ride with trend.",
                "path": path,
            }
        if path == "reversal":
            return {
                "phase": "CONTINUATION",  # post-acceptance expansion in the new direction
                "bias": direction,
                "note": "ISE reversal path complete (channel distribution/accumulation → sweep → manip → acceptance).",
                "path": path,
            }
        if path == "expansion":
            return {
                "phase": "DISPLACEMENT",
                "bias": direction,
                "note": "Range still expanding in impulse direction — live displacement leg.",
                "path": path,
            }

    # Acceptance confirmed but not yet broken channel/horizontal → REVERSION zone
    if acceptance.get("accepted"):
        bias = "BUY" if acceptance.get("side") == "BULLISH" else (
            "SELL" if acceptance.get("side") == "BEARISH" else direction
        )
        return {
            "phase": "REVERSION",
            "bias": bias if bias in ("BUY", "SELL") else "NEUTRAL",
            "note": acceptance.get("note", "Acceptance holding — high-probability entry zone."),
            "path": path,
        }

    # Manipulation confirmed, waiting acceptance
    if manipulation.get("confirmed"):
        hint = (sweep or {}).get("direction_hint") or direction
        return {
            "phase": "MANIPULATION",
            "bias": hint if hint in ("BUY", "SELL") else "NEUTRAL",
            "note": manipulation.get("note", "Manipulation confirmed — wait for acceptance."),
            "path": path,
        }

    # Sweep seen but not yet rejected
    if sweep is not None:
        hint = sweep.get("direction_hint") or "NEUTRAL"
        return {
            "phase": "MANIPULATION",
            "bias": hint if hint in ("BUY", "SELL") else "NEUTRAL",
            "note": sweep.get("note", "Liquidity swept — waiting for rejection / acceptance."),
            "path": path,
        }

    # Strong impulse + expansion pattern
    if impulse and not impulse.get("weak") and pattern == "EXPANSION":
        return {
            "phase": "DISPLACEMENT",
            "bias": impulse["direction"],
            "note": f"Strong impulse ({impulse['length_atr']}x ATR) with expanding range — displacement.",
            "path": path,
        }

    # Impulse exists and we are in a pullback structure
    if impulse and pullback:
        if pattern in ("BULL_FLAG", "BEAR_FLAG", "RISING_CHANNEL", "FALLING_CHANNEL", "TRIANGLE"):
            return {
                "phase": "REVERSION",
                "bias": pullback.get("bias_hint") or impulse["direction"],
                "note": f"Post-impulse {pattern.replace('_', ' ').title()} — reversion / pullback zone. {pullback.get('watch_for', '')}",
                "path": path,
            }
        if pattern in ("COMPRESSION", "RECTANGLE"):
            return {
                "phase": "ACCUMULATION",
                "bias": "NEUTRAL",
                "note": f"{pattern.title()} after impulse — energy building. Wait for liquidity grab.",
                "path": path,
            }
        # Generic pullback
        return {
            "phase": "REVERSION",
            "bias": impulse["direction"],
            "note": f"Pullback after {impulse['direction']} impulse — classify structure before entry.",
            "path": path,
        }

    # Clean impulse, no pullback classified yet → still in displacement
    if impulse and not impulse.get("weak"):
        return {
            "phase": "DISPLACEMENT",
            "bias": impulse["direction"],
            "note": f"Impulse {impulse['direction']} ({impulse['length_atr']}x ATR / {impulse['bars']} bars) — displacement leg active.",
            "path": path,
        }

    # Default: no structure → accumulation / range
    return {
        "phase": "ACCUMULATION",
        "bias": "NEUTRAL",
        "note": f"Market state {state} — no clean impulse/pullback yet. Accumulation / wait.",
        "path": None,
    }


def _build_phase_segments_from_ise(df: pd.DataFrame, ise: Dict[str, Any],
                                    lookback_range: int = 28):
    """
    Build chart-shade segments that stay visually compatible with the old AMD
    map, but are anchored to what the ISE actually detected (impulse origin,
    sweep bar, acceptance, etc.) instead of a pure range heuristic.
    Falls back to a simple accumulation block when ISE has nothing useful.
    """
    n = len(df)
    if n < 10:
        return [{"start_idx": 0, "end_idx": max(0, n - 1), "phase": "ACCUMULATION"}], "ACCUMULATION", {}

    phases = ["ACCUMULATION"] * n
    impulse = ise.get("impulse") or {}
    pullback = ise.get("pullback") or {}
    sweep = ise.get("sweep")
    manipulation = ise.get("manipulation") or {}
    acceptance = ise.get("acceptance") or {}
    mapped = _map_ise_to_amd_phase(ise)
    current_phase = mapped["phase"]

    # Paint from impulse origin forward as DISPLACEMENT
    if impulse.get("origin_index") is not None and impulse.get("extreme_index") is not None:
        o = int(impulse["origin_index"])
        e = int(impulse["extreme_index"])
        for i in range(max(0, o), min(e + 1, n)):
            phases[i] = "DISPLACEMENT"
        # Post-extreme = reversion / accumulation depending on pattern
        seg_start = e + 1
        if seg_start < n:
            post = mapped["phase"] if mapped["phase"] in (
                "REVERSION", "CONTINUATION", "ACCUMULATION", "MANIPULATION"
            ) else "REVERSION"
            for i in range(seg_start, n):
                phases[i] = post

    # Override with manipulation window around sweep
    if sweep and sweep.get("swept_pos") is not None:
        sp = int(sweep["swept_pos"])
        for i in range(max(0, sp), min(sp + 3, n)):
            phases[i] = "MANIPULATION"

    # Acceptance → paint last few bars as CONTINUATION or REVERSION
    if acceptance.get("accepted"):
        for i in range(max(0, n - 4), n):
            phases[i] = "CONTINUATION" if ise.get("valid") else "REVERSION"

    # Ensure last bar matches the dynamic current phase
    phases[-1] = current_phase

    # Collapse into segments
    segments = []
    cur = phases[0]
    seg_start = 0
    for i in range(1, n):
        if phases[i] != cur:
            segments.append({"start_idx": seg_start, "end_idx": i - 1, "phase": cur})
            cur = phases[i]
            seg_start = i
    segments.append({"start_idx": seg_start, "end_idx": n - 1, "phase": cur})

    # Range meta for report / chart (use pullback channel or recent window)
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and df["ATR"].iloc[-1] > 0 else float(
        (df["High"] - df["Low"]).tail(14).mean() or 1e-9
    )
    if pullback:
        rng_high = float(pullback.get("channel_high") or df["High"].iloc[-lookback_range:].max())
        rng_low = float(pullback.get("channel_low") or df["Low"].iloc[-lookback_range:].min())
    else:
        window = df.iloc[-lookback_range:]
        rng_high = float(window["High"].max())
        rng_low = float(window["Low"].min())

    meta = {
        "high": rng_high,
        "low": rng_low,
        "mid": (rng_high + rng_low) / 2.0,
        "height": rng_high - rng_low,
        "compressed": (rng_high - rng_low) < 2.5 * atr if atr > 0 else False,
        "atr": atr,
        "range_start_idx": max(0, n - lookback_range),
        "manip_idx": int(sweep["swept_pos"]) if sweep and sweep.get("swept_pos") is not None else None,
        "manip_side": ("HIGH" if (sweep or {}).get("side") == "BSL" else "LOW") if sweep else None,
        "disp_start": int(impulse["extreme_index"]) if impulse.get("extreme_index") is not None else None,
        "disp_dir": ("UP" if impulse.get("direction") == "BUY" else "DOWN") if impulse else None,
    }
    return segments, current_phase, meta


def run_amd_analysis(symbol):
    """
    Full AMD package — Structure Engine is the decision authority.

    1. Load multi-TF data
    2. Run ISE on 1H (same 10-stage pipeline as Trendline)
    3. Map ISE → AMD phase / bias dynamically
    4. Attach SMC zones, sessions, HTF context as confluence only
    """
    symbol = symbol.strip().upper()

    df_4h = mt5_data.fetch_candles(symbol, "4h", count=150)
    df_1h = mt5_data.fetch_candles(symbol, "1h", count=200)
    df_30 = mt5_data.fetch_candles(symbol, "30min", count=150)
    df_15 = mt5_data.fetch_candles(symbol, "15min", count=150)

    if df_1h is None or df_1h.empty or len(df_1h) < 40:
        return {"error": f"Insufficient 1H data for AMD analysis on {symbol}."}

    df_1h = _label_sessions(df_1h)
    structure_1h = analyse_structure(df_1h, left=2, right=2, lookback=60)
    structure_4h = (
        analyse_structure(df_4h, left=3, right=3, lookback=50)
        if df_4h is not None and len(df_4h) > 30 else None
    )

    # --- Core: Institutional Structure Engine on 1H ---
    ise = run_structure_engine(df_1h)
    mapped = _map_ise_to_amd_phase(ise)
    phase_segments, current_phase, rng_meta = _build_phase_segments_from_ise(df_1h, ise)

    rng = {
        "high": rng_meta["high"],
        "low": rng_meta["low"],
        "mid": rng_meta["mid"],
        "height": rng_meta["height"],
        "compressed": rng_meta["compressed"],
        "atr": rng_meta["atr"],
    }

    # Manipulation object for report compatibility
    manip = None
    sweep = ise.get("sweep")
    manipulation = ise.get("manipulation") or {}
    if sweep:
        manip = {
            "side": "BUY_SIDE_LIQUIDITY" if sweep.get("side") == "BSL" else "SELL_SIDE_LIQUIDITY",
            "direction_hint": sweep.get("direction_hint", mapped["bias"]),
            "index": sweep.get("swept_pos"),
            "note": sweep.get("note") or manipulation.get("note", ""),
            "confirmed": bool(manipulation.get("confirmed")),
        }
    elif manipulation.get("confirmed"):
        manip = {
            "side": "UNKNOWN",
            "direction_hint": mapped["bias"],
            "index": None,
            "note": manipulation.get("note", ""),
            "confirmed": True,
        }

    amd_bias = mapped["bias"]
    # Prefer ISE confirmed direction when valid
    if ise.get("valid") and ise.get("direction") in ("BUY", "SELL"):
        amd_bias = ise["direction"]

    fvgs = detect_fvgs(df_1h, min_gap_atr=0.12, max_zones=6)
    obs = detect_order_blocks(df_1h, structure=structure_1h, max_zones=5)
    idms = detect_inducement_zones(df_1h, max_zones=5)
    vp = compute_volume_profile(df_1h.iloc[:-1])

    last_session = str(df_1h["Session"].iloc[-1]) if "Session" in df_1h.columns else "Unknown"

    htf_note = ""
    if structure_4h:
        htf_note = structure_4h.get("note", "")
        if structure_4h.get("bias") == "BULLISH" and amd_bias == "SELL":
            htf_note += " | ⚠️ 1H AMD bearish vs 4H bullish structure"
        elif structure_4h.get("bias") == "BEARISH" and amd_bias == "BUY":
            htf_note += " | ⚠️ 1H AMD bullish vs 4H bearish structure"

    entry_notes = []
    for tf_name, dframe in (("30M", df_30), ("15M", df_15)):
        if dframe is not None and len(dframe) >= 40:
            st = analyse_structure(dframe, left=2, right=2, lookback=40)
            entry_notes.append(f"{tf_name}: {st.get('note', st.get('bias', ''))}")

    # Pattern scanner on 1H for confluence (fixes "pattern scanner not returning analysis")
    best_pattern = None
    all_patterns = []
    try:
        # [merged] was: from patterns import scan_all_patterns
        best_pattern, all_patterns = scan_all_patterns(df_1h.iloc[:-1], volume_profile=vp)
        if all_patterns:
            all_patterns = all_patterns[:3]
    except Exception:
        pass

    return {
        "symbol": symbol,
        "primary_tf": "1h",
        "amd_bias": amd_bias,
        "phase": current_phase,
        "phase_note": mapped["note"],
        "phase_segments": phase_segments,
        "manipulation": manip,
        "range": rng,
        "structure_1h": structure_1h,
        "structure_4h": structure_4h,
        "htf_note": htf_note,
        "last_session": last_session,
        "fvgs": fvgs,
        "order_blocks": obs,
        "inducements": idms,
        "bos_events": build_bos_events(df_1h, max_events=8),
        "volume_profile": vp,
        "entry_notes": entry_notes,
        "df_1h": df_1h,
        # ISE payload — full dynamic authority
        "ise": ise,
        "ise_valid": bool(ise.get("valid")),
        "ise_direction": ise.get("direction", "NEUTRAL"),
        "ise_score": int(ise.get("score", 0)),
        "ise_path": (ise.get("entry") or {}).get("path"),
        "ise_reasons": list(ise.get("reasons") or []),
        "best_pattern": best_pattern,
        "all_patterns": all_patterns,
    }


def format_amd_report(analysis):
    """SHORT AMD summary driven by ISE stages — chart shows zones."""
    if "error" in analysis:
        return analysis["error"]

    symbol = analysis["symbol"]
    lines = []
    lines.append(f"🕯 AMD {symbol}  |  1H  |  Bias: {analysis['amd_bias']}")
    lines.append(f"Phase: {analysis['phase']}  |  Session: {analysis['last_session']}")
    if analysis.get("phase_note"):
        lines.append(f"  {analysis['phase_note']}")

    # ISE stage summary (dynamic)
    ise = analysis.get("ise") or {}
    if ise and not ise.get("error"):
        st = ise.get("state") or {}
        lines.append(f"ISE State: {st.get('state', '?')} ({st.get('reason', '')})")
        imp = ise.get("impulse")
        if imp:
            lines.append(
                f"ISE Impulse: {imp['direction']} · {imp['length_atr']}x ATR / {imp['bars']} bars"
                + (" ⚠️ weak" if imp.get("weak") else "")
            )
        pb = ise.get("pullback")
        if pb:
            lines.append(f"ISE Pullback: {pb['pattern'].replace('_', ' ').title()}")
        if ise.get("sweep"):
            lines.append(f"ISE Liquidity: {ise['sweep'].get('note', 'sweep')}")
        man = ise.get("manipulation") or {}
        if man.get("note"):
            lines.append(f"ISE Manip: {man['note']}")
        acc = ise.get("acceptance") or {}
        if acc.get("note"):
            lines.append(f"ISE Accept: {acc['note']}")
        lines.append(
            f"ISE Verdict: {'TRADE ' + str(ise.get('direction')) if ise.get('valid') else 'WAIT'}"
            f" (score {ise.get('score', 0)})"
        )
    else:
        segs = analysis.get("phase_segments") or []
        if segs:
            path = " → ".join(s["phase"][:5] for s in segs[-5:])
            lines.append(f"Cycle: {path}")

    st = analysis.get("structure_1h") or {}
    if st.get("note"):
        lines.append(f"Struct: {st['note']}")

    rng = analysis.get("range")
    if rng:
        lines.append(
            f"Range: {rng['low']:.5f} – {rng['high']:.5f}"
            f"{' (compressed)' if rng.get('compressed') else ''}"
        )

    manip = analysis.get("manipulation")
    if manip:
        conf = "confirmed" if manip.get("confirmed") else "pending"
        lines.append(f"Manip: {manip['side']} → hint {manip['direction_hint']} [{conf}]")

    n_fvg = len(analysis.get("fvgs") or [])
    n_ob = len(analysis.get("order_blocks") or [])
    n_idm = len(analysis.get("inducements") or [])
    unmit = sum(1 for z in (analysis.get("inducements") or []) if not z.get("mitigated"))
    lines.append(f"Zones: {n_fvg} FVG · {n_ob} OB · {n_idm} IDM ({unmit} unmitigated)")

    pairs = pair_idm_with_extreme_ob(
        analysis.get("inducements") or [], analysis.get("order_blocks") or []
    )
    if pairs:
        p = pairs[0]
        lines.append(f"Setup: {p['direction']} IDM→OB (see chart)")
    else:
        lines.append("Setup: no clean IDM→OB pair")

    bp = analysis.get("best_pattern")
    if bp:
        lines.append(f"Pattern: {bp.name} ({bp.bias}) {bp.confidence:.0f}%")
        if bp.note:
            lines.append(f"  {bp.note[:120]}")

    vp = analysis.get("volume_profile")
    if vp:
        lines.append(
            f"POC {vp['poc_price']:.5f} | VA {vp['value_area_low']:.5f}–{vp['value_area_high']:.5f}"
        )

    if analysis.get("entry_notes"):
        lines.append("Entry: " + " · ".join(analysis["entry_notes"][:2]))

    if analysis.get("htf_note"):
        lines.append(f"HTF: {analysis['htf_note']}")

    lines.append("📷 Chart = FVG/OB/IDM/range + ISE structure (full story)")
    return "\n".join(lines)
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


from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from structure_engine import analyse_structure, find_swings
# [merged] was: from smc_zones import detect_fvgs, detect_order_blocks
import data


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
