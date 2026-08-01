"""
patterns.py
================
Professional-grade price-action pattern detection engine.

Design goals:
  - Detect swing highs/lows (fractal pivots).
  - Scan for common chart patterns (Head & Shoulders, Flags, Triangles, Wedges, etc.).
  - Return trigger lines and key points for clear execution mapping.
  - Prioritize continuation structures (Flags/Pennants) per strategic requirements.
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. SWING PIVOT DETECTION
# ----------------------------------------------------------------------------
def find_pivots(df, left=3, right=3):
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
    (x1, y1), (x2, y2) = p1, p2
    if x2 == x1:
        return 0.0, y1
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def _pct(a, b):
    if b == 0:
        return 0.0
    return (a - b) / abs(b)


def _atr(df):
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        return float(df['ATR'].iloc[-1])
    return float((df['High'] - df['Low']).tail(14).mean())


class Pattern:
    def __init__(self, name, category, bias, trigger_price, trigger_line,
                 key_points, confidence, note):
        self.name = name                 
        self.category = category         
        self.bias = bias                 
        self.trigger_price = trigger_price   
        self.trigger_line = trigger_line     
        self.key_points = key_points          
        self.confidence = confidence      
        self.note = note                  

    def to_dict(self):
        return {
            "name": self.name, "category": self.category, "bias": self.bias,
            "trigger_price": self.trigger_price, "trigger_line": self.trigger_line,
            "key_points": self.key_points, "confidence": self.confidence, "note": self.note,
        }


# ----------------------------------------------------------------------------
# 3. INDIVIDUAL PATTERN DETECTORS
# ----------------------------------------------------------------------------

def detect_double_top(df, ph, pl):
    if len(ph) < 2:
        return None
    i2, i1 = ph[-1], ph[-2]
    h1, h2 = df['High'].iloc[i1], df['High'].iloc[i2]
    if abs(_pct(h2, h1)) > 0.006:
        return None
    between_lows = [p for p in pl if i1 < p < i2]
    if not between_lows:
        return None
    trough_i = min(between_lows, key=lambda p: df['Low'].iloc[p])
    neckline = float(df['Low'].iloc[trough_i])
    current = float(df['Close'].iloc[-1])
    if current > max(h1, h2):
        return None
    conf = 60 + min(15, (1 - abs(_pct(h2, h1)) * 100) * 10)
    return Pattern(
        "Double Top", "reversal", "SELL",
        trigger_price=neckline,
        trigger_line=[(i1, neckline), (i2, neckline)],
        key_points=[(i1, h1, "Top 1"), (i2, h2, "Top 2"), (trough_i, neckline, "Neckline")],
        confidence=conf,
        note=f"Two near-equal highs ({h1:.5f} / {h2:.5f}) with neckline at {neckline:.5f}."
    )


def detect_double_bottom(df, ph, pl):
    if len(pl) < 2:
        return None
    i2, i1 = pl[-1], pl[-2]
    l1, l2 = df['Low'].iloc[i1], df['Low'].iloc[i2]
    if abs(_pct(l2, l1)) > 0.006:
        return None
    between_highs = [p for p in ph if i1 < p < i2]
    if not between_highs:
        return None
    peak_i = max(between_highs, key=lambda p: df['High'].iloc[p])
    neckline = float(df['High'].iloc[peak_i])
    current = float(df['Close'].iloc[-1])
    if current < min(l1, l2):
        return None
    conf = 60 + min(15, (1 - abs(_pct(l2, l1)) * 100) * 10)
    return Pattern(
        "Double Bottom", "reversal", "BUY",
        trigger_price=neckline,
        trigger_line=[(i1, neckline), (i2, neckline)],
        key_points=[(i1, l1, "Bottom 1"), (i2, l2, "Bottom 2"), (peak_i, neckline, "Neckline")],
        confidence=conf,
        note=f"Two near-equal lows ({l1:.5f} / {l2:.5f}) with neckline at {neckline:.5f}."
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
    if abs(_pct(rs, ls)) > 0.03:
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
        return None
    return Pattern(
        "Head and Shoulders", "reversal", "SELL",
        trigger_price=float(neckline_now),
        trigger_line=[(t1, float(df['Low'].iloc[t1])), (t2, float(df['Low'].iloc[t2]))],
        key_points=[(i1, ls, "L Shoulder"), (i2, head, "Head"), (i3, rs, "R Shoulder")],
        confidence=72,
        note=f"Classic H&S structure with neckline at {neckline_now:.5f}."
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
        note=f"Inverse H&S structure with neckline at {neckline_now:.5f}."
    )


def _fit_trend(points):
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if len(xs) < 2 or np.all(xs == xs[0]):
        return 0.0, float(ys[-1]) if len(ys) else 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def detect_triangle_or_wedge(df, ph, pl, lookback=60):
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
    up_norm = (up_slope * lookback) / avg_price
    lo_norm = (lo_slope * lookback) / avg_price
    FLAT = 0.003

    x_now = n - 1
    upper_now = up_slope * x_now + up_intercept
    lower_now = lo_slope * x_now + lo_intercept
    if upper_now <= lower_now:
        return None

    current = float(df['Close'].iloc[-1])
    line = [(upper_pts[0][0], upper_pts[0][1]), (upper_pts[-1][0], upper_pts[-1][1])]
    lower_line = [(lower_pts[0][0], lower_pts[0][1]), (lower_pts[-1][0], lower_pts[-1][1])]

    if abs(up_norm) < FLAT and lo_norm > FLAT:
        return Pattern(
            "Ascending Triangle", "continuation", "BUY",
            trigger_price=float(upper_now), trigger_line=line,
            key_points=[(p, y, "Resistance") for p, y in upper_pts] + [(p, y, "Higher Low") for p, y in lower_pts],
            confidence=65, note=f"Flat resistance at {upper_now:.5f} with rising support."
        )
    if up_norm < -FLAT and abs(lo_norm) < FLAT:
        return Pattern(
            "Descending Triangle", "continuation", "SELL",
            trigger_price=float(lower_now), trigger_line=lower_line,
            key_points=[(p, y, "Support") for p, y in lower_pts] + [(p, y, "Lower High") for p, y in upper_pts],
            confidence=65, note=f"Flat support at {lower_now:.5f} with falling resistance."
        )
    if up_norm > FLAT and lo_norm > FLAT and lo_norm > up_norm:
        return Pattern(
            "Rising Wedge", "reversal", "SELL",
            trigger_price=float(lower_now), trigger_line=lower_line,
            key_points=[(p, y, "Upper") for p, y in upper_pts] + [(p, y, "Lower") for p, y in lower_pts],
            confidence=63, note="Both lines rising but converging — bearish reversal potential."
        )
    if up_norm < -FLAT and lo_norm < -FLAT and up_norm < lo_norm:
        return Pattern(
            "Falling Wedge", "reversal", "BUY",
            trigger_price=float(upper_now), trigger_line=line,
            key_points=[(p, y, "Upper") for p, y in upper_pts] + [(p, y, "Lower") for p, y in lower_pts],
            confidence=63, note="Both lines falling but converging — bullish reversal potential."
        )
    if up_norm < -FLAT and lo_norm > FLAT:
        bias = "BUY" if current >= (upper_now + lower_now) / 2 else "SELL"
        return Pattern(
            "Symmetrical Triangle", "continuation", bias,
            trigger_price=float(upper_now if bias == "BUY" else lower_now),
            trigger_line=line if bias == "BUY" else lower_line,
            key_points=[(p, y, "Upper") for p, y in upper_pts] + [(p, y, "Lower") for p, y in lower_pts],
            confidence=55, note="Converging range structure."
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
        return None
    current = float(df['Close'].iloc[-1])
    bias = "BUY" if current <= bottom * 1.003 else ("SELL" if current >= top * 0.997 else None)
    if bias is None:
        return None
    return Pattern(
        "Rectangle / Range", "continuation", bias,
        trigger_price=top if bias == "SELL" else bottom,
        trigger_line=[(recent_ph[0], top), (recent_ph[-1], top)] if bias == "SELL" else [(recent_pl[0], bottom), (recent_pl[-1], bottom)],
        key_points=[(p, df['High'].iloc[p], "High") for p in recent_ph] + [(p, df['Low'].iloc[p], "Low") for p in recent_pl],
        confidence=52, note=f"Range between {bottom:.5f} and {top:.5f}."
    )


def detect_flag_or_pennant(df, lookback_pole=20, lookback_flag=15):
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

    if abs(pole_move) < atr * 3.0:
        return None
    pole_up = pole_move > 0

    closes = pole_df['Close'].values
    diffs = np.diff(closes)
    if len(diffs) == 0:
        return None
    clean_frac = np.mean(diffs > 0) if pole_up else np.mean(diffs < 0)
    if clean_frac < 0.55:
        return None

    flag_x = np.arange(len(flag_df))
    flag_slope, flag_intercept = np.polyfit(flag_x, flag_df['Close'].values, 1) if len(flag_df) >= 2 else (0, flag_df['Close'].iloc[-1])
    flag_range = float(flag_df['High'].max() - flag_df['Low'].min())
    retrace = flag_range / pole_range

    if retrace > 0.65:
        return None

    flag_norm_slope = (flag_slope * len(flag_df)) / (float(np.mean(flag_df['Close'])) or 1.0)
    counter_slope_ok = (flag_norm_slope < 0.004) if pole_up else (flag_norm_slope > -0.004)
    if not counter_slope_ok:
        return None

    upper = float(flag_df['High'].max())
    lower = float(flag_df['Low'].min())
    is_pennant = retrace < 0.35 and abs(flag_norm_slope) < 0.002

    name = ("Bull Flag" if pole_up else "Bear Flag") if not is_pennant else ("Bullish Pennant" if pole_up else "Bearish Pennant")
    bias = "BUY" if pole_up else "SELL"
    trigger_price = upper if pole_up else lower
    trigger_line = [(flag_start, upper if pole_up else lower), (n - 1, upper if pole_up else lower)]

    conf = 75 + min(15, clean_frac * 15) - min(10, retrace * 15)
    return Pattern(
        name, "continuation", bias,
        trigger_price=float(trigger_price),
        trigger_line=trigger_line,
        key_points=[(pole_start, float(pole_df['Close'].iloc[0]), "Pole Start"), (flag_start, float(pole_df['Close'].iloc[-1]), "Pole End")],
        confidence=float(np.clip(conf, 55, 92)),
        note=f"High-conviction {name} setup with trigger near {trigger_price:.5f}."
    )


# ----------------------------------------------------------------------------
# 4. TOP-LEVEL SCANNER
# ----------------------------------------------------------------------------
_PRIORITY = {
    "Bull Flag": 100, "Bear Flag": 100, "Bullish Pennant": 98, "Bearish Pennant": 98,
    "Ascending Triangle": 85, "Descending Triangle": 85, "Symmetrical Triangle": 80,
    "Rising Wedge": 75, "Falling Wedge": 75,
    "Head and Shoulders": 72, "Inverse Head and Shoulders": 72,
    "Double Top": 68, "Double Bottom": 68,
    "Triple Top": 66, "Triple Bottom": 66,
    "Rectangle / Range": 50,
}


def scan_all_patterns(df, left=3, right=3):
    ph, pl = find_pivots(df, left=left, right=right)
    ph = _dedupe_adjacent(ph, min_gap=left + right)
    pl = _dedupe_adjacent(pl, min_gap=left + right)

    detected = []
    for fn in (detect_flag_or_pennant,):
        try:
            res = fn(df)
        except Exception:
            res = None
        if res:
            detected.append(res)

    for fn in (detect_double_top, detect_double_bottom, detect_triple_top,
               detect_triple_bottom, detect_head_shoulders,
               detect_inverse_head_shoulders):
        try:
            res = fn(df, ph, pl)
        except Exception:
            res = None
        if res:
            detected.append(res)

    for fn in (detect_triangle_or_wedge, detect_rectangle):
        try:
            res = fn(df, ph, pl)
        except Exception:
            res = None
        if res:
            detected.append(res)

    if not detected:
        return None, []

    detected.sort(key=lambda p: (_PRIORITY.get(p.name, 40) + p.confidence), reverse=True)
    return detected[0], detected
