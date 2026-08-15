"""20-SMA-led trend/pattern map adapter.

The mapper follows the charting rule used by the manual Volatility 100 maps:
1) 20 SMA is median price (H+L)/2.
2) A sustained SMA slope establishes whether a trend exists.
3) The first leg that crosses/establishes beyond the 20 SMA supplies the
   starting structural anchor.
4) Rising SMA -> ascending support only. Falling SMA -> descending resistance
   only. Flat SMA -> no trendline; map consolidation/SR instead.
5) A trendline is sloped from the 20-SMA slope, but anchored to price
   structure, so the line stays close to the SMA instead of inheriting SMA lag.
"""

import numpy as np
import pandas as pd


def _sma20(df):
    return ((df["High"] + df["Low"]) / 2.0).rolling(20, min_periods=10).mean()


def _sma_state(df, lookback=8):
    sma = _sma20(df)
    valid = sma.dropna()
    if len(valid) < lookback + 2:
        return "FLAT", sma, 0.0
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and pd.notna(df["ATR"].iloc[-1]) else 0.0
    delta = float(valid.iloc[-1] - valid.iloc[-1 - lookback])
    threshold = max(atr * 0.08, abs(float(valid.iloc[-1])) * 0.00015)
    if delta > threshold:
        return "RISING", sma, delta / lookback
    if delta < -threshold:
        return "FALLING", sma, delta / lookback
    return "FLAT", sma, delta / lookback


def _atr(df, i):
    try:
        v = float(df["ATR"].iloc[i])
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    try:
        return max(float(df["High"].iloc[i] - df["Low"].iloc[i]), 1e-9)
    except Exception:
        return 1e-9


def _establishing_cross(df, sma, state, lookback=80):
    """Find the start of the current SMA-confirmed leg.

    We deliberately look for the price/SMA transition before choosing a
    pivot. This is the key difference from generic pivot-to-pivot trendline
    fitting: the line begins with the leg that actually established the
    trend above/below the 20 SMA.
    """
    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(float)
    s = sma.to_numpy(float)
    n = len(df)
    start = max(1, n - lookback)
    want_up = state == "RISING"
    crosses = []
    for i in range(start, n):
        if not (np.isfinite(s[i]) and np.isfinite(s[i - 1]) and np.isfinite(close[i]) and np.isfinite(close[i - 1])):
            continue
        if want_up and close[i - 1] <= s[i - 1] and close[i] > s[i]:
            crosses.append(i)
        elif not want_up and close[i - 1] >= s[i - 1] and close[i] < s[i]:
            crosses.append(i)
    if crosses:
        return crosses[-1]
    # If the current trend has persisted beyond the window, use the first
    # sustained same-side run available in the window.
    side = close > s if want_up else close < s
    for i in range(n - 10, start - 1, -1):
        if 0 <= i < n and bool(side[i]) and all(bool(side[j]) for j in range(i, min(i + 3, n)) if np.isfinite(s[j])):
            return i
    return start


def _structural_anchor(df, pivots, cross_i, state):
    """Choose the actual price anchor for the establishing leg."""
    if not pivots:
        return None
    want = "low" if state == "RISING" else "high"
    candidates = [p for p in pivots if p.get("type") == want]
    if not candidates:
        return None

    # The leg normally begins just before the SMA cross. Prefer the strongest
    # extreme in the 15-bar launch window, then fall back to the nearest one.
    lo = max(0, cross_i - 18)
    hi = min(len(df) - 1, cross_i + 3)
    window = [p for p in candidates if lo <= int(p["index"]) <= hi]
    if window:
        return (min(window, key=lambda p: p["price"]) if state == "RISING"
                else max(window, key=lambda p: p["price"]))

    before = [p for p in candidates if int(p["index"]) <= cross_i]
    if before:
        return before[-1]
    return candidates[0]


def _latest_structural_endpoint(pivots, anchor, state):
    if not anchor:
        return None
    idx = int(anchor["index"])
    if state == "RISING":
        pts = [p for p in pivots if p.get("type") == "low" and int(p["index"]) > idx and float(p["price"]) > float(anchor["price"])]
    else:
        pts = [p for p in pivots if p.get("type") == "high" and int(p["index"]) > idx and float(p["price"]) < float(anchor["price"])]
    if not pts:
        return None
    return pts[-1]


def _sma_guided_line(df, pivots, state, sma, sma_slope):
    if state not in ("RISING", "FALLING"):
        return None
    cross_i = _establishing_cross(df, sma, state)
    anchor = _structural_anchor(df, pivots, cross_i, state)
    endpoint = _latest_structural_endpoint(pivots, anchor, state) if anchor else None
    if anchor is None:
        return None

    # The 20 SMA supplies the slope. The price anchor supplies the vertical
    # position. This removes the SMA's lag without allowing a pivot fitter to
    # invent a slope that contradicts the live SMA direction.
    slope = float(sma_slope)
    if state == "RISING":
        slope = max(slope, abs(slope) * 0.25, 1e-9)
    else:
        slope = min(slope, -max(abs(slope) * 0.25, 1e-9))

    # If the SMA slope is extremely small but the state is still directional,
    # estimate it from the two structural points while preserving the sign.
    if abs(slope) < 1e-9 and endpoint is not None:
        raw = (float(endpoint["price"]) - float(anchor["price"])) / max(int(endpoint["index"]) - int(anchor["index"]), 1)
        slope = abs(raw) if state == "RISING" else -abs(raw)

    # Move the line vertically only enough to keep the latest structural
    # endpoint close to the same rail. Never change its slope sign.
    y0 = float(anchor["price"])
    if endpoint is not None:
        predicted = y0 + slope * (int(endpoint["index"]) - int(anchor["index"]))
        error = float(endpoint["price"]) - predicted
        limit = _atr(df, int(endpoint["index"])) * 0.75
        y0 += float(np.clip(error, -limit, limit))

    x0 = int(anchor["index"])
    x1 = int(endpoint["index"]) if endpoint is not None else min(len(df) - 1, x0 + 8)
    x1 = max(x1, x0 + 4)
    y1 = y0 + slope * (x1 - x0)
    y_end = y0 + slope * (len(df) - 1 - x0)

    kind = "support" if state == "RISING" else "resistance"
    touches = strategies._count_touches(df, x0, y0, x1, y1, kind, tol_atr=0.55)
    violations = strategies._count_violations(df, x0, y0, x1, y1, kind, tol_atr=0.35)
    quality = "confirmed" if touches >= 3 else "unconfirmed"

    return {
        "x0": x0, "y0": float(y0), "x1": x1, "y1": float(y1), "y_end": float(y_end),
        "slope": float(slope), "touches": max(2, int(touches)),
        "violations": int(violations), "confirmed": quality == "confirmed",
        "quality": quality, "kind": kind, "method": "20sma_establishing_leg",
        "establishing_cross": int(cross_i),
        "sma_slope": float(sma_slope),
    }


def _consolidation_pattern(df, pivots, sma, state):
    """Create a clean pattern map when the SMA is flat.

    This is deliberately a pattern/SR map, not a trendline classification.
    The renderer can use the two rails as a consolidation shape while the
    trade engine remains neutral until a breakout is confirmed.
    """
    if state != "FLAT" or len(pivots) < 6:
        return None
    n = len(df)
    recent = [p for p in pivots if int(p["index"]) >= max(0, n - 45)]
    highs = [p for p in recent if p.get("type") == "high"][-3:]
    lows = [p for p in recent if p.get("type") == "low"][-3:]
    if len(highs) < 2 or len(lows) < 2:
        return None

    def slope(a, b):
        return (float(b["price"]) - float(a["price"])) / max(int(b["index"]) - int(a["index"]), 1)
    hs = slope(highs[0], highs[-1])
    ls = slope(lows[0], lows[-1])
    atr = _atr(df, n - 1)
    threshold = max(atr * 0.04, abs(float(df["Close"].iloc[-1])) * 0.00005)
    high_flat = abs(hs) <= threshold
    low_flat = abs(ls) <= threshold

    if hs < -threshold and ls > threshold:
        name = "CONSOLIDATION"
    elif high_flat and ls > threshold:
        name = "ASCENDING CONSOLIDATION"
    elif hs < -threshold and low_flat:
        name = "DESCENDING CONSOLIDATION"
    elif high_flat and low_flat:
        name = "RANGE"
    else:
        return None

    upper = {"x0": int(highs[0]["index"]), "y0": float(highs[0]["price"]),
             "x1": int(highs[-1]["index"]), "y1": float(highs[-1]["price"]), "slope": float(hs)}
    lower = {"x0": int(lows[0]["index"]), "y0": float(lows[0]["price"]),
             "x1": int(lows[-1]["index"]), "y1": float(lows[-1]["price"]), "slope": float(ls)}
    upper["y_end"] = upper["y0"] + upper["slope"] * (n - 1 - upper["x0"])
    lower["y_end"] = lower["y0"] + lower["slope"] * (n - 1 - lower["x0"])
    return {"pattern": name, "upper": upper, "lower": lower,
            "apex_index": max(upper["x1"], lower["x1"]), "bias": "NEUTRAL"}


def _install():
    global strategies
    try:
        import strategies as _strategies
        strategies = _strategies
    except Exception as exc:
        print(f"[trendline_map] strategies unavailable: {exc!r}")
        return

    original = getattr(strategies, "build_trendline_family", None)
    if original is None or getattr(original, "_sma_map_wrapped", False):
        return

    def _wrapped(df, max_lines=4, lookback_bars=60):
        family = original(df, max_lines=max_lines, lookback_bars=lookback_bars)
        if not family or family.get("error"):
            return family

        pivots = family.get("pivots_full") or family.get("pivots") or []
        state, sma, sma_slope = _sma_state(df)
        line = _sma_guided_line(df, pivots, state, sma, sma_slope)
        pattern = _consolidation_pattern(df, pivots, sma, state)

        family["sma_series"] = sma
        family["sma_direction"] = state
        family["sma_applied_price"] = "median"
        family["sma_slope"] = float(sma_slope)
        family["sma_establishing_leg"] = _establishing_cross(df, sma, state) if state != "FLAT" else None

        if line is not None:
            # One and only one trendline. Never mix ascending support with
            # descending resistance in the trend map.
            role = "support" if state == "RISING" else "resistance"
            family["uptrends"] = [line] if role == "support" else []
            family["downtrends"] = [line] if role == "resistance" else []
            family["family_lines"] = [line]
            family["channel"] = None
            family["mode"] = "lines"
            family["master_trendline"] = line
            family["master_role"] = role
            family["family_kind"] = "ascending" if role == "support" else "descending"
            family["primary_quality"] = line["quality"]
            family["primary_touches"] = line["touches"]
            family["bias_touch_points"] = strategies._touch_points(df, line["x0"], line["y0"], line["x1"], line["y1"], role)
            family["trendline_color_state"] = "BULLISH" if role == "support" else "BEARISH"
            family["direction"] = "BUY" if role == "support" else "SELL"
            family["strength"] = max(int(family.get("strength") or 0), 55)
            family["active_pattern"] = "channel" if family.get("strength", 0) >= 55 else "none"
            family["mode"] = "lines"
            family["reasons"] = list(family.get("reasons") or []) + [
                f"20 SMA median-price state: {state}; trendline anchored from the establishing leg and slope-guided by the 20 SMA."
            ]

            # Recalculate lifecycle against the SAME master line.
            close = float(df["Close"].iloc[-1])
            line_now = strategies._line_value(line["x0"], line["y0"], line["x1"], line["y1"], len(df) - 1)
            break_kind = None
            if role == "support" and close < line_now:
                break_kind = "support_break_down"
            elif role == "resistance" and close > line_now:
                break_kind = "resistance_break_up"
            breakout = strategies._grade_breakout(df, line, break_kind, len(df)) if break_kind else None
            retest = strategies._trendline_retest_state(df, line, breakout, break_kind)
            family["breakout_grade"] = breakout
            family["trendline_retest"] = retest
            family["trendline_status"] = retest.get("status", "INTACT")
            family["trendline_break_kind"] = break_kind

            # Do not call the old geometry's opposite rail after this point.
            # It is the master line that controls the map.
            if retest.get("status") == "BREAK_RETEST_CONFIRMED":
                family["retest_entry_ready"] = True
                family["reasons"].append("Break + retest confirmed on the master trendline.")
            else:
                family["retest_entry_ready"] = False
        else:
            # Flat SMA = consolidation/SR map. Do not manufacture a diagonal
            # trendline merely because two pivots have a small slope.
            family["uptrends"] = []
            family["downtrends"] = []
            family["family_lines"] = []
            family["channel"] = None
            family["master_trendline"] = None
            family["master_role"] = "none"
            family["family_kind"] = "none"
            family["direction"] = "NEUTRAL"
            family["strength"] = min(int(family.get("strength") or 40), 45)
            family["trendline_color_state"] = "NEUTRAL"
            family["mode"] = "sr"
            family["master_decision"] = "20 SMA FLAT — consolidation/SR map"
            if pattern:
                family["wedge"] = pattern
                family["pattern_visual"] = {
                    "pattern_name": pattern["pattern"], "final_confidence": 70,
                    "upper_touches": 2, "lower_touches": 2, "fit_quality": 70
                }
                family["active_pattern"] = "wedge"
                family["pattern_confidence"] = 70
                family["reasons"] = list(family.get("reasons") or []) + [
                    f"20 SMA flat: {pattern['pattern']} mapped instead of a trendline."
                ]

        return family

    _wrapped._sma_map_wrapped = True
    _wrapped._original = original
    strategies.build_trendline_family = _wrapped
    print("[trendline_map] 20-SMA median-price structural map installed")


_install()

# Keep the existing confirmation/pullback adapter available. It runs after
# the mapper so historical pullbacks use the exact same master line.
try:
    import usercustomize
except Exception as exc:
    print(f"[trendline_map] usercustomize unavailable: {exc!r}")
