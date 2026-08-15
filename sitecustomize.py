"""20-SMA trendline mapper.

The 20 SMA is the geometry controller for the Trendline strategy:
- median price = (High + Low) / 2
- rising SMA -> ascending support trendline
- falling SMA -> descending resistance trendline
- flat SMA -> no diagonal trendline; keep pattern/SR mapping

The SMA supplies the slope. Price supplies the vertical anchor from the
establishing leg, removing SMA lag while preserving the trader's hand-drawn
ascending/descending structure.
"""

import numpy as np
import pandas as pd


def _sma20(df):
    return ((df["High"].astype(float) + df["Low"].astype(float)) / 2.0).rolling(
        20, min_periods=20
    ).mean()


def _sma_direction(df):
    sma = _sma20(df)
    valid = sma.dropna()
    if len(valid) < 8:
        return "FLAT", sma, 0.0

    try:
        atr = float(df["ATR"].iloc[-1])
    except Exception:
        atr = float((df["High"] - df["Low"]).tail(14).mean())
    if not np.isfinite(atr) or atr <= 0:
        atr = max(abs(float(valid.iloc[-1])) * 0.0001, 1e-9)

    lookback = 5
    slope = (float(valid.iloc[-1]) - float(valid.iloc[-1 - lookback])) / lookback
    threshold = max(atr * 0.06, abs(float(valid.iloc[-1])) * 0.00015)

    if slope > threshold:
        return "RISING", sma, slope
    if slope < -threshold:
        return "FALLING", sma, slope
    return "FLAT", sma, slope


def _establishing_cross(df, sma, state, lookback=80):
    """Locate the leg that establishes price on the trend side of the SMA."""
    if state not in ("RISING", "FALLING"):
        return None

    close = df["Close"].astype(float).to_numpy()
    s = sma.to_numpy()
    start = max(1, len(df) - lookback)
    crosses = []

    for i in range(start, len(df)):
        if not np.isfinite(s[i]) or not np.isfinite(s[i - 1]):
            continue
        if state == "RISING" and close[i - 1] <= s[i - 1] and close[i] > s[i]:
            crosses.append(i)
        elif state == "FALLING" and close[i - 1] >= s[i - 1] and close[i] < s[i]:
            crosses.append(i)

    if crosses:
        # The first cross of the current sustained leg is the structural start.
        return crosses[0]

    # If the move began before the visible window, find the first sustained run.
    for i in range(start, len(df) - 2):
        if not np.isfinite(s[i]):
            continue
        if state == "RISING" and np.all(close[i:i + 3] > s[i:i + 3]):
            return i
        if state == "FALLING" and np.all(close[i:i + 3] < s[i:i + 3]):
            return i

    return start


def _atr_at(df, i):
    try:
        value = float(df["ATR"].iloc[int(i)])
        if np.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    try:
        return max(float(df["High"].iloc[int(i)] - df["Low"].iloc[int(i)]), 1e-9)
    except Exception:
        return 1e-9


def _forced_sma_line(df, state, sma, sma_slope, strategies):
    """Build a real drawable trendline even when the legacy pivot fitter fails.

    Critical rule: the line slope comes from the median-price 20 SMA. The
    establishing price leg supplies the anchor. A later structural pullback
    is used only to vertically calibrate the line; it can never change slope
    or turn an ascending line into a descending one.
    """
    if state not in ("RISING", "FALLING") or df is None or len(df) < 25:
        return None

    n = len(df)
    cross = _establishing_cross(df, sma, state)
    if cross is None:
        return None

    launch_lo = max(0, int(cross) - 18)
    launch_hi = min(n - 1, int(cross) + 4)

    if state == "RISING":
        window = df["Low"].astype(float).iloc[launch_lo:launch_hi + 1].to_numpy()
        anchor = launch_lo + int(np.argmin(window))
        anchor_price = float(df["Low"].iloc[anchor])
        role = "support"
        values = df["Low"].astype(float).to_numpy()
        candidates = [
            i for i in range(max(int(cross) + 3, anchor + 3), n)
            if values[i] > anchor_price
        ]
    else:
        window = df["High"].astype(float).iloc[launch_lo:launch_hi + 1].to_numpy()
        anchor = launch_lo + int(np.argmax(window))
        anchor_price = float(df["High"].iloc[anchor])
        role = "resistance"
        values = df["High"].astype(float).to_numpy()
        candidates = [
            i for i in range(max(int(cross) + 3, anchor + 3), n)
            if values[i] < anchor_price
        ]

    endpoint = candidates[-1] if candidates else n - 1

    # THE CORE IMPLEMENTATION: slope is the actual 20-SMA slope.
    slope = float(sma_slope)
    if state == "RISING":
        slope = abs(slope)
        if slope <= 1e-12:
            slope = max((float(sma.iloc[-1]) - float(sma.iloc[-6])) / 5.0, 1e-9)
    else:
        slope = -abs(slope)
        if slope >= -1e-12:
            slope = min((float(sma.iloc[-1]) - float(sma.iloc[-6])) / 5.0, -1e-9)

    x0 = int(anchor)
    x1 = n - 1
    y0 = anchor_price

    # Keep the exact SMA-derived slope, but vertically align the rail with the
    # latest structural pullback when that pullback is reasonably close.
    actual = float(values[endpoint])
    predicted = y0 + slope * (endpoint - x0)
    atr = _atr_at(df, endpoint)
    y0 += float(np.clip(actual - predicted, -atr * 1.25, atr * 1.25))
    y1 = y0 + slope * (x1 - x0)

    touches = 0
    violations = 0
    for i in range(x0, n):
        line = y0 + slope * (i - x0)
        a = _atr_at(df, i)
        tolerance = a * 0.60
        if role == "support":
            if abs(float(df["Low"].iloc[i]) - line) <= tolerance:
                touches += 1
            if float(df["Close"].iloc[i]) < line - a * 0.35:
                violations += 1
        else:
            if abs(float(df["High"].iloc[i]) - line) <= tolerance:
                touches += 1
            if float(df["Close"].iloc[i]) > line + a * 0.35:
                violations += 1

    return {
        "x0": x0,
        "y0": float(y0),
        "x1": x1,
        "y1": float(y1),
        "y_end": float(y1),
        "slope": float(slope),
        "touches": int(touches),
        "violations": int(violations),
        "confirmed": touches >= 2,
        "quality": "confirmed" if touches >= 2 else "developing",
        "kind": role,
        "method": "20SMA_MEDIAN_PRICE_SLOPE",
        "establishing_cross": int(cross),
        "establishing_anchor": int(anchor),
        "sma_slope": float(slope),
        "sma_period": 20,
        "sma_applied_price": "median",
    }


def _install():
    try:
        import strategies
    except Exception as exc:
        print(f"[trendline_sma] strategies unavailable: {exc!r}")
        return

    original = getattr(strategies, "build_trendline_family", None)
    if original is None or getattr(original, "_sma20_geometry_wrapped", False):
        return

    def _wrapped(df, max_lines=4, lookback_bars=60):
        family = original(df, max_lines=max_lines, lookback_bars=lookback_bars)
        if not family or family.get("error"):
            return family

        state, sma, sma_slope = _sma_direction(df)
        family["sma_series"] = sma
        family["sma_direction"] = state
        family["sma_slope"] = float(sma_slope)
        family["sma_applied_price"] = "median"
        family["sma_period"] = 20

        line = _forced_sma_line(df, state, sma, sma_slope, strategies)

        if state == "FLAT" or line is None:
            # A flat SMA is not a trend. Do not manufacture a diagonal line.
            family["uptrends"] = []
            family["downtrends"] = []
            family["family_lines"] = []
            family["channel"] = None
            family["master_trendline"] = None
            family["master_role"] = "none"
            family["family_kind"] = "none"
            family["direction"] = "NEUTRAL"
            family["trendline_color_state"] = "NEUTRAL"
            family["master_decision"] = "20 SMA FLAT — map pattern / horizontal S/R"
            family["mode"] = "sr"
            return family

        role = line["kind"]

        # This is the critical renderer hand-off. Do not depend on the old
        # pivot engine successfully producing a line: give chart_engine the
        # actual SMA-guided line directly.
        if role == "support":
            family["uptrends"] = [line]
            family["downtrends"] = []
            family["direction"] = "BUY"
            family["family_kind"] = "ascending"
        else:
            family["uptrends"] = []
            family["downtrends"] = [line]
            family["direction"] = "SELL"
            family["family_kind"] = "descending"

        family["family_lines"] = [line]
        family["channel"] = None
        family["mode"] = "lines"
        family["master_trendline"] = line
        family["master_role"] = role
        family["primary_quality"] = line["quality"]
        family["primary_touches"] = line["touches"]
        family["master_line_value"] = float(
            strategies._line_value(line["x0"], line["y0"], line["x1"], line["y1"], len(df) - 1)
        )
        family["bias_touch_points"] = strategies._touch_points(
            df, line["x0"], line["y0"], line["x1"], line["y1"], role
        )
        family["trendline_color_state"] = "BULLISH" if role == "support" else "BEARISH"
        family["strength"] = max(int(family.get("strength") or 0), 55)
        family["reasons"] = list(family.get("reasons") or []) + [
            f"20 SMA median-price slope controls the {('ascending support' if role == 'support' else 'descending resistance')} trendline."
        ]

        # Break/retest is now evaluated against the same line that is drawn.
        close = float(df["Close"].iloc[-1])
        line_now = family["master_line_value"]
        break_kind = None
        if role == "support" and close < line_now:
            break_kind = "support_break_down"
        elif role == "resistance" and close > line_now:
            break_kind = "resistance_break_up"

        breakout = None
        retest = {"status": "INTACT", "note": "No confirmed trendline break."}
        try:
            if break_kind:
                breakout = strategies._grade_breakout(df, line, break_kind, len(df))
                retest = strategies._trendline_retest_state(df, line, breakout, break_kind)
        except Exception as exc:
            retest = {"status": "UNAVAILABLE", "note": str(exc)}

        family["breakout_grade"] = breakout
        family["trendline_retest"] = retest
        family["trendline_break_kind"] = break_kind
        family["trendline_status"] = retest.get("status", "INTACT")
        family["prefer_retest_entry"] = retest.get("status") == "BREAK_CONFIRMED"
        family["master_entry_ready"] = retest.get("status") == "BREAK_RETEST_CONFIRMED"

        return family

    _wrapped._sma20_geometry_wrapped = True
    _wrapped._original = original
    strategies.build_trendline_family = _wrapped
    print("[trendline_sma] 20-SMA median-price geometry installed")


_install()
