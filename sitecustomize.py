"""Runtime fixes for the Trendline strategy.

20 SMA is Median Price: (High + Low) / 2.
The SMA slope controls whether the master trendline is ascending support or
 descending resistance. A flat SMA means no diagonal trendline.

S/R is only an active setup when the level is actually close enough to price
to be tradable. A distant pivot cluster must never become a 35% S/R setup.
"""

import numpy as np


def _sma20(df):
    return ((df["High"].astype(float) + df["Low"].astype(float)) / 2.0).rolling(20, min_periods=20).mean()


def _atr_at(df, i):
    try:
        v = float(df["ATR"].iloc[int(i)])
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    try:
        return max(float(df["High"].iloc[int(i)] - df["Low"].iloc[int(i)]), 1e-9)
    except Exception:
        return 1e-9


def _sma_state(df):
    sma = _sma20(df)
    valid = sma.dropna()
    if len(valid) < 8:
        return "FLAT", sma, 0.0
    atr = _atr_at(df, len(df) - 1)
    slope = (float(valid.iloc[-1]) - float(valid.iloc[-6])) / 5.0
    threshold = max(atr * 0.06, abs(float(valid.iloc[-1])) * 0.00015)
    if slope > threshold:
        return "RISING", sma, slope
    if slope < -threshold:
        return "FALLING", sma, slope
    return "FLAT", sma, slope


def _establishing_cross(df, sma, state, lookback=100):
    if state not in ("RISING", "FALLING"):
        return None
    close = df["Close"].astype(float).to_numpy()
    s = sma.to_numpy()
    start = max(1, len(df) - lookback)
    crosses = []
    for i in range(start, len(df)):
        if not (np.isfinite(s[i]) and np.isfinite(s[i - 1])):
            continue
        if state == "RISING" and close[i - 1] <= s[i - 1] and close[i] > s[i]:
            crosses.append(i)
        elif state == "FALLING" and close[i - 1] >= s[i - 1] and close[i] < s[i]:
            crosses.append(i)
    if crosses:
        return crosses[0]
    for i in range(start, len(df) - 2):
        if not np.isfinite(s[i]):
            continue
        if state == "RISING" and np.all(close[i:i + 3] > s[i:i + 3]):
            return i
        if state == "FALLING" and np.all(close[i:i + 3] < s[i:i + 3]):
            return i
    return start


def _build_sma_line(df, state, sma, sma_slope):
    if state not in ("RISING", "FALLING") or len(df) < 25:
        return None
    n = len(df)
    cross = _establishing_cross(df, sma, state)
    if cross is None:
        return None

    lo = max(0, cross - 18)
    hi = min(n - 1, cross + 4)
    if state == "RISING":
        vals = df["Low"].astype(float).to_numpy()
        anchor = lo + int(np.argmin(vals[lo:hi + 1]))
        role = "support"
        slope = abs(float(sma_slope))
    else:
        vals = df["High"].astype(float).to_numpy()
        anchor = lo + int(np.argmax(vals[lo:hi + 1]))
        role = "resistance"
        slope = -abs(float(sma_slope))

    if slope <= 1e-12 if state == "RISING" else slope >= -1e-12:
        slope = (float(sma.iloc[-1]) - float(sma.iloc[-6])) / 5.0
        slope = abs(slope) if state == "RISING" else -abs(slope)

    x0 = int(anchor)
    y0 = float(vals[anchor])
    x1 = n - 1

    # Remove SMA lag vertically without changing its slope.
    actual = float(vals[-1])
    predicted = y0 + slope * (x1 - x0)
    atr = _atr_at(df, x1)
    y0 += float(np.clip(actual - predicted, -1.25 * atr, 1.25 * atr))
    y1 = y0 + slope * (x1 - x0)

    touches = 0
    violations = 0
    touch_points = []
    for i in range(x0, n):
        line = y0 + slope * (i - x0)
        a = _atr_at(df, i)
        if abs(float(vals[i]) - line) <= a * 0.60:
            touches += 1
            touch_points.append({"index": i, "price": float(vals[i])})
        if role == "support" and float(df["Close"].iloc[i]) < line - a * 0.35:
            violations += 1
        if role == "resistance" and float(df["Close"].iloc[i]) > line + a * 0.35:
            violations += 1

    return {
        "x0": x0, "y0": float(y0), "x1": x1, "y1": float(y1), "y_end": float(y1),
        "slope": float(slope), "touches": int(touches), "violations": int(violations),
        "confirmed": touches >= 2, "quality": "confirmed" if touches >= 2 else "developing",
        "kind": role, "method": "20SMA_MEDIAN_PRICE_SLOPE",
        "establishing_cross": int(cross), "establishing_anchor": int(anchor),
        "sma_slope": float(slope), "sma_period": 20, "sma_applied_price": "median",
        "touch_points": touch_points,
    }


def _install():
    try:
        import strategies
    except Exception as exc:
        print(f"[trendline_sma] strategies unavailable: {exc!r}")
        return

    original_build = getattr(strategies, "build_trendline_family", None)
    if original_build is not None and not getattr(original_build, "_sma20_geometry_wrapped", False):
        def wrapped_build(df, max_lines=4, lookback_bars=60):
            family = original_build(df, max_lines=max_lines, lookback_bars=lookback_bars)
            if not family or family.get("error"):
                return family

            state, sma, sma_slope = _sma_state(df)
            family["sma_series"] = sma
            family["sma_direction"] = state
            family["sma_slope"] = float(sma_slope)
            family["sma_applied_price"] = "median"
            family["sma_period"] = 20

            line = _build_sma_line(df, state, sma, sma_slope)
            if state == "FLAT" or line is None:
                family["uptrends"] = []
                family["downtrends"] = []
                family["family_lines"] = []
                family["channel"] = None
                family["master_trendline"] = None
                family["master_role"] = "none"
                family["family_kind"] = "none"
                family["mode"] = "sr"
                family["direction"] = "NEUTRAL"
                family["trendline_color_state"] = "NEUTRAL"
                family["master_decision"] = "20 SMA FLAT — map pattern / horizontal S/R"
                return family

            if line["kind"] == "support":
                family["uptrends"] = [line]
                family["downtrends"] = []
                family["family_kind"] = "ascending"
                family["direction"] = "BUY"
                family["trendline_color_state"] = "BULLISH"
            else:
                family["uptrends"] = []
                family["downtrends"] = [line]
                family["family_kind"] = "descending"
                family["direction"] = "SELL"
                family["trendline_color_state"] = "BEARISH"

            family["family_lines"] = [line]
            family["channel"] = None
            family["master_trendline"] = line
            family["master_role"] = line["kind"]
            family["primary_quality"] = line["quality"]
            family["primary_touches"] = line["touches"]
            family["bias_touch_points"] = line.get("touch_points", [])
            family["master_line_value"] = float(line["y_end"])
            family["mode"] = "lines"
            family["reasons"] = list(family.get("reasons") or []) + [
                "20 SMA median-price slope is the master trendline geometry."
            ]

            close = float(df["Close"].iloc[-1])
            broken = (line["kind"] == "support" and close < line["y_end"]) or (line["kind"] == "resistance" and close > line["y_end"])
            family["trendline_break_kind"] = "support_break_down" if line["kind"] == "support" and broken else ("resistance_break_up" if line["kind"] == "resistance" and broken else None)
            family["trendline_status"] = "BROKEN" if broken else "INTACT"

            if broken and hasattr(strategies, "_grade_breakout"):
                try:
                    bk = family["trendline_break_kind"]
                    family["breakout_grade"] = strategies._grade_breakout(df, line, bk, len(df))
                except Exception:
                    pass
            return family

        wrapped_build._sma20_geometry_wrapped = True
        wrapped_build._original = original_build
        strategies.build_trendline_family = wrapped_build

    # Fix the false S/R setup shown in the report: a pivot cluster can exist
    # on the chart, but it is NOT an active S/R setup when it is far away.
    original_sr = getattr(strategies, "_sr_setup_confidence", None)
    if original_sr is not None and not getattr(original_sr, "_distance_gate_wrapped", False):
        def wrapped_sr(df, horizontal_levels, close, atr_now):
            if not horizontal_levels or atr_now is None or not np.isfinite(float(atr_now)) or float(atr_now) <= 0:
                return None
            best = min(horizontal_levels, key=lambda lvl: abs(float(close) - float(lvl.get("price", close))))
            dist = abs(float(close) - float(best.get("price", close)))
            dist_atr = dist / float(atr_now)
            # A level more than 1.2 ATR away is context, not an active setup.
            if dist_atr > 1.2:
                return None
            touches = int(best.get("touches", 0))
            if touches < 2:
                return None
            prox = 30 if dist_atr <= 0.15 else 20 if dist_atr <= 0.5 else 8
            quality = {"confirmed": 15, "crowded": 8, "unconfirmed": 4}.get(best.get("quality"), 4)
            confidence = max(0, min(100, min(35, touches * 8) + quality + prox + 15))
            return {
                "confidence": confidence,
                "level": best,
                "bias": "BUY" if best.get("side") == "support" else "SELL",
                "distance_atr": round(dist_atr, 2),
            }
        wrapped_sr._distance_gate_wrapped = True
        wrapped_sr._original = original_sr
        strategies._sr_setup_confidence = wrapped_sr


_install()
