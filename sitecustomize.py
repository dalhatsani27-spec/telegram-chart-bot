"""Authoritative 20-SMA median-price trendline mapper.

Trendline direction is never inferred from arbitrary pivot pairs:
- rising 20 SMA -> ascending support
- falling 20 SMA -> descending resistance
- flat 20 SMA -> no diagonal trendline; map consolidation/SR
The establishing leg is the candle leg that crosses the 20 SMA.
"""
import numpy as np


def _install():
    try:
        import strategies
    except Exception as exc:
        print(f"[sma20_master] strategies unavailable: {exc!r}")
        return

    original = getattr(strategies, "build_trendline_family", None)
    if original is None or getattr(original, "_sma20_master", False):
        return

    def sma20(df):
        median = (df["High"].astype(float) + df["Low"].astype(float)) / 2.0
        return median.rolling(20, min_periods=20).mean()

    def atr(df, i):
        try:
            a = float(df["ATR"].iloc[i])
            if np.isfinite(a) and a > 0:
                return a
        except Exception:
            pass
        return max(float(df["High"].iloc[i] - df["Low"].iloc[i]), 1e-9)

    def state(sma, df):
        v = sma.dropna()
        if len(v) < 12:
            return "FLAT", 0.0
        lb = 8
        slope = (float(v.iloc[-1]) - float(v.iloc[-1-lb])) / lb
        threshold = max(atr(df, len(df)-1) * 0.035,
                        abs(float(v.iloc[-1])) * 0.00008)
        if slope > threshold:
            return "RISING", slope
        if slope < -threshold:
            return "FALLING", slope
        return "FLAT", slope

    def establishing_leg(df, sma, direction, lookback=100):
        close = df["Close"].astype(float).to_numpy()
        high = df["High"].astype(float).to_numpy()
        low = df["Low"].astype(float).to_numpy()
        sv = sma.to_numpy(float)
        n = len(df)
        start = max(1, n - lookback)
        crosses = []
        for i in range(start, n):
            if not (np.isfinite(sv[i]) and np.isfinite(sv[i-1])):
                continue
            if direction == "RISING" and close[i-1] <= sv[i-1] and close[i] > sv[i]:
                crosses.append(i)
            elif direction == "FALLING" and close[i-1] >= sv[i-1] and close[i] < sv[i]:
                crosses.append(i)
        cross = crosses[-1] if crosses else None
        if cross is None:
            return None

        # Walk back through the leg that was on the opposite side of SMA.
        leg_start = cross
        for j in range(cross-1, max(start, cross-35), -1):
            if not np.isfinite(sv[j]):
                continue
            opposite = close[j] <= sv[j] if direction == "RISING" else close[j] >= sv[j]
            if opposite:
                leg_start = j
            else:
                break

        lo = max(start, leg_start-3)
        hi = cross
        if direction == "RISING":
            anchor = lo + int(np.argmin(low[lo:hi+1]))
            return cross, anchor, "support", low
        anchor = lo + int(np.argmax(high[lo:hi+1]))
        return cross, anchor, "resistance", high

    def make_line(df, sma, direction, sma_slope):
        leg = establishing_leg(df, sma, direction)
        if leg is None:
            return None
        cross, anchor, role, extremes = leg
        n = len(df)
        # This is the critical rule: line slope comes ONLY from the 20 SMA.
        slope = abs(float(sma_slope)) if role == "support" else -abs(float(sma_slope))
        x0 = int(anchor)
        target = float(extremes[x0])
        sma_anchor = float(sma.iloc[x0])
        y0 = sma_anchor + (target - sma_anchor)
        # Preserve the manual anchor but prevent pathological displacement.
        y0 = sma_anchor + float(np.clip(y0 - sma_anchor, -3*atr(df, n-1), 3*atr(df, n-1)))
        y1 = y0 + slope * (n - 1 - x0)

        touches, violations, points = 0, 0, []
        for i in range(x0, n):
            line = y0 + slope * (i-x0)
            a = atr(df, i)
            if abs(float(extremes[i]) - line) <= 0.60*a:
                touches += 1
                points.append({"index": i, "price": float(extremes[i])})
            c = float(df["Close"].iloc[i])
            if role == "support" and c < line - 0.35*a:
                violations += 1
            if role == "resistance" and c > line + 0.35*a:
                violations += 1

        return {
            "x0": x0, "y0": y0, "x1": n-1, "y1": y1, "y_end": y1,
            "slope": slope, "touches": touches, "violations": violations,
            "confirmed": touches >= 2,
            "quality": "confirmed" if touches >= 2 else "developing",
            "kind": role, "method": "20SMA_MEDIAN_PRICE_ESTABLISHING_LEG",
            "establishing_cross": int(cross), "establishing_anchor": int(anchor),
            "sma_slope": slope, "sma_period": 20,
            "sma_applied_price": "median", "touch_points": points,
            "bars_since_last_touch": (n-1-points[-1]["index"]) if points else 999,
        }

    def wrapped(df, max_lines=4, lookback_bars=60):
        # Original function remains available for non-diagonal context, but
        # its pivot-derived trendlines and horizontal S/R are not authoritative.
        family = original(df, max_lines=max_lines, lookback_bars=lookback_bars)
        if not family or family.get("error"):
            return family

        sma = sma20(df)
        direction, slope = state(sma, df)
        line = make_line(df, sma, direction, slope) if direction != "FLAT" else None

        family["sma_series"] = sma.values
        family["sma_direction"] = direction
        family["sma_slope"] = float(slope)
        family["sma_last"] = float(sma.dropna().iloc[-1]) if not sma.dropna().empty else None
        family["sma_period"] = 20
        family["sma_applied_price"] = "median"

        # Flat SMA means consolidation: NEVER manufacture a diagonal line.
        if line is None:
            family.update({
                "uptrends": [], "downtrends": [], "family_lines": [],
                "channel": None, "master_trendline": None,
                "master_role": "none", "family_kind": "none",
                "direction": "NEUTRAL", "strength": 40,
                "trendline_color_state": "NEUTRAL", "active_pattern": "none",
                "mode": "sr", "horizontal_levels": [], "sr_setup": None,
                "primary_touches": 0, "primary_quality": None,
                "breakout_grade": None,
                "trendline_retest": {"status":"INTACT","note":"20 SMA flat; no diagonal trendline."},
                "sma_confluence": {"relationship":"N/A","distance_atr":None,
                                   "status":"SMA FLAT","strength":"N/A"},
            })
            family["reasons"] = list(family.get("reasons") or []) + [
                "20 SMA FLAT — consolidation/pattern/SR; no diagonal trendline."
            ]
            return family

        if line["kind"] == "support":
            family["uptrends"], family["downtrends"] = [line], []
            family["family_kind"], family["direction"] = "ascending", "BUY"
            family["trendline_color_state"] = "BULLISH"
        else:
            family["uptrends"], family["downtrends"] = [], [line]
            family["family_kind"], family["direction"] = "descending", "SELL"
            family["trendline_color_state"] = "BEARISH"

        family.update({
            "family_lines": [line], "channel": None,
            "master_trendline": line, "master_role": line["kind"],
            "primary_quality": line["quality"], "primary_touches": line["touches"],
            "bias_touch_points": line["touch_points"], "master_line_value": line["y_end"],
            "mode": "lines", "active_pattern": "none",
            # Do not let an unrelated old pivot cluster become the winning setup.
            "horizontal_levels": [], "sr_setup": None,
            "trendline_status": "INTACT", "trendline_break_kind": None,
            "breakout_grade": None,
            "trendline_retest": {"status":"INTACT","note":"No confirmed trendline break."},
            "sma_confluence": {"relationship":"ALIGNED","distance_atr":None,
                               "status":"SMA-SLOPE-MASTER","strength":"STRONG"},
        })
        family["reasons"] = list(family.get("reasons") or []) + [
            f"20 SMA median-price {direction.lower()} — {line['kind']} trendline follows SMA slope.",
            "Trendline anchor is the establishing leg that broke the 20 SMA."
        ]
        return family

    wrapped._sma20_master = True
    wrapped._original = original
    strategies.build_trendline_family = wrapped
    print("[sma20_master] authoritative 20-SMA median-price trendline mapper installed")


_install()
