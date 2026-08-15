"""Trendline master adapter.

The trendline map is deliberately subordinate to the 20 SMA (median price):
RISING SMA -> only an ascending support trendline is eligible;
FALLING SMA -> only a descending resistance trendline is eligible;
FLAT SMA -> no diagonal trendline, use horizontal S/R instead.
"""

import numpy as np
import pandas as pd


def _sma20_median(df):
    return ((df["High"] + df["Low"]) / 2.0).rolling(20, min_periods=10).mean()


def _sma_dir(df):
    sma = _sma20_median(df).dropna()
    if len(sma) < 6:
        return "FLAT"
    atr = None
    try:
        atr = float(df["ATR"].iloc[-1])
    except Exception:
        pass
    delta = float(sma.iloc[-1] - sma.iloc[-6])
    threshold = atr * 0.08 if atr and atr > 0 else abs(float(sma.iloc[-1])) * 0.0004
    if delta > threshold:
        return "RISING"
    if delta < -threshold:
        return "FALLING"
    return "FLAT"


def _structural_impulse_anchor_pair(pivots, df, kind):
    """Select the structural HL/LH pair for the active SMA trend.

    The 20 SMA establishes the direction; the price structure supplies the
    actual hand-drawn anchors. This prevents a mathematically valid but
    directionally wrong rail from becoming the master trendline.
    """
    if df is None or len(pivots or []) < 4:
        return None

    sma_dir = _sma_dir(df)
    if (kind == "support" and sma_dir != "RISING") or (kind == "resistance" and sma_dir != "FALLING"):
        return None

    ordered = sorted(pivots, key=lambda p: int(p.get("index", 0)))
    lows = [p for p in ordered if p.get("type") == "low"]
    highs = [p for p in ordered if p.get("type") == "high"]

    def atr_at(index):
        try:
            if "ATR" in df.columns:
                value = float(df["ATR"].iloc[int(index)])
                if value > 0:
                    return value
            high = float(df["High"].iloc[int(index)])
            low = float(df["Low"].iloc[int(index)])
            return max(high - low, 1e-9)
        except Exception:
            return 1e-9

    if kind == "support":
        if len(lows) < 3 or len(highs) < 2:
            return None
        candidates = []
        for i in range(1, len(lows) - 1):
            anchor = lows[i]
            previous_low = lows[i - 1]
            if float(anchor["price"]) <= float(previous_low["price"]):
                continue
            highs_after = [h for h in highs if h["index"] > anchor["index"]]
            if not highs_after:
                continue
            impulse_high = highs_after[0]
            highs_before = [h for h in highs if h["index"] < impulse_high["index"]]
            if not highs_before or float(impulse_high["price"]) <= float(highs_before[-1]["price"]):
                continue
            endpoints = [l for l in lows if l["index"] > impulse_high["index"] and float(l["price"]) > float(anchor["price"])]
            if not endpoints:
                continue
            endpoint = endpoints[-1]
            move = float(impulse_high["price"]) - float(anchor["price"])
            ref_atr = max((atr_at(anchor["index"]) + atr_at(impulse_high["index"])) / 2.0, 1e-9)
            if move / ref_atr < 1.25:
                continue
            candidates.append((endpoint["index"], impulse_high["index"], anchor, endpoint))
        if candidates:
            _, _, anchor, endpoint = max(candidates, key=lambda x: (x[0], x[1]))
            return anchor, endpoint

    if kind == "resistance":
        if len(highs) < 3 or len(lows) < 2:
            return None
        candidates = []
        for i in range(1, len(highs) - 1):
            anchor = highs[i]
            previous_high = highs[i - 1]
            if float(anchor["price"]) >= float(previous_high["price"]):
                continue
            lows_after = [l for l in lows if l["index"] > anchor["index"]]
            if not lows_after:
                continue
            impulse_low = lows_after[0]
            lows_before = [l for l in lows if l["index"] < impulse_low["index"]]
            if not lows_before or float(impulse_low["price"]) >= float(lows_before[-1]["price"]):
                continue
            endpoints = [h for h in highs if h["index"] > impulse_low["index"] and float(h["price"]) < float(anchor["price"])]
            if not endpoints:
                continue
            endpoint = endpoints[-1]
            move = float(anchor["price"]) - float(impulse_low["price"])
            ref_atr = max((atr_at(anchor["index"]) + atr_at(impulse_low["index"])) / 2.0, 1e-9)
            if move / ref_atr < 1.25:
                continue
            candidates.append((endpoint["index"], impulse_low["index"], anchor, endpoint))
        if candidates:
            _, _, anchor, endpoint = max(candidates, key=lambda x: (x[0], x[1]))
            return anchor, endpoint

    return None


def _patch_trendline_master():
    try:
        import strategies
    except Exception as exc:
        print(f"[trendline_master] strategies not ready: {exc!r}")
        return

    strategies._find_impulse_anchor_pair = _structural_impulse_anchor_pair
    original = getattr(strategies, "build_trendline_family", None)
    if original is None or getattr(original, "_trendline_master_wrapped", False):
        return

    def _master(df, max_lines=4, lookback_bars=60):
        family = original(df, max_lines=max_lines, lookback_bars=lookback_bars)
        if not family or family.get("error"):
            return family

        sma_dir = _sma_dir(df)
        supports = family.get("uptrends") or []
        resistances = family.get("downtrends") or []

        # HARD MAP RULE: the 20 SMA decides which diagonal family is allowed.
        if sma_dir == "RISING":
            candidates = [x for x in supports if float(x.get("slope", 0.0)) > 0]
            master = candidates[0] if candidates else None
            role = "support"
            base_bias = "BUY"
        elif sma_dir == "FALLING":
            candidates = [x for x in resistances if float(x.get("slope", 0.0)) < 0]
            master = candidates[0] if candidates else None
            role = "resistance"
            base_bias = "SELL"
        else:
            master = None
            role = "none"
            base_bias = "NEUTRAL"

        # Never leave the opposite rail visible. The renderer should receive
        # exactly the same one-line map a trader would draw by hand.
        if master is not None:
            if role == "support":
                family["uptrends"] = [master]
                family["downtrends"] = []
            else:
                family["uptrends"] = []
                family["downtrends"] = [master]
            family["family_lines"] = [master]
            family["channel"] = None
            family["mode"] = "lines"
        else:
            family["uptrends"] = []
            family["downtrends"] = []
            family["family_lines"] = []
            family["channel"] = None
            family["mode"] = "sr"

        family["sma_direction"] = sma_dir
        family["sma_applied_price"] = "median"
        family["master_trendline"] = master
        family["master_role"] = role

        if master is None:
            family["direction"] = "NEUTRAL"
            family["family_kind"] = "none"
            family["master_decision"] = "WAIT — 20 SMA FLAT / no trendline; map S/R"
            family["trendline_color_state"] = "NEUTRAL"
            family["prefer_retest_entry"] = False
            family["master_entry_ready"] = False
            family["reasons"] = list(family.get("reasons") or []) + [
                "20 SMA is FLAT — market is not classified as trending; use horizontal support/resistance."
            ]
            return family

        n = len(df)
        close = float(df["Close"].iloc[-1])
        line_now = strategies._line_value(master["x0"], master["y0"], master["x1"], master["y1"], n - 1)

        breakout = None
        retest = {"status": "INTACT", "note": "No confirmed trendline break."}
        break_kind = None
        if role == "support" and close < line_now:
            break_kind = "support_break_down"
        elif role == "resistance" and close > line_now:
            break_kind = "resistance_break_up"

        if break_kind:
            breakout = strategies._grade_breakout(df, master, break_kind, n)
            retest = strategies._trendline_retest_state(df, master, breakout, break_kind)

        direction = base_bias
        strength = max(int(family.get("strength") or 40), 55)
        reasons = list(family.get("reasons") or [])

        if retest.get("status") == "FAKEOUT":
            decision = "FAKEOUT — original SMA trend retained"
        elif breakout and breakout.get("strength") == "confirmed":
            direction = "SELL" if role == "support" else "BUY"
            decision = "BREAK CONFIRMED — BIAS FLIPPED"
            strength = max(strength, 68)
        elif breakout and breakout.get("strength") == "developing":
            decision = "BREAK DEVELOPING — WAIT FOR RETEST"
            strength = max(strength, 52)
        else:
            decision = "INTACT — BULLISH STRUCTURE" if role == "support" else "INTACT — BEARISH STRUCTURE"

        family["direction"] = direction
        family["strength"] = max(0, min(100, int(strength)))
        family["family_kind"] = "ascending" if role == "support" else "descending"
        family["primary_quality"] = master.get("quality")
        family["primary_touches"] = master.get("touches", 0)
        family["bias_touch_points"] = strategies._touch_points(
            df, int(master["x0"]), master["y0"], int(master["x1"]), master["y1"], role
        )
        family["master_line_value"] = float(line_now)
        family["master_decision"] = decision
        family["breakout_grade"] = breakout
        family["trendline_retest"] = retest
        family["trendline_break_kind"] = break_kind
        family["trendline_color_state"] = "BULLISH" if direction == "BUY" else "BEARISH"
        family["reasons"] = reasons + [
            f"20 SMA (median price) is {sma_dir}: only the {'ascending support' if role == 'support' else 'descending resistance'} trendline is allowed."
        ]

        if hasattr(strategies, "_entry_confirmation") and direction in ("BUY", "SELL"):
            family["entry_rules"] = strategies._entry_confirmation(df, direction)

        family["prefer_retest_entry"] = retest.get("status") == "BREAK_CONFIRMED"
        family["master_entry_ready"] = retest.get("status") == "BREAK_RETEST_CONFIRMED"
        return family

    _master._trendline_master_wrapped = True
    _master._original = original
    strategies.build_trendline_family = _master
    print("[trendline_master] 20-SMA-directed master trendline installed")


_patch_trendline_master()

# Preserve the existing pullback-history adapter, but use the single SMA-directed
# master line produced above. A pullback entry still requires the bot's existing
# rejection/confirmation candle logic.
try:
    import usercustomize
    import strategies

    original = getattr(strategies, "build_trendline_family", None)
    if original is not None and not getattr(original, "_historical_pullbacks_wrapped", False):
        def _historical_pullbacks(df, max_lines=4, lookback_bars=60):
            family = original(df, max_lines=max_lines, lookback_bars=lookback_bars)
            if not family or family.get("error"):
                return family
            master = family.get("master_trendline")
            role = str(family.get("master_role") or "none").lower()
            direction = str(family.get("direction") or "NEUTRAL").upper()
            if master is None or role not in ("support", "resistance") or direction not in ("BUY", "SELL"):
                return family

            n = len(df)
            pullbacks = []
            last_signal = -999
            for i in range(max(2, int(master["x0"]) + 1), n):
                try:
                    line = float(strategies._line_value(master["x0"], master["y0"], master["x1"], master["y1"], i))
                    atr = float(df["ATR"].iloc[i]) if "ATR" in df.columns else max(float(df["High"].iloc[i] - df["Low"].iloc[i]), 1e-9)
                    atr = max(atr, 1e-9)
                    if direction == "BUY":
                        touched = float(df["Low"].iloc[i]) <= line + atr * usercustomize.PULLBACK_ZONE_ATR
                        invalid = float(df["Close"].iloc[i]) < line - atr * usercustomize.PULLBACK_INVALIDATION_ATR
                    else:
                        touched = float(df["High"].iloc[i]) >= line - atr * usercustomize.PULLBACK_ZONE_ATR
                        invalid = float(df["Close"].iloc[i]) > line + atr * usercustomize.PULLBACK_INVALIDATION_ATR
                    if not touched or invalid or i - last_signal < 4:
                        continue
                    ok, name = usercustomize._rejection_confirmation(df, i, direction, line, atr)
                    if not ok:
                        continue
                    pullbacks.append({"index": i, "price": float(df["Close"].iloc[i]), "line_price": line,
                                      "direction": direction, "confirmation": name,
                                      "entry_price": float(df["Close"].iloc[i])})
                    last_signal = i
                except Exception:
                    continue

            family["pullback_entries"] = pullbacks
            family["pullback_entry_count"] = len(pullbacks)
            annotations = list(family.get("trendline_annotations") or [])
            for pb in pullbacks:
                annotations.append({"index": pb["index"], "price": pb["entry_price"],
                                    "type": "low" if direction == "BUY" else "high",
                                    "label": "PB", "pullback": True,
                                    "confirmation": pb["confirmation"]})
            family["trendline_annotations"] = annotations
            return family

        _historical_pullbacks._historical_pullbacks_wrapped = True
        _historical_pullbacks._original = original
        strategies.build_trendline_family = _historical_pullbacks
        print("[trendline_pullback_history] SMA-directed pullback markers installed")
except Exception as exc:
    print(f"[trendline_pullback_history] adapter unavailable: {exc!r}")
