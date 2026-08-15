"""Trendline-master adapter for the chart analyzer.

Loaded automatically by Python on Render. It leaves the existing analysis
and chart renderer intact, but makes the 20-SMA / structural-trendline
hierarchy explicit after the legacy family builder has produced its geometry.
"""


def _structural_impulse_anchor_pair(pivots, df, kind):
    """Return the hand-drawn trendline anchors from a completed impulse leg."""
    if df is None or len(pivots or []) < 4:
        return None

    ordered = sorted(pivots, key=lambda p: int(p.get("index", 0)))
    lows = [p for p in ordered if p.get("type") == "low"]
    highs = [p for p in ordered if p.get("type") == "high"]

    if len(lows) < 3 or len(highs) < 2:
        return None

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
        current = lows[-1]
        candidates = []
        for i in range(1, len(lows) - 1):
            anchor = lows[i]
            previous_low = lows[i - 1]
            if float(anchor["price"]) <= float(previous_low["price"]):
                continue
            highs_after_anchor = [h for h in highs if h["index"] > anchor["index"]]
            if not highs_after_anchor:
                continue
            impulse_high = highs_after_anchor[0]
            highs_before_impulse = [h for h in highs if h["index"] < impulse_high["index"]]
            if not highs_before_impulse or float(impulse_high["price"]) <= float(highs_before_impulse[-1]["price"]):
                continue
            lows_after_impulse = [l for l in lows if l["index"] > impulse_high["index"]]
            if not lows_after_impulse:
                continue
            endpoint = lows_after_impulse[0]
            if endpoint["index"] != current["index"] or float(endpoint["price"]) <= float(anchor["price"]):
                continue
            move = float(impulse_high["price"]) - float(anchor["price"])
            reference_atr = max((atr_at(anchor["index"]) + atr_at(impulse_high["index"])) / 2.0, 1e-9)
            impulse_atr = move / reference_atr
            if impulse_atr < 1.25:
                continue
            candidates.append((endpoint["index"], impulse_high["index"], impulse_atr, anchor, endpoint))
        if not candidates:
            return None
        _, _, _, anchor, endpoint = max(candidates, key=lambda x: (x[0], x[1], x[2]))
        return anchor, endpoint

    if kind == "resistance":
        current = highs[-1]
        candidates = []
        for i in range(1, len(highs) - 1):
            anchor = highs[i]
            previous_high = highs[i - 1]
            if float(anchor["price"]) >= float(previous_high["price"]):
                continue
            lows_after_anchor = [l for l in lows if l["index"] > anchor["index"]]
            if not lows_after_anchor:
                continue
            impulse_low = lows_after_anchor[0]
            lows_before_impulse = [l for l in lows if l["index"] < impulse_low["index"]]
            if not lows_before_impulse or float(impulse_low["price"]) >= float(lows_before_impulse[-1]["price"]):
                continue
            highs_after_impulse = [h for h in highs if h["index"] > impulse_low["index"]]
            if not highs_after_impulse:
                continue
            endpoint = highs_after_impulse[0]
            if endpoint["index"] != current["index"] or float(endpoint["price"]) >= float(anchor["price"]):
                continue
            move = float(anchor["price"]) - float(impulse_low["price"])
            reference_atr = max((atr_at(anchor["index"]) + atr_at(impulse_low["index"])) / 2.0, 1e-9)
            impulse_atr = move / reference_atr
            if impulse_atr < 1.25:
                continue
            candidates.append((endpoint["index"], impulse_low["index"], impulse_atr, anchor, endpoint))
        if not candidates:
            return None
        _, _, _, anchor, endpoint = max(candidates, key=lambda x: (x[0], x[1], x[2]))
        return anchor, endpoint

    return None


def _patch_trendline_master():
    try:
        import strategies
    except Exception as exc:
        print(f"[trendline_master] strategies not ready: {exc!r}")
        return

    strategies._find_impulse_anchor_pair = _structural_impulse_anchor_pair
    print("[trendline_master] impulse-leg structural anchor selector installed")

    original = getattr(strategies, "build_trendline_family", None)
    if original is None or getattr(original, "_trendline_master_wrapped", False):
        return

    def _master(df, max_lines=4, lookback_bars=60):
        family = original(df, max_lines=max_lines, lookback_bars=lookback_bars)
        if not family or family.get("error"):
            return family

        sma_dir = str(family.get("sma_direction") or "FLAT").upper()
        supports = family.get("uptrends") or []
        resistances = family.get("downtrends") or []

        if sma_dir == "RISING":
            master = supports[0] if supports else None
            master_role = "support"
            base_bias = "BUY"
        elif sma_dir == "FALLING":
            master = resistances[0] if resistances else None
            master_role = "resistance"
            base_bias = "SELL"
        else:
            master = None
            master_role = "none"
            base_bias = "NEUTRAL"

        if master is None:
            family["master_trendline"] = None
            family["master_role"] = "none"
            family["master_decision"] = "WAIT — no structural trendline in SMA direction"
            return family

        n = len(df)
        close = float(df["Close"].iloc[-1])
        line_now = strategies._line_value(master["x0"], master["y0"], master["x1"], master["y1"], n - 1)

        breakout = None
        retest = {"status": "INTACT", "note": "No confirmed trendline break."}
        break_kind = None
        if master_role == "support" and close < line_now:
            break_kind = "support_break_down"
        elif master_role == "resistance" and close > line_now:
            break_kind = "resistance_break_up"

        if break_kind:
            breakout = strategies._grade_breakout(df, master, break_kind, n)
            retest = strategies._trendline_retest_state(df, master, breakout, break_kind)

        direction = base_bias
        decision = "INTACT"
        strength = int(family.get("strength") or 40)
        reasons = list(family.get("reasons") or [])

        if retest.get("status") == "FAKEOUT":
            direction = base_bias
            decision = "FAKEOUT — original bias retained"
            reasons.append("🚫 Trendline excursion reclaimed — bias retained; no structural flip.")
        elif breakout and breakout.get("strength") == "confirmed":
            direction = "SELL" if master_role == "support" else "BUY"
            decision = "BREAK CONFIRMED — BIAS FLIPPED"
            strength = max(strength, 68)
            reasons.append(f"🔄 MASTER TRENDLINE BREAK: {master_role} broken with {breakout['consecutive_closes']} close(s), {breakout['penetration_atr']} ATR penetration. Bias flipped to {direction}.")
        elif breakout and breakout.get("strength") == "developing":
            direction = base_bias
            decision = "BREAK DEVELOPING — WAIT FOR CONFIRMATION"
            strength = max(strength, 52)
            reasons.append("⏳ Structural break developing — SMA bias retained until the trendline break confirms.")
        elif master_role == "support":
            direction = "BUY"
            decision = "INTACT — BULLISH STRUCTURE"
            reasons.append("🟢 SMA direction: RISING → green rising-support is the master trendline.")
        else:
            direction = "SELL"
            decision = "INTACT — BEARISH STRUCTURE"
            reasons.append("🔴 SMA direction: FALLING → red falling-resistance is the master trendline.")

        family["direction"] = direction
        family["strength"] = max(0, min(100, int(strength)))
        family["family_kind"] = "ascending" if master_role == "support" else "descending"
        family["primary_quality"] = master.get("quality")
        family["primary_touches"] = master.get("touches", 0)
        family["bias_touch_points"] = strategies._touch_points(df, int(master["x0"]), master["y0"], int(master["x1"]), master["y1"], master_role)
        family["master_trendline"] = master
        family["master_role"] = master_role
        family["master_decision"] = decision
        family["master_line_value"] = float(line_now)
        family["breakout_grade"] = breakout
        family["trendline_retest"] = retest
        family["trendline_break_kind"] = break_kind
        family["trendline_color_state"] = "BULLISH" if direction == "BUY" else "BEARISH" if direction == "SELL" else "NEUTRAL"
        family["reasons"] = reasons

        if direction in ("BUY", "SELL") and hasattr(strategies, "_entry_confirmation"):
            family["entry_rules"] = strategies._entry_confirmation(df, direction)

        if hasattr(strategies, "_measured_move_projections"):
            family["projections"] = strategies._measured_move_projections(df, family.get("pivots") or [], direction)

        family["prefer_retest_entry"] = retest.get("status") == "BREAK_CONFIRMED"
        family["master_entry_ready"] = retest.get("status") == "BREAK_RETEST_CONFIRMED"
        if family["master_entry_ready"]:
            reasons.append("✅ MASTER TRENDLINE BREAK + RETEST CONFIRMED — entry can now be evaluated.")
        elif break_kind and retest.get("status") in ("BREAK_CONFIRMED", "BREAK_DEVELOPING"):
            reasons.append("⏳ Wait for the master trendline retest before entry.")

        return family

    _master._trendline_master_wrapped = True
    _master._original = original
    strategies.build_trendline_family = _master
    print("[trendline_master] master decision adapter installed")


_patch_trendline_master()

# Load the finalized entry adapter after the protected master-trendline adapter.
import usercustomize


def _add_historical_pullbacks():
    """Keep the pullbacks that happened along the trendline, not only the last candle.

    These events are injected into trendline_annotations, which the existing
    educational chart renderer already draws. This is important for the
    user's chart: every meaningful touch/rejection on the rising rail should
    remain visible, not disappear when a later candle becomes the latest one.
    """
    try:
        import strategies
        import usercustomize
    except Exception:
        return

    original = getattr(strategies, "build_trendline_family", None)
    if original is None or getattr(original, "_historical_pullbacks_wrapped", False):
        return

    def _wrapped(df, max_lines=4, lookback_bars=60):
        family = original(df, max_lines=max_lines, lookback_bars=lookback_bars)
        if not family or family.get("error"):
            return family

        master = family.get("master_trendline")
        role = str(family.get("master_role") or "none").lower()
        direction = str(family.get("direction") or "NEUTRAL").upper()
        if master is None or role not in ("support", "resistance") or direction not in ("BUY", "SELL"):
            return family

        n = len(df)
        x0 = int(master.get("x0", 0))
        x1 = int(master.get("x1", x0))
        start = max(2, x0 + 1)
        if start >= n:
            return family

        pullbacks = []
        last_signal = -999
        for i in range(start, n):
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

                pullbacks.append({
                    "index": i,
                    "price": float(df["Close"].iloc[i]),
                    "line_price": line,
                    "direction": direction,
                    "confirmation": name,
                    "entry_price": float(df["Close"].iloc[i]),
                })
                last_signal = i
            except Exception:
                continue

        family["pullback_entries"] = pullbacks
        family["pullback_entry_count"] = len(pullbacks)

        annotations = list(family.get("trendline_annotations") or [])
        existing = {(int(a.get("index", -1)), str(a.get("label", ""))) for a in annotations if isinstance(a, dict)}
        for pb in pullbacks:
            key = (pb["index"], "PB")
            if key in existing:
                continue
            annotations.append({
                "index": pb["index"],
                "price": pb["entry_price"],
                "type": "low" if direction == "BUY" else "high",
                "label": "PB",
                "pullback": True,
                "confirmation": pb["confirmation"],
            })
        family["trendline_annotations"] = annotations
        return family

    _wrapped._historical_pullbacks_wrapped = True
    _wrapped._original = original
    strategies.build_trendline_family = _wrapped
    print("[trendline_pullback_history] historical pullback markers installed")


_add_historical_pullbacks()


# TRENDLINE_V2_FINAL_HOOK
try:
    import trendline_mapper_v2
    trendline_mapper_v2.install()
except Exception as _trendline_v2_exc:
    print(f"[trendline_v2] final hook failed: {_trendline_v2_exc!r}")
