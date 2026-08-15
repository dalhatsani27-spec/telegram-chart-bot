"""Trendline-master adapter for the chart analyzer.

Loaded automatically by Python on Render. It leaves the existing analysis
and chart renderer intact, but makes the 20-SMA / structural-trendline
hierarchy explicit after the legacy family builder has produced its geometry.
"""


def _patch_trendline_master():
    try:
        import strategies
    except Exception as exc:
        print(f"[trendline_master] strategies not ready: {exc!r}")
        return

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

        # SMA is a compass only: it selects which structural rail is the
        # directional/master rail. It is never used as an entry trigger.
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
        line_now = strategies._line_value(
            master["x0"], master["y0"], master["x1"], master["y1"], n - 1
        )

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

        # The trendline is the decision-maker. A confirmed structural break
        # flips the bias; the SMA is not allowed to veto that flip.
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
            reasons.append(
                f"🔄 MASTER TRENDLINE BREAK: {master_role} broken with "
                f"{breakout['consecutive_closes']} close(s), "
                f"{breakout['penetration_atr']} ATR penetration. Bias flipped to {direction}."
            )
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

        # Replace only the decision layer. Keep both rails, patterns, zones,
        # candles, and existing chart geometry untouched.
        family["direction"] = direction
        family["strength"] = max(0, min(100, int(strength)))
        family["family_kind"] = "ascending" if master_role == "support" else "descending"
        family["primary_quality"] = master.get("quality")
        family["primary_touches"] = master.get("touches", 0)
        family["bias_touch_points"] = strategies._touch_points(
            df, int(master["x0"]), master["y0"], int(master["x1"]), master["y1"], master_role
        )
        family["master_trendline"] = master
        family["master_role"] = master_role
        family["master_decision"] = decision
        family["master_line_value"] = float(line_now)
        family["breakout_grade"] = breakout
        family["trendline_retest"] = retest
        family["trendline_break_kind"] = break_kind
        family["trendline_color_state"] = "BULLISH" if direction == "BUY" else "BEARISH" if direction == "SELL" else "NEUTRAL"
        family["reasons"] = reasons

        # Re-evaluate confirmation using the master decision. This remains
        # an entry gate; SMA still has no entry role.
        if direction in ("BUY", "SELL") and hasattr(strategies, "_entry_confirmation"):
            family["entry_rules"] = strategies._entry_confirmation(df, direction)

        if hasattr(strategies, "_measured_move_projections"):
            family["projections"] = strategies._measured_move_projections(df, family.get("pivots") or [], direction)

        # A confirmed break is a transition until the retest confirms it.
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
