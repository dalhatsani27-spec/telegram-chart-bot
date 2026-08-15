"""Trendline-master adapter for the chart analyzer.

Loaded automatically by Python on Render. It leaves the existing analysis
and chart renderer intact, but makes the 20-SMA / structural-trendline
hierarchy explicit after the legacy family builder has produced its geometry.
"""


def _structural_impulse_anchor_pair(pivots, df, kind):
    """Return the hand-drawn trendline anchors from a completed impulse leg.

    The important rule is not "pick the last two lows/highs". The line must
    describe the leg that actually launched the current move:

      UP:   previous Low -> HL(anchor) -> HH(impulse) -> current HL
            trendline = HL(anchor) -> current HL

      DOWN: previous High -> LH(anchor) -> LL(impulse) -> current LH
            trendline = LH(anchor) -> current LH

    Pivots supplied by strategies are already structural/ATR filtered. This
    function adds the structural sequence requirement so a 1-2 candle wiggle
    cannot become a trendline anchor.
    """
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
            if not highs_before_impulse:
                continue
            previous_high = highs_before_impulse[-1]
            if float(impulse_high["price"]) <= float(previous_high["price"]):
                continue
            lows_after_impulse = [l for l in lows if l["index"] > impulse_high["index"]]
            if not lows_after_impulse:
                continue
            endpoint = lows_after_impulse[0]
            if endpoint["index"] != current["index"]:
                continue
            if float(endpoint["price"]) <= float(anchor["price"]):
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
            if not lows_before_impulse:
                continue
            previous_low = lows_before_impulse[-1]
            if float(impulse_low["price"]) >= float(previous_low["price"]):
                continue
            highs_after_impulse = [h for h in highs if h["index"] > impulse_low["index"]]
            if not highs_after_impulse:
                continue
            endpoint = highs_after_impulse[0]
            if endpoint["index"] != current["index"]:
                continue
            if float(endpoint["price"]) >= float(anchor["price"]):
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


def _patch_entry_risk_layer():
    """Attach the final entry/risk gate without modifying chart geometry.

    The existing strategy/chart code remains the source of truth for
    trendlines, SMA direction, breakouts, retests, patterns and drawings.
    This layer only consumes those results.
    """
    try:
        import strategies
        from entry_risk_engine import evaluate_trendline_entry
    except Exception as exc:
        print(f"[entry_risk] not installed: {exc!r}")
        return

    original_report = getattr(strategies, "format_trendline_report", None)
    original_position = getattr(strategies, "build_position_container", None)
    if original_report is None or original_position is None:
        print("[entry_risk] required strategy functions not found")
        return

    def _evaluate_and_attach(family):
        decision = evaluate_trendline_entry(family)
        family["entry_decision"] = decision
        family["entry_rules_final"] = decision["checks"]
        family["entry_status_final"] = decision["status"]
        return decision

    def _position(family, *args, **kwargs):
        # If the final layer has already evaluated this family, its result is
        # authoritative. If not, evaluate it now so EA/background callers
        # cannot accidentally receive a pre-final-gate ticket.
        decision = family.get("entry_decision")
        if decision is None and family.get("df") is not None and family.get("trendline_retest") is not None:
            decision = _evaluate_and_attach(family)

        if decision is not None:
            if decision.get("entry_ready") and decision.get("ticket"):
                return decision["ticket"]
            # A Trendline analysis that has not passed all three checks must
            # not create a trade ticket. Return None only for this isolated
            # Trendline entry layer; chart geometry itself is untouched.
            if family.get("trendline_retest") is not None:
                return None

        return original_position(family, *args, **kwargs)

    def _report(family, symbol):
        decision = _evaluate_and_attach(family)

        # The legacy report's old 4-check gate is not the final authority.
        # Feed it an equivalent confirmed 3/3 result only when our exact
        # break + retest + candle gate is actually satisfied. This changes
        # the decision text, not the chart/trendline engine.
        if decision.get("entry_ready"):
            checks = decision.get("checks") or {}
            family["entry_rules"] = {
                "checks": {
                    "break": (True, checks["break"]["detail"]),
                    "retest": (True, checks["retest"]["detail"]),
                    "candle": (True, checks["candle"]["detail"]),
                },
                "passed": 3,
                "required": 3,
                "confirmed": True,
            }
        else:
            # Keep the legacy details available for diagnostics, but make
            # the report's final decision explicitly reflect our 3-check
            # gate and prevent a false confirmed state.
            family["entry_rules"] = {
                "checks": {
                    name: (item["passed"], item["detail"])
                    for name, item in (decision.get("checks") or {}).items()
                },
                "passed": decision.get("passed", 0),
                "required": 3,
                "confirmed": False,
            }

        text = original_report(family, symbol)

        # Add the exact final checklist after the legacy report so the user
        # sees precisely what is preventing ENTER NOW, without changing any
        # chart or strategy geometry.
        lines = [
            "",
            "━━━━━━━━━━━━━━━━",
            "🔐 FINAL ENTRY GATE",
            f"1. Break confirmed: {'✅' if decision['checks']['break']['passed'] else '❌'}",
            f"2. Retest confirmed: {'✅' if decision['checks']['retest']['passed'] else '❌'}",
            f"3. Entry candle confirmed: {'✅' if decision['checks']['candle']['passed'] else '❌'}",
        ]
        if decision.get("entry_ready") and decision.get("ticket"):
            t = decision["ticket"]
            lines += [
                "",
                "🟢 ENTER NOW",
                f"ENTRY: {t['entry']:.5f}",
                f"SL: {t['sl']:.5f}",
                f"TP: {t['tp1']:.5f}",
                f"R:R: 1:{t['rr']:.1f}",
                f"RISK: {t['risk_percent']:.2f}%",
            ]
        else:
            missing = decision.get("missing") or []
            labels = {"break": "confirmed trendline break", "retest": "confirmed retest/hold", "candle": "directional entry candle"}
            lines += ["", "🔴 WAIT", "WAIT FOR:"]
            for name in missing:
                lines.append(f"• {labels.get(name, name)}")
            lines.append(f"ENTRY CONFIRMATION: {decision.get('passed', 0)}/3")

        return text + "\n" + "\n".join(lines)

    strategies.build_position_container = _position
    strategies.format_trendline_report = _report
    print("[entry_risk] final 3-check Trendline entry gate installed")


_patch_trendline_master()
_patch_entry_risk_layer()
