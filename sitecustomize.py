"""Activate the selected-timeframe 20-SMA geometry engine."""
from __future__ import annotations


def _install():
    try:
        import market_data
        import strategies
        import single_tf_sma_engine as stf

        def _selected_tf():
            try:
                from execution_engine import state
                return state.get_watch_timeframe() or "30min"
            except Exception:
                return "30min"

        def run_single(symbol, *args, **kwargs):
            tf = kwargs.get("timeframe") or kwargs.get("tf") or _selected_tf()
            df = market_data.fetch_candles(symbol, tf, 300)
            if df is None or df.empty:
                return {"error": f"{symbol} {stf.TF_LABELS.get(tf, tf)} data unavailable", "timeframe": tf}
            result = stf.analyze(df, symbol, tf)
            if result.get("error"):
                return result
            result["strength"] = 95 if result["entry_confirmed"] else 80 if result["strong"] else 60
            result["gating_notes"] = [
                f"Selected timeframe only: {result['timeframe_label']}",
                "20 SMA applied to Median Price",
                "20 SMA establishes direction; trendline is mapped independently",
                "Price touched/rejected trendline or candle confirmation" if result["entry_confirmed"] else "Wait for trendline touch/rejection or candle confirmation",
            ]
            return result

        def format_single(family, symbol, *args, **kwargs):
            return stf.report(family)

        def build_single(family, *args, **kwargs):
            if not family or family.get("error") or not family.get("entry_confirmed"):
                return None
            atr = float(family["atr"])
            entry = float(family["price"])
            direction = family["direction"]
            if direction == "BUY":
                sl = entry - atr
                tp1, tp2, tp3 = entry + atr * 1.5, entry + atr * 2.5, entry + atr * 3.5
            else:
                sl = entry + atr
                tp1, tp2, tp3 = entry - atr * 1.5, entry - atr * 2.5, entry - atr * 3.5
            return {
                "side": "LONG" if direction == "BUY" else "SHORT",
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "rr": 2.5,
                "confirmed": True,
                "order_type": "MARKET",
                "entry_note": "Price touched/rejected the directional trendline or gave candle confirmation.",
            }

        strategies.run_trendline_analysis = run_single
        strategies.format_trendline_report = format_single
        strategies.build_position_container = build_single

        try:
            import visual_pattern_engine
            visual_pattern_engine.render_trendline_map = lambda df, symbol, payload, title_suffix="": stf.render(payload)
        except Exception as exc:
            print(f"[single_tf] visual renderer hook skipped: {exc!r}")

        print("[single_tf] 20-SMA selected-timeframe geometry engine ACTIVE")
    except Exception as exc:
        # Never prevent the bot from starting because the experimental hook fails.
        print(f"[single_tf] activation failed; legacy strategy retained: {exc!r}")


_install()
