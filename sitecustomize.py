"""Activate the isolated single-timeframe 30-SMA geometry engine."""
from __future__ import annotations


def _install():
    try:
        import strategies
        import single_tf_sma_engine as stf

        def _selected_tf():
            try:
                from execution_engine import state
                return state.get_watch_timeframe() or "30min"
            except Exception:
                return "30min"

        def run_single(symbol, *args, **kwargs):
            tf = _selected_tf()
            df = strategies.market_data.fetch_candles(symbol, tf, 300)
            if df is None or df.empty:
                return {"error": f"{symbol} {stf.TF_LABELS.get(tf, tf)} data unavailable", "timeframe": tf}
            r = stf.analyze(df, symbol, tf)
            if r.get("error"):
                return r
            r["strength"] = 90 if r["strong"] else 65 if r["trendline"] else 50
            r["gating_notes"] = [
                f"Single timeframe: {r['timeframe_label']}",
                "30 SMA applied to Median Price",
                "SMA + trendline near/touching" if r["near"] else "SMA + trendline separated",
            ]
            r["entry_rules"] = {"confirmed": bool(r["strong"]), "confirmation_count": 1 if r["strong"] else 0}
            return r

        def format_single(family, symbol, *args, **kwargs):
            return stf.report(family)

        def build_single(family, *args, **kwargs):
            if not family or family.get("error") or not family.get("strong"):
                return None
            atr=float(family["atr"]); entry=float(family["price"]); direction=family["direction"]
            if direction == "BUY":
                sl=entry-atr; tp1=entry+atr*1.5; tp2=entry+atr*2.5
            else:
                sl=entry+atr; tp1=entry-atr*1.5; tp2=entry-atr*2.5
            return {"side":"LONG" if direction=="BUY" else "SHORT","direction":direction,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp2,"rr":2.5,"confirmed":True,"order_type":"MARKET","entry_note":"30 SMA and directional trendline are near/touching."}

        strategies.run_trendline_analysis = run_single
        strategies.format_trendline_report = format_single
        strategies.build_position_container = build_single

        import visual_pattern_engine
        visual_pattern_engine.render_trendline_map = lambda df, symbol, payload, title_suffix="": stf.render(payload)
        print("[single_tf] 30-SMA geometry engine ACTIVE")
    except Exception as exc:
        print(f"[single_tf] activation failed: {exc!r}")


_install()
