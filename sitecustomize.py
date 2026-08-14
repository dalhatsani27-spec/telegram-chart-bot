"""Activate the isolated single-timeframe 20-SMA geometry engine."""
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
            tf = kwargs.get("timeframe") or kwargs.get("tf") or _selected_tf()
            df = strategies.market_data.fetch_candles(symbol, tf, 300)
            if df is None or df.empty:
                return {"error": f"{symbol} {stf.TF_LABELS.get(tf, tf)} data unavailable", "timeframe": tf}
            r = stf.analyze(df, symbol, tf)
            if r.get("error"):
                return r
            r["strategy_name"] = "SINGLE-TF 20 SMA GEOMETRY"
            r["strength"] = 90 if r["strong"] else 65 if r["trendline"] else 50
            r["gating_notes"] = [
                f"Single timeframe: {r['timeframe_label']}",
                "20 SMA applied to Median Price",
                "SMA + trendline near/touching" if r["near"] else "SMA + trendline separated",
            ]
            r["entry_rules"] = {"confirmed": bool(r["entry_confirmed"]), "confirmation_count": 1 if r["entry_confirmed"] else 0}
            return r

        def format_single(family, symbol, *args, **kwargs):
            return stf.report(family)

        def build_single(family, *args, **kwargs):
            if not isinstance(family, dict) or family.get("error") or not family.get("entry_confirmed"):
                return None
            atr=float(family["atr"]); entry=float(family["price"]); direction=family["direction"]
            if direction == "BUY":
                sl=entry-atr; tp1=entry+atr*1.5; tp2=entry+atr*2.5; tp3=entry+atr*3.5
            elif direction == "SELL":
                sl=entry+atr; tp1=entry-atr*1.5; tp2=entry-atr*2.5; tp3=entry-atr*3.5
            else:
                return None
            return {"side":"LONG" if direction=="BUY" else "SHORT","direction":direction,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3,"rr":2.5,"confirmed":True,"order_type":"MARKET","entry_note":"20 SMA directional rail is near/touching; engine gate confirmed."}

        strategies.run_trendline_analysis = run_single
        strategies.format_trendline_report = format_single
        strategies.build_position_container = build_single

        import visual_pattern_engine
        visual_pattern_engine.render_trendline_map = lambda df, symbol, payload, title_suffix="": stf.render(payload)
        print("[single_tf] 20-SMA geometry engine ACTIVE")
    except Exception as exc:
        print(f"[single_tf] activation failed: {exc!r}")


_install()
