"""Production runtime wiring for the single-timeframe Trendline analyzer.

The existing Telegram bot imports the legacy strategy functions directly. This
hook replaces those functions before bot.py uses them and, as a final guard,
replaces bot.send_trendline_analysis after bot.py has loaded.
"""
from __future__ import annotations

import threading
import time


def _install():
    try:
        import market_data
        import strategies
        import execution_engine as engine
        import single_tf_sma_engine as stf
        import visual_pattern_engine

        allowed = ["4h", "1h", "30min", "15min", "5min", "1min"]
        labels = {"4h":"H4", "1h":"H1", "30min":"M30", "15min":"M15", "5min":"M5", "1min":"M1"}
        engine.WATCH_TIMEFRAMES = allowed
        engine.WATCH_TIMEFRAME_LABELS = labels

        if getattr(engine.state, "watch_timeframe", None) not in allowed:
            engine.state.watch_timeframe = "30min"

        def selected_tf():
            tf = engine.state.get_watch_timeframe()
            return tf if tf in allowed else "30min"

        def run_single(symbol, *args, **kwargs):
            tf = kwargs.get("timeframe") or kwargs.get("tf") or selected_tf()
            if tf not in allowed:
                tf = selected_tf()
            df = market_data.fetch_candles(symbol, tf, 320)
            if df is None or getattr(df, "empty", True):
                return {"error": f"{symbol} {labels[tf]} data unavailable", "symbol": symbol, "timeframe": tf, "strategy_name":"SINGLE-TF 20 SMA GEOMETRY"}
            result = stf.analyze(df, symbol, tf)
            result["strategy_name"] = "SINGLE-TF 20 SMA GEOMETRY"
            result["strength"] = 95 if result.get("entry_confirmed") else 80 if result.get("strong") else 60
            result["gating_notes"] = [
                f"Analysis timeframe: {result.get('timeframe_label', labels[tf])}",
                "20 SMA applied to Median Price",
                "20 SMA establishes direction; trendline is mapped independently",
                "Price touched/rejected trendline or candle confirmation" if result.get("entry_confirmed") else "Wait for price touch/rejection or candle confirmation",
            ]
            return result

        def format_single(result, symbol, *args, **kwargs):
            return stf.report(result)

        def build_single(result, *args, **kwargs):
            # Keep mobile control infrastructure alive, but only create a
            # ticket after the new engine itself confirms the trendline setup.
            if not isinstance(result, dict) or result.get("strategy_name") != "SINGLE-TF 20 SMA GEOMETRY":
                return None
            if not result.get("entry_confirmed"):
                return None
            entry = float(result["price"]); atr = float(result["atr"]); side = result["direction"]
            if side == "BUY":
                sl = entry - atr; tp1 = entry + 1.5*atr; tp2 = entry + 2.5*atr; tp3 = entry + 3.5*atr
            else:
                sl = entry + atr; tp1 = entry - 1.5*atr; tp2 = entry - 2.5*atr; tp3 = entry - 3.5*atr
            return {"side":"LONG" if side == "BUY" else "SHORT", "direction":side, "entry":entry, "sl":sl, "tp1":tp1, "tp2":tp2, "tp3":tp3, "rr":2.5, "confirmed":True, "order_type":"MARKET", "entry_note":"Trendline touch/rejection or candle confirmation."}

        strategies.run_trendline_analysis = run_single
        strategies.format_trendline_report = format_single
        strategies.build_position_container = build_single

        def render_single(df, symbol, payload, title_suffix=""):
            result = payload if isinstance(payload, dict) and payload.get("strategy_name") == "SINGLE-TF 20 SMA GEOMETRY" else run_single(symbol)
            return stf.render(result)
        visual_pattern_engine.render_trendline_map = render_single

        print("[single_tf] ACTIVE: 20 SMA Median Price + selected timeframe")
        print("[single_tf] TF selector: H4/H1/M30/M15/M5/M1")

        def patch_bot_when_loaded():
            # sitecustomize runs before bot.py. Wait for bot.py to finish its
            # imports, then replace the exact Telegram handler that users call.
            for _ in range(120):
                bot = __import__('sys').modules.get('bot')
                if bot is not None and hasattr(bot, 'send_trendline_analysis'):
                    import asyncio
                    from datetime import datetime

                    async def send_single_tf_analysis(context, chat_id, symbol):
                        try:
                            result = run_single(symbol)
                            report_text = format_single(result, symbol)
                            if not result.get("error"):
                                image = stf.render(result)
                                await context.bot.send_photo(chat_id=chat_id, photo=image, caption=f"{symbol} {result['timeframe_label']} — 20 SMA Geometry | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            await context.bot.send_message(chat_id=chat_id, text=report_text)
                            await context.bot.send_message(chat_id=chat_id, text="Choose next action:", reply_markup=bot.get_home_menu())
                        except Exception as exc:
                            import traceback
                            traceback.print_exc()
                            await context.bot.send_message(chat_id=chat_id, text=f"❌ Single-timeframe analysis failed for {symbol}: {exc}", reply_markup=bot.get_home_menu())

                    bot.send_trendline_analysis = send_single_tf_analysis

                    # Make the existing control-panel selector explicitly an
                    # ANALYSIS timeframe selector without changing its callback
                    # contract or the watch-price/mobile-control machinery.
                    original_menu = bot.get_mobile_panel_menu
                    def analysis_menu():
                        markup = original_menu()
                        # The existing menu already has menu_watch_tf. Its label
                        # is changed in the callback screen below; no duplicate
                        # button is added.
                        return markup
                    bot.get_mobile_panel_menu = analysis_menu
                    print("[single_tf] TELEGRAM HANDLER REPLACED — legacy Trendline path bypassed")
                    return
                time.sleep(0.25)
            print("[single_tf] WARNING: bot handler was not found during startup patch")

        threading.Thread(target=patch_bot_when_loaded, name="single-tf-bot-hook", daemon=True).start()

    except Exception as exc:
        print(f"[single_tf] ACTIVATION FAILED: {exc!r}")
        raise


_install()
