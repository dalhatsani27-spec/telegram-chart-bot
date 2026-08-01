"""
engine.py
================
Ties patterns.scan_all_patterns -> confirmation_engine -> trade_setup ->
trade_state (mode routing) into one call per (symbol, timeframe). This is
what the EA's poll hits every new bar, and what Telegram's on-demand
analysis buttons call too.
"""

import time
from patterns import scan_all_patterns
from confirmation_engine import ConfirmationEngine
from trade_setup import build_trade_setup
import trade_state as ts
import mt5_data

_confirmation_engine = ConfirmationEngine()

# Callbacks the Telegram layer registers so this module can push messages
# without importing bot.py directly (avoids circular imports).
_notify_callbacks = {"auto_fired": None, "approval_request": None, "manual_ticket": None}


def register_notifiers(auto_fired=None, approval_request=None, manual_ticket=None):
    if auto_fired: _notify_callbacks["auto_fired"] = auto_fired
    if approval_request: _notify_callbacks["approval_request"] = approval_request
    if manual_ticket: _notify_callbacks["manual_ticket"] = manual_ticket


def poll(symbol, tf, magic_number=None):
    """
    Main entry point called on every EA poll (once per new bar). Returns a
    dict describing what, if anything, the EA should do right now, plus
    dashboard info (mode, lot_mode) so the EA's on-chart panel stays synced.
    """
    mode = ts.state.get_mode()
    response = {"mode": mode, "lot_mode": ts.state.lot_mode, "command": None, "pattern": None}

    if mode == ts.MODE_OFF:
        return response

    df = mt5_data.fetch_candles(symbol, tf, count=250)
    if df is None or df.empty or len(df) < 41:
        response["error"] = "insufficient_data"
        return response

    # Establish/refresh the watched pattern using everything EXCEPT the newest
    # candle -- so the newest candle's own breakout move can't retroactively
    # change what pattern we consider "best" right when we need the watch to
    # stay locked onto the structure that was already forming.
    established_df = df.iloc[:-1]
    best, all_patterns = scan_all_patterns(established_df)
    decision = _confirmation_engine.step(symbol, tf, df, best)
    response["pattern"] = best.name if best else None
    response["reason"] = decision["reason"]
    response["trigger_price"] = best.trigger_price if best else None
    response["bias"] = best.bias if best else None

    if decision["action"] not in ("FIRE_MARKET", "FIRE_LIMIT"):
        # even with nothing to fire, hand back a queued command if one is
        # already waiting (e.g. an approval that was just granted)
        response["command"] = ts.state.pop_command(symbol)
        return response

    setup = build_trade_setup(df, decision["pattern"], decision)
    setup["symbol"] = symbol
    setup["timeframe"] = tf

    if mode == ts.MODE_AUTO:
        ts.state.queue_command(symbol, setup)
        if _notify_callbacks["auto_fired"]:
            _notify_callbacks["auto_fired"](setup)

    elif mode == ts.MODE_APPROVAL:
        summary = _format_setup_summary(setup)
        approval_id = ts.state.create_approval_request(symbol, summary, setup)
        if _notify_callbacks["approval_request"]:
            _notify_callbacks["approval_request"](approval_id, setup)

    elif mode == ts.MODE_COPY_TRADE:
        if _notify_callbacks["manual_ticket"]:
            _notify_callbacks["manual_ticket"](setup)

    response["command"] = ts.state.pop_command(symbol)
    return response


def _format_setup_summary(setup):
    return (f"{setup['symbol']} | {setup['pattern_name']} | {setup['bias']} "
            f"[{setup['order_type']}] entry {setup['entry']:.5f} SL {setup['sl']:.5f} "
            f"TP1 {setup['tp1']:.5f} TP2 {setup['tp2']:.5f}")


def report_event(symbol, event_type, details):
    """EA calls this (via the API) to report fills/TP hits/SL hits so Telegram can relay them."""
    # Left as a hook; bot.py registers a callback for this at startup the same
    # way it does for the fire notifiers, kept separate to avoid a 4th
    # optional callback cluttering register_notifiers().
    if "report_event" in _notify_callbacks and _notify_callbacks["report_event"]:
        _notify_callbacks["report_event"](symbol, event_type, details)


_notify_callbacks["report_event"] = None
