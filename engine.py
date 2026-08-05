"""
engine.py
================
Ties patterns.scan_all_patterns -> confirmation_engine -> trade_setup ->
trade_state (mode routing) into one call per (symbol, timeframe). This is
what the EA's poll hits every new bar, and what Telegram's on-demand
analysis buttons call too.
"""

import time
import numpy as np
from patterns import scan_all_patterns
from confirmation_engine import ConfirmationEngine
from trade_setup import build_trade_setup
from volume_profile import compute_volume_profile
import htf_context
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


def poll(symbol, tf, magic_number=None, pushed_df=None):
    """
    Main entry point. Called two ways:
      - By the EA's poll request (pushed_df = the candles it read straight
        from its own MT5 terminal -- free, accurate, avoids a redundant
        fetch). Also marks this symbol's EA heartbeat as alive.
      - By the background watchlist scanner when no EA has been seen
        recently for a symbol (pushed_df = None -> falls back through
        mt5_data.fetch_candles, which itself falls back to Twelve Data
        when MT5 isn't available, e.g. running in Termux while away).

    Regardless of Mode, if no EA has checked in recently for this symbol,
    a confirmed signal is automatically routed as a manual Copy Trade
    ticket instead of trying to command an executor that isn't there.
    """
    mode = ts.state.get_mode()
    response = {"mode": mode, "lot_mode": ts.state.lot_mode, "command": None, "pattern": None}

    if mode == ts.MODE_OFF:
        return response

    if pushed_df is not None:
        df = pushed_df
        ts.state.mark_ea_seen(symbol)
    else:
        df = mt5_data.fetch_candles(symbol, tf, count=250)

    if df is None or df.empty or len(df) < 41:
        response["error"] = "insufficient_data"
        return response

    established_df = df.iloc[:-1]
    volume_profile = compute_volume_profile(established_df)
    best, all_patterns = scan_all_patterns(established_df, volume_profile=volume_profile)

    htf_bias = htf_desc = None
    if best is not None:
        htf_bias, htf_desc = htf_context.get_htf_bias(symbol, tf)
        delta, htf_note = htf_context.htf_alignment_adjustment(best.bias, htf_bias, htf_desc)
        if delta != 0.0:
            best.confidence = float(np.clip(best.confidence + delta, 0.0, 100.0))
            best.note += f" {htf_note}"

    decision = _confirmation_engine.step(symbol, tf, df, best)
    response["pattern"] = best.name if best else None
    response["reason"] = decision["reason"]
    response["trigger_price"] = best.trigger_price if best else None
    response["bias"] = best.bias if best else None
    response["htf_bias"] = htf_bias
    response["htf_context"] = htf_desc
    if volume_profile is not None:
        response["poc_price"] = volume_profile["poc_price"]
        response["value_area_low"] = volume_profile["value_area_low"]
        response["value_area_high"] = volume_profile["value_area_high"]

    if decision["action"] not in ("FIRE_MARKET", "FIRE_LIMIT"):
        response["command"] = ts.state.pop_command(symbol)
        return response

    setup = build_trade_setup(df, decision["pattern"], decision)
    setup["symbol"] = symbol
    setup["timeframe"] = tf

    ea_available = ts.state.is_ea_available(symbol)
    effective_mode = mode if ea_available else ts.MODE_COPY_TRADE

    if effective_mode == ts.MODE_AUTO:
        ts.state.queue_command(symbol, setup)
        if _notify_callbacks["auto_fired"]:
            _notify_callbacks["auto_fired"](setup)

    elif effective_mode == ts.MODE_APPROVAL:
        summary = _format_setup_summary(setup)
        approval_id = ts.state.create_approval_request(symbol, summary, setup)
        if _notify_callbacks["approval_request"]:
            _notify_callbacks["approval_request"](approval_id, setup)

    elif effective_mode == ts.MODE_COPY_TRADE:
        if not ea_available and mode != ts.MODE_COPY_TRADE:
            setup["note"] = (setup.get("note", "") + " [Auto-routed to manual ticket: no EA connection detected.]").strip()
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
