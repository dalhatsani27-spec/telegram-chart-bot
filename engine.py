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
from patterns import scan_all_patterns, Pattern
from confirmation_engine import ConfirmationEngine
from trade_setup import build_trade_setup
from volume_profile import compute_volume_profile
from trendline_family import build_trendline_family, build_position_container
import htf_context
import trade_state as ts
import mt5_data

_confirmation_engine = ConfirmationEngine()


def _trendline_fallback(established_df):
    """
    Runs only when scan_all_patterns() finds NO classic chart pattern (no
    flag/pennant, double/triple top/bottom, H&S, triangle, wedge,
    rectangle). Falls back to the trendline-family read -- one clean
    parallel channel OR a converging wedge/triangle -- so a chart that
    doesn't fit a textbook pattern still gets a structured trade idea
    instead of going silent for the bar.

    Returns (Pattern, fire_decision, setup) or (None, None, None).

    This does NOT go through ConfirmationEngine's marubozu-wait logic --
    the trendline family already has its own confirmation layer
    (_grade_breakout's confirmed/developing/weak, with automatic
    retest-routing for anything not yet confirmed), so gating happens
    right here instead.
    """
    family = build_trendline_family(established_df, max_lines=3)
    if family.get("error"):
        return None, None, None
    direction = family.get("direction")
    if direction not in ("BUY", "SELL"):
        return None, None, None

    quality = family.get("primary_quality")
    strength = family.get("strength", 0)
    brk = family.get("breakout_grade")
    wedge = family.get("wedge")

    # Don't fire off a thin/unconfirmed read. Either the break itself is
    # graded confirmed/developing (weak breaks are dropped), or -- if
    # there's no break in play -- the underlying channel/wedge is at
    # least a validated 3+ touch structure with real strength.
    if brk:
        if brk["strength"] == "weak":
            return None, None, None
    else:
        if quality == "unconfirmed" or strength < 60:
            return None, None, None

    if wedge:
        name = wedge["pattern"]
        category = "reversal" if wedge["pattern"] != "Converging Channel" else "continuation"
        trigger_line = [(wedge["lower"]["x0"], wedge["lower"]["y0"]),
                         (wedge["lower"]["x1"], wedge["lower"]["y1"])]
    else:
        kind = str(family.get("family_kind") or "channel").title()
        name = f"{kind} Trendline Channel"
        category = "continuation"
        fl = (family.get("family_lines") or [None])[0]
        trigger_line = [(fl["x0"], fl["y0"]), (fl["x1"], fl["y1"])] if fl else []

    pos = build_position_container(family)
    if not pos:
        return None, None, None

    confidence = float(min(95, max(30, strength)))
    note = "; ".join(family.get("reasons") or [])[:280] or "Trendline-family structural read"

    pattern_obj = Pattern(
        name=name, category=category, bias=direction,
        trigger_price=pos["entry"], trigger_line=trigger_line,
        key_points=[], confidence=confidence, note=note,
    )
    order_type = pos.get("order_type", "MARKET")
    fire_decision = {
        "action": "FIRE_LIMIT" if order_type == "LIMIT" else "FIRE_MARKET",
        "pattern": pattern_obj, "fire_price": pos["entry"],
        "order_type": order_type,
        "expiry_bars": 15 if order_type == "LIMIT" else None,
        "reason": "trendline_fallback_" + ("retest" if pos.get("entry_note") else "structural"),
    }
    setup = {
        "entry": pos["entry"], "order_type": order_type, "bias": direction,
        "sl": pos["sl"], "tp1": pos["tp1"], "tp2": pos["tp2"],
        "trigger_price": pos["entry"], "pattern_name": name,
        "category": category, "confidence": confidence, "note": note,
        "expiry_bars": fire_decision["expiry_bars"],
    }
    return pattern_obj, fire_decision, setup

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

    # No classic chart pattern this bar -- fall back to the trendline/
    # channel/wedge read ("my logic") instead of going quiet. This bypasses
    # ConfirmationEngine (which is pattern-shaped: marubozu-through-trigger)
    # because the trendline fallback already carries its own confirmed/
    # developing/weak breakout grade and retest routing.
    trendline_setup = None
    trendline_fire_decision = None
    if best is None:
        best, trendline_fire_decision, trendline_setup = _trendline_fallback(established_df)

    htf_bias = htf_desc = None
    if best is not None:
        htf_bias, htf_desc = htf_context.get_htf_bias(symbol, tf)
        delta, htf_note = htf_context.htf_alignment_adjustment(best.bias, htf_bias, htf_desc)
        if delta != 0.0:
            best.confidence = float(np.clip(best.confidence + delta, 0.0, 100.0))
            best.note += f" {htf_note}"
            if trendline_setup is not None:
                trendline_setup["confidence"] = best.confidence
                trendline_setup["note"] = best.note

    if trendline_fire_decision is not None:
        decision = trendline_fire_decision
    else:
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

    setup = trendline_setup if trendline_setup is not None else build_trade_setup(df, decision["pattern"], decision)
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
