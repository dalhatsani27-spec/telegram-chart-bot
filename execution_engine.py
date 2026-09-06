"""
execution_engine.py
====================
The live-trading Control Panel backend: mode/state management, the EA
command queue, HTF-context checking, candle confirmation, trade-setup math,
and the Flask routes the EA itself polls. This is everything behind the
Telegram Control Panel (Master switch, Auto/Approval/Mobile-Manual modes,
lot sizing, EA heartbeat, Account & PnL, Open Positions).

Sections in this file (in call order, top to bottom):
  1. trade_state    -- Mode, strategy selection, EA command queue, approvals
  2. confirmation    -- marubozu / candlestick confirmation -> fire decision
  3. trade_setup     -- fire decision + pattern -> entry/SL/TP1/TP2 numbers
  4. htf_context     -- higher-timeframe bias check for a lower-TF signal
  5. engine.poll()   -- ties 1-4 together; called by the EA and by the
                        background watchlist scanner
  6. Flask routes    -- register_routes(app), called once from bot.py
"""

import json
import os
import time
import uuid
from types import SimpleNamespace

import numpy as np
from flask import request, jsonify

import market_data
from market_analysis import scan_all_patterns, Pattern, compute_volume_profile, detect_confirmation_candle, is_price_ranging_vs_sma
from strategies import build_trendline_family, build_position_container
import unified_strategy


# ============================================================
# 1. TRADE STATE -- Mode (set from Telegram), strategy selection, the
#    small command queue the EA polls, and the Telegram approval flow.
#
#    Modes:
#      OFF          - nothing happens
#      AUTO         - EA fires immediately on confirmed signal
#      APPROVAL     - Telegram Approve/Reject first
#      COPY_TRADE   - Mobile Manual: only send trade tickets (no EA execution)
# ============================================================

MODE_OFF = "OFF"
MODE_AUTO = "AUTO"
MODE_APPROVAL = "APPROVAL"
MODE_COPY_TRADE = "COPY_TRADE"
VALID_MODES = {MODE_OFF, MODE_AUTO, MODE_APPROVAL, MODE_COPY_TRADE}

STRATEGY_UNIFIED = "UNIFIED"
# Legacy constants retained only so stale callbacks do not crash; they are not selectable.
STRATEGY_TRENDLINE = "TRENDLINE"
STRATEGY_OTE = "OTE"
STRATEGY_SMC = "SMC"
VALID_STRATEGIES = {STRATEGY_UNIFIED}

APPROVAL_EXPIRY_SECONDS = 180

# Timeframes selectable from the Mobile Control Panel for the watched-symbol
# background scanner (Copy Trade / Mobile Manual tickets). Internal codes
# match market_data._TF_MAP.
WATCH_TIMEFRAMES = ["5min", "15min", "30min", "1h", "4h"]
WATCH_TIMEFRAME_LABELS = {
    "5min": "M5", "15min": "M15", "30min": "M30", "1h": "H1", "4h": "H4",
}
DEFAULT_WATCH_TIMEFRAME = "15min"


class TradeStateManager:
    def __init__(self):
        self.mode = MODE_OFF
        self.lot_mode = "MIN"
        self.watched_symbol = None
        self.watch_timeframe = DEFAULT_WATCH_TIMEFRAME
        # Personal price-watch levels are analysis alerts, not trade orders.
        self.watch_levels = {}

        # Strategy selection -- only Trendline and OTE remain, so this is
        # a straight either/or choice (no Hybrid/confluence mode).
        self.selected_strategy = STRATEGY_UNIFIED

        self._commands = {}
        self._pending_approvals = {}
        self._ea_last_seen = {}

        # Optional HTF context cache -- populated only when the user
        # explicitly requests it via the "HTF Context" button, keyed by
        # symbol. Trendline/SMC analysis reads this in (if present) rather
        # than fetching a 4H/1H cascade automatically on every run.
        self._htf_context = {}

    def set_last_htf_context(self, symbol, data):
        self._htf_context[symbol.upper()] = data

    def get_last_htf_context(self, symbol):
        return self._htf_context.get(symbol.upper())

    # ---------------- watched symbol ----------------
    def set_watched_symbol(self, symbol):
        self.watched_symbol = symbol.strip().upper() if symbol else None

    def get_watched_symbol(self):
        return self.watched_symbol

    def clear_watched_symbol(self):
        self.watched_symbol = None

    # ---------------- personal watch levels ----------------
    def add_watch_level(self, symbol, level):
        symbol = str(symbol).strip().upper()
        level = float(level)
        key = f"{symbol}|{level:.10f}"
        self.watch_levels[key] = {
            "symbol": symbol, "level": level, "last_price": None,
            "state": "WAITING", "triggered": False, "created_at": time.time(),
        }
        return self.watch_levels[key]

    def get_watch_levels(self):
        return list(self.watch_levels.values())

    def remove_watch_level(self, symbol, level):
        key = f"{str(symbol).strip().upper()}|{float(level):.10f}"
        return self.watch_levels.pop(key, None) is not None

    def clear_watch_levels(self):
        self.watch_levels.clear()

    def update_watch_level(self, item, price):
        """Return a one-time event when price crosses/touches a watched level."""
        level = float(item["level"]); prev = item.get("last_price")
        item["last_price"] = float(price)
        if prev is None:
            return None
        tol = max(abs(level) * 0.00005, 1e-8)
        crossed_up = prev < level and price >= level
        crossed_down = prev > level and price <= level
        touched = abs(price - level) <= tol
        if crossed_up:
            item["state"] = "CROSSED_UP"; item["triggered"] = True
            return "CROSSED_UP"
        if crossed_down:
            item["state"] = "CROSSED_DOWN"; item["triggered"] = True
            return "CROSSED_DOWN"
        if touched and item.get("state") == "WAITING":
            item["state"] = "TOUCHED"
            return "TOUCHED"
        return None

    # ---------------- watch timeframe ----------------
    def set_watch_timeframe(self, tf_code):
        if tf_code not in WATCH_TIMEFRAMES:
            raise ValueError(f"invalid watch timeframe: {tf_code}")
        self.watch_timeframe = tf_code

    def get_watch_timeframe(self):
        return self.watch_timeframe

    def watch_timeframe_label(self):
        return WATCH_TIMEFRAME_LABELS.get(self.watch_timeframe, self.watch_timeframe)

    # ---------------- EA heartbeat ----------------
    def mark_ea_seen(self, symbol):
        self._ea_last_seen[symbol] = time.time()

    def is_ea_available(self, symbol, timeout_seconds=180):
        last = self._ea_last_seen.get(symbol)
        if last is None:
            return False
        return (time.time() - last) <= timeout_seconds

    # ---------------- trading mode ----------------
    def set_mode(self, mode):
        if mode not in VALID_MODES:
            raise ValueError(f"invalid mode: {mode}")
        self.mode = mode

    def get_mode(self):
        return self.mode

    def set_lot_mode(self, lot_mode):
        if lot_mode not in ("MIN", "RISK"):
            raise ValueError("lot_mode must be MIN or RISK")
        self.lot_mode = lot_mode

    # ---------------- strategy selection ----------------
    def set_selected_strategy(self, strategy):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"invalid strategy: {strategy}")
        self.selected_strategy = strategy

    def get_selected_strategy(self):
        return self.selected_strategy

    def strategy_label(self):
        return "Unified Market Intelligence" if self.selected_strategy == STRATEGY_UNIFIED else self.selected_strategy.title().replace("_", " ")

    # ---------------- EA command queue ----------------
    def queue_command(self, symbol, command):
        command = dict(command)
        command["queued_at"] = time.time()
        self._commands[symbol] = command

    def pop_command(self, symbol):
        return self._commands.pop(symbol, None)

    def peek_command(self, symbol):
        return self._commands.get(symbol)

    # ---------------- Telegram approval flow ----------------
    def create_approval_request(self, symbol, setup_summary, on_approve_command):
        approval_id = uuid.uuid4().hex[:10]
        self._pending_approvals[approval_id] = {
            "symbol": symbol,
            "summary": setup_summary,
            "command": on_approve_command,
            "created_at": time.time(),
            "status": "PENDING",
        }
        return approval_id

    def resolve_approval(self, approval_id, approved):
        req = self._pending_approvals.get(approval_id)
        if req is None:
            return None, "not_found"
        age = time.time() - req["created_at"]
        if age > APPROVAL_EXPIRY_SECONDS:
            req["status"] = "EXPIRED"
            return req, "expired"
        req["status"] = "APPROVED" if approved else "REJECTED"
        if approved:
            self.queue_command(req["symbol"], req["command"])
        return req, req["status"].lower()

    def expire_stale_approvals(self):
        now = time.time()
        expired = []
        for aid, req in list(self._pending_approvals.items()):
            if req["status"] == "PENDING" and (now - req["created_at"]) > APPROVAL_EXPIRY_SECONDS:
                req["status"] = "EXPIRED"
                expired.append((aid, req))
        return expired


# Single process-wide instance
state = TradeStateManager()


# ============================================================
# 2. CONFIRMATION -- turns a detected chart pattern into an actual
#    "fire or wait" decision.
#
#    Rule:
#      1. A pattern only fires on a MARUBOZU candle closing beyond the
#         trigger (body >= 70% of range, reversal-side wick < 15% of
#         range, range >= 0.8x ATR) -- or a qualifying candlestick
#         confirmation pattern.
#      2. If that confirmation closes within 2x ATR of the trigger ->
#         fire at market, immediately.
#      3. If it's already stretched beyond 2x ATR -> don't chase. Compute
#         a Fibonacci discount/premium zone (50%-79% retracement) on the
#         trigger->extreme leg, and fire a LIMIT order at the 61.8%
#         anchor within that zone, with a 15-bar expiry.
#      4. If no confirmation appears within 20 bars of a pattern becoming
#         valid, the watch is abandoned (stale).
# ============================================================

STALE_BARS = 20
FIB_WAIT_BARS = 15
MARUBOZU_BODY_RATIO = 0.70
MARUBOZU_WICK_RATIO = 0.15
MARUBOZU_ATR_RATIO = 0.8
FAR_ATR_MULTIPLE = 2.0
FIB_ZONE_LOW = 0.50
FIB_ZONE_HIGH = 0.79
FIB_ENTRY_ANCHOR = 0.618


def is_marubozu(o, h, l, c, atr):
    rng = h - l
    if rng <= 0 or atr is None or atr <= 0:
        return False
    if rng < MARUBOZU_ATR_RATIO * atr:
        return False
    body = abs(c - o)
    if body / rng < MARUBOZU_BODY_RATIO:
        return False
    reversal_wick = (h - c) if c >= o else (c - l)
    if reversal_wick / rng > MARUBOZU_WICK_RATIO:
        return False
    return True


def fib_discount_premium_zone(trigger_price, extreme_price, bias):
    """Returns (zone_low, zone_high, entry_anchor_price) for the pullback zone."""
    if bias == "BUY":
        leg = extreme_price - trigger_price
        zone_low = extreme_price - leg * FIB_ZONE_HIGH
        zone_high = extreme_price - leg * FIB_ZONE_LOW
        entry = extreme_price - leg * FIB_ENTRY_ANCHOR
    else:
        leg = trigger_price - extreme_price
        zone_high = extreme_price + leg * FIB_ZONE_HIGH
        zone_low = extreme_price + leg * FIB_ZONE_LOW
        entry = extreme_price + leg * FIB_ENTRY_ANCHOR
    return zone_low, zone_high, entry


def check_current_confirmation(df, trigger_price, bias):
    """One-off check of the LATEST candle against a trigger (for on-demand display)."""
    if len(df) < 3:
        return False, None
    o = float(df['Open'].iloc[-1]); h = float(df['High'].iloc[-1])
    l = float(df['Low'].iloc[-1]);  c = float(df['Close'].iloc[-1])
    atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else None

    broke = (c > trigger_price) if bias == "BUY" else (c < trigger_price)
    if not broke:
        return False, None

    if is_marubozu(o, h, l, c, atr):
        return True, "Marubozu"

    candle_confirmed, candle_name = detect_confirmation_candle(df, bias)
    if candle_confirmed:
        return True, candle_name

    return False, None


class ConfirmationEngine:
    """Holds per (symbol, timeframe) watch state across successive polls."""

    def __init__(self):
        self._watches = {}  # (symbol, tf) -> dict

    def reset(self, symbol, tf):
        self._watches.pop((symbol, tf), None)

    def step(self, symbol, tf, df, best_pattern):
        """
        Returns a dict:
          {"action": "NONE"|"FIRE_MARKET"|"FIRE_LIMIT",
           "pattern": DetectedPattern or None,
           "fire_price": float or None,
           "order_type": "MARKET"|"LIMIT"|None,
           "expiry_bars": int or None,
           "reason": str}
        """
        key = (symbol, tf)

        if best_pattern is None:
            self._watches.pop(key, None)
            return {"action": "NONE", "pattern": None, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "no_pattern"}

        watch = self._watches.get(key)
        if watch is None or watch["pattern_name"] != best_pattern.name or watch["bias"] != best_pattern.bias:
            watch = {"pattern_name": best_pattern.name, "bias": best_pattern.bias,
                      "trigger_price": best_pattern.trigger_price, "bars_watched": 0, "state": "WATCHING"}
            self._watches[key] = watch
        else:
            watch["trigger_price"] = best_pattern.trigger_price  # keep fresh for sloped necklines (H&S)

        if watch["state"] != "WATCHING":
            return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "already_resolved"}

        watch["bars_watched"] += 1
        if watch["bars_watched"] > STALE_BARS:
            self._watches.pop(key, None)
            return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "stale_pattern_timeout"}

        o = float(df['Open'].iloc[-1]); h = float(df['High'].iloc[-1])
        l = float(df['Low'].iloc[-1]);  c = float(df['Close'].iloc[-1])
        atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else None
        trigger = watch["trigger_price"]
        bias = watch["bias"]

        broke = (c > trigger) if bias == "BUY" else (c < trigger)
        if not broke:
            return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "not_broken_yet"}

        if not is_marubozu(o, h, l, c, atr):
            candle_confirmed, candle_name = detect_confirmation_candle(df, bias)
            if not candle_confirmed:
                return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                        "order_type": None, "expiry_bars": None, "reason": "broke_but_not_confirmed"}
            confirmation_label = candle_name
        else:
            confirmation_label = "Marubozu"

        watch["state"] = "DONE"
        distance = abs(c - trigger)

        if atr and distance <= FAR_ATR_MULTIPLE * atr:
            return {"action": "FIRE_MARKET", "pattern": best_pattern, "fire_price": c,
                    "order_type": "MARKET", "expiry_bars": None,
                    "reason": f"{confirmation_label}_confirmed_near_trigger"}

        extreme = h if bias == "BUY" else l
        _, _, entry_anchor = fib_discount_premium_zone(trigger, extreme, bias)
        return {"action": "FIRE_LIMIT", "pattern": best_pattern, "fire_price": entry_anchor,
                "order_type": "LIMIT", "expiry_bars": FIB_WAIT_BARS,
                "reason": f"{confirmation_label}_confirmed_stretched_fib_pullback"}


_confirmation_engine = ConfirmationEngine()


# ============================================================
# 3. TRADE SETUP -- fire decision + detected pattern -> the final
#    entry/SL/TP1/TP2 numbers.
#
#    - SL is structural: bound to the pattern's own footprint (its
#      trigger_line span), not an arbitrary fixed lookback.
#    - TP for flags/pennants uses the measured-move (flagpole height),
#      anchored to the TRIGGER price (not the fill price).
#    - TP for everything else uses a 1.5R / 3R risk-multiple off SL.
# ============================================================

def _pattern_atr(df):
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        return float(df['ATR'].iloc[-1])
    return float((df['High'] - df['Low']).tail(14).mean())


def build_trade_setup(df, pattern, fire_decision):
    """Returns dict: entry, order_type, sl, tp1, tp2, trigger_price, bias, pattern_name."""
    entry = fire_decision["fire_price"]
    order_type = fire_decision["order_type"]
    bias = pattern.bias
    trigger = pattern.trigger_price
    atr = _pattern_atr(df)

    span_xs = [p[0] for p in (pattern.trigger_line or [])]
    n = len(df)
    if span_xs:
        window_start = max(0, min(span_xs) - 3)
    else:
        window_start = max(0, n - 60)
    local_window = df.iloc[window_start:]
    resistance_level = float(local_window['High'].max())
    support_level = float(local_window['Low'].min())

    if bias == "BUY":
        sl = min(entry, support_level) - atr * 0.5
        risk = max(abs(entry - sl), atr * 0.25)
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 3.0
    else:
        sl = max(entry, resistance_level) + atr * 0.5
        risk = max(abs(sl - entry), atr * 0.25)
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 3.0

    if "Flag" in pattern.name or "Pennant" in pattern.name:
        pole_pts = [p for p in (pattern.key_points or []) if "Pole" in p[2]]
        if len(pole_pts) >= 2:
            pole_height = abs(pole_pts[1][1] - pole_pts[0][1])
            if pole_height > 0:
                if bias == "BUY":
                    tp1 = trigger + pole_height * 0.618
                    tp2 = trigger + pole_height * 1.0
                else:
                    tp1 = trigger - pole_height * 0.618
                    tp2 = trigger - pole_height * 1.0

    return {
        "entry": entry, "order_type": order_type, "bias": bias,
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "trigger_price": trigger, "pattern_name": pattern.name,
        "category": pattern.category, "confidence": pattern.confidence,
        "note": pattern.note, "expiry_bars": fire_decision.get("expiry_bars"),
    }


# ============================================================
# 4. HTF CONTEXT -- for a lower-timeframe entry signal, check the higher
#    timeframe's own structure/trend first. Scalping timeframes are for
#    entry TIMING -- the higher timeframe establishes the actual
#    directional bias. Aligned signals get reinforced; counter-trend
#    signals get flagged.
# ============================================================

LTF_TO_HTF = {
    "1min": "15min",
    "3min": "30min",
    "5min": "1h",
    "15min": "4h",
}

HTF_ALIGN_BONUS = 8.0
HTF_COUNTER_PENALTY = -10.0


# ------------------------------------------------------------------
# DEAD CODE as of the unified-decision merge: get_htf_bias(),
# htf_alignment_adjustment(), and _trendline_fallback() below are no
# longer called by poll(). HTF alignment and trendline-only setups are
# now handled inside unified_strategy.analyze() (the single decision
# source for both this live path and Telegram reports) so they aren't
# scored twice by two different HTF adjustments. Left in place rather
# than deleted in case report/legacy code elsewhere still calls them
# directly -- if nothing does after a repo-wide check, safe to remove.
# ------------------------------------------------------------------
def get_htf_bias(symbol, ltf_timeframe):
    """Returns (bias, description) or (None, None) if no HTF mapping/data available."""
    htf_tf = LTF_TO_HTF.get(ltf_timeframe)
    if htf_tf is None:
        return None, None

    df = market_data.fetch_candles(symbol, htf_tf, count=150)
    if df is None or df.empty or len(df) < 40:
        return None, None

    best, _ = scan_all_patterns(df)
    if best is not None:
        return best.bias, f"{best.name} on {htf_tf}"

    if 'EMA50' in df.columns:
        current_close = float(df['Close'].iloc[-1])
        ema50 = float(df['EMA50'].iloc[-1])
        bias = "BUY" if current_close > ema50 else "SELL"
        return bias, f"price {'above' if bias=='BUY' else 'below'} EMA50 trend on {htf_tf}"

    return None, None


def htf_alignment_adjustment(ltf_bias, htf_bias, htf_desc):
    """Returns (confidence_delta, note_text). Zero delta if no HTF context is available."""
    if htf_bias is None:
        return 0.0, "No higher-timeframe context available."
    if ltf_bias == htf_bias:
        return HTF_ALIGN_BONUS, f"Aligned with higher-timeframe bias: {htf_desc}."
    return HTF_COUNTER_PENALTY, f"⚠️ COUNTER-TREND: against higher-timeframe bias ({htf_desc}) -- treat with extra caution."


# ============================================================
# 5. ENGINE -- ties scan_all_patterns -> ConfirmationEngine -> trade_setup
#    -> trade state (mode routing) into one call per (symbol, timeframe).
#    This is what the EA's poll hits every new bar, and what the
#    background watchlist scanner calls too.
# ============================================================

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
_notify_callbacks = {"auto_fired": None, "approval_request": None, "manual_ticket": None, "report_event": None}


def register_notifiers(auto_fired=None, approval_request=None, manual_ticket=None):
    if auto_fired: _notify_callbacks["auto_fired"] = auto_fired
    if approval_request: _notify_callbacks["approval_request"] = approval_request
    if manual_ticket: _notify_callbacks["manual_ticket"] = manual_ticket


def poll(symbol, tf, magic_number=None, pushed_df=None):
    """
    Main entry point. Called two ways:
      - By the EA's poll request (pushed_df = the candles it read straight
        from its own MT5 terminal). Also marks this symbol's EA heartbeat
        as alive.
      - By the background watchlist scanner when no EA has been seen
        recently for a symbol (pushed_df = None -> falls back through
        market_data.fetch_candles).

    Regardless of Mode, if no EA has checked in recently for this symbol,
    a confirmed signal is automatically routed as a manual Copy Trade
    ticket instead of trying to command an executor that isn't there.

    Decision source: unified_strategy.analyze() is now the ONLY brain that
    decides direction/readiness/ticket -- both for this live path and for
    Telegram's /analyze-style report. This function no longer runs its own
    separate pattern-scan+confirmation vote; it only (a) gets the shared
    classic-pattern read from analyze()'s trendline intelligence so the EA
    still has something to draw, and (b) waits for that SAME pattern's
    trigger to actually break and get a confirmation candle before firing,
    so a snapshot-level "ready" can't fire mid-bar on a level that hasn't
    really been taken out yet.
    """
    mode = state.get_mode()
    response = {"mode": mode, "lot_mode": state.lot_mode, "command": None, "pattern": None}

    if mode == MODE_OFF:
        return response

    if pushed_df is not None:
        df = pushed_df
        state.mark_ea_seen(symbol)
    else:
        df = market_data.fetch_candles(symbol, tf, count=250)

    if df is None or df.empty or len(df) < 41:
        response["error"] = "insufficient_data"
        return response

    established_df = df.iloc[:-1]
    volume_profile = compute_volume_profile(established_df)

    try:
        result = unified_strategy.analyze(symbol, timeframe=tf, include_htf=True, df=established_df)
    except Exception as exc:
        print(f"[execution] unified_strategy.analyze failed for {symbol}: {exc!r}")
        response["error"] = "analysis_failed"
        return response

    family = (result.get("trendline_intelligence") or {}).get("raw") or {}
    scanned = family.get("scanned_pattern")
    family_df = family.get("df") if family.get("df") is not None else established_df
    best = SimpleNamespace(**scanned) if scanned else None

    response["reason"] = result.get("reason")
    response["decision"] = result.get("decision")
    response["score"] = result.get("score")
    response["evidence"] = result.get("evidence")
    response["signal_id"] = result.get("signal_id")
    response["pattern"] = best.name if best else None
    response["trigger_price"] = getattr(best, "trigger_price", None) if best else None
    response["bias"] = getattr(best, "bias", None) if best else None
    response["htf_bias"] = (result.get("htf") or {}).get("direction")
    response["htf_context"] = (result.get("htf") or {}).get("bias_4h") or (result.get("htf") or {}).get("bias")
    if volume_profile is not None:
        response["poc_price"] = volume_profile["poc_price"]
        response["value_area_low"] = volume_profile["value_area_low"]
        response["value_area_high"] = volume_profile["value_area_high"]

    try:
        is_ranging, ranging_reason = is_price_ranging_vs_sma(established_df)
        response["sma_ranging"] = is_ranging
        response["sma_ranging_reason"] = ranging_reason
    except Exception:
        response["sma_ranging"] = None
        response["sma_ranging_reason"] = None

    if best is not None:
        response["category"] = getattr(best, "category", None)
        response["stage"] = getattr(best, "stage", None)
        response["confidence"] = getattr(best, "confidence", None)

        def _pts_to_epoch(pts):
            out = []
            for pt in (pts or []):
                try:
                    xi = int(round(float(pt[0])))
                    y = float(pt[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if 0 <= xi < len(family_df):
                    t = family_df.index[xi]
                    out.append([int(t.timestamp()), y])
            return out

        boundary_lines = getattr(best, "boundary_lines", None)
        if boundary_lines:
            response["upper_line"] = _pts_to_epoch(boundary_lines.get("upper"))
            response["lower_line"] = _pts_to_epoch(boundary_lines.get("lower"))
        else:
            # Classic marker patterns (H&S, Double Top...) don't have a
            # two-rail boundary_lines dict -- fall back to trigger_line
            # (the neckline) so the EA still has something to draw.
            response["upper_line"] = []
            response["lower_line"] = _pts_to_epoch(getattr(best, "trigger_line", None))

    # --- Single fire gate ------------------------------------------------
    # No pattern to watch, or the multi-engine confluence isn't ready, or
    # the two disagree on direction -> nothing to do. One "no" from either
    # side is enough; there's no second opinion left to overrule it because
    # there's only one decision now.
    if (
        best is None
        or not result.get("ready")
        or result.get("direction") not in ("BUY", "SELL")
        or getattr(best, "bias", None) != result.get("direction")
    ):
        _confirmation_engine.reset(symbol, tf)
        if best is not None and result.get("direction") in ("BUY", "SELL") and getattr(best, "bias", None) != result.get("direction"):
            response["reason"] = "pattern_confluence_direction_mismatch"
        response["command"] = state.pop_command(symbol)
        return response

    # Confluence says the setup is real; still wait for THIS pattern's own
    # trigger to actually break and confirm before risking money on it --
    # `result["ready"]` is a snapshot, this is the timing gate on top of it.
    decision = _confirmation_engine.step(symbol, tf, df, best)
    response["reason"] = decision["reason"]

    if decision["action"] not in ("FIRE_MARKET", "FIRE_LIMIT"):
        response["command"] = state.pop_command(symbol)
        return response

    ticket = result.get("ticket")
    if not ticket:
        # ready=True but analyze() didn't attach a ticket -- shouldn't
        # normally happen (see unified_strategy.analyze's ticket-building
        # block), but fail safe to the pattern-based setup builder rather
        # than firing with no numbers.
        print(f"[execution] ready=True with no ticket from unified_strategy for {symbol}; using pattern-based setup")
        ticket = build_trade_setup(df, best, decision)
    else:
        ticket = dict(ticket)
        ticket.setdefault("order_type", decision.get("order_type") or "MARKET")

    ticket["symbol"] = symbol
    ticket["timeframe"] = tf
    ticket["bias"] = result.get("direction")
    ticket["pattern_name"] = getattr(best, "name", None)
    ticket["category"] = getattr(best, "category", None)
    ticket.setdefault("confidence", result.get("score"))
    ticket["note"] = getattr(best, "note", "")
    ticket["expiry_bars"] = decision.get("expiry_bars")
    ticket["signal_id"] = result.get("signal_id")
    setup = ticket

    ea_available = state.is_ea_available(symbol)
    effective_mode = mode if ea_available else MODE_COPY_TRADE

    if effective_mode == MODE_AUTO:
        state.queue_command(symbol, setup)
        if _notify_callbacks["auto_fired"]:
            _notify_callbacks["auto_fired"](setup)

    elif effective_mode == MODE_APPROVAL:
        summary = _format_setup_summary(setup)
        approval_id = state.create_approval_request(symbol, summary, setup)
        if _notify_callbacks["approval_request"]:
            _notify_callbacks["approval_request"](approval_id, setup)

    elif effective_mode == MODE_COPY_TRADE:
        if not ea_available and mode != MODE_COPY_TRADE:
            setup["note"] = (setup.get("note", "") + " [Auto-routed to manual ticket: no EA connection detected.]").strip()
        if _notify_callbacks["manual_ticket"]:
            _notify_callbacks["manual_ticket"](setup)

    response["command"] = state.pop_command(symbol)
    return response


def _format_setup_summary(setup):
    return (f"{setup['symbol']} | {setup['pattern_name']} | {setup['bias']} "
            f"[{setup['order_type']}] entry {setup['entry']:.5f} SL {setup['sl']:.5f} "
            f"TP1 {setup['tp1']:.5f} TP2 {setup['tp2']:.5f}")


def report_event(symbol, event_type, details):
    """EA calls this (via the API) to report fills/TP hits/SL hits so Telegram can relay them."""
    _log_outcome(symbol, event_type, details)
    if _notify_callbacks.get("report_event"):
        _notify_callbacks["report_event"](symbol, event_type, details)


def _log_outcome(symbol, event_type, details):
    """
    Append trade outcomes to the same log unified_strategy.analyze() writes
    signals to, keyed by signal_id when present, so signals can eventually
    be joined to what actually happened and the scoring weights recalibrated
    against real results instead of staying fixed guesses.

    NOTE: this only links to a specific signal if the EA echoes back the
    `confirmation_signal_id` field from the command it was given (the EA
    source isn't part of this codebase, so that echo may need adding on
    that side). Until then this still logs every outcome unlinked, which
    is strictly better than not logging outcomes at all.
    """
    try:
        details = details or {}
        record = {
            "record_type": "outcome",
            "ts": time.time(),
            "symbol": symbol,
            "event_type": event_type,
            "signal_id": details.get("confirmation_signal_id") or details.get("signal_id"),
            "profit": details.get("profit"),
            "ticket": details.get("ticket"),
        }
        with open(unified_strategy.SIGNAL_LOG_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"[execution] outcome log write failed: {exc!r}")


# ============================================================
# 6. FLASK ROUTES -- the EA calls these. Registered onto the existing
#    Flask `app` in bot.py via register_routes(app). All routes require a
#    shared API key (header X-API-KEY) so nothing else on the machine can
#    fire trades.
# ============================================================

API_KEY = os.environ.get("EA_API_KEY", "change-me")


def _check_key():
    return request.headers.get("X-API-KEY") == API_KEY


def _df_from_posted_candles(candles):
    """
    Build a normalized OHLC dataframe from the candle list an EA POSTs
    (its own live MT5 data -- more authoritative than the bot re-fetching
    from Deriv/MT5 independently, and lets the bot mark this symbol's EA
    heartbeat as genuinely alive, per mark_ea_seen()'s contract).
    Expected shape: [{"time": <epoch seconds>, "open":, "high":, "low":, "close":}, ...]
    """
    if not candles:
        return None
    try:
        import pandas as pd
        rows = []
        for c in candles:
            rows.append({
                "time": c.get("time"),
                "Open": c.get("open"), "High": c.get("high"),
                "Low": c.get("low"), "Close": c.get("close"),
            })
        df = pd.DataFrame(rows)
        if df.empty or df[["time", "Open", "High", "Low", "Close"]].isnull().any().any():
            return None
        df["datetime"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("datetime").sort_index()
        df = df[["Open", "High", "Low", "Close"]].astype(float)
        return market_data.clean_and_normalize_data(df)
    except Exception:
        return None


def register_routes(app):

    @app.route("/api/ea/poll/<symbol>/<tf>", methods=["GET", "POST"])
    def ea_poll(symbol, tf):
        if not _check_key():
            return jsonify({"error": "unauthorized"}), 401
        pushed_df = None
        if request.method == "POST":
            data = request.get_json(force=True, silent=True) or {}
            pushed_df = _df_from_posted_candles(data.get("candles"))
        result = poll(symbol, tf, pushed_df=pushed_df)
        return jsonify(result)

    @app.route("/api/ea/report", methods=["POST"])
    def ea_report():
        if not _check_key():
            return jsonify({"error": "unauthorized"}), 401
        data = request.get_json(force=True, silent=True) or {}
        symbol = data.get("symbol")
        event_type = data.get("event_type")
        report_event(symbol, event_type, data)
        return jsonify({"ok": True})

    @app.route("/api/ea/lotmode", methods=["GET", "POST"])
    def ea_lotmode():
        if not _check_key():
            return jsonify({"error": "unauthorized"}), 401
        if request.method == "POST":
            data = request.get_json(force=True, silent=True) or {}
            new_mode = data.get("lot_mode")
            if new_mode in ("MIN", "RISK"):
                state.set_lot_mode(new_mode)
        return jsonify({"lot_mode": state.lot_mode})

    @app.route("/api/ea/dashboard", methods=["GET"])
    def ea_dashboard():
        """Lightweight status the EA can show on its own panel alongside its
        native account/broker/PnL info (which the EA already reads itself
        via MQL5 -- this just adds the mode/lot-mode Python owns)."""
        if not _check_key():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify({"mode": state.get_mode(), "lot_mode": state.lot_mode})

    @app.route("/api/status", methods=["GET"])
    def api_status():
        # No key required -- harmless read-only health check for local debugging.
        return jsonify({
            "mode": state.get_mode(),
            "lot_mode": state.lot_mode,
            "mt5_connected": market_data.is_mt5_ready(),
        })
