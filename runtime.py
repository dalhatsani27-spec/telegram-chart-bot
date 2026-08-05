"""
runtime.py — trade state, EA command queue, engine loop, Flask API, AI throttle.
"""
"""
trade_state.py
================
Owns the system-wide trading Mode (set from Telegram), strategy selection,
and the small command queue the EA polls.

Modes:
  OFF          - nothing happens
  AUTO         - EA fires immediately on confirmed signal
  APPROVAL     - Telegram Approve/Reject first
  COPY_TRADE   - Mobile Manual: only send trade tickets (no EA execution)

Strategy Modes:
  SINGLE       - run only the selected strategy
  HYBRID       - scan all enabled strategies, pick best confluence, explain why
"""

import time
import uuid

MODE_OFF = "OFF"
MODE_AUTO = "AUTO"
MODE_APPROVAL = "APPROVAL"
MODE_COPY_TRADE = "COPY_TRADE"
VALID_MODES = {MODE_OFF, MODE_AUTO, MODE_APPROVAL, MODE_COPY_TRADE}

STRATEGY_TRENDLINE = "TRENDLINE"
STRATEGY_SMC = "SMC"
STRATEGY_AMD = "AMD"
STRATEGY_SILVER_BULLET = "SILVER_BULLET"
VALID_STRATEGIES = {
    STRATEGY_TRENDLINE,
    STRATEGY_SMC,
    STRATEGY_AMD,
    STRATEGY_SILVER_BULLET,
}

STRATEGY_MODE_SINGLE = "SINGLE"
STRATEGY_MODE_HYBRID = "HYBRID"

APPROVAL_EXPIRY_SECONDS = 180

# Timeframes selectable from the Mobile Control Panel for the watched-symbol
# background scanner (Copy Trade / Mobile Manual tickets) and for on-demand
# Trendline analysis. Internal codes match mt5_data._TF_MAP / legacy_data.
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

        # Strategy selection
        self.strategy_mode = STRATEGY_MODE_SINGLE
        self.selected_strategy = STRATEGY_SMC
        self.enabled_strategies = {
            STRATEGY_TRENDLINE: True,
            STRATEGY_SMC: True,
            STRATEGY_AMD: True,
            STRATEGY_SILVER_BULLET: True,
        }
        self.min_confluence_score = 65
        self.prefer_silver_bullet = True

        self._commands = {}
        self._pending_approvals = {}
        self._ea_last_seen = {}

    # ---------------- watched symbol ----------------
    def set_watched_symbol(self, symbol):
        self.watched_symbol = symbol.strip().upper() if symbol else None

    def get_watched_symbol(self):
        return self.watched_symbol

    def clear_watched_symbol(self):
        self.watched_symbol = None

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
    def set_strategy_mode(self, mode):
        if mode not in (STRATEGY_MODE_SINGLE, STRATEGY_MODE_HYBRID):
            raise ValueError("strategy_mode must be SINGLE or HYBRID")
        self.strategy_mode = mode

    def get_strategy_mode(self):
        return self.strategy_mode

    def set_selected_strategy(self, strategy):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"invalid strategy: {strategy}")
        self.selected_strategy = strategy

    def get_selected_strategy(self):
        return self.selected_strategy

    def set_strategy_enabled(self, strategy, enabled: bool):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"invalid strategy: {strategy}")
        self.enabled_strategies[strategy] = bool(enabled)

    def is_strategy_enabled(self, strategy):
        return bool(self.enabled_strategies.get(strategy, False))

    def get_enabled_strategies(self):
        return [s for s, on in self.enabled_strategies.items() if on]

    def strategy_label(self):
        if self.strategy_mode == STRATEGY_MODE_SINGLE:
            return f"Single → {self.selected_strategy}"
        enabled = self.get_enabled_strategies()
        return f"Hybrid ({len(enabled)} active)"

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
import data as mt5_data
import analysis as htf_context
from analysis import scan_all_patterns
# self-alias so engine_api can call engine.poll
import runtime as engine  # noqa: F401 — set after load; see _bind_engine()
# lazy imports to avoid circular dependency with strategy.py
from data import compute_volume_profile
import analysis
# [merged] was: import trade_state
import data

_confirmation_engine = None

def _get_confirmation_engine():
    global _confirmation_engine
    if _confirmation_engine is None:
        from strategy import ConfirmationEngine
        _confirmation_engine = ConfirmationEngine()
    return _confirmation_engine

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

    decision = _get_confirmation_engine().step(symbol, tf, df, best)
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

    from strategy import build_trade_setup
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
"""
engine_api.py
================
Flask routes the EA calls. Registered onto the existing Flask `app` in
bot.py via register_routes(app). All routes require a shared API key
(header X-API-KEY) so nothing else on the machine can fire trades.
"""

import os
try:
    from flask import request, jsonify
except ImportError:
    request = jsonify = None
# [merged] was: import engine
# [merged] was: import trade_state
import data

API_KEY = os.environ.get("EA_API_KEY", "change-me")


def _check_key():
    return request.headers.get("X-API-KEY") == API_KEY


def register_routes(app):

    @app.route("/api/ea/poll/<symbol>/<tf>", methods=["GET"])
    def ea_poll(symbol, tf):
        if not _check_key():
            return jsonify({"error": "unauthorized"}), 401
        result = poll(symbol, tf)
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
                ts.state.set_lot_mode(new_mode)
        return jsonify({"lot_mode": ts.state.lot_mode})

    @app.route("/api/ea/dashboard", methods=["GET"])
    def ea_dashboard():
        """Lightweight status the EA can show on its own panel alongside its
        native account/broker/PnL info (which the EA already reads itself
        via MQL5 -- this just adds the mode/lot-mode Python owns)."""
        if not _check_key():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify({"mode": ts.state.get_mode(), "lot_mode": ts.state.lot_mode})

    @app.route("/api/status", methods=["GET"])
    def api_status():
        # No key required -- harmless read-only health check for local debugging.
        return jsonify({
            "mode": ts.state.get_mode(),
            "lot_mode": ts.state.lot_mode,
            "mt5_connected": mt5_data.is_mt5_ready(),
        })
"""
ai_throttle.py
================
Wraps the existing OpenRouter commentary/translation calls with:
  - A simple per-hour call budget (falls back to the plain template text once
    exhausted, never errors out).
  - A short-lived cache so repeated requests for the same text+language don't
    re-hit the API.
The caller decides WHEN to call this at all -- the real fix for "don't burn
the free tier" is calling this only on confirmed trade events, not on every
routine scan. This module is the safety net underneath that discipline.
"""

import time
import hashlib

MAX_CALLS_PER_HOUR = 20
CACHE_TTL_SECONDS = 600

_call_timestamps = []
_cache = {}  # key -> (timestamp, result)


def _budget_available():
    now = time.time()
    global _call_timestamps
    _call_timestamps = [t for t in _call_timestamps if now - t < 3600]
    return len(_call_timestamps) < MAX_CALLS_PER_HOUR


def _record_call():
    _call_timestamps.append(time.time())


def _cache_key(text, language):
    return hashlib.sha256(f"{language}:{text}".encode()).hexdigest()


def throttled_call(text, language, call_fn, fallback_text):
    """
    call_fn(text, language) -> str   (the actual OpenRouter call, e.g. translate_text or fetch_ai_commentary)
    fallback_text: what to return if we're out of budget or the call fails.
    """
    key = _cache_key(text, language)
    now = time.time()
    if key in _cache:
        ts, result = _cache[key]
        if now - ts < CACHE_TTL_SECONDS:
            return result

    if not _budget_available():
        return fallback_text

    try:
        result = call_fn(text, language)
        _record_call()
        _cache[key] = (now, result)
        return result
    except Exception:
        return fallback_text
