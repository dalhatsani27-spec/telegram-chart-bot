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
