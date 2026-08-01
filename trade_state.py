"""
trade_state.py
================
Owns the system-wide trading Mode (set from Telegram) and the small command
queue the EA polls. Kept intentionally simple/in-memory: this whole system
runs as a single local process on one PC, so there's no need for a database.

Modes:
  OFF          - nothing happens, ever, regardless of what the engine finds.
  AUTO         - EA fires immediately on a confirmed signal.
  APPROVAL     - Telegram sends an Approve/Reject prompt; EA only fires after
                 you tap Approve.
  COPY_TRADE   - EA never fires. Telegram sends you a manual trade ticket
                 (entry, order type, SL, TP1, TP2) to place yourself.
"""

import time
import uuid

MODE_OFF = "OFF"
MODE_AUTO = "AUTO"
MODE_APPROVAL = "APPROVAL"
MODE_COPY_TRADE = "COPY_TRADE"
VALID_MODES = {MODE_OFF, MODE_AUTO, MODE_APPROVAL, MODE_COPY_TRADE}

APPROVAL_EXPIRY_SECONDS = 180  # how long an Approve/Reject prompt stays valid


class TradeStateManager:
    def __init__(self):
        self.mode = MODE_OFF
        self.lot_mode = "MIN"  # "MIN" or "RISK" -- mirrored from the EA's on-chart toggle / Telegram
        self._commands = {}     # symbol -> command dict, consumed by EA on next poll
        self._pending_approvals = {}  # approval_id -> dict

    # ---------------- mode ----------------
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

    # ---------------- EA command queue ----------------
    def queue_command(self, symbol, command):
        """command: dict, e.g. {"type":"FIRE_MARKET"/"FIRE_LIMIT", "bias":..., "price":..., "sl":..., "tp1":..., "tp2":..., "expiry_bars":...}"""
        command = dict(command)
        command["queued_at"] = time.time()
        self._commands[symbol] = command

    def pop_command(self, symbol):
        """EA calls this each poll. Returns the command (and clears it) or None."""
        return self._commands.pop(symbol, None)

    def peek_command(self, symbol):
        return self._commands.get(symbol)

    # ---------------- Telegram approval flow ----------------
    def create_approval_request(self, symbol, setup_summary, on_approve_command):
        approval_id = uuid.uuid4().hex[:10]
        self._pending_approvals[approval_id] = {
            "symbol": symbol, "summary": setup_summary, "command": on_approve_command,
            "created_at": time.time(), "status": "PENDING",
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
        """Call periodically (e.g. from the background loop) to auto-expire old prompts."""
        now = time.time()
        expired = []
        for aid, req in list(self._pending_approvals.items()):
            if req["status"] == "PENDING" and (now - req["created_at"]) > APPROVAL_EXPIRY_SECONDS:
                req["status"] = "EXPIRED"
                expired.append((aid, req))
        return expired


# Single process-wide instance -- imported by bot.py and the Flask routes.
state = TradeStateManager()
