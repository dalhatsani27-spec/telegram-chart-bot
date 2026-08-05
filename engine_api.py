"""
engine_api.py
================
Flask routes the EA calls. Registered onto the existing Flask `app` in
bot.py via register_routes(app). All routes require a shared API key
(header X-API-KEY) so nothing else on the machine can fire trades.
"""

import os
from flask import request, jsonify
import engine
import trade_state as ts
import mt5_data

API_KEY = os.environ.get("EA_API_KEY", "change-me")


def _check_key():
    return request.headers.get("X-API-KEY") == API_KEY


def register_routes(app):

    @app.route("/api/ea/poll/<symbol>/<tf>", methods=["GET"])
    def ea_poll(symbol, tf):
        if not _check_key():
            return jsonify({"error": "unauthorized"}), 401
        result = engine.poll(symbol, tf)
        return jsonify(result)

    @app.route("/api/ea/report", methods=["POST"])
    def ea_report():
        if not _check_key():
            return jsonify({"error": "unauthorized"}), 401
        data = request.get_json(force=True, silent=True) or {}
        symbol = data.get("symbol")
        event_type = data.get("event_type")
        engine.report_event(symbol, event_type, data)
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
