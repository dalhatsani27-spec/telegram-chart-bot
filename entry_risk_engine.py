"""
entry_risk_engine.py
====================
Final execution gate for the Trendline strategy.

This module deliberately does NOT draw trendlines, find pivots, choose the
SMA direction, or modify pattern detection. It consumes the already-working
Trendline analysis and answers one question only:

    Is this an executable trade RIGHT NOW?

The final gate is exactly three checks:
    1. Confirmed trendline break
    2. Confirmed retest/hold
    3. Directional entry candle confirmation

Only when all three pass do we calculate Entry / SL / TP and create a
mobile-manual / future-MT5 execution ticket.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from market_analysis import detect_confirmation_candle


DEFAULT_RISK_PERCENT = float(os.environ.get("TRADE_RISK_PERCENT", "1.0"))
DEFAULT_RR = float(os.environ.get("TRADE_TARGET_RR", "2.0"))
DEFAULT_RR2 = float(os.environ.get("TRADE_TARGET_RR2", "3.0"))
SL_BUFFER_ATR = float(os.environ.get("TRADE_SL_BUFFER_ATR", "0.25"))
MIN_SL_ATR = float(os.environ.get("TRADE_MIN_SL_ATR", "0.50"))
FALLBACK_SL_ATR = float(os.environ.get("TRADE_FALLBACK_SL_ATR", "0.80"))
MAX_SL_ATR = float(os.environ.get("TRADE_MAX_SL_ATR", "3.0"))


def _atr(df) -> float:
    if df is None or len(df) == 0:
        return 0.0
    try:
        if "ATR" in df.columns:
            value = float(df["ATR"].iloc[-1])
            if value > 0:
                return value
        return max(float(df["High"].iloc[-1] - df["Low"].iloc[-1]), 1e-9)
    except Exception:
        return 0.0


def _directional_candle_confirmation(df, direction: str):
    """Use the existing candle detector as the FINAL trigger only."""
    if df is None or len(df) < 3 or direction not in ("BUY", "SELL"):
        return False, "No confirmation candle"
    try:
        ok, name = detect_confirmation_candle(df, direction)
        return bool(ok), (name or "Directional confirmation candle")
    except Exception as exc:
        return False, f"Candle confirmation unavailable: {exc}"


def _structural_stop(df, direction: str, retest: Dict[str, Any], family: Dict[str, Any], entry: float):
    """Place the stop beyond the actual retest candle/structure.

    The retest is the trade's invalidation point: if the retest candle's
    extreme is taken out, the break/retest thesis is no longer valid.
    """
    atr = _atr(df)
    if atr <= 0:
        return None

    idx = retest.get("retest_index")
    try:
        idx = int(idx) if idx is not None else len(df) - 1
    except (TypeError, ValueError):
        idx = len(df) - 1
    idx = max(0, min(idx, len(df) - 1))

    if direction == "BUY":
        invalidation = float(df["Low"].iloc[idx])
        sl = invalidation - atr * SL_BUFFER_ATR
        if sl >= entry or (entry - sl) < atr * MIN_SL_ATR:
            sl = entry - atr * FALLBACK_SL_ATR
        if (entry - sl) > atr * MAX_SL_ATR:
            sl = entry - atr * MAX_SL_ATR
        return sl

    invalidation = float(df["High"].iloc[idx])
    sl = invalidation + atr * SL_BUFFER_ATR
    if sl <= entry or (sl - entry) < atr * MIN_SL_ATR:
        sl = entry + atr * FALLBACK_SL_ATR
    if (sl - entry) > atr * MAX_SL_ATR:
        sl = entry + atr * MAX_SL_ATR
    return sl


def _build_ticket(family: Dict[str, Any], checks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    df = family.get("df")
    direction = str(family.get("direction") or "NEUTRAL").upper()
    if df is None or getattr(df, "empty", True) or direction not in ("BUY", "SELL"):
        return None

    entry = float(df["Close"].iloc[-1])
    retest = family.get("trendline_retest") or {}
    sl = _structural_stop(df, direction, retest, family, entry)
    if sl is None:
        return None

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    sign = 1.0 if direction == "BUY" else -1.0
    tp1 = entry + sign * risk * DEFAULT_RR
    tp2 = entry + sign * risk * DEFAULT_RR2

    return {
        "symbol": family.get("symbol"),
        "timeframe": family.get("timeframe"),
        "strategy": "TRENDLINE",
        "pattern_name": "Trendline Break + Retest",
        "side": "LONG" if direction == "BUY" else "SHORT",
        "direction": direction,
        "order_type": "MARKET",
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp2,
        "tp3_basis": f"fixed RR 1:{DEFAULT_RR2:g}",
        "risk_distance": risk,
        "risk_percent": DEFAULT_RISK_PERCENT,
        "rr": DEFAULT_RR,
        "rr_tp2": DEFAULT_RR2,
        "confirmed": True,
        "entry_reason": checks["candle"]["detail"],
        "break_index": retest.get("break_index"),
        "retest_index": retest.get("retest_index"),
        "retest_level": retest.get("retest_level"),
    }


def evaluate_trendline_entry(family: Dict[str, Any]) -> Dict[str, Any]:
    """Return the final 3-check decision and an executable ticket if ready."""
    df = family.get("df") if family else None
    direction = str((family or {}).get("direction") or "NEUTRAL").upper()
    breakout = (family or {}).get("breakout_grade") or {}
    retest = (family or {}).get("trendline_retest") or {}

    break_ok = (
        direction in ("BUY", "SELL")
        and breakout.get("strength") == "confirmed"
        and retest.get("status") == "BREAK_RETEST_CONFIRMED"
    )
    retest_ok = retest.get("status") == "BREAK_RETEST_CONFIRMED"
    candle_ok, candle_name = _directional_candle_confirmation(df, direction)

    checks = {
        "break": {
            "passed": bool(break_ok),
            "detail": "Confirmed candle close beyond master trendline" if break_ok else "Waiting for confirmed trendline break",
        },
        "retest": {
            "passed": bool(retest_ok),
            "detail": "Broken trendline retested and held" if retest_ok else "Waiting for confirmed retest/hold",
        },
        "candle": {
            "passed": bool(candle_ok),
            "detail": candle_name if candle_ok else candle_name,
        },
    }
    passed = sum(1 for item in checks.values() if item["passed"])
    entry_ready = passed == 3 and direction in ("BUY", "SELL")

    ticket = _build_ticket(family, checks) if entry_ready else None
    if entry_ready and ticket is None:
        entry_ready = False
        passed = min(passed, 2)

    missing = [name for name, item in checks.items() if not item["passed"]]
    status = "ENTER_NOW" if entry_ready else "WAIT"

    return {
        "status": status,
        "entry_ready": entry_ready,
        "direction": direction,
        "checks": checks,
        "passed": passed,
        "required": 3,
        "missing": missing,
        "ticket": ticket,
        "risk_percent": DEFAULT_RISK_PERCENT,
        "target_rr": DEFAULT_RR,
        "target_rr2": DEFAULT_RR2,
    }
