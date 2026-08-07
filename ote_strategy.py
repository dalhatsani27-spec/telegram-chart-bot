"""
ote_strategy.py
===============
OTE Strategy — Fibonacci Fan + Fibonacci Expansion

Professional implementation:
  1. Detect the most recent clear impulse swing (ZigZag)
  2. Draw Fibonacci Fan (38.2 / 50 / 61.8) from the impulse origin
  3. Entry zone = deeper fan lines (50–61.8%) acting as dynamic OTE
  4. Project Fibonacci Expansion targets (127.2 / 161.8 / 200 / 261.8)
  5. Score + ticket for AUTO / APPROVAL / COPY_TRADE modes

This is a standalone strategy registered as "OTE".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from market_structure import zigzag_swings, find_swings


# ---------------------------------------------------------------------------
# Core Fibonacci calculations
# ---------------------------------------------------------------------------

FAN_RATIOS = [0.382, 0.50, 0.618]
EXPANSION_RATIOS = [1.272, 1.618, 2.0, 2.618]


def _ensure_atr(df: pd.DataFrame) -> pd.DataFrame:
    if "ATR" not in df.columns or df["ATR"].isna().all():
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df = df.copy()
        df["ATR"] = tr.rolling(14, min_periods=1).mean()
    return df


def _line_price(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _find_impulse(df: pd.DataFrame, lookback: int = 120) -> Optional[Dict[str, Any]]:
    """
    Find the most recent clean impulse leg suitable for Fan + Expansion.

    Returns:
      {
        "direction": "BUY" | "SELL",
        "start": {index, price, type},
        "end":   {index, price, type},
        "retracement": {index, price, type} | None,   # point C if available
        "leg_size": float,
      }
    """
    if df is None or len(df) < 40:
        return None

    df = _ensure_atr(df)
    n = len(df)
    swings = zigzag_swings(df, depth=5, deviation_atr=0.40)
    if len(swings) < 2:
        swings = find_swings(df, left=3, right=3)
    if len(swings) < 2:
        return None

    # Restrict to recent window
    swings = [s for s in swings if s["index"] >= max(0, n - lookback)]
    if len(swings) < 2:
        return None

    # Walk backwards looking for a strong directional leg
    for i in range(len(swings) - 1, 0, -1):
        a = swings[i - 1]
        b = swings[i]
        if a["type"] == b["type"]:
            continue

        leg = abs(b["price"] - a["price"])
        atr = float(df["ATR"].iloc[min(b["index"], n - 1)])
        if atr <= 0:
            atr = leg * 0.1
        if leg < 1.2 * atr:          # require meaningful impulse
            continue

        # Bullish impulse: low → high
        if a["type"] == "low" and b["type"] == "high" and b["price"] > a["price"]:
            # Look for a subsequent pullback low (point C)
            retrace = None
            for s in swings[i + 1:]:
                if s["type"] == "low" and s["price"] < b["price"]:
                    retrace = s
                    break
            return {
                "direction": "BUY",
                "start": a,
                "end": b,
                "retracement": retrace,
                "leg_size": leg,
            }

        # Bearish impulse: high → low
        if a["type"] == "high" and b["type"] == "low" and b["price"] < a["price"]:
            retrace = None
            for s in swings[i + 1:]:
                if s["type"] == "high" and s["price"] > b["price"]:
                    retrace = s
                    break
            return {
                "direction": "SELL",
                "start": a,
                "end": b,
                "retracement": retrace,
                "leg_size": leg,
            }

    return None


def _build_fan(impulse: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """
    Build Fibonacci Fan rays from impulse start → end.
    Each ray is defined by two points and can be extended to any bar index.
    """
    x0 = impulse["start"]["index"]
    y0 = impulse["start"]["price"]
    x1 = impulse["end"]["index"]
    y1 = impulse["end"]["price"]
    dy = y1 - y0

    fans = []
    for r in FAN_RATIOS:
        # The classic construction: vertical at x1 is divided by ratio,
        # then a line is drawn from (x0,y0) through that division point.
        y_div = y0 + dy * r
        # Slope of the fan ray
        slope = (y_div - y0) / max(x1 - x0, 1)
        y_end = y0 + slope * (n - 1 - x0)
        fans.append({
            "ratio": r,
            "label": f"{r*100:.1f}%",
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y_div,
            "slope": slope,
            "y_at_end": y_end,
        })
    return fans


def _fan_price_at(fan: Dict, x: float) -> float:
    return fan["y0"] + fan["slope"] * (x - fan["x0"])


def _build_expansion(impulse: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fibonacci Expansion (3-point style when retracement exists,
    otherwise simple extension from the impulse leg).
    """
    start = impulse["start"]["price"]
    end = impulse["end"]["price"]
    leg = impulse["leg_size"]
    direction = impulse["direction"]
    retrace = impulse.get("retracement")

    expansions = []
    if retrace is not None:
        # Classic 3-point expansion: A=start, B=end, C=retrace
        # Projection from C using the AB length
        c = retrace["price"]
        for r in EXPANSION_RATIOS:
            if direction == "BUY":
                price = c + leg * r
            else:
                price = c - leg * r
            expansions.append({
                "ratio": r,
                "label": f"{r*100:.1f}%",
                "price": float(price),
                "from_point": "C",
            })
    else:
        # Simple extension beyond the end of the impulse
        for r in EXPANSION_RATIOS:
            if direction == "BUY":
                price = end + leg * (r - 1.0)
            else:
                price = end - leg * (r - 1.0)
            expansions.append({
                "ratio": r,
                "label": f"{r*100:.1f}%",
                "price": float(price),
                "from_point": "B",
            })
    return expansions


def _evaluate_entry(
    df: pd.DataFrame,
    impulse: Dict[str, Any],
    fans: List[Dict],
    expansions: List[Dict],
) -> Dict[str, Any]:
    """
    Decide if price is currently in a valid OTE entry zone on the Fan
    and build the trade ticket.
    """
    n = len(df)
    close = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else abs(impulse["leg_size"]) * 0.1
    direction = impulse["direction"]

    # Current fan prices at the last bar
    fan_prices = sorted(
        [{"ratio": f["ratio"], "label": f["label"], "price": _fan_price_at(f, n - 1)} for f in fans],
        key=lambda x: x["price"],
    )

    # Determine which fan zone price is interacting with
    in_zone = False
    nearest_fan = None
    min_dist = 1e18
    for fp in fan_prices:
        dist = abs(close - fp["price"])
        if dist < min_dist:
            min_dist = dist
            nearest_fan = fp
        # Accept if price is within 0.45 ATR of any fan line
        if dist <= atr * 0.45:
            in_zone = True

    # For BUY we prefer price near / above the deeper fans (50-61.8)
    # For SELL we prefer price near / below the deeper fans
    reasons = []
    score = 40

    if impulse["leg_size"] >= 2.0 * atr:
        score += 15
        reasons.append(f"Strong impulse ({impulse['leg_size']/atr:.1f} ATR)")
    else:
        reasons.append(f"Moderate impulse ({impulse['leg_size']/atr:.1f} ATR)")

    if in_zone:
        score += 25
        reasons.append(f"Price interacting with Fan {nearest_fan['label']}")
    else:
        # Soft score if still between the outer fans
        if direction == "BUY":
            lowest = fan_prices[0]["price"]
            highest = fan_prices[-1]["price"]
            if lowest <= close <= highest + atr * 0.3:
                score += 10
                reasons.append("Price inside Fan channel")
        else:
            lowest = fan_prices[0]["price"]
            highest = fan_prices[-1]["price"]
            if highest >= close >= lowest - atr * 0.3:
                score += 10
                reasons.append("Price inside Fan channel")

    # Prefer deeper OTE-style interaction (50 / 61.8)
    if nearest_fan and nearest_fan["ratio"] >= 0.50:
        score += 12
        reasons.append(f"Deep Fan zone ({nearest_fan['label']}) — OTE quality")

    # Expansion targets present
    if expansions:
        score += 8
        reasons.append(f"{len(expansions)} Expansion targets projected")

    # Build ticket
    entry = close
    if direction == "BUY":
        # SL below the lowest fan or the impulse origin
        sl_candidates = [fp["price"] for fp in fan_prices] + [impulse["start"]["price"]]
        sl = min(sl_candidates) - atr * 0.35
        tps = sorted([e["price"] for e in expansions if e["price"] > entry])
    else:
        sl_candidates = [fp["price"] for fp in fan_prices] + [impulse["start"]["price"]]
        sl = max(sl_candidates) + atr * 0.35
        tps = sorted([e["price"] for e in expansions if e["price"] < entry], reverse=True)

    tp1 = tps[0] if tps else (entry + atr * 1.8 if direction == "BUY" else entry - atr * 1.8)
    tp2 = tps[1] if len(tps) > 1 else (entry + atr * 3.0 if direction == "BUY" else entry - atr * 3.0)

    risk = abs(entry - sl)
    reward = abs(tp1 - entry)
    rr = (reward / risk) if risk > 0 else 0.0

    valid = direction in ("BUY", "SELL") and score >= 58 and in_zone and rr >= 1.2

    ticket = {
        "side": "LONG" if direction == "BUY" else "SHORT",
        "direction": direction,
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr": round(rr, 2),
        "risk": float(risk),
        "reward": float(reward),
        "order_type": "MARKET",
        "nearest_fan": nearest_fan["label"] if nearest_fan else None,
    }

    return {
        "in_zone": in_zone,
        "nearest_fan": nearest_fan,
        "score": min(100, score),
        "reasons": reasons,
        "valid": valid,
        "ticket": ticket if valid else None,
        "fan_prices": fan_prices,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ote_analysis(
    symbol: str,
    timeframe: str = None,
    df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Full OTE analysis for a symbol.
    Returns a structured dict compatible with strategy_engine.
    """
    import trade_state as ts
    import mt5_data

    timeframe = timeframe or ts.state.get_watch_timeframe()

    if df is None:
        df = mt5_data.fetch_candles(symbol, timeframe, count=220)
        if df is None or df.empty or len(df) < 50:
            df = mt5_data.fetch_candles(symbol, "30min", count=220)
        if df is None or df.empty or len(df) < 50:
            df = mt5_data.fetch_candles(symbol, "15min", count=220)

    if df is None or df.empty or len(df) < 50:
        return {
            "error": "Insufficient data for OTE analysis",
            "direction": "NEUTRAL",
            "score": 0,
            "valid": False,
        }

    df = _ensure_atr(df)
    n = len(df)

    impulse = _find_impulse(df)
    if impulse is None:
        return {
            "error": "No clear impulse swing found for Fan / Expansion",
            "direction": "NEUTRAL",
            "score": 0,
            "valid": False,
            "df": df,
            "timeframe": timeframe,
        }

    fans = _build_fan(impulse, n)
    expansions = _build_expansion(impulse)
    entry_eval = _evaluate_entry(df, impulse, fans, expansions)

    direction = impulse["direction"]
    score = entry_eval["score"]
    valid = entry_eval["valid"]
    reasons = entry_eval["reasons"]

    result = {
        "strategy": "OTE",
        "direction": direction if valid else "NEUTRAL",
        "score": score,
        "reasons": reasons,
        "valid": valid,
        "impulse": impulse,
        "fans": fans,
        "expansions": expansions,
        "fan_prices": entry_eval["fan_prices"],
        "nearest_fan": entry_eval["nearest_fan"],
        "in_zone": entry_eval["in_zone"],
        "position": entry_eval["ticket"],
        "ticket": entry_eval["ticket"],
        "df": df,
        "timeframe": timeframe,
        "symbol": symbol,
    }
    return result


def format_ote_report(analysis: Dict[str, Any]) -> str:
    if analysis.get("error"):
        return f"OTE  |  {analysis.get('symbol', '')}\n{analysis['error']}"

    symbol = analysis.get("symbol", "")
    direction = analysis.get("direction", "NEUTRAL")
    score = analysis.get("score", 0)
    valid = analysis.get("valid", False)
    impulse = analysis.get("impulse") or {}
    fans = analysis.get("fans") or []
    expansions = analysis.get("expansions") or []
    ticket = analysis.get("ticket")
    nearest = analysis.get("nearest_fan")

    lines = [
        f"📐 OTE  (Fib Fan + Expansion)  |  {symbol}",
        f"Direction: {direction}  |  Score: {score}/100  |  {'✅ VALID' if valid else '⏳ WAIT'}",
        f"Impulse: {impulse.get('start', {}).get('type', '?')} → {impulse.get('end', {}).get('type', '?')}  "
        f"({impulse.get('leg_size', 0):.5f})",
    ]

    if fans:
        fan_str = " · ".join(f"{f['label']}" for f in fans)
        lines.append(f"Fan rays: {fan_str}")
    if nearest:
        lines.append(f"Nearest Fan: {nearest.get('label')} @ {nearest.get('price', 0):.5f}")

    if expansions:
        exp_str = " · ".join(f"{e['label']} {e['price']:.5f}" for e in expansions[:3])
        lines.append(f"Expansion targets: {exp_str}")

    for r in analysis.get("reasons") or []:
        lines.append(f"  • {r}")

    if ticket:
        lines.append(
            f"Ticket: {ticket['side']}  Entry {ticket['entry']:.5f}  "
            f"SL {ticket['sl']:.5f}  TP1 {ticket['tp1']:.5f}  TP2 {ticket['tp2']:.5f}"
        )
        lines.append(f"R:R 1:{ticket.get('rr', 0):.2f}")

    return "\n".join(lines)


def build_ote_ticket(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return analysis.get("ticket")
