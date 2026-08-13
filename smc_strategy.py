"""
smc_strategy.py
===============
Institutional-style Smart Money Concepts (SMC) analysis engine.

The engine deliberately treats SMC as a sequence rather than a collection
of isolated indicators:

1. Establish higher-timeframe directional structure.
2. Identify external liquidity (equal highs/lows and recent swing pools).
3. Detect a liquidity sweep.
4. Require displacement away from the swept pool.
5. Confirm BOS / CHoCH / MSS on the execution structure.
6. Locate the last opposing candle as the order block.
7. Locate the displacement imbalance as FVG and detect IFVG when invalidated.
8. Classify premium/discount relative to the current dealing range.
9. Prefer an unmitigated OB/FVG confluence zone after the sweep + displacement.
10. Produce entry, invalidation and liquidity-based targets without forcing a trade.

This module is analysis-only. It does not place orders and has no Telegram
or broker dependencies, so it can be tested independently and safely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from market_analysis import zigzag_swings, live_closed_candles, live_analysis_meta


# -------------------------------
# Tunable SMC parameters
# -------------------------------
SWING_DEPTH = 5
MIN_DISPLACEMENT_ATR = 1.20
MIN_BODY_RATIO = 0.55
FVG_MIN_ATR = 0.10
OB_LOOKBACK = 12
LIQUIDITY_TOL_ATR = 0.12
MAX_ZONE_AGE = 80


def _atr(df: pd.DataFrame) -> pd.Series:
    if "ATR" in df.columns:
        return pd.to_numeric(df["ATR"], errors="coerce").bfill().ffill()
    h, l, c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=1).mean()


def _body_ratio(row: pd.Series) -> float:
    rng = float(row["High"] - row["Low"])
    return abs(float(row["Close"] - row["Open"])) / rng if rng > 0 else 0.0


def _direction_from_structure(swings: List[Dict[str, Any]]) -> str:
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "NEUTRAL"
    hh = highs[-1]["price"] > highs[-2]["price"]
    hl = lows[-1]["price"] > lows[-2]["price"]
    lh = highs[-1]["price"] < highs[-2]["price"]
    ll = lows[-1]["price"] < lows[-2]["price"]
    if hh and hl:
        return "BULLISH"
    if lh and ll:
        return "BEARISH"
    return "NEUTRAL"


def _swing_liquidity(swings: List[Dict[str, Any]], df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Group nearby swing extremes into practical liquidity pools."""
    atr = float(_atr(df).iloc[-1])
    tol = max(atr * LIQUIDITY_TOL_ATR, 1e-9)
    pools = {"buy_side": [], "sell_side": []}
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]

    for i, a in enumerate(highs):
        for b in highs[i + 1:]:
            if abs(a["price"] - b["price"]) <= tol:
                pools["buy_side"].append({
                    "price": float((a["price"] + b["price"]) / 2),
                    "indices": [a["index"], b["index"]],
                    "type": "equal_highs",
                })
    for i, a in enumerate(lows):
        for b in lows[i + 1:]:
            if abs(a["price"] - b["price"]) <= tol:
                pools["sell_side"].append({
                    "price": float((a["price"] + b["price"]) / 2),
                    "indices": [a["index"], b["index"]],
                    "type": "equal_lows",
                })

    # Always retain the latest obvious external swing as a liquidity target.
    if highs:
        pools["buy_side"].append({"price": float(highs[-1]["price"]), "indices": [highs[-1]["index"]], "type": "swing_high"})
    if lows:
        pools["sell_side"].append({"price": float(lows[-1]["price"]), "indices": [lows[-1]["index"]], "type": "swing_low"})
    return pools


def detect_liquidity_sweeps(df: pd.DataFrame, swings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect wick-through-and-close-back events around prior swing liquidity."""
    if len(df) < 5 or not swings:
        return []
    atr = _atr(df).to_numpy(float)
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    events = []

    for i in range(max(SWING_DEPTH, len(df) - MAX_ZONE_AGE), len(df)):
        a = max(float(atr[i]), 1e-9)
        h, l, c = map(float, (df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i]))
        prior_highs = [s for s in highs if s["index"] < i]
        prior_lows = [s for s in lows if s["index"] < i]
        if prior_highs:
            level = max(s["price"] for s in prior_highs)
            if h > level + 0.05 * a and c < level:
                events.append({"index": i, "type": "BUY_SIDE_SWEEP", "level": level, "price": h})
        if prior_lows:
            level = min(s["price"] for s in prior_lows)
            if l < level - 0.05 * a and c > level:
                events.append({"index": i, "type": "SELL_SIDE_SWEEP", "level": level, "price": l})
    return events


def detect_fvg(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Three-candle imbalance: bullish low[i] > high[i-2], bearish high[i] < low[i-2]."""
    if len(df) < 3:
        return []
    atr = _atr(df).to_numpy(float)
    zones = []
    for i in range(2, len(df)):
        h2 = float(df["High"].iloc[i - 2]); l2 = float(df["Low"].iloc[i - 2])
        h0 = float(df["High"].iloc[i]); l0 = float(df["Low"].iloc[i])
        min_gap = max(float(atr[i]) * FVG_MIN_ATR, 1e-9)
        if l0 - h2 >= min_gap:
            zones.append({"type": "BULLISH_FVG", "index": i, "low": h2, "high": l0, "mitigated": False})
        elif l2 - h0 >= min_gap:
            zones.append({"type": "BEARISH_FVG", "index": i, "low": h0, "high": l2, "mitigated": False})
    # Mark mitigation only after the FVG forms.
    for z in zones:
        after = df.iloc[z["index"] + 1:]
        if after.empty:
            continue
        if z["type"] == "BULLISH_FVG":
            z["mitigated"] = bool((after["Low"] <= z["low"]).any())
        else:
            z["mitigated"] = bool((after["High"] >= z["high"]).any())
    return zones


def detect_order_blocks(df: pd.DataFrame, displacement_index: Optional[int], direction: str) -> List[Dict[str, Any]]:
    """Last opposing candle before displacement, with freshness/mitigation state."""
    if displacement_index is None or displacement_index <= 0:
        return []
    start = max(0, displacement_index - OB_LOOKBACK)
    out = []
    want_bearish = direction == "BUY"
    for i in range(displacement_index - 1, start - 1, -1):
        o = float(df["Open"].iloc[i]); c = float(df["Close"].iloc[i])
        is_opposing = c < o if want_bearish else c > o
        if not is_opposing:
            continue
        low = float(df["Low"].iloc[i]); high = float(df["High"].iloc[i])
        after = df.iloc[i + 1:]
        if direction == "BUY":
            mitigated = bool((after["Low"] <= low).any())
        else:
            mitigated = bool((after["High"] >= high).any())
        out.append({"index": i, "type": "BULLISH_OB" if direction == "BUY" else "BEARISH_OB",
                    "low": low, "high": high, "mitigated": mitigated})
        break
    return out


def detect_bos_choch(df: pd.DataFrame, swings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Close-confirmed structure breaks using the most recent prior swing level."""
    events = []
    closes = df["Close"].to_numpy(float)
    for i in range(len(df)):
        prior_highs = [s for s in swings if s["index"] < i and s["type"] == "high"]
        prior_lows = [s for s in swings if s["index"] < i and s["type"] == "low"]
        if prior_highs:
            level = prior_highs[-1]["price"]
            if closes[i] > level:
                prev = [e for e in events if e["type"] in ("BOS_BULLISH", "CHOCH_BULLISH")]
                events.append({"index": i, "type": "BOS_BULLISH" if prev else "CHOCH_BULLISH", "level": level})
        if prior_lows:
            level = prior_lows[-1]["price"]
            if closes[i] < level:
                prev = [e for e in events if e["type"] in ("BOS_BEARISH", "CHOCH_BEARISH")]
                events.append({"index": i, "type": "BOS_BEARISH" if prev else "CHOCH_BEARISH", "level": level})
    # De-duplicate consecutive bars breaking the same level.
    cleaned = []
    for e in events:
        if cleaned and e["type"] == cleaned[-1]["type"] and abs(e["level"] - cleaned[-1]["level"]) < 1e-12:
            continue
        cleaned.append(e)
    return cleaned


def _displacement(df: pd.DataFrame, candidate_indices: List[int], direction: str) -> Optional[Dict[str, Any]]:
    atr = _atr(df)
    for i in sorted(candidate_indices, reverse=True):
        row = df.iloc[i]
        body = abs(float(row["Close"] - row["Open"]))
        rng = float(row["High"] - row["Low"])
        if rng <= 0:
            continue
        bullish = float(row["Close"]) > float(row["Open"])
        if (direction == "BUY" and not bullish) or (direction == "SELL" and bullish):
            continue
        if body >= MIN_DISPLACEMENT_ATR * max(float(atr.iloc[i]), 1e-9) and body / rng >= MIN_BODY_RATIO:
            return {"index": i, "direction": direction, "body_atr": body / max(float(atr.iloc[i]), 1e-9)}
    return None


def _premium_discount(df: pd.DataFrame, swings: List[Dict[str, Any]], price: float) -> Dict[str, Any]:
    highs = [s["price"] for s in swings if s["type"] == "high"]
    lows = [s["price"] for s in swings if s["type"] == "low"]
    if not highs or not lows:
        return {"zone": "EQUILIBRIUM", "equilibrium": price, "range_high": price, "range_low": price, "position": 50.0}
    hi, lo = max(highs[-3:]), min(lows[-3:])
    span = max(hi - lo, 1e-9)
    pos = (price - lo) / span * 100.0
    zone = "DISCOUNT" if pos < 50 else "PREMIUM" if pos > 50 else "EQUILIBRIUM"
    return {"zone": zone, "equilibrium": lo + span * 0.5, "range_high": hi, "range_low": lo, "position": pos}


def _zone_contains(price: float, zone: Optional[Dict[str, Any]]) -> bool:
    return bool(zone and float(zone["low"]) <= price <= float(zone["high"]))


def _nearest_target(price: float, pools: List[Dict[str, Any]], direction: str) -> Optional[float]:
    levels = [float(p["price"]) for p in pools]
    if direction == "BUY":
        levels = [x for x in levels if x > price]
        return min(levels) if levels else None
    levels = [x for x in levels if x < price]
    return max(levels) if levels else None


def analyse_smc(df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Return a complete SMC map and a conservative setup decision."""
    if df is None or len(df) < 60:
        return {"error": "insufficient_data", "strategy": "SMC", "signal": "WAIT"}

    work = live_closed_candles(df.copy().sort_index(), "30min")
    if work is None or len(work) < 60:
        return {"error": "insufficient_closed_data", "strategy": "SMC", "signal": "WAIT",
                "live_analysis": live_analysis_meta(work, "30min")}
    swings = zigzag_swings(work, depth=SWING_DEPTH, deviation_atr=0.35)
    structure_bias = _direction_from_structure(swings)
    events = detect_bos_choch(work, swings)
    sweeps = detect_liquidity_sweeps(work, swings)
    fvgs = detect_fvg(work)
    liquidity = _swing_liquidity(swings, work)

    price = float(work["Close"].iloc[-1])
    # A sweep is only actionable while it is recent. A week-old sweep must
    # never turn an otherwise neutral live chart into a new entry.
    fresh_window = max(6, min(12, SWING_DEPTH * 2))
    fresh_start = max(0, len(work) - fresh_window)
    recent_sweeps = [x for x in sweeps if int(x.get("index", -1)) >= fresh_start]
    last_sweep = recent_sweeps[-1] if recent_sweeps else None
    directional_hint = "BUY" if last_sweep and last_sweep["type"] == "SELL_SIDE_SWEEP" else "SELL" if last_sweep and last_sweep["type"] == "BUY_SIDE_SWEEP" else ("BUY" if structure_bias == "BULLISH" else "SELL" if structure_bias == "BEARISH" else None)

    recent_breaks = [e for e in events if e["index"] >= len(work) - 30]
    if directional_hint:
        matching = [e for e in recent_breaks if (directional_hint == "BUY" and "BULLISH" in e["type"]) or (directional_hint == "SELL" and "BEARISH" in e["type"])]
    else:
        matching = recent_breaks
    displacement = _displacement(work, [e["index"] for e in matching], directional_hint) if directional_hint else None

    if displacement is None and directional_hint and last_sweep:
        candidates = list(range(max(0, last_sweep["index"]), len(work)))
        displacement = _displacement(work, candidates, directional_hint)

    ob = detect_order_blocks(work, displacement["index"] if displacement else None, directional_hint) if directional_hint else []
    unmitigated_ob = next((z for z in ob if not z["mitigated"]), None)
    unmitigated_fvgs = [z for z in fvgs[-20:] if not z["mitigated"] and ((directional_hint == "BUY" and z["type"] == "BULLISH_FVG") or (directional_hint == "SELL" and z["type"] == "BEARISH_FVG"))]
    fvg = unmitigated_fvgs[-1] if unmitigated_fvgs else None

    pd_zone = _premium_discount(work, swings, price)
    htf_bias = None
    if htf_df is not None and len(htf_df) >= 40:
        htf_swings = zigzag_swings(htf_df, depth=SWING_DEPTH, deviation_atr=0.35)
        htf_bias = _direction_from_structure(htf_swings)

    confluence = 0
    reasons: List[str] = []
    if last_sweep:
        confluence += 25; reasons.append(f"Liquidity sweep: {last_sweep['type'].replace('_', ' ')}")
    if displacement:
        confluence += 25; reasons.append(f"Displacement: {displacement['body_atr']:.1f} ATR body")
    if recent_breaks:
        confluence += 20; reasons.append(f"Structure break: {recent_breaks[-1]['type']}")
    if unmitigated_ob:
        confluence += 15; reasons.append("Fresh unmitigated order block")
    if fvg:
        confluence += 10; reasons.append("Unmitigated FVG")
    if directional_hint == "BUY" and pd_zone["zone"] == "DISCOUNT":
        confluence += 5; reasons.append("Long setup is in discount")
    elif directional_hint == "SELL" and pd_zone["zone"] == "PREMIUM":
        confluence += 5; reasons.append("Short setup is in premium")
    if htf_bias and ((directional_hint == "BUY" and htf_bias == "BULLISH") or (directional_hint == "SELL" and htf_bias == "BEARISH")):
        confluence += 10; reasons.append("Higher timeframe agrees")
    elif htf_bias and directional_hint:
        confluence -= 15; reasons.append("Higher timeframe conflicts")

    entry_zone = unmitigated_ob or fvg
    # Do not resurrect an old zone as a new entry. The actionable zone must
    # be tied to the current sweep/displacement sequence.
    if entry_zone and last_sweep:
        zone_idx = int(entry_zone.get("formed_index", entry_zone.get("index", -1)))
        if zone_idx >= 0 and zone_idx < int(last_sweep.get("index", -1)):
            entry_zone = None
    entry = price
    if entry_zone:
        entry = (float(entry_zone["low"]) + float(entry_zone["high"])) / 2
    target_pools = liquidity["buy_side"] if directional_hint == "BUY" else liquidity["sell_side"] if directional_hint == "SELL" else []
    target = _nearest_target(entry, target_pools, directional_hint) if directional_hint else None

    if directional_hint == "BUY" and entry_zone:
        sl = float(entry_zone["low"]) - float(_atr(work).iloc[-1]) * 0.25
        if target is None or target <= entry:
            target = entry + abs(entry - sl) * 2.0
    elif directional_hint == "SELL" and entry_zone:
        sl = float(entry_zone["high"]) + float(_atr(work).iloc[-1]) * 0.25
        if target is None or target >= entry:
            target = entry - abs(sl - entry) * 2.0
    else:
        sl = None
        target = None

    signal = "WAIT"
    fresh_break = bool(matching and any(int(e.get("index", -1)) >= int(last_sweep.get("index", -1)) for e in matching)) if last_sweep else False
    displacement_after_sweep = bool(displacement and last_sweep and int(displacement.get("index", -1)) >= int(last_sweep.get("index", -1)))
    # LIVE ENTRY GATE: the complete chain must have happened recently and in
    # chronological order. Historical completed setups are analysis only.
    if directional_hint and displacement_after_sweep and last_sweep and entry_zone and fresh_break and confluence >= 65:
        signal = directional_hint

    return {
        "strategy": "SMC",
        "live_analysis": live_analysis_meta(work, "30min"),
        "signal": signal,
        "direction": directional_hint,
        "confidence": int(max(0, min(100, confluence))),
        "structure_bias": structure_bias,
        "htf_bias": htf_bias,
        "last_event": events[-1] if events else None,
        "last_sweep": last_sweep,
        "displacement": displacement,
        "order_block": unmitigated_ob,
        "fvg": fvg,
        "ifvg": None,
        "premium_discount": pd_zone,
        "liquidity": liquidity,
        "entry": entry if signal != "WAIT" else None,
        "sl": sl if signal != "WAIT" else None,
        "tp1": target if signal != "WAIT" else None,
        "tp2": (entry + 2 * abs(target - entry)) if signal == "BUY" and target is not None else (entry - 2 * abs(target - entry) if signal == "SELL" and target is not None else None),
        "reasons": reasons,
        "swings": swings[-12:],
        "events": events[-12:],
        "sweeps": sweeps[-12:],
        "fvgs": fvgs[-12:],
    }


def format_smc_report(result: Dict[str, Any], symbol: str, timeframe: str = "") -> str:
    if result.get("error"):
        return f"🧠 SMC | {symbol}\n\n⚪ WAIT\n{result['error']}"
    signal = result.get("signal", "WAIT")
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    pdz = result.get("premium_discount") or {}
    lines = [
        "════════════════════════════",
        f"🧠 SMC MARKET MAP | {symbol}",
        "════════════════════════════",
        f"{emoji} {signal}  |  Confidence: {result.get('confidence', 0)}/100",
        f"Structure : {result.get('structure_bias', 'NEUTRAL')}",
        f"HTF Bias  : {result.get('htf_bias') or 'N/A'}",
        f"Dealing Range: {pdz.get('zone', 'N/A')} ({pdz.get('position', 50):.1f}%)",
        "",
        "SMC SEQUENCE",
    ]
    for reason in result.get("reasons", [])[:7]:
        lines.append(f"• {reason}")
    if result.get("last_sweep"):
        s = result["last_sweep"]
        lines.append(f"• Sweep level: {s['level']:.5f}")
    if result.get("order_block"):
        z = result["order_block"]
        lines.append(f"• OB zone: {z['low']:.5f} → {z['high']:.5f}")
    if result.get("fvg"):
        z = result["fvg"]
        lines.append(f"• FVG: {z['low']:.5f} → {z['high']:.5f}")
    lines.append("")
    if signal != "WAIT":
        lines += [
            "TRADE MAP",
            f"Entry : {result['entry']:.5f}",
            f"SL    : {result['sl']:.5f}",
            f"TP1   : {result['tp1']:.5f}",
            f"TP2   : {result['tp2']:.5f}",
        ]
    else:
        lines.append("No trade: SMC sequence is not fully confirmed. Wait for sweep → displacement → structure break → fresh zone.")
    lines.append("════════════════════════════")
    return "\n".join(lines)
