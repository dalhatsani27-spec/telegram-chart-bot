"""
trendline_confidence.py
=======================
Deterministic confidence layer for the Trendline V3 analysis.

It does not create a trade by itself. It scores the already-detected
geometry/pattern/confirmation evidence so the Telegram report can show
exactly why a setup received its final confidence.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def _pattern_score(family: Dict[str, Any], direction: str) -> tuple[float, str]:
    sp = family.get("scanned_pattern") or {}
    name = str(sp.get("name") or "")
    bias = str(sp.get("bias") or "NEUTRAL")
    raw = float(sp.get("confidence") or 0.0)
    stage = str(sp.get("stage") or "").upper()

    if raw > 0:
        score = raw
        detail = name or "Named pattern"
    else:
        wedge = family.get("wedge")
        mw = family.get("mw_pattern")
        if wedge:
            score = 70.0 if wedge.get("bias") in ("BUY", "SELL") else 55.0
            detail = wedge.get("pattern", "Converging structure")
            bias = wedge.get("bias", "NEUTRAL")
        elif mw:
            score = 68.0
            detail = mw.get("name", "M/W structure")
            bias = mw.get("bias", "NEUTRAL")
        elif family.get("channel"):
            score = 58.0
            detail = f"{str(family.get('family_kind') or 'trend').capitalize()} channel"
            bias = direction
        else:
            score = 35.0
            detail = "No named pattern"
            bias = "NEUTRAL"

    if direction in ("BUY", "SELL") and bias in ("BUY", "SELL"):
        score += 10.0 if bias == direction else -15.0

    if stage == "CONFIRMED":
        score += 12.0
    elif stage == "TRIGGERED":
        score += 5.0
    elif stage == "FORMING":
        score -= 18.0
    elif stage == "FAKEOUT":
        score -= 35.0

    return _clamp(score), detail


def _geometry_score(family: Dict[str, Any]) -> tuple[float, str]:
    quality = family.get("primary_quality")
    touches = int(family.get("primary_touches") or 0)
    line = (family.get("uptrends") or family.get("downtrends") or family.get("family_lines") or [None])[0]
    violations = int((line or {}).get("violations") or 0)

    if quality == "confirmed":
        base = 88.0
    elif quality == "crowded":
        base = 70.0
    elif quality == "unconfirmed":
        base = 55.0
    elif line:
        base = 48.0
    else:
        base = 32.0

    base += min(10.0, max(0, touches - 2) * 4.0)
    base -= min(22.0, violations * 4.0)
    if family.get("channel"):
        base += 5.0
    if family.get("wedge"):
        base += 4.0

    detail = f"{touches} touches" + (f", {violations} close violation(s)" if violations else ", no close violations")
    return _clamp(base), detail


def _breakout_score(family: Dict[str, Any]) -> tuple[float, str]:
    brk = family.get("breakout_grade") or {}
    if not brk:
        return 55.0, "No fresh trendline breakout; continuation state"

    strength = str(brk.get("strength") or "weak").lower()
    score = {"confirmed": 95.0, "developing": 65.0, "weak": 32.0}.get(strength, 45.0)
    pen = float(brk.get("penetration_atr") or 0.0)
    consecutive = int(brk.get("consecutive_closes") or 0)
    body = float(brk.get("body_ratio") or 0.0)
    score += min(5.0, pen * 4.0)
    score += min(5.0, max(0, consecutive - 1) * 3.0)
    score += min(5.0, body * 4.0)
    detail = f"{strength}, {pen:.2f} ATR penetration, {consecutive} close(s), body {body:.2f}"
    return _clamp(score), detail


def _confirmation_score(family: Dict[str, Any]) -> tuple[float, str]:
    rules = family.get("entry_rules") or {}
    checks = rules.get("checks")
    if not checks:
        return 45.0, "Entry confirmation unavailable"
    passed = int(rules.get("passed") or 0)
    required = int(rules.get("required") or 3)
    score = (passed / max(4, len(checks))) * 100.0
    if passed >= required:
        score += 8.0
    return _clamp(score), f"{passed}/{len(checks)} checks passed (need {required}+)"


def _momentum_score(family: Dict[str, Any], direction: str) -> tuple[float, str]:
    rules = family.get("entry_rules") or {}
    checks = rules.get("checks") or {}
    momentum = checks.get("momentum", (False, "n/a"))
    rsi = checks.get("rsi", (False, "n/a"))
    candle = checks.get("candle", (False, "n/a"))
    score = 40.0 + (25.0 if momentum[0] else 0.0) + (20.0 if rsi[0] else 0.0) + (15.0 if candle[0] else 0.0)
    return _clamp(score), f"Momentum={'PASS' if momentum[0] else 'WAIT'}; RSI={'PASS' if rsi[0] else 'WAIT'}"


def _fib_score(family: Dict[str, Any], direction: str) -> tuple[float, str]:
    """Score confluence with a meaningful retracement of the latest impulse."""
    df = family.get("df")
    pivots = family.get("pivots") or []
    if df is None or len(pivots) < 2:
        return 35.0, "Insufficient impulse anchors"

    a, b = pivots[-2], pivots[-1]
    if a.get("type") == b.get("type"):
        return 35.0, "No clean alternating impulse"
    high, low = max(float(a["price"]), float(b["price"])), min(float(a["price"]), float(b["price"]))
    leg = high - low
    if leg <= 0:
        return 35.0, "Flat impulse"

    close = float(df["Close"].iloc[-1])
    if b.get("type") == "high":
        retr = (high - close) / leg
    else:
        retr = (close - low) / leg

    if direction == "BUY" and b.get("type") != "high":
        return 38.0, "Latest leg is not a bullish impulse"
    if direction == "SELL" and b.get("type") != "low":
        return 38.0, "Latest leg is not a bearish impulse"

    levels = (0.382, 0.50, 0.618, 0.786)
    nearest = min(levels, key=lambda r: abs(retr - r))
    distance = abs(retr - nearest)
    score = 85.0 if distance <= 0.035 else 72.0 if distance <= 0.075 else 52.0 if distance <= 0.13 else 35.0
    return score, f"{retr*100:.1f}% retracement, nearest {nearest*100:.1f}%"


def _topdown_score(family: Dict[str, Any], direction: str) -> tuple[float, str]:
    td = family.get("topdown") or {}
    if direction not in ("BUY", "SELL"):
        return 35.0, "No directional setup"
    bias4 = str(td.get("bias_4h") or "NEUTRAL")
    bias1 = str(td.get("direction") or "NEUTRAL")
    allowed = bool(td.get("allowed"))
    score = 50.0
    if bias4 == direction:
        score += 20.0
    elif bias4 in ("BUY", "SELL"):
        score -= 10.0
    if bias1 == direction:
        score += 20.0
    elif bias1 in ("BUY", "SELL"):
        score -= 10.0
    if allowed and bias1 == direction:
        score += 10.0
    return _clamp(score), f"4H={bias4}, 1H={bias1}, permission={'YES' if allowed else 'NO'}"


def _ob_score(family: Dict[str, Any], direction: str) -> tuple[float, str]:
    active = family.get("active_order_block")
    nearby = active
    if not nearby:
        aligned = "bullish" if direction == "BUY" else "bearish" if direction == "SELL" else None
        candidates = [o for o in (family.get("order_blocks") or []) if aligned and o.get("type") == aligned and o.get("freshness") == "untested"]
        nearby = candidates[0] if candidates else None
    if not nearby:
        return 50.0, "No active aligned order block"
    side = nearby.get("type", "")
    aligned = (direction == "BUY" and side == "bullish") or (direction == "SELL" and side == "bearish")
    confidence = float(nearby.get("confidence") or 50.0)
    score = confidence if aligned else max(0.0, 100.0 - confidence)
    return _clamp(score), f"{side} OB {'aligned' if aligned else 'opposite'}, {confidence:.0f}% quality"


WEIGHTS = {
    "geometry": 20,
    "pattern": 20,
    "breakout": 15,
    "momentum": 10,
    "confirmation": 10,
    "fibonacci": 10,
    "topdown": 10,
    "order_block": 5,
}


def calculate_confidence(family: Dict[str, Any]) -> Dict[str, Any]:
    direction = str(family.get("direction") or "NEUTRAL").upper()
    specs = {
        "geometry": _geometry_score(family),
        "pattern": _pattern_score(family, direction),
        "breakout": _breakout_score(family),
        "momentum": _momentum_score(family, direction),
        "confirmation": _confirmation_score(family),
        "fibonacci": _fib_score(family, direction),
        "topdown": _topdown_score(family, direction),
        "order_block": _ob_score(family, direction),
    }
    weighted = {}
    for name, (score, detail) in specs.items():
        weighted[name] = {
            "score": round(_clamp(score), 1),
            "weight": WEIGHTS[name],
            "contribution": round(_clamp(score) * WEIGHTS[name] / 100.0, 1),
            "detail": detail,
        }
    final = sum(v["contribution"] for v in weighted.values())
    if direction == "NEUTRAL":
        final = min(final, 49.0)
    stage = str((family.get("scanned_pattern") or {}).get("stage") or "").upper()
    if stage == "FAKEOUT":
        final = min(final, 34.0)
    elif stage == "FORMING":
        final = min(final, 54.0)
    final = int(round(_clamp(final)))
    grade = "HIGH" if final >= 80 else "GOOD" if final >= 70 else "MODERATE" if final >= 60 else "LOW" if final >= 50 else "WAIT"
    return {
        "score": final,
        "grade": grade,
        "direction": direction,
        "pattern": weighted["pattern"]["detail"],
        "components": weighted,
        "formula": "20% geometry + 20% pattern + 15% breakout + 10% momentum + 10% confirmation + 10% Fibonacci + 10% top-down + 5% OB",
    }


def format_confidence_block(result: Dict[str, Any]) -> str:
    if not result:
        return ""
    labels = {
        "geometry": "Geometry", "pattern": "Pattern", "breakout": "Breakout",
        "momentum": "Momentum", "confirmation": "Confirmation", "fibonacci": "Fibonacci",
        "topdown": "4H/1H Top-down", "order_block": "Order Block",
    }
    lines = [
        "════════════════════════════",
        f"🎯 FINAL CONFIDENCE: {result['score']}% · {result['grade']}",
        f"Direction: {result.get('direction', 'NEUTRAL')}",
        f"Pattern: {result.get('pattern', 'None')}",
        "Confidence details:",
    ]
    for key, item in result.get("components", {}).items():
        lines.append(f"  • {labels.get(key, key)}: {item['score']:.0f}/100 × {item['weight']}% = {item['contribution']:.1f}")
        lines.append(f"    {item['detail']}")
    lines.append(f"Method: {result.get('formula', '')}")
    lines.append("Confidence is evidence-weighted, not a probability of profit.")
    lines.append("════════════════════════════")
    return "\n".join(lines)
