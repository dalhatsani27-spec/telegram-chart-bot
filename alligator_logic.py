"""Williams Alligator Trifecta regime/filter layer.

Replaces EMA20/EMA50 trend filtering with three Alligator confirmations:
1) alignment of Lips/Teeth/Jaw,
2) price location relative to the trio,
3) expansion/contraction of the trio normalized by ATR.

The calculation is causal: displaced lines only shift already-known values and
never read future candles.
"""
from __future__ import annotations
from typing import Any, Dict
import pandas as pd


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def calculate_alligator(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    h = pd.to_numeric(d["High"], errors="coerce")
    l = pd.to_numeric(d["Low"], errors="coerce")
    median = (h + l) / 2.0
    jaw_raw = _rma(median, 13)
    teeth_raw = _rma(median, 8)
    lips_raw = _rma(median, 5)
    d["ALLIGATOR_JAW"] = jaw_raw.shift(8)
    d["ALLIGATOR_TEETH"] = teeth_raw.shift(5)
    d["ALLIGATOR_LIPS"] = lips_raw.shift(3)
    d["ALLIGATOR_JAW_RAW"] = jaw_raw
    d["ALLIGATOR_TEETH_RAW"] = teeth_raw
    d["ALLIGATOR_LIPS_RAW"] = lips_raw
    return d


def alligator_regime(df: pd.DataFrame) -> Dict[str, Any]:
    d = calculate_alligator(df)
    if len(d) < 30:
        return {"regime": "RANGE", "direction": "NEUTRAL", "trifecta": 0}
    row = d.iloc[-1]
    close = float(row["Close"])
    jaw = float(row["ALLIGATOR_JAW"])
    teeth = float(row["ALLIGATOR_TEETH"])
    lips = float(row["ALLIGATOR_LIPS"])
    h = pd.to_numeric(d["High"], errors="coerce")
    l = pd.to_numeric(d["Low"], errors="coerce")
    c = pd.to_numeric(d["Close"], errors="coerce")
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_now = max(float(atr.iloc[-1]), 1e-12)
    bull_align = lips > teeth > jaw
    bear_align = lips < teeth < jaw
    buffer = 0.10 * atr_now
    bull_price = close > lips + buffer
    bear_price = close < lips - buffer
    spread = (abs(lips-teeth) + abs(teeth-jaw)) / atr_now
    prev = d.iloc[-4]
    prev_spread = (abs(float(prev["ALLIGATOR_LIPS"])-float(prev["ALLIGATOR_TEETH"])) + abs(float(prev["ALLIGATOR_TEETH"])-float(prev["ALLIGATOR_JAW"]))) / max(float(atr.iloc[-4]), 1e-12)
    expansion = spread > prev_spread * 1.03
    lips_slope = (lips - float(d["ALLIGATOR_LIPS"].iloc[-6])) / atr_now
    jaw_slope = (jaw - float(d["ALLIGATOR_JAW"].iloc[-6])) / atr_now
    bull_expand = expansion and lips_slope > 0.05 and jaw_slope >= 0
    bear_expand = expansion and lips_slope < -0.05 and jaw_slope <= 0
    bull_score = int(bull_align) + int(bull_price) + int(bull_expand)
    bear_score = int(bear_align) + int(bear_price) + int(bear_expand)
    if bull_score == 3:
        regime, direction = "BULL_TREND", "BUY"
    elif bear_score == 3:
        regime, direction = "BEAR_TREND", "SELL"
    elif bull_score >= 2 and bull_score > bear_score:
        regime, direction = "BULL_TRANSITION", "BUY"
    elif bear_score >= 2 and bear_score > bull_score:
        regime, direction = "BEAR_TRANSITION", "SELL"
    else:
        regime, direction = "RANGE", "NEUTRAL"
    return {"regime": regime, "direction": direction, "trifecta": max(bull_score, bear_score), "bull_trifecta": bull_score, "bear_trifecta": bear_score, "jaw": jaw, "teeth": teeth, "lips": lips, "jaw_slope_atr": round(jaw_slope,3), "lips_slope_atr": round(lips_slope,3), "spread_atr": round(spread,3), "expanding": bool(expansion), "price_above_lips": bool(bull_price), "price_below_lips": bool(bear_price), "atr": atr_now, "close": close}


def apply_alligator(result: Dict[str, Any]) -> Dict[str, Any]:
    if not result or result.get("error"):
        return result
    df = result.get("df")
    if df is None or len(df) < 30:
        return result
    ag = alligator_regime(df)
    result["alligator"] = ag
    result["market_regime"] = ag["regime"]
    result["regime"] = dict(result.get("regime") or {})
    result["regime"].update(ag)
    direction = result.get("direction", "NEUTRAL")
    score = int(result.get("score", result.get("strength", 0)))
    old_score = score
    if direction in ("BUY", "SELL"):
        if ag["direction"] == direction and ag["trifecta"] == 3:
            score = min(100, score + 10)
            result.setdefault("reasons", []).append(f"Alligator Trifecta aligned: {direction} (3/3)")
        elif ag["direction"] == direction and ag["trifecta"] >= 2:
            score = min(100, score + 4)
            result.setdefault("reasons", []).append(f"Alligator transition aligned: {direction} ({ag['trifecta']}/3)")
        elif ag["direction"] in ("BUY", "SELL") and ag["direction"] != direction:
            score = max(0, score - 12)
            result.setdefault("reasons", []).append(f"Alligator conflict: setup {direction} vs regime {ag['direction']}")
        elif ag["regime"] == "RANGE":
            score = max(0, score - 8)
            result.setdefault("reasons", []).append("Alligator mouth closed/ranging")
    result["score"] = score
    result["strength"] = score
    result["alligator_adjustment"] = score - old_score
    result["gating_notes"] = result.get("reasons", [])
    if "valid" in result:
        opposite = (direction == "BUY" and ag["direction"] == "SELL") or (direction == "SELL" and ag["direction"] == "BUY")
        result["valid"] = bool(result.get("valid")) and not opposite and score >= 55
    return result
