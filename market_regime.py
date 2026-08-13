"""
Market-regime layer for the Trendline strategy.

Purpose:
    Identify the *character* of the market before the trendline engine spends
    time labeling every micro HH/HL/LH/LL.  This is deliberately a rule-based
    context layer, not a price-direction predictor.

Regimes:
    TREND_UP / TREND_DOWN
    RANGE
    ACCUMULATION
    DISTRIBUTION
    TRANSITION

The detector combines:
    - Kaufman Efficiency Ratio (move purity)
    - ADX + DI direction (trend strength)
    - relative ATR (compression/expansion)
    - regression slope normalized by ATR
    - range containment / contraction
    - directional expansion confirmation

No single indicator is allowed to decide the regime.  ADX is used for
strength, not direction; ATR is used for volatility state; slope and DI help
with direction; ER separates clean directional movement from noisy churn.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _safe_series(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(df[name], errors="coerce").to_numpy(float)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = _safe_series(df, "High")
    l = _safe_series(df, "Low")
    c = _safe_series(df, "Close")
    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return pd.Series(tr, index=df.index).ewm(alpha=1 / period, adjust=False).mean()


def _efficiency_ratio(close: pd.Series, period: int = 20) -> float:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) <= period:
        return 0.0
    net = abs(float(c.iloc[-1]) - float(c.iloc[-1 - period]))
    path = float(c.diff().abs().iloc[-period:].sum())
    return float(net / path) if path > 0 else 0.0


def _adx_di(df: pd.DataFrame, period: int = 14) -> tuple[float, float, float]:
    h = _safe_series(df, "High")
    l = _safe_series(df, "Low")
    c = _safe_series(df, "Close")
    if len(c) < period + 3:
        return 0.0, 0.0, 0.0

    up = h[1:] - h[:-1]
    down = l[:-1] - l[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - prev), abs(l[1:] - prev)))

    tr_s = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    p_s = pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean()
    m_s = pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * p_s / tr_s.replace(0, np.nan)
    minus_di = 100 * m_s / tr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return (
        float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 0.0,
        float(plus_di.iloc[-1]) if pd.notna(plus_di.iloc[-1]) else 0.0,
        float(minus_di.iloc[-1]) if pd.notna(minus_di.iloc[-1]) else 0.0,
    )


def _regression_slope(close: pd.Series, period: int, atr_value: float) -> float:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) < period or atr_value <= 0:
        return 0.0
    y = c.iloc[-period:].to_numpy(float)
    x = np.arange(period, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope / atr_value


def _range_stats(df: pd.DataFrame, period: int, atr_value: float) -> Dict[str, float]:
    d = df.tail(period)
    hi = float(d["High"].max())
    lo = float(d["Low"].min())
    close = float(df["Close"].iloc[-1])
    width_atr = (hi - lo) / max(atr_value, 1e-12)

    # Count how often the recent closes remain inside the same normalized
    # range. High containment + low directional efficiency is range-like.
    containment = 0.0
    if hi > lo:
        containment = float(((d["Close"] >= lo) & (d["Close"] <= hi)).mean())

    location = (close - lo) / max(hi - lo, 1e-12)
    return {"width_atr": width_atr, "containment": containment, "location": location}


def detect_market_regime(
    df: pd.DataFrame,
    trend_period: int = 20,
    range_period: int = 32,
    atr_period: int = 14,
    atr_baseline: int = 50,
) -> Dict[str, Any]:
    """Classify the current market character without predicting direction.

    The thresholds are intentionally broad and interpretable.  They are a
    starting regime map and should be validated per instrument/timeframe,
    rather than treated as universal profitable thresholds.
    """
    if df is None or len(df) < max(atr_baseline, trend_period, range_period) + 5:
        return {
            "state": "INSUFFICIENT_DATA",
            "direction": "NEUTRAL",
            "confidence": 0,
            "trade_permission": "WAIT",
            "reason": "Not enough candles for a stable regime classification.",
        }

    atr_s = _atr(df, atr_period)
    atr_now = float(atr_s.iloc[-1])
    atr_base = float(atr_s.iloc[-atr_baseline:].mean())
    atr_ratio = atr_now / max(atr_base, 1e-12)

    er = _efficiency_ratio(df["Close"], trend_period)
    adx, plus_di, minus_di = _adx_di(df, atr_period)
    slope = _regression_slope(df["Close"], trend_period, atr_now)
    rs = _range_stats(df, range_period, atr_now)

    directional = abs(slope) >= 0.12 and er >= 0.28
    strong_directional = abs(slope) >= 0.20 and er >= 0.38 and adx >= 20
    compressed = atr_ratio <= 0.78
    expanded = atr_ratio >= 1.08
    contained = rs["width_atr"] <= 4.5 and rs["containment"] >= 0.92

    # Accumulation/distribution are *candidate states*, not predictions.
    # They require compression/containment plus a location bias and weak
    # directional efficiency.  Confirmation comes only after expansion.
    accumulation = compressed and contained and er <= 0.30 and rs["location"] <= 0.42
    distribution = compressed and contained and er <= 0.30 and rs["location"] >= 0.58

    if strong_directional and plus_di > minus_di and slope > 0:
        state = "TREND_UP"
        direction = "BUY"
        permission = "TRENDLINE"
    elif strong_directional and minus_di > plus_di and slope < 0:
        state = "TREND_DOWN"
        direction = "SELL"
        permission = "TRENDLINE"
    elif accumulation:
        state = "ACCUMULATION"
        direction = "NEUTRAL"
        permission = "WAIT_EXPANSION"
    elif distribution:
        state = "DISTRIBUTION"
        direction = "NEUTRAL"
        permission = "WAIT_EXPANSION"
    elif contained and not directional and adx < 23:
        state = "RANGE"
        direction = "NEUTRAL"
        permission = "NO_TRENDLINE"
    elif directional and adx >= 18:
        state = "TREND_UP" if slope > 0 and plus_di >= minus_di else "TREND_DOWN"
        direction = "BUY" if state == "TREND_UP" else "SELL"
        permission = "TRENDLINE"
    else:
        state = "TRANSITION"
        direction = "NEUTRAL"
        permission = "WAIT_CONFIRMATION"

    # Confidence measures agreement between independent dimensions. It is
    # not a win probability.
    votes = []
    votes.append(min(adx / 30.0, 1.0))
    votes.append(min(er / 0.55, 1.0))
    votes.append(min(abs(slope) / 0.35, 1.0))
    if state in ("RANGE", "ACCUMULATION", "DISTRIBUTION"):
        votes.append(1.0 if contained else 0.0)
        votes.append(1.0 if compressed or adx < 23 else 0.0)
    else:
        votes.append(1.0 if expanded or atr_ratio >= 0.85 else 0.0)
    confidence = int(round(np.mean(votes) * 100))

    if state == "TREND_UP":
        reason = "Directional movement is clean: slope/ER/ADX agree on bullish trend."
    elif state == "TREND_DOWN":
        reason = "Directional movement is clean: slope/ER/ADX agree on bearish trend."
    elif state == "ACCUMULATION":
        reason = "Compressed lower-range behavior detected; wait for bullish expansion before biasing."
    elif state == "DISTRIBUTION":
        reason = "Compressed upper-range behavior detected; wait for bearish expansion before biasing."
    elif state == "RANGE":
        reason = "Price is contained with weak directional efficiency; suppress micro structure labels."
    else:
        reason = "Market character is transitioning; wait for directional expansion or clearer range boundaries."

    return {
        "state": state,
        "direction": direction,
        "confidence": max(0, min(100, confidence)),
        "trade_permission": permission,
        "reason": reason,
        "metrics": {
            "adx": round(adx, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "efficiency_ratio": round(er, 3),
            "regression_slope_atr": round(slope, 3),
            "atr_ratio": round(atr_ratio, 3),
            "range_width_atr": round(rs["width_atr"], 3),
            "range_location": round(rs["location"], 3),
        },
    }


def regime_summary(regime: Dict[str, Any]) -> str:
    """Compact text suitable for Telegram/chart headers."""
    return (
        f"Market State: {regime.get('state', 'UNKNOWN')} | "
        f"Confidence: {int(regime.get('confidence', 0))}% | "
        f"Permission: {regime.get('trade_permission', 'WAIT')}"
    )
