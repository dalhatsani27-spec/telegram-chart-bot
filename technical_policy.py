"""Single technical-indicator policy for the bot.

Decision indicators are intentionally limited to:
- 200 EMA for long-term directional location.
- Williams Alligator for market-state/trend structure.
ATR is used only as a volatility unit, not as a directional indicator.
RSI/ADX and other legacy indicator values are not used by this policy.
"""
from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from alligator_logic import alligator_regime


def market_regime(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or len(df) < 30:
        return {"regime":"RANGE","volatility":"UNKNOWN","direction":"NEUTRAL","atr":0.0,"close":None,"ema200":None}
    d = df.copy()
    for c in ("High","Low","Close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    h, l, c = d["High"], d["Low"], d["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr = max(float(atr_s.iloc[-1]), 1e-12)
    ag = alligator_regime(d)
    ema200 = ag.get("ema200")
    close = float(c.iloc[-1])
    # 200 EMA is a location/regime filter; Alligator is the active state engine.
    direction = ag.get("direction", "NEUTRAL")
    if direction == "BUY" and ema200 is not None and close <= ema200 + 0.05*atr:
        direction = "NEUTRAL"
    if direction == "SELL" and ema200 is not None and close >= ema200 - 0.05*atr:
        direction = "NEUTRAL"
    regime = ag.get("regime", "RANGE")
    if direction == "NEUTRAL" and regime not in ("RANGE",):
        regime = "TRANSITION"
    hist = atr_s.dropna().tail(80)
    med = float(hist.median()) if len(hist) else atr
    volatility = "LOW" if atr < med*.75 else ("HIGH" if atr > med*1.5 else "NORMAL")
    return {
        "regime": regime, "volatility": volatility, "direction": direction,
        "atr": atr, "close": close, "ema200": ema200,
        "alligator": ag,
        "indicator_policy": "200EMA+ALLIGATOR_ONLY",
    }
