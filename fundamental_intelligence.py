"""Lightweight fundamental intelligence layer.

This module is deliberately provider-agnostic and Render-Free friendly. It
normalizes economic surprises, policy stance, cross-asset context and event
risk into a compact state. It does NOT create trades by itself.

Providers can feed records through `ingest_event()` / `build_state()`.
No raw news/candle history is retained in memory.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

HIGH_IMPACT = {"CPI","CORE_CPI","NFP","PAYROLLS","GDP","FOMC","ECB","BOE","BOJ","PCE","PMI","RATE_DECISION","RATE_SPEECH"}

@dataclass
class FundamentalEvent:
    timestamp: str
    country: str
    event: str
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    importance: str = "MEDIUM"
    hawkish: Optional[float] = None  # -1 dovish, +1 hawkish
    assets: str = ""

    def surprise(self) -> float:
        if self.actual is None or self.forecast is None:
            return 0.0
        scale = abs(self.forecast) if abs(self.forecast) > 1e-9 else 1.0
        return max(-3.0, min(3.0, (self.actual - self.forecast) / scale))


def ingest_event(country: str, event: str, actual=None, forecast=None, previous=None,
                 importance="MEDIUM", hawkish=None, assets="") -> Dict:
    e = FundamentalEvent(
        timestamp=datetime.now(timezone.utc).isoformat(), country=country.upper(),
        event=event.upper(), actual=actual, forecast=forecast, previous=previous,
        importance=importance.upper(), hawkish=hawkish, assets=assets,
    )
    return asdict(e) | {"surprise": e.surprise()}


def _country_bias(events: Iterable[Dict], country: str) -> float:
    score = 0.0
    weight = 0.0
    for e in events:
        if str(e.get("country","")).upper() != country.upper():
            continue
        w = 2.0 if str(e.get("importance","MEDIUM")).upper() == "HIGH" else 1.0
        if e.get("hawkish") is not None:
            v = max(-1.0,min(1.0,float(e["hawkish"])))
        else:
            v = max(-1.0,min(1.0,float(e.get("surprise",0.0))))
        score += v*w; weight += w
    return score/weight if weight else 0.0


def _label(x: float) -> str:
    if x >= .25: return "BULLISH"
    if x <= -.25: return "BEARISH"
    return "NEUTRAL"


def build_state(events: Iterable[Dict], base_currency: str = "USD", quote_currency: str = "") -> Dict:
    events = list(events)
    b = _country_bias(events, base_currency); q = _country_bias(events, quote_currency) if quote_currency else 0.0
    differential = b-q if quote_currency else b
    high_impact = [e for e in events if str(e.get("importance","MEDIUM")).upper()=="HIGH" or str(e.get("event","")).upper() in HIGH_IMPACT]
    return {
        "available": bool(events),
        "base_currency": base_currency,
        "quote_currency": quote_currency,
        "base_bias": _label(b), "quote_bias": _label(q) if quote_currency else "NEUTRAL",
        "score": round(max(-100.0,min(100.0,differential*100)),1),
        "bias": _label(differential),
        "event_risk": "HIGH" if high_impact else "NORMAL",
        "high_impact_count": len(high_impact),
        "event_count": len(events),
        "method": "macro_surprise+policy_stance+differential",
    }


def analyze_symbol(symbol: str, events: Iterable[Dict]) -> Dict:
    s = symbol.upper().replace("/","").replace("-","")
    currency_pairs = {"EURUSD":("EUR","USD"),"GBPUSD":("GBP","USD"),"USDJPY":("USD","JPY"),"AUDUSD":("AUD","USD"),"USDCAD":("USD","CAD"),"NZDUSD":("NZD","USD"),"EURGBP":("EUR","GBP"),"GBPJPY":("GBP","JPY"),"EURJPY":("EUR","JPY"),"AUDJPY":("AUD","JPY"),"EURAUD":("EUR","AUD")}
    if s in currency_pairs:
        return build_state(events,*currency_pairs[s])
    # For non-FX instruments, USD is the default macro anchor; callers can add
    # asset-specific flow events and policy scores through the same schema.
    return build_state(events,"USD","")


def format_state(state: Dict) -> str:
    if not state.get("available"): return "Fundamental data: UNAVAILABLE"
    return (f"Fundamentals: {state.get('bias','NEUTRAL')} | score {state.get('score',0):+.1f} | "
            f"event risk {state.get('event_risk','NORMAL')} | {state.get('event_count',0)} events")
