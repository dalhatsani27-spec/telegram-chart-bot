"""Macro fundamental layer for the trading-analysis bot.

Uses Trading Economics when TRADINGECONOMICS_API_KEY is configured. The
engine is intentionally advisory: it scores macro direction and event risk,
but never invents data when the provider is unavailable.

Supported: FX, gold/silver, oil, indices and crypto proxies. The engine
maps an asset to the currencies/economies that primarily drive it, evaluates
recent high-impact releases and upcoming event risk, and produces a
fundamental bias that can be fused with technical strategies.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import requests

API_KEY = os.getenv("TRADINGECONOMICS_API_KEY", "").strip()
BASE_URL = "https://api.tradingeconomics.com"
TIMEOUT = 12

# Primary macro relationships. Values are directional from the asset's point
# of view: positive score = supportive of the asset, negative = bearish.
ASSET_MAP = {
    "EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"], "AUDUSD": ["AUD", "USD"],
    "NZDUSD": ["NZD", "USD"], "USDCAD": ["USD", "CAD"], "USDJPY": ["USD", "JPY"],
    "EURGBP": ["EUR", "GBP"], "EURJPY": ["EUR", "JPY"], "GBPJPY": ["GBP", "JPY"],
    "AUDJPY": ["AUD", "JPY"], "GBPAUD": ["GBP", "AUD"], "EURAUD": ["EUR", "AUD"],
    "XAUUSD": ["XAU", "USD"], "GOLD": ["XAU", "USD"], "XAGUSD": ["XAG", "USD"],
    "OIL": ["OIL", "USD"], "US30": ["USD"], "NAS100": ["USD"], "SPX500": ["USD"],
    "BTCUSD": ["BTC", "USD"], "ETHUSD": ["ETH", "USD"],
}

COUNTRY_CURRENCY = {
    "USD": "united states", "EUR": "euro area", "GBP": "united kingdom",
    "JPY": "japan", "CAD": "canada", "AUD": "australia", "NZD": "new zealand",
    "CHF": "switzerland", "CNY": "china", "XAU": "united states", "XAG": "united states",
    "OIL": "united states", "BTC": "united states", "ETH": "united states",
}

# Macro releases where surprises usually have a meaningful market impact.
EVENT_WEIGHTS = {
    "interest rate": 5.0, "central bank": 5.0, "inflation": 4.5, "cpi": 4.5,
    "core inflation": 4.5, "gdp": 3.5, "non farm payrolls": 4.5,
    "employment change": 4.0, "unemployment": 3.5, "retail sales": 2.5,
    "pmi": 2.5, "manufacturing": 2.0, "services": 2.0, "wage": 3.0,
    "jobless claims": 2.5, "trade balance": 1.5, "oil inventories": 2.5,
}


def _get(path: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    if not API_KEY:
        return []
    p = dict(params or {})
    p["c"] = API_KEY
    p["f"] = "json"
    r = requests.get(BASE_URL + path, params=p, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _event_weight(event: Dict[str, Any]) -> float:
    text = f"{event.get('Event','')} {event.get('Category','')}".lower()
    for key, weight in EVENT_WEIGHTS.items():
        if key in text:
            return weight
    return 1.0


def _importance(event: Dict[str, Any]) -> int:
    try:
        return int(event.get("Importance") or 0)
    except Exception:
        return 0


def _num(value):
    try:
        if value is None or value == "": return None
        return float(str(value).replace(",", "").replace("%", ""))
    except Exception:
        return None


def _surprise_score(event: Dict[str, Any]) -> float:
    """Score actual-vs-forecast surprises without assuming every indicator
    has the same sign. This is a raw surprise magnitude; interpretation is
    supplied by event-specific economic direction below."""
    actual, forecast = _num(event.get("Actual")), _num(event.get("Forecast"))
    if actual is None or forecast is None or forecast == 0:
        return 0.0
    return max(-3.0, min(3.0, (actual - forecast) / max(abs(forecast), 1e-9) * 10))


def _economic_direction(event: Dict[str, Any]) -> float:
    """Approximate currency impact: +1 is currency-supportive, -1 adverse.
    This is deliberately conservative; unknown releases contribute only risk."""
    text = f"{event.get('Event','')} {event.get('Category','')}".lower()
    surprise = _surprise_score(event)
    if not surprise:
        return 0.0
    if any(k in text for k in ("inflation", "cpi", "core inflation", "wage")):
        return 1.0 if surprise > 0 else -1.0
    if any(k in text for k in ("gdp", "retail sales", "pmi", "manufacturing", "services", "employment change")):
        return 1.0 if surprise > 0 else -1.0
    if "unemployment" in text or "jobless claims" in text:
        return -1.0 if surprise > 0 else 1.0
    if "interest rate" in text or "central bank" in text:
        return 1.0 if surprise > 0 else -1.0
    return 0.0


def _currency_fundamentals(currency: str, country: str, days: int = 21) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        events = _get(f"/calendar/country/{country}/{start}/{end}")
    except Exception as exc:
        return {"currency": currency, "country": country, "error": str(exc), "score": 0.0, "events": []}

    score = 0.0
    recent = []
    upcoming = []
    now_dt = datetime.now(timezone.utc)
    for e in events:
        if _importance(e) < 2:
            continue
        weight = _event_weight(e)
        dt_raw = str(e.get("Date") or "")
        try:
            dt = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            dt = now_dt
        if dt <= now_dt:
            score += _economic_direction(e) * weight
            recent.append({"event": e.get("Event"), "actual": e.get("Actual"), "forecast": e.get("Forecast"), "importance": _importance(e)})
        else:
            upcoming.append({"event": e.get("Event"), "date": dt_raw, "importance": _importance(e), "weight": weight})

    return {"currency": currency, "country": country, "score": round(score, 2),
            "recent": recent[-8:], "upcoming": sorted(upcoming, key=lambda x: x.get("date", ""))[:8]}


def analyze(symbol: str) -> Dict[str, Any]:
    """Return a structured fundamental snapshot for a tradable symbol."""
    symbol = symbol.upper().replace("/", "")
    if not API_KEY:
        return {"symbol": symbol, "available": False, "bias": "NEUTRAL", "score": 0,
                "event_risk": "UNKNOWN", "reason": "TRADINGECONOMICS_API_KEY is not configured."}
    assets = ASSET_MAP.get(symbol)
    if not assets:
        return {"symbol": symbol, "available": True, "bias": "NEUTRAL", "score": 0,
                "event_risk": "UNKNOWN", "reason": "No macro mapping for this symbol."}

    components = []
    for ccy in assets:
        country = COUNTRY_CURRENCY.get(ccy)
        if country:
            components.append(_currency_fundamentals(ccy, country))

    # Pair logic: first currency is the numerator, second the denominator.
    if len(assets) >= 2 and assets[0] not in ("XAU", "XAG", "OIL", "BTC", "ETH"):
        score = components[0]["score"] - components[1]["score"]
    elif assets[0] in ("XAU", "XAG"):
        # Gold/silver tend to benefit from weaker USD / easier policy and risk-off.
        score = -components[-1]["score"]
    elif assets[0] == "OIL":
        score = -components[-1]["score"]
    else:
        score = components[-1]["score"]

    upcoming = [x for c in components for x in c.get("upcoming", [])]
    high = sum(1 for x in upcoming if x.get("importance", 0) >= 3)
    event_risk = "HIGH" if high >= 2 else "MEDIUM" if upcoming else "LOW"
    bias = "BUY" if score >= 5 else "SELL" if score <= -5 else "NEUTRAL"
    confidence = min(100, int(50 + min(abs(score), 30) * 1.6)) if components else 0
    reasons = []
    for c in components:
        if abs(c.get("score", 0)) >= 3:
            side = "supportive" if c["score"] > 0 else "deteriorating"
            reasons.append(f"{c['currency']} macro backdrop {side} ({c['score']:+.1f})")
    if high:
        reasons.append(f"{high} high-impact event(s) in the next 7 days")

    return {"symbol": symbol, "available": True, "bias": bias, "score": round(score, 2),
            "confidence": confidence, "event_risk": event_risk, "components": components,
            "upcoming": sorted(upcoming, key=lambda x: x.get("date", ""))[:10], "reasons": reasons}


def format_report(result: Dict[str, Any]) -> str:
    if not result.get("available"):
        return (f"🌐 FUNDAMENTAL ANALYSIS — {result.get('symbol','—')}\n\n"
                "Status: DATA PROVIDER NOT CONFIGURED\n"
                "Set TRADINGECONOMICS_API_KEY on Render to enable live macro analysis.")
    lines = ["🌐 FUNDAMENTAL ANALYSIS — " + result["symbol"], "", f"Bias: {result['bias']}",
             f"Macro Score: {result['score']:+.1f}", f"Confidence: {result.get('confidence',0)}/100",
             f"Event Risk: {result['event_risk']}"]
    if result.get("reasons"):
        lines += ["", "Drivers:"] + [f"• {r}" for r in result["reasons"][:6]]
    if result.get("upcoming"):
        lines += ["", "Upcoming risk:"]
        for e in result["upcoming"][:5]:
            lines.append(f"• {e.get('date','')} — {e.get('event','')} (importance {e.get('importance',0)})")
    lines += ["", "Fundamental data is a context filter, not a standalone trade trigger."]
    return "\n".join(lines)
