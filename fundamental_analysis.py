"""Fundamental-analysis layer using the user's FRED and Alpha Vantage APIs.

The module is deliberately fail-soft: missing/limited API access never blocks
technical analysis. It produces a currency-level fundamental score and a
pair-level bias that unified_strategy.py can use as another evidence source.

Environment variables:
    FRED_API_KEY
    ALPHA_VANTAGE_API_KEY (ALPHA_VANTAGE_KEY is also accepted)

Alpha Vantage NEWS_SENTIMENT is used when available. FRED is the primary
macro source for rates, inflation, unemployment, GDP and Treasury yields.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()
ALPHA_VANTAGE_API_KEY = (
    os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    or os.environ.get("ALPHA_VANTAGE_KEY", "").strip()
)

FRED_BASE = "https://api.stlouisfed.org/fred"
AV_BASE = "https://www.alphavantage.co/query"
_TIMEOUT = 12
_CACHE_TTL = 15 * 60
_CACHE: Dict[str, Tuple[float, Any]] = {}

PAIR_CURRENCIES = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"), "AUDUSD": ("AUD", "USD"),
    "USDCAD": ("USD", "CAD"), "NZDUSD": ("NZD", "USD"),
    "EURGBP": ("EUR", "GBP"), "EURJPY": ("EUR", "JPY"),
    "GBPJPY": ("GBP", "JPY"), "GBPAUD": ("GBP", "AUD"),
    "AUDJPY": ("AUD", "JPY"), "EURAUD": ("EUR", "AUD"),
    "XAUUSD": ("XAU", "USD"), "XAGUSD": ("XAG", "USD"),
}

# Reliable FRED series for the US. Other currencies are augmented by
# Alpha Vantage news sentiment; this avoids hard-coding questionable local
# series IDs and keeps the engine useful across the user's full watchlist.
US_SERIES = {
    "policy_rate": "DFF",       # Effective Federal Funds Rate
    "inflation": "CPIAUCSL",   # CPI, all urban consumers
    "unemployment": "UNRATE",
    "gdp": "GDP",
    "yield10": "DGS10",
}


def _cached(key: str) -> Any:
    item = _CACHE.get(key)
    if item and time.time() - item[0] < _CACHE_TTL:
        return item[1]
    return None


def _put(key: str, value: Any) -> Any:
    _CACHE[key] = (time.time(), value)
    return value


def _fred_series(series_id: str, limit: int = 8) -> List[float]:
    if not FRED_API_KEY:
        return []
    key = f"fred:{series_id}:{limit}"
    cached = _cached(key)
    if cached is not None:
        return cached
    params = {
        "series_id": series_id, "api_key": FRED_API_KEY,
        "file_type": "json", "sort_order": "desc", "limit": limit,
    }
    try:
        r = requests.get(f"{FRED_BASE}/series/observations", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        rows = r.json().get("observations", [])
        values = []
        for row in rows:
            try:
                v = float(row.get("value"))
                if v == v:
                    values.append(v)
            except (TypeError, ValueError):
                continue
        return _put(key, values)
    except Exception as exc:
        print(f"[fundamental] FRED {series_id} failed: {exc!r}")
        return []


def _alpha_news(ticker: str) -> Dict[str, Any]:
    if not ALPHA_VANTAGE_API_KEY or not ticker:
        return {}
    key = f"avnews:{ticker}"
    cached = _cached(key)
    if cached is not None:
        return cached
    try:
        params = {
            "function": "NEWS_SENTIMENT", "tickers": ticker,
            "limit": 50, "apikey": ALPHA_VANTAGE_API_KEY,
        }
        r = requests.get(AV_BASE, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return _put(key, data)
    except Exception as exc:
        print(f"[fundamental] Alpha Vantage news failed for {ticker}: {exc!r}")
        return {}


def _news_score(ticker: str) -> Tuple[float, int, List[str]]:
    data = _alpha_news(ticker)
    feed = data.get("feed") or []
    if not feed:
        return 0.0, 0, []
    weighted = 0.0
    weight_sum = 0.0
    for item in feed[:30]:
        try:
            sentiment = float(item.get("overall_sentiment_score", 0.0))
            relevance = float(item.get("relevance_score", 0.5))
        except (TypeError, ValueError):
            continue
        w = max(0.05, min(1.0, relevance))
        weighted += sentiment * w
        weight_sum += w
    if not weight_sum:
        return 0.0, 0, []
    score = max(-30.0, min(30.0, weighted / weight_sum * 30.0))
    label = "bullish" if score > 7 else "bearish" if score < -7 else "mixed/neutral"
    return score, len(feed), [f"Alpha Vantage news sentiment {label} ({score:+.0f})"]


def _delta(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return float(values[0] - values[1])


def _us_macro() -> Tuple[float, List[str], Dict[str, Any]]:
    if not FRED_API_KEY:
        return 0.0, ["FRED_API_KEY not configured"], {}
    rate = _fred_series(US_SERIES["policy_rate"])
    cpi = _fred_series(US_SERIES["inflation"])
    unemp = _fred_series(US_SERIES["unemployment"])
    gdp = _fred_series(US_SERIES["gdp"])
    y10 = _fred_series(US_SERIES["yield10"])

    score = 0.0
    reasons: List[str] = []
    metrics: Dict[str, Any] = {}

    if rate:
        metrics["policy_rate"] = rate[0]
        d = _delta(rate)
        score += max(-20, min(20, d * 10))
        if abs(d) >= 0.05:
            reasons.append(f"Fed funds rate changed {d:+.2f} pp")
    if cpi:
        metrics["cpi"] = cpi[0]
        d = _delta(cpi)
        score += max(-8, min(8, d * 0.8))
        if abs(d) >= 0.1:
            reasons.append(f"US CPI changed {d:+.2f}")
    if unemp:
        metrics["unemployment"] = unemp[0]
        d = _delta(unemp)
        score += max(-8, min(8, -d * 4))
        if abs(d) >= 0.1:
            reasons.append(f"US unemployment changed {d:+.2f} pp")
    if gdp:
        metrics["gdp"] = gdp[0]
        d = _delta(gdp)
        score += max(-8, min(8, d * 0.15))
        if abs(d) >= 0.1:
            reasons.append(f"US GDP changed {d:+.2f}")
    if y10:
        metrics["10y_yield"] = y10[0]
        d = _delta(y10)
        score += max(-10, min(10, d * 4))
        if abs(d) >= 0.03:
            reasons.append(f"US 10Y yield changed {d:+.2f} pp")
    return max(-60.0, min(60.0, score)), reasons, metrics


def _currency_news_ticker(currency: str) -> Optional[str]:
    # Alpha Vantage's NEWS_SENTIMENT forex filter uses FOREX:XXX.
    if currency in {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD"}:
        return f"FOREX:{currency}"
    return None


def currency_fundamental(currency: str) -> Dict[str, Any]:
    currency = str(currency or "").upper()
    if currency == "USD":
        score, reasons, metrics = _us_macro()
    else:
        score, reasons, metrics = 0.0, [f"No dedicated FRED macro model for {currency}; using news sentiment"], {}
    ticker = _currency_news_ticker(currency)
    if ticker:
        nscore, count, nreasons = _news_score(ticker)
        score += nscore
        reasons.extend(nreasons)
        metrics["news_articles"] = count
    score = max(-100.0, min(100.0, score))
    bias = "BULLISH" if score >= 20 else "BEARISH" if score <= -20 else "NEUTRAL"
    return {"currency": currency, "score": round(score), "bias": bias, "reasons": reasons[:6], "metrics": metrics}


def analyze(symbol: str) -> Dict[str, Any]:
    """Return pair-level fundamental intelligence.

    Positive score means the base currency is stronger than the quote
    currency; negative means the quote currency is stronger.
    """
    clean = str(symbol or "").upper().replace("/", "")
    base, quote = PAIR_CURRENCIES.get(clean, (None, None))
    if not base or not quote:
        return {"symbol": symbol, "available": False, "bias": "NEUTRAL", "score": 0, "confidence": "LOW", "reason": "No currency mapping for this symbol."}
    base_data = currency_fundamental(base)
    quote_data = currency_fundamental(quote)
    score = max(-100, min(100, base_data["score"] - quote_data["score"]))
    bias = "BULLISH" if score >= 20 else "BEARISH" if score <= -20 else "NEUTRAL"
    confidence = "HIGH" if abs(score) >= 50 else "MEDIUM" if abs(score) >= 25 else "LOW"
    reasons = [f"{base}: {base_data['bias']} ({base_data['score']:+d})", f"{quote}: {quote_data['bias']} ({quote_data['score']:+d})"]
    reasons.extend(base_data["reasons"][:2]); reasons.extend(quote_data["reasons"][:2])
    return {"symbol": clean, "available": bool(FRED_API_KEY or ALPHA_VANTAGE_API_KEY), "bias": bias, "score": int(score), "confidence": confidence, "base": base_data, "quote": quote_data, "reasons": reasons[:6]}


def format_report(result: Dict[str, Any]) -> str:
    if not result.get("available"):
        return "FUNDAMENTAL ANALYSIS\n\nWAIT / UNAVAILABLE\nSet FRED_API_KEY and/or ALPHA_VANTAGE_API_KEY."
    lines = ["════════════════════════════", "📊 FUNDAMENTAL ANALYSIS", "════════════════════════════", f"{result.get('symbol', '—')}", f"BIAS: {result.get('bias', 'NEUTRAL')}", f"SCORE: {result.get('score', 0):+d}/100", f"CONFIDENCE: {result.get('confidence', 'LOW')}", "", "MACRO INTELLIGENCE:"]
    lines.extend(f"• {r}" for r in result.get("reasons", [])[:6])
    return "\n".join(lines)
