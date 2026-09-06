"""Unified market-intelligence strategy.

Trendline, HTF (4H->1H top-down), and fundamental analysis are evidence
extractors, not user-selectable strategies. The engine runs the real
engines and reasons over market state before deciding.

This module is the single decision layer used by Telegram analysis and
the live execution path (see execution_engine.poll()). SMC and OTE were
removed -- trendline is the only chart-geometry evidence source now;
requests.get calls in the fundamental section below are the sole network
dependency beyond price data.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import market_data
import strategies
from market_analysis import analyse_structure, structure_trade_permission

STRATEGY_NAME = "Unified Market Intelligence"
POLICY = "ONE_STRATEGY_TRENDLINE_FUNDAMENTAL_INTELLIGENCE"

# Every decision this engine makes gets appended here, so scoring weights
# (currently fixed constants -- see analyze()) can eventually be checked
# against and tuned by real outcomes, instead of staying guesses forever.
# report_event() in execution_engine.py appends the matching outcome line
# when a trade tied to a signal_id closes.
SIGNAL_LOG_PATH = os.environ.get("SIGNAL_LOG_PATH", "signal_log.jsonl")


def _log_signal(result: Dict[str, Any]) -> None:
    try:
        record = {
            "signal_id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "symbol": result["symbol"],
            "timeframe": result["timeframe"],
            "decision": result["decision"],
            "direction": result["direction"],
            "ready": result["ready"],
            "score": result["score"],
            "weights": result["weights"],
            "evidence_sources": result["evidence_sources"],
            "conflict": result["conflict"],
        }
        result["signal_id"] = record["signal_id"]
        with open(SIGNAL_LOG_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"[unified] signal log write failed: {exc!r}")


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()



# ============================================================
# TOP-DOWN BIAS (formerly topdown_engine.py) -- 4H -> 1H context.
# Folded in directly during the SMC/OTE removal + file consolidation
# so this is real code, not a separate module two other files had to
# import correctly.
# ============================================================
def _ensure_atr(df):
    if df is None or df.empty:
        return df
    if "ATR" in df.columns and not df["ATR"].isna().all():
        return df
    import numpy as np
    import pandas as pd
    h=pd.to_numeric(df["High"], errors="coerce")
    l=pd.to_numeric(df["Low"], errors="coerce")
    c=pd.to_numeric(df["Close"], errors="coerce")
    tr=pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    out=df.copy(); out["ATR"]=tr.rolling(14, min_periods=1).mean()
    return out


def _line_structural_swings(df, left=3, right=3, min_gap=2, min_leg_atr=0.55):
    """Detect pivots on closes (line chart), then map them to candle extremes."""
    import numpy as np
    if df is None or len(df) < left + right + 8:
        return []
    df=_ensure_atr(df)
    close=df["Close"].to_numpy(float); high=df["High"].to_numpy(float); low=df["Low"].to_numpy(float)
    atr=df["ATR"].to_numpy(float)
    candidates=[]
    for i in range(left, len(df)-right):
        w=close[i-left:i+right+1]
        hi=close[i] >= np.max(w); lo=close[i] <= np.min(w)
        if hi and lo: continue
        if hi: candidates.append({"index":i,"type":"high","close_price":float(close[i]),"price":float(high[i]),"time":df.index[i]})
        elif lo: candidates.append({"index":i,"type":"low","close_price":float(close[i]),"price":float(low[i]),"time":df.index[i]})
    if not candidates: return []
    accepted=[]
    for p in candidates:
        if not accepted:
            accepted.append(p); continue
        q=accepted[-1]
        if p["index"]-q["index"] < min_gap:
            # same side: retain the more extreme close; opposite side: keep the later only if meaningful
            if p["type"]==q["type"]:
                better=(p["close_price"]>q["close_price"]) if p["type"]=="high" else (p["close_price"]<q["close_price"])
                if better: accepted[-1]=p
            continue
        leg=abs(p["close_price"]-q["close_price"])/max(float(atr[p["index"]]),1e-9)
        if leg < min_leg_atr:
            if p["type"]==q["type"]:
                better=(p["close_price"]>q["close_price"]) if p["type"]=="high" else (p["close_price"]<q["close_price"])
                if better: accepted[-1]=p
            continue
        if p["type"]==q["type"]:
            better=(p["close_price"]>q["close_price"]) if p["type"]=="high" else (p["close_price"]<q["close_price"])
            if better: accepted[-1]=p
        else:
            accepted.append(p)
    # final alternating cleanup
    clean=[]
    for p in accepted:
        if clean and clean[-1]["type"]==p["type"]:
            better=(p["close_price"]>clean[-1]["close_price"]) if p["type"]=="high" else (p["close_price"]<clean[-1]["close_price"])
            if better: clean[-1]=p
        else: clean.append(p)
    return clean[-16:]


def _classify_swing_structure(swings):
    highs=[p for p in swings if p["type"]=="high"]; lows=[p for p in swings if p["type"]=="low"]
    if len(highs)<2 or len(lows)<2: return "NEUTRAL"
    hh=highs[-1]["close_price"]>highs[-2]["close_price"]
    hl=lows[-1]["close_price"]>lows[-2]["close_price"]
    lh=highs[-1]["close_price"]<highs[-2]["close_price"]
    ll=lows[-1]["close_price"]<lows[-2]["close_price"]
    if hh and hl: return "BULLISH"
    if lh and ll: return "BEARISH"
    return "TRANSITION"


def _cluster_htf_levels(df, swings, max_levels=3):
    if df is None or not swings: return []
    df=_ensure_atr(df); atr=float(df["ATR"].tail(50).mean()) if len(df) else 0.0
    tol=max(atr*0.55, 1e-9)
    clusters=[]
    for p in swings:
        px=float(p["price"]); ctype="resistance" if p["type"]=="high" else "support"
        placed=False
        for c in clusters:
            if abs(px-c["price"])<=tol:
                c["prices"].append(px); c["indices"].append(int(p["index"])); c["types"].add(ctype); c["price"]=sum(c["prices"])/len(c["prices"]); placed=True; break
        if not placed: clusters.append({"price":px,"prices":[px],"indices":[int(p["index"])],"types":{ctype}})
    close=float(df["Close"].iloc[-1]); out=[]
    for c in clusters:
        touches=len(c["prices"]);
        if touches<2: continue
        side="resistance" if c["price"]>close else "support"
        out.append({"price":float(c["price"]),"side":side,"touches":touches,"first_index":min(c["indices"]),"last_index":max(c["indices"]),"source":"4H"})
    out.sort(key=lambda x:(-x["touches"], abs(x["price"]-close)))
    return out[:max_levels]


def _timeframe_context(df, tf):
    df=_ensure_atr(df)
    swings=_line_structural_swings(df, left=3, right=3, min_gap=2, min_leg_atr=0.55)
    bias=_classify_swing_structure(swings)
    prev_h=prev_l=None
    for p in swings:
        if p["type"]=="high":
            p["label"]="HH" if prev_h is not None and p["close_price"]>prev_h else "LH" if prev_h is not None else "H"
            prev_h=p["close_price"]
        else:
            p["label"]="HL" if prev_l is not None and p["close_price"]>prev_l else "LL" if prev_l is not None else "L"
            prev_l=p["close_price"]
    return {"timeframe":tf,"structure_bias":bias,"swings":swings,"key_levels":_cluster_htf_levels(df,swings,max_levels=3),"df":df}


def _ema200_bias(df):
    if df is None or df.empty or "EMA200" not in df.columns:
        return "NEUTRAL", "EMA200 n/a", 0.0
    close = float(df["Close"].iloc[-1])
    ema200 = float(df["EMA200"].iloc[-1])
    if ema200 <= 0:
        return "NEUTRAL", "EMA200 n/a", 0.0
    dist = (close - ema200) / ema200 * 100.0
    if close > ema200 * 1.001:
        return "BUY", f"Above 200 EMA (+{dist:.2f}%)", dist
    if close < ema200 * 0.999:
        return "SELL", f"Below 200 EMA ({dist:.2f}%)", dist
    return "NEUTRAL", f"At 200 EMA ({dist:+.2f}%)", dist


def get_topdown_bias(symbol: str, count_4h: int = 200, count_1h: int = 200) -> Dict[str, Any]:
    """Full hierarchical 4H -> 1H -> 30M context.

    4H supplies macro structure and horizontal key levels. 1H supplies the
    intermediate transition. All swing detection is close/line-chart first,
    then mapped to candle High/Low for drawing. The 30M strategy remains the
    execution layer and is intentionally not changed here.
    """
    df_4h=market_data.fetch_candles(symbol,"4h",count=count_4h)
    df_1h=market_data.fetch_candles(symbol,"1h",count=count_1h)
    if df_4h is None or df_4h.empty or len(df_4h)<40:
        return {"direction":"NEUTRAL","allowed":False,"reasons":["Insufficient 4H data for top-down bias"],"bias_4h":"NEUTRAL","structure_4h":{},"structure_1h":{},"df_4h":df_4h,"df_1h":df_1h,"error":"insufficient_4h_data"}
    if df_1h is None or df_1h.empty or len(df_1h)<40:
        return {"direction":"NEUTRAL","allowed":False,"reasons":["Insufficient 1H data for top-down bias"],"bias_4h":"NEUTRAL","structure_4h":{},"structure_1h":{},"df_4h":df_4h,"df_1h":df_1h,"error":"insufficient_1h_data"}
    df_4h=_ensure_atr(df_4h); df_1h=_ensure_atr(df_1h)
    ema_bias,ema_note,_=_ema200_bias(df_4h)
    s4=analyse_structure(df_4h,left=3,right=3,lookback=80)
    s1=analyse_structure(df_1h,left=3,right=3,lookback=80)
    ctx4=_timeframe_context(df_4h,"4H"); ctx1=_timeframe_context(df_1h,"1H")
    macro=ema_bias
    if ctx4["structure_bias"]=="BULLISH" and macro!="SELL": macro="BUY"
    elif ctx4["structure_bias"]=="BEARISH" and macro!="BUY": macro="SELL"
    allowed,permission_reason,direction=structure_trade_permission(macro,s1)
    # Keep the existing permission engine, but expose the hierarchical swing
    # context explicitly so callers can distinguish macro structure from 30M.
    reasons=[f"4H regime: {ema_note}",f"4H line-structure: {ctx4['structure_bias']}",f"4H structure: {s4.get('note',s4.get('bias'))}",f"1H line-structure: {ctx1['structure_bias']}",f"1H structure: {s1.get('note',s1.get('bias'))}",permission_reason]
    if macro in ("BUY","SELL") and direction in ("BUY","SELL") and direction!=macro:
        reasons.append(f"⚠️ 1H confirmation ({direction}) is counter to the 4H regime ({macro}) -- caution")
    return {
        "direction":direction,"allowed":bool(allowed),"reasons":reasons,"bias_4h":macro,
        "structure_4h":s4,"structure_1h":s1,"df_4h":df_4h,"df_1h":df_1h,
        "swing_context_4h":ctx4,"swing_context_1h":ctx1,
        "swings_4h":ctx4["swings"],"swings_1h":ctx1["swings"],
        "key_levels_4h":ctx4["key_levels"],"key_levels_1h":ctx1["key_levels"],
    }


def format_topdown_summary(bias: Dict[str, Any]) -> str:
    """Short human-readable summary of the 4H/1H top-down read."""
    lines = [
        f"Top-down bias: {bias.get('bias_4h', 'NEUTRAL')} (4H)  →  "
        f"{bias.get('direction', 'NEUTRAL')} (1H confirmation)"
    ]
    for r in bias.get("reasons") or []:
        lines.append(f"  • {r}")
    return "\n".join(lines)

# ============================================================
# FUNDAMENTAL ANALYSIS (formerly fundamental_analysis.py) -- FRED +
# Alpha Vantage evidence source. Fail-soft: missing/rate-limited API
# access never blocks technical analysis, it just yields LOW
# confidence with reasons explaining why.
# ============================================================

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


def _fundamental_analyze(symbol: str) -> Dict[str, Any]:
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


def format_fundamental_report(result: Dict[str, Any]) -> str:
    if not result.get("available"):
        return "FUNDAMENTAL ANALYSIS\n\nWAIT / UNAVAILABLE\nSet FRED_API_KEY and/or ALPHA_VANTAGE_API_KEY."
    lines = ["════════════════════════════", "📊 FUNDAMENTAL ANALYSIS", "════════════════════════════", f"{result.get('symbol', '—')}", f"BIAS: {result.get('bias', 'NEUTRAL')}", f"SCORE: {result.get('score', 0):+d}/100", f"CONFIDENCE: {result.get('confidence', 'LOW')}", "", "MACRO INTELLIGENCE:"]
    lines.extend(f"• {r}" for r in result.get("reasons", [])[:6])
    return "\n".join(lines)

def alligator_state(df: pd.DataFrame) -> Dict[str, Any]:
    """Bill Williams-style Alligator state; no EMA20/50 is used."""
    median = (df["High"] + df["Low"]) / 2
    jaw = median.rolling(13, min_periods=13).mean().shift(8)
    teeth = median.rolling(8, min_periods=8).mean().shift(5)
    lips = median.rolling(5, min_periods=5).mean().shift(3)
    if len(df) < 20 or any(x.isna().iloc[-1] for x in (jaw, teeth, lips)):
        return {
            "state": "UNKNOWN",
            "direction": "NEUTRAL",
            "jaw": None,
            "teeth": None,
            "lips": None,
            "spread_atr": None,
            "opening": False,
        }
    j, t, li = [float(x.iloc[-1]) for x in (jaw, teeth, lips)]
    atr = max(float(_atr(df).iloc[-1]), 1e-12)
    spread = max(j, t, li) - min(j, t, li)
    prev = max(float(jaw.iloc[-4]), float(teeth.iloc[-4]), float(lips.iloc[-4])) - min(
        float(jaw.iloc[-4]), float(teeth.iloc[-4]), float(lips.iloc[-4])
    )
    bullish = li > t > j
    bearish = li < t < j
    opening = spread > prev * 1.08
    compressed = spread < atr * 0.35
    if compressed:
        state = "SLEEPING"
    elif bullish and opening:
        state = "AWAKENING_BULLISH"
    elif bearish and opening:
        state = "AWAKENING_BEARISH"
    elif bullish:
        state = "BULLISH"
    elif bearish:
        state = "BEARISH"
    else:
        state = "TRANSITION"
    return {
        "state": state,
        "direction": "BUY" if bullish else "SELL" if bearish else "NEUTRAL",
        "jaw": j,
        "teeth": t,
        "lips": li,
        "spread_atr": round(spread / atr, 2),
        "opening": opening,
    }


def _safe_dir(value: Any) -> str:
    v = str(value or "NEUTRAL").upper()
    if v in ("BUY", "BULLISH", "LONG"):
        return "BUY"
    if v in ("SELL", "BEARISH", "SHORT"):
        return "SELL"
    return "NEUTRAL"


def _extract_trendline_intel(family: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the full Trendline engine output into intelligence."""
    if not family or family.get("error"):
        return {
            "direction": "NEUTRAL",
            "quality": 0,
            "event": "NONE",
            "touches": 0,
            "strength": 0,
            "confirmed": False,
            "active_setup": "NONE",
            "error": family.get("error") if family else "no_trendline_data",
            "raw": family,
        }

    direction = _safe_dir(family.get("short_term_signal") or family.get("direction"))
    strength = int(family.get("strength") or 0)
    touches = int(family.get("primary_touches") or 0)
    quality = min(100, max(0, strength))

    retest = family.get("trendline_retest") or {}
    event = "NONE"
    status = str(retest.get("status") or "INTACT").upper()
    if status == "BREAK_RETEST_CONFIRMED":
        event = "BREAK_RETEST_CONFIRMED"
    elif status in ("BREAK_CONFIRMED", "BREAK_DEVELOPING"):
        event = "BREAKOUT"
    elif status == "FAKEOUT":
        event = "FAKEOUT"
    elif direction == "BUY":
        event = "SUPPORT_HOLD"
    elif direction == "SELL":
        event = "RESISTANCE_HOLD"

    pos = None
    try:
        pos = strategies.build_position_container(family)
    except Exception:
        pos = None

    confirmed = bool(pos and pos.get("confirmed"))
    if not confirmed and event == "BREAK_RETEST_CONFIRMED":
        confirmed = True

    return {
        "direction": direction,
        "quality": quality,
        "strength": strength,
        "event": event,
        "touches": touches,
        "confirmed": confirmed,
        "active_setup": family.get("active_setup") or "TRENDLINE",
        "setup_scores": family.get("setup_scores") or {},
        "continuation_state": family.get("continuation_state"),
        "family_kind": family.get("family_kind"),
        "primary_quality": family.get("primary_quality"),
        "gating_notes": family.get("gating_notes") or [],
        "reasons": family.get("reasons") or [],
        "position": pos,
        "error": None,
        "raw": family,
    }


def analyze(symbol: str, timeframe: str = "30min", include_htf: bool = True, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Run the real Trendline engine (Alligator regime + HTF + fundamentals
    as supporting evidence) and produce one decision.

    Pass `df` when you already have authoritative candles for this
    symbol/timeframe (e.g. an EA just pushed its own live MT5 bars) --
    the trendline engine then reads that same data instead of pulling its
    own independent copy, which is what let the live-trading brain and
    the Telegram-report brain drift onto different bars entirely.
    """
    symbol = str(symbol or "").strip().upper()
    timeframe = timeframe or "30min"

    if df is None:
        df = market_data.fetch_candles(symbol, timeframe, count=300)
    if df is None or df.empty or len(df) < 80:
        return {
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "timeframe": timeframe,
            "error": "insufficient_data",
            "decision": "WAIT",
            "direction": "NEUTRAL",
            "ready": False,
            "score": 0,
        }

    df = df.copy()
    if "ATR" not in df.columns or df["ATR"].isna().all():
        df["ATR"] = _atr(df)

    htf: Dict[str, Any] = {}
    if include_htf:
        try:
            htf = get_topdown_bias(symbol)
        except Exception as exc:
            print(f"[unified] topdown failed for {symbol}: {exc!r}")
            htf = {}

    hdir = _safe_dir(htf.get("direction") or htf.get("bias_4h") or htf.get("bias"))

    try:
        tl_raw = strategies.run_trendline_analysis(symbol, tf_code=timeframe, topdown=htf or None, df=df)
    except Exception as exc:
        print(f"[unified] trendline engine failed for {symbol}: {exc!r}")
        tl_raw = {"error": str(exc), "direction": "NEUTRAL"}
    tl = _extract_trendline_intel(tl_raw)

    state = alligator_state(df)

    try:
        fundamental = _fundamental_analyze(symbol)
    except Exception as exc:
        print(f"[unified] fundamental analysis failed for {symbol}: {exc!r}")
        fundamental = {
            "symbol": symbol,
            "available": False,
            "bias": "NEUTRAL",
            "score": 0,
            "confidence": "LOW",
            "reasons": ["Fundamental engine unavailable"],
        }

    # --- Weighted confidence per source, instead of one-vote-each -----
    # Each source contributes a 0-100 confidence to whichever direction
    # it's actually pointing at. Direction is picked by summed weight,
    # not by how many sources merely agree, so one strong signal can
    # (correctly) outweigh a weak/ambiguous one.
    alligator_conf = {
        "AWAKENING_BULLISH": 70, "AWAKENING_BEARISH": 70,
        "BULLISH": 55, "BEARISH": 55,
    }.get(state["state"], 0)

    weights: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0}
    sources = (
        (state["direction"], alligator_conf),
        (tl["direction"], tl.get("quality") or 0),
    )
    for d, conf in sources:
        if d in weights:
            weights[d] += conf

    total_weight = weights["BUY"] + weights["SELL"]
    if total_weight <= 0:
        dominant = "NEUTRAL"
    else:
        dominant = "BUY" if weights["BUY"] >= weights["SELL"] else "SELL"
    # Conflict = a real signal on both sides, and the losing side isn't
    # negligible (avoids flagging conflict when one source is at conf=2
    # against another at conf=80).
    margin = abs(weights["BUY"] - weights["SELL"])
    conflict = weights["BUY"] > 0 and weights["SELL"] > 0 and margin < 0.6 * total_weight

    evidence: List[str] = []
    evidence_sources: set = set()  # independent engines actually backing `dominant`

    if state["direction"] == dominant and dominant in ("BUY", "SELL"):
        evidence.append(f"Alligator {state['state']} aligned")
        evidence_sources.add("alligator")

    if tl["direction"] == dominant and tl["quality"] >= 50:
        evidence.append(f"Trendline geometry aligned ({tl['quality']}%)")
        evidence_sources.add("trendline")
    if tl.get("confirmed") and tl["direction"] == dominant:
        evidence.append("Trendline entry confirmed")
        evidence_sources.add("trendline")
    if tl.get("event") in ("BREAK_RETEST_CONFIRMED", "BREAKOUT") and tl["direction"] == dominant:
        evidence.append(f"Trendline event: {tl['event']}")
        evidence_sources.add("trendline")
    if tl.get("active_setup") and tl["active_setup"] not in ("NONE", "TRENDLINE"):
        evidence.append(f"Trendline best setup: {tl['active_setup']}")
        evidence_sources.add("trendline")

    if hdir == dominant and dominant in ("BUY", "SELL"):
        evidence.append("Higher-timeframe context aligned")
        evidence_sources.add("htf")
    if hdir in ("BUY", "SELL") and dominant in ("BUY", "SELL") and hdir != dominant:
        conflict = True
        evidence.append(f"HTF conflict ({hdir})")

    fbias = str(fundamental.get("bias", "NEUTRAL")).upper()
    if fbias in ("BULLISH", "BEARISH") and dominant in ("BUY", "SELL"):
        fdir = "BUY" if fbias == "BULLISH" else "SELL"
        if fdir == dominant:
            evidence.append(f"Fundamental bias aligned ({fundamental.get('score', 0):+d})")
            evidence_sources.add("fundamental")
        else:
            conflict = True
            evidence.append(f"Fundamental conflict ({fundamental.get('score', 0):+d})")

    event_ok = bool(
        tl.get("confirmed")
        or tl.get("event") in ("BREAK_RETEST_CONFIRMED", "BREAKOUT")
    )

    location_ok = bool(
        tl.get("event") in ("SUPPORT_HOLD", "RESISTANCE_HOLD", "BREAK_RETEST_CONFIRMED")
        or tl.get("confirmed")
    )

    fundamental_ok = (
        (not fundamental.get("available"))
        or fbias == "NEUTRAL"
        or (
            (fbias == "BULLISH" and dominant == "BUY")
            or (fbias == "BEARISH" and dominant == "SELL")
        )
    )

    # Gate on independent engines agreeing, not on raw evidence-line count:
    # one strong trendline setup used to be able to add 3-4 lines to
    # `evidence` on its own and clear this bar by itself. Now at least
    # two of {alligator, trendline, htf, fundamental} have to actually
    # back the direction.
    ready = (
        dominant in ("BUY", "SELL")
        and not conflict
        and fundamental_ok
        and state["state"] not in ("SLEEPING", "TRANSITION", "UNKNOWN")
        and event_ok
        and location_ok
        and len(evidence_sources) >= 2
    )

    tech_score = min(100, 30 + len(evidence_sources) * 14)
    if tl.get("quality"):
        tech_score = min(100, int(round(tech_score * 0.6 + tl["quality"] * 0.4)))
    if fundamental.get("available"):
        fscore = abs(int(fundamental.get("score", 0)))
        score = min(100, int(round(tech_score * 0.70 + fscore * 0.30)))
    else:
        score = tech_score

    ticket = None
    if ready:
        # tl["position"] was never actually populated by
        # run_trendline_analysis (only computed on the fly inside
        # format_trendline_report for display) -- build it the same way
        # here so a ready confluence can produce a real ticket instead of
        # `ready=True` with nothing to trade.
        try:
            pos = strategies.build_position_container(tl_raw) if tl_raw and not tl_raw.get("error") else None
        except Exception as exc:
            print(f"[unified] build_position_container failed for {symbol}: {exc!r}")
            pos = None
        if pos and pos.get("confirmed") and pos.get("entry") is not None:
            pos.setdefault("order_type", "MARKET")
            ticket = pos

    result = {
        "strategy": STRATEGY_NAME,
        "policy": POLICY,
        "symbol": symbol,
        "timeframe": timeframe,
        "decision": dominant if ready else "WAIT",
        "direction": dominant,
        "ready": ready,
        "conflict": conflict,
        "fundamental_ok": fundamental_ok,
        "evidence": evidence,
        "evidence_sources": sorted(evidence_sources),
        "weights": {"BUY": round(weights["BUY"], 1), "SELL": round(weights["SELL"], 1)},
        "alligator": state,
        "trendline_intelligence": tl,
        "htf": htf,
        "fundamental": fundamental,
        "ticket": ticket,
        "df": df,
        "score": score,
        "reason": "; ".join(evidence) if evidence else "No coherent market-state sequence",
    }
    _log_signal(result)
    return result


def format_report(r: Dict[str, Any]) -> str:
    if r.get("error"):
        return f"{STRATEGY_NAME} — {r.get('symbol', '?')}\n\nWAIT\n{r['error']}"

    a = r.get("alligator") or {}
    t = r.get("trendline_intelligence") or {}
    h = r.get("htf") or {}
    f = r.get("fundamental") or {}

    lines = [
        "════════════════════════════",
        "🧠 UNIFIED MARKET INTELLIGENCE",
        "════════════════════════════",
        f"{r.get('symbol', '?')} | {r.get('timeframe', '')}",
        f"DECISION: {r.get('decision', 'WAIT')}",
        f"STATE: {a.get('state', 'UNKNOWN')}",
        f"ALLIGATOR: {a.get('direction', 'NEUTRAL')}",
        f"TRENDLINE: {t.get('direction', 'NEUTRAL')} ({t.get('quality', 0)}%) | {t.get('event', 'NONE')}",
        f"HTF: {h.get('direction') or h.get('bias_4h') or h.get('bias') or 'NEUTRAL'}",
        f"FUNDAMENTAL: {f.get('bias', 'NEUTRAL')} ({f.get('score', 0):+d}) | {f.get('confidence', 'LOW')}",
        f"SCORE: {r.get('score', 0)}/100",
    ]

    if r.get("evidence"):
        lines += ["", "INTELLIGENCE:"] + [f"• {x}" for x in r["evidence"][:10]]

    if f.get("reasons"):
        lines += ["", "FUNDAMENTAL DRIVERS:"] + [f"• {x}" for x in f["reasons"][:5]]

    if not r.get("ready"):
        notes = []
        if t.get("gating_notes"):
            notes.extend(t["gating_notes"][:2])
        if notes:
            lines += ["", "ENGINE NOTES:"] + [f"• {n}" for n in notes[:4]]

    ticket = r.get("ticket")
    if r.get("ready") and ticket and ticket.get("entry") is not None:
        lines += [
            "",
            "🎯 TRADE MODEL",
            f"ENTRY: {ticket.get('entry')}",
            f"SL: {ticket.get('sl')}",
            f"TP1: {ticket.get('tp1')}",
            f"TP2: {ticket.get('tp2')}",
            f"ORDER: {ticket.get('order_type', 'MARKET')}",
        ]

    lines += [
        "",
        f"WHY: {r.get('reason', 'No coherent market-state sequence')}",
        "",
        "Trendline / Alligator / HTF / Fundamentals are internal intelligence sources — not separate strategies.",
    ]
    return "\n".join(lines)
