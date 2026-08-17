"""Live fundamental intelligence using free FRED + Alpha Vantage APIs.

The module is intentionally lightweight for Render's free tier. It fetches a
small set of macro series and a small Alpha Vantage news/economic snapshot,
then produces context for the unified strategy. It never creates a trade by
itself and fails closed when providers are unavailable.
"""
from __future__ import annotations
import os, time
from datetime import datetime, timezone
from typing import Any, Dict
import requests

FRED_KEY=os.getenv("FRED_API_KEY","").strip()
ALPHA_KEY=os.getenv("ALPHAVANTAGE_API_KEY","").strip()
FRED_URL="https://api.stlouisfed.org/fred/series/observations"
ALPHA_URL="https://www.alphavantage.co/query"
TIMEOUT=10
CACHE_TTL=900
_cache={}

# Small, high-value macro set. Values are used as context, not direct signals.
SERIES={
 "CPI":"CPIAUCSL", "UNEMPLOYMENT":"UNRATE", "FED_FUNDS":"FEDFUNDS",
 "GDP":"GDP", "10Y":"DGS10", "2Y":"DGS2", "PCE":"PCEPI"
}
ASSET_CURRENCIES={
 "EURUSD":("EUR","USD"),"GBPUSD":("GBP","USD"),"AUDUSD":("AUD","USD"),
 "NZDUSD":("NZD","USD"),"USDCAD":("USD","CAD"),"USDJPY":("USD","JPY"),
 "EURGBP":("EUR","GBP"),"EURJPY":("EUR","JPY"),"GBPJPY":("GBP","JPY"),
 "AUDJPY":("AUD","JPY"),"XAUUSD":("XAU","USD"),"GOLD":("XAU","USD"),
 "XAGUSD":("XAG","USD"),"US30":("USD","USD"),"NAS100":("USD","USD"),
 "SPX500":("USD","USD"),"BTCUSD":("BTC","USD"),"ETHUSD":("ETH","USD")
}


def _cached(key, fn):
    now=time.time(); hit=_cache.get(key)
    if hit and now-hit[0]<CACHE_TTL:return hit[1]
    try:value=fn()
    except Exception:return None
    _cache[key]=(now,value); return value


def _fred(series_id):
    if not FRED_KEY:return None
    def fetch():
        r=requests.get(FRED_URL,params={"series_id":series_id,"api_key":FRED_KEY,"file_type":"json","sort_order":"desc","limit":8},timeout=TIMEOUT); r.raise_for_status()
        obs=[]
        for x in r.json().get("observations",[]):
            try:
                if x.get("value") not in (None,"."):obs.append(float(x["value"]))
            except ValueError:pass
        return obs
    return _cached("fred:"+series_id,fetch)


def _alpha_economic(function):
    if not ALPHA_KEY:return None
    def fetch():
        r=requests.get(ALPHA_URL,params={"function":function,"apikey":ALPHA_KEY},timeout=TIMEOUT); r.raise_for_status(); return r.json()
    return _cached("alpha:"+function,fetch)


def _trend(values):
    if not values or len(values)<2:return 0.0
    return float(values[0]-values[-1])


def _macro_state():
    vals={k:_fred(v) for k,v in SERIES.items()} if FRED_KEY else {}
    # The most recent observation is first because FRED is requested desc.
    inflation=_trend(vals.get("CPI"))
    unemployment=_trend(vals.get("UNEMPLOYMENT"))
    growth=_trend(vals.get("GDP"))
    policy=_trend(vals.get("FED_FUNDS"))
    real_proxy=None
    if vals.get("10Y") and vals.get("PCE"):
        real_proxy=vals["10Y"][0]-vals["PCE"][0]
    score=0.0
    score += 1.5 if inflation<0 else -1.0 if inflation>0 else 0
    score += 1.0 if unemployment<0 else -1.0 if unemployment>0 else 0
    score += 1.0 if growth>0 else -1.0 if growth<0 else 0
    score += .5 if policy>0 else 0
    return {"available":bool(vals),"usd_macro_score":round(score,2),"series":{k:(v[0] if v else None) for k,v in vals.items()},"changes":{"cpi":round(inflation,4),"unemployment":round(unemployment,4),"gdp":round(growth,4),"fed_funds":round(policy,4)},"real_yield_proxy":round(real_proxy,4) if real_proxy is not None else None}


def _alpha_context():
    # Alpha Vantage is supplementary; economic data is provider-dependent.
    if not ALPHA_KEY:return {"available":False,"note":"ALPHAVANTAGE_API_KEY not configured"}
    data=_alpha_economic("REAL_GDP")
    return {"available":bool(data and not data.get("Note") and not data.get("Information")),"source":"Alpha Vantage","economic_snapshot":data.get("data",[])[:3] if isinstance(data,dict) else []}


def analyze(symbol:str)->Dict[str,Any]:
    symbol=symbol.upper().replace("/","")
    macro=_macro_state(); alpha=_alpha_context()
    if not macro.get("available") and not alpha.get("available"):
        return {"symbol":symbol,"available":False,"bias":"NEUTRAL","score":0,"confidence":0,"event_risk":"UNKNOWN","reason":"FRED_API_KEY and/or ALPHAVANTAGE_API_KEY are unavailable or returned no data."}
    pair=ASSET_CURRENCIES.get(symbol,("USD","USD")); base,quote=pair
    # FRED currently supplies the strongest official USD macro layer. For non-USD
    # pairs, we remain neutral rather than pretending we have foreign macro data.
    usd=float(macro.get("usd_macro_score",0)); score=usd if quote=="USD" else -usd if base=="USD" else 0.0
    # For USD-vs-USD instruments the macro layer is context only.
    if base==quote:score=0.0
    bias="BUY" if score>=1.5 else "SELL" if score<=-1.5 else "NEUTRAL"
    risk="MEDIUM" if macro.get("available") else "LOW"
    reasons=[]
    if macro.get("available"):reasons.append(f"US macro state score {usd:+.1f}; official FRED series active")
    if macro.get("real_yield_proxy") is not None:reasons.append(f"10Y minus PCE proxy: {macro['real_yield_proxy']:+.2f}")
    if alpha.get("available"):reasons.append("Alpha Vantage supplementary economic context active")
    return {"symbol":symbol,"available":True,"bias":bias,"score":round(score,2),"confidence":min(80,40+int(abs(score)*12)),"event_risk":risk,"fred":macro,"alpha_vantage":alpha,"reasons":reasons,"method":"FRED official macro state + Alpha Vantage supplementary context; no standalone trade trigger"}


def format_report(result):
    if not result.get("available"):return f"🌐 FUNDAMENTAL INTELLIGENCE — {result.get('symbol','—')}\n\nStatus: unavailable\nConfigure FRED_API_KEY and ALPHAVANTAGE_API_KEY in Render."
    lines=[f"🌐 FUNDAMENTAL INTELLIGENCE — {result['symbol']}","",f"Bias: {result['bias']}",f"Score: {result['score']:+.1f}",f"Confidence: {result.get('confidence',0)}/100",f"Event risk: {result.get('event_risk','UNKNOWN')}"]
    lines += ["","Drivers:"]+[f"• {x}" for x in result.get("reasons",[])[:5]]
    lines.append("\nFundamentals are context for the Unified Strategy, never a standalone entry signal.")
    return "\n".join(lines)
