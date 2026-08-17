"""Macro fundamental layer for the trading-analysis bot.

Uses Trading Economics when TRADINGECONOMICS_API_KEY is configured. The engine
is advisory: it scores macro direction and event risk, but never invents data
when the provider is unavailable.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import requests

API_KEY = os.getenv("TRADINGECONOMICS_API_KEY", "").strip()
BASE_URL = "https://api.tradingeconomics.com"
TIMEOUT = 12

ASSET_MAP = {
    "EURUSD":["EUR","USD"],"GBPUSD":["GBP","USD"],"AUDUSD":["AUD","USD"],"NZDUSD":["NZD","USD"],
    "USDCAD":["USD","CAD"],"USDJPY":["USD","JPY"],"EURGBP":["EUR","GBP"],"EURJPY":["EUR","JPY"],
    "GBPJPY":["GBP","JPY"],"AUDJPY":["AUD","JPY"],"GBPAUD":["GBP","AUD"],"EURAUD":["EUR","AUD"],
    "XAUUSD":["XAU","USD"],"GOLD":["XAU","USD"],"XAGUSD":["XAG","USD"],"OIL":["OIL","USD"],
    "US30":["USD"],"NAS100":["USD"],"SPX500":["USD"],"BTCUSD":["BTC","USD"],"ETHUSD":["ETH","USD"]}
COUNTRY_CURRENCY={"USD":"united states","EUR":"euro area","GBP":"united kingdom","JPY":"japan","CAD":"canada","AUD":"australia","NZD":"new zealand","CHF":"switzerland","CNY":"china","XAU":"united states","XAG":"united states","OIL":"united states","BTC":"united states","ETH":"united states"}
EVENT_WEIGHTS={"interest rate":5.0,"central bank":5.0,"inflation":4.5,"cpi":4.5,"core inflation":4.5,"gdp":3.5,"non farm payrolls":4.5,"employment change":4.0,"unemployment":3.5,"retail sales":2.5,"pmi":2.5,"manufacturing":2.0,"services":2.0,"wage":3.0,"jobless claims":2.5,"trade balance":1.5,"oil inventories":2.5}


def _get(path, params=None):
    if not API_KEY:return []
    p=dict(params or {});p["c"]=API_KEY;p["f"]="json"
    r=requests.get(BASE_URL+path,params=p,timeout=TIMEOUT);r.raise_for_status();data=r.json()
    return data if isinstance(data,list) else []

def _event_weight(e):
    text=f"{e.get('Event','')} {e.get('Category','')}".lower()
    for k,w in EVENT_WEIGHTS.items():
        if k in text:return w
    return 1.0

def _importance(e):
    try:return int(e.get("Importance") or 0)
    except Exception:return 0

def _num(v):
    try:
        if v is None or v=="":return None
        return float(str(v).replace(",","").replace("%",""))
    except Exception:return None

def _surprise(e):
    a,f=_num(e.get("Actual")),_num(e.get("Forecast"))
    if a is None or f is None or f==0:return 0.0
    return max(-3.0,min(3.0,(a-f)/max(abs(f),1e-9)*10))

def _economic_direction(e):
    text=f"{e.get('Event','')} {e.get('Category','')}".lower();s=_surprise(e)
    if not s:return 0.0
    if any(k in text for k in ("inflation","cpi","core inflation","wage","gdp","retail sales","pmi","manufacturing","services","employment change")):return 1.0 if s>0 else -1.0
    if "unemployment" in text or "jobless claims" in text:return -1.0 if s>0 else 1.0
    if "interest rate" in text or "central bank" in text:return 1.0 if s>0 else -1.0
    return 0.0

def _currency_fundamentals(currency,country,days=21):
    now=datetime.now(timezone.utc);start=(now-timedelta(days=days)).strftime("%Y-%m-%d");end=(now+timedelta(days=7)).strftime("%Y-%m-%d")
    try:events=_get(f"/calendar/country/{country}/{start}/{end}")
    except Exception as exc:return {"currency":currency,"country":country,"error":str(exc),"score":0.0,"recent":[],"upcoming":[]}
    score=0.0;recent=[];upcoming=[]
    for e in events:
        if _importance(e)<2:continue
        w=_event_weight(e);raw=str(e.get("Date") or "")
        try:
            dt=datetime.fromisoformat(raw.replace("Z","+00:00"));dt=dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:dt=now
        if dt<=now:
            score+=_economic_direction(e)*w;recent.append({"event":e.get("Event"),"actual":e.get("Actual"),"forecast":e.get("Forecast"),"importance":_importance(e)})
        else:upcoming.append({"event":e.get("Event"),"date":raw,"importance":_importance(e),"weight":w})
    return {"currency":currency,"country":country,"score":round(score,2),"recent":recent[-8:],"upcoming":sorted(upcoming,key=lambda x:x.get("date",""))[:8]}

def analyze(symbol):
    symbol=symbol.upper().replace("/","")
    if not API_KEY:return {"symbol":symbol,"available":False,"bias":"NEUTRAL","score":0,"event_risk":"UNKNOWN","reason":"TRADINGECONOMICS_API_KEY is not configured."}
    assets=ASSET_MAP.get(symbol)
    if not assets:return {"symbol":symbol,"available":True,"bias":"NEUTRAL","score":0,"event_risk":"UNKNOWN","reason":"No macro mapping for this symbol."}
    comps=[_currency_fundamentals(c,COUNTRY_CURRENCY[c]) for c in assets if c in COUNTRY_CURRENCY]
    if len(assets)>=2 and assets[0] not in ("XAU","XAG","OIL","BTC","ETH"):score=comps[0]["score"]-comps[1]["score"]
    elif assets[0] in ("XAU","XAG","OIL"):score=-comps[-1]["score"]
    else:score=comps[-1]["score"]
    upcoming=[x for c in comps for x in c.get("upcoming",[])];high=sum(1 for x in upcoming if x.get("importance",0)>=3)
    risk="HIGH" if high>=2 else "MEDIUM" if upcoming else "LOW";bias="BUY" if score>=5 else "SELL" if score<=-5 else "NEUTRAL"
    confidence=min(100,int(50+min(abs(score),30)*1.6));reasons=[]
    for c in comps:
        if abs(c.get("score",0))>=3:reasons.append(f"{c['currency']} macro backdrop {'supportive' if c['score']>0 else 'deteriorating'} ({c['score']:+.1f})")
    if high:reasons.append(f"{high} high-impact event(s) in the next 7 days")
    return {"symbol":symbol,"available":True,"bias":bias,"score":round(score,2),"confidence":confidence,"event_risk":risk,"components":comps,"upcoming":sorted(upcoming,key=lambda x:x.get("date",""))[:10],"reasons":reasons}

def format_report(result):
    if not result.get("available"):return f"🌐 FUNDAMENTAL ANALYSIS — {result.get('symbol','—')}\n\nStatus: DATA PROVIDER NOT CONFIGURED\nSet TRADINGECONOMICS_API_KEY on Render to enable live macro analysis."
    lines=["🌐 FUNDAMENTAL ANALYSIS — "+result["symbol"],"",f"Bias: {result['bias']}",f"Macro Score: {result['score']:+.1f}",f"Confidence: {result.get('confidence',0)}/100",f"Event Risk: {result['event_risk']}"]
    if result.get("reasons"):lines += ["","Drivers:"]+[f"• {r}" for r in result["reasons"][:6]]
    if result.get("upcoming"):
        lines += ["","Upcoming risk:"]
        for e in result["upcoming"][:5]:lines.append(f"• {e.get('date','')} — {e.get('event','')} (importance {e.get('importance',0)})")
    lines += ["","Fundamental data is a context filter, not a standalone trade trigger."]
    return "\n".join(lines)
