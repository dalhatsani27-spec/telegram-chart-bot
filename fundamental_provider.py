"""Live economic-calendar adapter for the unified fundamental engine.

Provider: Trading Economics Economic Calendar API.
Set TRADING_ECONOMICS_API_KEY in Render environment variables. The adapter is
fail-soft: if the key/provider is unavailable, technical intelligence continues
without fabricated fundamentals.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import requests
from fundamental_intelligence import ingest_event

BASE_URL="https://api.tradingeconomics.com/calendar/country/{countries}/{start}/{end}"
COUNTRIES={"USD":"united states","EUR":"euro area","GBP":"united kingdom","JPY":"japan","AUD":"australia","CAD":"canada","NZD":"new zealand","CHF":"switzerland","CNY":"china"}

def _importance(v):
    try:
        n=int(v)
        return "HIGH" if n>=3 else ("MEDIUM" if n==2 else "LOW")
    except (TypeError,ValueError): return "MEDIUM"

def _number(v):
    if v is None or v=="": return None
    s=str(v).strip().replace(",","").replace("%","").replace("K","").replace("M","").replace("B","")
    try:return float(s)
    except ValueError:return None

def fetch_calendar(currencies:List[str], days_ahead:int=3, days_back:int=2, timeout:int=8)->Dict:
    key=os.getenv("TRADING_ECONOMICS_API_KEY","").strip()
    if not key:return {"available":False,"reason":"TRADING_ECONOMICS_API_KEY not configured","events":[]}
    names=[COUNTRIES[c] for c in currencies if c in COUNTRIES]
    if not names:return {"available":False,"reason":"unsupported currency set","events":[]}
    now=datetime.now(timezone.utc); start=(now-timedelta(days=days_back)).strftime("%Y-%m-%d"); end=(now+timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    try:
        r=requests.get(BASE_URL.format(countries=','.join(names),start=start,end=end),params={"c":key,"importance":1,"f":"json"},timeout=timeout); r.raise_for_status(); raw=r.json()
    except Exception as exc:
        return {"available":False,"reason":f"provider_error:{type(exc).__name__}","events":[]}
    events=[]
    for x in raw if isinstance(raw,list) else []:
        currency=next((c for c in currencies if COUNTRIES.get(c)==x.get("Country")),"")
        if not currency:continue
        events.append(ingest_event(currency,x.get("Event") or x.get("Category",""),_number(x.get("Actual")),_number(x.get("Forecast") or x.get("TEForecast")),_number(x.get("Previous")),_importance(x.get("Importance")),source="Trading Economics"))
    return {"available":True,"events":events,"fetched_at":now.isoformat(),"count":len(events)}

def fetch_for_symbol(symbol:str, days_ahead:int=3, days_back:int=2)->Dict:
    s=symbol.upper().replace('/','').replace('-',''); pairs={"EURUSD":('EUR','USD'),"GBPUSD":('GBP','USD'),"USDJPY":('USD','JPY'),"AUDUSD":('AUD','USD'),"USDCAD":('USD','CAD'),"NZDUSD":('NZD','USD'),"EURGBP":('EUR','GBP'),"GBPJPY":('GBP','JPY'),"EURJPY":('EUR','JPY'),"AUDJPY":('AUD','JPY'),"EURAUD":('EUR','AUD')}
    cc=pairs.get(s,("USD",)); return fetch_calendar(list(cc),days_ahead,days_back)
