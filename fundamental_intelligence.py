"""Provider-neutral fundamental intelligence for the unified strategy.

Fundamentals are context, never an automatic trade signal. Providers may feed
normalized events containing actual/forecast/previous values and policy stance.
The module keeps only compact state; it does not retain raw news or candles.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

HIGH_IMPACT={"CPI","CORE_CPI","PCE","CORE_PCE","NFP","PAYROLLS","UNEMPLOYMENT","GDP","PMI","FOMC","ECB","BOE","BOJ","RBA","BOC","SNB","RBNZ","RATE_DECISION","RATE_SPEECH"}

@dataclass
class FundamentalEvent:
    timestamp:str; country:str; event:str; actual:Optional[float]=None; forecast:Optional[float]=None; previous:Optional[float]=None; importance:str="MEDIUM"; hawkish:Optional[float]=None; source:str=""
    def surprise(self)->float:
        if self.actual is None or self.forecast is None:return 0.0
        scale=abs(self.forecast) if abs(self.forecast)>1e-9 else 1.0
        return max(-3.0,min(3.0,(self.actual-self.forecast)/scale))

def ingest_event(country,event,actual=None,forecast=None,previous=None,importance="MEDIUM",hawkish=None,source=""):
    e=FundamentalEvent(datetime.now(timezone.utc).isoformat(),str(country).upper(),str(event).upper(),actual,forecast,previous,str(importance).upper(),hawkish,source)
    return asdict(e)|{"surprise":e.surprise()}

def _bias(events,country):
    score=weight=0.0
    for e in events:
        if str(e.get("country","")).upper()!=str(country).upper():continue
        w=2.0 if str(e.get("importance","MEDIUM")).upper()=="HIGH" else 1.0
        v=e.get("hawkish")
        if v is None:v=e.get("surprise",0.0)
        try:v=max(-1.0,min(1.0,float(v)))
        except (TypeError,ValueError):v=0.0
        score+=v*w;weight+=w
    return score/weight if weight else 0.0

def _label(x):return "BULLISH" if x>=.25 else ("BEARISH" if x<=-.25 else "NEUTRAL")

def build_state(events:Iterable[Dict],base_currency="USD",quote_currency=""):
    events=list(events); b=_bias(events,base_currency); q=_bias(events,quote_currency) if quote_currency else 0.0; differential=b-q
    high=[e for e in events if str(e.get("importance","MEDIUM")).upper()=="HIGH" or str(e.get("event","")).upper() in HIGH_IMPACT]
    return {"available":bool(events),"base_currency":base_currency,"quote_currency":quote_currency,"base_bias":_label(b),"quote_bias":_label(q) if quote_currency else "NEUTRAL","score":round(max(-100,min(100,differential*100)),1),"bias":_label(differential),"event_risk":"HIGH" if high else "NORMAL","high_impact_count":len(high),"event_count":len(events),"method":"actual-vs-forecast surprise + previous context + policy stance + currency differential"}

def analyze_symbol(symbol,events):
    s=str(symbol).upper().replace("/","").replace("-","")
    pairs={"EURUSD":("EUR","USD"),"GBPUSD":("GBP","USD"),"USDJPY":("USD","JPY"),"AUDUSD":("AUD","USD"),"USDCAD":("USD","CAD"),"NZDUSD":("NZD","USD"),"EURGBP":("EUR","GBP"),"GBPJPY":("GBP","JPY"),"EURJPY":("EUR","JPY"),"AUDJPY":("AUD","JPY"),"EURAUD":("EUR","AUD")}
    return build_state(events,*pairs[s]) if s in pairs else build_state(events,"USD","")

def format_state(state):
    if not state.get("available"):return "Fundamental data: UNAVAILABLE"
    return f"Fundamentals: {state.get('bias','NEUTRAL')} | score {state.get('score',0):+.1f} | event risk {state.get('event_risk','NORMAL')} | {state.get('event_count',0)} events"
