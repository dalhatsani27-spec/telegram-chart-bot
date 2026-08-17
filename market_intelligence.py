"""Practical market-intelligence layer built from observable OHLCV behavior.

Converts raw candles into structure, liquidity proxies, auction behavior,
volatility regime, evidence and contradictions. It does not invent hidden
order flow or claim to know what a market maker is doing.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd


def _atr(d, n=14):
    h,l,c=d.High,d.Low,d.Close
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()


def _rsi(c,n=14):
    x=c.diff(); up=x.clip(lower=0); dn=-x.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); ad=dn.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return 100-100/(1+au/ad.replace(0,np.nan))


def _pivots(d,left=3,right=3):
    h,l=d.High.to_numpy(float),d.Low.to_numpy(float); out=[]
    for i in range(left,len(d)-right):
        if h[i]>=np.max(h[i-left:i+right+1]): out.append((i,h[i],"high"))
        if l[i]<=np.min(l[i-left:i+right+1]): out.append((i,l[i],"low"))
    return sorted(out)


def _structure(d):
    p=_pivots(d); hs=[(i,v) for i,v,t in p if t=="high"][-5:]; ls=[(i,v) for i,v,t in p if t=="low"][-5:]
    hh=len(hs)>=2 and hs[-1][1]>hs[-2][1]; lh=len(hs)>=2 and hs[-1][1]<hs[-2][1]
    hl=len(ls)>=2 and ls[-1][1]>ls[-2][1]; ll=len(ls)>=2 and ls[-1][1]<ls[-2][1]
    bias="BUY" if hh and hl else "SELL" if lh and ll else "NEUTRAL"
    return {"bias":bias,"higher_high":hh,"higher_low":hl,"lower_high":lh,"lower_low":ll,
            "last_high":hs[-1][1] if hs else None,"last_low":ls[-1][1] if ls else None}


def _liquidity(d,atr):
    w=d.tail(min(20,len(d))); hi=float(w.High.max()); lo=float(w.Low.min()); close=float(d.Close.iloc[-1]); tol=max(.12*atr,1e-12)
    # These are liquidity proxies, not a claim of seeing the order book.
    eqh=int((w.High>=hi-tol).sum()); eql=int((w.Low<=lo+tol).sum()); prev=d.iloc[-2]
    sweep_high=float(prev.High)>hi-tol and close<hi-tol; sweep_low=float(prev.Low)<lo+tol and close>lo+tol
    return {"range_high":hi,"range_low":lo,"equal_high_count":eqh,"equal_low_count":eql,
            "sweep_high":bool(sweep_high),"sweep_low":bool(sweep_low)}


def _auction(d,atr):
    r=d.iloc[-1]; o,h,l,c=map(float,(r.Open,r.High,r.Low,r.Close)); body=abs(c-o); rng=max(h-l,1e-12)
    upper=h-max(c,o); lower=min(c,o)-l; rejection="NONE"
    if upper>body*1.5 and upper>lower*1.2: rejection="HIGH_REJECTION"
    elif lower>body*1.5 and lower>upper*1.2: rejection="LOW_REJECTION"
    med=float((d.High-d.Low).tail(20).median())
    return {"body_atr":round(body/max(atr,1e-12),3),"range_atr":round(rng/max(atr,1e-12),3),
            "rejection":rejection,"range_expansion":bool(rng>1.35*max(med,1e-12))}


def _regime(d,atr,structure):
    c=d.Close; ret20=float(c.iloc[-1]/c.iloc[-21]-1) if len(c)>=22 else 0.0; hist=_atr(d).dropna().tail(80); med=float(hist.median()) if len(hist) else atr
    if atr<med*.75:return "COMPRESSION"
    if structure["bias"]=="BUY" and ret20>.003:return "BULL_EXPANSION" if atr>med*1.5 else "BULL_TREND"
    if structure["bias"]=="SELL" and ret20<-.003:return "BEAR_EXPANSION" if atr>med*1.5 else "BEAR_TREND"
    if atr>med*1.5:return "VOLATILE_RANGE"
    return "RANGE"


def analyze(df,symbol=""):
    if df is None or len(df)<60:return {"available":False,"reason":"insufficient_data","symbol":symbol}
    d=df.copy()
    for c in ("Open","High","Low","Close"): d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["Open","High","Low","Close"]); atr=float(_atr(d).iloc[-1]); structure=_structure(d); liq=_liquidity(d,atr); auction=_auction(d,atr); regime=_regime(d,atr,structure); rsi=float(_rsi(d.Close).iloc[-1])
    evidence=[]; contradictions=[]
    if structure["bias"]!="NEUTRAL":evidence.append("swing structure supports "+structure["bias"])
    if liq["sweep_low"]:evidence.append("sell-side sweep/reclaim proxy")
    if liq["sweep_high"]:evidence.append("buy-side sweep/rejection proxy")
    if auction["rejection"]!="NONE":evidence.append(auction["rejection"].lower().replace("_"," "))
    if regime.startswith("BULL") and rsi<45:contradictions.append("bull regime but weak momentum")
    if regime.startswith("BEAR") and rsi>55:contradictions.append("bear regime but strong momentum")
    if structure["bias"]=="BUY" and liq["sweep_high"]:contradictions.append("bull structure facing high-side rejection")
    if structure["bias"]=="SELL" and liq["sweep_low"]:contradictions.append("bear structure facing low-side rejection")
    confidence=50+(15 if structure["bias"]!="NEUTRAL" else 0)+(10 if regime in ("BULL_TREND","BEAR_TREND","BULL_EXPANSION","BEAR_EXPANSION") else 0)+(8 if liq["sweep_low"] or liq["sweep_high"] else 0)+(5 if auction["rejection"]!="NONE" else 0)-10*len(contradictions)
    confidence=max(0,min(100,int(confidence)))
    decision="CONTINUE" if confidence>=65 and not contradictions else "WAIT" if confidence>=45 else "AVOID"
    return {"available":True,"symbol":symbol,"regime":regime,"direction":structure["bias"],"confidence":confidence,"atr":atr,"rsi":round(rsi,1),"structure":structure,"liquidity":liq,"auction":auction,"evidence":evidence,"contradictions":contradictions,"decision":decision}


def apply(result):
    if not result or result.get("error") or result.get("df") is None:return result
    mi=analyze(result["df"],result.get("symbol","")); result["market_intelligence"]=mi; result["market_state"]=mi.get("regime"); result["evidence"]=mi.get("evidence",[]); result["contradictions"]=mi.get("contradictions",[])
    result.setdefault("reasons",[])
    result["reasons"] += ["MI: "+x for x in mi.get("evidence",[])] + ["MI conflict: "+x for x in mi.get("contradictions",[])]
    direction=result.get("direction","NEUTRAL"); md=mi.get("direction","NEUTRAL"); score=int(result.get("score",result.get("strength",0)))
    before=score
    if direction in ("BUY","SELL") and md==direction:score+=8
    elif direction in ("BUY","SELL") and md in ("BUY","SELL") and md!=direction:score-=12
    if mi.get("regime") in ("RANGE","VOLATILE_RANGE"):score-=6
    score-=min(20,7*len(mi.get("contradictions",[]))); score=max(0,min(100,score)); result["score"]=score; result["strength"]=score; result["intelligence_adjustment"]=score-before
    if "valid" in result and direction in ("BUY","SELL") and md in ("BUY","SELL"):
        result["valid"]=bool(result["valid"]) and direction==md and not mi.get("contradictions")
    return result
