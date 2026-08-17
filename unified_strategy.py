"""Unified market-intelligence strategy.

Trendline, SMC and OTE are evidence extractors, not user-selectable strategies.
The engine reasons over market state and a causal sequence before deciding.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
import market_data
from market_analysis import find_swings, analyse_structure, detect_order_blocks
from smc_engine import detect_liquidity_pools, detect_fair_value_gaps, select_smc_zone
from topdown_engine import get_topdown_bias

STRATEGY_NAME = "Unified Market Intelligence"
POLICY = "ONE_STRATEGY_TRENDLINE_SMC_OTE_INTELLIGENCE"


def _atr(df, n=14):
    h,l,c=df["High"],df["Low"],df["Close"]
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()


def alligator_state(df):
    """Bill Williams-style Alligator state; no EMA20/50 is used."""
    median=(df["High"]+df["Low"])/2
    jaw=median.rolling(13,min_periods=13).mean().shift(8)
    teeth=median.rolling(8,min_periods=8).mean().shift(5)
    lips=median.rolling(5,min_periods=5).mean().shift(3)
    if len(df)<20 or any(x.isna().iloc[-1] for x in (jaw,teeth,lips)):
        return {"state":"UNKNOWN","direction":"NEUTRAL","jaw":None,"teeth":None,"lips":None}
    j,t,li=[float(x.iloc[-1]) for x in (jaw,teeth,lips)]
    atr=max(float(_atr(df).iloc[-1]),1e-12)
    spread=max(j,t,li)-min(j,t,li)
    prev=max(float(jaw.iloc[-4]),float(teeth.iloc[-4]),float(lips.iloc[-4]))-min(float(jaw.iloc[-4]),float(teeth.iloc[-4]),float(lips.iloc[-4]))
    bullish=li>t>j; bearish=li<t<j; opening=spread>prev*1.08; compressed=spread<atr*.35
    if compressed: state="SLEEPING"
    elif bullish and opening: state="AWAKENING_BULLISH"
    elif bearish and opening: state="AWAKENING_BEARISH"
    elif bullish: state="BULLISH"
    elif bearish: state="BEARISH"
    else: state="TRANSITION"
    return {"state":state,"direction":"BUY" if bullish else "SELL" if bearish else "NEUTRAL","jaw":j,"teeth":t,"lips":li,"spread_atr":round(spread/atr,2),"opening":opening}


def trendline_intelligence(df):
    swings=find_swings(df,left=3,right=3); highs=[s for s in swings if s.get("type")=="high"][-5:]; lows=[s for s in swings if s.get("type")=="low"][-5:]
    out={"direction":"NEUTRAL","quality":0,"event":"NONE","touches":0}
    if len(highs)<2 or len(lows)<2: return out
    hs=np.polyfit([x["index"] for x in highs[-3:]],[x["price"] for x in highs[-3:]],1)[0]; ls=np.polyfit([x["index"] for x in lows[-3:]],[x["price"] for x in lows[-3:]],1)[0]; atr=max(float(_atr(df).iloc[-1]),1e-12)
    if hs>atr*.02 and ls>atr*.02: out["direction"]="BUY"
    elif hs<-atr*.02 and ls<-atr*.02: out["direction"]="SELL"
    close=float(df["Close"].iloc[-1]); out["touches"]=min(len(highs),len(lows))*2
    if out["direction"]=="BUY" and close>highs[-1]["price"]: out["event"]="BREAKOUT_UP"
    elif out["direction"]=="SELL" and close<lows[-1]["price"]: out["event"]="BREAKOUT_DOWN"
    elif out["direction"]=="BUY" and close>=lows[-1]["price"]: out["event"]="SUPPORT_HOLD"
    elif out["direction"]=="SELL" and close<=highs[-1]["price"]: out["event"]="RESISTANCE_HOLD"
    out["quality"]=min(100,30+out["touches"]*8+(20 if out["event"].startswith("BREAKOUT") else 0)); return out


def ote_intelligence(df,direction):
    swings=find_swings(df,left=3,right=3)
    if len(swings)<6: return {"location":"UNKNOWN","retracement":None,"quality":0}
    last=swings[-1]; opp=next((s for s in reversed(swings[:-1]) if s.get("type")!=last.get("type")),None)
    if not opp:return {"location":"UNKNOWN","retracement":None,"quality":0}
    hi=max(float(last["price"]),float(opp["price"])); lo=min(float(last["price"]),float(opp["price"])); leg=hi-lo; close=float(df["Close"].iloc[-1])
    if leg<=0:return {"location":"UNKNOWN","retracement":None,"quality":0}
    retr=(hi-close)/leg if direction=="BUY" else (close-lo)/leg
    loc="DEEP_RETRACEMENT" if .62<=retr<=.79 else "SHALLOW_RETRACEMENT" if .50<=retr<.62 else "OVER_RETRACED" if retr>.79 else "EXTENDED"
    return {"location":loc,"retracement":round(retr,3),"quality":int(max(0,min(100,100-abs(retr-.705)*220)))}


def smc_intelligence(df,direction):
    structure=analyse_structure(df); obs=detect_order_blocks(df); pools=detect_liquidity_pools(df); fvgs=detect_fair_value_gaps(df)
    zone=select_smc_zone(obs,fvgs,direction) if direction in ("BUY","SELL") else {"status":"NONE","confluence":False}
    sweep=None
    if direction=="BUY" and pools.get("sell_side") and pools["sell_side"].get("status")=="SWEPT": sweep=pools["sell_side"]
    if direction=="SELL" and pools.get("buy_side") and pools["buy_side"].get("status")=="SWEPT": sweep=pools["buy_side"]
    return {"structure":structure,"liquidity":pools,"fvg":fvgs,"zone":zone,"sweep":sweep}


def _structure_direction(s):
    if isinstance(s,dict):
        for k in ("direction","bias","trend"):
            v=str(s.get(k,"")).upper()
            if v in ("BUY","SELL"): return v
        txt=str(s).upper()
        if "BULL" in txt:return "BUY"
        if "BEAR" in txt:return "SELL"
    return "NEUTRAL"


def analyze(symbol,timeframe="30min",include_htf=True):
    df=market_data.fetch_candles(symbol,timeframe,count=300)
    if df is None or df.empty or len(df)<80:return {"strategy":STRATEGY_NAME,"symbol":symbol,"error":"insufficient_data","decision":"WAIT"}
    df=df.copy(); df["ATR"]=_atr(df)
    state=alligator_state(df); tl=trendline_intelligence(df); hint=state["direction"] if state["direction"] in ("BUY","SELL") else tl["direction"]
    smc=smc_intelligence(df,hint); ote=ote_intelligence(df,hint); structure=_structure_direction(smc["structure"])
    dirs=[x for x in (state["direction"],tl["direction"],structure) if x in ("BUY","SELL")]
    dominant=max(set(dirs),key=dirs.count) if dirs else "NEUTRAL"; conflict=len(set(dirs))>1; evidence=[]
    if state["direction"]==dominant:evidence.append("Alligator state aligned")
    if tl["direction"]==dominant and tl["quality"]>=50:evidence.append("Trend geometry aligned")
    if structure==dominant:evidence.append("Structure aligned")
    if smc.get("sweep"):evidence.append("Liquidity sweep")
    if smc.get("zone",{}).get("confluence"):evidence.append("OB/FVG location")
    if ote.get("location")=="DEEP_RETRACEMENT":evidence.append("Favorable retracement location")
    htf={}
    if include_htf:
        try: htf=get_topdown_bias(symbol)
        except Exception: htf={}
    hdir=str(htf.get("direction") or htf.get("bias") or "NEUTRAL").upper()
    if hdir not in ("BUY","SELL"):hdir="NEUTRAL"
    if hdir==dominant:evidence.append("Higher-timeframe context aligned")
    if hdir in ("BUY","SELL") and dominant in ("BUY","SELL") and hdir!=dominant:conflict=True
    event_ok=bool(smc.get("sweep") or smc.get("zone",{}).get("confluence") or tl.get("event","").startswith("BREAKOUT"))
    location_ok=ote.get("location") in ("DEEP_RETRACEMENT","SHALLOW_RETRACEMENT")
    ready=dominant in ("BUY","SELL") and not conflict and state["state"] not in ("SLEEPING","TRANSITION","UNKNOWN") and event_ok and location_ok and len(evidence)>=3
    return {"strategy":STRATEGY_NAME,"policy":POLICY,"symbol":symbol,"timeframe":timeframe,"decision":dominant if ready else "WAIT","direction":dominant,"ready":ready,"conflict":conflict,"evidence":evidence,"alligator":state,"trendline_intelligence":tl,"smc_intelligence":smc,"ote_intelligence":ote,"htf":htf,"df":df,"score":min(100,40+len(evidence)*10),"reason":"; ".join(evidence) if evidence else "No coherent market-state sequence"}


def format_report(r):
    if r.get("error"):return f"{STRATEGY_NAME} — {r['symbol']}\n\nWAIT\n{r['error']}"
    a=r["alligator"]; s=r["smc_intelligence"]; o=r["ote_intelligence"]; h=r.get("htf",{})
    lines=["════════════════════════════","🧠 UNIFIED MARKET INTELLIGENCE","════════════════════════════",f"{r['symbol']} | {r['timeframe']}",f"DECISION: {r['decision']}",f"STATE: {a['state']}",f"ALLIGATOR: {a['direction']}",f"STRUCTURE: {_structure_direction(s.get('structure'))}",f"LIQUIDITY: {'SWEPT' if s.get('sweep') else 'NO CONFIRMED SWEEP'}",f"LOCATION: {o.get('location')}",f"HTF: {h.get('direction') or h.get('bias') or 'NEUTRAL'}"]
    if r.get("evidence"):lines += ["","INTELLIGENCE:"]+[f"• {x}" for x in r["evidence"][:7]]
    lines += ["",f"WHY: {r['reason']}","","Trendline / SMC / OTE are internal intelligence sources — not separate strategies."]
    return "\n".join(lines)
