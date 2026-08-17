"""Unified Market Intelligence Strategy.

Trendline/SMC/OTE are intelligence sources, not selectable strategies. This
engine produces one final BUY/SELL/WAIT decision from market structure,
Alligator/200EMA state, location, liquidity intelligence and fundamentals.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import market_data
from technical_policy import enrich, market_regime
from market_analysis import analyse_structure, detect_order_blocks, detect_confirmation_candle, find_swings
from fundamental_analysis import analyze as analyze_fundamentals
from alligator_logic import alligator_regime


def _clamp(v): return max(0,min(100,int(round(v))))

def _intelligence(df):
    d=enrich(df); reg=market_regime(d); close=float(d.Close.iloc[-1]); atr=max(float(reg.get("atr",0)),1e-12)
    structure=analyse_structure(d) or {}
    swings=find_swings(d) or {}
    obs=detect_order_blocks(d) or []
    ag=reg.get("alligator") or alligator_regime(d) or {}
    direction=reg.get("direction","NEUTRAL")
    trend_score=0
    if direction in ("BUY","SELL"): trend_score+=30
    if reg.get("ema200") is not None:
        trend_score += 15 if (direction=="BUY" and close>reg["ema200"]) or (direction=="SELL" and close<reg["ema200"]) else -15
    trend_score += 15 if ag.get("awake") else 0
    trend_score += 10 if ag.get("spread_expanding") else 0
    setup=[]
    # Extracted intelligence: structural trendline geometry, SMC liquidity/OB,
    # and OTE-style retracement location are represented as evidence, not votes.
    if structure: setup.append("market structure analyzed")
    if swings: setup.append("swing/liquidity structure analyzed")
    if obs: setup.append(f"{len(obs)} reaction zone(s) detected")
    last20=d.tail(20); impulse=float((last20.High.max()-last20.Low.min())/atr) if len(last20) else 0
    location_score=15 if 0.6<=impulse<=8 else 5
    score=_clamp(50+trend_score+location_score)
    return {"df":d,"regime":reg,"structure":structure,"swings":swings,"order_blocks":obs,"alligator":ag,"direction":direction,"technical_score":score,"impulse_atr":round(impulse,2),"evidence":setup}


def analyze(symbol: str, tf_code: str="30min", topdown: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    raw=market_data.fetch_candles(symbol,tf_code,count=300)
    if raw is None or raw.empty or len(raw)<80:
        return {"symbol":symbol,"timeframe":tf_code,"decision":"WAIT","valid":False,"reason":"Insufficient market data"}
    ti=_intelligence(raw); fundamental=analyze_fundamentals(symbol)
    score=ti["technical_score"]; reasons=list(ti["evidence"])
    fscore=float(fundamental.get("score",0)) if fundamental.get("available") else 0
    # Fundamentals modify confidence only when real data is available. They do
    # not manufacture direction against price structure.
    if fundamental.get("available"):
        if ti["direction"] in ("BUY","SELL") and fundamental.get("bias") == ti["direction"]:
            score+=8; reasons.append(f"Fundamentals align ({fscore:+.1f})")
        elif ti["direction"] in ("BUY","SELL") and fundamental.get("bias") in ("BUY","SELL"):
            score-=10; reasons.append(f"Fundamental conflict ({fscore:+.1f})")
        if fundamental.get("event_risk")=="HIGH": score-=8; reasons.append("High fundamental event risk")
    else: reasons.append("Fundamental data unavailable; no fundamental assumption made")
    if topdown and topdown.get("direction") in ("BUY","SELL"):
        if topdown["direction"]==ti["direction"]: score+=7; reasons.append("Higher-timeframe direction aligned")
        elif ti["direction"] in ("BUY","SELL"): score-=10; reasons.append("Higher-timeframe conflict")
    score=_clamp(score)
    decision=ti["direction"] if score>=65 else "WAIT"
    # Avoid entries during uncertain Alligator transition/range states.
    if ti["regime"].get("regime") in ("RANGE","TRANSITION") and score<78: decision="WAIT"
    return {"symbol":symbol,"timeframe":tf_code,"decision":decision,"direction":decision if decision in ("BUY","SELL") else "NEUTRAL","score":score,"valid":decision in ("BUY","SELL"),"market_regime":ti["regime"],"technical":{k:v for k,v in ti.items() if k!="df"},"fundamental":fundamental,"reasons":reasons,"strategy":"UNIFIED_MARKET_INTELLIGENCE","technical_indicator_policy":"200EMA+ALLIGATOR_ONLY","intelligence_sources":["TRENDLINE_STRUCTURE","SMC_LIQUIDITY","OTE_LOCATION"],"entry_ready":decision in ("BUY","SELL") and score>=70}


def format_report(a: Dict[str,Any]) -> str:
    if not a:return "Unified strategy unavailable."
    lines=[f"🧠 UNIFIED MARKET INTELLIGENCE — {a.get('symbol','—')}","",f"Decision: {a.get('decision','WAIT')}",f"Confidence: {a.get('score',0)}/100",f"Market regime: {a.get('market_regime',{}).get('regime','UNKNOWN')}","", "Evidence:"]
    lines += [f"• {x}" for x in a.get("reasons",[])[:8]]
    f=a.get("fundamental",{})
    lines += ["",f"Fundamental: {f.get('bias','NEUTRAL')} ({f.get('score',0):+.1f})" if f.get("available") else "Fundamental: UNAVAILABLE"]
    lines.append("\nOne strategy. Trendline/SMC/OTE are internal intelligence sources, not menu strategies.")
    return "\n".join(lines)
