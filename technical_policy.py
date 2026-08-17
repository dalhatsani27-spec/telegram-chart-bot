"""Single technical-indicator policy for the bot.

Decision moving averages are intentionally limited to 200 EMA. The Williams
Alligator is the active market-state engine. ATR is only a volatility unit.
No EMA20/EMA50 or other moving-average trend filters are calculated at runtime.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from alligator_logic import calculate_alligator, alligator_regime


def _rsi(c, n=14):
    delta=c.diff(); up=delta.clip(lower=0); dn=-delta.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); ad=dn.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return 100-(100/(1+(au/ad.replace(0,np.nan))))


def _adx(d,n=14):
    h,l,c=d["High"],d["Low"],d["Close"]
    up=h.diff(); down=-l.diff(); plus=up.where((up>down)&(up>0),0.0); minus=down.where((down>up)&(down>0),0.0)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    p=100*plus.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atr.replace(0,np.nan); m=100*minus.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atr.replace(0,np.nan)
    dx=100*(p-m).abs()/(p+m).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Runtime replacement for legacy enrichment: no EMA20/EMA50."""
    d=df.copy()
    for c in ("Open","High","Low","Close"): d[c]=pd.to_numeric(d[c],errors="coerce")
    h,l,c=d["High"],d["Low"],d["Close"]; prev=c.shift(1)
    tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    d["ATR"]=tr.ewm(alpha=1/14,adjust=False,min_periods=5).mean()
    d["ATR_PCT"]=d["ATR"]/d["Close"].replace(0,np.nan)*100
    d["EMA200"]=c.ewm(span=200,adjust=False,min_periods=200).mean()
    d["RSI14"]=_rsi(c,14); d["ADX14"]=_adx(d,14)
    if "Volume" in d.columns: d["VOL_MED"]=pd.to_numeric(d["Volume"],errors="coerce").rolling(20,min_periods=5).median()
    d=calculate_alligator(d)
    return d.dropna(subset=["Close","High","Low"])


def market_regime(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or len(df) < 30:
        return {"regime":"RANGE","volatility":"UNKNOWN","direction":"NEUTRAL","atr":0.0,"close":None,"ema200":None,"indicator_policy":"200EMA+ALLIGATOR_ONLY"}
    d=df.copy()
    for c in ("High","Low","Close"): d[c]=pd.to_numeric(d[c],errors="coerce")
    h,l,c=d["High"],d["Low"],d["Close"]
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr_s=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); atr=max(float(atr_s.iloc[-1]),1e-12)
    ag=alligator_regime(d); ema200=ag.get("ema200"); close=float(c.iloc[-1])
    direction=ag.get("direction","NEUTRAL")
    if direction=="BUY" and ema200 is not None and close<=ema200+0.05*atr: direction="NEUTRAL"
    if direction=="SELL" and ema200 is not None and close>=ema200-0.05*atr: direction="NEUTRAL"
    regime=ag.get("regime","RANGE")
    if direction=="NEUTRAL" and regime not in ("RANGE",): regime="TRANSITION"
    hist=atr_s.dropna().tail(80); med=float(hist.median()) if len(hist) else atr
    volatility="LOW" if atr<med*.75 else ("HIGH" if atr>med*1.5 else "NORMAL")
    return {"regime":regime,"volatility":volatility,"direction":direction,"atr":atr,"close":close,"ema200":ema200,"alligator":ag,"indicator_policy":"200EMA+ALLIGATOR_ONLY"}
