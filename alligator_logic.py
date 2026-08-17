"""Williams Alligator Trifecta regime/filter layer."""
from __future__ import annotations
from typing import Any, Dict
import pandas as pd

def _rma(s: pd.Series,n:int)->pd.Series:return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def calculate_alligator(df):
 d=df.copy(); m=(pd.to_numeric(d.High,errors="coerce")+pd.to_numeric(d.Low,errors="coerce"))/2
 d["ALLIGATOR_JAW"]=_rma(m,13).shift(8); d["ALLIGATOR_TEETH"]=_rma(m,8).shift(5); d["ALLIGATOR_LIPS"]=_rma(m,5).shift(3); return d

def alligator_regime(df)->Dict[str,Any]:
 d=calculate_alligator(df)
 if len(d)<30:return {"regime":"RANGE","direction":"NEUTRAL","trifecta":0}
 r=d.iloc[-1]; close=float(r.Close); jaw=float(r.ALLIGATOR_JAW); teeth=float(r.ALLIGATOR_TEETH); lips=float(r.ALLIGATOR_LIPS)
 h=pd.to_numeric(d.High,errors="coerce"); l=pd.to_numeric(d.Low,errors="coerce"); c=pd.to_numeric(d.Close,errors="coerce"); tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); a=max(float(atr.iloc[-1]),1e-12)
 ba=lips>teeth>jaw; be=lips<teeth<jaw; bp=close>lips+.1*a; sp=close<lips-.1*a; spread=(abs(lips-teeth)+abs(teeth-jaw))/a; p=d.iloc[-4]; ps=(abs(float(p.ALLIGATOR_LIPS)-float(p.ALLIGATOR_TEETH))+abs(float(p.ALLIGATOR_TEETH)-float(p.ALLIGATOR_JAW)))/max(float(atr.iloc[-4]),1e-12); ex=spread>ps*1.03; ls=(lips-float(d.ALLIGATOR_LIPS.iloc[-6]))/a; js=(jaw-float(d.ALLIGATOR_JAW.iloc[-6]))/a; bx=ex and ls>.05 and js>=0; sx=ex and ls<-.05 and js<=0; bs=int(ba)+int(bp)+int(bx); ss=int(be)+int(sp)+int(sx)
 if bs==3:reg,di="BULL_TREND","BUY"
 elif ss==3:reg,di="BEAR_TREND","SELL"
 elif bs>=2 and bs>ss:reg,di="BULL_TRANSITION","BUY"
 elif ss>=2 and ss>bs:reg,di="BEAR_TRANSITION","SELL"
 else:reg,di="RANGE","NEUTRAL"
 return {"regime":reg,"direction":di,"trifecta":max(bs,ss),"bull_trifecta":bs,"bear_trifecta":ss,"jaw":jaw,"teeth":teeth,"lips":lips,"jaw_slope_atr":round(js,3),"lips_slope_atr":round(ls,3),"spread_atr":round(spread,3),"expanding":bool(ex),"atr":a,"close":close}

def apply_alligator(result):
 if not result or result.get("error") or result.get("df") is None:return result
 ag=alligator_regime(result["df"]); result["alligator"]=ag; result["market_regime"]=ag["regime"]; result["regime"]=dict(result.get("regime") or {}); result["regime"].update(ag); di=result.get("direction","NEUTRAL"); score=int(result.get("score",result.get("strength",0))); old=score
 if di in ("BUY","SELL"):
  if ag["direction"]==di and ag["trifecta"]==3:score=min(100,score+10);result.setdefault("reasons",[]).append(f"Alligator Trifecta aligned: {di} (3/3)")
  elif ag["direction"]==di and ag["trifecta"]>=2:score=min(100,score+4);result.setdefault("reasons",[]).append(f"Alligator transition aligned: {di} ({ag['trifecta']}/3)")
  elif ag["direction"] in ("BUY","SELL") and ag["direction"]!=di:score=max(0,score-12);result.setdefault("reasons",[]).append(f"Alligator conflict: setup {di} vs regime {ag['direction']}")
  elif ag["regime"]=="RANGE":score=max(0,score-8);result.setdefault("reasons",[]).append("Alligator mouth closed/ranging")
 result["score"]=score;result["strength"]=score;result["alligator_adjustment"]=score-old;result["gating_notes"]=result.get("reasons",[])
 if "valid" in result:
  opp=(di=="BUY" and ag["direction"]=="SELL") or (di=="SELL" and ag["direction"]=="BUY");result["valid"]=bool(result.get("valid")) and not opp and score>=55
 return result
