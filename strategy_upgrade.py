"""Modern multi-strategy engine.

Shared regime/volatility/structure layer used by Trendline, OTE and SMC.
The engine is deliberately deterministic: no repainting pivots are used for
entry confirmation, ATR-normalised distances are preferred over fixed
percentages, and every setup receives a quality score plus explicit reasons.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd

import market_data
from market_analysis import analyse_structure, detect_order_blocks, detect_confirmation_candle, find_swings

TF_LABELS = {"1min":"M1","3min":"M3","5min":"M5","15min":"M15","30min":"M30","1h":"H1","4h":"H4"}


def _num(s, default=0.0):
    try: return float(s)
    except Exception: return float(default)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("Open","High","Low","Close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    h,l,c = d["High"],d["Low"],d["Close"]
    prev = c.shift(1)
    tr = pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False, min_periods=5).mean()
    d["ATR_PCT"] = d["ATR"] / d["Close"].replace(0,np.nan) * 100
    d["EMA20"] = c.ewm(span=20,adjust=False).mean()
    d["EMA50"] = c.ewm(span=50,adjust=False).mean()
    d["EMA200"] = c.ewm(span=200,adjust=False).mean()
    d["RSI14"] = _rsi(c,14)
    d["ADX14"] = _adx(d,14)
    d["VOL_MED"] = d.get("Volume",pd.Series(index=d.index,dtype=float)).rolling(20,min_periods=5).median()
    return d.dropna(subset=["Close","High","Low"])


def _rsi(c, n=14):
    delta=c.diff(); up=delta.clip(lower=0); dn=-delta.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); ad=dn.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return 100-(100/(1+(au/ad.replace(0,np.nan))))


def _adx(d,n=14):
    h,l,c=d["High"],d["Low"],d["Close"]
    up=h.diff(); down=-l.diff()
    plus=up.where((up>down)&(up>0),0.0); minus=down.where((down>up)&(down>0),0.0)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    p=100*plus.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atr.replace(0,np.nan)
    m=100*minus.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atr.replace(0,np.nan)
    dx=100*(p-m).abs()/(p+m).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean()


def market_regime(df: pd.DataFrame) -> Dict[str,Any]:
    d=enrich(df); c=float(d.Close.iloc[-1]); atr=max(_num(d.ATR.iloc[-1]),1e-12)
    e20,e50,e200=map(lambda x:_num(d[x].iloc[-1]),("EMA20","EMA50","EMA200"))
    slope20=(_num(d.EMA20.iloc[-1])-_num(d.EMA20.iloc[-6]))/atr if len(d)>=7 else 0
    slope50=(_num(d.EMA50.iloc[-1])-_num(d.EMA50.iloc[-11]))/atr if len(d)>=12 else 0
    adx=_num(d.ADX14.iloc[-1],20); rsi=_num(d.RSI14.iloc[-1],50)
    if abs(e20-e50) < .25*atr and adx < 18: regime="RANGE"
    elif abs(slope20)<.12 and adx < 22: regime="TRANSITION"
    elif e20>e50>e200 and slope20>.08: regime="BULL_TREND"
    elif e20<e50<e200 and slope20<-.08: regime="BEAR_TREND"
    elif slope20>0: regime="BULL_TRANSITION"
    elif slope20<0: regime="BEAR_TRANSITION"
    else: regime="RANGE"
    vol="LOW" if _num(d.ATR_PCT.iloc[-1]) < _num(d.ATR_PCT.rolling(50).median().iloc[-1])*.75 else ("HIGH" if _num(d.ATR_PCT.iloc[-1]) > _num(d.ATR_PCT.rolling(50).median().iloc[-1])*1.5 else "NORMAL")
    return {"regime":regime,"volatility":vol,"adx":round(adx,1),"rsi":round(rsi,1),"atr":atr,"ema20":e20,"ema50":e50,"ema200":e200,"ema20_slope_atr":round(slope20,3),"ema50_slope_atr":round(slope50,3),"close":c}


def _pivots(df,left=4,right=4):
    if len(df)<left+right+5:return []
    h=df.High.to_numpy(float); l=df.Low.to_numpy(float); out=[]
    for i in range(left,len(df)-right):
        if h[i]>=np.max(h[i-left:i+right+1]): out.append({"index":i,"price":float(h[i]),"type":"high"})
        if l[i]<=np.min(l[i-left:i+right+1]): out.append({"index":i,"price":float(l[i]),"type":"low"})
    return sorted(out,key=lambda x:x["index"])


def _line(points):
    if len(points)<2:return None
    slopes=[]
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            dx=points[j]["index"]-points[i]["index"]
            if dx: slopes.append((points[j]["price"]-points[i]["price"])/dx)
    if not slopes:return None
    slope=float(np.median(slopes)); intercept=float(np.median([p["price"]-slope*p["index"] for p in points]))
    x0,x1=points[0]["index"],points[-1]["index"]
    return {"x0":x0,"y0":slope*x0+intercept,"x1":x1,"y1":slope*x1+intercept,"y_end":slope*(len(points)+x1-x0-1)+intercept,"slope":slope,"intercept":intercept}


def trendline_analysis(symbol:str,tf_code:str="30min",topdown:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
    raw=market_data.fetch_candles(symbol,tf_code,count=300)
    if raw is None or raw.empty or len(raw)<80:return {"error":f"Insufficient {TF_LABELS.get(tf_code,tf_code)} data","symbol":symbol,"timeframe":tf_code}
    d=enrich(raw); reg=market_regime(d); piv=_pivots(d)
    highs=[p for p in piv if p["type"]=="high"][-8:]; lows=[p for p in piv if p["type"]=="low"][-8:]
    close=reg["close"]; atr=reg["atr"]
    up=None; down=None
    if len(lows)>=2:
        cand=_line(lows[-5:])
        if cand and cand["slope"]>0: up=cand
    if len(highs)>=2:
        cand=_line(highs[-5:])
        if cand and cand["slope"]<0: down=cand
    lines=[]
    if up: lines.append(("ascending",up,lows[-5:]))
    if down: lines.append(("descending",down,highs[-5:]))
    best=None
    for kind,line,pts in lines:
        touches=0; violations=0
        for p in piv:
            if p["index"]<line["x0"]:continue
            lv=line["slope"]*p["index"]+line["intercept"]
            if abs(p["price"]-lv)<=.55*atr: touches+=1
        for i in range(line["x0"],len(d)):
            lv=line["slope"]*i+line["intercept"]
            if kind=="ascending" and d.Close.iloc[i] < lv-.25*atr: violations+=1
            if kind=="descending" and d.Close.iloc[i] > lv+.25*atr: violations+=1
        dist=abs(close-(line["slope"]*(len(d)-1)+line["intercept"]))/atr
        score=min(100,30+min(touches,6)*8+max(0,20-violations*5)+max(0,25-12*dist))
        if best is None or score>best["score"]:best={"kind":kind,"line":line,"touches":touches,"violations":violations,"distance_atr":round(dist,2),"score":int(score)}
    direction="BUY" if best and best["kind"]=="ascending" else ("SELL" if best else "NEUTRAL")
    # Break/reclaim logic prevents stale rails from producing entries.
    if best:
        lv=best["line"]["slope"]*(len(d)-1)+best["line"]["intercept"]
        if direction=="BUY" and close<lv-.35*atr: direction="SELL" if reg["regime"]=="BEAR_TREND" else "NEUTRAL"
        elif direction=="SELL" and close>lv+.35*atr: direction="BUY" if reg["regime"]=="BULL_TREND" else "NEUTRAL"
    reasons=[]
    if best: reasons += [f"{best['kind']} structural rail: {best['touches']} touches",f"Rail distance: {best['distance_atr']} ATR"]
    reasons += [f"Market regime: {reg['regime']}",f"ADX: {reg['adx']}"]
    if topdown:
        td=topdown.get("direction","NEUTRAL")
        if td==direction: reasons.append("HTF direction aligned")
        elif td in ("BUY","SELL") and direction in ("BUY","SELL"): reasons.append(f"HTF conflict: {td} vs {direction}")
    score=int((best["score"] if best else 0))
    if topdown and topdown.get("direction")==direction: score=min(100,score+10)
    if reg["regime"]=="RANGE" and direction in ("BUY","SELL"): score=max(0,score-12)
    return {"symbol":symbol,"timeframe":tf_code,"timeframe_label":TF_LABELS.get(tf_code,tf_code),"df":d,"direction":direction,"strength":score,"score":score,"valid":direction in ("BUY","SELL") and score>=55,"regime":reg,"market_regime":reg["regime"],"trendline":best["line"] if best else None,"family_kind":best["kind"] if best else "none","uptrends":[best["line"]] if best and best["kind"]=="ascending" else [],"downtrends":[best["line"]] if best and best["kind"]=="descending" else [],"reasons":reasons,"gating_notes":reasons,"topdown":topdown,"active_setup":"TRENDLINE" if best else "NONE","setup_scores":{"TRENDLINE":score,"PATTERN":0,"S/R":0}}


def _impulse(d):
    piv=_pivots(d,3,3)
    if len(piv)<2:return None
    # Choose latest meaningful opposite pivots; reject tiny legs.
    for i in range(len(piv)-1,0,-1):
        a,b=piv[i-1],piv[i]
        leg=abs(b["price"]-a["price"]); atr=_num(d.ATR.iloc[b["index"]],1)
        if leg>=1.2*atr:
            return {"a":a,"b":b,"leg":leg,"atr":atr}
    return None


def ote_analysis(symbol:str,tf_code:str="30min",topdown:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
    raw=market_data.fetch_candles(symbol,tf_code,count=300)
    if raw is None or raw.empty or len(raw)<80:return {"error":f"Insufficient {TF_LABELS.get(tf_code,tf_code)} data","symbol":symbol}
    d=enrich(raw); reg=market_regime(d); imp=_impulse(d)
    if not imp:return {"error":"No clean impulse leg found","symbol":symbol,"timeframe":tf_code}
    a,b=imp["a"],imp["b"]; hi=max(a["price"],b["price"]); lo=min(a["price"],b["price"]); rng=hi-lo; close=reg["close"]
    bull=b["price"]>a["price"]
    z62=hi-rng*.62 if bull else lo+rng*.62; z79=hi-rng*.79 if bull else lo+rng*.79
    zone_hi=max(z62,z79); zone_lo=min(z62,z79)
    in_zone=zone_lo<=close<=zone_hi or abs(close-zone_hi)<=.15*reg["atr"] or abs(close-zone_lo)<=.15*reg["atr"]
    conf,_=detect_confirmation_candle(d,"BUY" if bull else "SELL")
    direction="BUY" if bull else "SELL"
    score=40
    score += 20 if in_zone else 0
    score += 15 if conf else 0
    score += 15 if ((reg["regime"]=="BULL_TREND" and bull) or (reg["regime"]=="BEAR_TREND" and not bull)) else 0
    if reg["regime"]=="RANGE":score-=12
    if topdown and topdown.get("direction") in ("BUY","SELL"):
        score += 10 if topdown["direction"]==direction else -8
    score=max(0,min(100,int(score)))
    entry=close if in_zone and conf else (zone_hi if bull else zone_lo)
    sl=(lo-.35*reg["atr"]) if bull else (hi+.35*reg["atr"])
    risk=abs(entry-sl)
    tp1=(entry+risk*1.5) if bull else (entry-risk*1.5); tp2=(entry+risk*2.5) if bull else (entry-risk*2.5); tp3=(entry+risk*3.5) if bull else (entry-risk*3.5)
    reasons=[f"Clean {'bullish' if bull else 'bearish'} impulse: {round(rng/reg['atr'],1)} ATR",f"OTE zone 62–79%: {zone_lo:.6f}–{zone_hi:.6f}",f"Regime: {reg['regime']}"]
    return {"symbol":symbol,"timeframe":tf_code,"timeframe_label":TF_LABELS.get(tf_code,tf_code),"df":d,"direction":direction,"score":score,"strength":score,"valid":score>=65,"reasons":reasons,"regime":reg,"market_regime":reg["regime"],"impulse":imp,"zone":{"low":zone_lo,"high":zone_hi,"fib62":z62,"fib79":z79},"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3,"ticket":{"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3,"order_type":"LIMIT" if not (in_zone and conf) else "MARKET"},"entry_ready":in_zone and conf,"confirmation":conf,"gating_notes":reasons}


def _fmt(x):
    if x is None:return "—"
    return f"{x:.5f}" if abs(x)<100 else f"{x:.2f}"


def format_trendline_report(a,symbol=None):
    if a.get("error"):return f"📐 TRENDLINE — {symbol or a.get('symbol','?')}\n\n⚠️ {a['error']}"
    return "\n".join([f"📐 TRENDLINE — {a['symbol']} ({a['timeframe_label']})","",f"REGIME: {a['market_regime']}",f"DIRECTION: {a['direction']}",f"CONFIDENCE: {a['strength']}/100",f"SETUP: {a.get('active_setup','NONE')}","",*(f"• {r}" for r in a.get('reasons',[])[:6]),"",f"STATUS: {'ENTRY BIAS VALID' if a.get('valid') else 'WAIT — no high-quality setup'}"])


def format_ote_report(a):
    if a.get("error"):return f"🎯 OTE — {a.get('symbol','?')}\n\n⚠️ {a['error']}"
    z=a.get('zone',{}); return "\n".join([f"🎯 OTE — {a['symbol']} ({a['timeframe_label']})","",f"REGIME: {a['market_regime']}",f"DIRECTION: {a['direction']}",f"CONFIDENCE: {a['score']}/100",f"ZONE: {_fmt(z.get('low'))} – {_fmt(z.get('high'))}",f"ENTRY: {_fmt(a.get('entry'))}",f"SL: {_fmt(a.get('sl'))}",f"TP1: {_fmt(a.get('tp1'))}",f"TP2: {_fmt(a.get('tp2'))}",f"TP3: {_fmt(a.get('tp3'))}","",*(f"• {r}" for r in a.get('reasons',[])[:6]),"",f"STATUS: {'CONFIRMED' if a.get('valid') else 'WAIT'}"])


def build_position_container(a):
    if not a or a.get('direction') not in ('BUY','SELL') or not a.get('valid'):return None
    return {"entry":a.get('entry'),"sl":a.get('sl'),"tp1":a.get('tp1'),"tp2":a.get('tp2'),"tp3":a.get('tp3'),"order_type":a.get('ticket',{}).get('order_type','MARKET'),"tp3_basis":"3.5R"}


def smc_analysis(symbol:str,tf_code:str="30min",topdown:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
    raw=market_data.fetch_candles(symbol,tf_code,count=300)
    if raw is None or raw.empty or len(raw)<80:return {"error":f"Insufficient {TF_LABELS.get(tf_code,tf_code)} data","symbol":symbol}
    d=enrich(raw); reg=market_regime(d); structure=analyse_structure(d,left=3,right=3,lookback=150); obs=detect_order_blocks(d)
    swings=_pivots(d); highs=[p for p in swings if p['type']=='high']; lows=[p for p in swings if p['type']=='low']
    close=reg['close']; atr=reg['atr']
    # Liquidity levels are clustered structural highs/lows, with explicit sweep state.
    def pool(points,side):
        if not points:return None
        p=max(points,key=lambda x:x['index']); level=p['price']; swept=None
        for i in range(p['index']+1,len(d)):
            if side=='buy' and d.High.iloc[i]>level:
                swept=i if d.Close.iloc[i]<level else None; break
            if side=='sell' and d.Low.iloc[i]<level:
                swept=i if d.Close.iloc[i]>level else None; break
        return {"level":level,"index":p['index'],"status":"SWEPT" if swept is not None else "INTACT","swept_index":swept}
    buy=pool(highs[-10:],'buy'); sell=pool(lows[-10:],'sell')
    biasword=structure.get('bias','NEUTRAL'); direction='BUY' if biasword=='BULLISH' else ('SELL' if biasword=='BEARISH' else 'NEUTRAL')
    if direction=='NEUTRAL':
        direction='BUY' if reg['regime']=='BULL_TREND' else ('SELL' if reg['regime']=='BEAR_TREND' else 'NEUTRAL')
    want='bullish' if direction=='BUY' else 'bearish'
    candidates=[o for o in (obs or []) if o.get('type')==want and not o.get('is_inducement')]
    zone=candidates[0] if candidates else None
    # FVG detector is local to avoid coupling to stale legacy thresholds.
    fvgs=[]
    for i in range(2,len(d)):
        if d.Low.iloc[i]>d.High.iloc[i-2] and d.Low.iloc[i]-d.High.iloc[i-2]>=.12*atr:
            fvgs.append({'type':'bullish','bottom':float(d.High.iloc[i-2]),'top':float(d.Low.iloc[i]),'index':i})
        if d.High.iloc[i]<d.Low.iloc[i-2] and d.Low.iloc[i-2]-d.High.iloc[i]>=.12*atr:
            fvgs.append({'type':'bearish','bottom':float(d.High.iloc[i]),'top':float(d.Low.iloc[i-2]),'index':i})
    fvgs=[g for g in fvgs if g['type']==want][-5:]
    fvg=fvgs[-1] if fvgs else None
    confluence=bool(zone and fvg and not (zone['top']<fvg['bottom'] or zone['bottom']>fvg['top']))
    if confluence:zt=max(zone['top'],fvg['top']); zb=min(zone['bottom'],fvg['bottom'])
    elif zone:zt,zb=zone['top'],zone['bottom']
    elif fvg:zt,zb=fvg['top'],fvg['bottom']
    else:zt=zb=None
    in_zone=zt is not None and zb is not None and zb-.15*atr<=close<=zt+.15*atr
    conf,_=detect_confirmation_candle(d,direction) if direction in ('BUY','SELL') else (False,None)
    score=35
    score += 20 if confluence else (10 if zone or fvg else 0)
    score += 15 if conf else 0
    score += 15 if ((direction=='BUY' and sell and sell['status']=='SWEPT') or (direction=='SELL' and buy and buy['status']=='SWEPT')) else 0
    score += 10 if reg['regime'] in ('BULL_TREND','BEAR_TREND') else 0
    if reg['regime']=='RANGE':score-=12
    if topdown and topdown.get('direction') in ('BUY','SELL'):score += 8 if topdown['direction']==direction else -8
    score=max(0,min(100,int(score)))
    entry=close if in_zone else (zt if direction=='BUY' else zb)
    if entry is None:return {"error":"No SMC zone available","symbol":symbol,"timeframe":tf_code}
    sl=(zb-.35*atr) if direction=='BUY' else (zt+.35*atr); risk=abs(entry-sl)
    tp1=(entry+1.5*risk) if direction=='BUY' else (entry-1.5*risk); tp2=(entry+2.5*risk) if direction=='BUY' else (entry-2.5*risk)
    ready=direction in ('BUY','SELL') and in_zone and conf and score>=65
    reasons=[f"Regime: {reg['regime']}",f"OB/FVG confluence: {'YES' if confluence else 'NO'}",f"Liquidity sweep: {'YES' if score>=50 and ((direction=='BUY' and sell and sell['status']=='SWEPT') or (direction=='SELL' and buy and buy['status']=='SWEPT')) else 'NO'}"]
    return {"symbol":symbol,"timeframe":tf_code,"timeframe_label":TF_LABELS.get(tf_code,tf_code),"df":d,"bias":direction,"bias_word":biasword,"direction":direction,"score":score,"strength":score,"valid":ready,"entry_ready":ready,"entry":entry if ready else None,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":None,"structure":structure,"liquidity":{"buy_side":buy,"sell_side":sell,"pools":[p for p in (buy,sell) if p]},"trap_pool":sell if direction=='BUY' else buy,"target_pool":buy if direction=='BUY' else sell,"order_blocks":obs,"fvgs":fvgs,"zone":{"ob":zone,"fvg":fvg,"zone_top":zt,"zone_bottom":zb,"status":"CONFLUENCE" if confluence else "SINGLE_ZONE","confluence":confluence},"price_in_zone":in_zone,"candle_confirmed":conf,"candle_name":None,"status":"CONFIRMED" if ready else "WAIT","regime":reg,"market_regime":reg['regime'],"reasons":reasons,"topdown":topdown}


def format_smc_report(a):
    if a.get('error'):return f"🧠 SMC — {a.get('symbol','?')}\n\n⚠️ {a['error']}"
    z=a.get('zone',{}); li=a.get('liquidity',{}); return "\n".join([f"🧠 SMC — {a['symbol']} ({a['timeframe_label']})","",f"REGIME: {a['market_regime']}",f"BIAS: {a['direction']}",f"CONFIDENCE: {a['score']}/100",f"OB/FVG: {'CONFLUENCE' if z.get('confluence') else 'SINGLE ZONE'}",f"ZONE: {_fmt(z.get('zone_bottom'))} – {_fmt(z.get('zone_top'))}",f"BUY-SIDE LIQUIDITY: {li.get('buy_side',{}).get('status','—')}",f"SELL-SIDE LIQUIDITY: {li.get('sell_side',{}).get('status','—')}",f"ENTRY: {_fmt(a.get('entry'))}",f"SL: {_fmt(a.get('sl'))}",f"TP1: {_fmt(a.get('tp1'))}",f"TP2: {_fmt(a.get('tp2'))}","",*(f"• {r}" for r in a.get('reasons',[])[:6]),"",f"STATUS: {'CONFIRMED' if a.get('entry_ready') else 'WAIT'}"])
