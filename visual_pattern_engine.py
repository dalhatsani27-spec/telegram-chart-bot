"""Geometric Trendline Pattern Engine.
Patterns are selected from recent price geometry, never from HH/HL/LH/LL labels.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

PATTERN_RULES = {
    "Uptrend Line": "one rising support rail with repeated reactions",
    "Downtrend Line": "one falling resistance rail with repeated reactions",
    "Horizontal Support & Resistance": "flat horizontal boundary",
    "Ascending Triangle": "flat resistance + rising support",
    "Descending Triangle": "falling resistance + flat support",
    "Symmetrical Triangle": "falling resistance + rising support",
    "Rising Wedge": "two rising converging rails",
    "Falling Wedge": "two falling converging rails",
    "Ascending Channel": "two rising parallel rails",
    "Descending Channel": "two falling parallel rails",
    "Horizontal Channel": "two horizontal parallel rails",
    "Bull Flag": "rising impulse + compact falling parallel correction",
    "Bear Flag": "falling impulse + compact rising parallel correction",
    "Pennant": "strong impulse + converging correction",
}

def _ensure_atr(df):
    df=df.copy()
    if "ATR" not in df or df["ATR"].isna().all():
        tr=pd.concat([df["High"]-df["Low"],(df["High"]-df["Close"].shift(1)).abs(),(df["Low"]-df["Close"].shift(1)).abs()],axis=1).max(axis=1)
        df["ATR"]=tr.rolling(14,min_periods=1).mean()
    return df

def _pivots(df,left=3,right=3):
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float); out=[]
    for i in range(left,len(df)-right):
        if h[i]>=np.max(h[i-left:i+right+1]): out.append({"i":i,"p":float(h[i]),"t":"H"})
        if l[i]<=np.min(l[i-left:i+right+1]): out.append({"i":i,"p":float(l[i]),"t":"L"})
    out.sort(key=lambda x:x["i"]); return out

def _dedupe(points,min_gap=4):
    out=[]
    for p in points:
        if not out: out.append(p); continue
        q=out[-1]
        if p["t"]==q["t"] and p["i"]-q["i"]<min_gap:
            if (p["p"]>q["p"]) if p["t"]=="H" else (p["p"]<q["p"]): out[-1]=p
        else: out.append(p)
    return out

def _fit(points):
    if len(points)<2:return None
    x=np.array([p["i"] for p in points],float); y=np.array([p["p"] for p in points],float)
    m,b=np.polyfit(x,y,1); err=float(np.mean(np.abs(y-(m*x+b))))
    return {"m":float(m),"b":float(b),"x0":int(x.min()),"x1":int(x.max()),"err":err,"n":len(points),"pts":points}

def _line(line,x): return line["m"]*x+line["b"]

def _candidate_lines(points,kind,recent_start):
    pts=[p for p in points if p["t"]==kind and p["i"]>=recent_start]
    if len(pts)<2:return []
    c=[]
    for n in range(2,min(5,len(pts))+1):
        ln=_fit(pts[-n:])
        if ln:c.append(ln)
    c.sort(key=lambda z:(z["n"],-z["err"]),reverse=True); return c

def _pair(df,points,recent_start):
    lows=_candidate_lines(points,"L",recent_start); highs=_candidate_lines(points,"H",recent_start)
    if not lows or not highs:return None
    scale=max(float(np.nanmedian(df["ATR"].to_numpy(float)[-30:])),1e-9); best=None
    for lo in lows:
        for hi in highs:
            start=max(lo["x0"],hi["x0"]); end=min(lo["x1"],hi["x1"])
            if end-start<8 or end<len(df)-8: continue
            gap0=_line(hi,start)-_line(lo,start); gap1=_line(hi,end)-_line(lo,end)
            if gap0<=0 or gap1<=0: continue
            convergence=1-gap1/max(gap0,1e-9)
            slope_diff=abs(lo["m"]-hi["m"])/max(abs(lo["m"]),abs(hi["m"]),scale*.01)
            parallel=max(0.0,1.0-min(slope_diff,1.0))
            fit=max(0.0,1-(lo["err"]+hi["err"])/(max((gap0+gap1)/2,scale)*.8+scale))
            recency=max(0.0,min(1.0,(end-recent_start)/max(len(df)-recent_start,1)))
            score=.38*fit+.27*parallel+.20*max(0,convergence)+.15*recency
            if best is None or score>best[0]: best=(score,lo,hi,convergence,parallel)
    return best

def _classify(df,lo,hi,conv,parallel):
    atr=max(float(np.nanmedian(df["ATR"].to_numpy(float)[-30:])),1e-9); ls=lo["m"]/atr; hs=hi["m"]/atr; eps=.055
    if ls < -eps and hs < -eps:
        if conv>=.12 and abs(abs(ls)-abs(hs))/max(abs(ls),abs(hs),1e-9)>.16:return "Falling Wedge"
        if abs(ls-hs)/max(abs(ls),abs(hs),1e-9)<=.18:return "Descending Channel"
        return "Falling Wedge"
    if ls > eps and hs > eps:
        if conv>=.12 and abs(abs(ls)-abs(hs))/max(abs(ls),abs(hs),1e-9)>.16:return "Rising Wedge"
        if abs(ls-hs)/max(abs(ls),abs(hs),1e-9)<=.18:return "Ascending Channel"
        return "Rising Wedge"
    if hs < -eps and ls > eps:return "Symmetrical Triangle"
    if abs(hs)<=eps and ls>eps:return "Ascending Triangle"
    if abs(ls)<=eps and hs<-eps:return "Descending Triangle"
    if abs(ls)<=eps and abs(hs)<=eps:return "Horizontal Channel"
    return "Downtrend Line" if hs<0 else "Uptrend Line"

def _single(df,points,recent_start):
    c=[]
    for kind,name in [("L","Uptrend Line"),("H","Downtrend Line")]:
        pts=[p for p in points if p["t"]==kind and p["i"]>=recent_start]
        if len(pts)>=2:
            ln=_fit(pts[-5:]); atr=max(float(np.nanmedian(df["ATR"].to_numpy(float)[-30:])),1e-9)
            if ln and abs(ln["m"])/atr>=.045:c.append((ln["n"]*2-max(0,ln["err"]/atr),name,ln))
    if not c:return None
    c.sort(reverse=True,key=lambda z:z[0]); s,n,ln=c[0]
    return {"name":n,"confidence":min(88,int(58+s*4)),"lines":[ln],"pivots":points,"convergence":0.0,"touches":[ln["n"],0]}

def detect_visual_pattern(df: pd.DataFrame) -> Dict[str,Any]:
    if df is None or len(df)<35:return {"name":"None","confidence":0,"lines":[],"pivots":[]}
    df=_ensure_atr(df); work=df.tail(min(110,len(df))).copy().reset_index(drop=True); n=len(work); recent_start=max(0,n-75)
    piv=_dedupe(_pivots(work),4); pair=_pair(work,piv,recent_start); candidates=[]
    if pair:
        score,lo,hi,conv,parallel=pair; name=_classify(work,lo,hi,conv,parallel); touches=[lo["n"],hi["n"]]
        confidence=min(97,max(55,int(58+score*34+sum(touches)*2)))
        fit=max(0,1-(lo["err"]+hi["err"])/(max(float(np.nanmedian(work["ATR"].to_numpy(float)[-30:])),1e-9)*2))*100
        candidates.append({"name":name,"confidence":confidence,"lines":[lo,hi],"pivots":piv,"convergence":round(max(0,conv)*100,1),"parallelism":round(parallel*100,1),"touches":touches,"fit":round(fit,1)})
    one=_single(work,piv,recent_start)
    if one:candidates.append(one)
    if not candidates:return {"name":"None","confidence":0,"lines":[],"pivots":piv}
    candidates.sort(key=lambda z:(len(z["lines"])==2,z["confidence"]),reverse=True); result=candidates[0]; base=len(df)-n
    lines=[]
    for ln in result["lines"]:
        x=dict(ln); x["x0"]+=base; x["x1"]+=base; x["b"]=ln["b"]-ln["m"]*base; x["pts"]=[{**p,"i":p["i"]+base} for p in ln.get("pts",[])]; lines.append(x)
    result["lines"]=lines; result["pivots"]=[{**p,"i":p["i"]+base} for p in piv]; result["window_start"]=base; result["window_end"]=len(df)-1; result["rule"]=PATTERN_RULES.get(result["name"],"geometric price structure"); return result

def _draw(ax,ln,start,end,color="#ff334f",lw=2.6):
    if end<=start:return
    xs=np.arange(start,end+1); ax.plot(xs,[_line(ln,x) for x in xs],color=color,lw=lw,zorder=8)

def render_trendline_map(df,symbol,setup,title_suffix=""):
    family=setup.get("family") or setup.get("analysis") or setup; data=family.get("df") if isinstance(family,dict) else None
    if data is not None:df=data
    df=_ensure_atr(df); pattern=family.get("visual_pattern") or detect_visual_pattern(df); chart_len=min(110,len(df)); start=len(df)-chart_len; chart=df.tail(chart_len).copy()
    name=pattern.get("name","None"); conf=int(pattern.get("confidence",0) or 0); side=str(family.get("direction") or "NEUTRAL").upper()
    mc=mpf.make_marketcolors(up="#00a889",down="#e53e4f",edge="inherit",wick="inherit"); style=mpf.make_mpf_style(marketcolors=mc,gridstyle=":",gridcolor="#202833",y_on_right=True,facecolor="#0b0f14",figcolor="#0b0f14")
    fig=plt.figure(figsize=(15.5,8.2),facecolor="#0b0f14"); gs=fig.add_gridspec(1,2,width_ratios=[4.2,1.25],wspace=.025); ax=fig.add_subplot(gs[0,0]); panel=fig.add_subplot(gs[0,1])
    mpf.plot(chart,type="candle",style=style,volume=False,ax=ax); ax.set_title(f"{symbol} | M30 | {side} BIAS | {name}",loc="left",color="white",fontsize=13,fontweight="bold",pad=12)
    for ln in pattern.get("lines",[]):
        s=max(start,int(ln["x0"])); e=min(len(df)-1,int(ln["x1"]));
        if e>s:_draw(ax,ln,s-start,e-start)
        if e>s and e>=len(df)-12:_draw(ax,ln,s-start,min(chart_len-1,e-start+12))
    ax.set_xlim(-1,chart_len); panel.set_facecolor("#0b0f14"); panel.axis("off")
    panel.text(.03,.95,"VISUAL PATTERN",color="white",fontsize=12,fontweight="bold",transform=panel.transAxes); panel.text(.03,.87,name,color="#00e676",fontsize=15,fontweight="bold",transform=panel.transAxes); panel.text(.03,.81,f"Confidence: {conf}%",color="#00e676",fontsize=12,transform=panel.transAxes)
    panel.text(.03,.70,"DETECTED FROM GEOMETRY",color="white",fontsize=9,fontweight="bold",transform=panel.transAxes); ts=pattern.get("touches",[0,0]); vals=[f"• Rails: {len(pattern.get('lines',[]))}",f"• Touches: {ts[0]} / {ts[1] if len(ts)>1 else 0}",f"• Convergence: {pattern.get('convergence',0)}%",f"• Parallelism: {pattern.get('parallelism','—')}%","• HH/LH/HL/LL labels: OFF"]
    y=.64
    for t in vals: panel.text(.03,y,t,color="#d7dde7",fontsize=9,transform=panel.transAxes); y-=.055
    panel.text(.03,.30,"PATTERN RULE",color="white",fontsize=9,fontweight="bold",transform=panel.transAxes); panel.text(.03,.24,pattern.get("rule","geometric price structure"),color="#d7dde7",fontsize=9,transform=panel.transAxes,wrap=True)
    fig.tight_layout()
    from io import BytesIO
    buf=BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.close(fig); buf.seek(0); return buf

def _confidence_v4(family,pattern):
    geometry=min(100,int(pattern.get("confidence",0))) if pattern else 0; pattern_score=geometry; brk=family.get("breakout_grade") or {}; breakout={"confirmed":95,"developing":68,"weak":35}.get(brk.get("strength"),45); er=family.get("entry_rules") or {}; confirmation=min(100,int(er.get("passed",0)/max(er.get("required",3),1)*100)); checks=er.get("checks",{}); mom=80 if checks.get("momentum",(False,""))[0] else 35; rsi=80 if checks.get("rsi",(False,""))[0] else 35; momentum=int((mom+rsi)/2); fib=50; td=family.get("topdown") or {}; td_dir=str(td.get("direction","NEUTRAL")).upper(); direction=str(family.get("direction","NEUTRAL")).upper(); topdown=90 if td_dir==direction and direction in ("BUY","SELL") else 55 if td_dir=="NEUTRAL" else 30; obs=family.get("order_blocks") or []; ob=80 if any((o.get("type")=="bullish" if direction=="BUY" else o.get("type")=="bearish") and o.get("freshness")=="untested" for o in obs) else 40
    values={"Geometry":geometry,"Pattern":pattern_score,"Breakout":breakout,"Momentum":momentum,"Confirmation":confirmation,"Fibonacci":fib,"4H/1H Top-down":topdown,"Order Block":ob}; weights={"Geometry":.20,"Pattern":.20,"Breakout":.15,"Momentum":.10,"Confirmation":.10,"Fibonacci":.10,"4H/1H Top-down":.10,"Order Block":.05}; final=round(sum(values[k]*weights[k] for k in values)); return final,values,weights

def _v4_report(family,symbol,original_report):
    pattern=family.get("visual_pattern") or detect_visual_pattern(family.get("df")); final,values,weights=_confidence_v4(family,pattern); direction=str(family.get("direction") or "NEUTRAL").upper(); state=(family.get("continuation_state") or {}).get("state","TRANSITION"); er=family.get("entry_rules") or {}; trend_status=(family.get("trendline_retest") or {}).get("status","INTACT"); structure_ok=state=="REVERSAL_CONFIRMED" if trend_status=="BREAK_RETEST_CONFIRMED" else True; entry_ok=direction in ("BUY","SELL") and bool(er.get("confirmed")) and structure_ok and final>=70 and not family.get("force_wait_pattern"); label="HIGH" if final>=80 else "GOOD" if final>=70 else "MODERATE" if final>=55 else "LOW"
    lines=["","════════════════════════════","🎯 VISUAL PATTERN + CONFIDENCE V4",f"Pattern: {pattern.get('name','None')}",f"Pattern confidence: {int(pattern.get('confidence',0))}%",f"Geometry rule: {pattern.get('rule','—')}",f"Rails: {len(pattern.get('lines',[]))}",f"Touches: {' / '.join(str(x) for x in pattern.get('touches',[])) or '—'}",f"Convergence: {pattern.get('convergence',0)}%","",f"FINAL CONFIDENCE: {final}% · {label}","Confidence details:"]
    for k in values: lines.append(f"• {k}: {values[k]}/100 × {int(weights[k]*100)}% = {values[k]*weights[k]:.1f}")
    lines += ["",f"VISUAL DECISION: {'🔥 ENTRY CONFIRMED' if entry_ok else '⏳ WAIT'}",f"Bias: {direction}",f"Market state: {state}",f"Confirmation gate: {er.get('passed',0)}/{er.get('required',3)}"]
    if not entry_ok:
        reasons=[]
        if not er.get("confirmed"): reasons.append("entry confirmation")
        if not structure_ok: reasons.append("structural reversal")
        if final<70: reasons.append("confidence ≥70%")
        if family.get("force_wait_pattern"): reasons.append("pattern trigger/confirmation")
        lines.append("Missing: "+(", ".join(reasons) or "required confluence"))
    lines.append("════════════════════════════"); return original_report+"\n"+"\n".join(lines)

def _install_runtime_patch():
    try:
        import strategies as _s
        if getattr(_s,"_VISUAL_V4_PATCHED",False): return
        _old_fmt=_s.format_trendline_report; _old_run=_s.run_trendline_analysis
        def _run(symbol,*args,**kwargs):
            fam=_old_run(symbol,*args,**kwargs)
            try:
                df=fam.get("df")
                if df is not None and not getattr(df,"empty",True): fam["visual_pattern"]=detect_visual_pattern(df); fam["visual_confidence"]=_confidence_v4(fam,fam["visual_pattern"])[0]
            except Exception as e: print(f"[visual_v4] detection failed: {e!r}")
            return fam
        def _fmt(family,symbol):
            base=_old_fmt(family,symbol)
            try:return _v4_report(family,symbol,base)
            except Exception as e: print(f"[visual_v4] report failed: {e!r}"); return base
        _s.run_trendline_analysis=_run; _s.format_trendline_report=_fmt; _s._VISUAL_V4_PATCHED=True; print("[visual_v4] runtime integration active")
    except Exception as e: print(f"[visual_v4] runtime patch failed: {e!r}")

try: _install_runtime_patch()
except Exception: pass
