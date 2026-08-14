"""Single-timeframe 30-SMA + market-geometry analyzer.

The selected timeframe is the only analysis timeframe. 30 SMA is calculated
on Median Price. A rising SMA maps bullish support from swing lows; a falling
SMA maps bearish resistance from swing highs. If no clean directional rail is
available, the engine falls back to patterns and then horizontal S/R.
"""
from __future__ import annotations
import io
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

TF_LABELS = {"4h":"H4", "1h":"H1", "30min":"M30", "15min":"M15", "5min":"M5", "1min":"M1"}


def _atr(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-prev).abs(), (df["Low"]-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=1).mean()


def _pivots(df: pd.DataFrame, span: int = 3) -> Tuple[List[Dict], List[Dict]]:
    highs, lows = [], []
    h = df["High"].to_numpy(float); l = df["Low"].to_numpy(float)
    for i in range(span, len(df)-span):
        if h[i] >= np.max(h[i-span:i+span+1]): highs.append({"index":i,"price":float(h[i])})
        if l[i] <= np.min(l[i-span:i+span+1]): lows.append({"index":i,"price":float(l[i])})
    return highs, lows


def _fit(points: List[Dict], df: pd.DataFrame) -> Optional[Dict]:
    if len(points) < 3: return None
    xs=np.array([p["index"] for p in points[-6:]],float); ys=np.array([p["price"] for p in points[-6:]],float)
    slope, intercept=np.polyfit(xs,ys,1)
    pred=slope*xs+intercept; resid=np.abs(ys-pred); atr=float(df["ATR"].iloc[-1])
    touches=int(np.sum(resid <= max(atr*0.45, 1e-9))); move=abs(slope*(xs[-1]-xs[0]))
    if move < atr*0.30: return None
    return {"x0":int(xs[0]),"x1":int(xs[-1]),"y0":float(slope*xs[0]+intercept),"y1":float(slope*xs[-1]+intercept),"slope":float(slope),"touches":touches,"y_end":float(slope*(len(df)-1)+intercept),"points":[dict(p) for p in points[-6:]]}


def _levels(df: pd.DataFrame, highs: List[Dict], lows: List[Dict]) -> Tuple[List[float],List[float]]:
    atr=float(df["ATR"].iloc[-1]); tol=max(atr*0.45,1e-9)
    def cluster(vals):
        out=[]
        for v in sorted(vals):
            if not out or abs(v-out[-1])>tol: out.append(v)
            else: out[-1]=(out[-1]+v)/2
        return out
    return cluster([p["price"] for p in lows[-12:]]), cluster([p["price"] for p in highs[-12:]])


def _pattern(df: pd.DataFrame, highs: List[Dict], lows: List[Dict]) -> Optional[Dict]:
    if len(highs)<2 or len(lows)<2: return None
    rh,rl=highs[-4:],lows[-4:]
    hs=np.polyfit([p["index"] for p in rh],[p["price"] for p in rh],1)[0]
    ls=np.polyfit([p["index"] for p in rl],[p["price"] for p in rl],1)[0]
    atr=float(df["ATR"].iloc[-1])
    if hs>0 and ls>0 and abs(hs)>atr/len(df)*0.2 and abs(ls)>atr/len(df)*0.2: return {"name":"ASCENDING CHANNEL","confidence":82,"type":"channel"}
    if hs<0 and ls<0: return {"name":"DESCENDING CHANNEL","confidence":82,"type":"channel"}
    if hs<0 and ls>0: return {"name":"SYMMETRICAL TRIANGLE","confidence":84,"type":"triangle"}
    return None


def analyze(df: pd.DataFrame, symbol: str, tf: str) -> Dict[str,Any]:
    df=df.copy(); df["SMA30"]=(df["High"]+df["Low"])/2.0; df["SMA30"]=df["SMA30"].rolling(30,min_periods=30).mean(); df["ATR"]=_atr(df); df=df.dropna().copy()
    if len(df)<45: return {"error":f"Insufficient {TF_LABELS.get(tf,tf)} candles"}
    sma_now=float(df["SMA30"].iloc[-1]); sma_prev=float(df["SMA30"].iloc[-4]); sma_slope=sma_now-sma_prev
    direction="BUY" if sma_slope>0 else "SELL" if sma_slope<0 else "NEUTRAL"
    highs,lows=_pivots(df); support=_fit(lows,df); resistance=_fit(highs,df); primary=support if direction=="BUY" else resistance if direction=="SELL" else None
    rail_kind="support" if direction=="BUY" else "resistance" if direction=="SELL" else None; atr=float(df["ATR"].iloc[-1]); near=False; distance=None
    if primary: distance=abs(sma_now-primary["y_end"]); near=distance<=atr*0.35
    pattern=_pattern(df,highs,lows) if primary is None else None; supports,resistances=_levels(df,highs,lows); price=float(df["Close"].iloc[-1]); bias=direction
    strong=bool(primary and near and ((direction=="BUY" and primary["slope"]>0) or (direction=="SELL" and primary["slope"]<0))); status="STRONG DIRECTION" if strong else "WAIT / TRANSITION" if primary else "PATTERN" if pattern else "S/R FALLBACK"
    return {"df":df,"symbol":symbol,"timeframe":tf,"timeframe_label":TF_LABELS.get(tf,tf),"sma":sma_now,"sma_slope":sma_slope,"direction":direction,"bias":bias,"trendline":primary,"trendline_kind":rail_kind,"distance":distance,"near":near,"strong":strong,"entry_confirmed":strong,"entry":{"confirmed":strong},"pattern":pattern,"supports":supports,"resistances":resistances,"atr":atr,"price":price,"status":status,"high_pivots":highs,"low_pivots":lows}


def report(r: Dict[str,Any]) -> str:
    if r.get("error"): return "❌ SINGLE-TF ANALYSIS\n\n"+r["error"]
    tl=r.get("trendline"); lines=[f"📐 SINGLE-TF MARKET MAP — {r['symbol']}","━━━━━━━━━━━━━━━━━━━━",f"ANALYSIS TIMEFRAME: {r['timeframe_label']}","30 SMA: MEDIAN PRICE · 30 PERIOD",f"SMA DIRECTION: {'RISING 🟢' if r['direction']=='BUY' else 'FALLING 🔴' if r['direction']=='SELL' else 'FLAT ⚪'}"]
    if tl:
        colour="🟢" if r["trendline_kind"]=="support" else "🔴"; name="BULLISH SUPPORT TRENDLINE" if r["trendline_kind"]=="support" else "BEARISH RESISTANCE TRENDLINE"; lines += [f"TRENDLINE: {colour} {name}",f"TOUCHES: {tl['touches']}",f"SMA ↔ TRENDLINE: {r['distance']:.4g} · {'NEAR/TOUCHING ✅' if r['near'] else 'SEPARATED ⚠️'}"]
    elif r.get("pattern"):
        p=r["pattern"]; lines += [f"PATTERN: {p['name']}",f"PATTERN CONFIDENCE: {p['confidence']}%"]
    else: lines += ["TRENDLINE: NONE CLEAN","STRUCTURE: S/R FALLBACK"]
    if r["supports"]: lines.append("SUPPORT: "+", ".join(f"{x:.5f}" for x in r["supports"][-3:]))
    if r["resistances"]: lines.append("RESISTANCE: "+", ".join(f"{x:.5f}" for x in r["resistances"][-3:]))
    lines += ["━━━━━━━━━━━━━━━━━━━━",f"BIAS: {r['bias']}",f"MARKET STATE: {r['status']}",f"ENTRY RELATIONSHIP: {'CONFIRMED' if r['strong'] else 'WAIT'}"]
    return "\n".join(lines)


def render(r: Dict[str,Any]):
    import matplotlib.pyplot as plt
    df=r["df"].tail(140); offset=len(r["df"])-len(df); fig,ax=plt.subplots(figsize=(12,6.5)); x=np.arange(len(df)); ax.plot(x,df["Close"].values,label="Close",linewidth=1); ax.plot(x,df["SMA30"].values,label="30 SMA (Median)",linewidth=1.7); tl=r.get("trendline")
    if tl:
        color="green" if r["trendline_kind"]=="support" else "red"; xs=np.array([tl["x0"]-offset,len(df)-1]); ys=np.array([tl["y0"],tl["y_end"]]); ax.plot(xs,ys,color=color,linewidth=2.2,label=("Bullish Support" if color=="green" else "Bearish Resistance"))
        for p in tl["points"]:
            px=p["index"]-offset
            if 0<=px<len(df): ax.scatter(px,p["price"],s=22,color=color,zorder=5)
    for lv in r.get("supports",[])[:-1]: ax.axhspan(lv-r["atr"]*.12,lv+r["atr"]*.12,alpha=.10)
    for lv in r.get("resistances",[])[:-1]: ax.axhspan(lv-r["atr"]*.12,lv+r["atr"]*.12,alpha=.10)
    ax.set_title(f"{r['symbol']} {r['timeframe_label']} — {r['status']}"); ax.grid(alpha=.15); ax.legend(loc="upper left"); fig.tight_layout(); buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=140,bbox_inches="tight"); plt.close(fig); buf.seek(0); return buf
