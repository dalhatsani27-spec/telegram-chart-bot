from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional

import mplfinance as mpf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import market_data
from topdown_engine import get_topdown_bias


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[Dict[str, Any]]:
    if len(df) < left + right + 5:
        return []
    h, l, c = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)
    out = []
    for i in range(left, len(df) - right):
        if h[i] >= h[i-left:i+right+1].max() and h[i] > h[i-1] and h[i] >= h[i+1]:
            out.append({"i": i, "price": float(h[i]), "close": float(c[i]), "type": "high"})
        if l[i] <= l[i-left:i+right+1].min() and l[i] < l[i-1] and l[i] <= l[i+1]:
            out.append({"i": i, "price": float(l[i]), "close": float(c[i]), "type": "low"})
    out.sort(key=lambda x: x["i"])
    clean = []
    for p in out:
        if clean and clean[-1]["type"] == p["type"]:
            better = p["price"] > clean[-1]["price"] if p["type"] == "high" else p["price"] < clean[-1]["price"]
            if better:
                clean[-1] = p
        else:
            clean.append(p)
    return clean[-40:]


def _classify_swings(pivots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last_high = last_low = None
    out = []
    for p in pivots:
        q = dict(p)
        if p["type"] == "high":
            q["label"] = "H" if last_high is None else ("HH" if p["price"] > last_high else "LH")
            last_high = p["price"]
        else:
            q["label"] = "L" if last_low is None else ("HL" if p["price"] > last_low else "LL")
            last_low = p["price"]
        out.append(q)
    return out


def _bias(pivots: List[Dict[str, Any]]) -> str:
    hs = [p for p in pivots if p["type"] == "high"]
    ls = [p for p in pivots if p["type"] == "low"]
    if len(hs) < 2 or len(ls) < 2:
        return "NEUTRAL"
    if hs[-1]["price"] > hs[-2]["price"] and ls[-1]["price"] > ls[-2]["price"]:
        return "BULLISH"
    if hs[-1]["price"] < hs[-2]["price"] and ls[-1]["price"] < ls[-2]["price"]:
        return "BEARISH"
    return "TRANSITION"


def _liquidity(pivots: List[Dict[str, Any]], atr: float) -> List[Dict[str, Any]]:
    tol = max(atr * 0.20, 1e-9)
    pools = []
    for typ, side in (("high", "BSL"), ("low", "SSL")):
        q = [p for p in pivots if p["type"] == typ]
        for p in q[-16:]:
            pools.append({"side": side, "kind": "SWING", "price": p["price"], "i": p["i"]})
        for a, b in zip(q[-16:-1], q[-15:]):
            if abs(a["price"] - b["price"]) <= tol:
                pools.append({"side": side, "kind": "EQH" if side == "BSL" else "EQL", "price": (a["price"] + b["price"]) / 2, "i": b["i"]})
    pools.sort(key=lambda x: x["i"])
    clean = []
    for p in pools:
        dup = next((q for q in clean if q["side"] == p["side"] and abs(q["price"] - p["price"]) <= tol * 0.5), None)
        if dup:
            if p["kind"] in ("EQH", "EQL"):
                dup.update(p)
        else:
            clean.append(p)
    return clean


def _find_sweeps(df: pd.DataFrame, pools: List[Dict[str, Any]], start: int = 0) -> List[Dict[str, Any]]:
    events = []
    for i in range(max(1, start), len(df)):
        row = df.iloc[i]
        for p in pools:
            if p["i"] >= i:
                continue
            if p["side"] == "SSL" and float(row.Low) < p["price"] and float(row.Close) > p["price"]:
                events.append({**p, "i": i, "event": "SSL_SWEEP", "direction": "BUY"})
            elif p["side"] == "BSL" and float(row.High) > p["price"] and float(row.Close) < p["price"]:
                events.append({**p, "i": i, "event": "BSL_SWEEP", "direction": "SELL"})
    return events[-20:]


def _structure_events(df: pd.DataFrame, pivots: List[Dict[str, Any]], sweeps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map confirmed closed-candle BOS and CHoCH; promote a sweep→CHoCH to MSS."""
    events = []
    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return events
    trend = _bias(pivots)
    broken_high, broken_low = set(), set()
    for i in range(1, len(df)):
        c = float(df.Close.iloc[i])
        ph = next((p for p in reversed(highs) if p["i"] < i and p["i"] not in broken_high), None)
        pl = next((p for p in reversed(lows) if p["i"] < i and p["i"] not in broken_low), None)
        if trend == "BULLISH" and ph and c > ph["price"]:
            broken_high.add(ph["i"]); events.append({"i": i, "level": ph["price"], "direction": "BUY", "event": "BOS", "pivot_i": ph["i"]})
        elif trend == "BULLISH" and pl and c < pl["price"]:
            broken_low.add(pl["i"]); events.append({"i": i, "level": pl["price"], "direction": "SELL", "event": "CHoCH", "pivot_i": pl["i"]}); trend = "BEARISH"
        elif trend == "BEARISH" and pl and c < pl["price"]:
            broken_low.add(pl["i"]); events.append({"i": i, "level": pl["price"], "direction": "SELL", "event": "BOS", "pivot_i": pl["i"]})
        elif trend == "BEARISH" and ph and c > ph["price"]:
            broken_high.add(ph["i"]); events.append({"i": i, "level": ph["price"], "direction": "BUY", "event": "CHoCH", "pivot_i": ph["i"]}); trend = "BULLISH"
    for e in events:
        if e["event"] == "CHoCH":
            s = next((s for s in reversed(sweeps) if s["i"] < e["i"] and s["direction"] == e["direction"] and e["i"] - s["i"] <= 12), None)
            if s:
                e["mss"] = True
    return events[-30:]


def _fvg_zones(df: pd.DataFrame, direction: Optional[str] = None, start: int = 2) -> List[Dict[str, Any]]:
    out = []
    for i in range(max(2, start), len(df)):
        h2, l2 = float(df.High.iloc[i-2]), float(df.Low.iloc[i-2])
        h0, l0 = float(df.High.iloc[i]), float(df.Low.iloc[i])
        atr = max(float(df.ATR.iloc[i]), 1e-9)
        if l0 > h2 and l0 - h2 >= 0.08 * atr and direction in (None, "BUY"):
            out.append({"low": h2, "high": l0, "direction": "BUY", "i": i})
        if h0 < l2 and l2 - h0 >= 0.08 * atr and direction in (None, "SELL"):
            out.append({"low": h0, "high": l2, "direction": "SELL", "i": i})
    return out[-20:]


def _displacement(df: pd.DataFrame, direction: str, start: int = 0) -> Optional[Dict[str, Any]]:
    for i in range(len(df)-1, max(start, len(df)-25)-1, -1):
        o, h, l, c = [float(df[x].iloc[i]) for x in ("Open", "High", "Low", "Close")]
        rng = max(h-l, 1e-9); a = max(float(df.ATR.iloc[i]), 1e-9); d = "BUY" if c > o else "SELL"
        if d == direction and rng >= 1.20*a and abs(c-o)/rng >= 0.60:
            return {"i": i, "direction": d, "atr_multiple": rng/a}
    return None


def _last_opposite_candle(df: pd.DataFrame, start: int, direction: str) -> Optional[Dict[str, Any]]:
    for i in range(min(start-1, len(df)-1), max(-1, start-15), -1):
        o, c = float(df.Open.iloc[i]), float(df.Close.iloc[i])
        if (c < o) if direction == "BUY" else (c > o):
            return {"i": i, "low": float(df.Low.iloc[i]), "high": float(df.High.iloc[i]), "direction": direction}
    return None


def _overlap(a: Dict[str, Any], b: Dict[str, Any], tol: float = 0.0) -> bool:
    return min(a["high"], b["high"]) >= max(a["low"], b["low"]) - tol


def _ote(df: pd.DataFrame, pivots: List[Dict[str, Any]], direction: str) -> Dict[str, Any]:
    if direction not in ("BUY", "SELL"):
        return {"direction": direction, "valid": False}
    highs = [p for p in pivots if p["type"] == "high"]; lows = [p for p in pivots if p["type"] == "low"]
    if not highs or not lows:
        return {"direction": direction, "valid": False}
    if direction == "BUY":
        low = lows[-1]; high = next((p for p in reversed(highs) if p["i"] > low["i"]), None)
        if not high or high["price"] <= low["price"]: return {"direction": direction, "valid": False}
        leg = high["price"] - low["price"]; zl, zh = high["price"]-leg*.79, high["price"]-leg*.62
        px = float(df.Close.iloc[-1]); retr = (high["price"]-px)/leg
    else:
        high = highs[-1]; low = next((p for p in reversed(lows) if p["i"] > high["i"]), None)
        if not low or low["price"] >= high["price"]: return {"direction": direction, "valid": False}
        leg = high["price"] - low["price"]; zl, zh = low["price"]+leg*.62, low["price"]+leg*.79
        px = float(df.Close.iloc[-1]); retr = (px-low["price"])/leg
    return {"direction": direction, "valid": True, "low": min(zl,zh), "high": max(zl,zh), "in_zone": min(zl,zh) <= px <= max(zl,zh), "retracement": round(retr,3), "swing_low": low["price"], "swing_high": high["price"], "low_i": low["i"], "high_i": high["i"]}


def _trendline(df: pd.DataFrame, pivots: List[Dict[str, Any]], direction: str) -> Dict[str, Any]:
    want = "low" if direction == "BUY" else "high"; pts = [p for p in pivots if p["type"] == want][-4:]
    if len(pts) < 2: return {"direction": direction, "valid": False}
    slopes = [(b["price"]-a["price"])/(b["i"]-a["i"]) for a,b in zip(pts[:-1],pts[1:]) if b["i"] != a["i"]]
    if not slopes: return {"direction": direction, "valid": False}
    slope = float(np.median(slopes)); intercept = float(np.median([p["price"]-slope*p["i"] for p in pts])); current = slope*(len(df)-1)+intercept; atr = max(float(df.ATR.iloc[-1]),1e-9)
    touches = sum(abs(p["price"]-(slope*p["i"]+intercept)) <= .35*max(float(df.ATR.iloc[p["i"]]),1e-9) for p in pts); close=float(df.Close.iloc[-1])
    return {"direction":direction,"valid":True,"slope":slope,"intercept":intercept,"current":current,"touches":int(touches),"held":close>=current if direction=="BUY" else close<=current,"near":abs(close-current)<=.5*atr,"x0":pts[0]["i"],"x1":pts[-1]["i"],"y0":slope*pts[0]["i"]+intercept,"y1":slope*pts[-1]["i"]+intercept}


def _score_ob(ob, sweep, structure, fvg, displacement, ote, df):
    score=0; confluence=[]
    if sweep: score+=25; confluence.append(sweep["event"])
    if structure and structure["event"] in ("BOS","CHoCH"): score+=25; confluence.append(structure["event"]); confluence += ["MSS"] if structure.get("mss") else []
    if structure and structure.get("mss"): score+=10
    if fvg and _overlap(ob,fvg,tol=float(df.ATR.iloc[-1])*.20): score+=20; confluence.append("FVG")
    if displacement: score+=10; confluence.append("DISPLACEMENT")
    if ote.get("in_zone"): score+=5; confluence.append("OTE")
    if ob["direction"]=="BUY": invalid=bool((df.Close.iloc[ob["i"]+1:]<ob["low"]).any())
    else: invalid=bool((df.Close.iloc[ob["i"]+1:]>ob["high"]).any())
    if not invalid: score+=5; confluence.append("FRESH")
    high_prob=score>=70 and ("BOS" in confluence or "CHoCH" in confluence) and ("FVG" in confluence or "DISPLACEMENT" in confluence) and any(x in confluence for x in ("BSL_SWEEP","SSL_SWEEP"))
    return {**ob,"score":min(score,100),"confluence":list(dict.fromkeys(confluence)),"high_probability":high_prob}


def _best_order_blocks(df, direction, sweeps, structures, fvgs, ote):
    if direction not in ("BUY","SELL"): return []
    out=[]; ss=[e for e in structures if e["direction"]==direction and e["event"] in ("BOS","CHoCH")]; sw=[s for s in sweeps if s["direction"]==direction]; fs=[f for f in fvgs if f["direction"]==direction]; disp=_displacement(df,direction)
    for st in ss[-8:]:
        ob=_last_opposite_candle(df,st["i"],direction)
        if not ob: continue
        sweep=next((s for s in reversed(sw) if s["i"]<st["i"] and st["i"]-s["i"]<=20),None)
        fvg=next((f for f in reversed(fs) if st["i"]-3<=f["i"]<=st["i"]+8),None)
        out.append(_score_ob(ob,sweep,st,fvg,disp,ote,df)|{"structure_i":st["i"],"sweep_i":sweep["i"] if sweep else None,"fvg_i":fvg["i"] if fvg else None})
    out.sort(key=lambda x:(x["score"],x["i"]),reverse=True); return out[:8]


def _targets(pools, entry, direction, risk):
    side="BSL" if direction=="BUY" else "SSL"; q=[p for p in pools if p["side"]==side and (p["price"]>entry if direction=="BUY" else p["price"]<entry)]; q.sort(key=lambda x:abs(x["price"]-entry))
    return [{**p,"rr":abs(p["price"]-entry)/max(risk,1e-9)} for p in q if abs(p["price"]-entry)/max(risk,1e-9)>=1.2][:3]


def run_smc_analysis(symbol: str, count: int = 260) -> Dict[str, Any]:
    symbol=str(symbol).strip().upper()
    try:
        h4=market_data.fetch_candles(symbol,"4h",count); h1=market_data.fetch_candles(symbol,"1h",count); m30=market_data.fetch_candles(symbol,"30min",count)
        if any(x is None or len(x)<60 for x in (h4,h1,m30)): return {"error":f"Insufficient data for {symbol}."}
        for x in (h4,h1,m30): x["ATR"]=_atr(x)
        td=get_topdown_bias(symbol,count_4h=count,count_1h=count)
        p4,p1,p30=_classify_swings(_pivots(h4)),_classify_swings(_pivots(h1)),_classify_swings(_pivots(m30)); b4,b1,b30=_bias(p4),_bias(p1),_bias(p30)
        atr=max(float(m30.ATR.iloc[-1]),1e-9); pools=_liquidity(p4+p1+p30,atr); sweeps=_find_sweeps(m30,pools,max(0,len(m30)-120)); structures=_structure_events(m30,p30,sweeps)
        direction=td.get("direction") if td.get("direction") in ("BUY","SELL") else ("BUY" if b30=="BULLISH" else "SELL" if b30=="BEARISH" else "WAIT")
        recent_sweep=next((s for s in reversed(sweeps) if s["direction"]==direction),None); recent_struct=next((e for e in reversed(structures) if e["direction"]==direction),None)
        displacement=_displacement(m30,direction,recent_sweep["i"] if recent_sweep else 0) if direction in ("BUY","SELL") else None; fvgs=_fvg_zones(m30,direction if direction in ("BUY","SELL") else None); ote=_ote(m30,p30,direction); trendline=_trendline(m30,p30,direction); obs=_best_order_blocks(m30,direction,sweeps,structures,fvgs,ote)
        for ob in obs:
            fvg=next((f for f in fvgs if f["i"]==ob.get("fvg_i")),None); ob.update(_score_ob(ob,recent_sweep,recent_struct,fvg,displacement,ote,m30))
        obs.sort(key=lambda x:(x["score"],x["i"]),reverse=True); best_ob=obs[0] if obs else None; hp=[o for o in obs if o.get("high_probability")]; mss=next((e for e in reversed(structures) if e.get("mss") and e["direction"]==direction),None)
        entry=float(m30.Close.iloc[-1]); sl=(best_ob["low"]-atr*.15 if direction=="BUY" else best_ob["high"]+atr*.15) if best_ob else (entry-atr*1.5 if direction=="BUY" else entry+atr*1.5); risk=abs(entry-sl); targets=_targets(pools,entry,direction,risk) if direction in ("BUY","SELL") else []
        sweep_ok=bool(recent_sweep); structure_ok=bool(recent_struct and recent_struct["event"] in ("BOS","CHoCH")); fvg_ok=bool(best_ob and "FVG" in best_ob.get("confluence",[])); ob_ok=bool(best_ob and best_ob.get("high_probability")); displacement_ok=bool(displacement); aligned=(direction=="BUY" and b4=="BULLISH" and b1!="BEARISH") or (direction=="SELL" and b4=="BEARISH" and b1!="BULLISH")
        score=sum([20 if sweep_ok else 0,20 if structure_ok else 0,15 if fvg_ok else 0,20 if ob_ok else 0,10 if displacement_ok else 0,10 if aligned else 0,5 if ote.get("in_zone") else 0]); confirmed=direction in ("BUY","SELL") and sweep_ok and structure_ok and ob_ok and score>=70
        if not sweep_ok: waiting="BSL/SSL LIQUIDITY SWEEP"
        elif not structure_ok: waiting="BOS / CHoCH / MSS"
        elif not ob_ok: waiting="HIGH-PROBABILITY OB (SWEEP + STRUCTURE + FVG/DISPLACEMENT)"
        elif not confirmed: waiting="OB RETEST / ENTRY WINDOW"
        else: waiting="ENTRY WINDOW / OB RETEST"
        return {"symbol":symbol,"df":m30,"direction":direction,"status":"CONFIRMED" if confirmed else "WAIT","waiting_for":waiting,"score":min(score,100),"bias_4h":b4,"bias_1h":b1,"bias_30m":b30,"topdown":td,"liquidity":pools,"sweeps":sweeps,"sweep":recent_sweep,"structures":structures,"structure_event":recent_struct,"choch":next((e for e in reversed(structures) if e["event"]=="CHoCH"),None),"bos":next((e for e in reversed(structures) if e["event"]=="BOS"),None),"mss":mss,"order_block":best_ob,"order_blocks":obs,"high_probability_obs":hp,"fvg":next((f for f in reversed(fvgs) if f["direction"]==direction),None),"fvgs":fvgs,"displacement":displacement,"ote":ote,"trendline":trendline,"entry":entry,"sl":sl,"risk":risk,"targets":targets,"swing_labels":p30,"requirements":{"sweep":sweep_ok,"structure":structure_ok,"fvg":fvg_ok,"high_probability_ob":ob_ok,"displacement":displacement_ok,"mss":bool(mss),"htf_alignment":aligned,"ote":bool(ote.get("in_zone"))}}
    except Exception as e:
        return {"error":f"SMC analysis failed for {symbol}: {e}"}


def _p(x): return f"{float(x):.5f}" if isinstance(x,(int,float,np.floating)) else "—"


def format_smc_report(a):
    if a.get("error"): return "🏦 TRUE SMC ANALYSIS\n\n❌ "+a["error"]
    r=a.get("requirements",{}); s,ch,bos,mss=a.get("sweep"),a.get("choch"),a.get("bos"),a.get("mss"); ob=a.get("order_block"); ote,tl=a.get("ote",{}),a.get("trendline",{})
    lines=[f"🏦 TRUE SMC — {a['symbol']} M30","",f"BIAS: {a['direction']} | STATUS: {a['status']} | SCORE: {a['score']}/100",f"4H {a['bias_4h']} | 1H {a['bias_1h']} | 30M {a['bias_30m']}","","MARKET STRUCTURE",f"CHoCH: {_p(ch['level']) if ch else '—'}",f"MSS: {_p(mss['level']) if mss else '—'}",f"BOS: {_p(bos['level']) if bos else '—'}","","LIQUIDITY",f"BSL pools: {sum(1 for x in a['liquidity'] if x['side']=='BSL')}",f"SSL pools: {sum(1 for x in a['liquidity'] if x['side']=='SSL')}",f"Latest sweep: {s['event'] if s else '—'} @ {_p(s['price']) if s else '—'}","","INSTITUTIONAL ZONE",f"OB: {ob['direction']} {_p(ob['low'])} — {_p(ob['high'])}" if ob else "OB: —",f"OB SCORE: {ob.get('score')}/100 | CONFLUENCE: {', '.join(ob.get('confluence',[]))}" if ob else "OB CONFLUENCE: —",f"FVG: {_p(a['fvg']['low'])} — {_p(a['fvg']['high'])}" if a.get('fvg') else "FVG: —","","TRENDLINE / OTE",f"Trendline: {'VALID' if tl.get('valid') else '—'} | touches {tl.get('touches','—')} | {'NEAR' if tl.get('near') else 'AWAY'}",f"OTE 62–79%: {'IN ZONE' if ote.get('in_zone') else 'OUTSIDE'}" if ote.get('valid') else "OTE: —","","SEQUENCE CHECK",f"Sweep {'✅' if r.get('sweep') else '❌'} | Structure {'✅' if r.get('structure') else '❌'} | FVG {'✅' if r.get('fvg') else '❌'}",f"High-prob OB {'✅' if r.get('high_probability_ob') else '❌'} | Displacement {'✅' if r.get('displacement') else '❌'}",f"MSS {'✅' if r.get('mss') else '—'} | HTF alignment {'✅' if r.get('htf_alignment') else '❌'} | OTE {'✅' if r.get('ote') else '—'}","",f"WAITING FOR: {a['waiting_for']}","",f"ENTRY: {_p(a['entry'])} | SL: {_p(a['sl'])}"]
    for i,t in enumerate(a.get("targets",[])[:3],1): lines.append(f"TP{i}: {_p(t['price'])} ({t['kind']} {t['rr']:.2f}R)")
    return "\n".join(lines)


def _label(ax,i,y,text,va="center"):
    ax.annotate(text,xy=(i,y),xytext=(i,y),fontsize=7,fontweight="bold",ha="left",va=va,bbox=dict(boxstyle="round,pad=.20",alpha=.78),zorder=30)


def generate_smc_chart(a,title_prefix="TRUE SMC"):
    df=a["df"].tail(160).copy(); offset=len(a["df"])-len(df); df["ATR"]=_atr(df)
    fig,axes=mpf.plot(df[["Open","High","Low","Close"]],type="candle",style="charles",volume=False,returnfig=True,figsize=(15,9),warn_too_much_data=10000); ax=axes[0]; atr=max(float(df.ATR.iloc[-1]),1e-9); n=len(df); lo=float(df.Low.min())-3*atr; hi=float(df.High.max())+3*atr
    for p in a.get("liquidity",[]):
        px=float(p["price"])
        if lo<=px<=hi:
            ax.axhline(px,linestyle=":" if p["kind"] in ("EQH","EQL") else "--",linewidth=.75,alpha=.30); ax.text(n-1,px,p["kind"] if p["kind"] in ("EQH","EQL") else p["side"],fontsize=6.5,ha="right",va="bottom")
    for e in a.get("structures",[]):
        x=int(e["i"])-offset
        if 0<=x<n and lo<=e["level"]<=hi:
            ax.axhline(e["level"],linestyle="--" if e["event"]=="BOS" else "-",linewidth=1.0,alpha=.50); _label(ax,x,float(e["level"]),"MSS" if e.get("mss") else e["event"])
    for p in a.get("swing_labels",[]):
        x=int(p["i"])-offset
        if 0<=x<n: _label(ax,x,float(p["price"]),p["label"],va="bottom" if p["type"]=="high" else "top")
    for ob in a.get("order_blocks",[]):
        left=int(ob["i"])-offset
        if left>=n or left<-20: continue
        bottom,top=float(ob["low"]),float(ob["high"])
        if top<lo or bottom>hi: continue
        hp=bool(ob.get("high_probability")); alpha=.24 if hp else .06; width=max(1,n-1-max(0,left)); face="#6abf69" if ob["direction"]=="BUY" else "#e76f51"; edge="#2f6f2f" if ob["direction"]=="BUY" else "#9b2c2c"
        ax.add_patch(Rectangle((max(0,left),bottom),width,max(top-bottom,atr*.03),facecolor=face,edgecolor=edge,alpha=alpha,zorder=2))
        if hp: _label(ax,max(0,left),top,f"HIGH-PROB OB {ob['score']}/100 | {' + '.join(ob.get('confluence',[]))}",va="bottom")
    for f in a.get("fvgs",[])[-8:]:
        left=int(f["i"])-offset-2
        if left<n and left>=-10:
            ax.add_patch(Rectangle((max(0,left),f["low"]),max(1,n-max(0,left)),f["high"]-f["low"],facecolor="#4c9bd4",edgecolor="#21618c",alpha=.10,zorder=1)); _label(ax,max(0,left),f["high"],"FVG",va="bottom")
    tl=a.get("trendline") or {}
    if tl.get("valid"):
        xx=np.array([max(0,int(tl["x0"])-offset),n-1]); yy=tl["slope"]*(xx+offset)+tl["intercept"]; ax.plot(xx,yy,linewidth=1.5,alpha=.65); ax.text(n-1,yy[-1],"TRENDLINE",fontsize=7,ha="right",va="bottom")
    ote=a.get("ote") or {}
    if ote.get("valid"):
        left=max(0,int(min(ote["low_i"],ote["high_i"]))-offset); ax.add_patch(Rectangle((left,ote["low"]),max(1,n-left),ote["high"]-ote["low"],facecolor="#b38bd4",edgecolor="#7d4ca6",alpha=.10,zorder=1)); ax.text(left+1,ote["high"],"OTE 62–79%",fontsize=7,fontweight="bold",va="bottom")
    if a.get("sweep"):
        s=a["sweep"]; x=int(s["i"])-offset
        if 0<=x<n: _label(ax,x,float(s["price"]),s["event"])
    ax.axhline(float(a["entry"]),linestyle="-",linewidth=.8,alpha=.35); ax.text(n-1,float(a["entry"]),"ENTRY",fontsize=7,ha="right",va="bottom")
    ax.axhline(float(a["sl"]),linestyle=":",linewidth=.8,alpha=.40); ax.text(n-1,float(a["sl"]),"SL",fontsize=7,ha="right",va="bottom")
    for i,t in enumerate(a.get("targets",[])[:3],1): ax.axhline(float(t["price"]),linestyle=":",linewidth=.7,alpha=.30); ax.text(n-1,float(t["price"]),f"TP{i}",fontsize=7,ha="right",va="bottom")
    ax.set_title(f"{a['symbol']} | M30 | {title_prefix} | {a['direction']} | {a['status']} | {a['score']}/100"); fig.tight_layout(); out=BytesIO(); fig.savefig(out,format="png",dpi=160,bbox_inches="tight"); plt.close(fig); out.seek(0); return out


def run_hybrid_analysis(symbol):
    import strategies
    smc=run_smc_analysis(symbol); trend=strategies.run_trendline_analysis(symbol); ote=strategies.run_ote_analysis(symbol)
    if smc.get("error"): return {"error":smc["error"],"smc":smc,"trendline":trend,"ote":ote}
    sdir=smc.get("direction"); tdir=trend.get("direction"); odir=ote.get("direction"); dirs=[x for x in (tdir,odir) if x in ("BUY","SELL")]; agree=sdir in ("BUY","SELL") and bool(dirs) and all(x==sdir for x in dirs)
    score=int(smc.get("score",0))+(10 if tdir==sdir else 0)+(10 if odir==sdir else 0)
    return {"symbol":symbol,"direction":sdir if agree else "WAIT","status":"CONFIRMED" if agree and smc.get("status")=="CONFIRMED" else "WAIT","score":min(score,100),"agreement":agree,"smc":smc,"trendline":trend,"ote":ote}


def format_hybrid_report(h):
    if h.get("error"): return "🔀 HYBRID ANALYSIS\n\n❌ "+h["error"]
    s,t,o=h["smc"],h["trendline"],h["ote"]
    return "\n".join([f"🔀 HYBRID — {h['symbol']} M30","",f"SMC: {s.get('direction')} | {s.get('status')} | {s.get('score')}/100",f"Trendline: {t.get('direction','—')} | {t.get('strength',0)}/100",f"OTE: {o.get('direction','—')} | {'IN ZONE' if o.get('in_ote') or o.get('in_zone') else 'OUTSIDE'}",f"Liquidity sweep: {(s.get('sweep') or {}).get('event','—')}",f"OB: {(s.get('order_block') or {}).get('score','—')}/100","",f"CONFLUENCE: {'AGREE' if h['agreement'] else 'WAIT / CONFLICT'}",f"BIAS: {h['direction']} | STATUS: {h['status']} | SCORE: {h['score']}/100","SMC remains the structure/liquidity engine; Trendline and OTE are independent confluence layers."])


def generate_hybrid_chart(h):
    if h.get("error") or not h.get("smc") or h["smc"].get("error"): return None
    return generate_smc_chart(h["smc"],title_prefix="HYBRID")
