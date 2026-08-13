"""Fresh ICT/SMC analysis engine for the Telegram chart bot.

The engine is intentionally independent from the legacy Trendline/OTE logic.
It performs a 4H -> 1H -> M30 read and only promotes an execution setup when
structure, liquidity, displacement/POI and confirmation line up.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import market_data


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("Open", "High", "Low", "Close"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    tr = pd.concat([out.High-out.Low, (out.High-out.Close.shift()).abs(), (out.Low-out.Close.shift()).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
    out["body"] = (out.Close-out.Open).abs()
    out["range"] = out.High-out.Low
    out["body_ratio"] = out.body / out.range.replace(0, np.nan)
    return out


def _pivots(df: pd.DataFrame, left=3, right=3, min_gap=3, min_leg_atr=.55) -> List[Dict[str, Any]]:
    if df is None or len(df) < left+right+10: return []
    c,h,l,a = df.Close.to_numpy(), df.High.to_numpy(), df.Low.to_numpy(), df.ATR.to_numpy()
    raw=[]
    for i in range(left, len(df)-right):
        if c[i] >= np.max(c[i-left:i+right+1]) and c[i] > c[i-1] and c[i] >= c[i+1]: raw.append({"index":i,"type":"high","price":float(h[i]),"close":float(c[i])})
        elif c[i] <= np.min(c[i-left:i+right+1]) and c[i] < c[i-1] and c[i] <= c[i+1]: raw.append({"index":i,"type":"low","price":float(l[i]),"close":float(c[i])})
    out=[]
    for p in raw:
        if not out: out.append(p); continue
        q=out[-1]
        if p["index"]-q["index"] < min_gap:
            if p["type"]==q["type"]:
                better = p["close"]>q["close"] if p["type"]=="high" else p["close"]<q["close"]
                if better: out[-1]=p
            continue
        leg=abs(p["close"]-q["close"])/max(float(a[p["index"]]),1e-9)
        if p["type"]==q["type"]:
            better = p["close"]>q["close"] if p["type"]=="high" else p["close"]<q["close"]
            if better and leg >= min_leg_atr: out[-1]=p
        elif leg >= min_leg_atr:
            out.append(p)
    return out[-20:]


def _structure(df: pd.DataFrame) -> Dict[str, Any]:
    piv=_pivots(df)
    highs=[p for p in piv if p["type"]=="high"]; lows=[p for p in piv if p["type"]=="low"]
    labels=[]
    for i,p in enumerate(piv):
        prev = [q for q in piv[:i] if q["type"]==p["type"]]
        lab="H" if p["type"]=="high" else "L"
        if prev:
            if p["type"]=="high": lab="HH" if p["close"]>prev[-1]["close"] else "LH"
            else: lab="HL" if p["close"]>prev[-1]["close"] else "LL"
        p=dict(p); p["label"]=lab; labels.append(p)
    if len(highs)>=2 and len(lows)>=2:
        hh=highs[-1]["close"]>highs[-2]["close"]; hl=lows[-1]["close"]>lows[-2]["close"]
        lh=highs[-1]["close"]<highs[-2]["close"]; ll=lows[-1]["close"]<lows[-2]["close"]
        bias="BULLISH" if hh and hl else "BEARISH" if lh and ll else "TRANSITION"
    else: bias="NEUTRAL"
    close=float(df.Close.iloc[-1]); bos=None; mss=None
    # Confirmed close through the latest opposing structural level.
    for p in reversed(piv[:-1]):
        if p["type"]=="high" and close>p["price"]:
            bos={"direction":"BULLISH","level":p["price"],"index":p["index"]}; break
        if p["type"]=="low" and close<p["price"]:
            bos={"direction":"BEARISH","level":p["price"],"index":p["index"]}; break
    # MSS/CHoCH = break against the previous prevailing structure.
    if bos and bias in ("BULLISH","BEARISH") and bos["direction"] != bias:
        mss=dict(bos); mss["type"]="MSS/CHoCH"
    elif bos:
        bos["type"]="BOS"
    return {"bias":bias,"pivots":labels,"bos":bos,"mss":mss}


def _liquidity(df: pd.DataFrame, pivots: List[Dict[str,Any]]) -> Dict[str, Any]:
    highs=[p for p in pivots if p["type"]=="high"]; lows=[p for p in pivots if p["type"]=="low"]
    close=float(df.Close.iloc[-1]); atr=float(df.ATR.iloc[-1])
    def pools(items, side):
        vals=[]
        for i,p in enumerate(items):
            near=[q for q in items[max(0,i-4):i] if abs(q["price"]-p["price"])<=atr*.35]
            if near: vals.append({"price":float(np.mean([p["price"]]+[q["price"] for q in near])),"type":side,"index":p["index"],"strength":len(near)+1})
        # unique by price
        vals=sorted(vals,key=lambda x:x["price"])
        out=[]
        for v in vals:
            if not out or abs(v["price"]-out[-1]["price"])>atr*.25: out.append(v)
            elif v["strength"]>out[-1]["strength"]: out[-1]=v
        return out
    bsl=pools(highs,"BSL"); ssl=pools(lows,"SSL")
    # Any recent wick through a pool followed by a close back inside = sweep.
    sweep=None
    for pool_type,pools_ in (("BSL",bsl),("SSL",ssl)):
        for pool in reversed(pools_):
            if pool_type=="BSL":
                hit=(df.High.iloc[-1]>pool["price"] and close<pool["price"])
            else: hit=(df.Low.iloc[-1]<pool["price"] and close>pool["price"])
            if hit:
                sweep={"type":pool_type,"price":pool["price"],"index":len(df)-1}; break
        if sweep: break
    return {"BSL":bsl,"SSL":ssl,"sweep":sweep}


def _dealing_range(df: pd.DataFrame, structure: Dict[str,Any]) -> Dict[str,Any]:
    piv=structure.get("pivots",[]); highs=[p for p in piv if p["type"]=="high"]; lows=[p for p in piv if p["type"]=="low"]
    if not highs or not lows:
        hi=float(df.High.max()); lo=float(df.Low.min())
    else:
        hi=float(highs[-1]["price"]); lo=float(lows[-1]["price"])
        if hi<=lo: hi=float(df.High.max()); lo=float(df.Low.min())
    eq=(hi+lo)/2; close=float(df.Close.iloc[-1]); pos="PREMIUM" if close>eq else "DISCOUNT" if close<eq else "EQUILIBRIUM"
    return {"high":hi,"low":lo,"equilibrium":eq,"position":pos,"range":hi-lo}


def _displacement(df: pd.DataFrame) -> Optional[Dict[str,Any]]:
    if len(df)<25:return None
    med=float(df.ATR.tail(40).median())
    for i in range(len(df)-1,max(-1,len(df)-8),-1):
        r=df.iloc[i]; body=float(r.body); atr=float(r.ATR)
        if atr>0 and body>=max(1.25*atr,1.35*med) and float(r.body_ratio)>=.60:
            direction="BULLISH" if r.Close>r.Open else "BEARISH"
            return {"index":i,"direction":direction,"body":body,"atr_multiple":body/max(atr,1e-9)}
    return None


def _fvgs(df: pd.DataFrame) -> List[Dict[str,Any]]:
    out=[]
    for i in range(2,len(df)):
        a,b,c=df.iloc[i-2],df.iloc[i-1],df.iloc[i]
        if a.High < c.Low and float(b.body_ratio)>=.5:
            out.append({"type":"bullish","index":i,"bottom":float(a.High),"top":float(c.Low),"displacement":float(b.body/max(b.ATR,1e-9))})
        elif a.Low > c.High and float(b.body_ratio)>=.5:
            out.append({"type":"bearish","index":i,"bottom":float(c.High),"top":float(a.Low),"displacement":float(b.body/max(b.ATR,1e-9))})
    # Keep recent, and mark mitigation.
    close=float(df.Close.iloc[-1]);
    for z in out:
        future=df.iloc[z["index"]+1:]
        z["mitigated"]=bool((future.Low<=z["top"]).any() and (future.High>=z["bottom"]).any())
        z["active"]=not z["mitigated"]
        z["distance_atr"]=abs(close-((z["bottom"]+z["top"])/2))/max(float(df.ATR.iloc[-1]),1e-9)
    return sorted(out,key=lambda z:z["index"])[-12:]


def _order_blocks(df: pd.DataFrame, structure: Dict[str,Any], displacement: Optional[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out=[]; piv=structure.get("pivots",[])
    if not displacement:return out
    di=int(displacement["index"]); direction=displacement["direction"]
    # Search the last opposite candle before displacement; require that the
    # subsequent displacement broke a structural level in the same direction.
    start=max(0,di-8)
    for i in range(di-1,start-1,-1):
        r=df.iloc[i]
        opposite=(direction=="BULLISH" and r.Close<r.Open) or (direction=="BEARISH" and r.Close>r.Open)
        if not opposite: continue
        bottom=float(r.Low); top=float(r.High)
        if direction=="BULLISH" and not any(p["type"]=="high" and float(df.Close.iloc[di])>p["price"] for p in piv if p["index"]<di): continue
        if direction=="BEARISH" and not any(p["type"]=="low" and float(df.Close.iloc[di])<p["price"] for p in piv if p["index"]<di): continue
        future=df.iloc[i+1:]
        mitigated=bool((future.Low<=top).any() and (future.High>=bottom).any())
        out.append({"type":"bullish" if direction=="BULLISH" else "bearish","index":i,"bottom":bottom,"top":top,"mitigated":mitigated,"active":not mitigated,"quality":min(100,int(60+20*min(displacement["atr_multiple"],2)))} )
        break
    return out


def _confirmation(df: pd.DataFrame, direction: str) -> Dict[str,Any]:
    r=df.iloc[-1]; prev=df.iloc[-2];
    bull=(r.Close>r.Open and r.Close>prev.High and float(r.body_ratio)>=.45)
    bear=(r.Close<r.Open and r.Close<prev.Low and float(r.body_ratio)>=.45)
    return {"confirmed": bull if direction=="BUY" else bear if direction=="SELL" else False,
            "candle": "bullish displacement/engulf" if bull else "bearish displacement/engulf" if bear else "no decisive close",
            "body_ratio":float(r.body_ratio)}


def _targets(liq: Dict[str,Any], direction: str, entry: float, risk: float) -> List[float]:
    pools=liq["BSL"] if direction=="BUY" else liq["SSL"]
    vals=[p["price"] for p in pools if (p["price"]>entry if direction=="BUY" else p["price"]<entry)]
    vals=sorted(vals, reverse=direction=="SELL")
    if not vals:
        sign=1 if direction=="BUY" else -1; return [entry+sign*risk*1.5,entry+sign*risk*3]
    return vals[:3]


def run_smc_analysis(symbol: str, count: int=220) -> Dict[str,Any]:
    dfs={tf:_prep(market_data.fetch_candles(symbol,tf,count=count)) for tf in ("4h","1h","30min")}
    if any(df is None or df.empty or len(df)<40 for df in dfs.values()):
        return {"error":"Insufficient 4H/1H/30M data for SMC analysis","symbol":symbol,"direction":"NEUTRAL"}
    s4,s1,s3=(_structure(dfs[tf]) for tf in ("4h","1h","30min"))
    dr=_dealing_range(dfs["4h"],s4); l3=_liquidity(dfs["30min"],s3["pivots"]); disp=_displacement(dfs["30min"]); fvg=_fvgs(dfs["30min"]); obs=_order_blocks(dfs["30min"],s3,disp)
    htf_bias=s4["bias"]; mid_bias=s1["bias"]; exec_bias=s3["bias"]
    direction="BUY" if htf_bias==mid_bias=="BULLISH" else "SELL" if htf_bias==mid_bias=="BEARISH" else "NEUTRAL"
    reasons=[f"4H structure: {htf_bias}",f"1H structure: {mid_bias}",f"30M structure: {exec_bias}",f"4H dealing range: {dr['position']}"]
    score=35
    if direction in ("BUY","SELL"): score+=20
    sweep=l3.get("sweep")
    if sweep:
        if (direction=="BUY" and sweep["type"]=="SSL") or (direction=="SELL" and sweep["type"]=="BSL"): score+=15; reasons.append(f"Liquidity sweep: {sweep['type']} taken")
        else: score-=8; reasons.append(f"Opposing liquidity sweep: {sweep['type']}")
    if disp:
        if (direction=="BUY" and disp["direction"]=="BULLISH") or (direction=="SELL" and disp["direction"]=="BEARISH"): score+=12; reasons.append(f"Displacement: {disp['direction']} {disp['atr_multiple']:.1f}x ATR")
    active_ob=[z for z in obs if z["active"]]; active_fvg=[z for z in fvg if z["active"]]
    if active_ob: score+=7; reasons.append("Fresh structural order block available")
    if active_fvg: score+=5; reasons.append("Active FVG available")
    conf=_confirmation(dfs["30min"],direction)
    if conf["confirmed"]: score+=10; reasons.append(f"Candle confirmation: {conf['candle']}")
    else: reasons.append("Candle confirmation: waiting")
    # Premium/discount discipline: buys prefer discount, sells prefer premium.
    if direction=="BUY" and dr["position"]=="DISCOUNT": score+=6
    elif direction=="SELL" and dr["position"]=="PREMIUM": score+=6
    elif direction in ("BUY","SELL"): score-=6; reasons.append("Entry is on the wrong side of the 4H equilibrium")
    score=max(0,min(100,int(score)))
    setup_ok=direction in ("BUY","SELL") and score>=72 and conf["confirmed"] and bool(active_ob or active_fvg) and ((direction=="BUY" and dr["position"]=="DISCOUNT") or (direction=="SELL" and dr["position"]=="PREMIUM"))
    entry_zone=None; ticket=None
    poi=(active_ob[0] if active_ob else active_fvg[0] if active_fvg else None)
    if poi and direction in ("BUY","SELL"):
        entry_zone=(float(poi["bottom"]),float(poi["top"]))
        entry=(entry_zone[0]+entry_zone[1])/2
        atr=float(dfs["30min"].ATR.iloc[-1]); sl=(entry_zone[0]-atr*.25 if direction=="BUY" else entry_zone[1]+atr*.25); risk=abs(entry-sl)
        tps=_targets(l3,direction,entry,risk)
        ticket={"entry":entry,"sl":sl,"tp1":tps[0],"tp2":tps[1] if len(tps)>1 else tps[0],"tp3":tps[2] if len(tps)>2 else None,"rr":abs(tps[0]-entry)/max(risk,1e-9),"confirmed":setup_ok,"order_type":"LIMIT"}
    return {"symbol":symbol,"direction":direction,"score":score,"confirmed":setup_ok,"reasons":reasons,"topdown":{"4h":s4,"1h":s1,"30m":s3},"dealing_range":dr,"liquidity":l3,"displacement":disp,"order_blocks":obs,"fvgs":fvg,"entry_zone":entry_zone,"ticket":ticket,"df":dfs["30min"]}


def format_smc_report(a: Dict[str,Any]) -> str:
    if a.get("error"): return f"🏦 SMC\n{a['symbol']}\n\n❌ {a['error']}"
    d=a.get("direction","NEUTRAL"); state="CONFIRMED" if a.get("confirmed") else "WAIT"
    td=a["topdown"]; dr=a["dealing_range"]; liq=a["liquidity"]; t=a.get("ticket") or {}
    lines=["══════════════════════════","🏦 SMC MARKET STRUCTURE","══════════════════════════",f"{a['symbol']} | 4H → 1H → 30M",f"Bias: {d}",f"Score: {a['score']}/100",f"Decision: {state}","",f"4H: {td['4h']['bias']}",f"1H: {td['1h']['bias']}",f"30M: {td['30m']['bias']}",f"Dealing Range: {dr['position']}",f"BSL pools: {len(liq['BSL'])}",f"SSL pools: {len(liq['SSL'])}",f"Sweep: {(liq['sweep'] or {}).get('type','None')}","","SETUP LOGIC:"]
    lines += [f"• {r}" for r in a.get("reasons",[])[:8]]
    if t: lines += ["","ENTRY MAP",f"Entry: {t.get('entry'):.5f}",f"SL: {t.get('sl'):.5f}",f"TP1: {t.get('tp1'):.5f}",f"TP2: {t.get('tp2'):.5f}",f"R:R: 1:{t.get('rr',0):.2f}"]
    else: lines += ["","WAIT: structure/POI/confirmation has not aligned yet."]
    return "\n".join(lines)
