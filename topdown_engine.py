"""
topdown_engine.py
==================
Plain multi-timeframe top-down bias engine: 4H -> 1H, feeding into a 30M
entry (the 30M part is left to the caller -- Trendline or OTE -- since
each has its own entry mechanics).

This REPLACES the old institutional_analysis.py "SMC Top-Down" stack.
There is deliberately no FVG / Order Block / Inducement zone detection
here -- just a normal top-down read:

  1. 4H  -- macro regime via 200 EMA + swing structure bias (context only)
  2. 1H  -- swing structure + structure-based trade permission, gated
            against the 4H macro bias (this is the actual confirmation
            direction -- see market_analysis.structure_trade_permission)

Both remaining strategies (strategies.py) call get_topdown_bias() so
Trendline and OTE agree on overall market direction instead of each
reading the market independently.
"""

from __future__ import annotations

from typing import Any, Dict

import market_data
from market_analysis import analyse_structure, structure_trade_permission




def _ensure_atr(df):
    if df is None or df.empty:
        return df
    if "ATR" in df.columns and not df["ATR"].isna().all():
        return df
    import numpy as np
    import pandas as pd
    h=pd.to_numeric(df["High"], errors="coerce")
    l=pd.to_numeric(df["Low"], errors="coerce")
    c=pd.to_numeric(df["Close"], errors="coerce")
    tr=pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    out=df.copy(); out["ATR"]=tr.rolling(14, min_periods=1).mean()
    return out


def _line_structural_swings(df, left=3, right=3, min_gap=2, min_leg_atr=0.55):
    """Detect pivots on closes (line chart), then map them to candle extremes."""
    import numpy as np
    if df is None or len(df) < left + right + 8:
        return []
    df=_ensure_atr(df)
    close=df["Close"].to_numpy(float); high=df["High"].to_numpy(float); low=df["Low"].to_numpy(float)
    atr=df["ATR"].to_numpy(float)
    candidates=[]
    for i in range(left, len(df)-right):
        w=close[i-left:i+right+1]
        hi=close[i] >= np.max(w); lo=close[i] <= np.min(w)
        if hi and lo: continue
        if hi: candidates.append({"index":i,"type":"high","close_price":float(close[i]),"price":float(high[i]),"time":df.index[i]})
        elif lo: candidates.append({"index":i,"type":"low","close_price":float(close[i]),"price":float(low[i]),"time":df.index[i]})
    if not candidates: return []
    accepted=[]
    for p in candidates:
        if not accepted:
            accepted.append(p); continue
        q=accepted[-1]
        if p["index"]-q["index"] < min_gap:
            # same side: retain the more extreme close; opposite side: keep the later only if meaningful
            if p["type"]==q["type"]:
                better=(p["close_price"]>q["close_price"]) if p["type"]=="high" else (p["close_price"]<q["close_price"])
                if better: accepted[-1]=p
            continue
        leg=abs(p["close_price"]-q["close_price"])/max(float(atr[p["index"]]),1e-9)
        if leg < min_leg_atr:
            if p["type"]==q["type"]:
                better=(p["close_price"]>q["close_price"]) if p["type"]=="high" else (p["close_price"]<q["close_price"])
                if better: accepted[-1]=p
            continue
        if p["type"]==q["type"]:
            better=(p["close_price"]>q["close_price"]) if p["type"]=="high" else (p["close_price"]<q["close_price"])
            if better: accepted[-1]=p
        else:
            accepted.append(p)
    # final alternating cleanup
    clean=[]
    for p in accepted:
        if clean and clean[-1]["type"]==p["type"]:
            better=(p["close_price"]>clean[-1]["close_price"]) if p["type"]=="high" else (p["close_price"]<clean[-1]["close_price"])
            if better: clean[-1]=p
        else: clean.append(p)
    return clean[-16:]


def _classify_swing_structure(swings):
    highs=[p for p in swings if p["type"]=="high"]; lows=[p for p in swings if p["type"]=="low"]
    if len(highs)<2 or len(lows)<2: return "NEUTRAL"
    hh=highs[-1]["close_price"]>highs[-2]["close_price"]
    hl=lows[-1]["close_price"]>lows[-2]["close_price"]
    lh=highs[-1]["close_price"]<highs[-2]["close_price"]
    ll=lows[-1]["close_price"]<lows[-2]["close_price"]
    if hh and hl: return "BULLISH"
    if lh and ll: return "BEARISH"
    return "TRANSITION"


def _cluster_htf_levels(df, swings, max_levels=3):
    if df is None or not swings: return []
    df=_ensure_atr(df); atr=float(df["ATR"].tail(50).mean()) if len(df) else 0.0
    tol=max(atr*0.55, 1e-9)
    clusters=[]
    for p in swings:
        px=float(p["price"]); ctype="resistance" if p["type"]=="high" else "support"
        placed=False
        for c in clusters:
            if abs(px-c["price"])<=tol:
                c["prices"].append(px); c["indices"].append(int(p["index"])); c["types"].add(ctype); c["price"]=sum(c["prices"])/len(c["prices"]); placed=True; break
        if not placed: clusters.append({"price":px,"prices":[px],"indices":[int(p["index"])],"types":{ctype}})
    close=float(df["Close"].iloc[-1]); out=[]
    for c in clusters:
        touches=len(c["prices"]);
        if touches<2: continue
        side="resistance" if c["price"]>close else "support"
        out.append({"price":float(c["price"]),"side":side,"touches":touches,"first_index":min(c["indices"]),"last_index":max(c["indices"]),"source":"4H"})
    out.sort(key=lambda x:(-x["touches"], abs(x["price"]-close)))
    return out[:max_levels]


def _timeframe_context(df, tf):
    df=_ensure_atr(df)
    swings=_line_structural_swings(df, left=3, right=3, min_gap=2, min_leg_atr=0.55)
    bias=_classify_swing_structure(swings)
    prev_h=prev_l=None
    for p in swings:
        if p["type"]=="high":
            p["label"]="HH" if prev_h is not None and p["close_price"]>prev_h else "LH" if prev_h is not None else "H"
            prev_h=p["close_price"]
        else:
            p["label"]="HL" if prev_l is not None and p["close_price"]>prev_l else "LL" if prev_l is not None else "L"
            prev_l=p["close_price"]
    return {"timeframe":tf,"structure_bias":bias,"swings":swings,"key_levels":_cluster_htf_levels(df,swings,max_levels=3),"df":df}


def _ema200_bias(df):
    if df is None or df.empty or "EMA200" not in df.columns:
        return "NEUTRAL", "EMA200 n/a", 0.0
    close = float(df["Close"].iloc[-1])
    ema200 = float(df["EMA200"].iloc[-1])
    if ema200 <= 0:
        return "NEUTRAL", "EMA200 n/a", 0.0
    dist = (close - ema200) / ema200 * 100.0
    if close > ema200 * 1.001:
        return "BUY", f"Above 200 EMA (+{dist:.2f}%)", dist
    if close < ema200 * 0.999:
        return "SELL", f"Below 200 EMA ({dist:.2f}%)", dist
    return "NEUTRAL", f"At 200 EMA ({dist:+.2f}%)", dist


def get_topdown_bias(symbol: str, count_4h: int = 200, count_1h: int = 200) -> Dict[str, Any]:
    """Full hierarchical 4H -> 1H -> 30M context.

    4H supplies macro structure and horizontal key levels. 1H supplies the
    intermediate transition. All swing detection is close/line-chart first,
    then mapped to candle High/Low for drawing. The 30M strategy remains the
    execution layer and is intentionally not changed here.
    """
    df_4h=market_data.fetch_candles(symbol,"4h",count=count_4h)
    df_1h=market_data.fetch_candles(symbol,"1h",count=count_1h)
    if df_4h is None or df_4h.empty or len(df_4h)<40:
        return {"direction":"NEUTRAL","allowed":False,"reasons":["Insufficient 4H data for top-down bias"],"bias_4h":"NEUTRAL","structure_4h":{},"structure_1h":{},"df_4h":df_4h,"df_1h":df_1h,"error":"insufficient_4h_data"}
    if df_1h is None or df_1h.empty or len(df_1h)<40:
        return {"direction":"NEUTRAL","allowed":False,"reasons":["Insufficient 1H data for top-down bias"],"bias_4h":"NEUTRAL","structure_4h":{},"structure_1h":{},"df_4h":df_4h,"df_1h":df_1h,"error":"insufficient_1h_data"}
    df_4h=_ensure_atr(df_4h); df_1h=_ensure_atr(df_1h)
    ema_bias,ema_note,_=_ema200_bias(df_4h)
    s4=analyse_structure(df_4h,left=3,right=3,lookback=80)
    s1=analyse_structure(df_1h,left=3,right=3,lookback=80)
    ctx4=_timeframe_context(df_4h,"4H"); ctx1=_timeframe_context(df_1h,"1H")
    macro=ema_bias
    if ctx4["structure_bias"]=="BULLISH" and macro!="SELL": macro="BUY"
    elif ctx4["structure_bias"]=="BEARISH" and macro!="BUY": macro="SELL"
    allowed,permission_reason,direction=structure_trade_permission(macro,s1)
    # Keep the existing permission engine, but expose the hierarchical swing
    # context explicitly so callers can distinguish macro structure from 30M.
    reasons=[f"4H regime: {ema_note}",f"4H line-structure: {ctx4['structure_bias']}",f"4H structure: {s4.get('note',s4.get('bias'))}",f"1H line-structure: {ctx1['structure_bias']}",f"1H structure: {s1.get('note',s1.get('bias'))}",permission_reason]
    if macro in ("BUY","SELL") and direction in ("BUY","SELL") and direction!=macro:
        reasons.append(f"⚠️ 1H confirmation ({direction}) is counter to the 4H regime ({macro}) -- caution")
    return {
        "direction":direction,"allowed":bool(allowed),"reasons":reasons,"bias_4h":macro,
        "structure_4h":s4,"structure_1h":s1,"df_4h":df_4h,"df_1h":df_1h,
        "swing_context_4h":ctx4,"swing_context_1h":ctx1,
        "swings_4h":ctx4["swings"],"swings_1h":ctx1["swings"],
        "key_levels_4h":ctx4["key_levels"],"key_levels_1h":ctx1["key_levels"],
    }


def format_topdown_summary(bias: Dict[str, Any]) -> str:
    """Short human-readable summary of the 4H/1H top-down read."""
    lines = [
        f"Top-down bias: {bias.get('bias_4h', 'NEUTRAL')} (4H)  →  "
        f"{bias.get('direction', 'NEUTRAL')} (1H confirmation)"
    ]
    for r in bias.get("reasons") or []:
        lines.append(f"  • {r}")
    return "\n".join(lines)
