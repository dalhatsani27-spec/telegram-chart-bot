"""Independent Smart Money Concepts analysis engine.

The engine uses OHLC evidence only. It deliberately avoids claiming direct
visibility into institutional orders. 4H/1H provide context; M30 is the
execution layer. Existing Trendline/OTE engines remain independent.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional

import mplfinance as mpf
import numpy as np
import pandas as pd

import market_data
from market_analysis import detect_confirmation_candle
from topdown_engine import get_topdown_bias


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _structural_pivots(df: pd.DataFrame, left: int = 3, right: int = 3, min_gap: int = 2, min_leg_atr: float = 0.55) -> List[Dict[str, Any]]:
    if df is None or len(df) < left + right + 10:
        return []
    work = df.copy()
    if "ATR" not in work:
        work["ATR"] = _atr(work)
    close, high, low = work["Close"].to_numpy(float), work["High"].to_numpy(float), work["Low"].to_numpy(float)
    atr = work["ATR"].to_numpy(float)
    candidates: List[Dict[str, Any]] = []
    for i in range(left, len(work) - right):
        window = close[i-left:i+right+1]
        is_high = close[i] >= np.max(window) and close[i] > np.min(window)
        is_low = close[i] <= np.min(window) and close[i] < np.max(window)
        if is_high and not is_low:
            candidates.append({"i": i, "price": float(high[i]), "close": float(close[i]), "type": "high"})
        elif is_low and not is_high:
            candidates.append({"i": i, "price": float(low[i]), "close": float(close[i]), "type": "low"})
    accepted: List[Dict[str, Any]] = []
    for p in candidates:
        if not accepted:
            accepted.append(p); continue
        q = accepted[-1]
        if p["i"] - q["i"] < min_gap:
            if p["type"] == q["type"]:
                better = p["close"] > q["close"] if p["type"] == "high" else p["close"] < q["close"]
                if better: accepted[-1] = p
            continue
        leg_atr = abs(p["close"] - q["close"]) / max(float(atr[p["i"]]), 1e-9)
        if leg_atr < min_leg_atr:
            if p["type"] == q["type"]:
                better = p["close"] > q["close"] if p["type"] == "high" else p["close"] < q["close"]
                if better: accepted[-1] = p
            continue
        if p["type"] == q["type"]:
            better = p["close"] > q["close"] if p["type"] == "high" else p["close"] < q["close"]
            if better: accepted[-1] = p
        else:
            accepted.append(p)
    clean: List[Dict[str, Any]] = []
    for p in accepted:
        if clean and clean[-1]["type"] == p["type"]:
            better = p["close"] > clean[-1]["close"] if p["type"] == "high" else p["close"] < clean[-1]["close"]
            if better: clean[-1] = p
        else:
            clean.append(p)
    return clean[-24:]


def _bias(pivots: List[Dict[str, Any]]) -> str:
    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]
    if len(highs) < 2 or len(lows) < 2: return "NEUTRAL"
    hh, hl = highs[-1]["close"] > highs[-2]["close"], lows[-1]["close"] > lows[-2]["close"]
    lh, ll = highs[-1]["close"] < highs[-2]["close"], lows[-1]["close"] < lows[-2]["close"]
    if hh and hl: return "BULLISH"
    if lh and ll: return "BEARISH"
    return "TRANSITION"


def _liquidity(pivots: List[Dict[str, Any]], atr: float) -> List[Dict[str, Any]]:
    tol = max(float(atr) * 0.20, 1e-9)
    pools: List[Dict[str, Any]] = []
    for typ, side in (("high", "BSL"), ("low", "SSL")):
        q = [p for p in pivots if p["type"] == typ]
        for p in q[-12:]: pools.append({"side": side, "price": p["price"], "kind": "SWING", "i": p["i"]})
        for a, b in zip(q[-12:-1], q[-11:]):
            if abs(a["price"] - b["price"]) <= tol:
                pools.append({"side": side, "price": (a["price"] + b["price"]) / 2.0, "kind": "EQUAL", "i": b["i"]})
    unique = {}
    for p in pools: unique[(p["side"], p["kind"], round(p["price"], 8))] = p
    return sorted(unique.values(), key=lambda x: x["i"], reverse=True)


def _liquidity_sweep(df: pd.DataFrame, pools: List[Dict[str, Any]], atr: float) -> Optional[Dict[str, Any]]:
    if len(df) < 2: return None
    x = df.iloc[-1]; threshold = max(float(atr) * 0.03, 1e-9)
    for p in pools:
        if p["side"] == "SSL" and float(x.Low) < p["price"] - threshold and float(x.Close) > p["price"]: return {**p, "event": "SSL_SWEEP"}
        if p["side"] == "BSL" and float(x.High) > p["price"] + threshold and float(x.Close) < p["price"]: return {**p, "event": "BSL_SWEEP"}
    return None


def _displacement(df: pd.DataFrame, atr: pd.Series) -> Optional[Dict[str, Any]]:
    for i in range(max(1, len(df) - 8), len(df)):
        o, h, l, c = map(float, [df.Open.iloc[i], df.High.iloc[i], df.Low.iloc[i], df.Close.iloc[i]])
        rng, body, a = max(h-l, 1e-9), abs(c-o), max(float(atr.iloc[i]), 1e-9)
        if rng >= 1.25*a and body/rng >= .65:
            return {"i": i, "direction": "BUY" if c > o else "SELL", "atr_multiple": rng/a, "body_ratio": body/rng}
    return None


def _structure_event(df: pd.DataFrame, pivots: List[Dict[str, Any]], prior_bias: str) -> Optional[Dict[str, Any]]:
    if len(df) < 2 or not pivots: return None
    close = float(df.Close.iloc[-1]); highs = [p for p in pivots if p["type"] == "high"]; lows = [p for p in pivots if p["type"] == "low"]
    if prior_bias in ("BUY", "BULLISH") and highs and close > highs[-1]["price"]: return {"type": "BOS", "direction": "BUY", "level": highs[-1]["price"]}
    if prior_bias in ("SELL", "BEARISH") and lows and close < lows[-1]["price"]: return {"type": "BOS", "direction": "SELL", "level": lows[-1]["price"]}
    if highs and close > highs[-1]["price"]: return {"type": "CHoCH", "direction": "BUY", "level": highs[-1]["price"]}
    if lows and close < lows[-1]["price"]: return {"type": "CHoCH", "direction": "SELL", "level": lows[-1]["price"]}
    return None


def _promote_mss(event, displacement):
    if event and event["type"] == "CHoCH" and displacement and displacement["direction"] == event["direction"]:
        return {**event, "type": "MSS"}
    return event


def _order_block(df: pd.DataFrame, displacement, atr: float):
    if not displacement: return None
    i = int(displacement["i"])
    for j in range(i-1, max(-1, i-8), -1):
        o, h, l, c = map(float, [df.Open.iloc[j], df.High.iloc[j], df.Low.iloc[j], df.Close.iloc[j]])
        if h-l < max(.25*atr, 1e-9): continue
        opposite = displacement["direction"] == "BUY" and c < o or displacement["direction"] == "SELL" and c > o
        if opposite:
            return {"type": "Bullish OB" if displacement["direction"] == "BUY" else "Bearish OB", "low": l, "high": h, "i": j, "fresh": True, "direction": displacement["direction"]}
    return None


def _fvg(df: pd.DataFrame, atr: pd.Series, direction: str):
    if direction not in ("BUY", "SELL"): return None
    for i in range(len(df)-1, 1, -1):
        h2, l2 = float(df.High.iloc[i-2]), float(df.Low.iloc[i-2]); h, l = float(df.High.iloc[i]), float(df.Low.iloc[i]); minimum = max(float(atr.iloc[i])*.10, 1e-9)
        if direction == "BUY" and l > h2 and l-h2 >= minimum: return {"low": h2, "high": l, "direction": "BUY", "i": i, "size": l-h2}
        if direction == "SELL" and h < l2 and l2-h >= minimum: return {"low": h, "high": l2, "direction": "SELL", "i": i, "size": l2-h}
    return None


def _dealing_range(df: pd.DataFrame, pivots):
    highs = [p["price"] for p in pivots if p["type"] == "high"]; lows = [p["price"] for p in pivots if p["type"] == "low"]
    hi, lo = max(highs[-3:]) if highs else float(df.High.tail(80).max()), min(lows[-3:]) if lows else float(df.Low.tail(80).min())
    eq, px = (hi+lo)/2, float(df.Close.iloc[-1])
    return {"high": hi, "low": lo, "equilibrium": eq, "location": "PREMIUM" if px > eq else "DISCOUNT" if px < eq else "EQUILIBRIUM"}


def _targets(pools, price, direction, risk):
    side = "BSL" if direction == "BUY" else "SSL"
    q = [p for p in pools if p["side"] == side and (p["price"] > price if direction == "BUY" else p["price"] < price)]
    q.sort(key=lambda x: abs(x["price"]-price)); out=[]
    for p in q:
        rr=abs(p["price"]-price)/max(risk,1e-9)
        if rr >= 1.2: out.append({**p,"rr":rr})
        if len(out)==3: break
    return out


def _direction_from_htf(bias, b30):
    if bias.get("direction") in ("BUY","SELL"): return bias["direction"]
    if bias.get("bias_4h") == "BUY" and b30 == "BULLISH": return "BUY"
    if bias.get("bias_4h") == "SELL" and b30 == "BEARISH": return "SELL"
    return "WAIT"


def run_smc_analysis(symbol: str, count: int = 260):
    symbol=str(symbol).strip().upper()
    try:
        h4=market_data.fetch_candles(symbol,"4h",count); h1=market_data.fetch_candles(symbol,"1h",count); m30=market_data.fetch_candles(symbol,"30min",count)
        if any(x is None or len(x)<50 for x in (h4,h1,m30)): return {"error":f"Insufficient data for {symbol}."}
        for frame in (h4,h1,m30): frame["ATR"]=_atr(frame)
        topdown=get_topdown_bias(symbol,count_4h=count,count_1h=count)
        p4,p1,p30=_structural_pivots(h4),_structural_pivots(h1),_structural_pivots(m30)
        b4,b1,b30=_bias(p4),_bias(p1),_bias(p30)
        direction=_direction_from_htf(topdown,b30); atr=float(m30.ATR.iloc[-1]); pools=_liquidity(p4+p1+p30,atr)
        sweep=_liquidity_sweep(m30,pools,atr); displacement=_displacement(m30,m30.ATR)
        event=_promote_mss(_structure_event(m30,p30,direction if direction!="WAIT" else b30),displacement)
        ob=_order_block(m30,displacement,atr); fvg=_fvg(m30,m30.ATR,direction); dr=_dealing_range(h4,p4); entry=float(m30.Close.iloc[-1])
        if direction=="BUY": sl=ob["low"]-atr*.15 if ob else entry-atr*1.5
        elif direction=="SELL": sl=ob["high"]+atr*.15 if ob else entry+atr*1.5
        else: sl=entry
        risk=abs(entry-sl); targets=_targets(pools,entry,direction,risk) if direction in ("BUY","SELL") else []
        ok,candle_name=detect_confirmation_candle(m30,direction) if direction in ("BUY","SELL") else (False,None)
        aligned_4h=(direction=="BUY" and b4=="BULLISH") or (direction=="SELL" and b4=="BEARISH")
        aligned_1h=(direction=="BUY" and b1=="BULLISH") or (direction=="SELL" and b1=="BEARISH")
        sweep_ok=bool(sweep and sweep["event"]==("SSL_SWEEP" if direction=="BUY" else "BSL_SWEEP"))
        disp_ok=bool(displacement and displacement["direction"]==direction); event_ok=bool(event and event["direction"]==direction and event["type"] in ("BOS","MSS")); ob_ok=bool(ob and ob.get("direction")==direction); fvg_ok=bool(fvg and fvg.get("direction")==direction)
        score=sum([20 if aligned_4h else 0,15 if aligned_1h else 0,15 if sweep_ok else 0,15 if disp_ok else 0,15 if event_ok else 0,10 if ob_ok else 0,5 if fvg_ok else 0,5 if ok else 0])
        confirmed=direction in ("BUY","SELL") and score>=70 and event_ok and disp_ok and ob_ok
        return {"symbol":symbol,"df":m30,"direction":direction,"status":"CONFIRMED" if confirmed else "WAIT","score":min(score,100),"bias_4h":b4,"bias_1h":b1,"bias_30m":b30,"topdown":topdown,"dealing_range":dr,"liquidity":pools,"sweep":sweep,"displacement":displacement,"structure_event":event,"order_block":ob,"fvg":fvg,"candle_confirmation":candle_name if ok else None,"entry":entry,"sl":sl,"risk":risk,"targets":targets,"key_levels_4h":topdown.get("key_levels_4h",[]),"pivots_4h":p4,"pivots_1h":p1,"pivots_30m":p30}
    except Exception as e: return {"error":f"SMC analysis failed for {symbol}: {e}"}


def _fmt_price(value):
    return f"{float(value):.5f}" if isinstance(value,(int,float,np.floating)) else "—"


def format_smc_report(a):
    if a.get("error"): return "🏦 SMC ANALYSIS\n\n❌ "+a["error"]
    dr=a["dealing_range"]; sweep=a.get("sweep"); ob=a.get("order_block"); fvg=a.get("fvg"); event=a.get("structure_event"); disp=a.get("displacement")
    lines=[f"🏦 SMC ANALYSIS — {a['symbol']} M30","","TOP-DOWN CONTEXT",f"4H structure: {a['bias_4h']}",f"1H structure: {a['bias_1h']}",f"30M structure: {a['bias_30m']}",f"4H dealing range: {_fmt_price(dr['low'])} — {_fmt_price(dr['high'])}",f"Equilibrium: {_fmt_price(dr['equilibrium'])}",f"Price location: {dr['location']}","","LIQUIDITY",f"Sweep: {sweep['event'] if sweep else '—'}",f"Nearest BSL: {_fmt_price(next((q['price'] for q in a['liquidity'] if q['side']=='BSL'),None))}",f"Nearest SSL: {_fmt_price(next((q['price'] for q in a['liquidity'] if q['side']=='SSL'),None))","","STRUCTURE / ORDER FLOW",f"Event: {event['type']} {event['direction']} @ {_fmt_price(event['level'])}" if event else "Event: —",f"Displacement: {disp['direction']} ({disp['atr_multiple']:.2f} ATR)" if disp else "Displacement: —",f"Order Block: {ob['type']} {_fmt_price(ob['low'])} — {_fmt_price(ob['high'])}" if ob else "Order Block: —",f"FVG: {_fmt_price(fvg['low'])} — {_fmt_price(fvg['high'])}" if fvg else "FVG: —",f"Candle confirmation: {a.get('candle_confirmation') or '—'}","","━━━━━━━━━━━━━━━━","🎯 DECISION",f"BIAS: {a['direction']}",f"STATUS: {a['status']}",f"SMC SCORE: {a['score']}/100",f"ENTRY: {_fmt_price(a['entry'])}",f"SL: {_fmt_price(a['sl'])}"]
    for i,target in enumerate(a.get("targets",[])[:3],1): lines.append(f"TP{i}: {_fmt_price(target['price'])} ({target['kind']} liquidity, {target['rr']:.2f}R)")
    if not a.get("targets"): lines.append("TP: No qualifying liquidity target ≥1.2R")
    return "\n".join(lines)


def generate_smc_chart(a):
    df=a["df"].tail(120).copy()
    if "ATR" not in df: df["ATR"]=_atr(df)
    fig,axes=mpf.plot(df[["Open","High","Low","Close"]],type="candle",style="charles",volume=False,returnfig=True,figsize=(12,7),warn_too_much_data=10000)
    ax=axes[0]; atr=float(df["ATR"].iloc[-1]); n=len(df)
    for level in a.get("key_levels_4h",[]):
        price=float(level["price"])
        if df.Low.min()-5*atr<=price<=df.High.max()+5*atr:
            ax.axhline(price,linestyle="--",linewidth=.9,alpha=.55); ax.text(n-1,price,f"  4H {level['side']} ({level['touches']})",fontsize=6.5,ha="right")
    for pool in a.get("liquidity",[])[:6]:
        price=float(pool["price"])
        if df.Low.min()-4*atr<=price<=df.High.max()+4*atr: ax.axhline(price,linestyle=":",linewidth=.7,alpha=.28)
    dr=a.get("dealing_range") or {}
    if dr.get("equilibrium") is not None: ax.axhline(float(dr["equilibrium"]),linestyle="-.",linewidth=.7,alpha=.35)
    ob=a.get("order_block")
    if ob: ax.axhspan(ob["low"],ob["high"],alpha=.18); ax.text(2,ob["high"],ob["type"],fontsize=7,fontweight="bold")
    fvg=a.get("fvg")
    if fvg: ax.axhspan(fvg["low"],fvg["high"],alpha=.12); ax.text(2,fvg["high"],"FVG",fontsize=7,fontweight="bold")
    ax.set_title(f"{a['symbol']} | M30 | SMC {a['direction']} | {a['status']}"); fig.tight_layout(); out=BytesIO(); fig.savefig(out,format="png",dpi=150,bbox_inches="tight"); out.seek(0); return out


def run_hybrid_analysis(symbol):
    import strategies
    smc=run_smc_analysis(symbol); trendline=strategies.run_trendline_analysis(symbol)
    if smc.get("error") or trendline.get("error"): return {"error":smc.get("error") or trendline.get("error"),"smc":smc,"trendline":trendline}
    agree=smc.get("direction") in ("BUY","SELL") and smc.get("direction")==trendline.get("direction")
    score=min(100,int((smc.get("score",0)+trendline.get("strength",0))/2+(15 if agree else 0)))
    return {"symbol":symbol,"direction":smc["direction"] if agree else "WAIT","status":"CONFIRMED" if agree and smc["status"]=="CONFIRMED" else "WAIT","score":score,"agreement":agree,"smc":smc,"trendline":trendline}


def format_hybrid_report(h):
    if h.get("error"): return "🔀 HYBRID ANALYSIS\n\n❌ "+h["error"]
    s,t=h["smc"],h["trendline"]; sweep=s.get("sweep"); ob=s.get("order_block")
    return "\n".join([f"🔀 HYBRID ANALYSIS — {h['symbol']} M30","",f"SMC: {s.get('direction')} | {s.get('status')} | {s.get('score')}/100",f"Trendline: {t.get('direction','—')} | {t.get('strength',0)}/100",f"Liquidity sweep: {sweep['event'] if sweep else '—'}",f"Order Block: {ob['type'] if ob else '—'}","","━━━━━━━━━━━━━━━━","🎯 DECISION",f"SMC: {s.get('direction')}",f"Trendline: {t.get('direction')}",f"CONFLUENCE: {'AGREE' if h['agreement'] else 'CONFLICT'}",f"BIAS: {h['direction']}",f"STATUS: {h['status']}",f"HYBRID SCORE: {h['score']}/100","","SMC and Trendline remain independent; conflict = WAIT."])
