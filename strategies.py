"""Compatibility API for Trendline/OTE plus shared intelligence filters."""
from strategy_upgrade import trendline_analysis, ote_analysis, format_trendline_report, format_ote_report
from fundamental_analysis import analyze as analyze_fundamentals, format_report as _format_fundamental_report
from alligator_logic import apply_alligator
from market_intelligence import apply as apply_market_intelligence


def _apply_fundamental(result):
    if not result or result.get("error"): return result
    fundamental=analyze_fundamentals(result.get("symbol","")); result["fundamental"]=fundamental
    if not fundamental.get("available"): return result
    direction=result.get("direction","NEUTRAL"); fbias=fundamental.get("bias","NEUTRAL"); old=int(result.get("score",result.get("strength",0))); score=old
    fscore=float(fundamental.get("score",0))
    if direction in ("BUY","SELL") and fbias in ("BUY","SELL"):
        if direction==fbias: score=min(100,score+7); result.setdefault("reasons",[]).append(f"Fundamentals align: {fbias} ({fscore:+.1f})")
        else: score=max(0,score-10); result.setdefault("reasons",[]).append(f"Fundamental conflict: technical {direction} vs macro {fbias} ({fscore:+.1f})")
    if fundamental.get("event_risk")=="HIGH": score=max(0,score-5); result.setdefault("reasons",[]).append("High-impact macro event risk: reduce size / wait for release")
    result["score"]=score; result["strength"]=score; result["fundamental_adjustment"]=score-old
    return result


def _apply_filters(result):
    result=apply_market_intelligence(result)
    result=apply_alligator(result)
    result=_apply_fundamental(result)
    if result and not result.get("error"):
        result["gating_notes"]=result.get("reasons",[])
        if "valid" in result: result["valid"]=bool(result.get("valid")) and int(result.get("score",0))>=55
    return result


def run_trendline_analysis(symbol: str, tf_code: str = "30min", topdown=None): return _apply_filters(trendline_analysis(symbol,tf_code=tf_code,topdown=topdown))
def run_ote_analysis(symbol: str, tf_code: str = "30min", topdown=None): return _apply_filters(ote_analysis(symbol,tf_code=tf_code,topdown=topdown))
def run_fundamental_analysis(symbol: str): return analyze_fundamentals(symbol)
def format_fundamental_report(symbol: str): return _format_fundamental_report(analyze_fundamentals(symbol))


def build_position_container(a):
    if not a or a.get("direction") not in ("BUY","SELL") or not a.get("valid"): return None
    d=a.get("df"); atr=float(d["ATR"].iloc[-1]) if d is not None and "ATR" in d.columns else 0.0; close=float(d["Close"].iloc[-1]) if d is not None else None
    line=a.get("trendline") or {}; x=len(d)-1 if d is not None else 0; entry=float(line.get("slope",0.0))*x+float(line.get("intercept",close or 0.0)) if line else close; entry=close if entry is None else entry
    risk=max(.8*atr,abs(entry)*.0005,1e-9); sl=entry-risk if a["direction"]=="BUY" else entry+risk
    return {"entry":entry,"sl":sl,"tp1":entry+(1.5*risk if a["direction"]=="BUY" else -1.5*risk),"tp2":entry+(2.5*risk if a["direction"]=="BUY" else -2.5*risk),"tp3":entry+(3.5*risk if a["direction"]=="BUY" else -3.5*risk),"order_type":"MARKET","tp3_basis":"3.5R"}
