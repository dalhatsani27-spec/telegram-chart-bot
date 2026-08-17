"""Compatibility API for SMC plus shared market intelligence."""
from strategy_upgrade import smc_analysis, format_smc_report
from fundamental_analysis import analyze as analyze_fundamentals
from alligator_logic import apply_alligator
from market_intelligence import apply as apply_market_intelligence


def _apply_fundamental(result):
    if not result or result.get("error"): return result
    fundamental=analyze_fundamentals(result.get("symbol","")); result["fundamental"]=fundamental
    if not fundamental.get("available"): return result
    direction=result.get("direction","NEUTRAL"); fbias=fundamental.get("bias","NEUTRAL"); old=int(result.get("score",result.get("strength",0))); score=old
    if direction in ("BUY","SELL") and fbias in ("BUY","SELL"):
        if direction==fbias: score=min(100,score+7); result.setdefault("reasons",[]).append(f"Fundamentals align: {fbias} ({fundamental.get('score',0):+.1f})")
        else: score=max(0,score-10); result.setdefault("reasons",[]).append(f"Fundamental conflict: technical {direction} vs macro {fundamental.get('score',0):+.1f}")
    if fundamental.get("event_risk")=="HIGH": score=max(0,score-5); result.setdefault("reasons",[]).append("High-impact macro event risk: reduce size / wait for release")
    result["score"]=score; result["strength"]=score; result["fundamental_adjustment"]=score-old; return result


def run_smc_analysis(symbol: str, tf_code: str = "30min", topdown=None):
    result=smc_analysis(symbol,tf_code=tf_code,topdown=topdown)
    result=apply_market_intelligence(result); result=apply_alligator(result); result=_apply_fundamental(result)
    if result and not result.get("error"):
        result["gating_notes"]=result.get("reasons",[])
        if "valid" in result: result["valid"]=bool(result.get("valid")) and int(result.get("score",0))>=55
    return result


def build_smc_ticket(analysis):
    if not analysis or not analysis.get("entry_ready") or not analysis.get("valid",True): return None
    return {"entry":analysis.get("entry"),"sl":analysis.get("sl"),"tp1":analysis.get("tp1"),"tp2":analysis.get("tp2"),"tp3":analysis.get("tp3"),"direction":analysis.get("direction"),"order_type":"MARKET"}
