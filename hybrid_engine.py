"""Confluence layer: SMC + existing Trendline. No forced trades on conflict."""
from __future__ import annotations
from typing import Any, Dict
import smc_engine
import strategies


def run_hybrid_analysis(symbol: str) -> Dict[str,Any]:
    tl=strategies.run_trendline_analysis(symbol)
    smc=smc_engine.run_smc_analysis(symbol)
    if tl.get("error") or smc.get("error"):
        return {"error":tl.get("error") or smc.get("error"),"symbol":symbol,"direction":"NEUTRAL","trendline":tl,"smc":smc}
    td=tl.get("direction","NEUTRAL"); sd=smc.get("direction","NEUTRAL")
    score=min(100,int((float(tl.get("strength",0))*.45)+(float(smc.get("score",0))*.55)))
    if td in ("BUY","SELL") and td==sd and smc.get("confirmed"):
        direction=td; confirmed=True; decision="CONFIRMED"
    elif td in ("BUY","SELL") and sd in ("BUY","SELL") and td!=sd:
        direction="NEUTRAL"; confirmed=False; decision="WAIT — CONFLICT"
    else:
        direction=sd if sd in ("BUY","SELL") and smc.get("confirmed") else td if td in ("BUY","SELL") and tl.get("entry_rules",{}).get("confirmed") else "NEUTRAL"
        confirmed=False; decision="WAIT — NEED CONFLUENCE"
    reasons=[f"Trendline: {td} ({tl.get('strength',0)}/100)",f"SMC: {sd} ({smc.get('score',0)}/100)"]
    if td==sd and td in ("BUY","SELL"): reasons.append("Trendline and SMC agree on direction")
    elif td in ("BUY","SELL") and sd in ("BUY","SELL"): reasons.append("Trendline and SMC disagree — no forced trade")
    ticket=smc.get("ticket") if confirmed else None
    return {"symbol":symbol,"direction":direction,"score":score,"confirmed":confirmed,"decision":decision,"reasons":reasons,"trendline":tl,"smc":smc,"ticket":ticket,"df":smc.get("df")}


def format_hybrid_report(a: Dict[str,Any]) -> str:
    if a.get("error"): return f"🔀 HYBRID\n{a['symbol']}\n\n❌ {a['error']}"
    t=a["trendline"]; s=a["smc"]; ticket=a.get("ticket") or {}
    lines=["══════════════════════════","🔀 SMC + TRENDLINE HYBRID","══════════════════════════",f"{a['symbol']} | 4H → 1H → 30M",f"Decision: {a['decision']}",f"Direction: {a['direction']}",f"Confluence Score: {a['score']}/100","",f"Trendline: {t.get('direction')} / {t.get('strength',0)}/100",f"SMC: {s.get('direction')} / {s.get('score',0)}/100"]
    lines += ["","WHY:"]+[f"• {x}" for x in a.get("reasons",[])]
    if ticket: lines += ["","ENTRY MAP",f"Entry: {ticket.get('entry'):.5f}",f"SL: {ticket.get('sl'):.5f}",f"TP1: {ticket.get('tp1'):.5f}",f"TP2: {ticket.get('tp2'):.5f}",f"R:R: 1:{ticket.get('rr',0):.2f}"]
    else: lines += ["","No trade. Wait for both engines to align and confirmation to complete."]
    return "\n".join(lines)
