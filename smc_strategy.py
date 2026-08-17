"""Compatibility API for the bot's SMC strategy.

SMC is powered by strategy_upgrade.smc_analysis and now receives the same
macro fundamental filter used by Trendline and OTE.
"""
from strategy_upgrade import smc_analysis, format_smc_report
from fundamental_analysis import analyze as analyze_fundamentals


def run_smc_analysis(symbol: str, tf_code: str = "30min", topdown=None):
    result = smc_analysis(symbol, tf_code=tf_code, topdown=topdown)
    if not result or result.get("error"):
        return result
    fundamental = analyze_fundamentals(symbol)
    result["fundamental"] = fundamental
    if not fundamental.get("available"):
        return result
    direction = result.get("direction", "NEUTRAL")
    fbias = fundamental.get("bias", "NEUTRAL")
    score = int(result.get("score", result.get("strength", 0)))
    old_score = score
    if direction in ("BUY", "SELL") and fbias in ("BUY", "SELL"):
        if direction == fbias:
            score = min(100, score + 7)
            result.setdefault("reasons", []).append(f"Fundamentals align: {fbias} ({fundamental.get('score',0):+.1f})")
        else:
            score = max(0, score - 10)
            result.setdefault("reasons", []).append(f"Fundamental conflict: technical {direction} vs macro {fbias} ({fundamental.get('score',0):+.1f})")
    if fundamental.get("event_risk") == "HIGH":
        score = max(0, score - 5)
        result.setdefault("reasons", []).append("High-impact macro event risk: reduce size / wait for release")
    result["score"] = score
    result["strength"] = score
    result["fundamental_adjustment"] = score - old_score
    result["gating_notes"] = result.get("reasons", [])
    if "valid" in result:
        result["valid"] = bool(result.get("valid")) and score >= 55
    return result


def build_smc_ticket(analysis):
    if not analysis or not analysis.get("entry_ready"):
        return None
    return {
        "entry": analysis.get("entry"),
        "sl": analysis.get("sl"),
        "tp1": analysis.get("tp1"),
        "tp2": analysis.get("tp2"),
        "tp3": analysis.get("tp3"),
        "direction": analysis.get("direction"),
        "order_type": "MARKET",
    }
