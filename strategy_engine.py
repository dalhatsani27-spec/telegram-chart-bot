"""
strategy_engine.py
==================
Routes analysis and signal generation according to:
  - SINGLE mode  → only the selected strategy
  - HYBRID mode  → all enabled strategies, pick best confluence, explain why
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import trade_state as ts
from institutional_analysis import run_topdown_analysis, format_institutional_report
from amd_analysis import run_amd_analysis, format_amd_report
from silver_bullet import (
    run_silver_bullet_analysis,
    format_silver_bullet_report,
    build_silver_bullet_ticket,
)
from chart_engine import generate_smc_map, generate_amd_map, generate_trendline_map, generate_ticket_chart
import mt5_data


def _score_smc(analysis: Dict) -> Dict[str, Any]:
    bias = str(analysis.get("overall_bias", "NEUTRAL")).upper()
    score = 40
    reasons = []
    if bias in ("BUY", "BULLISH"):
        score += 20
        reasons.append("HTF bias bullish")
        direction = "BUY"
    elif bias in ("SELL", "BEARISH"):
        score += 20
        reasons.append("HTF bias bearish")
        direction = "SELL"
    else:
        direction = "NEUTRAL"
        reasons.append("No clear HTF bias")

    frames = analysis.get("frames") or []
    if frames:
        f = frames[0]
        n_fvg = len(f.get("fvgs") or [])
        n_ob = len(f.get("order_blocks") or [])
        if n_fvg:
            score += 10
            reasons.append(f"{n_fvg} FVG(s) present")
        if n_ob:
            score += 10
            reasons.append(f"{n_ob} Order Block(s) present")
    return {
        "strategy": ts.STRATEGY_SMC,
        "direction": direction,
        "score": min(100, score),
        "reasons": reasons,
        "analysis": analysis,
        "valid": direction in ("BUY", "SELL") and score >= 55,
    }


def _score_amd(analysis: Dict) -> Dict[str, Any]:
    if "error" in analysis:
        return {
            "strategy": ts.STRATEGY_AMD,
            "direction": "NEUTRAL",
            "score": 0,
            "reasons": [analysis["error"]],
            "analysis": analysis,
            "valid": False,
        }
    phase = str(analysis.get("phase", "")).upper()
    bias = str(analysis.get("amd_bias", "NEUTRAL")).upper()
    score = 35
    reasons = [f"Phase: {phase}"]

    if phase in ("DISPLACEMENT", "CONTINUATION"):
        score += 25
        reasons.append("Strong directional phase")
    elif phase == "REVERSION":
        score += 20
        reasons.append("Reversion / pullback zone — high probability entry area")
    elif phase == "MANIPULATION":
        score += 15
        reasons.append("Manipulation detected — wait for displacement")
    elif phase == "ACCUMULATION":
        score += 5
        reasons.append("Still in accumulation — no expansion yet")

    direction = "NEUTRAL"
    if bias in ("BUY", "BULLISH"):
        direction = "BUY"
        score += 15
    elif bias in ("SELL", "BEARISH"):
        direction = "SELL"
        score += 15

    if analysis.get("manipulation"):
        score += 10
        reasons.append(analysis["manipulation"].get("note", "Liquidity grab seen"))

    return {
        "strategy": ts.STRATEGY_AMD,
        "direction": direction,
        "score": min(100, score),
        "reasons": reasons,
        "analysis": analysis,
        "valid": direction in ("BUY", "SELL") and score >= 55,
    }


def _score_silver_bullet(analysis: Dict) -> Dict[str, Any]:
    if "error" in analysis:
        return {
            "strategy": ts.STRATEGY_SILVER_BULLET,
            "direction": "NEUTRAL",
            "score": 0,
            "reasons": [analysis["error"]],
            "analysis": analysis,
            "valid": False,
        }
    return {
        "strategy": ts.STRATEGY_SILVER_BULLET,
        "direction": analysis.get("direction", "NEUTRAL"),
        "score": analysis.get("score", 0),
        "reasons": analysis.get("reasons") or [],
        "analysis": analysis,
        "valid": bool(analysis.get("valid")),
    }


def _score_trendline(symbol: str, timeframe: str = "15min") -> Dict[str, Any]:
    """Lightweight trendline / structure score using market structure."""
    try:
        from market_structure import analyse_structure
        df = mt5_data.fetch_candles(symbol, timeframe, count=150)
        if df is None or df.empty or len(df) < 40:
            return {
                "strategy": ts.STRATEGY_TRENDLINE,
                "direction": "NEUTRAL",
                "score": 0,
                "reasons": ["Insufficient data"],
                "analysis": {},
                "valid": False,
            }
        st = analyse_structure(df, left=2, right=2, lookback=50)
        bias = str(st.get("bias", "NEUTRAL")).upper()
        direction = "NEUTRAL"
        score = 40
        reasons = [st.get("note") or "Structure scan"]
        if bias in ("BULLISH", "BUY"):
            direction = "BUY"
            score += 25
            reasons.append("Bullish structure / higher lows")
        elif bias in ("BEARISH", "SELL"):
            direction = "SELL"
            score += 25
            reasons.append("Bearish structure / lower highs")
        return {
            "strategy": ts.STRATEGY_TRENDLINE,
            "direction": direction,
            "score": min(100, score),
            "reasons": reasons,
            "analysis": {"structure": st, "df": df, "geometry_data": {"df": df}},
            "valid": direction in ("BUY", "SELL") and score >= 55,
        }
    except Exception as e:
        return {
            "strategy": ts.STRATEGY_TRENDLINE,
            "direction": "NEUTRAL",
            "score": 0,
            "reasons": [str(e)],
            "analysis": {},
            "valid": False,
        }


def run_single_strategy(symbol: str, strategy: Optional[str] = None) -> Dict[str, Any]:
    strategy = strategy or ts.state.get_selected_strategy()
    symbol = symbol.strip().upper()

    if strategy == ts.STRATEGY_SMC:
        analysis = run_topdown_analysis(symbol)
        result = _score_smc(analysis)
        result["report"] = format_institutional_report(analysis)
        return result

    if strategy == ts.STRATEGY_AMD:
        analysis = run_amd_analysis(symbol)
        result = _score_amd(analysis)
        result["report"] = format_amd_report(analysis)
        return result

    if strategy == ts.STRATEGY_SILVER_BULLET:
        analysis = run_silver_bullet_analysis(symbol)
        result = _score_silver_bullet(analysis)
        result["report"] = format_silver_bullet_report(analysis)
        result["ticket"] = build_silver_bullet_ticket(analysis)
        return result

    if strategy == ts.STRATEGY_TRENDLINE:
        result = _score_trendline(symbol)
        result["report"] = (
            f"TRENDLINE {symbol}\n"
            f"Direction: {result['direction']} | Score: {result['score']}\n"
            + "\n".join(f"  • {r}" for r in result["reasons"])
        )
        return result

    return {
        "strategy": strategy,
        "direction": "NEUTRAL",
        "score": 0,
        "reasons": ["Unknown strategy"],
        "valid": False,
        "report": "Unknown strategy",
    }


def run_hybrid(symbol: str) -> Dict[str, Any]:
    """
    Run all enabled strategies, score them, pick the strongest narrative.
    Silver Bullet gets priority when inside its window and valid.
    """
    symbol = symbol.strip().upper()
    candidates: List[Dict] = []

    enabled = ts.state.get_enabled_strategies()

    if ts.STRATEGY_SILVER_BULLET in enabled:
        sb = run_single_strategy(symbol, ts.STRATEGY_SILVER_BULLET)
        candidates.append(sb)

    if ts.STRATEGY_AMD in enabled:
        candidates.append(run_single_strategy(symbol, ts.STRATEGY_AMD))

    if ts.STRATEGY_SMC in enabled:
        candidates.append(run_single_strategy(symbol, ts.STRATEGY_SMC))

    if ts.STRATEGY_TRENDLINE in enabled:
        candidates.append(run_single_strategy(symbol, ts.STRATEGY_TRENDLINE))

    if not candidates:
        return {
            "strategy": "NONE",
            "direction": "NEUTRAL",
            "score": 0,
            "reasons": ["No strategies enabled"],
            "valid": False,
            "report": "No strategies enabled",
            "all_results": [],
        }

    # Priority: valid Silver Bullet first if preferred
    if ts.state.prefer_silver_bullet:
        for c in candidates:
            if c["strategy"] == ts.STRATEGY_SILVER_BULLET and c.get("valid"):
                c["chosen_reason"] = "ICT Silver Bullet window active + setup complete (priority)"
                c["all_results"] = candidates
                return c

    # Otherwise highest score among valid, then any
    valid_ones = [c for c in candidates if c.get("valid")]
    pool = valid_ones if valid_ones else candidates
    best = max(pool, key=lambda x: x.get("score", 0))

    # Build narrative why this strategy won
    why = [
        f"Selected {best['strategy']} (score {best['score']})",
    ]
    for r in best.get("reasons") or []:
        why.append(r)
    others = [c for c in candidates if c["strategy"] != best["strategy"]]
    if others:
        summary = ", ".join(f"{c['strategy']}={c['score']}" for c in others)
        why.append(f"Other scores: {summary}")

    best["chosen_reason"] = " | ".join(why)
    best["all_results"] = candidates
    return best


def run_strategy_for_symbol(symbol: str) -> Dict[str, Any]:
    """Main entry used by Telegram handlers."""
    if ts.state.get_strategy_mode() == ts.STRATEGY_MODE_HYBRID:
        return run_hybrid(symbol)
    return run_single_strategy(symbol)


def format_trade_ticket(result: Dict[str, Any], symbol: str) -> str:
    """Human-readable Mobile Manual ticket."""
    direction = result.get("direction", "NEUTRAL")
    strategy = result.get("strategy", "")
    score = result.get("score", 0)
    reasons = result.get("reasons") or []
    chosen = result.get("chosen_reason") or ""

    lines = [
        "══════════════════════════",
        f"  TRADE TICKET  |  {symbol}",
        "══════════════════════════",
        f"Direction : {direction}",
        f"Strategy  : {strategy}",
        f"Score     : {score}/100",
        f"Mode      : {ts.state.strategy_label()}",
    ]

    ticket = result.get("ticket")
    if ticket:
        lines += [
            f"Entry     : {ticket.get('entry', '—')}",
            f"SL        : {ticket.get('sl', '—')}",
            f"TP1       : {ticket.get('tp1', '—')}",
            f"TP2       : {ticket.get('tp2', '—')}",
            f"Order     : {ticket.get('order_type', 'MARKET')}",
        ]

    if chosen:
        lines.append(f"Why       : {chosen}")
    elif reasons:
        lines.append("Why       :")
        for r in reasons[:6]:
            lines.append(f"  • {r}")

    lines.append("══════════════════════════")
    lines.append("Enter this trade manually on your phone.")
    return "\n".join(lines)
