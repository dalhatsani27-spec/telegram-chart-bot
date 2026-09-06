"""Unified market-intelligence strategy.

Trendline, SMC, OTE and fundamental analysis are evidence extractors, not
user-selectable strategies. The engine runs the real engines and reasons
over market state before deciding.

This module is the single decision layer used by Telegram analysis and
(should be) the live execution path.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import market_data
import fundamental_analysis
import strategies
import smc_strategy
from topdown_engine import get_topdown_bias

STRATEGY_NAME = "Unified Market Intelligence"
POLICY = "ONE_STRATEGY_TRENDLINE_SMC_OTE_FUNDAMENTAL_INTELLIGENCE"

# Every decision this engine makes gets appended here, so scoring weights
# (currently fixed constants -- see analyze()) can eventually be checked
# against and tuned by real outcomes, instead of staying guesses forever.
# report_event() in execution_engine.py appends the matching outcome line
# when a trade tied to a signal_id closes.
SIGNAL_LOG_PATH = os.environ.get("SIGNAL_LOG_PATH", "signal_log.jsonl")


def _log_signal(result: Dict[str, Any]) -> None:
    try:
        record = {
            "signal_id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "symbol": result["symbol"],
            "timeframe": result["timeframe"],
            "decision": result["decision"],
            "direction": result["direction"],
            "ready": result["ready"],
            "score": result["score"],
            "weights": result["weights"],
            "evidence_sources": result["evidence_sources"],
            "conflict": result["conflict"],
        }
        result["signal_id"] = record["signal_id"]
        with open(SIGNAL_LOG_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"[unified] signal log write failed: {exc!r}")


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def alligator_state(df: pd.DataFrame) -> Dict[str, Any]:
    """Bill Williams-style Alligator state; no EMA20/50 is used."""
    median = (df["High"] + df["Low"]) / 2
    jaw = median.rolling(13, min_periods=13).mean().shift(8)
    teeth = median.rolling(8, min_periods=8).mean().shift(5)
    lips = median.rolling(5, min_periods=5).mean().shift(3)
    if len(df) < 20 or any(x.isna().iloc[-1] for x in (jaw, teeth, lips)):
        return {
            "state": "UNKNOWN",
            "direction": "NEUTRAL",
            "jaw": None,
            "teeth": None,
            "lips": None,
            "spread_atr": None,
            "opening": False,
        }
    j, t, li = [float(x.iloc[-1]) for x in (jaw, teeth, lips)]
    atr = max(float(_atr(df).iloc[-1]), 1e-12)
    spread = max(j, t, li) - min(j, t, li)
    prev = max(float(jaw.iloc[-4]), float(teeth.iloc[-4]), float(lips.iloc[-4])) - min(
        float(jaw.iloc[-4]), float(teeth.iloc[-4]), float(lips.iloc[-4])
    )
    bullish = li > t > j
    bearish = li < t < j
    opening = spread > prev * 1.08
    compressed = spread < atr * 0.35
    if compressed:
        state = "SLEEPING"
    elif bullish and opening:
        state = "AWAKENING_BULLISH"
    elif bearish and opening:
        state = "AWAKENING_BEARISH"
    elif bullish:
        state = "BULLISH"
    elif bearish:
        state = "BEARISH"
    else:
        state = "TRANSITION"
    return {
        "state": state,
        "direction": "BUY" if bullish else "SELL" if bearish else "NEUTRAL",
        "jaw": j,
        "teeth": t,
        "lips": li,
        "spread_atr": round(spread / atr, 2),
        "opening": opening,
    }


def _safe_dir(value: Any) -> str:
    v = str(value or "NEUTRAL").upper()
    if v in ("BUY", "BULLISH", "LONG"):
        return "BUY"
    if v in ("SELL", "BEARISH", "SHORT"):
        return "SELL"
    return "NEUTRAL"


def _extract_trendline_intel(family: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the full Trendline engine output into intelligence."""
    if not family or family.get("error"):
        return {
            "direction": "NEUTRAL",
            "quality": 0,
            "event": "NONE",
            "touches": 0,
            "strength": 0,
            "confirmed": False,
            "active_setup": "NONE",
            "error": family.get("error") if family else "no_trendline_data",
            "raw": family,
        }

    direction = _safe_dir(family.get("short_term_signal") or family.get("direction"))
    strength = int(family.get("strength") or 0)
    touches = int(family.get("primary_touches") or 0)
    quality = min(100, max(0, strength))

    retest = family.get("trendline_retest") or {}
    event = "NONE"
    status = str(retest.get("status") or "INTACT").upper()
    if status == "BREAK_RETEST_CONFIRMED":
        event = "BREAK_RETEST_CONFIRMED"
    elif status in ("BREAK_CONFIRMED", "BREAK_DEVELOPING"):
        event = "BREAKOUT"
    elif status == "FAKEOUT":
        event = "FAKEOUT"
    elif direction == "BUY":
        event = "SUPPORT_HOLD"
    elif direction == "SELL":
        event = "RESISTANCE_HOLD"

    pos = None
    try:
        pos = strategies.build_position_container(family)
    except Exception:
        pos = None

    confirmed = bool(pos and pos.get("confirmed"))
    if not confirmed and event == "BREAK_RETEST_CONFIRMED":
        confirmed = True

    return {
        "direction": direction,
        "quality": quality,
        "strength": strength,
        "event": event,
        "touches": touches,
        "confirmed": confirmed,
        "active_setup": family.get("active_setup") or "TRENDLINE",
        "setup_scores": family.get("setup_scores") or {},
        "continuation_state": family.get("continuation_state"),
        "family_kind": family.get("family_kind"),
        "primary_quality": family.get("primary_quality"),
        "gating_notes": family.get("gating_notes") or [],
        "reasons": family.get("reasons") or [],
        "position": pos,
        "error": None,
        "raw": family,
    }


def _extract_smc_intel(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the full SMC engine output into intelligence."""
    if not analysis or analysis.get("error"):
        return {
            "direction": "NEUTRAL",
            "structure": {},
            "liquidity": {},
            "zone": {"status": "NONE", "confluence": False},
            "sweep": None,
            "entry_ready": False,
            "error": analysis.get("error") if analysis else "no_smc_data",
            "raw": analysis,
        }

    direction = _safe_dir(analysis.get("bias"))
    liquidity = analysis.get("liquidity") or {}
    trap = analysis.get("trap_pool")
    sweep = trap if trap and trap.get("status") == "SWEPT" else None

    zone = analysis.get("zone") or {"status": "NONE", "confluence": False}
    structure = analysis.get("structure") or {}

    return {
        "direction": direction,
        "structure": structure,
        "liquidity": liquidity,
        "zone": zone,
        "sweep": sweep,
        "entry_ready": bool(analysis.get("entry_ready")),
        "price_in_zone": bool(analysis.get("price_in_zone")),
        "candle_confirmed": bool(analysis.get("candle_confirmed")),
        "location": analysis.get("location"),
        "status": analysis.get("status"),
        "sl": analysis.get("sl"),
        "tp1": analysis.get("tp1"),
        "tp2": analysis.get("tp2"),
        "entry": analysis.get("entry"),
        "error": None,
        "raw": analysis,
    }


def _extract_ote_intel(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the full OTE engine output into intelligence."""
    if not analysis or analysis.get("error"):
        return {
            "direction": "NEUTRAL",
            "location": "UNKNOWN",
            "retracement": None,
            "quality": 0,
            "valid": False,
            "zone_state": "UNKNOWN",
            "error": analysis.get("error") if analysis else "no_ote_data",
            "raw": analysis,
        }

    direction = _safe_dir(analysis.get("direction"))
    zone_state = str(analysis.get("zone_state") or analysis.get("status") or "WAITING").upper()
    score = int(analysis.get("score") or 0)

    if zone_state == "ACTIVE":
        location = "DEEP_RETRACEMENT"
    elif zone_state in ("WAITING",):
        location = "WAITING"
    elif "MITIGATED" in zone_state or "PASSED" in zone_state:
        location = "MITIGATED"
    elif "TOO_DEEP" in zone_state or "INVALID" in zone_state:
        location = "OVER_RETRACED"
    else:
        location = zone_state

    return {
        "direction": direction,
        "location": location,
        "retracement": None,
        "quality": score,
        "valid": bool(analysis.get("valid")),
        "zone_state": zone_state,
        "zone": analysis.get("zone"),
        "confirmation": analysis.get("confirmation"),
        "ticket": analysis.get("ticket") or analysis.get("position"),
        "reasons": analysis.get("reasons") or [],
        "error": None,
        "raw": analysis,
    }


def analyze(symbol: str, timeframe: str = "30min", include_htf: bool = True, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Run the real Trendline + SMC + OTE engines and produce one decision.

    Pass `df` when you already have authoritative candles for this
    symbol/timeframe (e.g. an EA just pushed its own live MT5 bars) --
    every sub-engine then reads that same data instead of each pulling
    its own independent copy, which is what let the live-trading brain
    and the Telegram-report brain drift onto different bars entirely.
    """
    symbol = str(symbol or "").strip().upper()
    timeframe = timeframe or "30min"

    if df is None:
        df = market_data.fetch_candles(symbol, timeframe, count=300)
    if df is None or df.empty or len(df) < 80:
        return {
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "timeframe": timeframe,
            "error": "insufficient_data",
            "decision": "WAIT",
            "direction": "NEUTRAL",
            "ready": False,
            "score": 0,
        }

    df = df.copy()
    if "ATR" not in df.columns or df["ATR"].isna().all():
        df["ATR"] = _atr(df)

    htf: Dict[str, Any] = {}
    if include_htf:
        try:
            htf = get_topdown_bias(symbol)
        except Exception as exc:
            print(f"[unified] topdown failed for {symbol}: {exc!r}")
            htf = {}

    hdir = _safe_dir(htf.get("direction") or htf.get("bias_4h") or htf.get("bias"))

    try:
        tl_raw = strategies.run_trendline_analysis(symbol, tf_code=timeframe, topdown=htf or None, df=df)
    except Exception as exc:
        print(f"[unified] trendline engine failed for {symbol}: {exc!r}")
        tl_raw = {"error": str(exc), "direction": "NEUTRAL"}
    tl = _extract_trendline_intel(tl_raw)

    try:
        smc_raw = smc_strategy.run_smc_analysis(symbol, tf_code=timeframe, topdown=htf or None, df=df)
    except Exception as exc:
        print(f"[unified] smc engine failed for {symbol}: {exc!r}")
        smc_raw = {"error": str(exc), "bias": "NEUTRAL"}
    smc = _extract_smc_intel(smc_raw)

    try:
        ote_raw = strategies.run_ote_analysis(symbol, df=df)
    except Exception as exc:
        print(f"[unified] ote engine failed for {symbol}: {exc!r}")
        ote_raw = {"error": str(exc), "direction": "NEUTRAL"}
    ote = _extract_ote_intel(ote_raw)

    state = alligator_state(df)

    try:
        fundamental = fundamental_analysis.analyze(symbol)
    except Exception as exc:
        print(f"[unified] fundamental analysis failed for {symbol}: {exc!r}")
        fundamental = {
            "symbol": symbol,
            "available": False,
            "bias": "NEUTRAL",
            "score": 0,
            "confidence": "LOW",
            "reasons": ["Fundamental engine unavailable"],
        }

    # --- Weighted confidence per source, instead of one-vote-each -----
    # Each source contributes a 0-100 confidence to whichever direction
    # it's actually pointing at. Direction is picked by summed weight,
    # not by how many sources merely agree, so one strong signal can
    # (correctly) outweigh two weak/ambiguous ones.
    alligator_conf = {
        "AWAKENING_BULLISH": 70, "AWAKENING_BEARISH": 70,
        "BULLISH": 55, "BEARISH": 55,
    }.get(state["state"], 0)

    smc_flags = [
        bool(smc.get("entry_ready")),
        bool(smc.get("sweep")),
        bool(smc.get("zone", {}).get("confluence")),
        bool(smc.get("price_in_zone")),
    ]
    smc_conf = 0 if smc["direction"] == "NEUTRAL" else min(100, 30 + sum(smc_flags) * 18)

    weights: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0}
    sources = (
        (state["direction"], alligator_conf),
        (tl["direction"], tl.get("quality") or 0),
        (smc["direction"], smc_conf),
        (ote["direction"], ote.get("quality") or 0),
    )
    for d, conf in sources:
        if d in weights:
            weights[d] += conf

    dirs: List[str] = [d for d, _ in sources if d in ("BUY", "SELL")]
    total_weight = weights["BUY"] + weights["SELL"]
    if total_weight <= 0:
        dominant = "NEUTRAL"
    else:
        dominant = "BUY" if weights["BUY"] >= weights["SELL"] else "SELL"
    # Conflict = a real signal on both sides, and the losing side isn't
    # negligible (avoids flagging conflict when one source is at conf=2
    # against another at conf=80).
    margin = abs(weights["BUY"] - weights["SELL"])
    conflict = weights["BUY"] > 0 and weights["SELL"] > 0 and margin < 0.6 * total_weight

    evidence: List[str] = []
    evidence_sources: set = set()  # independent engines actually backing `dominant`

    if state["direction"] == dominant and dominant in ("BUY", "SELL"):
        evidence.append(f"Alligator {state['state']} aligned")
        evidence_sources.add("alligator")

    if tl["direction"] == dominant and tl["quality"] >= 50:
        evidence.append(f"Trendline geometry aligned ({tl['quality']}%)")
        evidence_sources.add("trendline")
    if tl.get("confirmed") and tl["direction"] == dominant:
        evidence.append("Trendline entry confirmed")
        evidence_sources.add("trendline")
    if tl.get("event") in ("BREAK_RETEST_CONFIRMED", "BREAKOUT") and tl["direction"] == dominant:
        evidence.append(f"Trendline event: {tl['event']}")
        evidence_sources.add("trendline")
    if tl.get("active_setup") and tl["active_setup"] not in ("NONE", "TRENDLINE"):
        evidence.append(f"Trendline best setup: {tl['active_setup']}")
        evidence_sources.add("trendline")

    if smc["direction"] == dominant:
        evidence.append("SMC structure aligned")
        evidence_sources.add("smc")
    if smc.get("sweep"):
        evidence.append("Liquidity sweep present")
        evidence_sources.add("smc")
    if smc.get("zone", {}).get("confluence"):
        evidence.append("OB/FVG confluence zone")
        evidence_sources.add("smc")
    if smc.get("entry_ready") and smc["direction"] == dominant:
        evidence.append("SMC entry ready (zone + candle)")
        evidence_sources.add("smc")

    if ote["direction"] == dominant and ote.get("location") == "DEEP_RETRACEMENT":
        evidence.append("Price inside 62–79% OTE zone")
        evidence_sources.add("ote")
    if ote.get("valid") and ote["direction"] == dominant:
        evidence.append("OTE setup confirmed")
        evidence_sources.add("ote")
    if ote.get("quality", 0) >= 60 and ote["direction"] == dominant:
        evidence.append(f"OTE quality {ote['quality']}%")
        evidence_sources.add("ote")

    if hdir == dominant and dominant in ("BUY", "SELL"):
        evidence.append("Higher-timeframe context aligned")
        evidence_sources.add("htf")
    if hdir in ("BUY", "SELL") and dominant in ("BUY", "SELL") and hdir != dominant:
        conflict = True
        evidence.append(f"HTF conflict ({hdir})")

    fbias = str(fundamental.get("bias", "NEUTRAL")).upper()
    if fbias in ("BULLISH", "BEARISH") and dominant in ("BUY", "SELL"):
        fdir = "BUY" if fbias == "BULLISH" else "SELL"
        if fdir == dominant:
            evidence.append(f"Fundamental bias aligned ({fundamental.get('score', 0):+d})")
            evidence_sources.add("fundamental")
        else:
            conflict = True
            evidence.append(f"Fundamental conflict ({fundamental.get('score', 0):+d})")

    event_ok = bool(
        smc.get("sweep")
        or smc.get("zone", {}).get("confluence")
        or smc.get("entry_ready")
        or tl.get("confirmed")
        or tl.get("event") in ("BREAK_RETEST_CONFIRMED", "BREAKOUT")
        or ote.get("valid")
    )

    location_ok = bool(
        ote.get("location") in ("DEEP_RETRACEMENT", "WAITING")
        or smc.get("price_in_zone")
        or tl.get("event") in ("SUPPORT_HOLD", "RESISTANCE_HOLD", "BREAK_RETEST_CONFIRMED")
        or tl.get("confirmed")
    )

    fundamental_ok = (
        (not fundamental.get("available"))
        or fbias == "NEUTRAL"
        or (
            (fbias == "BULLISH" and dominant == "BUY")
            or (fbias == "BEARISH" and dominant == "SELL")
        )
    )

    # Gate on independent engines agreeing, not on raw evidence-line count:
    # one strong trendline setup used to be able to add 3-4 lines to
    # `evidence` on its own and clear this bar by itself. Now at least
    # two of {alligator, trendline, smc, ote, htf, fundamental} have to
    # actually back the direction.
    ready = (
        dominant in ("BUY", "SELL")
        and not conflict
        and fundamental_ok
        and state["state"] not in ("SLEEPING", "TRANSITION", "UNKNOWN")
        and event_ok
        and location_ok
        and len(evidence_sources) >= 2
    )

    tech_score = min(100, 30 + len(evidence_sources) * 14)
    if tl.get("quality"):
        tech_score = min(
            100,
            int(round(tech_score * 0.55 + tl["quality"] * 0.25 + (ote.get("quality") or 0) * 0.20)),
        )
    if fundamental.get("available"):
        fscore = abs(int(fundamental.get("score", 0)))
        score = min(100, int(round(tech_score * 0.70 + fscore * 0.30)))
    else:
        score = tech_score

    ticket = None
    if ready:
        if ote.get("valid") and ote.get("ticket"):
            ticket = ote["ticket"]
        elif smc.get("entry_ready") and smc.get("entry") is not None:
            ticket = {
                "entry": smc.get("entry"),
                "sl": smc.get("sl"),
                "tp1": smc.get("tp1"),
                "tp2": smc.get("tp2"),
                "direction": dominant,
                "order_type": "MARKET",
            }
        else:
            # Trendline-only setups: tl["position"] was never actually
            # populated by run_trendline_analysis (only computed on the fly
            # inside format_trendline_report for display) -- build it the
            # same way here so a pure-trendline confluence can still
            # produce a real ticket instead of `ready=True` with nothing
            # to trade.
            try:
                pos = strategies.build_position_container(tl_raw) if tl_raw and not tl_raw.get("error") else None
            except Exception as exc:
                print(f"[unified] build_position_container failed for {symbol}: {exc!r}")
                pos = None
            if pos and pos.get("confirmed") and pos.get("entry") is not None:
                pos.setdefault("order_type", "MARKET")
                ticket = pos

    result = {
        "strategy": STRATEGY_NAME,
        "policy": POLICY,
        "symbol": symbol,
        "timeframe": timeframe,
        "decision": dominant if ready else "WAIT",
        "direction": dominant,
        "ready": ready,
        "conflict": conflict,
        "fundamental_ok": fundamental_ok,
        "evidence": evidence,
        "evidence_sources": sorted(evidence_sources),
        "weights": {"BUY": round(weights["BUY"], 1), "SELL": round(weights["SELL"], 1)},
        "alligator": state,
        "trendline_intelligence": tl,
        "smc_intelligence": smc,
        "ote_intelligence": ote,
        "htf": htf,
        "fundamental": fundamental,
        "ticket": ticket,
        "df": df,
        "score": score,
        "reason": "; ".join(evidence) if evidence else "No coherent market-state sequence",
    }
    _log_signal(result)
    return result


def format_report(r: Dict[str, Any]) -> str:
    if r.get("error"):
        return f"{STRATEGY_NAME} — {r.get('symbol', '?')}\n\nWAIT\n{r['error']}"

    a = r.get("alligator") or {}
    s = r.get("smc_intelligence") or {}
    o = r.get("ote_intelligence") or {}
    t = r.get("trendline_intelligence") or {}
    h = r.get("htf") or {}
    f = r.get("fundamental") or {}

    structure_dir = _safe_dir(
        (s.get("structure") or {}).get("bias")
        or (s.get("structure") or {}).get("direction")
        or s.get("direction")
    )

    lines = [
        "════════════════════════════",
        "🧠 UNIFIED MARKET INTELLIGENCE",
        "════════════════════════════",
        f"{r.get('symbol', '?')} | {r.get('timeframe', '')}",
        f"DECISION: {r.get('decision', 'WAIT')}",
        f"STATE: {a.get('state', 'UNKNOWN')}",
        f"ALLIGATOR: {a.get('direction', 'NEUTRAL')}",
        f"TRENDLINE: {t.get('direction', 'NEUTRAL')} ({t.get('quality', 0)}%) | {t.get('event', 'NONE')}",
        f"STRUCTURE: {structure_dir}",
        f"LIQUIDITY: {'SWEPT' if s.get('sweep') else 'NO CONFIRMED SWEEP'}",
        f"SMC ZONE: {s.get('zone', {}).get('status', 'NONE')}"
        + (" + confluence" if s.get("zone", {}).get("confluence") else ""),
        f"OTE: {o.get('location', 'UNKNOWN')} | {o.get('zone_state', '—')} ({o.get('quality', 0)}%)",
        f"HTF: {h.get('direction') or h.get('bias_4h') or h.get('bias') or 'NEUTRAL'}",
        f"FUNDAMENTAL: {f.get('bias', 'NEUTRAL')} ({f.get('score', 0):+d}) | {f.get('confidence', 'LOW')}",
        f"SCORE: {r.get('score', 0)}/100",
    ]

    if r.get("evidence"):
        lines += ["", "INTELLIGENCE:"] + [f"• {x}" for x in r["evidence"][:10]]

    if f.get("reasons"):
        lines += ["", "FUNDAMENTAL DRIVERS:"] + [f"• {x}" for x in f["reasons"][:5]]

    if not r.get("ready"):
        notes = []
        if t.get("gating_notes"):
            notes.extend(t["gating_notes"][:2])
        if o.get("reasons"):
            notes.extend([x for x in o["reasons"] if "not" in x.lower() or "wait" in x.lower()][:2])
        if notes:
            lines += ["", "ENGINE NOTES:"] + [f"• {n}" for n in notes[:4]]

    ticket = r.get("ticket")
    if r.get("ready") and ticket and ticket.get("entry") is not None:
        lines += [
            "",
            "🎯 TRADE MODEL",
            f"ENTRY: {ticket.get('entry')}",
            f"SL: {ticket.get('sl')}",
            f"TP1: {ticket.get('tp1')}",
            f"TP2: {ticket.get('tp2')}",
            f"ORDER: {ticket.get('order_type', 'MARKET')}",
        ]

    lines += [
        "",
        f"WHY: {r.get('reason', 'No coherent market-state sequence')}",
        "",
        "Trendline / SMC / OTE / Fundamentals are internal intelligence sources — not separate strategies.",
    ]
    return "\n".join(lines)
