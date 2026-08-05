"""
strategy.py — strategy routing, scoring, hybrid, trade tickets, confirmation.
"""
from __future__ import annotations
"""
strategy_engine.py
==================
Routes analysis and signal generation according to:
  - SINGLE mode  → only the selected strategy
  - HYBRID mode  → all enabled strategies, pick best confluence, explain why
"""


from typing import Any, Dict, List, Optional

import runtime as ts
import data as mt5_data
from analysis import run_topdown_analysis, format_institutional_report
from analysis import run_amd_analysis, format_amd_report
from analysis import (
    run_silver_bullet_analysis,
    format_silver_bullet_report,
    build_silver_bullet_ticket,
)
# charts imported lazily where needed
import data


def _score_smc(analysis: Dict) -> Dict[str, Any]:
    """
    SMC scoring with Institutional Structure Engine gate.
    Top-down bias / zones / IDM→OB remain confluence; ISE (when runnable on
    the chart frame) supplies the same Liquidity → Manipulation → Acceptance
    permission used by Trendline and AMD.
    """
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
    chart_df = None
    if frames:
        f = frames[0]
        chart_df = f.get("df")
        n_fvg = len(f.get("fvgs") or [])
        n_ob = len(f.get("order_blocks") or [])
        if n_fvg:
            score += 10
            reasons.append(f"{n_fvg} FVG(s) present")
        if n_ob:
            score += 10
            reasons.append(f"{n_ob} Order Block(s) present")
        # Surface pattern scanner result when present
        bp = f.get("best_pattern")
        if bp is not None:
            reasons.append(f"Pattern: {bp.name} ({bp.bias}) {getattr(bp, 'confidence', 0):.0f}%")
            if getattr(bp, "bias", None) in ("BUY", "SELL") and bp.bias == direction:
                score += 5

    alignment = analysis.get("alignment", "MIXED")
    if alignment.startswith("ALIGNED"):
        score += 15
        reasons.append(f"Timeframes aligned ({alignment})")
    elif alignment.startswith("MOSTLY"):
        score += 5
        reasons.append(f"Timeframes mostly aligned ({alignment})")
    elif alignment == "MIXED":
        score -= 10
        reasons.append("Timeframes MIXED — conflicting bias across the ladder")

    allowed = analysis.get("structure_allowed")
    structure_reason = analysis.get("structure_reason", "")
    if allowed:
        score += 15
        reasons.append(f"Structure permission granted — {structure_reason}")
    else:
        score -= 20
        reasons.append(f"Structure permission WITHHELD — {structure_reason}")

    pairs = analysis.get("idm_ob_pairs") or []
    matching_pair = next((p for p in pairs if p.get("direction") == direction), None)
    if matching_pair:
        score += 10
        reasons.append("Clean IDM→OB pair aligned with direction")
    elif pairs:
        reasons.append("IDM→OB pair exists but doesn't match current direction")

    # ISE gate on the primary chart frame (dynamic structure permission)
    ise_valid = None
    ise_dir = None
    if chart_df is not None and len(chart_df) >= 60:
        try:
            from structure_engine import run_structure_engine
            ise = run_structure_engine(chart_df)
            analysis["ise"] = ise
            if not ise.get("error"):
                ise_dir = ise.get("direction")
                ise_valid = bool(ise.get("valid"))
                if ise_valid and ise_dir in ("BUY", "SELL"):
                    if direction == "NEUTRAL" or direction == ise_dir:
                        direction = ise_dir
                        score = max(score, int(ise.get("score", score)))
                        reasons.append(
                            f"ISE confirmed {ise_dir} (score {ise.get('score', 0)}, "
                            f"path={ (ise.get('entry') or {}).get('path') })"
                        )
                    elif direction != ise_dir:
                        score -= 15
                        reasons.append(
                            f"ISE conflicts with HTF bias ({ise_dir} vs {direction}) — reduced conviction"
                        )
                elif not ise_valid:
                    reasons.append("ISE verdict: WAIT — structure not fully accepted yet")
                    # Soft penalty; still allow high-confluence SMC if structure_allowed
                    score -= 5
        except Exception as e:
            reasons.append(f"ISE skip: {e}")

    valid = direction in ("BUY", "SELL") and score >= 55 and bool(allowed)
    # If ISE ran and explicitly rejected, do not override a hard WAIT unless
    # structure_allowed + IDM→OB + aligned HTF are all present (high confluence).
    if ise_valid is False and not (allowed and matching_pair and alignment.startswith("ALIGNED")):
        valid = False

    final_dir = direction if (valid and direction in ("BUY", "SELL")) else "NEUTRAL"
    return {
        "strategy": ts.STRATEGY_SMC,
        "direction": final_dir,
        "score": max(0, min(100, score)),
        "reasons": reasons,
        "analysis": analysis,
        "valid": bool(valid and final_dir in ("BUY", "SELL")),
    }


def _score_amd(analysis: Dict) -> Dict[str, Any]:
    """
    AMD scoring is now gated by the Institutional Structure Engine.
    Phase language is kept for the report; direction / valid / score prefer
    the ISE verdict (same pipeline as Trendline).
    """
    if "error" in analysis:
        return {
            "strategy": ts.STRATEGY_AMD,
            "direction": "NEUTRAL",
            "score": 0,
            "reasons": [analysis["error"]],
            "analysis": analysis,
            "valid": False,
        }

    ise = analysis.get("ise") or {}
    phase = str(analysis.get("phase", "")).upper()
    bias = str(analysis.get("amd_bias", "NEUTRAL")).upper()
    reasons: List[str] = [f"Phase: {phase}"]

    # Prefer ISE as decision authority when available
    if ise and not ise.get("error"):
        direction = ise.get("direction", "NEUTRAL")
        score = int(ise.get("score", 0))
        reasons.extend(list(ise.get("reasons") or [])[:6])
        # Soft phase bonus so AMD narrative still influences ranking in hybrid
        if phase in ("DISPLACEMENT", "CONTINUATION"):
            score = min(100, score + 5)
            reasons.append("AMD phase supports directional expansion")
        elif phase == "REVERSION":
            score = min(100, score + 3)
            reasons.append("AMD reversion zone aligned with structure")
        valid = bool(ise.get("valid")) and direction in ("BUY", "SELL")
        return {
            "strategy": ts.STRATEGY_AMD,
            "direction": direction if valid else "NEUTRAL",
            "score": max(0, min(100, score)),
            "reasons": reasons,
            "analysis": analysis,
            "valid": valid,
        }

    # Fallback (ISE unavailable): legacy phase scoring
    score = 35
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


def _score_trendline(symbol: str, timeframe: str = None) -> Dict[str, Any]:
    """Full trendline family score with projections + position container."""
    # Default to the user's chosen Entry Timeframe (Mobile Control Panel)
    # instead of a hardcoded value, so on-demand Trendline calls match
    # whatever the background scanner is also using.
    timeframe = timeframe or ts.state.get_watch_timeframe()
    try:
        from analysis import (
            build_trendline_family,
            build_position_container,
            format_trendline_report,
        )
        df = mt5_data.fetch_candles(symbol, timeframe, count=220)
        if df is None or df.empty or len(df) < 40:
            df = mt5_data.fetch_candles(symbol, "30min", count=220)
        if df is None or df.empty or len(df) < 40:
            df = mt5_data.fetch_candles(symbol, "15min", count=220)
        if df is None or df.empty or len(df) < 40:
            return {
                "strategy": ts.STRATEGY_TRENDLINE,
                "direction": "NEUTRAL",
                "score": 0,
                "reasons": ["Insufficient data"],
                "analysis": {},
                "valid": False,
            }
        family = build_trendline_family(df, max_lines=2)
        if family.get("error"):
            return {
                "strategy": ts.STRATEGY_TRENDLINE,
                "direction": "NEUTRAL",
                "score": 0,
                "reasons": [family["error"]],
                "analysis": {},
                "valid": False,
            }

        # --- Institutional Structure Engine is now the decision authority ---
        # Price -> Structure -> Liquidity -> Manipulation -> Acceptance -> Trade,
        # not raw channel geometry. trendline_family() above still supplies the
        # chart payload (rails/channel/POC/projections/liquidity targets) and
        # its own reasons are kept in the report as supporting confluence, but
        # direction/valid/score now come from the structure engine -- a
        # channel bounce or "break" that the ISE hasn't confirmed through
        # Liquidity -> Manipulation -> Acceptance no longer fires a ticket.
        from structure_engine import run_structure_engine, format_structure_report
        ise = run_structure_engine(df)

        if ise.get("error"):
            # Fall back to geometry-only read if the ISE can't run (e.g. too
            # little data) rather than blocking the strategy entirely.
            direction = family.get("direction", "NEUTRAL")
            score = int(family.get("strength", 0))
            reasons = family.get("reasons") or []
            reasons.append(f"(ISE unavailable: {ise['error']} — using channel-geometry read only)")
            valid = direction in ("BUY", "SELL") and score >= 55
        else:
            direction = ise["direction"]
            score = int(ise.get("score", 0))
            reasons = list(ise.get("reasons") or [])
            valid = bool(ise.get("valid"))
            family["ise"] = ise
            # Override the geometry-only direction so build_position_container
            # (SL from structure, TP from liquidity) builds its ticket off the
            # ISE-confirmed direction, not the raw channel-position read.
            family["direction"] = direction if direction in ("BUY", "SELL") else family.get("direction", "NEUTRAL")

        pos = build_position_container(family) if direction in ("BUY", "SELL") else None
        family["position"] = pos
        # HTF context (details only — final map stays M30)
        htf_notes = []
        try:
            from structure_engine import analyse_structure
            for tf, lab in (("1h", "H1"), ("4h", "H4")):
                hdf = mt5_data.fetch_candles(symbol, tf, count=120)
                if hdf is not None and len(hdf) >= 40:
                    st = analyse_structure(hdf, left=2, right=2, lookback=50)
                    htf_notes.append(f"{lab}: {st.get('note', st.get('bias', ''))}")
        except Exception:
            pass
        family["htf_notes"] = htf_notes
        family["timeframe"] = timeframe
        report = format_trendline_report(family, symbol)
        if not ise.get("error"):
            report = format_structure_report(ise, symbol) + "\n\n" + report
        if htf_notes:
            report += "\nHTF context:\n" + "\n".join(f"  • {n}" for n in htf_notes)
        return {
            "strategy": ts.STRATEGY_TRENDLINE,
            "direction": direction,
            "score": score,
            "reasons": reasons,
            "analysis": family,
            "family": family,
            "position": pos,
            "ticket": pos,
            "report": report,
            "valid": valid,
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



def _attach_projections(result: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """Add measured-move projections + optional position for any strategy chart."""
    try:
        from analysis import build_trendline_family, build_position_container
        analysis = result.get("analysis") or {}
        df = None
        if isinstance(analysis, dict):
            df = analysis.get("df_1h") or analysis.get("df")
            if df is None:
                frames = analysis.get("frames") or []
                if frames:
                    df = frames[0].get("df")
        if df is None or getattr(df, "empty", True):
            return result
        family = build_trendline_family(df, max_lines=3)
        projs = family.get("projections") or []
        # Only keep projections aligned with strategy direction
        direction = result.get("direction", "NEUTRAL")
        if direction in ("BUY", "SELL"):
            projs = [p for p in projs if p.get("side") == direction] or projs
        result["projections"] = projs
        if isinstance(analysis, dict):
            analysis["projections"] = projs
            analysis["swings"] = family.get("pivots")
            if not analysis.get("volume_profile"):
                analysis["volume_profile"] = family.get("volume_profile")
        # Position if valid setup
        if result.get("valid") and direction in ("BUY", "SELL"):
            family["direction"] = direction
            pos = build_position_container(family)
            result["position"] = pos
            if isinstance(analysis, dict):
                analysis["position"] = pos
            if not result.get("ticket") and pos:
                result["ticket"] = pos
        result["analysis"] = analysis
    except Exception:
        pass
    return result

def run_single_strategy(symbol: str, strategy: Optional[str] = None) -> Dict[str, Any]:
    strategy = strategy or ts.state.get_selected_strategy()
    symbol = symbol.strip().upper()

    if strategy == ts.STRATEGY_SMC:
        analysis = run_topdown_analysis(symbol)
        result = _score_smc(analysis)
        result["report"] = format_institutional_report(analysis)
        return _attach_projections(result, symbol)

    if strategy == ts.STRATEGY_AMD:
        analysis = run_amd_analysis(symbol)
        result = _score_amd(analysis)
        result["report"] = format_amd_report(analysis)
        return _attach_projections(result, symbol)

    if strategy == ts.STRATEGY_SILVER_BULLET:
        analysis = run_silver_bullet_analysis(symbol)
        result = _score_silver_bullet(analysis)
        result["report"] = format_silver_bullet_report(analysis)
        result["ticket"] = build_silver_bullet_ticket(analysis)
        return _attach_projections(result, symbol)

    if strategy == ts.STRATEGY_TRENDLINE:
        result = _score_trendline(symbol)
        if not result.get("report"):
            result["report"] = (
                f"TRENDLINE {symbol}\n"
                f"Direction: {result['direction']} | Score: {result['score']}\n"
                + "\n".join(f"  • {r}" for r in result.get("reasons") or [])
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
"""
trade_setup.py
================
Takes a confirmation_engine fire decision + the detected pattern and computes
the final entry/SL/TP1/TP2 numbers. Kept separate from confirmation_engine.py
so "when do we act" and "what are the numbers" stay independently testable.

Rules carried over from the earlier design:
  - SL is structural: bound to the pattern's own footprint (its trigger_line
    span), not an arbitrary fixed lookback -- avoids grabbing a flagpole's
    origin and producing an oversized stop.
  - TP for flags/pennants uses the measured-move (flagpole height), ANCHORED
    TO THE TRIGGER PRICE (not the actual fill price) -- so a stretched/Fib
    entry doesn't get an inflated target just because the fill was late.
  - TP for everything else uses a 1.5R / 3R risk-multiple off the SL distance.
"""


def _pattern_atr(df):
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        return float(df['ATR'].iloc[-1])
    return float((df['High'] - df['Low']).tail(14).mean())


def build_trade_setup(df, pattern, fire_decision):
    """
    df: OHLC dataframe used for the scan (with ATR column).
    pattern: the DetectedPattern that was confirmed.
    fire_decision: dict from ConfirmationEngine.step() with action in
                   {"FIRE_MARKET","FIRE_LIMIT"}.

    Returns dict: entry, order_type, sl, tp1, tp2, trigger_price, bias, pattern_name
    """
    entry = fire_decision["fire_price"]
    order_type = fire_decision["order_type"]
    bias = pattern.bias
    trigger = pattern.trigger_price
    atr = _pattern_atr(df)

    span_xs = [p[0] for p in (pattern.trigger_line or [])]
    n = len(df)
    if span_xs:
        window_start = max(0, min(span_xs) - 3)
    else:
        window_start = max(0, n - 60)
    local_window = df.iloc[window_start:]
    resistance_level = float(local_window['High'].max())
    support_level = float(local_window['Low'].min())

    if bias == "BUY":
        sl = min(entry, support_level) - atr * 0.5
        risk = max(abs(entry - sl), atr * 0.25)
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 3.0
    else:
        sl = max(entry, resistance_level) + atr * 0.5
        risk = max(abs(sl - entry), atr * 0.25)
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 3.0

    if "Flag" in pattern.name or "Pennant" in pattern.name:
        pole_pts = [p for p in (pattern.key_points or []) if "Pole" in p[2]]
        if len(pole_pts) >= 2:
            pole_height = abs(pole_pts[1][1] - pole_pts[0][1])
            if pole_height > 0:
                # Anchored to the TRIGGER, not the fill price.
                if bias == "BUY":
                    tp1 = trigger + pole_height * 0.618
                    tp2 = trigger + pole_height * 1.0
                else:
                    tp1 = trigger - pole_height * 0.618
                    tp2 = trigger - pole_height * 1.0

    return {
        "entry": entry, "order_type": order_type, "bias": bias,
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "trigger_price": trigger, "pattern_name": pattern.name,
        "category": pattern.category, "confidence": pattern.confidence,
        "note": pattern.note, "expiry_bars": fire_decision.get("expiry_bars"),
    }
"""
confirmation_engine.py
================
Turns a detected chart pattern into an actual "fire or wait" decision.

Rule (agreed design):
  1. A pattern only fires on a MARUBOZU candle closing beyond the trigger
     (body >= 70% of range, reversal-side wick < 15% of range, range >= 0.8x ATR).
  2. If that marubozu closes within 2x ATR of the trigger -> fire at market,
     immediately.
  3. If it's already stretched beyond 2x ATR -> don't chase. Compute a
     Fibonacci discount/premium zone (50%-79% retracement) on the trigger->
     extreme leg, and fire a LIMIT order at the 61.8% anchor within that
     zone, with a 15-bar expiry.
  4. If no marubozu appears within 20 bars of a pattern becoming valid, the
     watch is abandoned (stale).

This module only decides WHEN and AT WHAT PRICE/ORDER-TYPE to act. SL/TP
math and what to actually DO with that decision (auto-fire / ask approval /
send a manual mobile ticket) live in the caller (bot.py's trade layer).

Known simplification: if price reverses hard against a still-WATCHING
pattern without a scan ever confirming it, the watch is cleared naturally
next time scan_all_patterns() stops returning that pattern (structure has
changed), rather than via an explicit "opposite invalidation" check here.
"""

STALE_BARS = 20
FIB_WAIT_BARS = 15
MARUBOZU_BODY_RATIO = 0.70
MARUBOZU_WICK_RATIO = 0.15
MARUBOZU_ATR_RATIO = 0.8
FAR_ATR_MULTIPLE = 2.0
FIB_ZONE_LOW = 0.50
FIB_ZONE_HIGH = 0.79
FIB_ENTRY_ANCHOR = 0.618

from analysis import detect_confirmation_candle


def is_marubozu(o, h, l, c, atr):
    rng = h - l
    if rng <= 0 or atr is None or atr <= 0:
        return False
    if rng < MARUBOZU_ATR_RATIO * atr:
        return False
    body = abs(c - o)
    if body / rng < MARUBOZU_BODY_RATIO:
        return False
    reversal_wick = (h - c) if c >= o else (c - l)
    if reversal_wick / rng > MARUBOZU_WICK_RATIO:
        return False
    return True


def fib_discount_premium_zone(trigger_price, extreme_price, bias):
    """
    Returns (zone_low, zone_high, entry_anchor_price) for the pullback zone
    on the trigger->extreme breakout leg.
    """
    if bias == "BUY":
        leg = extreme_price - trigger_price
        zone_low = extreme_price - leg * FIB_ZONE_HIGH
        zone_high = extreme_price - leg * FIB_ZONE_LOW
        entry = extreme_price - leg * FIB_ENTRY_ANCHOR
    else:
        leg = trigger_price - extreme_price
        zone_high = extreme_price + leg * FIB_ZONE_HIGH
        zone_low = extreme_price + leg * FIB_ZONE_LOW
        entry = extreme_price + leg * FIB_ENTRY_ANCHOR
    return zone_low, zone_high, entry


def check_current_confirmation(df, trigger_price, bias):
    """
    One-off check of the LATEST candle against a trigger -- used by the
    Telegram informational display, which doesn't need the stateful
    bars_watched/stale-timeout tracking the live polling engine uses. Just
    answers: "as of right now, is this confirmed, and by what?"

    Returns (confirmed: bool, confirmation_type: str or None).
    """
    if len(df) < 3:
        return False, None
    o = float(df['Open'].iloc[-1]); h = float(df['High'].iloc[-1])
    l = float(df['Low'].iloc[-1]);  c = float(df['Close'].iloc[-1])
    atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else None

    broke = (c > trigger_price) if bias == "BUY" else (c < trigger_price)
    if not broke:
        return False, None

    if is_marubozu(o, h, l, c, atr):
        return True, "Marubozu"

    candle_confirmed, candle_name = detect_confirmation_candle(df, bias)
    if candle_confirmed:
        return True, candle_name

    return False, None


class ConfirmationEngine:
    """Holds per (symbol, timeframe) watch state across successive polls."""

    def __init__(self):
        self._watches = {}  # (symbol, tf) -> dict

    def reset(self, symbol, tf):
        self._watches.pop((symbol, tf), None)

    def step(self, symbol, tf, df, best_pattern):
        """
        df: cleaned OHLC dataframe (with 'ATR' column), chronological, latest
            bar last.
        best_pattern: the top result from patterns.scan_all_patterns(df), or
            None if nothing currently qualifies.

        Returns a dict:
          {"action": "NONE"|"FIRE_MARKET"|"FIRE_LIMIT",
           "pattern": DetectedPattern or None,
           "fire_price": float or None,
           "order_type": "MARKET"|"LIMIT"|None,
           "expiry_bars": int or None,
           "reason": str}
        """
        key = (symbol, tf)

        if best_pattern is None:
            self._watches.pop(key, None)
            return {"action": "NONE", "pattern": None, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "no_pattern"}

        watch = self._watches.get(key)
        if watch is None or watch["pattern_name"] != best_pattern.name or watch["bias"] != best_pattern.bias:
            watch = {"pattern_name": best_pattern.name, "bias": best_pattern.bias,
                      "trigger_price": best_pattern.trigger_price, "bars_watched": 0, "state": "WATCHING"}
            self._watches[key] = watch
        else:
            watch["trigger_price"] = best_pattern.trigger_price  # keep fresh for sloped necklines (H&S)

        if watch["state"] != "WATCHING":
            return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "already_resolved"}

        watch["bars_watched"] += 1
        if watch["bars_watched"] > STALE_BARS:
            self._watches.pop(key, None)
            return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "stale_pattern_timeout"}

        o = float(df['Open'].iloc[-1]); h = float(df['High'].iloc[-1])
        l = float(df['Low'].iloc[-1]);  c = float(df['Close'].iloc[-1])
        atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else None
        trigger = watch["trigger_price"]
        bias = watch["bias"]

        broke = (c > trigger) if bias == "BUY" else (c < trigger)
        if not broke:
            return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "not_broken_yet"}

        if not is_marubozu(o, h, l, c, atr):
            candle_confirmed, candle_name = detect_confirmation_candle(df, bias)
            if not candle_confirmed:
                return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                        "order_type": None, "expiry_bars": None, "reason": "broke_but_not_confirmed"}
            confirmation_label = candle_name
        else:
            confirmation_label = "Marubozu"

        # Confirmed (either a marubozu close or a qualifying candlestick pattern) -- resolve this watch (one-shot fire).
        watch["state"] = "DONE"
        distance = abs(c - trigger)

        if atr and distance <= FAR_ATR_MULTIPLE * atr:
            return {"action": "FIRE_MARKET", "pattern": best_pattern, "fire_price": c,
                    "order_type": "MARKET", "expiry_bars": None,
                    "reason": f"{confirmation_label}_confirmed_near_trigger"}

        extreme = h if bias == "BUY" else l
        _, _, entry_anchor = fib_discount_premium_zone(trigger, extreme, bias)
        return {"action": "FIRE_LIMIT", "pattern": best_pattern, "fire_price": entry_anchor,
                "order_type": "LIMIT", "expiry_bars": FIB_WAIT_BARS,
                "reason": f"{confirmation_label}_confirmed_stretched_fib_pullback"}
