"""
desk_engine.py
==============
Institutional Desk Decision Engine — single strict decision system.

Philosophy:
  The bot is no longer a collection of independent strategies that can
  fire on their own. It becomes one disciplined desk process:

  1. Higher-timeframe bias must be clear
  2. Structure (ISE) must confirm direction + acceptance
  3. Price must be in a high-quality entry zone (OTE Fan / deep Fib)
  4. Liquidity context must not be mid-range chase
  5. Minimum risk/reward
  6. Only then → VALID setup with ticket

  All other modules (SMC, AMD, Trendline, Silver Bullet, OTE) become
  confluence sources that add or subtract from the desk score.
  They do not generate independent fire signals under DESK mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

import mt5_data
import trade_state as ts
from structure_engine import run_structure_engine
from ote_strategy import run_ote_analysis
from market_structure import analyse_structure


# ---------------------------------------------------------------------------
# Scoring constants — deliberately strict
# ---------------------------------------------------------------------------
MIN_DESK_SCORE = 72          # was effectively ~55 in old strategies
MIN_RR = 1.5
REQUIRE_HTF_AGREE = True
REQUIRE_STRUCTURE = True
REQUIRE_ENTRY_ZONE = True


def _safe_fetch(symbol: str, tf: str, count: int = 220) -> Optional[pd.DataFrame]:
    try:
        df = mt5_data.fetch_candles(symbol, tf, count=count)
        if df is not None and not df.empty and len(df) >= 40:
            return df
    except Exception:
        pass
    return None


def _htf_gate(symbol: str) -> Dict[str, Any]:
    """
    Higher-timeframe bias gate using structure analysis on H1 and H4.
    """
    notes = []
    biases = []

    for tf, label in (("1h", "H1"), ("4h", "H4")):
        try:
            hdf = _safe_fetch(symbol, tf, count=150)
            if hdf is None or len(hdf) < 40:
                notes.append(f"{label}: no data")
                continue
            st = analyse_structure(hdf, left=2, right=2, lookback=50)
            bias = str(st.get("bias") or st.get("direction") or "NEUTRAL").upper()
            note = st.get("note") or st.get("bias") or ""
            if bias in ("BUY", "SELL", "BULLISH", "BEARISH"):
                clean = "BUY" if bias in ("BUY", "BULLISH") else "SELL"
                biases.append(clean)
                notes.append(f"{label}: {clean} — {note}")
            else:
                notes.append(f"{label}: neutral / unclear — {note}")
        except Exception as e:
            notes.append(f"{label}: unavailable ({e})")

    direction = "NEUTRAL"
    if biases:
        if all(b == "BUY" for b in biases):
            direction = "BUY"
        elif all(b == "SELL" for b in biases):
            direction = "SELL"
        elif biases.count("BUY") > biases.count("SELL"):
            direction = "BUY"
            notes.append("HTF mixed but leaning BUY")
        elif biases.count("SELL") > biases.count("BUY"):
            direction = "SELL"
            notes.append("HTF mixed but leaning SELL")
        else:
            notes.append("HTF conflict — no clear desk bias")

    return {
        "direction": direction,
        "notes": notes,
        "aligned": direction in ("BUY", "SELL") and len(set(biases)) == 1 and len(biases) >= 2,
    }


def _structure_gate(df: pd.DataFrame) -> Dict[str, Any]:
    """Run Institutional Structure Engine and extract decision fields."""
    try:
        ise = run_structure_engine(df)
        if ise.get("error"):
            return {
                "valid": False,
                "direction": "NEUTRAL",
                "score": 0,
                "reasons": [f"ISE error: {ise['error']}"],
                "raw": ise,
            }
        return {
            "valid": bool(ise.get("valid")),
            "direction": ise.get("direction", "NEUTRAL"),
            "score": int(ise.get("score", 0)),
            "reasons": list(ise.get("reasons") or []),
            "path": (ise.get("entry") or {}).get("path"),
            "raw": ise,
        }
    except Exception as e:
        return {
            "valid": False,
            "direction": "NEUTRAL",
            "score": 0,
            "reasons": [f"ISE failed: {e}"],
            "raw": {},
        }


def _entry_zone_gate(symbol: str, timeframe: str, df: pd.DataFrame, want_dir: str) -> Dict[str, Any]:
    """
    Entry quality gate using OTE (Fib Fan + Expansion).
    Price must be interacting with the Fan in the desired direction.
    """
    try:
        ote = run_ote_analysis(symbol, timeframe=timeframe, df=df)
    except Exception as e:
        return {
            "in_zone": False,
            "quality": 0,
            "reasons": [f"OTE unavailable: {e}"],
            "ticket": None,
            "raw": {},
        }

    if ote.get("error"):
        return {
            "in_zone": False,
            "quality": 0,
            "reasons": [ote["error"]],
            "ticket": None,
            "raw": ote,
        }

    ote_dir = ote.get("direction", "NEUTRAL")
    in_zone = bool(ote.get("in_zone"))
    nearest = ote.get("nearest_fan")
    ticket = ote.get("ticket")
    reasons = list(ote.get("reasons") or [])

    quality = 0
    if in_zone and ote_dir == want_dir:
        quality += 40
        reasons.append("Price in OTE Fan zone aligned with desk direction")
        if nearest and nearest.get("ratio", 0) >= 0.50:
            quality += 20
            reasons.append(f"Deep Fan ({nearest.get('label')}) — higher quality OTE")
    elif in_zone and ote_dir != want_dir:
        reasons.append("Fan interaction exists but against desk direction — rejected")
    else:
        reasons.append("Price not interacting with Fib Fan — no entry zone")

    # R:R check from OTE ticket if present
    if ticket and ticket.get("rr", 0) >= MIN_RR:
        quality += 15
        reasons.append(f"R:R {ticket['rr']:.2f} meets minimum {MIN_RR}")
    elif ticket:
        reasons.append(f"R:R {ticket.get('rr', 0):.2f} below minimum {MIN_RR}")

    return {
        "in_zone": in_zone and ote_dir == want_dir,
        "quality": quality,
        "reasons": reasons,
        "ticket": ticket if (in_zone and ote_dir == want_dir) else None,
        "raw": ote,
        "nearest_fan": nearest,
        "expansions": ote.get("expansions") or [],
    }


def run_desk_analysis(symbol: str, timeframe: str = None) -> Dict[str, Any]:
    """
    Full Institutional Desk decision.

    Returns a single, strict verdict: VALID or WAIT, with clear reasons.
    """
    symbol = symbol.strip().upper()
    timeframe = timeframe or ts.state.get_watch_timeframe()

    df = _safe_fetch(symbol, timeframe)
    if df is None:
        df = _safe_fetch(symbol, "30min")
    if df is None:
        df = _safe_fetch(symbol, "15min")
    if df is None:
        return {
            "strategy": "DESK",
            "direction": "NEUTRAL",
            "score": 0,
            "valid": False,
            "verdict": "WAIT",
            "reasons": ["Insufficient market data"],
            "report": f"DESK  |  {symbol}\nNo usable price data.",
            "ticket": None,
        }

    reasons: List[str] = []
    score = 0
    gates_passed = 0
    gates_total = 4

    # ------------------------------------------------------------------
    # GATE 1 — Higher-timeframe bias
    # ------------------------------------------------------------------
    htf = _htf_gate(symbol)
    reasons.extend(htf["notes"])

    if htf["direction"] in ("BUY", "SELL"):
        score += 20
        gates_passed += 1
        if htf["aligned"]:
            score += 10
            reasons.append("HTF fully aligned")
    else:
        reasons.append("GATE 1 FAIL: No clear higher-timeframe bias")
        if REQUIRE_HTF_AGREE:
            return _fail(symbol, timeframe, df, score, reasons, htf=htf)

    desk_dir = htf["direction"]

    # ------------------------------------------------------------------
    # GATE 2 — Structure (ISE)
    # ------------------------------------------------------------------
    struct = _structure_gate(df)
    reasons.extend(struct["reasons"][:4])

    if struct["valid"] and struct["direction"] == desk_dir:
        score += 25
        gates_passed += 1
        score += min(15, struct["score"] // 6)
        reasons.append(f"GATE 2 PASS: Structure confirms {desk_dir} (ISE score {struct['score']})")
    elif struct["valid"] and struct["direction"] != desk_dir:
        reasons.append(f"GATE 2 FAIL: Structure wants {struct['direction']} vs HTF {desk_dir}")
        score -= 15
        if REQUIRE_STRUCTURE:
            return _fail(symbol, timeframe, df, score, reasons, htf=htf, struct=struct)
    else:
        reasons.append("GATE 2 FAIL: Structure not accepted yet")
        if REQUIRE_STRUCTURE:
            return _fail(symbol, timeframe, df, score, reasons, htf=htf, struct=struct)

    # ------------------------------------------------------------------
    # GATE 3 — Entry zone (OTE Fan)
    # ------------------------------------------------------------------
    entry = _entry_zone_gate(symbol, timeframe, df, desk_dir)
    reasons.extend(entry["reasons"][:5])

    if entry["in_zone"]:
        score += entry["quality"]
        gates_passed += 1
        reasons.append("GATE 3 PASS: High-quality OTE / Fan entry zone")
    else:
        reasons.append("GATE 3 FAIL: No valid entry zone")
        if REQUIRE_ENTRY_ZONE:
            return _fail(
                symbol, timeframe, df, score, reasons,
                htf=htf, struct=struct, entry=entry,
            )

    # ------------------------------------------------------------------
    # GATE 4 — Ticket quality (R:R + not chasing)
    # ------------------------------------------------------------------
    ticket = entry.get("ticket")
    if ticket and ticket.get("rr", 0) >= MIN_RR:
        score += 10
        gates_passed += 1
        reasons.append(f"GATE 4 PASS: Ticket R:R {ticket['rr']:.2f}")
    else:
        reasons.append("GATE 4 FAIL: Ticket quality insufficient")
        ticket = None

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    score = max(0, min(100, score))
    valid = (
        gates_passed >= 3
        and score >= MIN_DESK_SCORE
        and desk_dir in ("BUY", "SELL")
        and ticket is not None
    )

    verdict = "VALID" if valid else "WAIT"
    if valid:
        reasons.insert(0, f"✅ DESK VERDICT: {verdict} — {desk_dir}")
    else:
        reasons.insert(0, f"⏳ DESK VERDICT: {verdict} — conditions not met")

    report = _format_desk_report(
        symbol, timeframe, desk_dir, score, valid, verdict,
        reasons, htf, struct, entry, ticket,
    )

    return {
        "strategy": "DESK",
        "direction": desk_dir if valid else "NEUTRAL",
        "score": score,
        "valid": valid,
        "verdict": verdict,
        "reasons": reasons,
        "gates_passed": gates_passed,
        "gates_total": gates_total,
        "htf": htf,
        "structure": struct,
        "entry": entry,
        "position": ticket,
        "ticket": ticket,
        "analysis": {
            "df": df,
            "ote": entry.get("raw") or {},
            "ise": struct.get("raw") or {},
            "impulse": (entry.get("raw") or {}).get("impulse"),
            "fans": (entry.get("raw") or {}).get("fans") or [],
            "expansions": entry.get("expansions") or [],
            "position": ticket,
            "ticket": ticket,
            "direction": desk_dir if valid else "NEUTRAL",
            "score": score,
        },
        "df": df,
        "timeframe": timeframe,
        "symbol": symbol,
        "report": report,
    }


def _fail(
    symbol, timeframe, df, score, reasons,
    htf=None, struct=None, entry=None,
) -> Dict[str, Any]:
    score = max(0, min(100, score))
    report = _format_desk_report(
        symbol, timeframe, "NEUTRAL", score, False, "WAIT",
        reasons, htf or {}, struct or {}, entry or {}, None,
    )
    return {
        "strategy": "DESK",
        "direction": "NEUTRAL",
        "score": score,
        "valid": False,
        "verdict": "WAIT",
        "reasons": reasons,
        "htf": htf or {},
        "structure": struct or {},
        "entry": entry or {},
        "position": None,
        "ticket": None,
        "analysis": {"df": df},
        "df": df,
        "timeframe": timeframe,
        "symbol": symbol,
        "report": report,
    }


def _format_desk_report(
    symbol, timeframe, direction, score, valid, verdict,
    reasons, htf, struct, entry, ticket,
) -> str:
    lines = [
        f"🏛 INSTITUTIONAL DESK  |  {symbol}  ({timeframe})",
        f"Verdict: {'✅ VALID' if valid else '⏳ WAIT'}  |  Direction: {direction}  |  Score: {score}/100",
        "",
        "── Gates ──",
    ]
    for r in reasons:
        lines.append(f"  • {r}")

    if ticket:
        lines.append("")
        lines.append("── Ticket ──")
        lines.append(
            f"  {ticket.get('side')}  Entry {ticket.get('entry', 0):.5f}  "
            f"SL {ticket.get('sl', 0):.5f}  TP1 {ticket.get('tp1', 0):.5f}  "
            f"TP2 {ticket.get('tp2', 0):.5f}"
        )
        lines.append(f"  R:R 1:{ticket.get('rr', 0):.2f}")

    lines.append("")
    lines.append("Desk rule: only trade when HTF + Structure + Entry Zone all agree.")
    return "\n".join(lines)
