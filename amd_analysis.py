"""
amd_analysis.py
===============
AMD = Accumulation → Manipulation → Distribution (ICT / power-of-three style).

NOW DRIVEN BY the Institutional Structure Engine (ISE):

    Price → Structure → Liquidity → Manipulation → Acceptance → Trade

The legacy range/phase segmenter is kept only for chart shading and session
context. Direction, validity, phase identity, and trade permission come from
structure_engine.run_structure_engine() — the same pipeline already used by
the Trendline strategy — so AMD is fully dynamic and consistent with the rest
of the system.

Primary chart: **1 Hour**
Context: 4H bias → 1H ISE/AMD → 30M / 15M for entry refinement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from market_structure import analyse_structure
from smc_zones import (
    detect_fvgs, detect_order_blocks, detect_inducement_zones,
    pair_idm_with_extreme_ob, build_bos_events,
)
from volume_profile import compute_volume_profile
from structure_engine import run_structure_engine, format_structure_report
import mt5_data


# UTC hour ranges (inclusive start, exclusive end-style checks use hour)
SESSION_WINDOWS = {
    "Asian": (0, 8),
    "London": (7, 16),
    "NewYork": (12, 21),
}


def _session_for_ts(ts):
    """Return session name(s) for a pandas Timestamp (assumes UTC-like index)."""
    try:
        h = ts.hour
    except Exception:
        return "Unknown"
    names = []
    for name, (start, end) in SESSION_WINDOWS.items():
        if start <= h < end:
            names.append(name)
    return "+".join(names) if names else "Off-session"


def _label_sessions(df):
    """Add a Session column for reporting."""
    if df is None or df.empty:
        return df
    out = df.copy()
    sessions = []
    for ts in out.index:
        sessions.append(_session_for_ts(ts))
    out["Session"] = sessions
    return out


def _map_ise_to_amd_phase(ise: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate Institutional Structure Engine stages into AMD phase language.

    Mapping (dynamic, not hardcoded calendar):
      • No impulse / RANGE / COMPRESSION / RECTANGLE  → ACCUMULATION
      • Liquidity sweep present, manip not confirmed   → MANIPULATION
      • Strong impulse / EXPANSION / flag breakout     → DISPLACEMENT
      • Pullback after impulse (flag / channel / retest) → REVERSION
      • Acceptance + continuation path confirmed       → CONTINUATION
    """
    if ise.get("error"):
        return {
            "phase": "ACCUMULATION",
            "bias": "NEUTRAL",
            "note": f"ISE unavailable: {ise['error']}",
            "path": None,
        }

    state = (ise.get("state") or {}).get("state", "RANGE")
    impulse = ise.get("impulse")
    pullback = ise.get("pullback")
    sweep = ise.get("sweep")
    manipulation = ise.get("manipulation") or {}
    acceptance = ise.get("acceptance") or {}
    entry = ise.get("entry") or {}
    direction = ise.get("direction", "NEUTRAL")
    path = entry.get("path")
    pattern = (pullback or {}).get("pattern") or ""

    # Highest priority: confirmed trade path from ISE
    if ise.get("valid") and direction in ("BUY", "SELL"):
        if path == "continuation":
            return {
                "phase": "CONTINUATION",
                "bias": direction,
                "note": "ISE continuation path accepted (flag/expansion break + structure). Ride with trend.",
                "path": path,
            }
        if path == "reversal":
            return {
                "phase": "CONTINUATION",  # post-acceptance expansion in the new direction
                "bias": direction,
                "note": "ISE reversal path complete (channel distribution/accumulation → sweep → manip → acceptance).",
                "path": path,
            }
        if path == "expansion":
            return {
                "phase": "DISPLACEMENT",
                "bias": direction,
                "note": "Range still expanding in impulse direction — live displacement leg.",
                "path": path,
            }

    # Acceptance confirmed but not yet broken channel/horizontal → REVERSION zone
    if acceptance.get("accepted"):
        bias = "BUY" if acceptance.get("side") == "BULLISH" else (
            "SELL" if acceptance.get("side") == "BEARISH" else direction
        )
        return {
            "phase": "REVERSION",
            "bias": bias if bias in ("BUY", "SELL") else "NEUTRAL",
            "note": acceptance.get("note", "Acceptance holding — high-probability entry zone."),
            "path": path,
        }

    # Manipulation confirmed, waiting acceptance
    if manipulation.get("confirmed"):
        hint = (sweep or {}).get("direction_hint") or direction
        return {
            "phase": "MANIPULATION",
            "bias": hint if hint in ("BUY", "SELL") else "NEUTRAL",
            "note": manipulation.get("note", "Manipulation confirmed — wait for acceptance."),
            "path": path,
        }

    # Sweep seen but not yet rejected
    if sweep is not None:
        hint = sweep.get("direction_hint") or "NEUTRAL"
        return {
            "phase": "MANIPULATION",
            "bias": hint if hint in ("BUY", "SELL") else "NEUTRAL",
            "note": sweep.get("note", "Liquidity swept — waiting for rejection / acceptance."),
            "path": path,
        }

    # Strong impulse + expansion pattern
    if impulse and not impulse.get("weak") and pattern == "EXPANSION":
        return {
            "phase": "DISPLACEMENT",
            "bias": impulse["direction"],
            "note": f"Strong impulse ({impulse['length_atr']}x ATR) with expanding range — displacement.",
            "path": path,
        }

    # Impulse exists and we are in a pullback structure
    if impulse and pullback:
        if pattern in ("BULL_FLAG", "BEAR_FLAG", "RISING_CHANNEL", "FALLING_CHANNEL", "TRIANGLE"):
            return {
                "phase": "REVERSION",
                "bias": pullback.get("bias_hint") or impulse["direction"],
                "note": f"Post-impulse {pattern.replace('_', ' ').title()} — reversion / pullback zone. {pullback.get('watch_for', '')}",
                "path": path,
            }
        if pattern in ("COMPRESSION", "RECTANGLE"):
            return {
                "phase": "ACCUMULATION",
                "bias": "NEUTRAL",
                "note": f"{pattern.title()} after impulse — energy building. Wait for liquidity grab.",
                "path": path,
            }
        # Generic pullback
        return {
            "phase": "REVERSION",
            "bias": impulse["direction"],
            "note": f"Pullback after {impulse['direction']} impulse — classify structure before entry.",
            "path": path,
        }

    # Clean impulse, no pullback classified yet → still in displacement
    if impulse and not impulse.get("weak"):
        return {
            "phase": "DISPLACEMENT",
            "bias": impulse["direction"],
            "note": f"Impulse {impulse['direction']} ({impulse['length_atr']}x ATR / {impulse['bars']} bars) — displacement leg active.",
            "path": path,
        }

    # Default: no structure → accumulation / range
    return {
        "phase": "ACCUMULATION",
        "bias": "NEUTRAL",
        "note": f"Market state {state} — no clean impulse/pullback yet. Accumulation / wait.",
        "path": None,
    }


def _build_phase_segments_from_ise(df: pd.DataFrame, ise: Dict[str, Any],
                                    lookback_range: int = 28):
    """
    Build chart-shade segments that stay visually compatible with the old AMD
    map, but are anchored to what the ISE actually detected (impulse origin,
    sweep bar, acceptance, etc.) instead of a pure range heuristic.
    Falls back to a simple accumulation block when ISE has nothing useful.
    """
    n = len(df)
    if n < 10:
        return [{"start_idx": 0, "end_idx": max(0, n - 1), "phase": "ACCUMULATION"}], "ACCUMULATION", {}

    phases = ["ACCUMULATION"] * n
    impulse = ise.get("impulse") or {}
    pullback = ise.get("pullback") or {}
    sweep = ise.get("sweep")
    manipulation = ise.get("manipulation") or {}
    acceptance = ise.get("acceptance") or {}
    mapped = _map_ise_to_amd_phase(ise)
    current_phase = mapped["phase"]

    # Paint from impulse origin forward as DISPLACEMENT
    if impulse.get("origin_index") is not None and impulse.get("extreme_index") is not None:
        o = int(impulse["origin_index"])
        e = int(impulse["extreme_index"])
        for i in range(max(0, o), min(e + 1, n)):
            phases[i] = "DISPLACEMENT"
        # Post-extreme = reversion / accumulation depending on pattern
        seg_start = e + 1
        if seg_start < n:
            post = mapped["phase"] if mapped["phase"] in (
                "REVERSION", "CONTINUATION", "ACCUMULATION", "MANIPULATION"
            ) else "REVERSION"
            for i in range(seg_start, n):
                phases[i] = post

    # Override with manipulation window around sweep
    if sweep and sweep.get("swept_pos") is not None:
        sp = int(sweep["swept_pos"])
        for i in range(max(0, sp), min(sp + 3, n)):
            phases[i] = "MANIPULATION"

    # Acceptance → paint last few bars as CONTINUATION or REVERSION
    if acceptance.get("accepted"):
        for i in range(max(0, n - 4), n):
            phases[i] = "CONTINUATION" if ise.get("valid") else "REVERSION"

    # Ensure last bar matches the dynamic current phase
    phases[-1] = current_phase

    # Collapse into segments
    segments = []
    cur = phases[0]
    seg_start = 0
    for i in range(1, n):
        if phases[i] != cur:
            segments.append({"start_idx": seg_start, "end_idx": i - 1, "phase": cur})
            cur = phases[i]
            seg_start = i
    segments.append({"start_idx": seg_start, "end_idx": n - 1, "phase": cur})

    # Range meta for report / chart (use pullback channel or recent window)
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and df["ATR"].iloc[-1] > 0 else float(
        (df["High"] - df["Low"]).tail(14).mean() or 1e-9
    )
    if pullback:
        rng_high = float(pullback.get("channel_high") or df["High"].iloc[-lookback_range:].max())
        rng_low = float(pullback.get("channel_low") or df["Low"].iloc[-lookback_range:].min())
    else:
        window = df.iloc[-lookback_range:]
        rng_high = float(window["High"].max())
        rng_low = float(window["Low"].min())

    meta = {
        "high": rng_high,
        "low": rng_low,
        "mid": (rng_high + rng_low) / 2.0,
        "height": rng_high - rng_low,
        "compressed": (rng_high - rng_low) < 2.5 * atr if atr > 0 else False,
        "atr": atr,
        "range_start_idx": max(0, n - lookback_range),
        "manip_idx": int(sweep["swept_pos"]) if sweep and sweep.get("swept_pos") is not None else None,
        "manip_side": ("HIGH" if (sweep or {}).get("side") == "BSL" else "LOW") if sweep else None,
        "disp_start": int(impulse["extreme_index"]) if impulse.get("extreme_index") is not None else None,
        "disp_dir": ("UP" if impulse.get("direction") == "BUY" else "DOWN") if impulse else None,
    }
    return segments, current_phase, meta


def run_amd_analysis(symbol):
    """
    Full AMD package — Structure Engine is the decision authority.

    1. Load multi-TF data
    2. Run ISE on 1H (same 10-stage pipeline as Trendline)
    3. Map ISE → AMD phase / bias dynamically
    4. Attach SMC zones, sessions, HTF context as confluence only
    """
    symbol = symbol.strip().upper()

    df_4h = mt5_data.fetch_candles(symbol, "4h", count=150)
    df_1h = mt5_data.fetch_candles(symbol, "1h", count=200)
    df_30 = mt5_data.fetch_candles(symbol, "30min", count=150)
    df_15 = mt5_data.fetch_candles(symbol, "15min", count=150)

    if df_1h is None or df_1h.empty or len(df_1h) < 40:
        return {"error": f"Insufficient 1H data for AMD analysis on {symbol}."}

    df_1h = _label_sessions(df_1h)
    structure_1h = analyse_structure(df_1h, left=2, right=2, lookback=60)
    structure_4h = (
        analyse_structure(df_4h, left=3, right=3, lookback=50)
        if df_4h is not None and len(df_4h) > 30 else None
    )

    # --- Core: Institutional Structure Engine on 1H ---
    ise = run_structure_engine(df_1h)
    mapped = _map_ise_to_amd_phase(ise)
    phase_segments, current_phase, rng_meta = _build_phase_segments_from_ise(df_1h, ise)

    rng = {
        "high": rng_meta["high"],
        "low": rng_meta["low"],
        "mid": rng_meta["mid"],
        "height": rng_meta["height"],
        "compressed": rng_meta["compressed"],
        "atr": rng_meta["atr"],
    }

    # Manipulation object for report compatibility
    manip = None
    sweep = ise.get("sweep")
    manipulation = ise.get("manipulation") or {}
    if sweep:
        manip = {
            "side": "BUY_SIDE_LIQUIDITY" if sweep.get("side") == "BSL" else "SELL_SIDE_LIQUIDITY",
            "direction_hint": sweep.get("direction_hint", mapped["bias"]),
            "index": sweep.get("swept_pos"),
            "note": sweep.get("note") or manipulation.get("note", ""),
            "confirmed": bool(manipulation.get("confirmed")),
        }
    elif manipulation.get("confirmed"):
        manip = {
            "side": "UNKNOWN",
            "direction_hint": mapped["bias"],
            "index": None,
            "note": manipulation.get("note", ""),
            "confirmed": True,
        }

    amd_bias = mapped["bias"]
    # Prefer ISE confirmed direction when valid
    if ise.get("valid") and ise.get("direction") in ("BUY", "SELL"):
        amd_bias = ise["direction"]

    # LOCKED clean-chart defaults: fewer zones, only structure-confirmed OBs
    fvgs = detect_fvgs(df_1h, min_gap_atr=0.18, max_zones=4)
    obs = detect_order_blocks(df_1h, structure=structure_1h, max_zones=3, require_bos=True)
    idms = detect_inducement_zones(df_1h, max_zones=4)
    vp = compute_volume_profile(df_1h.iloc[:-1])

    last_session = str(df_1h["Session"].iloc[-1]) if "Session" in df_1h.columns else "Unknown"

    htf_note = ""
    if structure_4h:
        htf_note = structure_4h.get("note", "")
        if structure_4h.get("bias") == "BULLISH" and amd_bias == "SELL":
            htf_note += " | ⚠️ 1H AMD bearish vs 4H bullish structure"
        elif structure_4h.get("bias") == "BEARISH" and amd_bias == "BUY":
            htf_note += " | ⚠️ 1H AMD bullish vs 4H bearish structure"

    entry_notes = []
    for tf_name, dframe in (("30M", df_30), ("15M", df_15)):
        if dframe is not None and len(dframe) >= 40:
            st = analyse_structure(dframe, left=2, right=2, lookback=40)
            entry_notes.append(f"{tf_name}: {st.get('note', st.get('bias', ''))}")

    # Pattern scanner on 1H for confluence (fixes "pattern scanner not returning analysis")
    best_pattern = None
    all_patterns = []
    try:
        from patterns import scan_all_patterns
        best_pattern, all_patterns = scan_all_patterns(df_1h.iloc[:-1], volume_profile=vp)
        if all_patterns:
            all_patterns = all_patterns[:3]
    except Exception:
        pass

    return {
        "symbol": symbol,
        "primary_tf": "1h",
        "amd_bias": amd_bias,
        "phase": current_phase,
        "phase_note": mapped["note"],
        "phase_segments": phase_segments,
        "manipulation": manip,
        "range": rng,
        "structure_1h": structure_1h,
        "structure_4h": structure_4h,
        "htf_note": htf_note,
        "last_session": last_session,
        "fvgs": fvgs,
        "order_blocks": obs,
        "inducements": idms,
        "bos_events": build_bos_events(df_1h, max_events=8),
        "volume_profile": vp,
        "entry_notes": entry_notes,
        "df_1h": df_1h,
        # ISE payload — full dynamic authority
        "ise": ise,
        "ise_valid": bool(ise.get("valid")),
        "ise_direction": ise.get("direction", "NEUTRAL"),
        "ise_score": int(ise.get("score", 0)),
        "ise_path": (ise.get("entry") or {}).get("path"),
        "ise_reasons": list(ise.get("reasons") or []),
        "best_pattern": best_pattern,
        "all_patterns": all_patterns,
    }


def format_amd_report(analysis):
    """SHORT AMD summary driven by ISE stages — chart shows zones."""
    if "error" in analysis:
        return analysis["error"]

    symbol = analysis["symbol"]
    lines = []
    lines.append(f"🕯 AMD {symbol}  |  1H  |  Bias: {analysis['amd_bias']}")
    lines.append(f"Phase: {analysis['phase']}  |  Session: {analysis['last_session']}")
    if analysis.get("phase_note"):
        lines.append(f"  {analysis['phase_note']}")

    # ISE stage summary (dynamic)
    ise = analysis.get("ise") or {}
    if ise and not ise.get("error"):
        st = ise.get("state") or {}
        lines.append(f"ISE State: {st.get('state', '?')} ({st.get('reason', '')})")
        imp = ise.get("impulse")
        if imp:
            lines.append(
                f"ISE Impulse: {imp['direction']} · {imp['length_atr']}x ATR / {imp['bars']} bars"
                + (" ⚠️ weak" if imp.get("weak") else "")
            )
        pb = ise.get("pullback")
        if pb:
            lines.append(f"ISE Pullback: {pb['pattern'].replace('_', ' ').title()}")
        if ise.get("sweep"):
            lines.append(f"ISE Liquidity: {ise['sweep'].get('note', 'sweep')}")
        man = ise.get("manipulation") or {}
        if man.get("note"):
            lines.append(f"ISE Manip: {man['note']}")
        acc = ise.get("acceptance") or {}
        if acc.get("note"):
            lines.append(f"ISE Accept: {acc['note']}")
        lines.append(
            f"ISE Verdict: {'TRADE ' + str(ise.get('direction')) if ise.get('valid') else 'WAIT'}"
            f" (score {ise.get('score', 0)})"
        )
    else:
        segs = analysis.get("phase_segments") or []
        if segs:
            path = " → ".join(s["phase"][:5] for s in segs[-5:])
            lines.append(f"Cycle: {path}")

    st = analysis.get("structure_1h") or {}
    if st.get("note"):
        lines.append(f"Struct: {st['note']}")

    rng = analysis.get("range")
    if rng:
        lines.append(
            f"Range: {rng['low']:.5f} – {rng['high']:.5f}"
            f"{' (compressed)' if rng.get('compressed') else ''}"
        )

    manip = analysis.get("manipulation")
    if manip:
        conf = "confirmed" if manip.get("confirmed") else "pending"
        lines.append(f"Manip: {manip['side']} → hint {manip['direction_hint']} [{conf}]")

    n_fvg = len(analysis.get("fvgs") or [])
    n_ob = len(analysis.get("order_blocks") or [])
    n_idm = len(analysis.get("inducements") or [])
    unmit = sum(1 for z in (analysis.get("inducements") or []) if not z.get("mitigated"))
    lines.append(f"Zones: {n_fvg} FVG · {n_ob} OB · {n_idm} IDM ({unmit} unmitigated)")

    pairs = pair_idm_with_extreme_ob(
        analysis.get("inducements") or [], analysis.get("order_blocks") or []
    )
    if pairs:
        p = pairs[0]
        lines.append(f"Setup: {p['direction']} IDM→OB (see chart)")
    else:
        lines.append("Setup: no clean IDM→OB pair")

    bp = analysis.get("best_pattern")
    if bp:
        lines.append(f"Pattern: {bp.name} ({bp.bias}) {bp.confidence:.0f}%")
        if bp.note:
            lines.append(f"  {bp.note[:120]}")

    vp = analysis.get("volume_profile")
    if vp:
        lines.append(
            f"POC {vp['poc_price']:.5f} | VA {vp['value_area_low']:.5f}–{vp['value_area_high']:.5f}"
        )

    if analysis.get("entry_notes"):
        lines.append("Entry: " + " · ".join(analysis["entry_notes"][:2]))

    if analysis.get("htf_note"):
        lines.append(f"HTF: {analysis['htf_note']}")

    lines.append("📷 Chart = FVG/OB/IDM/range + ISE structure (full story)")
    return "\n".join(lines)
