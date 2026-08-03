"""
amd_analysis.py
===============
AMD = Accumulation → Manipulation → Distribution (ICT / power-of-three style).

Primary chart: **1 Hour** (AMD cycles are most readable on 1H).
Context: 4H bias → 1H AMD phases → 30M / 15M for entry refinement.

Session separators (TradingView-style conceptual template):
  We mark Asian / London / New York session blocks on the 1H series so the
  report can say which phase sits inside which session.

  Default UTC session windows (adjustable):
    Asian   : 00:00 – 08:00 UTC
    London  : 07:00 – 16:00 UTC
    NewYork : 12:00 – 21:00 UTC

Phase detection (practical rules on 1H):
  Accumulation : tight range after a directional move; compression of highs/lows
  Manipulation : liquidity sweep beyond the range (raid of highs or lows) then reclaim
  Distribution : expansion away from the range in the true direction (often after NY open)
"""

import numpy as np
import pandas as pd
from market_structure import analyse_structure
from smc_zones import detect_fvgs, detect_order_blocks, detect_inducement_zones, pair_idm_with_extreme_ob, summarise_smc_zones
from volume_profile import compute_volume_profile
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


def _detect_range(df, lookback=24):
    """Recent consolidation range on 1H (last `lookback` bars)."""
    if df is None or len(df) < lookback:
        return None
    window = df.iloc[-lookback:]
    high = float(window["High"].max())
    low = float(window["Low"].min())
    mid = (high + low) / 2.0
    height = high - low
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else height / 4
    compressed = height < 2.5 * atr if atr > 0 else False
    return {
        "high": high,
        "low": low,
        "mid": mid,
        "height": height,
        "compressed": compressed,
        "atr": atr,
    }


def _detect_manipulation(df, rng):
    """
    Manipulation = liquidity raid beyond range high/low then reclaim back inside.
    Looks at the last ~12 bars relative to the range.
    """
    if rng is None or df is None or len(df) < 8:
        return None

    recent = df.iloc[-12:]
    highs = recent["High"].values
    lows = recent["Low"].values
    closes = recent["Close"].values

    # Raid above range high then close back below
    raid_high = False
    for i in range(len(recent)):
        if highs[i] > rng["high"] * 1.0002 and closes[i] < rng["high"]:
            raid_high = True
            break

    # Raid below range low then close back above
    raid_low = False
    for i in range(len(recent)):
        if lows[i] < rng["low"] * 0.9998 and closes[i] > rng["low"]:
            raid_low = True
            break

    if raid_high and not raid_low:
        return {
            "side": "BUY_SIDE_LIQUIDITY",
            "direction_hint": "SELL",  # classic: raid highs → distribute down
            "note": "Buy-side liquidity raided (highs swept) then reclaimed — classic manipulation before distribution lower",
        }
    if raid_low and not raid_high:
        return {
            "side": "SELL_SIDE_LIQUIDITY",
            "direction_hint": "BUY",
            "note": "Sell-side liquidity raided (lows swept) then reclaimed — classic manipulation before expansion higher",
        }
    if raid_high and raid_low:
        return {
            "side": "BOTH",
            "direction_hint": "NEUTRAL",
            "note": "Both sides of range raided — wait for clearer distribution leg",
        }
    return None


def _detect_distribution(df, rng, manip):
    """Distribution / expansion away from the range."""
    if rng is None or df is None or len(df) < 5:
        return None
    close = float(df["Close"].iloc[-1])
    atr = rng.get("atr") or 1e-9

    if close > rng["high"] + 0.3 * atr:
        return {
            "phase": "DISTRIBUTION_UP",
            "bias": "BUY",
            "note": f"Price expanding above range ({rng['high']:.5f}) — distribution / true move higher",
        }
    if close < rng["low"] - 0.3 * atr:
        return {
            "phase": "DISTRIBUTION_DOWN",
            "bias": "BUY" if False else "SELL",
            "note": f"Price expanding below range ({rng['low']:.5f}) — distribution / true move lower",
        }

    if manip:
        return {
            "phase": "POST_MANIPULATION",
            "bias": manip.get("direction_hint", "NEUTRAL"),
            "note": "Inside/near range after manipulation — wait for expansion (distribution leg)",
        }

    if rng.get("compressed"):
        return {
            "phase": "ACCUMULATION",
            "bias": "NEUTRAL",
            "note": "Compressed range — accumulation phase (wait for manipulation + expansion)",
        }

    return {
        "phase": "RANGE",
        "bias": "NEUTRAL",
        "note": "Price still rotating inside the 1H range",
    }


def run_amd_analysis(symbol):
    """
    Full AMD package.
    Primary: 1H chart + phases.
    Context: 4H structure, 30M/15M for entry notes.
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
    structure_4h = analyse_structure(df_4h, left=3, right=3, lookback=50) if df_4h is not None and len(df_4h) > 30 else None

    rng = _detect_range(df_1h, lookback=24)
    manip = _detect_manipulation(df_1h, rng)
    dist = _detect_distribution(df_1h, rng, manip)

    fvgs = detect_fvgs(df_1h, min_gap_atr=0.12, max_zones=6)
    obs = detect_order_blocks(df_1h, structure=structure_1h, max_zones=5)
    idms = detect_inducement_zones(df_1h, max_zones=5)
    vp = compute_volume_profile(df_1h.iloc[:-1])

    # Session of latest bar
    last_session = str(df_1h["Session"].iloc[-1]) if "Session" in df_1h.columns else "Unknown"

    # Overall AMD bias
    amd_bias = "NEUTRAL"
    if dist and dist.get("bias") in ("BUY", "SELL"):
        amd_bias = dist["bias"]
    elif manip and manip.get("direction_hint") in ("BUY", "SELL"):
        amd_bias = manip["direction_hint"]

    # Align with 4H structure if available
    htf_note = ""
    if structure_4h:
        htf_note = structure_4h["note"]
        if structure_4h["bias"] == "BULLISH" and amd_bias == "SELL":
            htf_note += " | ⚠️ 1H AMD bearish vs 4H bullish structure"
        elif structure_4h["bias"] == "BEARISH" and amd_bias == "BUY":
            htf_note += " | ⚠️ 1H AMD bullish vs 4H bearish structure"

    # Entry refinement notes from 30M / 15M structure
    entry_notes = []
    for tf_name, dframe in (("30M", df_30), ("15M", df_15)):
        if dframe is not None and len(dframe) >= 40:
            st = analyse_structure(dframe, left=2, right=2, lookback=40)
            entry_notes.append(f"{tf_name}: {st['note']}")

    return {
        "symbol": symbol,
        "primary_tf": "1h",
        "amd_bias": amd_bias,
        "phase": dist["phase"] if dist else "UNKNOWN",
        "phase_note": dist["note"] if dist else "",
        "manipulation": manip,
        "range": rng,
        "structure_1h": structure_1h,
        "structure_4h": structure_4h,
        "htf_note": htf_note,
        "last_session": last_session,
        "fvgs": fvgs,
        "order_blocks": obs,
        "inducements": idms,
        "volume_profile": vp,
        "entry_notes": entry_notes,
        "df_1h": df_1h,
    }


def format_amd_report(analysis):
    """SHORT AMD summary — chart shows zones."""
    if "error" in analysis:
        return analysis["error"]

    symbol = analysis["symbol"]
    lines = []
    lines.append(f"🕯 AMD {symbol}  |  1H  |  Bias: {analysis['amd_bias']}")
    lines.append(f"Phase: {analysis['phase']}  |  Session: {analysis['last_session']}")
    if analysis.get("phase_note"):
        lines.append(f"  {analysis['phase_note']}")

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
        lines.append(f"Manip: {manip['side']} → hint {manip['direction_hint']}")

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

    vp = analysis.get("volume_profile")
    if vp:
        lines.append(f"POC {vp['poc_price']:.5f} | VA {vp['value_area_low']:.5f}–{vp['value_area_high']:.5f}")

    if analysis.get("entry_notes"):
        lines.append("Entry: " + " · ".join(analysis["entry_notes"][:2]))

    lines.append("📷 Chart = FVG/OB/IDM/range (full story)")
    return "\n".join(lines)
