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
    """
    Map current price action into the 5-phase AMD cycle used in the educational maps:
      1. ACCUMULATION  – range / compression
      2. MANIPULATION  – liquidity grab (stop hunt)
      3. DISPLACEMENT  – strong expansion away from range
      4. REVERSION     – pullback into FVG / OB / discount-premium
      5. CONTINUATION  – trend continuation after reversion
    """
    if rng is None or df is None or len(df) < 5:
        return None
    close = float(df["Close"].iloc[-1])
    atr = rng.get("atr") or 1e-9
    high = float(df["High"].iloc[-1])
    low = float(df["Low"].iloc[-1])

    # Strong expansion = Displacement
    if close > rng["high"] + 0.35 * atr:
        # Check if we already pulled back (reversion) then continued
        recent = df.iloc[-8:]
        pulled_back = any(float(c) < rng["high"] + 0.15 * atr for c in recent["Close"].values[:-1])
        if pulled_back and close > rng["high"] + 0.5 * atr:
            return {
                "phase": "CONTINUATION",
                "bias": "BUY",
                "note": f"Continuation higher after displacement & reversion — trend leg in progress above {rng['high']:.5f}",
            }
        return {
            "phase": "DISPLACEMENT",
            "bias": "BUY",
            "note": f"Displacement higher — strong expansion above range high {rng['high']:.5f}",
        }

    if close < rng["low"] - 0.35 * atr:
        recent = df.iloc[-8:]
        pulled_back = any(float(c) > rng["low"] - 0.15 * atr for c in recent["Close"].values[:-1])
        if pulled_back and close < rng["low"] - 0.5 * atr:
            return {
                "phase": "CONTINUATION",
                "bias": "SELL",
                "note": f"Continuation lower after displacement & reversion — trend leg in progress below {rng['low']:.5f}",
            }
        return {
            "phase": "DISPLACEMENT",
            "bias": "SELL",
            "note": f"Displacement lower — strong expansion below range low {rng['low']:.5f}",
        }

    # After manipulation, price still near range → possible reversion zone
    if manip:
        hint = manip.get("direction_hint", "NEUTRAL")
        # If we raided highs and are still high → reversion short setup forming
        # If we raided lows and are still low → reversion long setup forming
        return {
            "phase": "REVERSION",
            "bias": hint,
            "note": "Post-manipulation — price in reversion / pullback zone. Wait for confirmation into OB/FVG before continuation.",
        }

    if rng.get("compressed"):
        return {
            "phase": "ACCUMULATION",
            "bias": "NEUTRAL",
            "note": "Compressed range — Accumulation phase. Smart money building positions. Wait for liquidity grab.",
        }

    return {
        "phase": "ACCUMULATION",
        "bias": "NEUTRAL",
        "note": "Price rotating inside 1H range — Accumulation / range phase.",
    }


def _build_phase_segments(df, lookback_range=28):
    """
    Walk the 1H series and assign each bar a phase so the chart can shade
    real boundaries (not a crude 40/60 split).

    Logic (practical, ICT-style):
      1. Find the most recent consolidation range on a rolling window.
      2. Mark bars inside that range as ACCUMULATION.
      3. First bar that sweeps range high/low then reclaims → MANIPULATION.
      4. Bars that close clearly outside the range with momentum → DISPLACEMENT.
      5. First meaningful pullback toward the range / mid after displacement → REVERSION.
      6. Bars that resume in the displacement direction after reversion → CONTINUATION.

    Returns list of {start_idx, end_idx, phase} in dataframe index space
    (absolute indices into `df`).
    """
    if df is None or len(df) < 30:
        return [{"start_idx": 0, "end_idx": max(0, len(df) - 1), "phase": "ACCUMULATION"}]

    n = len(df)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    atr_series = df["ATR"].values.astype(float) if "ATR" in df.columns else None

    def atr_at(i):
        if atr_series is not None and i < len(atr_series) and atr_series[i] > 0:
            return float(atr_series[i])
        return max(float(highs[i] - lows[i]), 1e-9)

    # --- 1) Locate the primary range used for the current cycle ---
    # Use the last `lookback_range` bars that still look like consolidation,
    # then expand slightly backward if needed.
    end = n - 1
    start_search = max(0, end - lookback_range)
    win_high = float(np.max(highs[start_search:end + 1]))
    win_low = float(np.min(lows[start_search:end + 1]))
    mid = (win_high + win_low) / 2.0
    height = win_high - win_low
    atr_ref = atr_at(end)
    compressed = height < 2.8 * atr_ref if atr_ref > 0 else False

    # Refine range start: first bar of a relatively tight stretch before expansion
    range_start = start_search
    for i in range(start_search, end + 1):
        local = highs[max(0, i - 6):i + 1]
        local_l = lows[max(0, i - 6):i + 1]
        if (float(np.max(local)) - float(np.min(local_l))) <= 2.5 * atr_at(i):
            range_start = max(0, i - 6)
            break

    # Recompute range from refined window (before any big expansion)
    # Use bars from range_start until first clear break
    rng_high = float(np.max(highs[range_start:min(range_start + lookback_range, end) + 1]))
    rng_low = float(np.min(lows[range_start:min(range_start + lookback_range, end) + 1]))
    atr0 = atr_at(min(range_start + 5, end))

    phases = ["ACCUMULATION"] * n  # default

    # Mark accumulation for the range window
    for i in range(range_start, end + 1):
        phases[i] = "ACCUMULATION"

    # --- 2) Find manipulation (first sweep beyond range then reclaim) ---
    manip_idx = None
    manip_side = None  # "HIGH" or "LOW"
    for i in range(range_start + 2, end + 1):
        a = atr_at(i)
        # Sweep high then close back inside
        if highs[i] > rng_high + 0.05 * a and closes[i] < rng_high:
            manip_idx = i
            manip_side = "HIGH"
            break
        # Sweep low then close back inside
        if lows[i] < rng_low - 0.05 * a and closes[i] > rng_low:
            manip_idx = i
            manip_side = "LOW"
            break

    if manip_idx is not None:
        # Manipulation often spans 1–3 bars
        for i in range(manip_idx, min(manip_idx + 3, end + 1)):
            phases[i] = "MANIPULATION"

    # --- 3) Displacement: first strong close outside range after manip (or after range) ---
    disp_start = None
    disp_dir = None  # "UP" or "DOWN"
    search_from = (manip_idx + 1) if manip_idx is not None else (range_start + 4)
    for i in range(search_from, end + 1):
        a = atr_at(i)
        if closes[i] > rng_high + 0.3 * a:
            disp_start = i
            disp_dir = "UP"
            break
        if closes[i] < rng_low - 0.3 * a:
            disp_start = i
            disp_dir = "DOWN"
            break

    if disp_start is not None:
        # Displacement leg until first meaningful pullback
        for i in range(disp_start, end + 1):
            phases[i] = "DISPLACEMENT"

        # --- 4) Reversion: first pullback of >= 0.4 ATR toward range ---
        rev_start = None
        extreme = closes[disp_start]
        for i in range(disp_start + 1, end + 1):
            a = atr_at(i)
            if disp_dir == "UP":
                extreme = max(extreme, highs[i])
                if closes[i] < extreme - 0.4 * a:
                    rev_start = i
                    break
            else:
                extreme = min(extreme, lows[i])
                if closes[i] > extreme + 0.4 * a:
                    rev_start = i
                    break

        if rev_start is not None:
            for i in range(rev_start, end + 1):
                phases[i] = "REVERSION"

            # --- 5) Continuation: resume in displacement direction after reversion ---
            cont_start = None
            for i in range(rev_start + 1, end + 1):
                a = atr_at(i)
                if disp_dir == "UP" and closes[i] > closes[rev_start] + 0.35 * a:
                    cont_start = i
                    break
                if disp_dir == "DOWN" and closes[i] < closes[rev_start] - 0.35 * a:
                    cont_start = i
                    break
            if cont_start is not None:
                for i in range(cont_start, end + 1):
                    phases[i] = "CONTINUATION"

    # Collapse consecutive same-phase bars into segments
    segments = []
    cur = phases[0]
    seg_start = 0
    for i in range(1, n):
        if phases[i] != cur:
            segments.append({"start_idx": seg_start, "end_idx": i - 1, "phase": cur})
            cur = phases[i]
            seg_start = i
    segments.append({"start_idx": seg_start, "end_idx": n - 1, "phase": cur})

    # Current phase = last segment
    current_phase = segments[-1]["phase"] if segments else "ACCUMULATION"
    return segments, current_phase, {
        "high": rng_high,
        "low": rng_low,
        "mid": (rng_high + rng_low) / 2.0,
        "height": rng_high - rng_low,
        "compressed": compressed,
        "atr": atr_ref,
        "range_start_idx": range_start,
        "manip_idx": manip_idx,
        "manip_side": manip_side,
        "disp_start": disp_start,
        "disp_dir": disp_dir,
    }


def run_amd_analysis(symbol):
    """
    Full AMD package.
    Primary: 1H chart + precise phase segments.
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

    # Precise phase segmentation
    phase_segments, current_phase, rng_meta = _build_phase_segments(df_1h, lookback_range=28)

    # Keep legacy helpers for bias / notes (compatible with existing report)
    rng = {
        "high": rng_meta["high"],
        "low": rng_meta["low"],
        "mid": rng_meta["mid"],
        "height": rng_meta["height"],
        "compressed": rng_meta["compressed"],
        "atr": rng_meta["atr"],
    }
    manip = None
    if rng_meta.get("manip_idx") is not None:
        side = "BUY_SIDE_LIQUIDITY" if rng_meta["manip_side"] == "HIGH" else "SELL_SIDE_LIQUIDITY"
        hint = "SELL" if rng_meta["manip_side"] == "HIGH" else "BUY"
        manip = {
            "side": side,
            "direction_hint": hint,
            "index": rng_meta["manip_idx"],
            "note": f"{side} swept at bar {rng_meta['manip_idx']}",
        }

    # Map current phase to bias + note
    phase_notes = {
        "ACCUMULATION": "Compressed/rotating range — Accumulation. Wait for liquidity grab.",
        "MANIPULATION": "Liquidity grab in progress — wait for displacement confirmation.",
        "DISPLACEMENT": "Strong expansion away from range — true directional leg.",
        "REVERSION": "Pullback into discount/premium — high-probability entry zone.",
        "CONTINUATION": "Trend resumed after reversion — ride with structure.",
    }
    dist = {
        "phase": current_phase,
        "bias": "NEUTRAL",
        "note": phase_notes.get(current_phase, ""),
    }
    if current_phase in ("DISPLACEMENT", "CONTINUATION", "REVERSION"):
        if rng_meta.get("disp_dir") == "UP":
            dist["bias"] = "BUY"
        elif rng_meta.get("disp_dir") == "DOWN":
            dist["bias"] = "SELL"
        elif manip:
            dist["bias"] = manip.get("direction_hint", "NEUTRAL")
    elif manip:
        dist["bias"] = manip.get("direction_hint", "NEUTRAL")

    fvgs = detect_fvgs(df_1h, min_gap_atr=0.12, max_zones=6)
    obs = detect_order_blocks(df_1h, structure=structure_1h, max_zones=5)
    idms = detect_inducement_zones(df_1h, max_zones=5)
    vp = compute_volume_profile(df_1h.iloc[:-1])

    last_session = str(df_1h["Session"].iloc[-1]) if "Session" in df_1h.columns else "Unknown"

    amd_bias = dist.get("bias", "NEUTRAL")
    if amd_bias == "NEUTRAL" and manip:
        amd_bias = manip.get("direction_hint", "NEUTRAL")

    htf_note = ""
    if structure_4h:
        htf_note = structure_4h["note"]
        if structure_4h["bias"] == "BULLISH" and amd_bias == "SELL":
            htf_note += " | ⚠️ 1H AMD bearish vs 4H bullish structure"
        elif structure_4h["bias"] == "BEARISH" and amd_bias == "BUY":
            htf_note += " | ⚠️ 1H AMD bullish vs 4H bearish structure"

    entry_notes = []
    for tf_name, dframe in (("30M", df_30), ("15M", df_15)):
        if dframe is not None and len(dframe) >= 40:
            st = analyse_structure(dframe, left=2, right=2, lookback=40)
            entry_notes.append(f"{tf_name}: {st['note']}")

    return {
        "symbol": symbol,
        "primary_tf": "1h",
        "amd_bias": amd_bias,
        "phase": current_phase,
        "phase_note": dist["note"],
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
    segs = analysis.get("phase_segments") or []
    if segs:
        # Compact cycle path e.g. ACCUM → MANIP → DISP → REV
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
