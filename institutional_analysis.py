"""
institutional_analysis.py
=========================
True Top-Down Institutional Analysis.

Priority stack:
  1. 200 EMA on HTF          → Who is ruling the market
  2. Trendline Families      → Primary structure + projections
  3. BOS / CHoCH / MSS       → Structure permission
  4. VWAP / Volume Profile   → Dynamic levels
  5. FVG / OB / IDM          → SMC zones (drawn on chart)
  6. Chart Patterns          → Confluence only

Reports are SHORT — the chart carries the visual story.
"""

import numpy as np
import pandas as pd
from patterns import scan_all_patterns, find_pivots, _atr, Pattern
from volume_profile import compute_volume_profile
from market_structure import analyse_structure, structure_trade_permission
from smc_zones import (
    detect_fvgs, detect_order_blocks, detect_inducement_zones,
    detect_base_zones, pair_idm_with_extreme_ob, summarise_smc_zones,
    build_bos_events,
)

import mt5_data

TOPDOWN_LADDER = [
    ("4h", "4 Hour"),
    ("1h", "1 Hour"),
    ("30min", "30 Minute"),
]
ALT_LADDER = [
    ("1h", "1 Hour"),
    ("30min", "30 Minute"),
    ("15min", "15 Minute"),
]


def _ema200_bias(df):
    if df is None or df.empty or "EMA200" not in df.columns:
        return "NEUTRAL", "EMA200 n/a", 0.0
    close = float(df["Close"].iloc[-1])
    ema200 = float(df["EMA200"].iloc[-1])
    if ema200 <= 0:
        return "NEUTRAL", "EMA200 n/a", 0.0
    dist = (close - ema200) / ema200 * 100.0
    if close > ema200 * 1.001:
        return "BUY", f"Above 200 EMA (+{dist:.2f}%)", dist
    if close < ema200 * 0.999:
        return "SELL", f"Below 200 EMA ({dist:.2f}%)", dist
    return "NEUTRAL", f"At 200 EMA ({dist:+.2f}%)", dist


def _vwap_context(df):
    if df is None or df.empty or "VWAP" not in df.columns:
        return None
    close = float(df["Close"].iloc[-1])
    vwap = float(df["VWAP"].iloc[-1])
    if vwap <= 0:
        return None
    dist_pct = (close - vwap) / vwap * 100.0
    if close > vwap * 1.0005:
        pos, note = "ABOVE", f"Above VWAP (+{dist_pct:.2f}%)"
    elif close < vwap * 0.9995:
        pos, note = "BELOW", f"Below VWAP ({dist_pct:.2f}%)"
    else:
        pos, note = "AT", f"At VWAP ({dist_pct:+.2f}%)"
    return {"vwap": vwap, "position": pos, "distance_pct": dist_pct, "note": note}


def _fit_trendline_family(df, lookback=80):
    if df is None or len(df) < 30:
        return None
    ph, pl = find_pivots(df, left=4, right=4)
    n = len(df)
    start = max(0, n - lookback)
    recent_ph = [p for p in ph if p >= start][-4:]
    recent_pl = [p for p in pl if p >= start][-4:]
    if len(recent_ph) < 2 or len(recent_pl) < 2:
        return None

    upper_pts = [(p, float(df["High"].iloc[p])) for p in recent_ph]
    lower_pts = [(p, float(df["Low"].iloc[p])) for p in recent_pl]

    def _fit(pts):
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        if len(xs) < 2 or np.all(xs == xs[0]):
            return 0.0, float(ys[-1])
        slope, intercept = np.polyfit(xs, ys, 1)
        return float(slope), float(intercept)

    up_slope, up_int = _fit(upper_pts)
    lo_slope, lo_int = _fit(lower_pts)
    x_now = n - 1
    upper_now = up_slope * x_now + up_int
    lower_now = lo_slope * x_now + lo_int
    if upper_now <= lower_now:
        return None
    height = upper_now - lower_now
    close = float(df["Close"].iloc[-1])
    pos = (close - lower_now) / height if height > 0 else 0.5
    avg_price = float(df["Close"].tail(lookback).mean()) or 1.0
    up_norm = (up_slope * lookback) / avg_price
    lo_norm = (lo_slope * lookback) / avg_price
    FLAT = 0.004
    if abs(up_norm) < FLAT and lo_norm > FLAT:
        family, bias = "Ascending Channel", "BUY"
    elif up_norm < -FLAT and abs(lo_norm) < FLAT:
        family, bias = "Descending Channel", "SELL"
    elif up_norm > FLAT and lo_norm > FLAT:
        family, bias = "Rising Channel", "BUY"
    elif up_norm < -FLAT and lo_norm < -FLAT:
        family, bias = "Falling Channel", "SELL"
    else:
        family, bias = "Range / Contract", "NEUTRAL"
    return {
        "family": family, "bias": bias,
        "upper": float(upper_now), "lower": float(lower_now),
        "mid": float((upper_now + lower_now) / 2),
        "height": float(height),
        "position": float(np.clip(pos, 0, 1)),
        "proj_up": float(upper_now + height),
        "proj_down": float(lower_now - height),
        "upper_pts": upper_pts, "lower_pts": lower_pts,
    }


def _analyse_single_tf(symbol, tf_code, tf_label):
    df = mt5_data.fetch_candles(symbol, tf_code, count=150)
    if df is None or df.empty or len(df) < 40:
        return None
    ema_bias, ema_note, ema_dist = _ema200_bias(df)
    vwap = _vwap_context(df)
    # Full detectors only on HTF/primary chart TF — keeps Render free tier under timeout
    heavy = tf_code in ("4h", "1h")
    trend = _fit_trendline_family(df) if heavy else None
    vp = compute_volume_profile(df.iloc[:-1]) if heavy else None
    if heavy:
        best, all_pats = scan_all_patterns(df.iloc[:-1], volume_profile=vp)
    else:
        best, all_pats = None, []
    structure = analyse_structure(df, left=3, right=3, lookback=50 if heavy else 40)
    fvgs = detect_fvgs(df, min_gap_atr=0.18, max_zones=4 if heavy else 2)
    obs = detect_order_blocks(df, structure=structure, max_zones=3 if heavy else 2, require_bos=True)
    idms = detect_inducement_zones(df, max_zones=3 if heavy else 2)
    base_zones = detect_base_zones(df, max_zones=3) if heavy else []
    bos_events = build_bos_events(df, max_events=6 if heavy else 3)
    return {
        "tf": tf_code, "tf_label": tf_label, "df": df,
        "close": float(df["Close"].iloc[-1]),
        "ema200_bias": ema_bias, "ema200_note": ema_note, "ema200_dist": ema_dist,
        "vwap": vwap, "trendline": trend, "volume_profile": vp,
        "best_pattern": best, "all_patterns": all_pats[:3] if all_pats else [],
        "structure": structure, "fvgs": fvgs, "order_blocks": obs,
        "inducements": idms, "base_zones": base_zones,
        "bos_events": bos_events,
    }



import concurrent.futures


def _fetch_ladder_parallel(symbol, ladder):
    """
    Fetches every timeframe in `ladder` concurrently instead of serially.
    Each _analyse_single_tf call can take up to ~30s in the worst case
    (Twelve Data timeout + yfinance fallback timeout stacked, which is
    common for symbols like BTCUSD that aren't on the free Twelve Data
    tier). Fetching 3 timeframes serially could take up to ~90s -- and if
    the ladder came up short and ALT_LADDER had to run too, that doubled
    to ~180s, blowing past the 120s outer timeout in bot.py. Running them
    in parallel bounds total wait to roughly the single slowest fetch.
    """
    frames = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ladder)) as ex:
        futures = {
            ex.submit(_analyse_single_tf, symbol, tf_code, tf_label): tf_code
            for tf_code, tf_label in ladder
        }
        results = {}
        for fut in concurrent.futures.as_completed(futures):
            tf_code = futures[fut]
            try:
                snap = fut.result()
            except Exception as e:
                print(f"[institutional_analysis] {symbol} {tf_code} failed: {e!r}")
                snap = None
            if snap:
                results[tf_code] = snap
    # Preserve ladder order (HTF first) regardless of completion order --
    # downstream code assumes frames[0] is the highest timeframe.
    for tf_code, _ in ladder:
        if tf_code in results:
            frames.append(results[tf_code])
    return frames


def run_topdown_analysis(symbol):
    symbol = symbol.strip().upper()
    frames = _fetch_ladder_parallel(symbol, TOPDOWN_LADDER)
    if len(frames) < 2:
        # Only fetch the ALT_LADDER timeframes we don't already have, and do
        # it in the same parallel batch style -- re-running the whole ladder
        # here used to double the worst-case wait (each ladder can take up
        # to ~30s per TF on the slow-provider path), which was enough to
        # blow past the outer timeout in bot.py. Topping up just the gaps
        # keeps this bounded to roughly one more slowest-fetch, not two.
        have = {f["tf"] for f in frames}
        missing = [(tf_code, tf_label) for tf_code, tf_label in ALT_LADDER if tf_code not in have]
        if missing:
            extra = _fetch_ladder_parallel(symbol, missing)
            by_tf = {f["tf"]: f for f in frames}
            for f in extra:
                by_tf[f["tf"]] = f
            # Preserve ALT_LADDER's HTF-first order for downstream code
            # (frames[0] must stay the highest timeframe available).
            order = [tf for tf, _ in ALT_LADDER] if len(extra) >= len(frames) else [tf for tf, _ in TOPDOWN_LADDER]
            ordered = [by_tf[tf] for tf in order if tf in by_tf]
            # Include anything fetched that fell outside the chosen order list.
            for tf, f in by_tf.items():
                if f not in ordered:
                    ordered.append(f)
            frames = ordered
    if not frames:
        return {
            "error": (
                f"No market data for {symbol}. "
                "On Render, MT5 is unavailable — set TWELVE_DATA_API_KEY or check yfinance access."
            )
        }


    htf = frames[0]
    overall_bias = htf["ema200_bias"]
    if htf.get("trendline") and htf["trendline"]["bias"] != "NEUTRAL":
        if htf["trendline"]["bias"] == overall_bias or overall_bias == "NEUTRAL":
            overall_bias = htf["trendline"]["bias"]

    biases = [f["ema200_bias"] for f in frames if f["ema200_bias"] != "NEUTRAL"]
    if not biases:
        alignment = "MIXED"
    else:
        buy_c = sum(1 for b in biases if b == "BUY")
        sell_c = sum(1 for b in biases if b == "SELL")
        if buy_c == len(biases):
            alignment = "ALIGNED BULLISH"
        elif sell_c == len(biases):
            alignment = "ALIGNED BEARISH"
        elif buy_c > sell_c:
            alignment = "MOSTLY BULLISH"
        elif sell_c > buy_c:
            alignment = "MOSTLY BEARISH"
        else:
            alignment = "MIXED"

    primary_proj = None
    if htf.get("trendline"):
        t = htf["trendline"]
        if overall_bias == "BUY":
            primary_proj = {"direction": "UP", "target": t["proj_up"], "invalidation": t["lower"]}
        elif overall_bias == "SELL":
            primary_proj = {"direction": "DOWN", "target": t["proj_down"], "invalidation": t["upper"]}

    pairs = pair_idm_with_extreme_ob(htf.get("inducements") or [], htf.get("order_blocks") or [])
    ltf = frames[-1]
    allowed, reason, pref = structure_trade_permission(
        overall_bias, ltf.get("structure") or {}
    )

    return {
        "symbol": symbol,
        "overall_bias": overall_bias,
        "alignment": alignment,
        "htf_regime": htf["ema200_note"],
        "frames": frames,
        "primary_projection": primary_proj,
        "htf_trendline": htf.get("trendline"),
        "htf_vwap": htf.get("vwap"),
        "htf_poc": htf["volume_profile"]["poc_price"] if htf.get("volume_profile") else None,
        "idm_ob_pairs": pairs,
        "structure_allowed": allowed,
        "structure_reason": reason,
        "structure_prefer": pref,
        # Chart payload (HTF preferred for institutional map; LTF for entry view)
        "chart_frame": htf,
    }


def format_institutional_report(analysis):
    """Vital-info only. Chart carries the visual detail."""
    if "error" in analysis:
        return analysis["error"]

    symbol = analysis["symbol"]
    bias = analysis["overall_bias"]
    align = analysis["alignment"]
    htf = analysis["frames"][0]
    lines = []

    lines.append(f"🏛 {symbol}  |  Bias: {bias}  |  {align}")
    lines.append(f"HTF: {analysis.get('htf_regime', '—')}")

    bits = []
    for f in analysis["frames"]:
        ev = (f.get("structure") or {}).get("last_event") or "—"
        bits.append(f"{f['tf_label']}:{ev}")
    lines.append("Struct: " + " · ".join(bits))

    keys = []
    if htf.get("vwap") and isinstance(htf["vwap"], dict):
        keys.append(f"VWAP {htf['vwap'].get('vwap', 0):.5f}")
    if analysis.get("htf_poc"):
        keys.append(f"POC {analysis['htf_poc']:.5f}")
    if htf.get("trendline"):
        t = htf["trendline"]
        keys.append(f"Ch {t.get('lower', 0):.5f}/{t.get('upper', 0):.5f}")
    if keys:
        lines.append("Levels: " + " | ".join(keys))

    proj = analysis.get("primary_projection")
    if proj:
        lines.append(
            f"Proj: {proj.get('direction')} → {proj.get('target', 0):.5f}  "
            f"(inv {proj.get('invalidation', 0):.5f})"
        )

    n_fvg = len(htf.get("fvgs") or [])
    n_ob = len(htf.get("order_blocks") or [])
    n_idm = len(htf.get("inducements") or [])
    n_base = len(htf.get("base_zones") or [])
    unmit_idm = sum(1 for z in (htf.get("inducements") or []) if not z.get("mitigated"))
    lines.append(f"Zones: {n_fvg} FVG · {n_ob} OB · {n_idm} IDM · {n_base} Base ({unmit_idm} IDM open)")


    pairs = analysis.get("idm_ob_pairs") or []
    if pairs:
        lines.append(f"Setup: {pairs[0].get('direction')} IDM→OB")
    else:
        lines.append("Setup: none clean")

    allowed = analysis.get("structure_allowed")
    prefer = analysis.get("structure_prefer", "—")
    lines.append(f"Permission: {'YES' if allowed else 'WAIT'} → {prefer}")
    if analysis.get("structure_reason"):
        lines.append(f"  {analysis['structure_reason']}")

    if htf.get("best_pattern"):
        p = htf["best_pattern"]
        conf = getattr(p, "confidence", 0)
        lines.append(f"Pattern: {getattr(p, 'name', '?')} ({getattr(p, 'bias', '?')}) {conf:.0f}%")

    return "\n".join(lines)

