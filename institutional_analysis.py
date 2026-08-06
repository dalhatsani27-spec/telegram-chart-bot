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
from direction_banner import direction_banner
from smc_zones import (
    detect_fvgs, detect_order_blocks, detect_inducement_zones,
    pair_idm_with_extreme_ob, summarise_smc_zones, build_bos_events,
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
    df = mt5_data.fetch_candles(symbol, tf_code, count=250)
    if df is None or df.empty or len(df) < 40:
        return None
    ema_bias, ema_note, ema_dist = _ema200_bias(df)
    vwap = _vwap_context(df)
    trend = _fit_trendline_family(df)
    vp = compute_volume_profile(df.iloc[:-1])
    best, all_pats = scan_all_patterns(df.iloc[:-1], volume_profile=vp)
    structure = analyse_structure(df, left=3, right=3, lookback=70)
    fvgs = detect_fvgs(df, min_gap_atr=0.15, max_zones=5)
    obs = detect_order_blocks(df, structure=structure, max_zones=4)
    idms = detect_inducement_zones(df, max_zones=4)
    bos_events = build_bos_events(df, max_events=8)
    return {
        "tf": tf_code, "tf_label": tf_label, "df": df,
        "close": float(df["Close"].iloc[-1]),
        "ema200_bias": ema_bias, "ema200_note": ema_note, "ema200_dist": ema_dist,
        "vwap": vwap, "trendline": trend, "volume_profile": vp,
        "best_pattern": best, "all_patterns": all_pats[:3] if all_pats else [],
        "structure": structure, "fvgs": fvgs, "order_blocks": obs, "inducements": idms,
        "bos_events": bos_events,
    }


def run_topdown_analysis(symbol):
    symbol = symbol.strip().upper()
    frames = []
    for tf_code, tf_label in TOPDOWN_LADDER:
        snap = _analyse_single_tf(symbol, tf_code, tf_label)
        if snap:
            frames.append(snap)
    if len(frames) < 2:
        frames = []
        for tf_code, tf_label in ALT_LADDER:
            snap = _analyse_single_tf(symbol, tf_code, tf_label)
            if snap:
                frames.append(snap)
    if not frames:
        return {"error": f"No data for {symbol}."}

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
    """SHORT summary — chart shows the zones."""
    if "error" in analysis:
        return analysis["error"]

    symbol = analysis["symbol"]
    bias = analysis["overall_bias"]
    align = analysis["alignment"]
    htf = analysis["frames"][0]
    lines = []

    lines.append(f"🏛 {symbol}  |  {align}")
    lines.append(direction_banner(bias, extra=symbol))
    lines.append(f"HTF: {analysis['htf_regime']}")

    # One-line structure per TF
    bits = []
    for f in analysis["frames"]:
        ev = (f.get("structure") or {}).get("last_event") or "—"
        bits.append(f"{f['tf_label']}: {ev}")
    lines.append("Struct: " + " · ".join(bits))

    # Key levels only (not every zone price)
    keys = []
    if htf.get("vwap"):
        keys.append(f"VWAP {htf['vwap']['vwap']:.5f}")
    if analysis.get("htf_poc"):
        keys.append(f"POC {analysis['htf_poc']:.5f}")
    if htf.get("trendline"):
        t = htf["trendline"]
        keys.append(f"Ch {t['lower']:.5f}/{t['upper']:.5f}")
    if keys:
        lines.append("Levels: " + " | ".join(keys))

    proj = analysis.get("primary_projection")
    if proj:
        lines.append(f"Proj: {proj['direction']} → {proj['target']:.5f}  (inv {proj['invalidation']:.5f})")

    # Zone counts (details are on the chart)
    n_fvg = len(htf.get("fvgs") or [])
    n_ob = len(htf.get("order_blocks") or [])
    n_idm = len(htf.get("inducements") or [])
    unmit_idm = sum(1 for z in (htf.get("inducements") or []) if not z.get("mitigated"))
    lines.append(f"Zones: {n_fvg} FVG · {n_ob} OB · {n_idm} IDM ({unmit_idm} unmitigated)")

    pairs = analysis.get("idm_ob_pairs") or []
    if pairs:
        p = pairs[0]
        lines.append(f"Setup: {p['direction']} — IDM→OB (see chart)")
    else:
        lines.append("Setup: no clean IDM→OB pair")

    allowed = analysis.get("structure_allowed")
    lines.append(
        f"Permission: {'YES' if allowed else 'WAIT'} — {analysis.get('structure_prefer', 'n/a')}"
    )
    if analysis.get("structure_reason"):
        lines.append(f"  {analysis['structure_reason']}")

    if htf.get("best_pattern"):
        p = htf["best_pattern"]
        lines.append(f"Pattern: {p.name} ({p.bias}) {p.confidence:.0f}%")

    lines.append("📷 Chart = full story (FVG/OB/IDM/EMA/structure)")
    return "\n".join(lines)
