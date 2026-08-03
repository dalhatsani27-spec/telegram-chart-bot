"""
institutional_analysis.py
=========================
True Top-Down Institutional Analysis.

Priority stack:
  1. 200 EMA on HTF          → Who is ruling the market (bulls / bears)
  2. Trendline Families      → Primary structure + measured projections
  3. VWAP                    → Dynamic support / resistance
  4. Volume Profile          → POC + Value Area (high-probability zones)
  5. Chart Patterns          → Confluence only (Double Top/Bottom, Flags, etc.)

Designed for the Telegram "Run Institutional Analysis" button.
Produces a clean multi-timeframe bias + key levels + projection targets.
"""

import numpy as np
import pandas as pd
from patterns import scan_all_patterns, find_pivots, _atr, Pattern
from volume_profile import compute_volume_profile
import mt5_data

# Timeframe ladder for top-down (highest → lowest)
TOPDOWN_LADDER = [
    ("4h",   "4 Hour"),
    ("1h",   "1 Hour"),
    ("30min", "30 Minute"),
]

# Fallback if 4h data is thin
ALT_LADDER = [
    ("1h",   "1 Hour"),
    ("30min", "30 Minute"),
    ("15min", "15 Minute"),
]


def _ema200_bias(df):
    """Returns ('BUY'|'SELL'|'NEUTRAL', description, distance_pct)."""
    if df is None or df.empty or 'EMA200' not in df.columns:
        return "NEUTRAL", "EMA200 unavailable", 0.0
    close = float(df['Close'].iloc[-1])
    ema200 = float(df['EMA200'].iloc[-1])
    if ema200 <= 0:
        return "NEUTRAL", "EMA200 unavailable", 0.0
    dist = (close - ema200) / ema200 * 100.0
    if close > ema200 * 1.001:
        return "BUY", f"Price above 200 EMA (+{dist:.2f}%) — bulls in control", dist
    if close < ema200 * 0.999:
        return "SELL", f"Price below 200 EMA ({dist:.2f}%) — bears in control", dist
    return "NEUTRAL", f"Price at 200 EMA ({dist:+.2f}%) — equilibrium", dist


def _vwap_context(df):
    """Returns dict with vwap, position, and note."""
    if df is None or df.empty or 'VWAP' not in df.columns:
        return None
    close = float(df['Close'].iloc[-1])
    vwap = float(df['VWAP'].iloc[-1])
    if vwap <= 0:
        return None
    dist_pct = (close - vwap) / vwap * 100.0
    if close > vwap * 1.0005:
        pos = "ABOVE"
        note = f"Price above VWAP (+{dist_pct:.2f}%) — dynamic support"
    elif close < vwap * 0.9995:
        pos = "BELOW"
        note = f"Price below VWAP ({dist_pct:.2f}%) — dynamic resistance"
    else:
        pos = "AT"
        note = f"Price at VWAP ({dist_pct:+.2f}%) — balance zone"
    return {"vwap": vwap, "position": pos, "distance_pct": dist_pct, "note": note}


def _fit_trendline_family(df, lookback=80):
    """
    Build a simple trendline family from recent pivot highs and lows.
    Returns upper/lower lines + channel height for measured-move projections.
    """
    if df is None or len(df) < 30:
        return None
    ph, pl = find_pivots(df, left=4, right=4)
    n = len(df)
    start = max(0, n - lookback)
    recent_ph = [p for p in ph if p >= start][-4:]
    recent_pl = [p for p in pl if p >= start][-4:]
    if len(recent_ph) < 2 or len(recent_pl) < 2:
        return None

    upper_pts = [(p, float(df['High'].iloc[p])) for p in recent_ph]
    lower_pts = [(p, float(df['Low'].iloc[p])) for p in recent_pl]

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
    close = float(df['Close'].iloc[-1])
    mid = (upper_now + lower_now) / 2.0
    pos = (close - lower_now) / height if height > 0 else 0.5

    # Simple regime
    avg_price = float(df['Close'].tail(lookback).mean()) or 1.0
    up_norm = (up_slope * lookback) / avg_price
    lo_norm = (lo_slope * lookback) / avg_price
    FLAT = 0.004

    if abs(up_norm) < FLAT and lo_norm > FLAT:
        family = "Ascending Channel"
        bias = "BUY"
    elif up_norm < -FLAT and abs(lo_norm) < FLAT:
        family = "Descending Channel"
        bias = "SELL"
    elif up_norm > FLAT and lo_norm > FLAT:
        family = "Rising Channel / Parallel"
        bias = "BUY"
    elif up_norm < -FLAT and lo_norm < -FLAT:
        family = "Falling Channel / Parallel"
        bias = "SELL"
    else:
        family = "Range / Contracting"
        bias = "NEUTRAL"

    # Measured move projections from channel height
    proj_up = upper_now + height
    proj_down = lower_now - height

    return {
        "family": family,
        "bias": bias,
        "upper": float(upper_now),
        "lower": float(lower_now),
        "mid": float(mid),
        "height": float(height),
        "position": float(np.clip(pos, 0.0, 1.0)),
        "proj_up": float(proj_up),
        "proj_down": float(proj_down),
        "upper_pts": upper_pts,
        "lower_pts": lower_pts,
    }


def _analyse_single_tf(symbol, tf_code, tf_label):
    """Full single-timeframe institutional snapshot."""
    df = mt5_data.fetch_candles(symbol, tf_code, count=250)
    if df is None or df.empty or len(df) < 40:
        return None

    ema_bias, ema_note, ema_dist = _ema200_bias(df)
    vwap = _vwap_context(df)
    trend = _fit_trendline_family(df)
    vp = compute_volume_profile(df.iloc[:-1])
    best, all_pats = scan_all_patterns(df.iloc[:-1], volume_profile=vp)

    return {
        "tf": tf_code,
        "tf_label": tf_label,
        "df": df,
        "close": float(df['Close'].iloc[-1]),
        "ema200_bias": ema_bias,
        "ema200_note": ema_note,
        "ema200_dist": ema_dist,
        "vwap": vwap,
        "trendline": trend,
        "volume_profile": vp,
        "best_pattern": best,
        "all_patterns": all_pats[:3] if all_pats else [],
    }


def run_topdown_analysis(symbol):
    """
    Main entry point for Institutional Top-Down Analysis.
    Returns a structured dict ready for professional Telegram display.
    """
    symbol = symbol.strip().upper()
    frames = []
    for tf_code, tf_label in TOPDOWN_LADDER:
        snap = _analyse_single_tf(symbol, tf_code, tf_label)
        if snap:
            frames.append(snap)

    if len(frames) < 2:
        # fallback ladder
        frames = []
        for tf_code, tf_label in ALT_LADDER:
            snap = _analyse_single_tf(symbol, tf_code, tf_label)
            if snap:
                frames.append(snap)

    if not frames:
        return {"error": f"Unable to retrieve sufficient data for {symbol}."}

    # Highest TF drives overall regime
    htf = frames[0]
    overall_bias = htf["ema200_bias"]
    if htf.get("trendline") and htf["trendline"]["bias"] != "NEUTRAL":
        # Trendline family can reinforce or slightly override pure EMA
        if htf["trendline"]["bias"] == overall_bias or overall_bias == "NEUTRAL":
            overall_bias = htf["trendline"]["bias"]

    # Alignment score across frames
    biases = [f["ema200_bias"] for f in frames if f["ema200_bias"] != "NEUTRAL"]
    if not biases:
        alignment = "MIXED"
    else:
        buy_count = sum(1 for b in biases if b == "BUY")
        sell_count = sum(1 for b in biases if b == "SELL")
        if buy_count == len(biases):
            alignment = "FULLY ALIGNED BULLISH"
        elif sell_count == len(biases):
            alignment = "FULLY ALIGNED BEARISH"
        elif buy_count > sell_count:
            alignment = "MOSTLY BULLISH"
        elif sell_count > buy_count:
            alignment = "MOSTLY BEARISH"
        else:
            alignment = "MIXED / CONFLICTING"

    # Key levels (collect from all frames)
    levels = []
    for f in frames:
        if f.get("trendline"):
            t = f["trendline"]
            levels.append({"tf": f["tf_label"], "type": "Trendline Upper", "price": t["upper"]})
            levels.append({"tf": f["tf_label"], "type": "Trendline Lower", "price": t["lower"]})
            levels.append({"tf": f["tf_label"], "type": "Projection Up", "price": t["proj_up"]})
            levels.append({"tf": f["tf_label"], "type": "Projection Down", "price": t["proj_down"]})
        if f.get("vwap"):
            levels.append({"tf": f["tf_label"], "type": "VWAP", "price": f["vwap"]["vwap"]})
        if f.get("volume_profile"):
            vp = f["volume_profile"]
            levels.append({"tf": f["tf_label"], "type": "POC", "price": vp["poc_price"]})
            levels.append({"tf": f["tf_label"], "type": "VA High", "price": vp["value_area_high"]})
            levels.append({"tf": f["tf_label"], "type": "VA Low", "price": vp["value_area_low"]})
        if f.get("best_pattern"):
            p = f["best_pattern"]
            levels.append({"tf": f["tf_label"], "type": f"{p.name} Trigger", "price": p.trigger_price})

    # Primary projection from HTF trendline family
    primary_proj = None
    if htf.get("trendline"):
        t = htf["trendline"]
        if overall_bias == "BUY":
            primary_proj = {"direction": "UP", "target": t["proj_up"], "invalidation": t["lower"]}
        elif overall_bias == "SELL":
            primary_proj = {"direction": "DOWN", "target": t["proj_down"], "invalidation": t["upper"]}

    return {
        "symbol": symbol,
        "overall_bias": overall_bias,
        "alignment": alignment,
        "htf_regime": htf["ema200_note"],
        "frames": frames,
        "key_levels": levels,
        "primary_projection": primary_proj,
        "htf_trendline": htf.get("trendline"),
        "htf_vwap": htf.get("vwap"),
        "htf_poc": htf["volume_profile"]["poc_price"] if htf.get("volume_profile") else None,
    }


def format_institutional_report(analysis):
    """
    Professional plain-text report for Telegram.
    Clean hierarchy, easy to read on mobile.
    """
    if "error" in analysis:
        return analysis["error"]

    symbol = analysis["symbol"]
    bias = analysis["overall_bias"]
    align = analysis["alignment"]
    lines = []

    lines.append(f"══════════════════════════════════")
    lines.append(f"  INSTITUTIONAL TOP-DOWN ANALYSIS")
    lines.append(f"  {symbol}")
    lines.append(f"══════════════════════════════════")
    lines.append("")
    lines.append(f"Overall Bias : {bias}")
    lines.append(f"Alignment    : {align}")
    lines.append(f"HTF Regime   : {analysis['htf_regime']}")
    lines.append("")

    # Timeframe breakdown
    lines.append("── TIMEFRAME STRUCTURE ──")
    for f in analysis["frames"]:
        lines.append(f"")
        lines.append(f"▶ {f['tf_label']}")
        lines.append(f"  200 EMA : {f['ema200_note']}")
        if f.get("trendline"):
            t = f["trendline"]
            lines.append(f"  Trendline Family : {t['family']} ({t['bias']})")
            lines.append(f"  Channel          : {t['lower']:.5f} → {t['upper']:.5f}")
            lines.append(f"  Position in ch.  : {t['position']*100:.0f}%")
        if f.get("vwap"):
            lines.append(f"  VWAP             : {f['vwap']['note']}")
        if f.get("volume_profile"):
            vp = f["volume_profile"]
            lines.append(f"  POC              : {vp['poc_price']:.5f}")
            lines.append(f"  Value Area       : {vp['value_area_low']:.5f} – {vp['value_area_high']:.5f}")
        if f.get("best_pattern"):
            p = f["best_pattern"]
            lines.append(f"  Pattern          : {p.name} ({p.bias}) conf {p.confidence:.0f}%")
            lines.append(f"  Trigger          : {p.trigger_price:.5f}")

    lines.append("")
    lines.append("── PRIMARY PROJECTION ──")
    proj = analysis.get("primary_projection")
    if proj:
        lines.append(f"  Direction     : {proj['direction']}")
        lines.append(f"  Target        : {proj['target']:.5f}")
        lines.append(f"  Invalidation  : {proj['invalidation']:.5f}")
    else:
        lines.append("  No clear measured projection (range / mixed structure)")

    # High-value limit order zones (POC + VA)
    lines.append("")
    lines.append("── HIGH-PROBABILITY ZONES (Limit candidates) ──")
    htf = analysis["frames"][0]
    if htf.get("volume_profile"):
        vp = htf["volume_profile"]
        lines.append(f"  POC (magnet)     : {vp['poc_price']:.5f}")
        lines.append(f"  Value Area High  : {vp['value_area_high']:.5f}")
        lines.append(f"  Value Area Low   : {vp['value_area_low']:.5f}")
    if htf.get("vwap"):
        lines.append(f"  VWAP             : {htf['vwap']['vwap']:.5f}")
    if htf.get("trendline"):
        t = htf["trendline"]
        lines.append(f"  Trendline Lower  : {t['lower']:.5f}")
        lines.append(f"  Trendline Upper  : {t['upper']:.5f}")

    lines.append("")
    lines.append("══════════════════════════════════")
    lines.append("Trendline family + 200 EMA drive bias.")
    lines.append("POC / VWAP / Channel edges = preferred limit zones.")
    lines.append("══════════════════════════════════")

    return "\n".join(lines)
