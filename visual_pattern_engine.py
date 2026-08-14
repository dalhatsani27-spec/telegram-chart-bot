"""Visual geometric pattern engine for Trendline charts.

The detector works from the *shape* of price action: local extrema, fitted
upper/lower rails, slope, parallelism, convergence and impulse/correction
geometry.  It does not use HH/LH/HL/LL market-structure labels to name a
pattern.  The renderer draws the same geometric primitives a trader would
put on the chart: two rails, horizontal boundaries, triangles/wedges and
flag/pennant corrections.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

PATTERNS = [
    "Uptrend Line", "Downtrend Line", "Horizontal Support & Resistance",
    "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle",
    "Rising Wedge", "Falling Wedge", "Ascending Channel", "Descending Channel",
    "Horizontal Channel", "Bull Flag", "Bear Flag", "Pennant",
]

BULL = {"Uptrend Line", "Ascending Triangle", "Rising Wedge", "Ascending Channel", "Bull Flag"}
BEAR = {"Downtrend Line", "Descending Triangle", "Falling Wedge", "Descending Channel", "Bear Flag"}
NEUTRAL = {"Horizontal Support & Resistance", "Symmetrical Triangle", "Horizontal Channel", "Pennant"}


def _atr(df: pd.DataFrame) -> np.ndarray:
    if "ATR" in df.columns:
        a = pd.to_numeric(df["ATR"], errors="coerce")
        fallback = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
        return a.fillna(fallback).to_numpy(float)
    return (df["High"] - df["Low"]).rolling(14, min_periods=1).mean().to_numpy(float)


def _pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[Dict[str, Any]]:
    """Local extrema used only as geometric anchor points."""
    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(df["High"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(df["Low"], errors="coerce").to_numpy(float)
    out: List[Dict[str, Any]] = []
    for i in range(left, len(df) - right):
        w = close[i - left:i + right + 1]
        if not np.isfinite(close[i]):
            continue
        if close[i] >= np.max(w) and close[i] > np.min(w):
            out.append({"i": i, "p": float(high[i]), "close": float(close[i]), "t": "H"})
        elif close[i] <= np.min(w) and close[i] < np.max(w):
            out.append({"i": i, "p": float(low[i]), "close": float(close[i]), "t": "L"})
    return out


def _fit(points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(points) < 2:
        return None
    x = np.array([p["i"] for p in points], dtype=float)
    y = np.array([p["p"] for p in points], dtype=float)
    m, b = np.polyfit(x, y, 1)
    residual = float(np.mean(np.abs(y - (m * x + b))))
    return {
        "m": float(m), "b": float(b), "x0": int(x[0]), "x1": int(x[-1]),
        "y0": float(m * x[0] + b), "y1": float(m * x[-1] + b),
        "err": residual, "n": len(points), "pts": points,
    }


def _v(line: Dict[str, Any], x: float) -> float:
    return line["m"] * x + line["b"]


def _line_candidates(pivots: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    pts = [p for p in pivots if p["t"] == kind]
    if len(pts) < 2:
        return []
    candidates: List[Dict[str, Any]] = []
    # Recent windows let the detector follow the visible shape instead of
    # fitting one line across unrelated old price action.
    for count in (2, 3, 4, 5, 6):
        if len(pts) >= count:
            line = _fit(pts[-count:])
            if line:
                candidates.append(line)
    candidates.sort(key=lambda z: (z["n"], -z["err"]), reverse=True)
    return candidates


def _best_pair(df: pd.DataFrame, pivots: List[Dict[str, Any]]) -> Optional[Tuple[float, Dict, Dict]]:
    lows = _line_candidates(pivots, "L")
    highs = _line_candidates(pivots, "H")
    best = None
    for lower in lows:
        for upper in highs:
            start = max(lower["x0"], upper["x0"])
            end = min(lower["x1"], upper["x1"])
            if end <= start + 4:
                continue
            gap0 = _v(upper, start) - _v(lower, start)
            gap1 = _v(upper, end) - _v(lower, end)
            if gap0 <= 0 or gap1 <= 0:
                continue
            convergence = 1.0 - gap1 / max(gap0, 1e-9)
            if convergence < 0.05:
                continue
            # Penalise poorly fitting rails.
            scale = max(float(df["High"].max() - df["Low"].min()), 1e-9)
            fit = max(0.0, 1.0 - (lower["err"] + upper["err"]) / scale * 12.0)
            score = convergence * 0.70 + fit * 0.30
            if best is None or score > best[0]:
                best = (score, lower, upper)
    return best


def _impulse(df: pd.DataFrame, pivots: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find a displacement leg followed by a compact correction window."""
    if len(pivots) < 4:
        return None
    atr = _atr(df)
    best = None
    for a, b in zip(pivots[:-1], pivots[1:]):
        bars = b["i"] - a["i"]
        if bars < 3:
            continue
        av = float(np.nanmedian(atr[max(0, a["i"] - 10):min(len(df), b["i"] + 2)]))
        av = max(av, 1e-9)
        displacement = abs(b["p"] - a["p"]) / av
        if best is None or displacement > best["score"]:
            best = {"score": displacement, "a": a, "b": b}
    return best if best and best["score"] >= 2.0 else None


def _classify_pair(df: pd.DataFrame, lower: Dict, upper: Dict) -> str:
    scale = max(float(np.nanmedian(_atr(df))), 1e-9)
    sm = (lower["m"] + upper["m"]) / 2.0
    # Slope thresholds are expressed in ATR per bar, not raw price units.
    lo_s = lower["m"] / scale
    hi_s = upper["m"] / scale
    slope_eps = 0.08

    if abs(lo_s) < slope_eps and abs(hi_s) < slope_eps:
        return "Horizontal Channel"
    if lo_s > slope_eps and hi_s > slope_eps:
        if lo_s > hi_s * 1.18:
            return "Rising Wedge"
        if abs(lo_s - hi_s) / max(abs(lo_s), abs(hi_s), 1e-9) < 0.28:
            return "Ascending Channel"
        return "Rising Wedge"
    if lo_s < -slope_eps and hi_s < -slope_eps:
        if abs(lo_s) > abs(hi_s) * 1.18:
            return "Falling Wedge"
        if abs(lo_s - hi_s) / max(abs(lo_s), abs(hi_s), 1e-9) < 0.28:
            return "Descending Channel"
        return "Falling Wedge"
    if hi_s < -slope_eps and lo_s > slope_eps:
        return "Symmetrical Triangle"
    if abs(hi_s) < slope_eps and lo_s > slope_eps:
        return "Ascending Triangle"
    if abs(lo_s) < slope_eps and hi_s < -slope_eps:
        return "Descending Triangle"
    if sm > 0:
        return "Ascending Triangle"
    if sm < 0:
        return "Descending Triangle"
    return "Symmetrical Triangle"


def _flag_or_pennant(df: pd.DataFrame, pivots: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    impulse = _impulse(df, pivots)
    if not impulse:
        return None
    a, b = impulse["a"], impulse["b"]
    # Correction must occupy the rightmost part of the chart and be much
    # smaller than the impulse.
    corr = [p for p in pivots if p["i"] > b["i"]]
    if len(corr) < 4:
        return None
    lows = [p for p in corr if p["t"] == "L"]
    highs = [p for p in corr if p["t"] == "H"]
    if len(lows) < 2 or len(highs) < 2:
        return None
    lower = _fit(lows[-4:])
    upper = _fit(highs[-4:])
    if not lower or not upper:
        return None
    start = max(lower["x0"], upper["x0"]); end = min(lower["x1"], upper["x1"])
    if end <= start + 3:
        return None
    g0 = _v(upper, start) - _v(lower, start)
    g1 = _v(upper, end) - _v(lower, end)
    if g0 <= 0 or g1 <= 0:
        return None
    convergence = 1 - g1 / max(g0, 1e-9)
    impulse_dir = "BULL" if b["p"] > a["p"] else "BEAR"
    if convergence >= 0.20:
        name = "Pennant"
    else:
        # A flag is a compact parallel channel against the impulse.
        slope_diff = abs(lower["m"] - upper["m"]) / max(abs(lower["m"]), abs(upper["m"]), 1e-9)
        if slope_diff > 0.45:
            return None
        correction_slope = (lower["m"] + upper["m"]) / 2
        if impulse_dir == "BULL" and correction_slope >= 0:
            return None
        if impulse_dir == "BEAR" and correction_slope <= 0:
            return None
        name = "Bull Flag" if impulse_dir == "BULL" else "Bear Flag"
    return {"name": name, "confidence": min(96, int(65 + impulse["score"] * 5 + convergence * 20)),
            "lines": [lower, upper], "impulse": impulse, "pivots": pivots, "convergence": convergence}


def detect_visual_pattern(df: pd.DataFrame) -> Dict[str, Any]:
    """Return the strongest visible geometric pattern on the current chart."""
    if df is None or len(df) < 30:
        return {"name": "None", "confidence": 0, "lines": [], "pivots": []}
    piv = _pivots(df)
    if len(piv) < 4:
        return {"name": "None", "confidence": 0, "lines": [], "pivots": piv}

    flag = _flag_or_pennant(df, piv)
    pair = _best_pair(df, piv)
    candidates = []
    if flag:
        candidates.append(flag)
    if pair:
        _, lower, upper = pair
        name = _classify_pair(df, lower, upper)
        confidence = min(98, int(58 + pair[0] * 35 + min(lower["n"] + upper["n"], 10) * 1.4))
        candidates.append({"name": name, "confidence": confidence, "lines": [lower, upper],
                           "pivots": piv, "convergence": pair[0]})

    # Single clean diagonal / horizontal line when no two-rail shape exists.
    lows = _line_candidates(piv, "L")
    highs = _line_candidates(piv, "H")
    for line in (lows[0] if lows else None, highs[0] if highs else None):
        if line:
            atr = max(float(np.nanmedian(_atr(df))), 1e-9)
            normalized = abs(line["m"]) / atr
            if normalized < 0.08:
                name = "Horizontal Support & Resistance"
            elif line["m"] > 0:
                name = "Uptrend Line"
            else:
                name = "Downtrend Line"
            candidates.append({"name": name, "confidence": min(88, 54 + line["n"] * 8),
                               "lines": [line], "pivots": piv})

    if not candidates:
        return {"name": "None", "confidence": 0, "lines": [], "pivots": piv}
    # Prefer specific two-rail geometry over a generic single line.
    candidates.sort(key=lambda x: (len(x.get("lines", [])) == 2, x.get("confidence", 0)), reverse=True)
    return candidates[0]


def _draw_line(ax, line: Dict, offset: int, chart_len: int, color: str, lw: float = 2.2, ls: str = "-"):
    x0 = max(0, int(line["x0"]) - offset)
    x1 = min(chart_len - 1, int(line["x1"]) - offset)
    if x1 <= x0:
        return
    xs = np.arange(x0, x1 + 1)
    orig = xs + offset
    ax.plot(xs, [_v(line, x) for x in orig], color=color, lw=lw, ls=ls, zorder=8, solid_capstyle="round")


def render_trendline_map(df: pd.DataFrame, symbol: str, setup: Dict[str, Any], title_suffix: str = ""):
    """Render a clean visual-pattern chart. No H/L/LH/HH labels are drawn."""
    family = setup.get("family") or setup.get("analysis") or setup
    data = family.get("df") if isinstance(family, dict) else None
    if data is not None:
        df = data
    chart_len = min(140, len(df))
    chart_df = df.tail(chart_len).copy()
    offset = len(df) - chart_len

    pattern = family.get("visual_pattern") or detect_visual_pattern(chart_df)
    name = pattern.get("name", "None")
    conf = int(pattern.get("confidence", 0) or 0)
    side = str(setup.get("direction") or family.get("direction") or "NEUTRAL").upper()

    mc = mpf.make_marketcolors(up="#089981", down="#f23645", edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridcolor="#1f2937",
                               y_on_right=True, facecolor="#0d1117", figcolor="#0d1117")
    fig = plt.figure(figsize=(15.4, 8.0), facecolor="#0d1117")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.6, 1.0], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0]); panel = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("#0d1117")
    mpf.plot(chart_df, type="candle", style=style, volume=False, ax=ax)

    # The pattern is literally drawn from its fitted rails.
    lines = pattern.get("lines") or []
    if name in BULL:
        colors = ["#00e676", "#00e676"]
    elif name in BEAR:
        colors = ["#ff1744", "#ff1744"]
    else:
        colors = ["#e5e7eb", "#e5e7eb"]
    if name in {"Ascending Triangle", "Descending Triangle", "Symmetrical Triangle", "Rising Wedge", "Falling Wedge", "Pennant"} and len(lines) >= 2:
        colors = ["#00e676", "#ff1744"]
    for i, line in enumerate(lines[:2]):
        _draw_line(ax, line, offset, chart_len, colors[i % len(colors)])

    # Draw a horizontal neckline/boundary cleanly for triangle-like patterns
    if name == "Ascending Triangle" and len(lines) >= 2:
        upper = lines[1]; y = float(np.median([p["p"] for p in upper["pts"]]))
        ax.axhline(y, color="#ff1744", lw=2.0, ls="-", alpha=0.9, zorder=8)
    elif name == "Descending Triangle" and len(lines) >= 2:
        lower = lines[0]; y = float(np.median([p["p"] for p in lower["pts"]]))
        ax.axhline(y, color="#00e676", lw=2.0, ls="-", alpha=0.9, zorder=8)

    # Optional breakout marker from the geometric boundary, not from HH/LL labels.
    if name in {"Ascending Triangle", "Descending Triangle", "Symmetrical Triangle", "Rising Wedge", "Falling Wedge", "Pennant"} and len(lines) >= 2:
        last_x = len(df) - 1
        upper_y = _v(lines[1], last_x); lower_y = _v(lines[0], last_x)
        close = float(df["Close"].iloc[-1])
        if close > upper_y:
            ax.annotate("BREAKOUT", xy=(chart_len-1, close), xytext=(chart_len-18, close + (chart_df["High"].max()-chart_df["Low"].min())*0.08),
                        color="#00e676", fontsize=8, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#00e676", lw=1.5), zorder=12)
        elif close < lower_y:
            ax.annotate("BREAKOUT", xy=(chart_len-1, close), xytext=(chart_len-18, close - (chart_df["High"].max()-chart_df["Low"].min())*0.08),
                        color="#ff1744", fontsize=8, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#ff1744", lw=1.5), zorder=12)

    title_color = "#00e676" if side in {"BUY", "LONG", "BULLISH"} else "#ff1744" if side in {"SELL", "SHORT", "BEARISH"} else "#e5e7eb"
    ax.text(0.015, 0.97, f"{symbol} | M30 | {side} BIAS | {name}", transform=ax.transAxes,
            fontsize=11, color="#ffffff", fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=title_color, edgecolor="none", alpha=0.88), zorder=15)

    # Clean side panel: pattern, confidence and geometric evidence.
    panel.set_facecolor("#0d1117"); panel.set_xticks([]); panel.set_yticks([])
    for s in panel.spines.values(): s.set_visible(False)
    panel.text(0.05, 0.94, "VISUAL PATTERN", color="#e5e7eb", fontsize=10, fontweight="bold")
    panel.text(0.05, 0.87, name, color=title_color, fontsize=13, fontweight="bold")
    panel.text(0.05, 0.81, f"Confidence: {conf}%", color="#00e676" if conf >= 75 else "#ffab00", fontsize=10, fontweight="bold")
    panel.text(0.05, 0.72, "DETECTED FROM GEOMETRY", color="#e5e7eb", fontsize=8, fontweight="bold")
    evidence = [
        f"Rails: {len(lines)}",
        f"Convergence: {pattern.get('convergence', 0)*100:.0f}%" if pattern.get("convergence") is not None else "Convergence: —",
        "Slope relationship: measured",
        "Touch/fit quality: measured",
        "Market-structure labels: OFF",
    ]
    y = 0.67
    for text in evidence:
        panel.text(0.05, y, "• " + text, color="#c7d0d9", fontsize=8, va="top")
        y -= 0.065
    panel.text(0.05, 0.28, "PATTERN RULE", color="#e5e7eb", fontsize=8, fontweight="bold")
    rule = {
        "Ascending Triangle":"flat resistance + rising support",
        "Descending Triangle":"flat support + falling resistance",
        "Symmetrical Triangle":"falling resistance + rising support",
        "Rising Wedge":"two rising converging rails",
        "Falling Wedge":"two falling converging rails",
        "Ascending Channel":"two rising parallel rails",
        "Descending Channel":"two falling parallel rails",
        "Horizontal Channel":"two flat parallel rails",
        "Bull Flag":"bull impulse + falling correction channel",
        "Bear Flag":"bear impulse + rising correction channel",
        "Pennant":"impulse + converging correction",
        "Uptrend Line":"rising support rail",
        "Downtrend Line":"falling resistance rail",
        "Horizontal Support & Resistance":"flat price boundary",
    }.get(name, "geometric match")
    panel.text(0.05, 0.23, rule, color="#c7d0d9", fontsize=8, va="top", wrap=True)
    fig.tight_layout()
    buf = __import__("io").BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf
