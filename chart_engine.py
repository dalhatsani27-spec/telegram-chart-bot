"""
chart_engine.py
===============
New institutional chart visual engine.

Replaces the old generate_smc_chart / generate_execution_chart style.

Matches the educational mapping style from:
  - SMC (Smart Money Concepts) maps
  - AMD (Algorithmic Market Dynamics) 5-phase cycle
  - Trendline strategy maps
  - ICT Silver Bullet (time windows)

Design rules:
  - Dark TradingView-like theme
  - Clear colored zone rectangles with labels
  - Vertical session separators (Asian / London / New York)
  - AMD phase background shading
  - Structure labels (BOS / CHoCH / HH / HL / LH / LL)
  - Liquidity pools marked
  - Clean legend
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import mplfinance as mpf


# ---------------------------------------------------------------------------
# Colour palette (aligned with the educational images)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0d1117",
    "grid": "#1f2937",
    "text": "#e5e7eb",
    "bull": "#089981",
    "bear": "#f23645",
    "fvg_bull": "#00c853",
    "fvg_bear": "#ff1744",
    "ob_bull": "#1e88e5",
    "ob_bear": "#6d4c41",
    "breaker": "#7e57c2",
    "idm": "#ff9100",
    "liquidity": "#ea80fc",
    "poc": "#ff9800",
    "session_asian": "#263238",
    "session_london": "#1a237e",
    "session_ny": "#004d40",
    "phase_accum": "#37474f",
    "phase_manip": "#b71c1c",
    "phase_disp": "#0d47a1",
    "phase_rev": "#4a148c",
    "phase_cont": "#1b5e20",
    "trendline": "#00e676",
    "entry": "#00e676",
    "sl": "#f23645",
    "tp": "#26c6da",
}


# Session windows in UTC (start hour inclusive, end hour exclusive)
# These match common TradingView / ICT session definitions
SESSION_UTC = {
    "Asian": (0, 8),
    "London": (7, 16),
    "NewYork": (12, 21),
}


def _to_naive_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ensure datetime index is timezone-naive UTC for plotting."""
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


def _session_name(hour: int) -> str:
    names = []
    for name, (start, end) in SESSION_UTC.items():
        if start <= hour < end:
            names.append(name)
    return "+".join(names) if names else "Off"


def _draw_zone(
    ax,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    color: str,
    alpha: float = 0.22,
    label: str | None = None,
    label_x: float | None = None,
    fontsize: int = 7,
):
    """Draw a rectangular zone with optional label."""
    lo, hi = min(y0, y1), max(y0, y1)
    rect = mpatches.FancyBboxPatch(
        (x0, lo),
        max(x1 - x0, 0.5),
        hi - lo,
        boxstyle="square,pad=0",
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(rect)
    if label:
        lx = label_x if label_x is not None else x0 + 0.5
        ax.text(
            lx,
            (lo + hi) / 2,
            label,
            fontsize=fontsize,
            color="#ffffff",
            va="center",
            ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=color, alpha=0.92, edgecolor="none"),
            zorder=8,
        )


def _draw_session_separators(ax, chart_df: pd.DataFrame, chart_len: int, minimal: bool = True):
    """
    Session separators. minimal=True (default): thin low-alpha lines, no labels
    so they don't fight the analysis. minimal=False: TradingView-style with labels.
    """
    if chart_df is None or chart_df.empty:
        return

    idx = _to_naive_utc(chart_df.index)
    # Only London + New York opens — skip Asian to cut clutter
    session_starts = {"London": 7, "NewYork": 12} if minimal else {"Asian": 0, "London": 7, "NewYork": 12}
    colors = {"Asian": "#546e7a", "London": "#5c6bc0", "NewYork": "#26a69a"}
    labels_drawn = set()
    lw = 0.6 if minimal else 1.1
    alpha = 0.35 if minimal else 0.85

    for i, ts in enumerate(idx):
        h, m = ts.hour, ts.minute
        for name, start_h in session_starts.items():
            if h == start_h and m == 0:
                ax.axvline(i, color=colors[name], linestyle=":", linewidth=lw, alpha=alpha, zorder=1)
                if not minimal and name not in labels_drawn:
                    ax.text(
                        i + 0.3,
                        ax.get_ylim()[1] * 0.995 if ax.get_ylim()[1] else chart_df["High"].max(),
                        name[:3].upper(),
                        fontsize=6.5,
                        color=colors[name],
                        rotation=90,
                        va="top",
                        ha="left",
                        alpha=0.9,
                        zorder=9,
                    )
                    labels_drawn.add(name)


def _draw_phase_backgrounds(
    ax,
    chart_len: int,
    phase_segments: List[Dict[str, Any]],
):
    """
    Shade background by AMD phase.
    phase_segments: list of {start_idx, end_idx, phase}
    """
    phase_colors = {
        "ACCUMULATION": COLORS["phase_accum"],
        "MANIPULATION": COLORS["phase_manip"],
        "DISPLACEMENT": COLORS["phase_disp"],
        "REVERSION": COLORS["phase_rev"],
        "CONTINUATION": COLORS["phase_cont"],
        "RANGE": COLORS["phase_accum"],
        "UNKNOWN": "#1a1a1a",
    }
    for seg in phase_segments:
        start = max(0, int(seg.get("start_idx", 0)))
        end = min(chart_len - 1, int(seg.get("end_idx", chart_len - 1)))
        if end <= start:
            continue
        color = phase_colors.get(str(seg.get("phase", "UNKNOWN")).upper(), "#1a1a1a")
        ax.axvspan(start, end, facecolor=color, alpha=0.18, zorder=0)


def _prepare_ohlc(df: pd.DataFrame, max_bars: int = 120) -> Tuple[pd.DataFrame, int]:
    if df is None or df.empty:
        raise ValueError("no chart data")
    chart_len = min(max_bars, len(df))
    chart_df = df.tail(chart_len).copy()
    chart_df.index = _to_naive_utc(pd.DatetimeIndex(chart_df.index))
    return chart_df, chart_len


def _draw_swings(ax, swings, offset: int, chart_len: int):
    """
    Map every pivot the way a discretionary trader marks the chart:
      - Red down-arrow on swing highs (HH / LH)
      - Blue up-arrow on swing lows (HL / LL)
      - HH/HL/LH/LL label + thin zigzag skeleton
    This is the structure anchors used to build channels.
    """
    if not swings:
        return
    vis = []
    for s in swings:
        idx = int(s.get("index", -1)) - offset
        if 0 <= idx < chart_len:
            vis.append({**s, "vidx": idx})
    if len(vis) < 2:
        return
    last_high = None
    last_low = None
    for s in vis:
        price = float(s["price"])
        idx = s["vidx"]
        if s.get("type") == "high":
            label = "HH" if last_high is not None and price > last_high else ("LH" if last_high is not None else "H")
            last_high = price
            color = "#ef5350" if label in ("HH", "H") else "#ffab40"
            # Down arrow at swing high (matches MT5-style pivot markers)
            ax.annotate(
                "", xy=(idx, price), xytext=(idx, price),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6),
                zorder=10,
            )
            ax.scatter([idx], [price], marker="v", s=70, c=color,
                       edgecolors="#ffffff", linewidths=0.8, zorder=11)
            ax.annotate(label, (idx, price), textcoords="offset points", xytext=(0, 10),
                        fontsize=7, color=color, ha="center", fontweight="bold", zorder=12)
        else:
            label = "LL" if last_low is not None and price < last_low else ("HL" if last_low is not None else "L")
            last_low = price
            color = "#29b6f6" if label in ("LL", "L") else "#26a69a"
            # Up arrow at swing low
            ax.scatter([idx], [price], marker="^", s=70, c=color,
                       edgecolors="#ffffff", linewidths=0.8, zorder=11)
            ax.annotate(label, (idx, price), textcoords="offset points", xytext=(0, -12),
                        fontsize=7, color=color, ha="center", fontweight="bold", zorder=12)
    if len(vis) >= 2:
        xs = [s["vidx"] for s in vis]
        ys = [float(s["price"]) for s in vis]
        ax.plot(xs, ys, color="#90a4ae", linewidth=0.9, alpha=0.50, zorder=3, linestyle="-")


def _draw_projections(ax, projections, chart_len: int):
    for p in (projections or [])[:4]:
        price = p.get("price")
        if price is None:
            continue
        ax.axhline(price, color="#7e57c2", linestyle="-.", linewidth=1.1, alpha=0.8)
        ax.text(chart_len * 0.01, price, p.get("label", "P"), fontsize=6.5, color="#ce93d8", va="bottom")


def _draw_volume_profile_levels(ax, vp, chart_len: int):
    """Draw only POC line. Value Area big orange box removed for clean charts."""
    if not vp:
        return
    if vp.get("poc_price") is not None:
        ax.axhline(vp["poc_price"], color=COLORS["poc"], linestyle=":", linewidth=1.35, alpha=0.9)
        ax.text(chart_len * 0.86, vp["poc_price"], "POC", fontsize=7, color=COLORS["poc"], va="bottom")
    # LOCKED: do NOT draw the big orange Value Area rectangle (too noisy)


def _draw_position_container(ax, pos, chart_len: int):
    """
    Big, unmissable long/short position container (matches the shaded
    entry->target boxes in TradingView-style SMC education charts):
      - solid shaded risk zone (entry->SL, red) and reward zone
        (entry->TP1/TP2, green)
      - thick bordered box around the whole container
      - large bold LONG / SHORT banner with R:R, colored green/red
      - thick entry/SL/TP lines with labels
    Used identically by every chart type (SMC map, AMD map, trendline map,
    ticket chart) so the position reads the same everywhere.
    """
    if not pos:
        return
    entry = pos.get("entry")
    sl = pos.get("sl")
    tp1 = pos.get("tp1")
    tp2 = pos.get("tp2")
    if entry is None:
        return

    side = str(pos.get("side") or pos.get("direction") or "").upper()
    is_long = ("LONG" in side) or (side == "BUY") or ("BULL" in side)
    box_color = "#00e676" if is_long else "#ff1744"
    label = "LONG" if is_long else "SHORT"

    x0_frac = 0.66
    x0 = chart_len * x0_frac
    x1 = chart_len - 1

    span_vals = [entry]
    if sl is not None:
        ax.axhspan(min(entry, sl), max(entry, sl), xmin=x0_frac, xmax=1.0,
                   facecolor="#ff1744", alpha=0.28, zorder=2)
        ax.axhline(sl, color="#ff1744", linestyle="-", linewidth=2.0, xmin=x0_frac, zorder=4)
        ax.text(x0 + 1, sl, "SL", fontsize=8, color="#ff1744", fontweight="bold", va="top", zorder=12)
        span_vals.append(sl)
    for tp, tp_label, tp_alpha in ((tp1, "TP1", 1.0), (tp2, "TP2", 0.7)):
        if tp is None:
            continue
        ax.axhspan(min(entry, tp), max(entry, tp), xmin=x0_frac, xmax=1.0,
                   facecolor="#00e676", alpha=0.28 * tp_alpha, zorder=2)
        ax.axhline(tp, color="#00e676", linestyle=":", linewidth=1.8, xmin=x0_frac, alpha=max(tp_alpha, 0.6), zorder=4)
        ax.text(x0 + 1, tp, tp_label, fontsize=8, color="#00e676", fontweight="bold", va="bottom", zorder=12)
        span_vals.append(tp)

    ax.axhline(entry, color=COLORS["entry"], linestyle="--", linewidth=2.2, xmin=x0_frac, zorder=5)
    ax.text(x0 + 1, entry, "ENTRY", fontsize=8, color=COLORS["entry"], fontweight="bold", va="bottom", zorder=12)

    span_lo, span_hi = min(span_vals), max(span_vals)
    if span_hi > span_lo:
        rect = mpatches.FancyBboxPatch(
            (x0, span_lo), (x1 - x0), (span_hi - span_lo),
            boxstyle="square,pad=0", facecolor="none", edgecolor=box_color,
            linewidth=2.4, alpha=0.9, zorder=3,
        )
        ax.add_patch(rect)

    risk = abs(entry - sl) if sl is not None else None
    reward = abs((tp1 if tp1 is not None else tp2) - entry) if (tp1 is not None or tp2 is not None) else None
    rr_txt = f"  R:R 1:{(reward / risk):.1f}" if risk and reward else ""

    header_y = span_hi + (span_hi - span_lo) * 0.14 if span_hi > span_lo else entry * 1.001
    ax.text(
        (x0 + x1) / 2, header_y, f"{label}{rr_txt}",
        fontsize=14, color="#000000" if is_long else "#ffffff",
        fontweight="bold", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=box_color, alpha=0.95,
                  edgecolor="#000000", linewidth=1.3),
        zorder=13,
    )


def _draw_vp_histogram(ax, vp, chart_len: int, price_min: float, price_max: float):
    """Volume profile histogram on the right edge (like TradingView fixed-range VP)."""
    if not vp or vp.get("bin_volumes") is None:
        return
    try:
        edges = vp.get("bin_edges")
        vols = vp.get("bin_volumes")
        if edges is None or vols is None:
            return
        import numpy as np
        edges = np.asarray(edges, dtype=float)
        vols = np.asarray(vols, dtype=float)
        if len(vols) == 0 or vols.max() <= 0:
            return
        max_w = chart_len * 0.14  # max bar width in x-units
        x_base = chart_len - 0.5
        for i, v in enumerate(vols):
            if i + 1 >= len(edges):
                break
            y0, y1 = float(edges[i]), float(edges[i + 1])
            w = (v / vols.max()) * max_w
            mid = (y0 + y1) / 2.0
            is_poc = abs(mid - float(vp.get("poc_price", 0))) < (y1 - y0)
            in_va = False
            if vp.get("value_area_low") is not None and vp.get("value_area_high") is not None:
                in_va = vp["value_area_low"] <= mid <= vp["value_area_high"]
            color = "#ff9800" if is_poc else ("#5c6bc0" if in_va else "#455a64")
            alpha = 0.85 if is_poc else (0.55 if in_va else 0.35)
            ax.barh(mid, w, height=(y1 - y0) * 0.9, left=x_base - w,
                    color=color, alpha=alpha, zorder=2, align="center")
    except Exception:
        pass


def _draw_tv_position_box(ax, pos, chart_len: int):
    """Alias kept for existing call sites -- delegates to the single shared
    _draw_position_container so every chart type renders an identical,
    equally bold container instead of maintaining two versions."""
    _draw_position_container(ax, pos, chart_len)


def generate_smc_map(
    df: pd.DataFrame,
    symbol: str,
    title_suffix: str = "",
    zones: Optional[Dict[str, Any]] = None,
    bias_label: str = "",
    show_sessions: bool = True,
) -> io.BytesIO:
    """
    SMC visual map matching the educational style:
      - FVG / Order Blocks / Breaker / IDM / Liquidity
      - Structure levels
      - Session separators
    """
    zones = zones or {}
    chart_df, chart_len = _prepare_ohlc(df, max_bars=160)
    offset = len(df) - chart_len

    mc = mpf.make_marketcolors(up=COLORS["bull"], down=COLORS["bear"], edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":",
        gridcolor=COLORS["grid"],
        y_on_right=True,
        facecolor=COLORS["bg"],
        figcolor=COLORS["bg"],
        rc={"axes.labelcolor": COLORS["text"], "xtick.color": COLORS["text"], "ytick.color": COLORS["text"]},
    )

    addplots = []
    for col, color, width in [
        ("EMA200", "#ffd600", 1.3),
        ("EMA50", "#2962ff", 1.1),
        ("EMA20", "#ab47bc", 0.9),
    ]:
        if col in chart_df.columns:
            addplots.append(mpf.make_addplot(chart_df[col], color=color, width=width))

    price_min = float(chart_df["Low"].min())
    price_max = float(chart_df["High"].max())
    padding = (price_max - price_min) * 0.10 or 0.0005

    fig, axlist = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        volume=False,
        addplot=addplots if addplots else None,
        returnfig=True,
        figsize=(13, 7.2),
        ylim=(price_min - padding, price_max + padding),
        datetime_format="%d %H:%M",
    )
    ax = axlist[0]

    # --- Zones (LOCKED clean style: only the most important sequential levels) ---
    fvgs = zones.get("fvgs") or []
    obs = zones.get("order_blocks") or []
    idms = zones.get("inducements") or []
    structure = zones.get("structure") or {}
    vp = zones.get("volume_profile")
    bos_events = zones.get("bos_events") or []

    # Prefer unmitigated structure-confirmed OBs only
    clean_obs = [z for z in obs if z.get("bos") or str(z.get("type", "")).upper() == "BREAKER"][:3]
    if not clean_obs:
        clean_obs = obs[:2]

    # FVG — max 3, prefer unmitigated
    for z in fvgs[:3]:
        mitigated = z.get("mitigated", False)
        bias = str(z.get("bias", "")).upper()
        color = COLORS["fvg_bull"] if bias in ("BULLISH", "BUY") else COLORS["fvg_bear"]
        if mitigated:
            color = "#607d8b"
        label = "FVG" if not mitigated else "IFVG"
        z_idx = int(z.get("index", chart_len - 10)) - offset
        x0 = max(0, z_idx - 2)
        x1 = min(chart_len - 1, z_idx + 8)
        _draw_zone(ax, x0, x1, z["bottom"], z["top"], color, 0.20 if not mitigated else 0.08, label)

    # Order Blocks / Breakers — max 3, structure-confirmed only
    for z in clean_obs:
        mitigated = z.get("mitigated", False)
        bias = str(z.get("bias", "")).upper()
        is_breaker = mitigated or str(z.get("type", "")).upper() == "BREAKER"
        if is_breaker:
            color = COLORS["breaker"]
            label = "BRK"
        else:
            color = COLORS["ob_bull"] if bias in ("BULLISH", "BUY") else COLORS["ob_bear"]
            label = "OB"
        z_idx = int(z.get("index", chart_len - 15)) - offset
        x0 = max(0, z_idx - 1)
        x1 = min(chart_len - 1, z_idx + 12)
        _draw_zone(ax, x0, x1, z["bottom"], z["top"], color, 0.24 if not mitigated else 0.10, label)

    # Inducement — max 2 (only important liquidity traps)
    for z in idms[:2]:
        mitigated = z.get("mitigated") or z.get("swept")
        color = "#9e9e9e" if mitigated else COLORS["idm"]
        label = "IDM✗" if mitigated else "IDM"
        z_idx = int(z.get("index", chart_len - 20)) - offset
        x0 = max(0, z_idx - 1)
        x1 = min(chart_len - 1, z_idx + 6)
        _draw_zone(ax, x0, x1, z["bottom"], z["top"], color, 0.22 if not mitigated else 0.08, label)

    # Structure High / Low
    if structure.get("structure_high"):
        ax.axhline(structure["structure_high"], color=COLORS["liquidity"], linestyle="--", linewidth=1.15, alpha=0.85)
        ax.text(chart_len * 0.02, structure["structure_high"], "RANGE HIGH / BSL", fontsize=7,
                color=COLORS["liquidity"], va="bottom", fontweight="bold")
    if structure.get("structure_low"):
        ax.axhline(structure["structure_low"], color=COLORS["liquidity"], linestyle="--", linewidth=1.15, alpha=0.85)
        ax.text(chart_len * 0.02, structure["structure_low"], "RANGE LOW / SSL", fontsize=7,
                color=COLORS["liquidity"], va="top", fontweight="bold")

    # BOS / CHoCH / MSS — dotted line + clear educational labels (max 5)
    # Prefer the most recent structure events for clean charts
    for ev in bos_events[-5:]:
        idx = int(ev.get("index", 0)) - offset
        price = ev.get("price")
        if price is None:
            continue
        kind = str(ev.get("type", "BOS")).upper()
        bias = str(ev.get("bias", "")).upper()
        color = "#00e676" if bias == "BULLISH" else "#ff5252"
        x0 = max(0, idx - 2) if idx > 0 else 0
        ax.plot(
            [x0, chart_len - 1],
            [price, price],
            color=color,
            linestyle=":",
            linewidth=1.45,
            alpha=0.9,
            zorder=6,
        )
        label_x = min(max(idx, 0), chart_len - 1)
        # Educational labels: CHoCH = REVERSAL signal, BOS after CHoCH = CONTINUATION
        display = kind
        if kind == "CHOCH":
            display = "CHoCH"
        elif kind == "MSS":
            display = "MSS"
        ax.annotate(
            display,
            xy=(label_x, price),
            xytext=(4, 10 if bias == "BULLISH" else -12),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            fontweight="bold",
            ha="left",
            zorder=10,
        )

    # Swings (HH/HL/LH/LL) for verification
    swings = zones.get("swings") or (structure.get("swings") if structure else None)
    if not swings:
        try:
            from market_structure import zigzag_swings
            swings = zigzag_swings(df, depth=4, deviation_atr=0.28)
        except Exception:
            swings = []
    _draw_swings(ax, swings, offset, chart_len)

    # Volume profile POC + VA
    _draw_volume_profile_levels(ax, vp, chart_len)

    # Projections + position container (shared across strategies)
    _draw_projections(ax, zones.get("projections"), chart_len)
    _draw_position_container(ax, zones.get("position"), chart_len)

    if show_sessions:
        _draw_session_separators(ax, chart_df, chart_len)

    legend = "Swings HH/HL  |  OB→FVG  |  BOS dotted  |  POC/VA  |  Proj  |  Sessions"
    ax.set_title(
        f"{symbol}  SMC MAP  |  {title_suffix}  |  Bias: {bias_label or '—'}\n{legend}",
        color=COLORS["text"],
        fontsize=9.5,
        fontweight="bold",
        pad=10,
    )

    img_buf = io.BytesIO()
    fig.savefig(img_buf, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf


def generate_amd_map(
    df: pd.DataFrame,
    symbol: str,
    analysis: Dict[str, Any],
    title_suffix: str = "",
) -> io.BytesIO:
    """
    AMD 5-phase visual map matching the educational images:
      1. Accumulation (range) – grey/teal background
      2. Manipulation (liquidity grab) – red background
      3. Displacement (expansion) – blue background
      4. Reversion (pullback) – purple background
      5. Continuation (trend) – green background

    + Session separators (Asian / London / New York)
    + Range High / Range Low
    + FVG / OB zones
    + Clear phase labels
    """
    chart_df, chart_len = _prepare_ohlc(df, max_bars=160)
    offset = len(df) - chart_len

    mc = mpf.make_marketcolors(up=COLORS["bull"], down=COLORS["bear"], edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":",
        gridcolor=COLORS["grid"],
        y_on_right=True,
        facecolor=COLORS["bg"],
        figcolor=COLORS["bg"],
        rc={"axes.labelcolor": COLORS["text"], "xtick.color": COLORS["text"], "ytick.color": COLORS["text"]},
    )

    price_min = float(chart_df["Low"].min())
    price_max = float(chart_df["High"].max())
    for p in (analysis.get("projections") or []):
        if p.get("price") is not None:
            price_min = min(price_min, float(p["price"]))
            price_max = max(price_max, float(p["price"]))
    pos = analysis.get("position")
    if pos:
        for k in ("entry", "sl", "tp1", "tp2"):
            if pos.get(k) is not None:
                price_min = min(price_min, float(pos[k]))
                price_max = max(price_max, float(pos[k]))
    padding = (price_max - price_min) * 0.12 or 0.0005

    fig, axlist = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        volume=False,
        returnfig=True,
        figsize=(13.5, 7.5),
        ylim=(price_min - padding, price_max + padding),
        datetime_format="%d %H:%M",
    )
    ax = axlist[0]

    # --- Phase background from precise bar-by-bar segments ---
    phase = str(analysis.get("phase", "UNKNOWN")).upper()
    phase_map = {
        "ACCUMULATION": "ACCUMULATION",
        "RANGE": "ACCUMULATION",
        "MANIPULATION": "MANIPULATION",
        "POST_MANIPULATION": "MANIPULATION",
        "DISTRIBUTION_UP": "DISPLACEMENT",
        "DISTRIBUTION_DOWN": "DISPLACEMENT",
        "DISPLACEMENT": "DISPLACEMENT",
        "REVERSION": "REVERSION",
        "CONTINUATION": "CONTINUATION",
    }
    mapped_phase = phase_map.get(phase, "UNKNOWN")

    raw_segments = analysis.get("phase_segments") or []
    phase_segments = []
    if raw_segments:
        for seg in raw_segments:
            # analysis indices are absolute on full df; chart is tail(chart_len)
            abs_start = int(seg.get("start_idx", 0))
            abs_end = int(seg.get("end_idx", 0))
            vis_start = abs_start - offset
            vis_end = abs_end - offset
            if vis_end < 0 or vis_start >= chart_len:
                continue
            vis_start = max(0, vis_start)
            vis_end = min(chart_len - 1, vis_end)
            if vis_end >= vis_start:
                phase_segments.append({
                    "start_idx": vis_start,
                    "end_idx": vis_end,
                    "phase": phase_map.get(str(seg.get("phase", "")).upper(), seg.get("phase")),
                })
    if not phase_segments:
        phase_segments = [{"start_idx": 0, "end_idx": chart_len - 1, "phase": mapped_phase}]

    _draw_phase_backgrounds(ax, chart_len, phase_segments)

    # Session separators (TradingView style)
    _draw_session_separators(ax, chart_df, chart_len)

    # Range High / Low (Liquidity pools)
    rng = analysis.get("range")
    if rng:
        ax.axhline(rng["high"], color=COLORS["liquidity"], linestyle="--", linewidth=1.3, alpha=0.9)
        ax.axhline(rng["low"], color=COLORS["liquidity"], linestyle="--", linewidth=1.3, alpha=0.9)
        ax.text(1, rng["high"], "RANGE HIGH  (BSL)", fontsize=7.5, color=COLORS["liquidity"],
                va="bottom", fontweight="bold")
        ax.text(1, rng["low"], "RANGE LOW  (SSL)", fontsize=7.5, color=COLORS["liquidity"],
                va="top", fontweight="bold")

    # FVG / OB from analysis
    for z in (analysis.get("fvgs") or [])[:5]:
        mitigated = z.get("mitigated", False)
        bias = str(z.get("bias", "")).upper()
        color = COLORS["fvg_bull"] if bias in ("BULLISH", "BUY") else COLORS["fvg_bear"]
        if mitigated:
            color = "#607d8b"
        z_idx = int(z.get("index", chart_len - 10)) - offset
        x0 = max(0, z_idx - 1)
        x1 = min(chart_len - 1, z_idx + 7)
        _draw_zone(ax, x0, x1, z["bottom"], z["top"], color, 0.22 if not mitigated else 0.08,
                   "FVG" if not mitigated else "IFVG")

    for z in (analysis.get("order_blocks") or [])[:4]:
        mitigated = z.get("mitigated", False)
        bias = str(z.get("bias", "")).upper()
        color = COLORS["ob_bull"] if bias in ("BULLISH", "BUY") else COLORS["ob_bear"]
        if mitigated:
            color = COLORS["breaker"]
        z_idx = int(z.get("index", chart_len - 15)) - offset
        x0 = max(0, z_idx - 1)
        x1 = min(chart_len - 1, z_idx + 10)
        _draw_zone(ax, x0, x1, z["bottom"], z["top"], color, 0.25 if not mitigated else 0.10,
                   "OB" if not mitigated else "BRK")

    # Swings for verification
    swings = analysis.get("swings")
    if not swings:
        try:
            from market_structure import zigzag_swings
            swings = zigzag_swings(df, depth=4, deviation_atr=0.28)
        except Exception:
            swings = []
    _draw_swings(ax, swings, offset, chart_len)

    # BOS events on AMD chart too
    for ev in (analysis.get("bos_events") or [])[:6]:
        idx = int(ev.get("index", 0)) - offset
        price = ev.get("price")
        if price is None:
            continue
        kind = str(ev.get("type", "BOS")).upper()
        b = str(ev.get("bias", "")).upper()
        color = "#00e676" if b == "BULLISH" else "#ff5252"
        x0 = max(0, idx - 2) if idx > 0 else 0
        ax.plot([x0, chart_len - 1], [price, price], color=color, linestyle=":", linewidth=1.3, alpha=0.85, zorder=6)
        ax.annotate(kind, xy=(min(max(idx, 0), chart_len - 1), price),
                    xytext=(4, 9 if b == "BULLISH" else -11), textcoords="offset points",
                    fontsize=6.5, color=color, fontweight="bold", zorder=10)

    _draw_volume_profile_levels(ax, analysis.get("volume_profile"), chart_len)
    _draw_projections(ax, analysis.get("projections"), chart_len)
    _draw_position_container(ax, analysis.get("position"), chart_len)

    # Phase label box
    phase_note = analysis.get("phase_note") or mapped_phase
    bias = analysis.get("amd_bias", "NEUTRAL")
    session = analysis.get("last_session", "")
    ax.text(
        0.98,
        0.03,
        f"PHASE: {mapped_phase}   |   Bias: {bias}   |   Session: {session}",
        transform=ax.transAxes,
        fontsize=8,
        color="#ffffff",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827", alpha=0.88, edgecolor="#374151"),
        zorder=12,
    )

    # Legend
    legend_txt = (
        "Phases · Swings HH/HL · BOS dotted · POC/VA · Proj · Sessions"
    )
    ax.set_title(
        f"{symbol}  AMD CYCLE MAP  |  {title_suffix}\n{legend_txt}",
        color=COLORS["text"],
        fontsize=9.5,
        fontweight="bold",
        pad=10,
    )

    img_buf = io.BytesIO()
    fig.savefig(img_buf, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf


def generate_trendline_map(
    df: pd.DataFrame,
    symbol: str,
    setup: Dict[str, Any],
    title_suffix: str = "",
) -> io.BytesIO:
    """
    Full trendline family map:
      - Multiple uptrend / downtrend lines
      - Parallel channel
      - Measured-move projections (P1/P2/P3)
      - Volume Profile POC + Value Area
      - Long/Short position container (Entry/SL/TP)
    """
    # Prefer family payload if present
    family = setup.get("family") or setup.get("analysis") or setup
    if family.get("df") is not None:
        df = family["df"]
    chart_df, chart_len = _prepare_ohlc(df, max_bars=160)
    offset = len(df) - chart_len

    mc = mpf.make_marketcolors(up=COLORS["bull"], down=COLORS["bear"], edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":",
        gridcolor=COLORS["grid"],
        y_on_right=True,
        facecolor=COLORS["bg"],
        figcolor=COLORS["bg"],
    )

    addplots = []
    for key, color, ls, w in [
        ("upper_line", COLORS["trendline"], "-", 1.7),
        ("middle_line", "#ff9800", "--", 1.15),
        ("lower_line", COLORS["trendline"], "-", 1.7),
    ]:
        arr = family.get(key)
        if arr is not None and len(arr) >= chart_len:
            series = pd.Series(arr[-chart_len:], index=chart_df.index)
            addplots.append(mpf.make_addplot(series, color=color, width=w, linestyle=ls))

    # Expand ylim for projections / position
    price_min = float(chart_df["Low"].min())
    price_max = float(chart_df["High"].max())
    for p in (family.get("projections") or []):
        price_min = min(price_min, float(p["price"]))
        price_max = max(price_max, float(p["price"]))
    for key in ("entry", "sl", "tp1", "tp2"):
        if setup.get(key) is not None:
            price_min = min(price_min, float(setup[key]))
            price_max = max(price_max, float(setup[key]))
    pos = setup.get("position") or family.get("position")
    if pos:
        for key in ("entry", "sl", "tp1", "tp2"):
            if pos.get(key) is not None:
                price_min = min(price_min, float(pos[key]))
                price_max = max(price_max, float(pos[key]))
    padding = (price_max - price_min) * 0.10 or 0.0005

    fig, axlist = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        volume=False,
        addplot=addplots if addplots else None,
        returnfig=True,
        figsize=(13.2, 7.4),
        ylim=(price_min - padding, price_max + padding),
    )
    ax = axlist[0]

    # --- ALWAYS map pivot points (structure anchors) ---
    pivots = family.get("pivots") or []
    if not pivots:
        try:
            from market_structure import zigzag_swings
            pivots = zigzag_swings(df, depth=4, deviation_atr=0.28)
        except Exception:
            pivots = []
    # Full HH/HL/LH/LL labeling + zigzag skeleton
    _draw_swings(ax, pivots, offset, chart_len)
    # Emphasize every visible pivot with a clear marker
    for s in pivots:
        idx = int(s.get("index", -1)) - offset
        if not (0 <= idx < chart_len):
            continue
        price = float(s["price"])
        is_high = s.get("type") == "high"
        color = "#ffeb3b" if is_high else "#00e5ff"
        ax.scatter([idx], [price], s=42, c=color, edgecolors="#000000", linewidths=0.6, zorder=10)

    # Clean parallel family only (MT5-style rails — anchored on pivots)
    def _line_at(tl, x):
        x0, y0, x1, y1 = tl["x0"], tl["y0"], tl["x1"], tl["y1"]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    family_kind = family.get("family_kind", "")
    rail_color = "#26a69a" if family_kind == "ascending" else "#ef5350"
    rails = family.get("family_lines") or []
    if not rails:
        if family.get("channel"):
            rails = [family["channel"].get("lower"), family["channel"].get("upper")]
            rails = [r for r in rails if r]
    for i, tl in enumerate(rails[:5]):
        if not tl:
            continue
        xs = [max(0, int(tl["x0"]) - offset), chart_len - 1]
        ys = [_line_at(tl, max(int(tl["x0"]), 0)), float(tl.get("y_end", tl.get("y1", 0)))]
        if xs[0] < chart_len:
            lw = 1.8 if i in (0, len(rails) - 1) else 1.2
            ax.plot(xs, ys, color=rail_color, linewidth=lw, alpha=0.92, zorder=4)
        # Mark the two pivot anchors of the primary rail
        if i == 0:
            for ax_key, ay_key in (("x0", "y0"), ("x1", "y1")):
                if ax_key in tl and ay_key in tl:
                    px = int(tl[ax_key]) - offset
                    if 0 <= px < chart_len:
                        ax.scatter([px], [float(tl[ay_key])], s=55, c=rail_color,
                                   edgecolors="#ffffff", linewidths=1.0, zorder=11, marker="D")

    # M/W neckline (double top / double bottom)
    mw = family.get("mw_pattern")
    if mw and mw.get("neckline") is not None:
        ax.axhline(mw["neckline"], color="#ff9800", linestyle="-", linewidth=1.6, alpha=0.95, zorder=5)
        ax.text(chart_len * 0.02, mw["neckline"], f"NECKLINE ({mw.get('pattern', '')})",
                fontsize=7.5, color="#ffb74d", fontweight="bold", va="bottom")

    # Discrete pattern-scanner output (Double Top/Bottom, H&S, Triangle,
    # Wedge, Flag, Rectangle -- from patterns.py). This payload shape is
    # different from the trendline-family payload above (single trigger
    # line + labelled key points instead of a rail family), and previously
    # had NO render path here at all -- the Pattern Scanner chart would come
    # back with candles only and none of the actual detected structure.
    trigger_line = family.get("trigger_line") or setup.get("trigger_line")
    if trigger_line and len(trigger_line) >= 2:
        xs = [max(0, int(p[0]) - offset) for p in trigger_line]
        ys = [float(p[1]) for p in trigger_line]
        if xs[-1] < chart_len:
            # Extend the trigger/neckline to the right edge of the chart so
            # it's visible as the level to watch, not just where it was formed.
            xs = xs + [chart_len - 1]
            ys = ys + [ys[-1]]
        ax.plot(xs, ys, color="#ff9800", linewidth=1.7, linestyle="-", alpha=0.95, zorder=5)
        label = family.get("category", "pattern")
        ax.text(chart_len * 0.02, ys[0], f"TRIGGER ({label})", fontsize=7.5,
                color="#ffb74d", fontweight="bold", va="bottom")

    key_points = family.get("key_points") or setup.get("key_points")
    if key_points:
        for kp in key_points:
            if len(kp) < 2:
                continue
            px = int(kp[0]) - offset
            if not (0 <= px < chart_len):
                continue
            py = float(kp[1])
            label = kp[2] if len(kp) > 2 else ""
            ax.scatter([px], [py], s=60, c="#ffca28", edgecolors="#000000", linewidths=0.7, zorder=11, marker="o")
            if label:
                ax.annotate(label, (px, py), fontsize=6.5, color="#ffe0b2",
                            xytext=(4, 4), textcoords="offset points")

    # Projections — subtle
    for p in (family.get("projections") or [])[:3]:
        ax.axhline(p["price"], color="#7e57c2", linestyle="-.", linewidth=0.9, alpha=0.65)
        ax.text(chart_len * 0.01, p["price"], p["label"], fontsize=6.5, color="#b39ddb", va="bottom", alpha=0.85)

    # Volume Profile histogram (right side) + POC/VA
    vp = family.get("volume_profile") or setup.get("volume_profile")
    _draw_vp_histogram(ax, vp, chart_len, price_min, price_max)
    if vp and vp.get("poc_price") is not None:
        ax.axhline(vp["poc_price"], color=COLORS["poc"], linestyle=":", linewidth=1.15, alpha=0.8)
        ax.text(chart_len * 0.55, vp["poc_price"], "POC", fontsize=6.5, color=COLORS["poc"], va="bottom")

    # TradingView-style position container (R:R) — projection levels
    pos = setup.get("position") or family.get("position")
    if not pos and setup.get("entry") is not None:
        pos = {
            "entry": setup.get("entry"), "sl": setup.get("sl"),
            "tp1": setup.get("tp1"), "tp2": setup.get("tp2"),
            "side": setup.get("direction", ""),
        }
    _draw_tv_position_box(ax, pos, chart_len)

    _draw_session_separators(ax, chart_df, chart_len)

    direction = family.get("direction") or setup.get("direction", "")
    strength = family.get("strength") or setup.get("confidence") or setup.get("score") or 0
    ax.set_title(
        f"{symbol}  TRENDLINE FAMILY  |  TF: 30m  |  {title_suffix}  |  {direction}  |  Str {strength:.0f}",
        color=COLORS["text"],
        fontsize=9.5,
        fontweight="bold",
        pad=10,
    )

    img_buf = io.BytesIO()
    fig.savefig(img_buf, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf


def generate_ticket_chart(
    df: pd.DataFrame,
    symbol: str,
    ticket: Dict[str, Any],
) -> io.BytesIO:
    """
    Clean execution ticket chart for Mobile Manual Trade.
    Shows entry zone, SL, TP1, TP2 clearly.
    """
    chart_df, chart_len = _prepare_ohlc(df, max_bars=80)

    mc = mpf.make_marketcolors(up=COLORS["bull"], down=COLORS["bear"], edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":",
        gridcolor=COLORS["grid"],
        y_on_right=True,
        facecolor=COLORS["bg"],
        figcolor=COLORS["bg"],
    )

    price_min = float(chart_df["Low"].min())
    price_max = float(chart_df["High"].max())
    # Expand limits so SL/TP are visible
    levels = [ticket.get("entry"), ticket.get("sl"), ticket.get("tp1"), ticket.get("tp2")]
    levels = [x for x in levels if x is not None]
    if levels:
        price_min = min(price_min, min(levels))
        price_max = max(price_max, max(levels))
    padding = (price_max - price_min) * 0.08 or 0.0003

    fig, axlist = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        volume=False,
        returnfig=True,
        figsize=(12, 6.5),
        ylim=(price_min - padding, price_max + padding),
    )
    ax = axlist[0]

    pos = {
        "entry": ticket.get("entry"),
        "sl": ticket.get("sl"),
        "tp1": ticket.get("tp1"),
        "tp2": ticket.get("tp2"),
        "side": ticket.get("direction", ""),
    }
    _draw_position_container(ax, pos, chart_len)

    direction = ticket.get("direction", "")
    strategy = ticket.get("strategy", "")
    ax.set_title(
        f"{symbol}  TRADE TICKET  |  {direction}  |  {strategy}",
        color=COLORS["text"],
        fontsize=10,
        fontweight="bold",
        pad=10,
    )

    img_buf = io.BytesIO()
    fig.savefig(img_buf, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf
