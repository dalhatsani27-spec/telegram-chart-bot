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


def _draw_session_separators(ax, chart_df: pd.DataFrame, chart_len: int):
    """
    Draw vertical session separators exactly like TradingView.
    Marks the open of Asian, London and New York sessions.
    """
    if chart_df is None or chart_df.empty:
        return

    idx = _to_naive_utc(chart_df.index)
    session_starts = {"Asian": 0, "London": 7, "NewYork": 12}
    colors = {
        "Asian": "#546e7a",
        "London": "#5c6bc0",
        "NewYork": "#26a69a",
    }
    labels_drawn = set()

    for i, ts in enumerate(idx):
        h = ts.hour
        m = ts.minute
        # Only mark the first bar of each session hour
        for name, start_h in session_starts.items():
            if h == start_h and m == 0:
                ax.axvline(i, color=colors[name], linestyle="--", linewidth=1.1, alpha=0.85, zorder=1)
                if name not in labels_drawn:
                    ax.text(
                        i + 0.3,
                        ax.get_ylim()[1] * 0.995 if ax.get_ylim()[1] else chart_df["High"].max(),
                        name[:3].upper(),
                        fontsize=6.5,
                        color=colors[name],
                        rotation=90,
                        va="top",
                        ha="left",
                        alpha=0.95,
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
    chart_df, chart_len = _prepare_ohlc(df, max_bars=110)
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

    # --- Zones ---
    fvgs = zones.get("fvgs") or []
    obs = zones.get("order_blocks") or []
    idms = zones.get("inducements") or []
    structure = zones.get("structure") or {}
    vp = zones.get("volume_profile")
    bos_events = zones.get("bos_events") or []  # list of {index, price, type}

    # FVG
    for z in fvgs[:7]:
        mitigated = z.get("mitigated", False)
        bias = str(z.get("bias", "")).upper()
        color = COLORS["fvg_bull"] if bias in ("BULLISH", "BUY") else COLORS["fvg_bear"]
        if mitigated:
            color = "#607d8b"
        label = "FVG" if not mitigated else "IFVG"
        # Map original index into visible window
        z_idx = int(z.get("index", chart_len - 10)) - offset
        x0 = max(0, z_idx - 2)
        x1 = min(chart_len - 1, z_idx + 8)
        _draw_zone(ax, x0, x1, z["bottom"], z["top"], color, 0.20 if not mitigated else 0.08, label)

    # Order Blocks / Breakers
    for z in obs[:6]:
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

    # Inducement / Liquidity
    for z in idms[:5]:
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

    # BOS / CHoCH markers
    for ev in bos_events[:8]:
        idx = int(ev.get("index", 0)) - offset
        if 0 <= idx < chart_len:
            price = ev.get("price")
            kind = str(ev.get("type", "BOS")).upper()
            color = "#00e676" if "BULL" in kind or kind == "BOS" else "#ff5252"
            ax.annotate(
                kind,
                xy=(idx, price),
                xytext=(0, 12 if "BULL" in kind or kind == "BOS" else -14),
                textcoords="offset points",
                fontsize=7,
                color=color,
                fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                zorder=10,
            )

    # POC
    if vp and vp.get("poc_price"):
        ax.axhline(vp["poc_price"], color=COLORS["poc"], linestyle=":", linewidth=1.2, alpha=0.85)
        ax.text(chart_len * 0.88, vp["poc_price"], "POC", fontsize=7, color=COLORS["poc"], va="bottom")

    if show_sessions:
        _draw_session_separators(ax, chart_df, chart_len)

    legend = "FVG  |  OB / BRK  |  IDM  |  BSL/SSL  |  Sessions (vertical)"
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
    chart_df, chart_len = _prepare_ohlc(df, max_bars=120)
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
        "ACCUM (grey) → MANIP (red) → DISPLACE (blue) → REVERT (purple) → CONT (green)   |   "
        "Vertical lines = Sessions"
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
    Trendline strategy visual map:
      - Valid trendline(s)
      - HH / HL / LH / LL labels
      - BOS markers
      - Entry / SL / TP boxes when available
    """
    chart_df, chart_len = _prepare_ohlc(df, max_bars=100)
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

    g = setup.get("geometry_data") or {}
    addplots = []
    if g.get("mode") == "channel":
        n = len(chart_df)
        for key, color, ls, w in [
            ("upper_line", COLORS["trendline"], "-", 1.8),
            ("middle_line", "#ff9800", "--", 1.2),
            ("lower_line", COLORS["trendline"], "-", 1.8),
        ]:
            if key in g and g[key] is not None:
                series = pd.Series(g[key][-n:], index=chart_df.index)
                addplots.append(mpf.make_addplot(series, color=color, width=w, linestyle=ls))

    price_min = float(chart_df["Low"].min())
    price_max = float(chart_df["High"].max())
    padding = (price_max - price_min) * 0.12 or 0.0005

    fig, axlist = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        volume=False,
        addplot=addplots if addplots else None,
        returnfig=True,
        figsize=(13, 7.2),
        ylim=(price_min - padding, price_max + padding),
    )
    ax = axlist[0]

    # Pattern trigger line
    if g.get("mode") == "pattern":
        trigger_line = g.get("trigger_line") or []
        if len(trigger_line) >= 2:
            xs = [pt[0] - offset for pt in trigger_line]
            ys = [pt[1] for pt in trigger_line]
            if xs[-1] != xs[0]:
                slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
                xs.append(chart_len - 1)
                ys.append(ys[-1] + slope * (chart_len - 1 - xs[-2]))
            xs = [max(0, min(chart_len - 1, x)) for x in xs]
            ax.plot(xs, ys, color="#ffeb3b", linewidth=2.0, linestyle="--", zorder=5)

        for px, py, label in (g.get("key_points") or []):
            cx = px - offset
            if 0 <= cx <= chart_len - 1:
                ax.scatter([cx], [py], color="#ffffff", edgecolor="#000000", s=50, zorder=6)
                ax.annotate(label, (cx, py), textcoords="offset points", xytext=(0, 11),
                            fontsize=7, color="#ffffff", ha="center", zorder=6)

    # Entry / SL / TP
    if setup.get("entry") is not None:
        ax.axhline(setup["entry"], color=COLORS["entry"], linestyle="--", linewidth=1.4, alpha=0.95)
        ax.text(chart_len * 0.75, setup["entry"], "ENTRY", fontsize=7, color=COLORS["entry"], va="bottom")
    if setup.get("sl") is not None:
        ax.axhline(setup["sl"], color=COLORS["sl"], linestyle="-", linewidth=1.3, alpha=0.9)
        ax.text(chart_len * 0.75, setup["sl"], "SL", fontsize=7, color=COLORS["sl"], va="top")
    for i, key in enumerate(("tp1", "tp2"), 1):
        if setup.get(key) is not None:
            ax.axhline(setup[key], color=COLORS["tp"], linestyle=":", linewidth=1.2, alpha=0.85)
            ax.text(chart_len * 0.75, setup[key], f"TP{i}", fontsize=7, color=COLORS["tp"], va="bottom")

    _draw_session_separators(ax, chart_df, chart_len)

    status = setup.get("trendline_status") or setup.get("direction", "")
    conf = setup.get("confidence", 0)
    ax.set_title(
        f"{symbol}  TRENDLINE MAP  |  {title_suffix}  |  {status}  |  Conf: {conf:.0f}%",
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

    if ticket.get("entry") is not None:
        ax.axhline(ticket["entry"], color=COLORS["entry"], linestyle="--", linewidth=1.6)
        ax.text(2, ticket["entry"], "ENTRY", fontsize=8, color=COLORS["entry"], fontweight="bold", va="bottom")
    if ticket.get("sl") is not None:
        ax.axhline(ticket["sl"], color=COLORS["sl"], linestyle="-", linewidth=1.5)
        ax.text(2, ticket["sl"], "SL", fontsize=8, color=COLORS["sl"], fontweight="bold", va="top")
    if ticket.get("tp1") is not None:
        ax.axhline(ticket["tp1"], color=COLORS["tp"], linestyle=":", linewidth=1.3)
        ax.text(2, ticket["tp1"], "TP1", fontsize=8, color=COLORS["tp"], fontweight="bold", va="bottom")
    if ticket.get("tp2") is not None:
        ax.axhline(ticket["tp2"], color=COLORS["tp"], linestyle=":", linewidth=1.3)
        ax.text(2, ticket["tp2"], "TP2", fontsize=8, color=COLORS["tp"], fontweight="bold", va="bottom")

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
