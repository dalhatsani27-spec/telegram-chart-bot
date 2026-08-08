"""
chart_engine.py
================
Chart rendering for the two remaining strategies: Trendline and OTE.

Dark TradingView-like theme, session separators, swing/structure labels,
and the shared LONG/SHORT position container (entry/SL/TP1/TP2 shaded
zones) used identically by both chart types.
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
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0d1117",
    "grid": "#1f2937",
    "text": "#e5e7eb",
    "bull": "#089981",
    "bear": "#f23645",
    "liquidity": "#ea80fc",
    "poc": "#ff9800",
    "trendline": "#00e676",
    "entry": "#00e676",
    "sl": "#f23645",
    "tp": "#26c6da",
}

# Session windows in UTC (start hour inclusive, end hour exclusive)
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



def _prepare_ohlc(df: pd.DataFrame, max_bars: int = 120) -> Tuple[pd.DataFrame, int]:
    if df is None or df.empty:
        raise ValueError("no chart data")
    chart_len = min(max_bars, len(df))
    chart_df = df.tail(chart_len).copy()
    chart_df.index = _to_naive_utc(pd.DatetimeIndex(chart_df.index))
    return chart_df, chart_len



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
    tp3 = pos.get("tp3")
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
    for tp, tp_label, tp_alpha in ((tp1, "TP1", 1.0), (tp2, "TP2", 0.75), (tp3, "TP3", 0.55)):
        if tp is None:
            continue
        ax.axhspan(min(entry, tp), max(entry, tp), xmin=x0_frac, xmax=1.0,
                   facecolor="#00e676", alpha=0.28 * tp_alpha, zorder=2)
        ax.axhline(tp, color="#00e676", linestyle=":", linewidth=1.8, xmin=x0_frac, alpha=max(tp_alpha, 0.55), zorder=4)
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




# ============================================================
# TRENDLINE CHART -- 30M entry chart for the Trendline strategy
# ============================================================
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
    for key in ("entry", "sl", "tp1", "tp2", "tp3"):
        if setup.get(key) is not None:
            price_min = min(price_min, float(setup[key]))
            price_max = max(price_max, float(setup[key]))
    pos = setup.get("position") or family.get("position")
    if pos:
        for key in ("entry", "sl", "tp1", "tp2", "tp3"):
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
            from market_analysis import zigzag_swings
            pivots = zigzag_swings(df, depth=4, deviation_atr=0.28)
        except Exception:
            pivots = []
    # LOCKED clean style (match hand-drawn maps): NO HH/HL spam, no zigzag skeleton
    # Only subtle markers on the few pivots that anchor the rails

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
    for i, tl in enumerate(rails[:3]):
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

    # Converging wedge/triangle: two independent-slope rails, NOT the
    # same-slope parallel family above. Rendered separately because they
    # represent a different structure (rails meeting at an apex) than a
    # parallel channel, and the old code had no path for this at all.
    wedge = family.get("wedge")
    if wedge:
        for rail, color in ((wedge["lower"], "#26a69a"), (wedge["upper"], "#ef5350")):
            x0 = max(0, int(rail["x0"]) - offset)
            # Extend to the apex (or chart edge, whichever comes first) so
            # the convergence is visible, matching how it's drawn by hand.
            apex = wedge.get("apex_index")
            x1 = chart_len - 1
            if apex is not None:
                apex_local = apex - offset
                if 0 < apex_local < chart_len * 1.15:
                    x1 = min(chart_len - 1, max(0, apex_local))
            y0 = _line_at(rail, max(int(rail["x0"]), 0))
            y1 = _line_at(rail, x1 + offset)
            if x0 < chart_len:
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.8, alpha=0.95, zorder=4)
        label_x = max(0, int(wedge["lower"]["x0"]) - offset)
        label_y = wedge["lower"]["y0"]
        ax.text(label_x, label_y, wedge["pattern"], fontsize=8, color="#ffffff",
                fontweight="bold", va="top", zorder=12,
                bbox=dict(boxstyle="round,pad=0.2", fc="#1c202a", ec="none", alpha=0.8))

    # M/W neckline (double top / double bottom)
    mw = family.get("mw_pattern")
    if mw and mw.get("neckline") is not None:
        ax.axhline(mw["neckline"], color="#ff9800", linestyle="--", linewidth=1.1, alpha=0.70, zorder=5)
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

    # Horizontal Support / Resistance -- clustered from the FULL pivot
    # history (see _detect_horizontal_levels), not just the last 6 swings.
    # A flip zone that's been tested repeatedly over the life of the chart
    # stays on the map even if its most recent touch has aged out of a
    # short recency window, which is what a trader marking levels by eye
    # actually does.
    try:
        hz = family.get("horizontal_levels") or []
        for lvl in hz:
            color = "#ef5350" if lvl["side"] == "resistance" else "#26a69a"
            tag = "R" if lvl["side"] == "resistance" else "S"
            ax.axhline(lvl["price"], color=color, linestyle="--", linewidth=1.0, alpha=0.6, zorder=3)
            ax.text(chart_len * 0.01, lvl["price"], f"{tag} ({lvl['touches']}x)",
                    fontsize=7, color=color, va="bottom" if tag == "R" else "top", fontweight="bold")
        if not hz:
            # Fallback to the old recent-pivot read only if clustering found nothing
            recent_piv = [p for p in (pivots or []) if 0 <= int(p.get("index", -1)) - offset < chart_len][-6:]
            highs = sorted([float(p["price"]) for p in recent_piv if p.get("type") == "high"], reverse=True)
            lows = sorted([float(p["price"]) for p in recent_piv if p.get("type") == "low"])
            for price in highs[:2]:
                ax.axhline(price, color="#ef5350", linestyle="--", linewidth=1.0, alpha=0.55, zorder=3)
                ax.text(chart_len * 0.01, price, "R", fontsize=7, color="#ef5350", va="bottom", fontweight="bold")
            for price in lows[:2]:
                ax.axhline(price, color="#26a69a", linestyle="--", linewidth=1.0, alpha=0.55, zorder=3)
                ax.text(chart_len * 0.01, price, "S", fontsize=7, color="#26a69a", va="top", fontweight="bold")
    except Exception:
        pass

    # Fibonacci pullback levels (for trend entries) — 0.5 / 0.618 / 0.705 / 0.79
    # Anchored to the last clear impulse leg from non-ranging pivots
    try:
        direction = str(family.get("direction") or setup.get("direction") or "").upper()
        if direction in ("BUY", "SELL", "LONG", "SHORT", "BULLISH", "BEARISH") and pivots and len(pivots) >= 2:
            # Last impulse: previous swing -> latest swing in trend direction
            leg = pivots[-2:]
            a, b = leg[0], leg[1]
            hi = max(float(a["price"]), float(b["price"]))
            lo = min(float(a["price"]), float(b["price"]))
            span = hi - lo
            if span > 0:
                # Standard ICT/SMC OTE-style pullback zone
                fibs = [
                    (0.50, "0.5"),
                    (0.618, "0.618"),
                    (0.705, "0.705"),
                    (0.79, "0.79"),
                ]
                is_buy = direction in ("BUY", "LONG", "BULLISH")
                for ratio, label in fibs:
                    if is_buy:
                        # Pullback down into discount of bullish impulse
                        price = hi - span * ratio
                    else:
                        # Pullback up into premium of bearish impulse
                        price = lo + span * ratio
                    ax.axhline(price, color="#ab47bc", linestyle=":", linewidth=0.95, alpha=0.75, zorder=3)
                    ax.text(chart_len * 0.70, price, f"Fib {label}", fontsize=6.5,
                            color="#ce93d8", va="bottom", alpha=0.9)
                # Highlight OTE band (0.618–0.79) lightly
                if is_buy:
                    ote_top = hi - span * 0.618
                    ote_bot = hi - span * 0.79
                else:
                    ote_bot = lo + span * 0.618
                    ote_top = lo + span * 0.79
                ax.axhspan(min(ote_bot, ote_top), max(ote_bot, ote_top),
                           facecolor="#ab47bc", alpha=0.06, zorder=1)
    except Exception:
        pass

    # Volume Profile — keep POC + Value Area (useful), no heavy histogram clutter
    vp = family.get("volume_profile") or setup.get("volume_profile")
    if vp:
        if vp.get("poc_price") is not None:
            ax.axhline(vp["poc_price"], color=COLORS["poc"], linestyle=":", linewidth=1.25, alpha=0.9)
            ax.text(chart_len * 0.55, vp["poc_price"], "POC", fontsize=7, color=COLORS["poc"], va="bottom")
        if vp.get("value_area_low") is not None and vp.get("value_area_high") is not None:
            # Light Value Area band (not the old heavy orange box)
            ax.axhspan(vp["value_area_low"], vp["value_area_high"],
                       facecolor="#ff9800", alpha=0.05, zorder=1)
            ax.axhline(vp["value_area_high"], color="#ffb74d", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.axhline(vp["value_area_low"], color="#ffb74d", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.text(chart_len * 0.01, vp["value_area_high"], "VA-H", fontsize=6, color="#ffb74d", va="bottom")
            ax.text(chart_len * 0.01, vp["value_area_low"], "VA-L", fontsize=6, color="#ffb74d", va="top")

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


# ============================================================
# OTE CHART -- 30M Fibonacci Fan + Expansion chart
# ============================================================
def generate_ote_map(
    df: pd.DataFrame,
    symbol: str,
    analysis: Dict[str, Any],
    title_suffix: str = "",
) -> io.BytesIO:
    """
    OTE visual map:
      - Candlesticks
      - Fibonacci Fan rays (38.2 / 50 / 61.8) in green
      - Fibonacci Expansion levels (127.2 / 161.8 / 200 / 261.8)
      - Impulse start / end markers
      - Position container (Entry / SL / TP) when available
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

    fig, axlist = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        volume=False,
        returnfig=True,
        figsize=(12, 6.8),
        warn_too_much_data=10000,
    )
    ax = axlist[0]

    # --- Impulse anchors ---
    impulse = analysis.get("impulse") or {}
    start = impulse.get("start")
    end = impulse.get("end")
    if start:
        x = start["index"] - offset
        if 0 <= x < chart_len:
            ax.scatter([x], [start["price"]], color="#00e676", s=55, zorder=8, marker="o")
            ax.text(x, start["price"], "  Start", fontsize=7, color="#00e676", va="bottom")
    if end:
        x = end["index"] - offset
        if 0 <= x < chart_len:
            ax.scatter([x], [end["price"]], color="#ffab00", s=55, zorder=8, marker="o")
            ax.text(x, end["price"], "  End", fontsize=7, color="#ffab00", va="bottom")

    # --- Fibonacci Fan rays ---
    fans = analysis.get("fans") or []
    fan_colors = ["#69f0ae", "#00e676", "#00c853"]
    for i, fan in enumerate(fans):
        x0 = fan["x0"] - offset
        x1 = chart_len - 1
        if x0 >= chart_len:
            continue
        y0 = fan["y0"]
        y1 = fan["y0"] + fan["slope"] * ((offset + chart_len - 1) - fan["x0"])
        color = fan_colors[i % len(fan_colors)]
        y_left = y0 if x0 >= 0 else (fan["y0"] + fan["slope"] * (offset - fan["x0"]))
        ax.plot([max(0, x0), x1], [y_left, y1],
                color=color, linewidth=1.6, alpha=0.90, zorder=5)
        ax.text(chart_len - 2, y1, f" {fan['label']}", fontsize=7,
                color=color, va="center", fontweight="bold")

    # --- Expansion levels ---
    expansions = analysis.get("expansions") or []
    exp_colors = ["#26c6da", "#00bcd4", "#0097a7", "#00838f"]
    for i, exp in enumerate(expansions):
        price = exp["price"]
        color = exp_colors[i % len(exp_colors)]
        ax.axhline(price, color=color, linestyle="--", linewidth=1.25, alpha=0.85, zorder=4)
        ax.text(chart_len * 0.72, price, f" Exp {exp['label']}", fontsize=7,
                color=color, va="bottom")

    # --- Position container ---
    pos = analysis.get("position") or analysis.get("ticket")
    if pos:
        _draw_position_container(ax, pos, chart_len)

    direction = analysis.get("direction", "")
    score = analysis.get("score", 0)
    title = f"{symbol}  OTE (Fib Fan + Expansion)  |  {direction}  |  Score {score}"
    if title_suffix:
        title += f"  |  {title_suffix}"
    ax.set_title(title, color=COLORS["text"], fontsize=10, fontweight="bold", pad=10)

    # Price padding
    prices = list(chart_df["High"]) + list(chart_df["Low"])
    for f in fans:
        prices.append(f.get("y_at_end", f["y0"]))
    for e in expansions:
        prices.append(e["price"])
    if prices:
        pmin, pmax = min(prices), max(prices)
        pad = (pmax - pmin) * 0.08
        ax.set_ylim(pmin - pad, pmax + pad)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf
