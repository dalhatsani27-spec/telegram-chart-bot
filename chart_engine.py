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
import textwrap
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


def _draw_position_panel(fig, panel_ax, pos: Optional[Dict], reasons: Optional[List[str]] = None):
    """Side info panel (Entry/SL/TP + confirmation checklist) instead of
    the text stack overlaid on the candles. Mirrors the reference layout:
    a colored header, a plain-text readout of the levels, then the actual
    entry-rule checklist (candle / structure / momentum / RSI) with
    pass/fail marks -- not just prose notes."""
    panel_ax.set_facecolor(COLORS["bg"])
    for spine in panel_ax.spines.values():
        spine.set_visible(False)
    panel_ax.set_xticks([])
    panel_ax.set_yticks([])
    panel_ax.set_xlim(0, 1)
    panel_ax.set_ylim(0, 1)

    if not pos or pos.get("entry") is None:
        panel_ax.text(0.5, 0.5, "No active setup", ha="center", va="center",
                       color=COLORS["text"], fontsize=9, alpha=0.6)
        return

    entry = pos.get("entry")
    sl = pos.get("sl")
    tp1 = pos.get("tp1")
    tp2 = pos.get("tp2")
    tp3 = pos.get("tp3")
    side = str(pos.get("side") or pos.get("direction") or "").upper()
    is_long = ("LONG" in side) or (side == "BUY") or ("BULL" in side)
    confirmed = bool(pos.get("confirmed"))
    label = ("LONG" if is_long else "SHORT") if confirmed else "WAIT"
    # Core Rule: "never force a trade" -- an unconfirmed geometric bias
    # gets an amber WAIT header instead of a green/red LONG/SHORT one, so
    # it visually reads as "not yet", not as a live signal.
    box_color = ("#00e676" if is_long else "#ff1744") if confirmed else "#ffab00"

    risk = abs(entry - sl) if sl is not None else None
    # Prefer the pre-computed R:R from build_position_container -- it's
    # quoted against TP2 (a real "next resistance/support" target), not
    # TP1 (a deliberately-close first partial), so it doesn't make every
    # setup look worse than the actual plan by dividing by the smallest target.
    if pos.get("rr") is not None:
        rr_txt = f"R:R 1:{pos['rr']:.1f}"
    else:
        reward = abs((tp1 if tp1 is not None else tp2) - entry) if (tp1 is not None or tp2 is not None) else None
        rr_txt = f"R:R 1:{(reward / risk):.1f}" if risk and reward else ""

    def fmt(p):
        return f"{p:.5f}" if p < 100 else f"{p:.2f}"

    y = 0.97
    panel_ax.add_patch(mpatches.FancyBboxPatch(
        (0.03, y - 0.05), 0.94, 0.05, boxstyle="round,pad=0.01",
        facecolor=box_color, edgecolor="none", alpha=0.95,
        transform=panel_ax.transAxes, zorder=5))
    panel_ax.text(0.5, y - 0.025, f"{label}  {rr_txt}", ha="center", va="center",
                   fontsize=11, fontweight="bold",
                   color="#000000" if (is_long or not confirmed) else "#ffffff", zorder=6)
    y -= 0.09

    rows = [("ENTRY", entry, COLORS["entry"], None)]
    if tp3 is not None:
        tp3_sub = pos.get("tp3_basis") or "RR 1:2/1:3 target"
        rows.append(("TP3", tp3, "#00e676", tp3_sub))
    if tp2 is not None:
        rows.append(("TP2", tp2, "#00e676", "next resistance/support"))
    if tp1 is not None:
        tp1_rr = pos.get("rr_tp1")
        rows.append(("TP1", tp1, "#00e676", f"partial, RR 1:{tp1_rr:.1f}" if tp1_rr is not None else "partial target"))
    if sl is not None:
        rows.append(("STOP LOSS", sl, "#ff1744", None))

    for name, price, color, note in rows:
        panel_ax.text(0.05, y, name, fontsize=8.5, fontweight="bold", color=color, va="top")
        panel_ax.text(0.95, y, fmt(price), fontsize=8.5, color=color, va="top", ha="right")
        y -= 0.036
        if note:
            panel_ax.text(0.05, y, note, fontsize=6.3, color=color, va="top", alpha=0.65)
            y -= 0.03
        y -= 0.012

    y -= 0.025
    panel_ax.axhline(y, xmin=0.03, xmax=0.97, color=COLORS["grid"], linewidth=1, alpha=0.6)
    y -= 0.045

    # --- Entry rules checklist (the actual "Confirmation" section from
    # the reference image: candle pattern / structure break / momentum /
    # RSI vs 50), rendered as pass/fail marks, not prose. ---
    entry_rules = pos.get("entry_rules")
    panel_ax.text(0.05, y, "ENTRY RULES", fontsize=8, fontweight="bold",
                   color=COLORS["text"], va="top", alpha=0.9)
    y -= 0.042
    check_labels = {
        "candle": "Candle confirmation",
        "structure": "Break of minor structure",
        "momentum": "Volume / momentum",
        "rsi": "RSI confirms direction",
    }
    if entry_rules:
        for key, title in check_labels.items():
            ok, detail = entry_rules["checks"].get(key, (False, ""))
            mark = "✓" if ok else "✗"
            mark_color = "#00e676" if ok else "#78828e"
            panel_ax.text(0.05, y, mark, fontsize=8.5, fontweight="bold", color=mark_color, va="top")
            panel_ax.text(0.12, y, title, fontsize=7.3, color=COLORS["text"], va="top", alpha=0.9)
            y -= 0.036
            if detail:
                for line in textwrap.wrap(str(detail), width=32)[:1]:
                    panel_ax.text(0.12, y, line, fontsize=6.5, color=COLORS["text"], va="top", alpha=0.6)
                    y -= 0.032
            y -= 0.006
        y -= 0.012
        panel_ax.text(0.05, y, f"{entry_rules['passed']}/4 checks passed"
                       f" (need {entry_rules['required']}+)",
                       fontsize=7, fontweight="bold", va="top",
                       color="#00e676" if confirmed else "#ffab00")
        y -= 0.05
    else:
        panel_ax.text(0.05, y, "n/a", fontsize=7.3, color=COLORS["text"], va="top", alpha=0.6)
        y -= 0.05

    y -= 0.01
    panel_ax.axhline(y, xmin=0.03, xmax=0.97, color=COLORS["grid"], linewidth=1, alpha=0.6)
    y -= 0.045

    panel_ax.text(0.05, y, "NOTES", fontsize=8, fontweight="bold",
                   color=COLORS["text"], va="top", alpha=0.85)
    y -= 0.042
    # Skip the entry-confirmation reason here -- it's already shown in
    # full detail by the checklist above; just show the structural notes.
    other_reasons = [r for r in (reasons or []) if not str(r).startswith(("Entry confirmed", "⚠ Entry"))]
    for note in other_reasons[:2]:
        wrapped = textwrap.wrap(str(note), width=32)
        for line in wrapped[:2]:
            panel_ax.text(0.05, y, f"• {line}" if line == wrapped[0] else f"  {line}",
                           fontsize=7, color=COLORS["text"], va="top", alpha=0.8)
            y -= 0.034
        y -= 0.01
        if y < 0.03:
            break


def _draw_price_reference_lines(ax, pos: Optional[Dict], chart_len: int):
    """Thin dashed lines + shaded risk/reward zones only -- no text, no
    banner. The actual numbers live in the side panel now, so the chart
    itself just shows where those levels sit relative to price action."""
    if not pos or pos.get("entry") is None:
        return
    entry = pos.get("entry")
    sl = pos.get("sl")
    tp1 = pos.get("tp1")
    tp2 = pos.get("tp2")
    tp3 = pos.get("tp3")
    x0_frac = 0.82

    if sl is not None:
        ax.axhspan(min(entry, sl), max(entry, sl), xmin=x0_frac, xmax=1.0,
                   facecolor="#ff1744", alpha=0.16, zorder=2)
        ax.axhline(sl, color="#ff1744", linestyle="-", linewidth=1.4, xmin=x0_frac, zorder=4)
    for tp, tp_alpha in ((tp1, 1.0), (tp2, 0.6), (tp3, 0.35)):
        if tp is None:
            continue
        ax.axhspan(min(entry, tp), max(entry, tp), xmin=x0_frac, xmax=1.0,
                   facecolor="#00e676", alpha=0.14 * tp_alpha, zorder=2)
        ax.axhline(tp, color="#00e676", linestyle=":", linewidth=1.3, xmin=x0_frac,
                   alpha=max(tp_alpha, 0.55), zorder=4)
    ax.axhline(entry, color=COLORS["entry"], linestyle="--", linewidth=1.6, xmin=x0_frac, zorder=5)


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

    # NOTE: the old code plotted upper_line/middle_line/lower_line here via
    # mplfinance addplot AND ALSO drew the same rails again further down via
    # ax.plot() in the "family rails" loop -- the channel was being drawn
    # twice on top of itself, which is a big part of why the map looked
    # noisy. We now draw the pattern/channel exactly once, further down,
    # after we've decided which single pattern actually gets the chart.
    addplots = []

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

    fig = plt.figure(figsize=(15.4, 7.6), facecolor=COLORS["bg"])
    gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.0], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    panel_ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(COLORS["bg"])

    mpf_kwargs = dict(type="candle", style=style, volume=False, ax=ax)
    if addplots:
        mpf_kwargs["addplot"] = addplots
    mpf.plot(chart_df, **mpf_kwargs)
    ax.set_ylim(price_min - padding, price_max + padding)

    # --- ALWAYS map pivot points (structure anchors) ---
    pivots = family.get("pivots") or []
    if not pivots:
        try:
            from market_analysis import zigzag_swings
            pivots = zigzag_swings(df, depth=4, deviation_atr=0.28)
        except Exception:
            pivots = []

    def _line_at(tl, x):
        x0, y0, x1, y1 = tl["x0"], tl["y0"], tl["x1"], tl["y1"]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def _pivot_dot(px, py, text, color, offset_idx=0):
        """Single small dot + one-word label, offset so it never stacks
        with other annotations (matches the reference 'Upper'/'Lower'
        pivot-map style instead of dense boxed text). offset_idx staggers
        the label vertically when two pivots land close together (e.g. a
        tight double top/bottom) so the words don't print on top of each other."""
        if not (0 <= px < chart_len):
            return
        ax.scatter([px], [py], s=46, c="#ffffff", edgecolors=color, linewidths=1.6, zorder=11)
        y_off = 9 + offset_idx * 11
        ax.annotate(text, (px, py), fontsize=7.5, color="#e8e8e8", fontweight="bold",
                    xytext=(0, y_off), textcoords="offset points", ha="center", zorder=12)

    family_kind = family.get("family_kind", "")
    bias_color = "#00e676" if family_kind == "ascending" else "#ff5252"

    # --- Order block zones -- "likely to be respected" ones only (see
    # detect_order_blocks: structure-confirmed, decent displacement, not
    # already invalidated). Drawn as translucent boxes from where the zone
    # formed out to the current bar, color-coded so bullish (demand) reads
    # green and bearish (supply) reads red rather than one flat color.
    for ob in (family.get("order_blocks") or []):
        x0 = int(ob["formed_index"]) - offset
        if x0 >= chart_len:
            continue
        x0 = max(0, x0)
        is_bull = ob["type"] == "bullish"
        is_unmitigated = ob["freshness"] == "untested"
        # Deep, solid fill colors matching how a trader actually marks these
        # by hand (solid maroon/navy blocks), not a light translucent tint.
        color = "#0d3ea3" if is_bull else "#7a0d16"
        edge_color = "#26a69a" if is_bull else "#ef5350"
        # Unmitigated (never traded back into) zones are the ones still in
        # play -- draw them as bold, nearly-opaque blocks. Mitigated
        # ("tested-held") zones already did their job once; keep them on
        # the chart for context but shrink them to a thin outline so they
        # don't compete visually with the zones that actually matter.
        if is_unmitigated:
            alpha = 0.85 if ob["grade"] == "strong" else 0.65
            linestyle = "solid"
            linewidth = 1.4
        else:
            alpha = 0.10
            linestyle = "dashed"
            linewidth = 0.8
        ax.add_patch(mpatches.Rectangle(
            (x0, ob["bottom"]), (chart_len - 1 - x0), (ob["top"] - ob["bottom"]),
            facecolor=color, edgecolor=edge_color, linewidth=linewidth, linestyle=linestyle,
            alpha=alpha, zorder=1))
        # Solid level line at the edge price actually has to clear to
        # invalidate the zone (top for a bearish/supply OB since price
        # approaches from below, bottom for a bullish/demand OB since price
        # approaches from above) -- the single number a trader actually
        # watches, extended to the right edge of the chart like a live level.
        if is_unmitigated:
            edge_price = ob["bottom"] if is_bull else ob["top"]
            ax.plot([x0, chart_len - 1], [edge_price, edge_price],
                     color=edge_color, linestyle="-", linewidth=1.2, alpha=0.9, zorder=6)
        status = "UNMITIGATED" if is_unmitigated else "mitigated"
        label = f"{'Bullish' if is_bull else 'Bearish'} OB · {status} · {ob['confidence']}%"
        ax.text(x0 + 1, ob["top"] if is_bull else ob["bottom"], label, fontsize=6.5,
                color="#ffffff" if is_unmitigated else edge_color, fontweight="bold",
                va="bottom" if is_bull else "top",
                alpha=0.95 if is_unmitigated else 0.55, zorder=9)

    # --- Classic chart pattern (triangle/wedge/flag/pennant/rectangle/H&S) -
    # Drawn straight from strategies.py's scanned_pattern (market_analysis
    # .scan_all_patterns output). One generic renderer handles every
    # pattern type since they all share the same Pattern schema:
    # trigger_line (the breakout level to watch) + key_points (labeled
    # boundary/marker points). For triangles/wedges/rectangles, key_points
    # carries BOTH boundary sides labeled separately, so group-by-label
    # reconstructs the full two-line shape instead of just the one trigger
    # side that gets stored in trigger_line.
    sp = family.get("scanned_pattern")
    if sp:
        p_bias = sp.get("bias")
        p_color = "#26a69a" if p_bias == "BUY" else "#ef5350" if p_bias == "SELL" else "#ffb74d"
        key_points = sp.get("key_points") or []
        trigger_line = sp.get("trigger_line") or []
        name = sp.get("name", "")

        def _cx(idx):
            return idx - offset

        if name in ("Bull Flag", "Bear Flag", "Bullish Pennant", "Bearish Pennant"):
            # Pole: diagonal line through the two labeled pole points.
            pole_pts = sorted(key_points, key=lambda kp: kp[0])
            if len(pole_pts) >= 2:
                (px0, py0, _), (px1, py1, _) = pole_pts[0], pole_pts[-1]
                ax.plot([_cx(px0), _cx(px1)], [py0, py1], color=p_color,
                        linewidth=1.6, alpha=0.85, zorder=6)
            # Flag/pennant box: the consolidation boundary stored as trigger_line.
            if len(trigger_line) == 2:
                (fx0, fy0), (fx1, fy1) = trigger_line
                ax.plot([_cx(fx0), _cx(fx1)], [fy0, fy1], color=p_color,
                        linewidth=1.4, linestyle="--", alpha=0.9, zorder=6)
        else:
            # Group labeled key_points -> reconstruct both boundary lines
            # (triangle/wedge/rectangle) or plot bare markers (H&S, double
            # top/bottom, where each label is a single point, not a line).
            groups: Dict[str, List[Tuple[float, float]]] = {}
            for kp in key_points:
                if len(kp) >= 3:
                    x, y, lbl = kp[0], kp[1], kp[2]
                else:
                    x, y, lbl = kp[0], kp[1], "pt"
                groups.setdefault(lbl, []).append((x, y))

            any_line_drawn = False
            for lbl, pts in groups.items():
                if len(pts) < 2:
                    continue
                pts_sorted = sorted(pts, key=lambda p: p[0])
                (x0, y0), (x1, y1) = pts_sorted[0], pts_sorted[-1]
                x_end = chart_len - 1  # extend the boundary out to the current bar
                if x1 != x0:
                    slope = (y1 - y0) / (x1 - x0)
                    y_end = y0 + slope * ((x_end + offset) - x0)
                else:
                    y_end = y1
                ax.plot([_cx(x0), x_end], [y0, y_end], color=p_color,
                        linewidth=1.5, alpha=0.85, zorder=6)
                any_line_drawn = True

            if not any_line_drawn:
                # Marker-point pattern (H&S / double top-bottom / triple).
                for lbl, pts in groups.items():
                    for x, y in pts:
                        ax.plot(_cx(x), y, marker="o", markersize=4,
                                color=p_color, zorder=7)
                        ax.annotate(lbl, (_cx(x), y), fontsize=6, color=p_color,
                                    xytext=(3, 3), textcoords="offset points", zorder=8)
                # Neckline / trigger line, still drawn straight across.
                if len(trigger_line) == 2:
                    (tx0, ty0), (tx1, ty1) = trigger_line
                    ax.plot([_cx(tx0), chart_len - 1], [ty0, ty0 if ty0 == ty1 else ty1],
                            color=p_color, linewidth=1.2, linestyle="--", alpha=0.8, zorder=6)

        trig_price = sp.get("trigger_price")
        if trig_price is not None:
            ax.axhline(y=trig_price, color=p_color, linestyle=":", linewidth=1.0, alpha=0.55, zorder=5)

        label_x = _cx(min((kp[0] for kp in key_points), default=offset))
        label_y = max((kp[1] for kp in key_points), default=trig_price or 0)
        ax.text(max(2, label_x), label_y, f"{name} ({sp.get('confidence', 0):.0f}%)",
                fontsize=7, color=p_color, fontweight="bold", va="bottom", zorder=10,
                bbox=dict(boxstyle="round", facecolor="black", edgecolor=p_color, alpha=0.6, pad=0.2))

    # --- Pick exactly ONE structure to draw as "the pattern" -------------
    # Priority set upstream in strategies.py (active_pattern): a specific
    # named reversal (M/W) beats a converging wedge/triangle beats a plain
    # parallel channel. Whichever it is, it's the only shape drawn here --
    # no more wedge + neckline + channel stacked on the same candles.
    active_pattern = family.get("active_pattern", "none")
    pattern_title = None
    pattern_conf = family.get("pattern_confidence")

    wedge = family.get("wedge")
    mw = family.get("mw_pattern")

    if active_pattern == "mw" and mw and mw.get("neckline") is not None:
        pattern_title = mw.get("name", "M/W Pattern")
        # Neckline: one line, extended to the chart edge (the level to watch)
        neck_x0 = max(0, int(mw.get("neck_index", 0)) - offset)
        ax.plot([neck_x0, chart_len - 1], [mw["neckline"], mw["neckline"]],
                color="#ff9800", linestyle="--", linewidth=1.4, alpha=0.85, zorder=5)
        ax.text(chart_len * 0.015, mw["neckline"], "Neckline", fontsize=7.5,
                color="#ffb74d", fontweight="bold", va="bottom", zorder=12)
        # The two matching peaks/troughs that define the pattern
        left, right = mw.get("left"), mw.get("right")
        tag = "Top" if mw["pattern"] == "M" else "Bottom"
        close_x = (left and right and abs(int(left["index"]) - int(right["index"])) < chart_len * 0.05)
        for i, p in enumerate((left, right)):
            if p:
                px = int(p["index"]) - offset
                _pivot_dot(px, float(p["price"]), tag, "#ff9800", offset_idx=(i if close_x else 0))

    elif active_pattern == "wedge" and wedge:
        pattern_title = wedge["pattern"]
        apex = wedge.get("apex_index")
        for rail, color, tag in ((wedge["lower"], "#26a69a", "Lower"), (wedge["upper"], "#ef5350", "Upper")):
            x0 = max(0, int(rail["x0"]) - offset)
            x1 = chart_len - 1
            if apex is not None:
                apex_local = apex - offset
                if 0 < apex_local < chart_len * 1.15:
                    x1 = min(chart_len - 1, max(0, apex_local))
            y0 = _line_at(rail, max(int(rail["x0"]), 0))
            y1 = _line_at(rail, x1 + offset)
            if x0 < chart_len:
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.7, alpha=0.9, zorder=4)
            # Label only the two real pivots that anchor this rail
            for ax_key, ay_key in (("x0", "y0"), ("x1", "y1")):
                px = int(rail[ax_key]) - offset
                _pivot_dot(px, float(rail[ay_key]), tag, color)

    elif active_pattern == "channel" and family.get("channel"):
        pattern_title = f"{family_kind.capitalize()} Channel" if family_kind != "none" else "Channel"
        ch = family["channel"]
        for rail, tag in ((ch.get("lower"), "Lower"), (ch.get("upper"), "Upper")):
            if not rail:
                continue
            x0 = max(0, int(rail["x0"]) - offset)
            y0 = _line_at(rail, max(int(rail["x0"]), 0))
            y1 = float(rail.get("y_end", rail.get("y1", 0)))
            if x0 < chart_len:
                ax.plot([x0, chart_len - 1], [y0, y1], color=bias_color, linewidth=1.4, alpha=0.75, zorder=4)

    # --- Directional bias trendline: ALWAYS drawn on top -------------
    # Connects swing lows in an uptrend / swing highs in a downtrend --
    # this is the single line that answers "what's the bias", independent
    # of whichever pattern shape (if any) is drawn above.
    primary = (family.get("uptrends") or family.get("downtrends") or [None])[0]
    if primary:
        x0 = max(0, int(primary["x0"]) - offset)
        y0 = _line_at(primary, max(int(primary["x0"]), 0))
        y1 = float(primary.get("y_end", primary.get("y1", 0)))
        if x0 < chart_len:
            ax.plot([x0, chart_len - 1], [y0, y1], color="#ffd600", linestyle="--",
                     linewidth=2.2, alpha=0.95, zorder=8)
        end_pts_x = set()
        for ax_key, ay_key in (("x0", "y0"), ("x1", "y1")):
            px = int(primary[ax_key]) - offset
            end_pts_x.add(px)
            if 0 <= px < chart_len:
                ax.scatter([px], [float(primary[ay_key])], s=60, c="#ffd600",
                           edgecolors="#000000", linewidths=1.0, zorder=11, marker="D")
        # Every other wick that actually touched the line -- small hollow
        # circles, like a trader circling each bounce off the trendline.
        # Capped and evenly re-sampled so a very tight/long-lived line
        # (dozens of touches) still reads as "circled bounces", not a
        # second solid line drawn out of overlapping markers.
        touches = [tp for tp in (family.get("bias_touch_points") or [])
                   if (int(tp["index"]) - offset) not in end_pts_x]
        MAX_TOUCH_MARKERS = 10
        if len(touches) > MAX_TOUCH_MARKERS:
            step = len(touches) / MAX_TOUCH_MARKERS
            touches = [touches[int(i * step)] for i in range(MAX_TOUCH_MARKERS)]
        for tp in touches:
            px = int(tp["index"]) - offset
            if not (0 <= px < chart_len):
                continue
            ax.scatter([px], [float(tp["price"])], s=42, facecolors="none",
                       edgecolors="#ffd600", linewidths=1.4, zorder=10)

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

    # Fibonacci pullback levels intentionally NOT drawn on this chart.
    # They anchor near the last impulse leg -- i.e. right next to current
    # price -- which is exactly where the entry/SL/TP box also sits, so
    # they were the main cause of the illegible "0.5 / 0.618 / ..." text
    # pile-up over the signal box. The OTE chart (generate_ote_map) is the
    # dedicated place for fib levels; keep this chart to pattern + bias +
    # position only.

    # Volume Profile — POC only, kept subtle. Value Area band/lines removed:
    # they were adding two more dotted lines + labels right in the same
    # price region as everything else (OBs, trendlines, S/R, signal box),
    # for a level that duplicates what S/R clustering already shows.
    vp = family.get("volume_profile") or setup.get("volume_profile")
    if vp and vp.get("poc_price") is not None:
        ax.axhline(vp["poc_price"], color=COLORS["poc"], linestyle=":", linewidth=0.8, alpha=0.5, zorder=2)
        ax.text(chart_len * 0.55, vp["poc_price"], "POC", fontsize=6, color=COLORS["poc"],
                va="bottom", alpha=0.6)

    # Position: thin reference lines on the chart, full readout in the side panel
    pos = setup.get("position") or family.get("position")
    if not pos and setup.get("entry") is not None:
        pos = {
            "entry": setup.get("entry"), "sl": setup.get("sl"),
            "tp1": setup.get("tp1"), "tp2": setup.get("tp2"),
            "side": setup.get("direction", ""),
        }
    _draw_price_reference_lines(ax, pos, chart_len)
    _draw_position_panel(fig, panel_ax, pos, family.get("reasons"))

    _draw_session_separators(ax, chart_df, chart_len)

    direction = family.get("direction") or setup.get("direction", "")
    strength = family.get("strength") or setup.get("confidence") or setup.get("score") or 0
    line1 = f"{symbol}  |  TF: 30m  |  {title_suffix}  |  {direction}  |  Str {strength:.0f}"
    if pattern_title:
        conf = pattern_conf if pattern_conf is not None else strength
        line2 = f"Pattern: {pattern_title}  |  Confidence: {conf:.0f}%"
    else:
        line2 = "No named pattern — trading directional bias only"

    # 4H/1H top-down alignment -- was already computed and penalizing/boosting
    # strength, but only showed up in the text caption next to the chart, not
    # on the image itself. Surface it directly so it's visible at a glance.
    topdown = family.get("topdown") or {}
    td_dir = topdown.get("direction", "NEUTRAL")
    line3 = None
    title_color = COLORS["text"]
    if direction in ("BUY", "SELL") and td_dir in ("BUY", "SELL"):
        if td_dir == direction:
            tag = "✅ aligned" if topdown.get("allowed") else "aligned, 1H permission pending"
            line3 = f"4H/1H bias: {td_dir} — {tag}"
        else:
            line3 = f"⚠️ 4H/1H bias: {td_dir} — conflicts with this {direction}, counter-trend risk"
            title_color = "#ffb74d"

    title_text = f"{line1}\n{line2}"
    if line3:
        title_text += f"\n{line3}"
    fig.suptitle(
        title_text,
        color=title_color,
        fontsize=9.5,
        fontweight="bold",
        y=0.99,
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
