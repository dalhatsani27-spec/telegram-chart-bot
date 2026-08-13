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
    too_extended = bool(pos.get("too_extended"))

    if too_extended or entry is None:
        label = "NO ENTRY"
        box_color = "#546e7a"
        rr_txt = "extended"
    else:
        label = ("LONG" if is_long else "SHORT") if confirmed else "WAIT"
        # Core Rule: "never force a trade" -- unconfirmed geometric bias
        # gets amber WAIT instead of a live green/red signal.
        box_color = ("#00e676" if is_long else "#ff1744") if confirmed else "#ffab00"
        risk = abs(entry - sl) if (sl is not None and entry is not None) else None
        if pos.get("rr") is not None:
            rr_txt = f"R:R 1:{pos['rr']:.1f}"
        else:
            reward = abs((tp1 if tp1 is not None else tp2) - entry) if (entry is not None and (tp1 is not None or tp2 is not None)) else None
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
        role = " · INDUCEMENT" if ob.get("is_inducement") else ""
        label = f"{'Bullish' if is_bull else 'Bearish'} OB · {status}{role} · {ob['confidence']}%"
        ax.text(x0 + 1, ob["top"] if is_bull else ob["bottom"], label, fontsize=6.5,
                color="#ffffff" if is_unmitigated else edge_color, fontweight="bold",
                va="bottom" if is_bull else "top",
                alpha=0.95 if is_unmitigated else 0.55, zorder=9)

    # --- Classic chart pattern (triangle/wedge/flag/pennant/rectangle/H&S) -
    # Clean educational-style rendering: thick clear lines, proper labels,
    # minimal clutter. Only high-confidence patterns reach this point.
    sp = family.get("scanned_pattern")
    if sp:
        p_bias = sp.get("bias")
        # Strong educational-chart colors
        p_color = "#00c853" if p_bias == "BUY" else "#ff1744" if p_bias == "SELL" else "#ffb300"
        key_points = sp.get("key_points") or []
        trigger_line = sp.get("trigger_line") or []
        name = sp.get("name", "")

        def _cx(idx):
            return idx - offset

        if name in ("Bull Flag", "Bear Flag", "Bullish Pennant", "Bearish Pennant"):
            # Pole: solid diagonal
            pole_pts = sorted(key_points, key=lambda kp: kp[0])
            if len(pole_pts) >= 2:
                (px0, py0, _), (px1, py1, _) = pole_pts[0], pole_pts[-1]
                ax.plot([_cx(px0), _cx(px1)], [py0, py1], color=p_color,
                        linewidth=2.4, alpha=0.95, zorder=6, solid_capstyle="round")
            # Flag/pennant consolidation boundary
            if len(trigger_line) == 2:
                (fx0, fy0), (fx1, fy1) = trigger_line
                ax.plot([_cx(fx0), _cx(fx1)], [fy0, fy1], color=p_color,
                        linewidth=2.0, linestyle="--", alpha=0.9, zorder=6)
        else:
            # Group labeled key_points
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
                x_end = chart_len - 1
                if x1 != x0:
                    slope = (y1 - y0) / (x1 - x0)
                    y_end = y0 + slope * ((x_end + offset) - x0)
                else:
                    y_end = y1
                ax.plot([_cx(x0), x_end], [y0, y_end], color=p_color,
                        linewidth=2.2, alpha=0.92, zorder=6, solid_capstyle="round")
                any_line_drawn = True

            if not any_line_drawn:
                # Marker-point patterns (H&S, Inverse H&S, Double Top/Bottom)
                # Large clear markers + bold labels like educational charts
                for lbl, pts in groups.items():
                    for x, y in pts:
                        cx = _cx(x)
                        if not (0 <= cx < chart_len):
                            continue
                        ax.scatter([cx], [y], s=90, c=p_color, edgecolors="#ffffff",
                                   linewidths=1.8, zorder=9, marker="o")
                        # Place label below for lows, above for highs
                        is_low_label = any(k in str(lbl).lower() for k in ("shoulder", "head", "bottom"))
                        y_off = -14 if is_low_label else 12
                        ax.annotate(str(lbl), (cx, y), fontsize=8.5, color="#ffffff",
                                    fontweight="bold", xytext=(0, y_off),
                                    textcoords="offset points", ha="center", zorder=10,
                                    bbox=dict(boxstyle="round,pad=0.25", facecolor=p_color,
                                              edgecolor="none", alpha=0.92))
                # Clean neckline
                if len(trigger_line) >= 2:
                    (tx0, ty0), (tx1, ty1) = trigger_line[0], trigger_line[-1]
                    # Draw actual sloped neckline when the two points differ
                    y_left, y_right = ty0, ty1
                    if abs(ty0 - ty1) < 1e-9:
                        # Flat neckline – extend cleanly
                        ax.plot([_cx(tx0), chart_len - 1], [ty0, ty0],
                                color=p_color, linewidth=2.0, linestyle="--", alpha=0.9, zorder=6)
                    else:
                        slope = (ty1 - ty0) / max(tx1 - tx0, 1)
                        y_end = ty0 + slope * ((chart_len - 1 + offset) - tx0)
                        ax.plot([_cx(tx0), chart_len - 1], [ty0, y_end],
                                color=p_color, linewidth=2.0, linestyle="--", alpha=0.9, zorder=6)
                    ax.text(chart_len * 0.02, ty0, "Neckline", fontsize=8,
                            color=p_color, fontweight="bold", va="bottom", zorder=12)

        # Pattern title badge (clean, high-contrast)
        conf = sp.get("confidence", 0)
        title_txt = f"{name}  ·  {conf:.0f}%"
        ax.text(0.015, 0.97, title_txt, transform=ax.transAxes,
                fontsize=10, color="#ffffff", fontweight="bold", va="top", zorder=15,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=p_color, edgecolor="none", alpha=0.92))

    # --- Pick exactly ONE structure to draw as "the pattern" -------------
    # Priority set upstream in strategies.py (active_pattern):
    #   scanned (strong Inverse H&S / H&S etc. that won a conflict) >
    #   M/W (tightened double top/bottom) >
    #   wedge/triangle >
    #   plain channel.
    # Only one shape is drawn as the primary pattern to avoid label pile-up.
    active_pattern = family.get("active_pattern", "none")
    pattern_title = None
    pattern_conf = family.get("pattern_confidence")

    wedge = family.get("wedge")
    mw = family.get("mw_pattern")
    sp = family.get("scanned_pattern")

    if active_pattern == "scanned" and sp:
        pattern_title = sp.get("name", "Chart Pattern")
        pattern_conf = sp.get("confidence", pattern_conf)
        # The full key_points + trigger_line are already drawn earlier in the
        # scanned-pattern block, so here we only set the title.

    elif active_pattern == "mw" and mw and mw.get("neckline") is not None:
        pattern_title = mw.get("name", "M/W Pattern")
        mw_color = "#ff1744" if mw.get("pattern") == "M" else "#00c853"
        # Clean neckline
        neck_x0 = max(0, int(mw.get("neck_index", 0)) - offset)
        ax.plot([neck_x0, chart_len - 1], [mw["neckline"], mw["neckline"]],
                color=mw_color, linestyle="--", linewidth=2.1, alpha=0.9, zorder=5)
        ax.text(chart_len * 0.015, mw["neckline"], "Neckline", fontsize=8.5,
                color=mw_color, fontweight="bold", va="bottom", zorder=12)
        # Clear Top / Bottom markers
        left, right = mw.get("left"), mw.get("right")
        tag = "Top" if mw["pattern"] == "M" else "Bottom"
        close_x = (left and right and abs(int(left["index"]) - int(right["index"])) < chart_len * 0.05)
        for i, p in enumerate((left, right)):
            if p:
                px = int(p["index"]) - offset
                if 0 <= px < chart_len:
                    ax.scatter([px], [float(p["price"])], s=90, c=mw_color,
                               edgecolors="#ffffff", linewidths=1.8, zorder=9)
                    ax.annotate(tag, (px, float(p["price"])), fontsize=8.5, color="#ffffff",
                                fontweight="bold", xytext=(0, 12 if tag == "Top" else -14),
                                textcoords="offset points", ha="center", zorder=10,
                                bbox=dict(boxstyle="round,pad=0.25", facecolor=mw_color,
                                          edgecolor="none", alpha=0.92))

    elif active_pattern == "wedge" and wedge:
        pattern_title = wedge["pattern"]
        apex = wedge.get("apex_index")
        # Educational-style: green lower, red upper for rising/falling wedges
        for rail, color, tag in ((wedge["lower"], "#00c853", "Lower"), (wedge["upper"], "#ff1744", "Upper")):
            x0 = max(0, int(rail["x0"]) - offset)
            x1 = chart_len - 1
            if apex is not None:
                apex_local = apex - offset
                if 0 < apex_local < chart_len * 1.15:
                    x1 = min(chart_len - 1, max(0, apex_local))
            y0 = _line_at(rail, max(int(rail["x0"]), 0))
            y1 = _line_at(rail, x1 + offset)
            if x0 < chart_len:
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.3, alpha=0.95,
                        zorder=5, solid_capstyle="round")
            for ax_key, ay_key in (("x0", "y0"), ("x1", "y1")):
                px = int(rail[ax_key]) - offset
                _pivot_dot(px, float(rail[ay_key]), tag, color)

    elif active_pattern == "channel" and family.get("channel"):
        pattern_title = f"{family_kind.capitalize()} Channel" if family_kind != "none" else "Channel"
        ch = family["channel"]
        # Clean parallel channel rails (matching educational ascending/descending channel)
        ch_color = "#00c853" if family_kind == "ascending" else "#ff1744" if family_kind == "descending" else "#90a4ae"
        for rail, tag in ((ch.get("lower"), "Lower"), (ch.get("upper"), "Upper")):
            if not rail:
                continue
            x0 = max(0, int(rail["x0"]) - offset)
            y0 = _line_at(rail, max(int(rail["x0"]), 0))
            y1 = float(rail.get("y_end", rail.get("y1", 0)))
            if x0 < chart_len:
                ax.plot([x0, chart_len - 1], [y0, y1], color=ch_color, linewidth=2.1,
                        alpha=0.88, zorder=4, solid_capstyle="round")

    # --- Dual trendlines (MT5 hand-drawn style) -----------------------
    # ALWAYS draw ascending support + descending resistance when present.
    # Thick bright green so the lines are impossible to miss.
    def _draw_one_tl(tl, color="#00e676", width=2.8):
        if not tl:
            return
        try:
            x0 = max(0, int(tl["x0"]) - offset)
            # Use actual anchor y, then extend to current bar
            y0 = float(tl["y0"])
            y1 = float(tl.get("y_end", tl.get("y1", y0)))
            if x0 >= chart_len:
                return
            ax.plot([x0, chart_len - 1], [y0, y1], color=color, linestyle="-",
                    linewidth=width, alpha=1.0, zorder=10, solid_capstyle="round")
            # Anchor dots
            for ax_key, ay_key in (("x0", "y0"), ("x1", "y1")):
                px = int(tl[ax_key]) - offset
                if 0 <= px < chart_len:
                    ax.scatter([px], [float(tl[ay_key])], s=90, c=color,
                               edgecolors="#ffffff", linewidths=1.8, zorder=12, marker="o")
        except Exception:
            pass

    # Swing labels: HH/HL/LH/LL are shown only for the pivots that build the
    # current trendline structure. This keeps the chart readable while making
    # the line's anchors obvious.
    for ann in (family.get("trendline_annotations") or []):
        px = int(ann.get("index", -1)) - offset
        py = float(ann.get("price", 0))
        if not (0 <= px < chart_len):
            continue
        label = str(ann.get("label", ""))
        if label in ("H", "L"):
            continue
        is_high = ann.get("type") == "high"
        ax.scatter([px], [py], s=42, c="#ffffff", edgecolors="#00e676",
                   linewidths=1.1, zorder=12)
        ax.annotate(label, (px, py), fontsize=7.2, color="#ffffff",
                    fontweight="bold", xytext=(0, 10 if is_high else -13),
                    textcoords="offset points", ha="center", zorder=13)

    drawn = 0
    for tl in (family.get("uptrends") or []):
        _draw_one_tl(tl, color="#00e676", width=2.8)
        drawn += 1
    for tl in (family.get("downtrends") or []):
        _draw_one_tl(tl, color="#00e676", width=2.8)
        drawn += 1

    # Fallback: if uptrends/downtrends empty, try family_lines
    if drawn == 0:
        for tl in (family.get("family_lines") or []):
            _draw_one_tl(tl, color="#00e676", width=2.8)

    # Trendline lifecycle markers: BREAK / RETEST are derived from candle
    # closes and the line itself, never from prediction.
    tr = family.get("trendline_retest") or {}
    status = tr.get("status")
    if status in ("BREAK_CONFIRMED", "BREAK_DEVELOPING", "BREAK_RETEST_CONFIRMED", "FAKEOUT"):
        bi = tr.get("break_index")
        ri = tr.get("retest_index")
        level = tr.get("retest_level")
        if bi is not None:
            bx = int(bi) - offset
            if 0 <= bx < chart_len:
                by = float(chart_df["Close"].iloc[bx])
                ax.scatter([bx], [by], s=72,
                           c="#ffab00" if status != "FAKEOUT" else "#ff1744",
                           edgecolors="#ffffff", linewidths=1.4, zorder=15,
                           marker="D")
                ax.annotate("BREAK", (bx, by), fontsize=7.5,
                            color="#ffab00" if status != "FAKEOUT" else "#ff1744",
                            fontweight="bold", xytext=(6, 8),
                            textcoords="offset points", zorder=16)
        if ri is not None and level is not None:
            rx = int(ri) - offset
            if 0 <= rx < chart_len:
                ax.scatter([rx], [float(level)], s=78,
                           c="#00e676" if status == "BREAK_RETEST_CONFIRMED" else "#ff1744",
                           edgecolors="#ffffff", linewidths=1.5, zorder=16,
                           marker="o")
                ax.annotate("RETEST" if status == "BREAK_RETEST_CONFIRMED" else "RECLAIM",
                            (rx, float(level)), fontsize=7.5,
                            color="#00e676" if status == "BREAK_RETEST_CONFIRMED" else "#ff1744",
                            fontweight="bold", xytext=(6, -14),
                            textcoords="offset points", zorder=16)
        if level is not None and status != "INTACT":
            ax.axhline(float(level), color="#ffab00" if status != "FAKEOUT" else "#ff1744",
                       linestyle=":", linewidth=1.0, alpha=0.55, zorder=7)

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
    fig.suptitle(
        f"{line1}\n{line2}",
        color=COLORS["text"],
        fontsize=9.5,
        fontweight="bold",
        y=0.99,
    )


    img_buf = io.BytesIO()
    fig.savefig(img_buf, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf


def generate_trendline_educational_map(
    df: pd.DataFrame,
    symbol: str,
    setup: Dict[str, Any],
    title_suffix: str = "",
) -> io.BytesIO:
    """Clean educational Trendline chart.

    This intentionally draws only the information needed to understand the
    market story: candles, meaningful HH/HL/LH/LL points, one or two real
    trendlines, the breakout/retest lifecycle, and one relevant POI. It does
    not paint every detector output onto the candles.
    """
    family = setup.get("family") or setup.get("analysis") or setup
    chart_df, chart_len = _prepare_ohlc(df, max_bars=150)
    offset = len(df) - chart_len

    mc = mpf.make_marketcolors(
        up=COLORS["bull"], down=COLORS["bear"],
        edge="inherit", wick="inherit",
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":",
        gridcolor="#25303b",
        y_on_right=True,
        facecolor="#0b0f14",
        figcolor="#0b0f14",
        rc={"font.size": 9},
    )
    fig, axes = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        volume=False,
        figsize=(15, 8.5),
        returnfig=True,
        tight_layout=True,
        datetime_format="%d %b %H:%M",
        xrotation=0,
        warn_too_much_data=1000,
    )
    ax = axes[0]

    # Educational trendlines: maximum two meaningful rails.
    lines = []
    for tl in (family.get("uptrends") or []):
        if tl:
            lines.append((tl, "#22c55e", "RISING SUPPORT"))
    for tl in (family.get("downtrends") or []):
        if tl:
            lines.append((tl, "#ef4444", "FALLING RESISTANCE"))

    # Deduplicate by kind and keep only the cleanest line of each type.
    seen = set()
    clean_lines = []
    for tl, color, label in lines:
        kind = tl.get("kind")
        if kind in seen:
            continue
        seen.add(kind)
        clean_lines.append((tl, color, label))

    for tl, color, label in clean_lines[:2]:
        x0 = int(tl["x0"]) - offset
        x1 = int(tl["x1"]) - offset
        if x1 <= 0 or x0 >= chart_len:
            continue
        slope = (float(tl["y1"]) - float(tl["y0"])) / max(float(tl["x1"]) - float(tl["x0"]), 1.0)
        xa = max(0, x0)
        xb = chart_len - 1
        ya = float(tl["y0"]) + slope * ((xa + offset) - float(tl["x0"]))
        yb = float(tl["y0"]) + slope * ((xb + offset) - float(tl["x0"]))
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=2.8, alpha=0.95,
                solid_capstyle="round", zorder=6)

        # Only the defining anchors get circles — no marker explosion.
        for px_raw, py in ((tl["x0"], tl["y0"]), (tl["x1"], tl["y1"])):
            px = int(px_raw) - offset
            if 0 <= px < chart_len:
                ax.scatter([px], [float(py)], s=48, color=color, edgecolors="#ffffff",
                           linewidths=1.1, zorder=9)

        lx = max(2, min(chart_len - 18, int(x0 + (xb - x0) * 0.72)))
        ly = float(tl["y0"]) + slope * ((lx + offset) - float(tl["x0"]))
        ax.text(lx, ly, label, fontsize=7.5, color=color, fontweight="bold",
                va="bottom" if "SUPPORT" in label else "top", zorder=10)

    # Structure labels: only the most recent meaningful labels.
    anns = [a for a in (family.get("trendline_annotations") or [])
            if a.get("label") in ("HH", "HL", "LH", "LL")]
    anns = anns[-7:]
    for ann in anns:
        px = int(ann["index"]) - offset
        py = float(ann["price"])
        if not (0 <= px < chart_len):
            continue
        label = str(ann["label"])
        is_high = ann.get("type") == "high"
        ax.scatter([px], [py], s=28, color="#f8fafc", edgecolors="#94a3b8",
                   linewidths=0.8, zorder=10)
        ax.annotate(
            label, (px, py), fontsize=8, color="#f8fafc", fontweight="bold",
            xytext=(0, 10 if is_high else -14), textcoords="offset points",
            ha="center", zorder=11,
        )

    # Break/retest lifecycle — this is the educational core.
    tr = family.get("trendline_retest") or {}
    status = str(tr.get("status") or "INTACT")
    bi = tr.get("break_index")
    ri = tr.get("retest_index")
    level = tr.get("retest_level")
    if bi is not None:
        bx = int(bi) - offset
        if 0 <= bx < chart_len:
            by = float(chart_df["Close"].iloc[bx])
            ax.scatter([bx], [by], s=95, color="#f59e0b", edgecolors="#ffffff",
                       linewidths=1.3, marker="D", zorder=14)
            ax.annotate("BREAK", (bx, by), xytext=(8, 12), textcoords="offset points",
                        color="#f59e0b", fontsize=8, fontweight="bold", zorder=15)
    if ri is not None and level is not None:
        rx = int(ri) - offset
        if 0 <= rx < chart_len:
            retest_ok = status == "BREAK_RETEST_CONFIRMED"
            rcolor = "#22c55e" if retest_ok else "#ef4444"
            ax.scatter([rx], [float(level)], s=90, color=rcolor, edgecolors="#ffffff",
                       linewidths=1.3, zorder=14)
            ax.annotate("RETEST" if retest_ok else "RECLAIM", (rx, float(level)),
                        xytext=(8, -18), textcoords="offset points", color=rcolor,
                        fontsize=8, fontweight="bold", zorder=15)

    # Pattern geometry: draw only the selected pattern's compact structure.
    # Long rails describe trend direction; these short lines/points describe
    # the local pattern without covering the candle field.
    sp = family.get("scanned_pattern") or {}
    trigger_line = sp.get("trigger_line") or []
    if len(trigger_line) >= 2:
        pts = [(float(p[0]) - offset, float(p[1])) for p in trigger_line]
        pts = [(x, y) for x, y in pts if 0 <= x < chart_len]
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, linestyle="--", linewidth=1.8,
                    color="#f59e0b", alpha=0.9, zorder=7)
    seen_pattern_points = set()
    pattern_label_counts = {}
    for kp in (sp.get("key_points") or []):
        try:
            raw_x, py, label = kp[0], kp[1], kp[2]
            key = (int(raw_x), str(label).strip().upper())
            if key in seen_pattern_points:
                continue
            seen_pattern_points.add(key)
            px = int(raw_x) - offset
            if 0 <= px < chart_len:
                ax.scatter([px], [float(py)], s=34, color="#f59e0b",
                           edgecolors="#ffffff", linewidths=0.8, zorder=10)
                label_text = str(label)
                upper_label = label_text.upper()
                # Stagger repeated pattern labels (Top 1/2/3, Bottom 1/2/3,
                # etc.) so the educational map remains readable when points
                # are clustered tightly together.
                base_key = ("TOP" if "TOP" in upper_label else
                            "BOTTOM" if "BOTTOM" in upper_label else
                            "HEAD" if "HEAD" in upper_label else "OTHER")
                n_seen = pattern_label_counts.get(base_key, 0)
                pattern_label_counts[base_key] = n_seen + 1
                yoff = 10 + min(n_seen, 3) * 11
                if "BOTTOM" in upper_label:
                    yoff = -(10 + min(n_seen, 3) * 11)
                ax.annotate(label_text, (px, float(py)), xytext=(0, yoff),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color="#fef3c7", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.12",facecolor="#0b0f14",
                                      edgecolor="#f59e0b",alpha=0.78,linewidth=0.6),
                            zorder=11)
        except (TypeError, ValueError, IndexError):
            continue
    if sp.get("trigger_price") is not None:
        try:
            trigger = float(sp["trigger_price"])
            ax.axhline(trigger, linestyle=":", linewidth=1.2,
                       color="#f59e0b", alpha=0.65, zorder=5)
            ax.text(chart_len - 2, trigger, f" {sp.get('name', 'Pattern')} trigger",
                    fontsize=6.8, color="#fbbf24", va="bottom", ha="right", zorder=9)
        except (TypeError, ValueError):
            pass

    # Keep the candle field clean: no position box, no side panel, no
    # session labels, no decorative legend. The decision state belongs in
    # the Telegram text report; the chart is the structural visual.

    # Minimal title: bias + structural state. Everything else is deliberately
    # left off the candle field so the eye can read the price action.
    pattern_name = None
    sp = family.get("scanned_pattern") or {}
    if sp.get("name"):
        pattern_name = sp.get("name")
    elif family.get("active_pattern") and family.get("active_pattern") != "none":
        pattern_name = str(family.get("active_pattern")).replace("_", " ").title()
    direction = str(
        setup.get("direction")
        or family.get("direction")
        or family.get("bias")
        or family.get("short_term_direction")
        or "NEUTRAL"
    ).upper()
    title = f"{symbol}  |  M30  |  {direction} BIAS  |  {status.replace('_', ' ')}"
    if pattern_name:
        title += f"  |  {pattern_name}"
    ax.set_title(title, color="#f8fafc", fontsize=12.5, fontweight="bold", pad=10)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf


# ============================================================
# OTE CHART -- 30M Fibonacci Fan + Expansion chart
# ============================================================
# OTE CHART -- clean structural OTE map
def generate_ote_map(df: pd.DataFrame, symbol: str, analysis: Dict[str, Any], title_suffix: str = "") -> io.BytesIO:
    """Educational OTE chart: candles + impulse anchors + 62-79% zone.
    No fan rays, expansion clutter, or unconfirmed trade box.
    """
    chart_df, chart_len = _prepare_ohlc(df, max_bars=160)
    offset = len(df) - chart_len
    mc = mpf.make_marketcolors(up=COLORS["bull"], down=COLORS["bear"], edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridcolor=COLORS["grid"], y_on_right=True,
        facecolor=COLORS["bg"], figcolor=COLORS["bg"],
        rc={"axes.labelcolor":COLORS["text"],"xtick.color":COLORS["text"],"ytick.color":COLORS["text"]})
    fig, axlist = mpf.plot(chart_df, type="candle", style=style, volume=False, returnfig=True,
        figsize=(12,6.8), warn_too_much_data=10000)
    ax=axlist[0]
    imp=analysis.get("impulse") or {}; zone=analysis.get("zone") or {}
    start=imp.get("start"); end=imp.get("end")
    # Anchor labels are deliberately high-contrast.  The previous default
    # Matplotlib text color rendered these labels almost black on the dark
    # educational chart, making the two most important OTE anchors hard to see.
    if start:
        x=start["index"]-offset
        if 0<=x<chart_len:
            start_color = "#38bdf8"  # structural origin / cool anchor
            ax.scatter([x],[start["price"]],s=58,zorder=10,marker="o",
                       color=start_color,edgecolors="#ffffff",linewidths=1.0)
            ax.annotate("SWING ORIGIN", (x,start["price"]),
                        xytext=(7, 10 if start.get("type") == "low" else -22),
                        textcoords="offset points",fontsize=7.5,fontweight="bold",
                        color="#e0f2fe",va="bottom" if start.get("type") == "low" else "top",
                        bbox=dict(boxstyle="round,pad=0.22",facecolor="#0b0f14",
                                  edgecolor=start_color,alpha=0.88,linewidth=0.8),
                        zorder=12)
    if end:
        x=end["index"]-offset
        if 0<=x<chart_len:
            end_color = "#f59e0b"  # impulse / expansion anchor
            ax.scatter([x],[end["price"]],s=58,zorder=10,marker="o",
                       color=end_color,edgecolors="#ffffff",linewidths=1.0)
            ax.annotate("IMPULSE EXTREME", (x,end["price"]),
                        xytext=(7, -22 if end.get("type") == "high" else 10),
                        textcoords="offset points",fontsize=7.5,fontweight="bold",
                        color="#fef3c7",va="top" if end.get("type") == "high" else "bottom",
                        bbox=dict(boxstyle="round,pad=0.22",facecolor="#0b0f14",
                                  edgecolor=end_color,alpha=0.88,linewidth=0.8),
                        zorder=12)
    if zone:
        lo,hi=zone.get("low"),zone.get("high")
        if lo is not None and hi is not None:
            ax.axhspan(lo,hi,alpha=0.16,zorder=1)
            ax.axhline(zone.get("62",lo),linestyle="--",linewidth=1,alpha=.65)
            ax.axhline(zone.get("70.5",(lo+hi)/2),linestyle="-",linewidth=1.2,alpha=.8)
            ax.axhline(zone.get("79",hi),linestyle="--",linewidth=1,alpha=.65)
            ax.text(chart_len-2,zone.get("70.5")," OTE 70.5%",fontsize=7,va="center",fontweight="bold")
    direction=analysis.get("direction",""); state=analysis.get("zone_state",analysis.get("status","WAIT"))
    ax.set_title(f"{symbol}  OTE 62–79% | {direction} | {state}" + (f" | {title_suffix}" if title_suffix else ""), color=COLORS["text"],fontsize=10,fontweight="bold",pad=10)
    prices=list(chart_df["High"])+list(chart_df["Low"])
    if zone:
        prices += [zone.get("low",0),zone.get("high",0),zone.get("70.5",0)]
    if prices:
        pmin,pmax=min(prices),max(prices); pad=(pmax-pmin)*.05 if pmax>pmin else 1
        ax.set_ylim(pmin-pad,pmax+pad)
    img_buf=io.BytesIO(); fig.savefig(img_buf,dpi=180,bbox_inches="tight",facecolor=fig.get_facecolor()); img_buf.seek(0); plt.close(fig); return img_buf
