"""
strategies.py
=============
The two chart-analysis strategies this bot runs: Trendline and OTE.

Both call topdown_engine.get_topdown_bias() first to establish a 4H/1H
directional read, then do their own timeframe-specific work on the 30M
chart (the geometry/entry engine), and finally gate/score that 30M read
against the top-down bias so Trendline and OTE never disagree with the
bigger picture without saying so.

  - Trendline: classic educational rules (wicks, 2-3+ touches, retest,
    candlestick confirmation, TradingView-style position template).
  - OTE: Fibonacci Fan + Expansion off the most recent clean impulse leg,
    entry on 30M, gated by the same 4H -> 1H top-down read.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import market_data
from market_analysis import zigzag_swings, find_swings, compute_volume_profile
from topdown_engine import get_topdown_bias, format_topdown_summary




# ============================================================
# TRENDLINE STRATEGY
# Simple human-style drawing (matches MT5 hand-drawn lines)
#
# 1. Find clean swing pivots (line-chart style)
# 2. Connect the pivots that define the latest impulse
# 3. Count touches for strength
# 4. Direction = price above or below the line
# 5. Entry only on retest + confirmation + decent R:R
# ============================================================


def _line_value(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _count_touches(df: pd.DataFrame, x0: int, y0: float, x1: int, y1: float,
                   kind: str, tol_atr: float = 0.40) -> int:
    """How many times price touched the line (wicks)."""
    if df is None or len(df) < 5:
        return 0
    atr = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).values
    highs = df["High"].values
    lows = df["Low"].values
    touches = 0
    lo, hi = min(x0, x1), max(x0, x1)
    for i in range(max(0, lo), min(hi + 1, len(df))):
        lv = _line_value(x0, y0, x1, y1, i)
        a = float(atr[i]) if i < len(atr) and atr[i] > 0 else abs(y1 - y0) * 0.05
        tol = max(a * tol_atr, 1e-9)
        if kind == "support" and abs(lows[i] - lv) <= tol:
            touches += 1
        elif kind == "resistance" and abs(highs[i] - lv) <= tol:
            touches += 1
    return touches


def _get_clean_pivots(df: pd.DataFrame) -> List[Dict]:
    """
    Clean swing pivots the way a trader sees them on a line chart.
    Prefer significant swings only.
    """
    pivots = zigzag_swings(df, depth=4, deviation_atr=0.30)
    if len(pivots) < 4:
        pivots = zigzag_swings(df, depth=3, deviation_atr=0.22)
    if len(pivots) < 3:
        pivots = zigzag_swings(df, depth=3, deviation_atr=0.18)
    return pivots


def _draw_simple_trendline(pivots: List[Dict], kind: str, n: int, df: pd.DataFrame) -> Optional[Dict]:
    """
    Human-style trendline:
      - Uptrend   = last clear higher lows (support)
      - Downtrend = last clear lower highs (resistance)
    Connects the two most recent valid pivots that form the impulse structure.
    """
    pts = [p for p in pivots if p["type"] == ("low" if kind == "support" else "high")]
    if len(pts) < 2:
        return None

    # Walk from most recent backwards to find a clean pair
    # Prefer the most recent pair that still forms a proper higher-low / lower-high
    best = None
    for j in range(len(pts) - 1, 0, -1):
        for i in range(j - 1, -1, -1):
            a, b = pts[i], pts[j]
            if b["index"] <= a["index"]:
                continue
            # Structural requirement
            if kind == "support" and b["price"] <= a["price"]:
                continue  # must be higher low
            if kind == "resistance" and b["price"] >= a["price"]:
                continue  # must be lower high

            # Prefer pairs that span a meaningful distance
            span = b["index"] - a["index"]
            if span < 5:
                continue

            slope = (b["price"] - a["price"]) / max(span, 1)
            touches = _count_touches(df, a["index"], a["price"], b["index"], b["price"], kind)
            if touches < 2:
                continue

            y_end = _line_value(a["index"], a["price"], b["index"], b["price"], n - 1)
            quality = "unconfirmed" if touches < 3 else ("confirmed" if touches <= 4 else "crowded")

            candidate = {
                "x0": a["index"], "y0": a["price"],
                "x1": b["index"], "y1": b["price"],
                "y_end": y_end, "slope": slope,
                "touches": touches, "confirmed": touches >= 3,
                "quality": quality, "kind": kind,
                "span": span,
            }

            # Prefer more recent + more touches + reasonable span
            # Most recent valid pair wins (human draws the latest structure)
            if best is None:
                best = candidate
            else:
                # Prefer the one whose second pivot is more recent
                if b["index"] > best["x1"]:
                    best = candidate
                elif b["index"] == best["x1"] and touches > best["touches"]:
                    best = candidate
            break  # take the nearest valid partner for this j, then move on
        if best and best["x1"] == pts[j]["index"]:
            break  # already found a good recent pair

    return best


def build_trendline_family(df: pd.DataFrame, max_lines: int = 4, lookback_bars: int = 120) -> Dict[str, Any]:
    """
    Simple trendline engine that draws like a human on MT5.

    - Find clean pivots
    - Draw one primary uptrend line (higher lows) and/or downtrend line (lower highs)
    - Pick the one price is currently interacting with
    - Direction from price vs line (above = hold uptrend, below = break, etc.)
    """
    if df is None or len(df) < 30:
        return {"error": "Insufficient data for trendline", "direction": "NEUTRAL", "pivots": []}

    n = len(df)
    close = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else abs(float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]))
    if atr <= 0:
        atr = abs(float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1])) or close * 0.002

    pivots = _get_clean_pivots(df)

    # Only use relatively recent pivots for the active line
    cutoff = max(0, n - lookback_bars)
    recent = [p for p in pivots if p["index"] >= cutoff]
    if len(recent) < 3:
        recent = pivots

    support = _draw_simple_trendline(recent, "support", n, df)      # rising line
    resistance = _draw_simple_trendline(recent, "resistance", n, df)  # falling line

    # Choose the line price is closest to / interacting with
    primary = None
    family_kind = None

    def _dist(line):
        return abs(close - line["y_end"]) if line else 1e18

    if support and resistance:
        # Prefer the line that price is nearer to (active structure)
        if _dist(support) <= _dist(resistance) * 1.15:
            primary, family_kind = support, "ascending"
        else:
            primary, family_kind = resistance, "descending"
    elif support:
        primary, family_kind = support, "ascending"
    elif resistance:
        primary, family_kind = resistance, "descending"

    # Optional simple parallel channel (only if clean)
    channel = None
    if primary:
        opposite_type = "high" if primary["kind"] == "support" else "low"
        opp = [p for p in recent if p["type"] == opposite_type]
        if opp:
            # Use the opposite pivot furthest from the primary line
            best_opp = None
            best_d = 0
            for p in opp:
                lv = _line_value(primary["x0"], primary["y0"], primary["x1"], primary["y1"], p["index"])
                d = abs(p["price"] - lv)
                if d > best_d:
                    best_d = d
                    best_opp = p
            if best_opp and best_d > atr * 0.8:
                slope = primary["slope"]
                y_end = best_opp["price"] + slope * (n - 1 - best_opp["index"])
                rail = {
                    "x0": best_opp["index"], "y0": best_opp["price"],
                    "x1": n - 1, "y1": y_end, "y_end": y_end,
                    "slope": slope, "kind": "resistance" if primary["kind"] == "support" else "support",
                }
                if primary["kind"] == "support":
                    channel = {"lower": primary, "upper": rail, "width": abs(rail["y_end"] - primary["y_end"])}
                else:
                    channel = {"upper": primary, "lower": rail, "width": abs(primary["y_end"] - rail["y_end"])}

    # ---------- Direction (price vs line) ----------
    direction = "NEUTRAL"
    strength = 40
    reasons = []
    breakout_grade = None

    if primary:
        line_val = float(primary["y_end"])
        dist_atr = (close - line_val) / atr

        if primary["kind"] == "support":  # rising trendline
            reasons.append(f"Rising trendline · {primary['touches']} touches · {primary['quality']}")
            if close >= line_val - atr * 0.12:
                direction = "BUY"
                strength = 55 + min(28, (primary["touches"] - 2) * 8)
                if abs(dist_atr) <= 1.0:
                    reasons.append("Price holding ABOVE rising trendline")
                    strength = min(100, strength + 10)
                else:
                    reasons.append("Price ABOVE rising trendline — trend intact")
            else:
                direction = "SELL"
                strength = 52 + min(20, int(abs(dist_atr) * 6))
                breakout_grade = {
                    "side": "support_break_down",
                    "strength": "confirmed" if close < line_val - atr * 0.35 else "developing",
                    "penetration_atr": round(abs(dist_atr), 2),
                }
                reasons.append(f"Price BELOW rising trendline — break ({breakout_grade['penetration_atr']} ATR)")

        else:  # falling trendline
            reasons.append(f"Falling trendline · {primary['touches']} touches · {primary['quality']}")
            if close <= line_val + atr * 0.12:
                direction = "SELL"
                strength = 55 + min(28, (primary["touches"] - 2) * 8)
                if abs(dist_atr) <= 1.0:
                    reasons.append("Price holding BELOW falling trendline")
                    strength = min(100, strength + 10)
                else:
                    reasons.append("Price BELOW falling trendline — trend intact")
            else:
                direction = "BUY"
                strength = 52 + min(20, int(abs(dist_atr) * 6))
                breakout_grade = {
                    "side": "resistance_break_up",
                    "strength": "confirmed" if close > line_val + atr * 0.35 else "developing",
                    "penetration_atr": round(abs(dist_atr), 2),
                }
                reasons.append(f"Price ABOVE falling trendline — break ({breakout_grade['penetration_atr']} ATR)")

    # Chart series
    upper_line = np.full(n, np.nan)
    lower_line = np.full(n, np.nan)
    mid_line = np.full(n, np.nan)

    if channel:
        u, lo = channel["upper"], channel["lower"]
        for i in range(n):
            upper_line[i] = _line_value(u["x0"], u["y0"], u["x1"], u["y1"], i)
            lower_line[i] = _line_value(lo["x0"], lo["y0"], lo["x1"], lo["y1"], i)
            mid_line[i] = (upper_line[i] + lower_line[i]) / 2.0
    elif primary:
        for i in range(n):
            val = _line_value(primary["x0"], primary["y0"], primary["x1"], primary["y1"], i)
            if primary["kind"] == "support":
                lower_line[i] = val
            else:
                upper_line[i] = val

    candle = _last_candle_pattern(df)
    rsi_val = _rsi(df["Close"])
    reasons.append(f"RSI(14): {rsi_val:.1f}")
    if candle.get("name"):
        reasons.append(f"Candle: {candle['name']}")

    return {
        "direction": direction,
        "strength": max(0, min(100, int(strength))),
        "reasons": reasons,
        "family_kind": family_kind,
        "family_lines": [primary] if primary else [],
        "uptrends": [primary] if family_kind == "ascending" and primary else [],
        "downtrends": [primary] if family_kind == "descending" and primary else [],
        "channel": channel,
        "wedge": None,
        "horizontal_levels": [],
        "projections": [],
        "mw_pattern": None,
        "pivots": pivots[-16:],
        "volume_profile": {},
        "upper_line": upper_line,
        "lower_line": lower_line,
        "middle_line": mid_line,
        "df": df,
        "mode": "channel" if channel else "lines",
        "breakout_grade": breakout_grade,
        "primary_quality": primary.get("quality") if primary else None,
        "primary_touches": primary.get("touches") if primary else 0,
        "candle": candle,
        "rsi": rsi_val,
        "primary": primary,
    }



def build_position_container(family: Dict[str, Any], atr_mult_sl: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    TradingView-style Long / Short position template
    following the educational image rules STRICTLY:

      Entry  : only when price is near the trendline (retest) or after confirmed break
      SL     : beyond recent swing + beyond the trendline, minimum 0.6 ATR risk
      TP1    : at least 1.5–2× risk (structure or measured)
      TP2/TP3: further structure / 1:3 RR
      Reject  : any setup with R:R < 1.5 (never force a bad trade)
    """
    if not family or family.get("error"):
        return None
    df = family.get("df")
    if df is None or df.empty:
        return None

    close = float(df["Close"].iloc[-1])
    high = float(df["High"].iloc[-1])
    low = float(df["Low"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else abs(high - low)
    if atr <= 0:
        atr = abs(high - low) or close * 0.002

    direction = family.get("direction", "NEUTRAL")
    if direction not in ("BUY", "SELL"):
        return None

    primary = family.get("primary")
    pivots = family.get("pivots") or []
    channel = family.get("channel")
    candle = family.get("candle") or {}
    rsi_val = family.get("rsi", 50.0)
    brk = family.get("breakout_grade")

    min_risk = atr * 0.60          # never risk less than 0.6 ATR
    min_rr = 1.50                  # educational rule: never worse than ~1:1.5

    # ---------- BUY (Long) ----------
    if direction == "BUY":
        # Ideal entry = support trendline / lower channel rail
        line_price = None
        if channel and channel.get("lower"):
            line_price = float(channel["lower"].get("y_end", close))
        elif primary and primary.get("kind") == "support":
            line_price = float(primary.get("y_end", close))

        # Only treat as valid retest if price is close to the line
        # (within ~1.2 ATR). Otherwise wait — do not chase.
        if line_price is not None and abs(close - line_price) <= atr * 1.20:
            entry = min(close, line_price + atr * 0.15)
        elif brk and brk.get("side") == "resistance_break_up":
            # Breakout entry — use current close
            entry = close
        else:
            # Price too far from the trendline → no forced entry
            return None

        # SL: recent swing low below entry, or below the trendline
        swing_lows = sorted(
            [float(p["price"]) for p in pivots if p.get("type") == "low" and p["price"] < entry]
        )
        if swing_lows:
            # Use the most recent meaningful low (last 1–3)
            sl = swing_lows[-1] if len(swing_lows) == 1 else swing_lows[-min(3, len(swing_lows))]
            # Prefer the lowest of the recent ones for safety
            sl = min(swing_lows[-min(3, len(swing_lows)):])
        else:
            sl = entry - atr * 1.0

        if line_price is not None:
            sl = min(sl, line_price - atr * 0.35)

        # Enforce minimum risk distance
        if (entry - sl) < min_risk:
            sl = entry - min_risk

        risk = entry - sl
        if risk <= 0:
            return None

        # TPs: structure highs first, then measured RR
        swing_highs = sorted(
            [float(p["price"]) for p in pivots if p.get("type") == "high" and p["price"] > entry + risk * 0.8]
        )
        tp1 = swing_highs[0] if swing_highs else entry + risk * 2.0
        # Ensure TP1 meets minimum RR
        if (tp1 - entry) < risk * min_rr:
            tp1 = entry + risk * 2.0

        tp2 = swing_highs[1] if len(swing_highs) > 1 else entry + risk * 2.8
        if (tp2 - entry) <= (tp1 - entry):
            tp2 = entry + risk * 2.8

        tp3 = entry + risk * 3.0

        conf_ok = False
        conf_notes = []
        if candle.get("bullish"):
            conf_ok = True
            conf_notes.append(candle.get("name", "Bullish candle"))
        if rsi_val >= 48:
            conf_notes.append(f"RSI {rsi_val:.1f}")
            conf_ok = conf_ok or rsi_val > 52
        if brk and brk.get("side") == "resistance_break_up":
            conf_notes.append("Bullish trendline break")
            conf_ok = True

        rr = (tp1 - entry) / risk
        if rr < min_rr:
            return None  # never force a bad R:R trade

        return {
            "side": "BUY",
            "entry": round(entry, 5),
            "sl": round(sl, 5),
            "tp1": round(tp1, 5),
            "tp2": round(tp2, 5),
            "tp3": round(tp3, 5),
            "risk": round(risk, 5),
            "reward": round(tp1 - entry, 5),
            "rr": round(rr, 2),
            "confirmation": conf_ok,
            "conf_notes": conf_notes,
            "entry_note": (
                "Buy on trendline retest after bullish confirmation"
                if conf_ok else
                "Potential long — waiting for bullish confirmation"
            ),
        }

    # ---------- SELL (Short) ----------
    else:
        line_price = None
        if channel and channel.get("upper"):
            line_price = float(channel["upper"].get("y_end", close))
        elif primary and primary.get("kind") == "resistance":
            line_price = float(primary.get("y_end", close))

        if line_price is not None and abs(close - line_price) <= atr * 1.20:
            entry = max(close, line_price - atr * 0.15)
        elif brk and brk.get("side") == "support_break_down":
            entry = close
        else:
            return None

        swing_highs = sorted(
            [float(p["price"]) for p in pivots if p.get("type") == "high" and p["price"] > entry]
        )
        if swing_highs:
            sl = max(swing_highs[-min(3, len(swing_highs)):])
        else:
            sl = entry + atr * 1.0

        if line_price is not None:
            sl = max(sl, line_price + atr * 0.35)

        if (sl - entry) < min_risk:
            sl = entry + min_risk

        risk = sl - entry
        if risk <= 0:
            return None

        swing_lows = sorted(
            [float(p["price"]) for p in pivots if p.get("type") == "low" and p["price"] < entry - risk * 0.8],
            reverse=True,
        )
        tp1 = swing_lows[0] if swing_lows else entry - risk * 2.0
        if (entry - tp1) < risk * min_rr:
            tp1 = entry - risk * 2.0

        tp2 = swing_lows[1] if len(swing_lows) > 1 else entry - risk * 2.8
        if (entry - tp2) <= (entry - tp1):
            tp2 = entry - risk * 2.8

        tp3 = entry - risk * 3.0

        conf_ok = False
        conf_notes = []
        if candle.get("bearish"):
            conf_ok = True
            conf_notes.append(candle.get("name", "Bearish candle"))
        if rsi_val <= 52:
            conf_notes.append(f"RSI {rsi_val:.1f}")
            conf_ok = conf_ok or rsi_val < 48
        if brk and brk.get("side") == "support_break_down":
            conf_notes.append("Bearish trendline break")
            conf_ok = True

        rr = (entry - tp1) / risk
        if rr < min_rr:
            return None

        return {
            "side": "SELL",
            "entry": round(entry, 5),
            "sl": round(sl, 5),
            "tp1": round(tp1, 5),
            "tp2": round(tp2, 5),
            "tp3": round(tp3, 5),
            "risk": round(risk, 5),
            "reward": round(entry - tp1, 5),
            "rr": round(rr, 2),
            "confirmation": conf_ok,
            "conf_notes": conf_notes,
            "entry_note": (
                "Sell on trendline retest after bearish confirmation"
                if conf_ok else
                "Potential short — waiting for bearish confirmation"
            ),
        }



def format_trendline_report(family: Dict[str, Any], symbol: str) -> str:
    """Human-readable report matching the educational image language."""
    if family.get("error"):
        return f"📐 TRENDLINE | {symbol}\n{family['error']}"

    lines = [
        f"📐 TRENDLINE STRATEGY  |  {symbol}  (4H → 1H → 30M)",
        "─" * 42,
    ]

    direction = family.get("direction", "NEUTRAL")
    strength = family.get("strength", 0)
    lines.append(f"Direction: {direction}   Strength: {strength}/100")

    quality = family.get("primary_quality")
    touches = family.get("primary_touches", 0)
    if quality:
        lines.append(f"Trendline: {quality.upper()} · {touches} touches (wicks)")

    if family.get("channel"):
        w = family["channel"].get("width")
        if w:
            lines.append(f"Channel width: {w:.5f}")

    for r in (family.get("reasons") or [])[:6]:
        lines.append(f"• {r}")

    # Top-down gating notes
    for g in (family.get("gating_notes") or []):
        lines.append(g)

    # Position template — only shown when R:R and retest rules are satisfied
    pos = family.get("position")
    if pos is None:
        pos = build_position_container(family)
        if pos:
            family["position"] = pos
    if pos:
        side = pos["side"]
        lines.append("─" * 42)
        lines.append(f"{'🟢 LONG' if side == 'BUY' else '🔴 SHORT'} POSITION (TradingView style)")
        lines.append(f"Entry : {pos['entry']}")
        lines.append(f"SL    : {pos['sl']}")
        lines.append(f"TP1   : {pos['tp1']}   (structure)")
        lines.append(f"TP2   : {pos['tp2']}")
        lines.append(f"TP3   : {pos['tp3']}   (RR 1:3)")
        lines.append(f"R:R   : 1:{pos['rr']:.2f}")
        if pos.get("conf_notes"):
            lines.append("Confirmation: " + ", ".join(pos["conf_notes"]))
        if pos.get("entry_note"):
            lines.append(f"→ {pos['entry_note']}")
        lines.append("Risk rule: 1-2% of capital · Move SL to BE after TP1")
    else:
        lines.append("─" * 42)
        lines.append("⏳ No valid entry yet")
        lines.append("Reason: price not retesting the trendline, or R:R would be < 1.5")
        lines.append("Action: wait for a clean retest + confirmation (educational rule)")

    return "\n".join(lines)

# ============================================================
# OTE STRATEGY -- Fibonacci Fan + Fibonacci Expansion
# (Aligned with the educational OTE image)
#
#   1. Get 4H → 1H top-down bias
#   2. Detect the most recent clear impulse swing on the 30M chart
#   3. Draw Fibonacci Fan (38.2 / 50 / 61.8) from the impulse origin
#   4. PRIME entry zone = 50% – 61.8% (deeper Fan levels)
#   5. Require candlestick confirmation (Engulfing / Pin / Hammer / Shooting Star)
#   6. Project Fibonacci Expansion targets (127.2 / 161.8 / 200 / 261.8)
#   7. TradingView-style position template (Entry / SL / TP1 / TP2 / TP3)
#   8. Gate/score against the 4H/1H top-down bias
#
# Always runs and displays on the 30M timeframe.
# ============================================================

FAN_RATIOS = [0.382, 0.50, 0.618]
EXPANSION_RATIOS = [1.272, 1.618, 2.0, 2.618]


def _ensure_atr(df: pd.DataFrame) -> pd.DataFrame:
    if "ATR" not in df.columns or df["ATR"].isna().all():
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df = df.copy()
        df["ATR"] = tr.rolling(14, min_periods=1).mean()
    return df


def _find_impulse(df: pd.DataFrame, lookback: int = 120) -> Optional[Dict[str, Any]]:
    """
    Find the most recent clean impulse leg suitable for Fan + Expansion.

    Returns:
      {
        "direction": "BUY" | "SELL",
        "start": {index, price, type},
        "end":   {index, price, type},
        "retracement": {index, price, type} | None,   # point C if available
        "leg_size": float,
      }
    """
    if df is None or len(df) < 40:
        return None

    df = _ensure_atr(df)
    n = len(df)
    swings = zigzag_swings(df, depth=5, deviation_atr=0.40)
    if len(swings) < 2:
        swings = find_swings(df, left=3, right=3)
    if len(swings) < 2:
        return None

    # Restrict to recent window
    swings = [s for s in swings if s["index"] >= max(0, n - lookback)]
    if len(swings) < 2:
        return None

    # Walk backwards looking for a strong directional leg
    for i in range(len(swings) - 1, 0, -1):
        a = swings[i - 1]
        b = swings[i]
        if a["type"] == b["type"]:
            continue

        leg = abs(b["price"] - a["price"])
        atr = float(df["ATR"].iloc[min(b["index"], n - 1)])
        if atr <= 0:
            atr = leg * 0.1
        if leg < 1.2 * atr:          # require meaningful impulse
            continue

        # Bullish impulse: low -> high
        if a["type"] == "low" and b["type"] == "high" and b["price"] > a["price"]:
            retrace = None
            for s in swings[i + 1:]:
                if s["type"] == "low" and s["price"] < b["price"]:
                    retrace = s
                    break
            return {"direction": "BUY", "start": a, "end": b, "retracement": retrace, "leg_size": leg}

        # Bearish impulse: high -> low
        if a["type"] == "high" and b["type"] == "low" and b["price"] < a["price"]:
            retrace = None
            for s in swings[i + 1:]:
                if s["type"] == "high" and s["price"] > b["price"]:
                    retrace = s
                    break
            return {"direction": "SELL", "start": a, "end": b, "retracement": retrace, "leg_size": leg}

    return None


def _build_fan(impulse: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """Build Fibonacci Fan rays from impulse start -> end, extendable to any bar index."""
    x0 = impulse["start"]["index"]
    y0 = impulse["start"]["price"]
    x1 = impulse["end"]["index"]
    y1 = impulse["end"]["price"]
    dy = y1 - y0

    fans = []
    for r in FAN_RATIOS:
        y_div = y0 + dy * r
        slope = (y_div - y0) / max(x1 - x0, 1)
        y_end = y0 + slope * (n - 1 - x0)
        fans.append({
            "ratio": r, "label": f"{r*100:.1f}%",
            "x0": x0, "y0": y0, "x1": x1, "y1": y_div,
            "slope": slope, "y_at_end": y_end,
        })
    return fans


def _fan_price_at(fan: Dict, x: float) -> float:
    return fan["y0"] + fan["slope"] * (x - fan["x0"])


def _build_expansion(impulse: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fibonacci Expansion (3-point when a retracement point exists, else simple extension)."""
    start = impulse["start"]["price"]
    end = impulse["end"]["price"]
    leg = impulse["leg_size"]
    direction = impulse["direction"]
    retrace = impulse.get("retracement")

    expansions = []
    if retrace is not None:
        c = retrace["price"]
        for r in EXPANSION_RATIOS:
            price = c + leg * r if direction == "BUY" else c - leg * r
            expansions.append({"ratio": r, "label": f"{r*100:.1f}%", "price": float(price), "from_point": "C"})
    else:
        for r in EXPANSION_RATIOS:
            price = end + leg * (r - 1.0) if direction == "BUY" else end - leg * (r - 1.0)
            expansions.append({"ratio": r, "label": f"{r*100:.1f}%", "price": float(price), "from_point": "B"})
    return expansions


def _evaluate_entry(
    df: pd.DataFrame,
    impulse: Dict[str, Any],
    fans: List[Dict],
    expansions: List[Dict],
    topdown: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Educational-image OTE entry logic:
      - Prime zone = 50% – 61.8% Fibonacci Fan
      - Require price interaction with the zone
      - Candlestick confirmation (Engulfing / Pin / Hammer / Shooting Star)
      - RSI filter
      - TradingView-style position (Entry / SL / TP1 / TP2 / TP3)
      - Gate against 4H→1H top-down bias
    """
    n = len(df)
    close = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else abs(impulse["leg_size"]) * 0.1
    direction = impulse["direction"]

    # Current fan prices at the right edge of the chart
    fan_prices = sorted(
        [{"ratio": f["ratio"], "label": f["label"], "price": _fan_price_at(f, n - 1)} for f in fans],
        key=lambda x: x["price"],
    )

    # Identify the deeper (prime) OTE zone: 50% and 61.8%
    prime_fans = [fp for fp in fan_prices if fp["ratio"] >= 0.50]
    shallow_fans = [fp for fp in fan_prices if fp["ratio"] < 0.50]

    in_zone = False
    in_prime_zone = False
    nearest_fan = None
    min_dist = 1e18

    for fp in fan_prices:
        dist = abs(close - fp["price"])
        if dist < min_dist:
            min_dist = dist
            nearest_fan = fp
        if dist <= atr * 0.50:
            in_zone = True
            if fp["ratio"] >= 0.50:
                in_prime_zone = True

    # Also treat price sitting between the 50 and 61.8 rays as inside the zone
    if len(prime_fans) >= 2:
        lo = min(p["price"] for p in prime_fans)
        hi = max(p["price"] for p in prime_fans)
        if lo - atr * 0.25 <= close <= hi + atr * 0.25:
            in_zone = True
            in_prime_zone = True

    reasons = []
    score = 35

    # --- Impulse quality ---
    leg_atr = impulse["leg_size"] / max(atr, 1e-9)
    if leg_atr >= 2.5:
        score += 18
        reasons.append(f"Strong impulse ({leg_atr:.1f} ATR) — high quality")
    elif leg_atr >= 1.5:
        score += 12
        reasons.append(f"Solid impulse ({leg_atr:.1f} ATR)")
    else:
        score += 5
        reasons.append(f"Moderate impulse ({leg_atr:.1f} ATR)")

    # --- Zone interaction (core of the educational image) ---
    if in_prime_zone:
        score += 28
        reasons.append(f"Price in PRIME OTE zone (50–61.8%) near {nearest_fan['label'] if nearest_fan else '?'}")
    elif in_zone:
        score += 15
        reasons.append(f"Price interacting with Fan {nearest_fan['label'] if nearest_fan else '?'}")
    else:
        reasons.append("Price not yet in OTE zone — waiting for pullback")

    # --- Candlestick confirmation (same patterns as the educational image) ---
    candle = _last_candle_pattern(df)
    if direction == "BUY" and candle.get("bullish"):
        score += 12
        reasons.append(f"Bullish confirmation: {candle.get('name')}")
    elif direction == "SELL" and candle.get("bearish"):
        score += 12
        reasons.append(f"Bearish confirmation: {candle.get('name')}")
    elif candle.get("name"):
        reasons.append(f"Candle: {candle.get('name')} (not yet aligned)")

    # --- RSI filter ---
    rsi_val = _rsi(df["Close"])
    if direction == "BUY" and rsi_val > 45:
        score += 6
        reasons.append(f"RSI supportive ({rsi_val:.1f})")
    elif direction == "SELL" and rsi_val < 55:
        score += 6
        reasons.append(f"RSI supportive ({rsi_val:.1f})")
    else:
        reasons.append(f"RSI: {rsi_val:.1f}")

    if expansions:
        score += 6
        reasons.append(f"{len(expansions)} Expansion targets projected (127.2 / 161.8 / 200 / 261.8)")

    # --- Gate against 4H → 1H top-down bias ---
    td_dir = (topdown or {}).get("direction", "NEUTRAL")
    td_allowed = bool((topdown or {}).get("allowed"))
    if td_dir in ("BUY", "SELL"):
        if td_dir == direction and td_allowed:
            score += 15
            reasons.append(f"✅ Aligned with 4H/1H top-down bias ({td_dir}) — structure permission granted")
        elif td_dir == direction and not td_allowed:
            reasons.append(f"Aligned with top-down direction ({td_dir}) but 1H structure permission not yet granted")
        else:
            score -= 28
            reasons.append(f"⚠️ 30M impulse ({direction}) conflicts with 4H/1H bias ({td_dir}) — high risk")
    else:
        reasons.append("4H/1H top-down is NEUTRAL — 30M impulse stands on its own")

    # ---------- TradingView-style position template ----------
    entry = close

    if direction == "BUY":
        # SL beyond the deepest fan / impulse origin (classic rule)
        sl_candidates = [fp["price"] for fp in fan_prices] + [impulse["start"]["price"]]
        sl = min(sl_candidates) - atr * 0.30
        # Prefer entry near the prime zone if we are in it
        if in_prime_zone and nearest_fan:
            entry = min(close, nearest_fan["price"] + atr * 0.15)
        tps = sorted([e["price"] for e in expansions if e["price"] > entry])
    else:
        sl_candidates = [fp["price"] for fp in fan_prices] + [impulse["start"]["price"]]
        sl = max(sl_candidates) + atr * 0.30
        if in_prime_zone and nearest_fan:
            entry = max(close, nearest_fan["price"] - atr * 0.15)
        tps = sorted([e["price"] for e in expansions if e["price"] < entry], reverse=True)

    tp1 = tps[0] if tps else (entry + atr * 1.8 if direction == "BUY" else entry - atr * 1.8)
    tp2 = tps[1] if len(tps) > 1 else (entry + atr * 2.8 if direction == "BUY" else entry - atr * 2.8)
    tp3 = tps[2] if len(tps) > 2 else (entry + atr * 4.0 if direction == "BUY" else entry - atr * 4.0)

    risk = abs(entry - sl)
    reward = abs(tp1 - entry)
    rr = (reward / risk) if risk > 0 else 0.0

    score = max(0, min(100, int(score)))

    # Validity: must be in zone, decent score, positive RR, no hard conflict with higher TF
    conf_ok = (direction == "BUY" and candle.get("bullish")) or (direction == "SELL" and candle.get("bearish"))
    valid = (
        direction in ("BUY", "SELL")
        and score >= 55
        and in_zone
        and rr >= 1.2
        and not (td_dir in ("BUY", "SELL") and td_dir != direction)
    )

    ticket = {
        "side": "LONG" if direction == "BUY" else "SHORT",
        "direction": direction,
        "entry": round(float(entry), 5),
        "sl": round(float(sl), 5),
        "tp1": round(float(tp1), 5),
        "tp2": round(float(tp2), 5),
        "tp3": round(float(tp3), 5),
        "rr": round(rr, 2),
        "risk": round(float(risk), 5),
        "reward": round(float(reward), 5),
        "order_type": "MARKET",
        "nearest_fan": nearest_fan["label"] if nearest_fan else None,
        "in_prime_zone": in_prime_zone,
        "confirmation": conf_ok,
        "candle": candle.get("name"),
        "entry_note": (
            f"{'Buy' if direction == 'BUY' else 'Sell'} in OTE zone"
            + (f" after {candle.get('name')}" if conf_ok else " — waiting for candle confirmation")
        ),
    }

    return {
        "in_zone": in_zone,
        "in_prime_zone": in_prime_zone,
        "nearest_fan": nearest_fan,
        "score": score,
        "reasons": reasons,
        "valid": valid,
        "ticket": ticket,          # always return the template; validity is separate
        "fan_prices": fan_prices,
        "candle": candle,
        "rsi": rsi_val,
    }



def run_ote_analysis(symbol: str, df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Full OTE analysis for a symbol: 4H/1H top-down bias, then impulse +
    Fan + Expansion detection and entry evaluation on the 30M chart.
    Always fetches/displays on 30M (falls back to 15M only if 30M truly
    doesn't have enough bars yet).
    """
    topdown = get_topdown_bias(symbol)

    timeframe = "30min"
    if df is None:
        df = market_data.fetch_candles(symbol, "30min", count=220)
        if df is None or df.empty or len(df) < 50:
            df = market_data.fetch_candles(symbol, "15min", count=220)
            timeframe = "15min (30M had insufficient history)"

    if df is None or df.empty or len(df) < 50:
        return {
            "error": "Insufficient 30M data for OTE analysis",
            "direction": "NEUTRAL", "score": 0, "valid": False,
            "symbol": symbol, "topdown": topdown,
        }

    df = _ensure_atr(df)
    n = len(df)

    impulse = _find_impulse(df)
    if impulse is None:
        return {
            "error": "No clear impulse swing found for Fan / Expansion",
            "direction": "NEUTRAL", "score": 0, "valid": False,
            "df": df, "timeframe": timeframe, "symbol": symbol, "topdown": topdown,
        }

    fans = _build_fan(impulse, n)
    expansions = _build_expansion(impulse)
    entry_eval = _evaluate_entry(df, impulse, fans, expansions, topdown=topdown)

    direction = impulse["direction"]
    score = entry_eval["score"]
    valid = entry_eval["valid"]
    reasons = entry_eval["reasons"]

    return {
        "strategy": "OTE",
        "direction": direction if valid else "NEUTRAL",
        "score": score,
        "reasons": reasons,
        "valid": valid,
        "impulse": impulse,
        "fans": fans,
        "expansions": expansions,
        "fan_prices": entry_eval["fan_prices"],
        "nearest_fan": entry_eval["nearest_fan"],
        "in_zone": entry_eval["in_zone"],
        "in_prime_zone": entry_eval.get("in_prime_zone", False),
        "position": entry_eval["ticket"],
        "ticket": entry_eval["ticket"],
        "candle": entry_eval.get("candle"),
        "rsi": entry_eval.get("rsi"),
        "df": df,
        "timeframe": timeframe,
        "symbol": symbol,
        "topdown": topdown,
    }


def format_ote_report(analysis: Dict[str, Any]) -> str:
    """Human-readable report matching the educational OTE image language."""
    symbol = analysis.get("symbol", "")
    if analysis.get("error"):
        lines = [f"🎯 OTE STRATEGY  |  {symbol}  (4H → 1H → 30M)"]
        topdown = analysis.get("topdown")
        if topdown:
            lines.append(format_topdown_summary(topdown))
            lines.append("—")
        lines.append(analysis["error"])
        return "\n".join(lines)

    direction = analysis.get("direction", "NEUTRAL")
    score = analysis.get("score", 0)
    valid = analysis.get("valid", False)
    impulse = analysis.get("impulse") or {}
    fans = analysis.get("fans") or []
    expansions = analysis.get("expansions") or []
    ticket = analysis.get("ticket") or analysis.get("position")
    nearest = analysis.get("nearest_fan")
    topdown = analysis.get("topdown")
    in_prime = analysis.get("in_prime_zone", False)

    lines = [
        f"🎯 OTE STRATEGY  |  {symbol}  (4H → 1H → 30M)",
        "─" * 44,
    ]
    if topdown:
        lines.append(format_topdown_summary(topdown))
        lines.append("—")

    status = "✅ VALID SETUP" if valid else "⏳ WAITING / WATCH"
    lines.append(f"30M Direction: {direction}  |  Score: {score}/100  |  {status}")

    start_t = impulse.get("start", {}).get("type", "?")
    end_t = impulse.get("end", {}).get("type", "?")
    leg = impulse.get("leg_size", 0)
    lines.append(f"Impulse: {start_t} → {end_t}  (leg {leg:.5f})")

    if fans:
        lines.append("Fan rays: " + " · ".join(f["label"] for f in fans) + "  (prime zone = 50–61.8%)")
    if nearest:
        zone_tag = "PRIME ZONE" if in_prime else "near Fan"
        lines.append(f"Nearest: {nearest.get('label')} @ {nearest.get('price', 0):.5f}  ({zone_tag})")
    if expansions:
        lines.append("Expansion targets: " + " · ".join(
            f"{e['label']} {e['price']:.5f}" for e in expansions[:4]
        ))

    for r in (analysis.get("reasons") or [])[:7]:
        lines.append(f"  • {r}")

    if ticket:
        side = ticket.get("side", ticket.get("direction", ""))
        lines.append("─" * 44)
        lines.append(f"{'🟢 LONG' if 'LONG' in str(side).upper() or side == 'BUY' else '🔴 SHORT'} POSITION (TradingView style)")
        lines.append(f"Entry : {ticket.get('entry')}")
        lines.append(f"SL    : {ticket.get('sl')}   (beyond OTE zone / swing)")
        lines.append(f"TP1   : {ticket.get('tp1')}   (Expansion)")
        lines.append(f"TP2   : {ticket.get('tp2')}")
        if ticket.get("tp3") is not None:
            lines.append(f"TP3   : {ticket.get('tp3')}   (RR extension)")
        lines.append(f"R:R   : 1:{ticket.get('rr', 0):.2f}")
        if ticket.get("candle"):
            lines.append(f"Candle: {ticket['candle']}")
        if ticket.get("entry_note"):
            lines.append(f"→ {ticket['entry_note']}")
        lines.append("Risk rule: 1–2% of capital · Move SL to BE after TP1 · Trail for Expansion targets")

    return "\n".join(lines)



def build_ote_ticket(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return analysis.get("ticket")




# ============================================================
# TRENDLINE STRATEGY ORCHESTRATION -- full top-down cascade
#   4H bias → 1H structure permission → 30M classic trendline entry
# ============================================================

def run_trendline_analysis(symbol: str) -> Dict[str, Any]:
    topdown = get_topdown_bias(symbol)
    df_30m = market_data.fetch_candles(symbol, "30min", count=250)
    if df_30m is None or df_30m.empty or len(df_30m) < 30:
        return {
            "error": "Insufficient 30M data for Trendline analysis",
            "direction": "NEUTRAL", "symbol": symbol, "topdown": topdown,
        }

    family = build_trendline_family(df_30m, max_lines=4, lookback_bars=90)
    family["symbol"] = symbol
    family["timeframe"] = "30min"
    family["topdown"] = topdown
    if family.get("error"):
        return family

    # Attach TradingView-style position
    pos = build_position_container(family)
    if pos:
        family["position"] = pos
        # also expose flat keys for chart_engine compatibility
        family["entry"] = pos["entry"]
        family["sl"] = pos["sl"]
        family["tp1"] = pos["tp1"]
        family["tp2"] = pos["tp2"]
        family["tp3"] = pos.get("tp3")

    direction = family.get("direction", "NEUTRAL")
    strength = family.get("strength", 0)
    td_dir = topdown.get("direction", "NEUTRAL")
    gating_notes = []

    if direction in ("BUY", "SELL"):
        if td_dir == direction and topdown.get("allowed"):
            strength = min(100, strength + 15)
            gating_notes.append(f"✅ Aligned with 4H/1H top-down bias ({td_dir}) — structure permission granted")
        elif td_dir == direction and not topdown.get("allowed"):
            gating_notes.append(
                f"Aligned with top-down direction ({td_dir}) but 1H structure permission not yet granted — lower conviction"
            )
        elif td_dir == "NEUTRAL":
            gating_notes.append("4H/1H top-down read is NEUTRAL — 30M trendline direction stands on its own")
        else:
            strength = max(0, strength - 25)
            gating_notes.append(
                f"⚠️ 30M trendline direction ({direction}) conflicts with 4H/1H top-down bias ({td_dir}) — high risk of counter-trend trade"
            )

    family["strength"] = strength
    family["gating_notes"] = gating_notes
    return family
