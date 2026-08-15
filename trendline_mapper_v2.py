"""
20-SMA event-driven trendline mapper.

This module deliberately models the manual mapping process:

1. The decisive leg that establishes price above/below the 20 SMA is the
   starting event for the directional map.
2. While the SMA is clearly rising/falling, the map follows the corresponding
   higher-low/lower-high structure.
3. When the 20 SMA flattens, the old impulse line is demoted and the mapper
   switches to the new consolidation support/resistance rails.
4. A breakout is not an entry. A valid breakout/retest confirmation is a
   directional engulfing candle OR a directional marubozu at the broken rail.
5. No confirmation candle means WAIT; a wick-only excursion is treated as a
   liquidity grab/fakeout.

The renderer-compatible line dictionaries intentionally mirror the fields
used by the existing trendline chart engine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def _line_value(line: Dict[str, Any], x: int) -> float:
    x0, y0, x1, y1 = line["x0"], line["y0"], line["x1"], line["y1"]
    if x1 == x0:
        return float(y0)
    return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))


def _atr(df: pd.DataFrame) -> np.ndarray:
    if "ATR" in df.columns:
        a = pd.to_numeric(df["ATR"], errors="coerce").to_numpy(float)
    else:
        h = pd.to_numeric(df["High"], errors="coerce").to_numpy(float)
        l = pd.to_numeric(df["Low"], errors="coerce").to_numpy(float)
        c = pd.to_numeric(df["Close"], errors="coerce").to_numpy(float)
        prev = np.roll(c, 1)
        tr = np.maximum(h - l, np.maximum(abs(h - prev), abs(l - prev)))
        a = pd.Series(tr).rolling(14, min_periods=2).mean().to_numpy(float)
    fallback = np.nanmedian(a[np.isfinite(a) & (a > 0)]) if np.any(np.isfinite(a) & (a > 0)) else 1.0
    a = np.where(np.isfinite(a) & (a > 0), a, fallback)
    return a


def sma20(df: pd.DataFrame) -> pd.Series:
    price = (pd.to_numeric(df["High"], errors="coerce") + pd.to_numeric(df["Low"], errors="coerce")) / 2.0
    return price.rolling(20, min_periods=10).mean()


def _sma_state_series(df: pd.DataFrame, sma: pd.Series, lookback: int = 5) -> List[str]:
    a = _atr(df)
    s = sma.to_numpy(float)
    states: List[str] = ["FLAT"] * len(df)
    for i in range(len(df)):
        if i < lookback or not np.isfinite(s[i]) or not np.isfinite(s[i - lookback]):
            continue
        ref = float(np.nanmean(a[max(0, i - lookback + 1): i + 1]))
        delta = float(s[i] - s[i - lookback])
        # Small SMA movement relative to the market's own ATR is flat.
        norm = delta / max(ref, 1e-9)
        if norm > 0.12:
            states[i] = "RISING"
        elif norm < -0.12:
            states[i] = "FALLING"
        else:
            states[i] = "FLAT"
    return states


def _fractal_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[Dict[str, Any]]:
    if len(df) < left + right + 3:
        return []
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    a = _atr(df)
    out: List[Dict[str, Any]] = []
    for i in range(left, len(df) - right):
        if h[i] >= np.max(h[i-left:i+right+1]):
            out.append({"index": i, "price": float(h[i]), "type": "high", "atr": float(a[i])})
        if l[i] <= np.min(l[i-left:i+right+1]):
            out.append({"index": i, "price": float(l[i]), "type": "low", "atr": float(a[i])})
    return sorted(out, key=lambda p: p["index"])


def _stable_sma_cross(df: pd.DataFrame, sma: pd.Series, direction: str, end: int) -> Optional[int]:
    """Return the index of the first decisive crossing into the current leg."""
    close = df["Close"].to_numpy(float)
    s = sma.to_numpy(float)
    start = max(20, end - 80)
    for i in range(end, start, -1):
        if not (np.isfinite(s[i]) and np.isfinite(s[i-1])):
            continue
        if direction == "BUY":
            crossed = close[i] > s[i] and close[i-1] <= s[i-1]
            stable = sum(1 for j in range(i, min(i + 3, len(close))) if close[j] > s[j]) >= 2
        else:
            crossed = close[i] < s[i] and close[i-1] >= s[i-1]
            stable = sum(1 for j in range(i, min(i + 3, len(close))) if close[j] < s[j]) >= 2
        if crossed and stable:
            return i
    return None


def _starting_leg_anchor(df: pd.DataFrame, sma: pd.Series, pivots: List[Dict[str, Any]], direction: str, end: int) -> Optional[Dict[str, Any]]:
    cross = _stable_sma_cross(df, sma, direction, end)
    if cross is None:
        return None
    if direction == "BUY":
        candidates = [p for p in pivots if p["type"] == "low" and p["index"] <= cross]
        if not candidates:
            return None
        # The low that launched the decisive leg: lowest meaningful low in
        # the short pre-cross window, not an arbitrary historical fractal.
        pool = [p for p in candidates if p["index"] >= max(0, cross - 14)] or candidates[-3:]
        anchor = min(pool, key=lambda p: p["price"])
    else:
        candidates = [p for p in pivots if p["type"] == "high" and p["index"] <= cross]
        if not candidates:
            return None
        pool = [p for p in candidates if p["index"] >= max(0, cross - 14)] or candidates[-3:]
        anchor = max(pool, key=lambda p: p["price"])
    out = dict(anchor)
    out["cross_index"] = cross
    out["map_start"] = True
    return out


def _best_hl_after(anchor: Dict[str, Any], pivots: List[Dict[str, Any]], end: int) -> Optional[Dict[str, Any]]:
    lows = [p for p in pivots if p["type"] == "low" and anchor["index"] < p["index"] <= end]
    if not lows:
        return None
    valid = [p for p in lows if p["price"] > anchor["price"]]
    if not valid:
        return None
    # Prefer the latest meaningful higher low, with a minimum separation.
    for p in reversed(valid):
        if p["index"] - anchor["index"] >= 4:
            return p
    return valid[-1]


def _best_lh_after(anchor: Dict[str, Any], pivots: List[Dict[str, Any]], end: int) -> Optional[Dict[str, Any]]:
    highs = [p for p in pivots if p["type"] == "high" and anchor["index"] < p["index"] <= end]
    if not highs:
        return None
    valid = [p for p in highs if p["price"] < anchor["price"]]
    if not valid:
        return None
    for p in reversed(valid):
        if p["index"] - anchor["index"] >= 4:
            return p
    return valid[-1]


def _line(a: Dict[str, Any], b: Dict[str, Any], n: int, kind: str, method: str) -> Dict[str, Any]:
    dx = max(int(b["index"]) - int(a["index"]), 1)
    slope = (float(b["price"]) - float(a["price"])) / dx
    y_end = float(b["price"] + slope * (n - 1 - int(b["index"])))
    return {
        "x0": int(a["index"]), "y0": float(a["price"]),
        "x1": int(b["index"]), "y1": float(b["price"]),
        "y_end": y_end, "slope": float(slope), "kind": kind,
        "touches": 2, "violations": 0, "confirmed": True,
        "quality": "confirmed", "method": method,
        "bars_since_last_touch": max(0, n - 1 - int(b["index"])),
    }


def _consolidation_start(states: List[str], end: int, min_flat: int = 4) -> Optional[int]:
    run = 0
    for i in range(end, -1, -1):
        if states[i] == "FLAT":
            run += 1
            if run >= min_flat:
                return i
        else:
            run = 0
    return None


def _consolidation_rails(df: pd.DataFrame, pivots: List[Dict[str, Any]], start: int, end: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if start is None:
        return None, None
    local = [p for p in pivots if start <= p["index"] <= end]
    highs = [p for p in local if p["type"] == "high"]
    lows = [p for p in local if p["type"] == "low"]
    support = resistance = None
    if len(lows) >= 2:
        # Green rail: first meaningful low -> latest higher low.
        pairs = []
        for i in range(len(lows)-1):
            for j in range(i+1, len(lows)):
                if lows[j]["price"] >= lows[i]["price"] and lows[j]["index"] - lows[i]["index"] >= 4:
                    pairs.append((lows[j]["index"] - lows[i]["index"], lows[i], lows[j]))
        if pairs:
            _, a, b = max(pairs, key=lambda x: (x[0], x[2]["index"]))
            support = _line(a, b, len(df), "support", "sma_flat_consolidation")
    if len(highs) >= 2:
        # Red rail: first meaningful high -> latest lower high.
        pairs = []
        for i in range(len(highs)-1):
            for j in range(i+1, len(highs)):
                if highs[j]["price"] <= highs[i]["price"] and highs[j]["index"] - highs[i]["index"] >= 4:
                    pairs.append((highs[j]["index"] - highs[i]["index"], highs[i], highs[j]))
        if pairs:
            _, a, b = max(pairs, key=lambda x: (x[0], x[2]["index"]))
            resistance = _line(a, b, len(df), "resistance", "sma_flat_consolidation")
    return support, resistance


def _touches(df: pd.DataFrame, line: Dict[str, Any]) -> int:
    a = _atr(df)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    count = 0
    for i in range(line["x0"], min(line["x1"] + 1, len(df))):
        lv = _line_value(line, i)
        tol = max(a[i] * 0.30, 1e-9)
        if line["kind"] == "support" and abs(l[i] - lv) <= tol:
            count += 1
        elif line["kind"] == "resistance" and abs(h[i] - lv) <= tol:
            count += 1
    return count


def _is_bull_engulfing(o, h, l, c, po, ph, pl, pc) -> bool:
    return pc < po and c > o and o <= pc and c >= po and (c-o) >= max(abs(pc-po), 1e-9) * 0.95


def _is_bear_engulfing(o, h, l, c, po, ph, pl, pc) -> bool:
    return pc > po and c < o and o >= pc and c <= po and (o-c) >= max(abs(pc-po), 1e-9) * 0.95


def _is_marubozu(o, h, l, c, direction: str) -> bool:
    rng = max(h-l, 1e-9)
    body = abs(c-o)
    if body / rng < 0.75:
        return False
    upper = h - max(o, c)
    lower = min(o, c) - l
    if max(upper, lower) / rng > 0.15:
        return False
    return c > o if direction == "BUY" else c < o


def _confirmation(df: pd.DataFrame, line: Dict[str, Any], break_kind: str, end: int) -> Dict[str, Any]:
    out = {"status": "WAIT", "break_index": None, "confirmation_index": None,
           "confirmation": None, "retest_level": None, "fakeout": False,
           "note": "No engulfing or marubozu confirmation."}
    if not line:
        return out
    close = df["Close"].to_numpy(float); open_ = df["Open"].to_numpy(float)
    high = df["High"].to_numpy(float); low = df["Low"].to_numpy(float)
    a = _atr(df)
    down = break_kind == "support_break_down"
    start = max(int(line["x1"]) + 1, end - 40)
    break_i = None
    for i in range(start, end + 1):
        lv = _line_value(line, i)
        body = abs(close[i] - open_[i]); rng = max(high[i] - low[i], 1e-9)
        pen = abs(close[i] - lv) / max(a[i], 1e-9)
        crossed = close[i] < lv if down else close[i] > lv
        if crossed and pen >= 0.08 and body / rng >= 0.35:
            break_i = i
            break
    if break_i is None:
        return out
    out["break_index"] = break_i
    # Scan from the break itself. A marubozu/engulfing breakout candle is a
    # valid confirmation; otherwise a later return to the rail must produce
    # the candle pattern. There is no separate confirmation layer.
    for i in range(break_i, end + 1):
        lv = _line_value(line, i)
        tol = max(a[i] * 0.40, 1e-9)
        near = (low[i] <= lv + tol) if not down else (high[i] >= lv - tol)
        bull_eng = bear_eng = False
        if i > 0:
            bull_eng = _is_bull_engulfing(open_[i], high[i], low[i], close[i], open_[i-1], high[i-1], low[i-1], close[i-1])
            bear_eng = _is_bear_engulfing(open_[i], high[i], low[i], close[i], open_[i-1], high[i-1], low[i-1], close[i-1])
        maru = _is_marubozu(open_[i], high[i], low[i], close[i], "SELL" if down else "BUY")
        correct = bear_eng if down else bull_eng
        # The candle must interact with the broken rail. For the actual break
        # candle, its close beyond the rail is enough; for later candles, the
        # wick/body must return to the rail zone.
        interaction = near or i == break_i
        if interaction and (correct or maru):
            name = ("Bearish Engulfing" if bear_eng else "Bullish Engulfing" if bull_eng else
                    "Bearish Marubozu" if down else "Bullish Marubozu")
            out.update(status="CONFIRMED", confirmation_index=i,
                       confirmation=name, retest_level=float(lv),
                       note=f"Valid {name}: breakout/retest confirmed.")
            return out
        if i > break_i:
            reclaimed = close[i] > lv if down else close[i] < lv
            if reclaimed:
                out.update(status="FAKEOUT", confirmation_index=i, fakeout=True,
                           retest_level=float(lv), note="Break reclaimed before engulfing/marubozu confirmation.")
                return out
    out["retest_level"] = float(_line_value(line, end))
    return out


def _directional_map(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    s = sma20(df)
    states = _sma_state_series(df, s, lookback=5)
    pivots = _fractal_pivots(df)
    current_state = states[-1] if states else "FLAT"
    close = float(df["Close"].iloc[-1])

    support = resistance = None
    map_start = None
    regime = current_state

    if current_state in ("RISING", "FALLING"):
        direction = "BUY" if current_state == "RISING" else "SELL"
        map_start = _starting_leg_anchor(df, s, pivots, direction, n - 1)
        if map_start:
            if direction == "BUY":
                endpoint = _best_hl_after(map_start, pivots, n - 1)
                if endpoint:
                    support = _line(map_start, endpoint, n, "support", "20sma_starting_leg")
            else:
                endpoint = _best_lh_after(map_start, pivots, n - 1)
                if endpoint:
                    resistance = _line(map_start, endpoint, n, "resistance", "20sma_starting_leg")
    else:
        # The SMA has flattened: remap the current consolidation rather than
        # extending the old impulse trendline.
        cstart = _consolidation_start(states, n - 1, min_flat=4)
        support, resistance = _consolidation_rails(df, pivots, cstart, n - 1) if cstart is not None else (None, None)
        map_start = {"index": cstart, "map_start": True, "phase": "CONSOLIDATION"} if cstart is not None else None
        regime = "CONSOLIDATION"
        direction = "NEUTRAL"

    for line in (support, resistance):
        if line:
            line["touches"] = _touches(df, line)
            line["quality"] = "confirmed" if line["touches"] >= 2 else "unconfirmed"
            line["confirmed"] = line["touches"] >= 2

    primary = None
    break_kind = None
    base_bias = direction
    if current_state == "RISING" and support:
        primary, break_kind = support, "support_break_down"
    elif current_state == "FALLING" and resistance:
        primary, break_kind = resistance, "resistance_break_up"
    elif current_state == "FLAT":
        # In consolidation both rails are valid; choose the rail actually
        # being challenged, but keep both in the map.
        if support and close < _line_value(support, n - 1):
            primary, break_kind = support, "support_break_down"
        elif resistance and close > _line_value(resistance, n - 1):
            primary, break_kind = resistance, "resistance_break_up"

    confirmation = _confirmation(df, primary, break_kind, n - 1) if primary and break_kind else {
        "status": "WAIT", "break_index": None, "confirmation_index": None,
        "confirmation": None, "retest_level": None, "fakeout": False,
        "note": "No active structural break."
    }

    if confirmation["status"] == "CONFIRMED":
        final_direction = "SELL" if break_kind == "support_break_down" else "BUY"
        decision = f"CONFIRMED {final_direction} — {confirmation['confirmation']}"
    elif confirmation["status"] == "FAKEOUT":
        final_direction = base_bias
        decision = "FAKEOUT — NO ENTRY"
    else:
        final_direction = base_bias
        decision = "WAIT — BREAK NOT CONFIRMED"

    lines = [x for x in (support, resistance) if x]
    family_kind = "ascending" if current_state == "RISING" else "descending" if current_state == "FALLING" else "consolidation"
    reasons = [
        f"20 SMA regime: {current_state}",
        "Map starts from the decisive leg that established price across the 20 SMA." if map_start else "No decisive 20 SMA starting leg found in the current window.",
    ]
    if current_state == "FLAT":
        reasons.append("20 SMA is flat → old impulse line is demoted; map current consolidation support/resistance.")
    if confirmation["status"] == "CONFIRMED":
        reasons.append(f"Retest/continuation confirmed by {confirmation['confirmation']} — this is the entry confirmation.")
    elif confirmation["status"] == "FAKEOUT":
        reasons.append("Break was reclaimed before engulfing/marubozu confirmation → liquidity grab/fakeout.")
    else:
        reasons.append("No engulfing or marubozu confirmation → WAIT.")

    entry_rules = {
        "checks": {"candle": (confirmation["status"] == "CONFIRMED", confirmation.get("confirmation") or "waiting for engulfing/marubozu")},
        "passed": 1 if confirmation["status"] == "CONFIRMED" else 0,
        "required": 1,
        "confirmed": confirmation["status"] == "CONFIRMED",
    }

    annotations = []
    for p in pivots[-18:]:
        prior = [q for q in pivots if q["type"] == p["type"] and q["index"] < p["index"]]
        if not prior:
            label = "H" if p["type"] == "high" else "L"
        elif p["type"] == "high":
            label = "HH" if p["price"] > prior[-1]["price"] else "LH"
        else:
            label = "HL" if p["price"] > prior[-1]["price"] else "LL"
        annotations.append({"index": p["index"], "price": p["price"], "label": label, "type": p["type"]})

    return {
        "sma": s, "sma_direction": current_state if current_state != "CONSOLIDATION" else "FLAT",
        "sma_last": float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else None,
        "sma_series": s.to_numpy(), "pivots": pivots,
        "support": support, "resistance": resistance, "primary": primary,
        "family_lines": lines, "uptrends": [support] if support else [],
        "downtrends": [resistance] if resistance else [],
        "family_kind": family_kind, "direction": final_direction,
        "strength": 75 if confirmation["status"] == "CONFIRMED" else 45 if current_state == "FLAT" else 65,
        "map_start": map_start, "map_start_index": map_start.get("index") if map_start else None,
        "regime": regime, "master_trendline": primary, "master_role": primary.get("kind") if primary else "none",
        "master_line_value": _line_value(primary, n - 1) if primary else None,
        "master_decision": decision, "decision": decision,
        "breakout_grade": None, "trendline_retest": confirmation,
        "breakout_confirmation": confirmation, "retest_level": confirmation.get("retest_level"),
        "entry_rules": entry_rules, "force_wait_pattern": not entry_rules["confirmed"],
        "prefer_retest_entry": entry_rules["confirmed"], "master_entry_ready": entry_rules["confirmed"],
        "trendline_annotations": annotations, "bias_touch_points": [],
        "reasons": reasons,
    }


def apply_map(family: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    mapped = _directional_map(df)
    # Preserve the existing strategy's unrelated layers (OB, volume profile,
    # targets, top-down context, etc.) and replace only the trendline/map state.
    for key, value in mapped.items():
        family[key] = value
    family["df"] = df
    family["trendline_v2"] = True
    family["sma_period"] = 20
    family["sma_applied_price"] = "median"
    return family


def install() -> None:
    try:
        import strategies
    except Exception as exc:
        print(f"[trendline_v2] strategies import failed: {exc!r}")
        return
    original = getattr(strategies, "build_trendline_family", None)
    if original is None or getattr(original, "_trendline_v2_wrapped", False):
        return

    def wrapped(df, max_lines=4, lookback_bars=60):
        family = original(df, max_lines=max_lines, lookback_bars=lookback_bars)
        if not family or family.get("error"):
            return family
        try:
            return apply_map(family, df)
        except Exception as exc:
            family.setdefault("reasons", []).append(f"Trendline v2 mapping error: {exc!r}")
            return family

    wrapped._trendline_v2_wrapped = True
    wrapped._original = original
    strategies.build_trendline_family = wrapped
    print("[trendline_v2] 20-SMA event-driven trendline mapper installed")
