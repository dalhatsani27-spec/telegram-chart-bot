"""
market_structure.py
===================
ICT / SMC Market Structure engine.

Detects:
  - Swing highs / swing lows (fractal)
  - BOS  (Break of Structure)     — continuation of current trend
  - CHoCH (Change of Character)   — first break against structure
  - MSS  (Market Structure Shift) — confirmed reversal after CHoCH + follow-through

Used by Institutional Analysis and by SMC / ICT / AMD modules.
"""

import numpy as np
import pandas as pd


def find_swings(df, left=3, right=3):
    """
    Fractal swing highs and lows.
    Returns list of dicts: {index, price, type: 'high'|'low'}
    """
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)
    swings = []

    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        if highs[i] == window_h.max() and int(np.argmax(window_h)) == left:
            swings.append({"index": i, "price": float(highs[i]), "type": "high"})
        window_l = lows[i - left:i + right + 1]
        if lows[i] == window_l.min() and int(np.argmin(window_l)) == left:
            swings.append({"index": i, "price": float(lows[i]), "type": "low"})

    swings.sort(key=lambda s: s["index"])
    return swings



def filter_non_ranging_swings(df, pivots, range_atr_mult=2.2, min_leg_atr=0.55):
    """
    LOCKED RULE: only keep swings that are part of directional (non-ranging) structure.

    A swing is kept when the leg into it is meaningful vs ATR.
    Swings that form inside tight sideways chop are discarded.
    """
    if not pivots or df is None or len(df) < 10:
        return pivots or []

    if "ATR" in df.columns and not df["ATR"].isna().all():
        atr = df["ATR"].values.astype(float)
    else:
        atr = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean().values

    kept = [pivots[0]]
    for i in range(1, len(pivots)):
        prev = kept[-1]
        cur = pivots[i]
        a = float(atr[min(cur["index"], len(atr) - 1)]) if len(atr) else 0.0
        if a <= 0:
            a = abs(cur["price"] - prev["price"]) or 1e-9
        leg = abs(cur["price"] - prev["price"])
        # Discard micro legs that are just range noise
        if leg < min_leg_atr * a:
            # Replace previous same-type extreme if this one is more extreme
            if kept and kept[-1]["type"] == cur["type"]:
                if cur["type"] == "high" and cur["price"] > kept[-1]["price"]:
                    kept[-1] = cur
                elif cur["type"] == "low" and cur["price"] < kept[-1]["price"]:
                    kept[-1] = cur
            continue
        kept.append(cur)

    # Second pass: drop internal swings that sit inside a tight box of neighbors
    if len(kept) < 4:
        return kept
    final = [kept[0]]
    for i in range(1, len(kept) - 1):
        a = float(atr[min(kept[i]["index"], len(atr) - 1)]) if len(atr) else 0.0
        window = kept[max(0, i - 2): i + 3]
        prices = [p["price"] for p in window]
        box = max(prices) - min(prices)
        if a > 0 and box < range_atr_mult * a:
            # Inside a ranging cluster — skip unless it is the extreme of the cluster
            if kept[i]["type"] == "high" and kept[i]["price"] < max(prices) * 0.9995:
                continue
            if kept[i]["type"] == "low" and kept[i]["price"] > min(prices) * 1.0005:
                continue
        final.append(kept[i])
    final.append(kept[-1])

    # Ensure alternating after filtering
    cleaned = []
    for p in final:
        if cleaned and cleaned[-1]["type"] == p["type"]:
            if p["type"] == "high" and p["price"] >= cleaned[-1]["price"]:
                cleaned[-1] = p
            elif p["type"] == "low" and p["price"] <= cleaned[-1]["price"]:
                cleaned[-1] = p
        else:
            cleaned.append(p)
    return cleaned


def zigzag_swings(df, depth=5, deviation_atr=0.35):
    """
    ZigZag-style alternating swing highs/lows (noise-filtered).
    depth  : minimum bars between pivots
    deviation_atr : minimum reversal size as fraction of ATR

    Returns alternating list of {index, price, type: 'high'|'low'}.
    This is the preferred pivot source for OB / BOS / liquidity mapping.
    """
    if df is None or len(df) < depth * 2 + 5:
        return find_swings(df, left=max(2, depth // 2), right=max(2, depth // 2))

    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    n = len(df)

    # ATR proxy
    if "ATR" in df.columns and not df["ATR"].isna().all():
        atr = df["ATR"].values.astype(float)
    else:
        tr = np.maximum(highs - lows, 1e-9)
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    pivots = []
    # Seed with first significant extreme
    direction = 0  # 1 = looking for high, -1 = looking for low
    last_pivot_idx = 0
    last_pivot_price = (highs[0] + lows[0]) / 2.0

    # Find initial direction from first depth bars
    seed_h = float(np.max(highs[:depth]))
    seed_l = float(np.min(lows[:depth]))
    if seed_h - last_pivot_price >= last_pivot_price - seed_l:
        direction = 1
        last_pivot_idx = int(np.argmax(highs[:depth]))
        last_pivot_price = highs[last_pivot_idx]
        pivots.append({"index": last_pivot_idx, "price": float(last_pivot_price), "type": "high"})
        direction = -1  # next look for low
    else:
        direction = -1
        last_pivot_idx = int(np.argmin(lows[:depth]))
        last_pivot_price = lows[last_pivot_idx]
        pivots.append({"index": last_pivot_idx, "price": float(last_pivot_price), "type": "low"})
        direction = 1

    i = last_pivot_idx + 1
    while i < n:
        a = float(atr[i]) if i < len(atr) and atr[i] > 0 else float(highs[i] - lows[i])
        min_move = max(a * deviation_atr, 1e-9)

        if direction == 1:  # seeking swing high
            # Track running high since last pivot
            if i - last_pivot_idx < depth:
                i += 1
                continue
            window = highs[last_pivot_idx + 1:i + 1]
            if len(window) == 0:
                i += 1
                continue
            cand_idx = last_pivot_idx + 1 + int(np.argmax(window))
            cand_price = highs[cand_idx]
            # Confirm when price reverses by min_move from candidate
            if cand_price - lows[i] >= min_move and i - cand_idx >= max(1, depth // 2):
                pivots.append({"index": cand_idx, "price": float(cand_price), "type": "high"})
                last_pivot_idx = cand_idx
                last_pivot_price = cand_price
                direction = -1
        else:  # seeking swing low
            if i - last_pivot_idx < depth:
                i += 1
                continue
            window = lows[last_pivot_idx + 1:i + 1]
            if len(window) == 0:
                i += 1
                continue
            cand_idx = last_pivot_idx + 1 + int(np.argmin(window))
            cand_price = lows[cand_idx]
            if highs[i] - cand_price >= min_move and i - cand_idx >= max(1, depth // 2):
                pivots.append({"index": cand_idx, "price": float(cand_price), "type": "low"})
                last_pivot_idx = cand_idx
                last_pivot_price = cand_price
                direction = 1
        i += 1

    # Ensure alternating
    cleaned = []
    for p in pivots:
        if cleaned and cleaned[-1]["type"] == p["type"]:
            # Keep the more extreme
            if p["type"] == "high" and p["price"] >= cleaned[-1]["price"]:
                cleaned[-1] = p
            elif p["type"] == "low" and p["price"] <= cleaned[-1]["price"]:
                cleaned[-1] = p
        else:
            cleaned.append(p)
    # LOCKED: drop ranging / choppy swings
    return filter_non_ranging_swings(df, cleaned)


def _last_swing(swings, swing_type, before_idx=None):
    candidates = [s for s in swings if s["type"] == swing_type]
    if before_idx is not None:
        candidates = [s for s in candidates if s["index"] < before_idx]
    return candidates[-1] if candidates else None


def analyse_structure(df, left=3, right=3, lookback=80):
    """
    Build current market structure state from recent swings.

    Returns dict:
      bias          : 'BULLISH' | 'BEARISH' | 'NEUTRAL'
      last_event    : 'BOS' | 'CHoCH' | 'MSS' | None
      event_bias    : 'BULLISH' | 'BEARISH' | None
      event_price   : float (broken level)
      event_index   : int
      swings        : recent swing list
      structure_high: last relevant swing high
      structure_low : last relevant swing low
      note          : human-readable summary
    """
    if df is None or len(df) < left + right + 10:
        return {
            "bias": "NEUTRAL", "last_event": None, "event_bias": None,
            "event_price": None, "event_index": None, "swings": [],
            "structure_high": None, "structure_low": None,
            "note": "Insufficient data for structure.",
        }

    swings = find_swings(df, left=left, right=right)
    n = len(df)
    start = max(0, n - lookback)
    swings = [s for s in swings if s["index"] >= start]

    if len(swings) < 4:
        return {
            "bias": "NEUTRAL", "last_event": None, "event_bias": None,
            "event_price": None, "event_index": None, "swings": swings,
            "structure_high": None, "structure_low": None,
            "note": "Not enough swings for clear structure.",
        }

    # Walk swings chronologically and track structure state
    # Start neutral; first clear HH/HL or LH/LL sets bias
    bias = "NEUTRAL"
    last_event = None
    event_bias = None
    event_price = None
    event_index = None
    choch_pending = None  # after CHoCH, next BOS in same direction = MSS

    # Keep rolling structure levels
    last_sh = None  # last swing high used as structure
    last_sl = None  # last swing low used as structure

    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values

    # Process swing-by-swing for event labels, then check price breaks after each swing
    for i, sw in enumerate(swings):
        if sw["type"] == "high":
            prev_high = _last_swing(swings[:i], "high")
            if prev_high and sw["price"] > prev_high["price"]:
                # Higher high
                if bias == "BEARISH":
                    # Break of prior structure high against downtrend → CHoCH or MSS
                    last_event = "MSS" if choch_pending == "BULLISH" else "CHoCH"
                    event_bias = "BULLISH"
                    event_price = prev_high["price"]
                    event_index = sw["index"]
                    choch_pending = "BULLISH" if last_event == "CHoCH" else None
                    bias = "BULLISH"
                elif bias == "BULLISH":
                    last_event = "BOS"
                    event_bias = "BULLISH"
                    event_price = prev_high["price"]
                    event_index = sw["index"]
                    choch_pending = None
                else:
                    bias = "BULLISH"
                    last_event = "BOS"
                    event_bias = "BULLISH"
                    event_price = sw["price"]
                    event_index = sw["index"]
                last_sh = sw
            else:
                last_sh = sw

        else:  # swing low
            prev_low = _last_swing(swings[:i], "low")
            if prev_low and sw["price"] < prev_low["price"]:
                # Lower low
                if bias == "BULLISH":
                    last_event = "MSS" if choch_pending == "BEARISH" else "CHoCH"
                    event_bias = "BEARISH"
                    event_price = prev_low["price"]
                    event_index = sw["index"]
                    choch_pending = "BEARISH" if last_event == "CHoCH" else None
                    bias = "BEARISH"
                elif bias == "BEARISH":
                    last_event = "BOS"
                    event_bias = "BEARISH"
                    event_price = prev_low["price"]
                    event_index = sw["index"]
                    choch_pending = None
                else:
                    bias = "BEARISH"
                    last_event = "BOS"
                    event_bias = "BEARISH"
                    event_price = sw["price"]
                    event_index = sw["index"]
                last_sl = sw
            else:
                last_sl = sw

    # Also check if latest price has broken the most recent structure level
    # (intrabar break after last swing)
    if last_sh and n > 0:
        if closes[-1] > last_sh["price"] and bias != "BULLISH":
            last_event = "MSS" if choch_pending == "BULLISH" else "CHoCH"
            event_bias = "BULLISH"
            event_price = last_sh["price"]
            event_index = n - 1
            bias = "BULLISH"
        elif bias == "BULLISH" and closes[-1] > last_sh["price"]:
            # already bullish and making new high — BOS already counted via swings
            pass

    if last_sl and n > 0:
        if closes[-1] < last_sl["price"] and bias != "BEARISH":
            last_event = "MSS" if choch_pending == "BEARISH" else "CHoCH"
            event_bias = "BEARISH"
            event_price = last_sl["price"]
            event_index = n - 1
            bias = "BEARISH"

    structure_high = last_sh["price"] if last_sh else None
    structure_low = last_sl["price"] if last_sl else None

    if last_event and event_bias:
        note = f"{last_event} ({event_bias}) @ {event_price:.5f} — structure bias {bias}"
    else:
        note = f"Structure bias: {bias} (no recent BOS/CHoCH/MSS)"

    return {
        "bias": bias,
        "last_event": last_event,
        "event_bias": event_bias,
        "event_price": event_price,
        "event_index": event_index,
        "swings": swings[-12:],
        "structure_high": structure_high,
        "structure_low": structure_low,
        "note": note,
    }


def structure_trade_permission(htf_bias, structure):
    """
    Decide trade permission from structure confirmation.

    LOCKED RULES:
      1. 200 EMA / HTF bias = overall context ONLY — never the entry signal.
      2. When CHoCH then BOS/MSS occurs → look for pullback entry in the
         direction of the NEW trend (confirmation direction wins).
      3. Trade the confirmation direction, not the EMA direction.

    Returns:
      allowed: bool
      reason: str
      preferred_direction: 'BUY' | 'SELL' | 'NEUTRAL'
    """
    struct_bias = structure.get("bias", "NEUTRAL")
    event = structure.get("last_event")
    event_bias = structure.get("event_bias")
    htf = htf_bias  # context only

    # --- Confirmation events drive the trade ---
    if event == "MSS" and event_bias:
        direction = "BUY" if event_bias == "BULLISH" else "SELL"
        note = f"MSS confirmed ({event_bias}) — pullback entry in new trend"
        if htf not in ("NEUTRAL", direction):
            note += f" (against HTF {htf} — still valid, HTF is bias only)"
        return True, note, direction

    if event == "BOS" and event_bias:
        # BOS after structure = continuation of the current/new trend
        direction = "BUY" if event_bias == "BULLISH" else "SELL"
        note = f"BOS ({event_bias}) — pullback entry with trend"
        if htf not in ("NEUTRAL", direction):
            note += f" (HTF {htf} is bias only, not entry)"
        return True, note, direction

    if event == "CHoCH" and event_bias:
        # CHoCH alone = early warning. Wait for BOS/MSS before full entry.
        direction = "BUY" if event_bias == "BULLISH" else "SELL"
        return False, (
            f"CHoCH ({event_bias}) — watch for pullback; wait for BOS/MSS "
            f"confirmation before entry"
        ), direction

    # No fresh confirmation event — use structure bias, EMA only as soft context
    if struct_bias == "BULLISH":
        return True, "Bullish structure — look for pullback longs (EMA is bias only)", "BUY"
    if struct_bias == "BEARISH":
        return True, "Bearish structure — look for pullback shorts (EMA is bias only)", "SELL"

    return False, "No clear structure confirmation", "NEUTRAL"
