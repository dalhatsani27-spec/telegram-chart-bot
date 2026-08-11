"""
market_analysis.py
===================
Shared price-action analysis toolkit: market structure (swings / BOS /
CHoCH / MSS), volume profile, support/resistance zone clustering,
candlestick confirmation patterns, the chart-pattern scanner, and the
shared LONG/SHORT/NEUTRAL direction banner.

Used by: the Trendline and OTE strategies (strategies.py), the top-down
bias engine (topdown_engine.py), and the auto-trading confirmation
pipeline (execution_engine.py).
"""

import numpy as np
import pandas as pd


# ============================================================
# DIRECTION BANNER -- shared LONG/SHORT/NEUTRAL banner used by
# every strategy report and every live trade message.
# ============================================================
_LONG_WORDS = {"BUY", "LONG", "BULLISH"}
_SHORT_WORDS = {"SELL", "SHORT", "BEARISH"}


def normalize_direction(direction) -> str:
    """Collapse BUY/LONG/BULLISH -> LONG, SELL/SHORT/BEARISH -> SHORT,
    everything else -> NEUTRAL. Case-insensitive, None-safe."""
    d = str(direction or "").strip().upper()
    if d in _LONG_WORDS:
        return "LONG"
    if d in _SHORT_WORDS:
        return "SHORT"
    return "NEUTRAL"


def direction_banner(direction, extra: str = "") -> str:
    """
    Big three-line block, unmissable even skimming on a phone.
    `extra` appends context on the label line (symbol, strategy name, etc).
    """
    norm = normalize_direction(direction)
    if norm == "LONG":
        emoji, label, bar = "🟢", "LONG", "🟩"
    elif norm == "SHORT":
        emoji, label, bar = "🔴", "SHORT", "🟥"
    else:
        emoji, label, bar = "⚪", "NEUTRAL", "⬜"
    row = bar * 12
    tail = f"  ·  {extra}" if extra else ""
    return f"{row}\n{emoji}  {label}{tail}  {emoji}\n{row}"


def direction_tag(direction) -> str:
    """Compact inline tag for use inside an existing line, e.g.
    f"Stage 8 — Entry: {direction_tag(direction)}" """
    norm = normalize_direction(direction)
    if norm == "LONG":
        return "🟢 LONG"
    if norm == "SHORT":
        return "🔴 SHORT"
    return "⚪ NEUTRAL"


# ============================================================
# MARKET STRUCTURE -- swing highs/lows, BOS / CHoCH / MSS,
# structure-based trade permission.
# ============================================================


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


def detect_market_sequence(df, lookback=80, min_leg_atr=0.9, max_base_atr=2.8):
    """
    Detect the four core market sequences from recent swing structure:

      RBR  Rally → Base → Rally   (bullish continuation)
      DBD  Drop  → Base → Drop    (bearish continuation)
      RBD  Rally → Base → Drop    (possible distribution / reversal)
      DBR  Drop  → Base → Rally   (possible accumulation / reversal)

    Method:
      1. Take recent significant swings (zigzag)
      2. Identify the last three major legs: impulse1 → base → impulse2
      3. Classify by direction of the two impulses and size of the base

    Returns dict or None:
      sequence, bias, confidence, legs, note
    """
    if df is None or len(df) < 30:
        return None

    pivots = zigzag_swings(df, depth=4, deviation_atr=0.30)
    n = len(df)
    start = max(0, n - lookback)
    pivots = [p for p in pivots if p["index"] >= start]
    if len(pivots) < 4:
        return None

    atr = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).rolling(14).mean().values
    atr = np.asarray(atr, dtype=float)

    def _atr_at(idx):
        i = min(max(int(idx), 0), len(atr) - 1)
        a = float(atr[i]) if not np.isnan(atr[i]) else 0.0
        return max(a, 1e-9)

    # Build alternating legs between consecutive pivots
    legs = []
    for i in range(1, len(pivots)):
        a, b = pivots[i - 1], pivots[i]
        move = b["price"] - a["price"]
        a_atr = _atr_at(a["index"])
        leg_atr = abs(move) / a_atr
        direction = "RALLY" if move > 0 else "DROP"
        legs.append({
            "from": a, "to": b,
            "direction": direction,
            "move": move,
            "leg_atr": leg_atr,
            "bars": b["index"] - a["index"],
        })

    if len(legs) < 3:
        return None

    # Walk from the end: find impulse → base → impulse pattern
    # Base = relatively small leg (or cluster) between two larger legs
    best = None
    for i in range(len(legs) - 2, 0, -1):
        leg1 = legs[i - 1] if i >= 1 else None
        mid = legs[i]
        leg2 = legs[i + 1] if i + 1 < len(legs) else None
        # Prefer using last three meaningful legs
        if leg1 is None or leg2 is None:
            continue

        # leg1 and leg2 should be the impulses (larger), mid the base
        # Allow mid to be a single leg that is smaller than both impulses
        if leg1["leg_atr"] < min_leg_atr or leg2["leg_atr"] < min_leg_atr:
            continue
        # Base should be corrective / smaller
        if mid["leg_atr"] > max_base_atr and mid["leg_atr"] >= min(leg1["leg_atr"], leg2["leg_atr"]) * 0.85:
            # too large to be a base — skip
            continue

        d1, d2 = leg1["direction"], leg2["direction"]
        if d1 == "RALLY" and d2 == "RALLY":
            seq, bias = "RBR", "BUY"
        elif d1 == "DROP" and d2 == "DROP":
            seq, bias = "DBD", "SELL"
        elif d1 == "RALLY" and d2 == "DROP":
            seq, bias = "RBD", "SELL"  # distribution lean
        elif d1 == "DROP" and d2 == "RALLY":
            seq, bias = "DBR", "BUY"   # accumulation lean
        else:
            continue

        # Confidence: stronger when impulses are clear and base is tight
        conf = 55.0
        conf += min(15.0, (leg1["leg_atr"] - min_leg_atr) * 4)
        conf += min(15.0, (leg2["leg_atr"] - min_leg_atr) * 4)
        conf += min(10.0, max(0, max_base_atr - mid["leg_atr"]) * 3)
        # Continuation sequences slightly higher base confidence
        if seq in ("RBR", "DBD"):
            conf += 5
        # Recency of leg2
        bars_since = n - 1 - leg2["to"]["index"]
        if bars_since > 30:
            conf -= 8
        conf = float(np.clip(conf, 50, 92))

        # Prefer the most recent valid sequence
        best = {
            "sequence": seq,
            "bias": bias,
            "confidence": conf,
            "leg1": leg1,
            "base": mid,
            "leg2": leg2,
            "note": (
                f"{seq}: {d1.title()} → Base ({mid['leg_atr']:.1f}×ATR) → {d2.title()} "
                f"({leg1['leg_atr']:.1f}×ATR / {leg2['leg_atr']:.1f}×ATR). "
                f"{'Continuation' if seq in ('RBR', 'DBD') else 'Reversal lean'} bias {bias}."
            ),
        }
        break  # most recent match wins

    return best


def detect_order_blocks(df, left=3, right=3, lookback=150, max_per_side=2,
                        min_confidence=45):
    """
    Order blocks: the last opposite-colored candle before a strong
    displacement move that breaks market structure.

    This is NOT "find every big candle" -- an order block is only kept if
    it's actually "likely to be respected":
      1. It caused a confirmed structure break (BOS) -- the impulse leaving
         it was strong enough to take out a prior swing high/low, not just
         a big wick that went nowhere.
      2. Displacement strength: how far price moved (in ATR) leaving the
         zone -- weak displacement = weak zone.
      3. Freshness: has price come back to the zone since it formed?
         - untested  -> highest quality, nothing has challenged it yet
         - tested-held -> price returned and respected it (bounced/rejected)
           -- actually a *positive* signal, the zone proved itself
         - broken -> a candle closed all the way through it -> DISCARDED,
           a broken zone isn't "likely to be respected", it already wasn't
      4. Zone tightness -- a huge sprawling range is a worse zone than a
         tight one.

    Returns list of dicts, most relevant first (nearest to current price),
    capped at max_per_side per direction so the chart doesn't fill up with
    marginal zones:
      type            : 'bullish' | 'bearish'
      top / bottom    : zone price bounds
      formed_index    : bar index of the OB candle
      break_index     : bar index of the structure break it caused
      displacement_atr: strength of the move away from the zone
      freshness       : 'untested' | 'tested-held'
      confidence      : 0-100
      grade           : 'strong' | 'moderate'
    """
    if df is None or len(df) < left + right + 20:
        return []

    n = len(df)
    start = max(0, n - lookback)
    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    atr = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).rolling(14, min_periods=1).mean().values

    swings = [s for s in find_swings(df, left=left, right=right) if s["index"] >= start]
    if len(swings) < 2:
        return []

    # Raw structure-break events: a close beyond a prior swing high/low.
    # (Deliberately simpler than analyse_structure's full CHoCH/MSS state
    # machine -- for order-block purposes we just need "structure broke,
    # here, in this direction", not the BOS/CHoCH/MSS labeling.)
    breaks = []
    last_high = None
    last_low = None
    for sw in swings:
        if sw["type"] == "high":
            if last_high is not None:
                for j in range(sw["index"], min(sw["index"] + 40, n)):
                    if closes[j] > last_high["price"]:
                        breaks.append({"index": j, "bias": "BULLISH", "level": last_high["price"]})
                        break
            last_high = sw
        else:
            if last_low is not None:
                for j in range(sw["index"], min(sw["index"] + 40, n)):
                    if closes[j] < last_low["price"]:
                        breaks.append({"index": j, "bias": "BEARISH", "level": last_low["price"]})
                        break
            last_low = sw

    candidates = []
    seen_ob_idx = set()
    for brk in breaks:
        b = brk["index"]
        bias = brk["bias"]
        # Find the origin candle: last opposite-colored candle in the
        # short window before the break bar.
        ob_idx = None
        for idx in range(b, max(0, b - 8), -1):
            is_down = closes[idx] < opens[idx]
            is_up = closes[idx] > opens[idx]
            if bias == "BULLISH" and is_down:
                ob_idx = idx
                break
            if bias == "BEARISH" and is_up:
                ob_idx = idx
                break
        if ob_idx is None or ob_idx in seen_ob_idx:
            continue
        seen_ob_idx.add(ob_idx)

        top = float(highs[ob_idx])
        bottom = float(lows[ob_idx])
        if top <= bottom:
            continue
        a = float(atr[b]) if b < len(atr) and atr[b] > 0 else (top - bottom)
        displacement_atr = abs(closes[min(b + 1, n - 1)] - closes[ob_idx]) / a if a > 0 else 0.0
        width_atr = (top - bottom) / a if a > 0 else 99

        # Freshness: has price returned into the zone since it formed?
        # A close all the way through invalidates it outright.
        freshness = "untested"
        broken = False
        for k in range(min(b + 2, n), n):
            if bias == "BULLISH":
                if closes[k] < bottom:
                    broken = True
                    break
                if lows[k] <= top:
                    freshness = "tested-held"
            else:
                if closes[k] > top:
                    broken = True
                    break
                if highs[k] >= bottom:
                    freshness = "tested-held"
        if broken:
            continue

        body_ratio = abs(closes[ob_idx] - opens[ob_idx]) / max(top - bottom, 1e-9)

        score = 0.0
        score += min(40, displacement_atr * 20)
        score += 25 if freshness == "untested" else 14
        score += 15  # always structure-confirmed by construction
        score += 10 if body_ratio < 0.55 else 0
        score += 10 if width_atr < 1.5 else 0
        confidence = max(0, min(100, int(round(score))))
        if confidence < min_confidence:
            continue

        candidates.append({
            "type": "bullish" if bias == "BULLISH" else "bearish",
            "top": top,
            "bottom": bottom,
            "formed_index": ob_idx,
            "break_index": b,
            "displacement_atr": round(displacement_atr, 2),
            "freshness": freshness,
            "confidence": confidence,
            "grade": "strong" if confidence >= 65 else "moderate",
        })

    # Nearest-to-current-price first within each side, capped so the chart
    # only shows the zones actually worth reacting to.
    bullish = sorted([c for c in candidates if c["type"] == "bullish"], key=lambda c: -c["formed_index"])[:max_per_side]
    bearish = sorted([c for c in candidates if c["type"] == "bearish"], key=lambda c: -c["formed_index"])[:max_per_side]

    # Inducement rule: when two OBs of the same type sit close to each other,
    # the first (older / further from current price in the impulse direction)
    # is treated as inducement — liquidity grab that often fails before the
    # real (deeper / fresher) OB is respected.
    def _tag_inducement(obs, side):
        if len(obs) < 2:
            for o in obs:
                o["role"] = "primary"
                o["is_inducement"] = False
            return obs
        # Sort by formation time (older first)
        ordered = sorted(obs, key=lambda c: c["formed_index"])
        # If the two zones are relatively close (within ~1.5× the average width)
        avg_width = sum(o["top"] - o["bottom"] for o in ordered) / len(ordered)
        close_enough = abs(ordered[0]["top"] - ordered[1]["top"]) < max(avg_width * 3.0, avg_width + 1e-9)
        if close_enough:
            # Older one = inducement, newer/deeper one = primary
            ordered[0]["role"] = "inducement"
            ordered[0]["is_inducement"] = True
            ordered[0]["confidence"] = max(30, ordered[0]["confidence"] - 18)
            ordered[0]["grade"] = "moderate"
            ordered[1]["role"] = "primary"
            ordered[1]["is_inducement"] = False
        else:
            for o in ordered:
                o["role"] = "primary"
                o["is_inducement"] = False
        return ordered

    bullish = _tag_inducement(bullish, "bullish")
    bearish = _tag_inducement(bearish, "bearish")
    return sorted(bullish + bearish, key=lambda c: c["formed_index"])


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


# ============================================================
# VOLUME PROFILE -- POC / Value Area from tick volume.
# ============================================================


def compute_volume_profile(df, bins=24, value_area_pct=0.68):
    """
    df: OHLC dataframe, optionally with a 'Volume' column (tick_volume).
    Returns dict with bin_edges, bin_volumes, poc_price, value_area_low,
    value_area_high -- or None if there's not enough range to profile.
    """
    if df is None or df.empty:
        return None
    price_min = float(df['Low'].min())
    price_max = float(df['High'].max())
    if price_max <= price_min:
        return None

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_volumes = np.zeros(bins)
    has_volume = 'Volume' in df.columns

    highs = df['High'].values
    lows = df['Low'].values
    vols = df['Volume'].values if has_volume else np.ones(len(df))

    for lo, hi, vol in zip(lows, highs, vols):
        if hi <= lo:
            continue
        vol = float(vol) if vol and vol > 0 else 1.0
        start_idx = max(0, int(np.searchsorted(bin_edges, lo, side='right')) - 1)
        end_idx = min(bins, int(np.searchsorted(bin_edges, hi, side='left')))
        if end_idx <= start_idx:
            idx = min(bins - 1, max(0, start_idx))
            bin_volumes[idx] += vol
            continue
        span = hi - lo
        for b in range(start_idx, end_idx):
            b_lo = max(lo, bin_edges[b])
            b_hi = min(hi, bin_edges[b + 1])
            frac = max(0.0, (b_hi - b_lo)) / span
            bin_volumes[b] += vol * frac

    total = bin_volumes.sum()
    if total <= 0:
        return None

    poc_idx = int(np.argmax(bin_volumes))
    poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

    lo_i = hi_i = poc_idx
    included_vol = bin_volumes[poc_idx]
    while included_vol / total < value_area_pct and (lo_i > 0 or hi_i < bins - 1):
        left_vol = bin_volumes[lo_i - 1] if lo_i > 0 else -1
        right_vol = bin_volumes[hi_i + 1] if hi_i < bins - 1 else -1
        if right_vol >= left_vol and hi_i < bins - 1:
            hi_i += 1
            included_vol += bin_volumes[hi_i]
        elif lo_i > 0:
            lo_i -= 1
            included_vol += bin_volumes[lo_i]
        else:
            break

    return {
        "bin_edges": bin_edges, "bin_volumes": bin_volumes, "poc_price": poc_price,
        "value_area_low": float(bin_edges[lo_i]), "value_area_high": float(bin_edges[hi_i + 1]),
    }


def level_volume_bonus(profile, price_level, tolerance_frac=0.0015):
    """
    Confidence delta for a pattern's trigger/neckline sitting at a
    volume-significant level (+) or in a thin, low-activity gap (-).
    Range: roughly -5 to +8. Returns 0.0 if no profile is available.
    """
    if profile is None or price_level is None:
        return 0.0

    bin_edges = profile["bin_edges"]
    avg_price = float((bin_edges[0] + bin_edges[-1]) / 2) or 1.0
    tol = avg_price * tolerance_frac

    poc = profile["poc_price"]
    if abs(price_level - poc) <= tol:
        return 8.0

    va_lo, va_hi = profile["value_area_low"], profile["value_area_high"]
    if va_lo - tol <= price_level <= va_hi + tol:
        return 4.0

    bin_volumes = profile["bin_volumes"]
    idx = int(np.searchsorted(bin_edges, price_level)) - 1
    idx = min(max(idx, 0), len(bin_volumes) - 1)
    max_vol = bin_volumes.max() or 1.0
    if bin_volumes[idx] < 0.15 * max_vol:
        return -5.0
    return 0.0


# ============================================================
# SUPPORT/RESISTANCE ZONE CLUSTERING
# ============================================================


def cluster_sr_zones(prices, tolerance_frac=0.0015):
    """
    prices: flat list/array of price levels (pivot highs + pivot lows combined).
    Greedily clusters values within tolerance_frac of each other.
    Returns list of {"level": float, "touch_count": int}, sorted by touch_count desc.
    """
    if prices is None or len(prices) == 0:
        return []
    sorted_prices = sorted(float(p) for p in prices)
    zones = []
    current_cluster = [sorted_prices[0]]

    for p in sorted_prices[1:]:
        cluster_center = sum(current_cluster) / len(current_cluster)
        tol = cluster_center * tolerance_frac
        if abs(p - cluster_center) <= tol:
            current_cluster.append(p)
        else:
            zones.append({"level": sum(current_cluster) / len(current_cluster), "touch_count": len(current_cluster)})
            current_cluster = [p]
    zones.append({"level": sum(current_cluster) / len(current_cluster), "touch_count": len(current_cluster)})

    zones.sort(key=lambda z: z["touch_count"], reverse=True)
    return zones


def zone_strength_bonus(zones, price_level, tolerance_frac=0.0015, max_bonus=10.0):
    """
    Confidence delta for a trigger sitting at/near a well-touched S/R zone.
    +2 per touch beyond the first 2 (i.e. a 4-touch zone -> +4, capped at max_bonus).
    Returns 0.0 if no zone is nearby.
    """
    if not zones or price_level is None:
        return 0.0
    for z in zones:
        tol = z["level"] * tolerance_frac
        if abs(price_level - z["level"]) <= tol:
            bonus = max(0, z["touch_count"] - 2) * 2.0
            return float(min(max_bonus, bonus))
    return 0.0


# ============================================================
# CANDLESTICK CONFIRMATION PATTERNS
# ============================================================
def _body(o, c):
    return abs(c - o)


def _range(h, l):
    return h - l


def is_bullish_engulfing(prior, current):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    prior_bearish = pc < po
    current_bullish = cc > co
    engulfs = (co <= pc) and (cc >= po)
    return prior_bearish and current_bullish and engulfs


def is_bearish_engulfing(prior, current):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    prior_bullish = pc > po
    current_bearish = cc < co
    engulfs = (co >= pc) and (cc <= po)
    return prior_bullish and current_bearish and engulfs


def is_hammer(bar, atr):
    o, h, l, c = bar
    rng = _range(h, l)
    if rng <= 0 or atr is None or atr <= 0 or rng < 0.5 * atr:
        return False
    body = _body(o, c)
    if body / rng < 0.08:  # avoid doji-like near-zero bodies
        return False
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (lower_wick >= 2.0 * body) and (upper_wick <= 0.3 * body if body > 0 else upper_wick <= 0.05 * rng)


def is_shooting_star(bar, atr):
    o, h, l, c = bar
    rng = _range(h, l)
    if rng <= 0 or atr is None or atr <= 0 or rng < 0.5 * atr:
        return False
    body = _body(o, c)
    if body / rng < 0.08:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (upper_wick >= 2.0 * body) and (lower_wick <= 0.3 * body if body > 0 else lower_wick <= 0.05 * rng)


def is_inverted_hammer(bar, atr):
    # same shape as shooting star, but used at the base of a downtrend as a bullish signal
    return is_shooting_star(bar, atr)


def is_piercing_line(prior, current):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    prior_bearish = pc < po
    prior_mid = (po + pc) / 2.0
    current_bullish = cc > co
    opens_below_prior_low_zone = co < pc  # gaps down or opens near/below prior close
    closes_above_midpoint = cc > prior_mid and cc < po
    return prior_bearish and current_bullish and opens_below_prior_low_zone and closes_above_midpoint


def is_dark_cloud_cover(prior, current):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    prior_bullish = pc > po
    prior_mid = (po + pc) / 2.0
    current_bearish = cc < co
    opens_above_prior_high_zone = co > pc
    closes_below_midpoint = cc < prior_mid and cc > po
    return prior_bullish and current_bearish and opens_above_prior_high_zone and closes_below_midpoint


def is_morning_star(bar1, bar2, bar3):
    o1, h1, l1, c1 = bar1
    o2, h2, l2, c2 = bar2
    o3, h3, l3, c3 = bar3
    first_bearish = c1 < o1
    second_small = _body(o2, c2) < 0.4 * _body(o1, c1) if _body(o1, c1) > 0 else True
    gapped_down = max(o2, c2) < c1
    third_bullish = c3 > o3
    closes_into_first_body = c3 > (o1 + c1) / 2.0
    return first_bearish and second_small and gapped_down and third_bullish and closes_into_first_body


def is_evening_star(bar1, bar2, bar3):
    o1, h1, l1, c1 = bar1
    o2, h2, l2, c2 = bar2
    o3, h3, l3, c3 = bar3
    first_bullish = c1 > o1
    second_small = _body(o2, c2) < 0.4 * _body(o1, c1) if _body(o1, c1) > 0 else True
    gapped_up = min(o2, c2) > c1
    third_bearish = c3 < o3
    closes_into_first_body = c3 < (o1 + c1) / 2.0
    return first_bullish and second_small and gapped_up and third_bearish and closes_into_first_body


def is_three_white_soldiers(bar1, bar2, bar3):
    bars = [bar1, bar2, bar3]
    for (o, h, l, c) in bars:
        if c <= o:
            return False
    for i in range(1, 3):
        po, ph, pl, pc = bars[i-1]
        o, h, l, c = bars[i]
        if not (o > po and o < pc):  # opens within prior body
            return False
        if not (c > pc):  # each close higher than the last
            return False
    return True


def is_three_black_crows(bar1, bar2, bar3):
    bars = [bar1, bar2, bar3]
    for (o, h, l, c) in bars:
        if c >= o:
            return False
    for i in range(1, 3):
        po, ph, pl, pc = bars[i-1]
        o, h, l, c = bars[i]
        if not (o < po and o > pc):
            return False
        if not (c < pc):
            return False
    return True


def is_tweezer_bottom(prior, current, tolerance_frac=0.001):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    avg = (pl + cl) / 2.0 or 1.0
    return abs(pl - cl) / abs(avg) <= tolerance_frac and pc < po and cc > co


def is_tweezer_top(prior, current, tolerance_frac=0.001):
    po, ph, pl, pc = prior
    co, ch, cl, cc = current
    avg = (ph + ch) / 2.0 or 1.0
    return abs(ph - ch) / abs(avg) <= tolerance_frac and pc > po and cc < co


def is_doji(bar, atr):
    """Not used as confirmation -- indecision, not conviction. Exposed for
    optional caution-flagging elsewhere (e.g. 'setup forming but last candle
    was a doji, expect more chop before a real move')."""
    o, h, l, c = bar
    rng = _range(h, l)
    if rng <= 0:
        return False
    return _body(o, c) / rng < 0.08


def detect_confirmation_candle(df, bias):
    """
    Checks the most recent bars for a directionally-matching candlestick
    confirmation pattern. Returns (found: bool, pattern_name: str or None).
    """
    n = len(df)
    if n < 3:
        return False, None
    atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else None

    def bar_at(i):
        row = df.iloc[i]
        return (float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']))

    b1, b2, b3 = bar_at(-3), bar_at(-2), bar_at(-1)

    if bias == "BUY":
        if is_bullish_engulfing(b2, b3): return True, "Bullish Engulfing"
        if is_hammer(b3, atr): return True, "Hammer"
        if is_inverted_hammer(b3, atr): return True, "Inverted Hammer"
        if is_piercing_line(b2, b3): return True, "Piercing Line"
        if is_morning_star(b1, b2, b3): return True, "Morning Star"
        if is_three_white_soldiers(b1, b2, b3): return True, "Three White Soldiers"
        if is_tweezer_bottom(b2, b3): return True, "Tweezer Bottom"
    else:
        if is_bearish_engulfing(b2, b3): return True, "Bearish Engulfing"
        if is_shooting_star(b3, atr): return True, "Shooting Star"
        if is_dark_cloud_cover(b2, b3): return True, "Dark Cloud Cover"
        if is_evening_star(b1, b2, b3): return True, "Evening Star"
        if is_three_black_crows(b1, b2, b3): return True, "Three Black Crows"
        if is_tweezer_top(b2, b3): return True, "Tweezer Top"

    return False, None


# ============================================================
# CHART PATTERN SCANNER -- used internally by the auto-trading
# confirmation pipeline (execution_engine.py). Not exposed as a
# Telegram menu strategy.
# ============================================================


# ----------------------------------------------------------------------------
# 1. SWING PIVOT DETECTION
# ----------------------------------------------------------------------------
def find_pivots(df, left=3, right=3):
    """
    Swing pivot detection, sourced from the shared ZigZag engine
    (market_structure.zigzag_swings) so pattern scanning agrees with
    everything else in the bot (chart drawing, trendlines, SMC zones,
    structure engine) instead of running its own separate fractal calc
    that could disagree and produce conflicting reads across strategies.

    left/right are kept as the public knobs (unchanged call signature)
    and mapped onto ZigZag's `depth` so nothing else has to change.

    Returns two lists of integer indices (positions, not timestamps):
        pivot_highs, pivot_lows
    """
    depth = max(2, left + right)
    pivots = zigzag_swings(df, depth=depth, deviation_atr=0.30)
    pivot_highs = [p["index"] for p in pivots if p["type"] == "high"]
    pivot_lows = [p["index"] for p in pivots if p["type"] == "low"]
    return pivot_highs, pivot_lows


def _dedupe_adjacent(pivots, min_gap):
    """Collapse pivots that are within min_gap bars of each other, keeping the first."""
    if not pivots:
        return pivots
    out = [pivots[0]]
    for p in pivots[1:]:
        if p - out[-1] >= min_gap:
            out.append(p)
    return out


# ----------------------------------------------------------------------------
# 2. SHARED HELPERS
# ----------------------------------------------------------------------------
def _line_through(p1, p2):
    """Return (slope, intercept) of the line through two (x, y) points."""
    (x1, y1), (x2, y2) = p1, p2
    if x2 == x1:
        return 0.0, y1
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def _pct(a, b):
    """Percent difference of a relative to b."""
    if b == 0:
        return 0.0
    return (a - b) / abs(b)


def _atr(df):
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        return float(df['ATR'].iloc[-1])
    return float((df['High'] - df['Low']).tail(14).mean())


class Pattern:
    """Container for a detected pattern, with everything needed to render it."""
    def __init__(self, name, category, bias, trigger_price, trigger_line,
                 key_points, confidence, note):
        self.name = name                 # display name
        self.category = category         # 'reversal' | 'continuation'
        self.bias = bias                 # 'BUY' | 'SELL'
        self.trigger_price = trigger_price   # single price level to watch (breakout/neckline)
        self.trigger_line = trigger_line     # list of (x, y) points describing the line to draw (2+ pts)
        self.key_points = key_points          # list of (x, y, label) marker points to draw
        self.confidence = confidence      # 0-100 raw detector confidence
        self.note = note                  # human-readable rationale

    def to_dict(self):
        return {
            "name": self.name, "category": self.category, "bias": self.bias,
            "trigger_price": self.trigger_price, "trigger_line": self.trigger_line,
            "key_points": self.key_points, "confidence": self.confidence, "note": self.note,
        }


# ----------------------------------------------------------------------------
# 3. INDIVIDUAL PATTERN DETECTORS
#    Each takes (df, pivot_highs, pivot_lows) and returns a Pattern or None.
# ----------------------------------------------------------------------------

def detect_double_top(df, ph, pl, min_bars=12, min_depth_atr=1.4, max_peak_diff=0.0035):
    """
    High-quality Double Top only.
    Requirements for a clean setup:
    - Two peaks within 0.35% of each other
    - At least 12 bars between tops
    - Trough depth >= 1.4× ATR (meaningful pullback)
    - Right peak must be the most recent significant high (not buried)
    - Price must still be near or below the neckline zone (not already far below or making new highs)
    - Prefer patterns where the second top is slightly lower or equal (classic)
    """
    if len(ph) < 2:
        return None
    i2, i1 = ph[-1], ph[-2]
    if (i2 - i1) < min_bars:
        return None
    # Prefer the last two clear highs; reject if a much higher high sits between them
    h1, h2 = float(df['High'].iloc[i1]), float(df['High'].iloc[i2])
    if abs(_pct(h2, h1)) > max_peak_diff:
        return None
    # Second top should not be significantly higher than the first (classic DT is equal or lower)
    if h2 > h1 * 1.002:
        return None
    between_lows = [p for p in pl if i1 < p < i2]
    if not between_lows:
        return None
    trough_i = min(between_lows, key=lambda p: df['Low'].iloc[p])
    neckline = float(df['Low'].iloc[trough_i])
    atr = _atr(df) or 1e-9
    depth = max(h1, h2) - neckline
    if depth < min_depth_atr * atr:
        return None
    current = float(df['Close'].iloc[-1])
    # Reject if price has already made a new high above both tops
    if current > max(h1, h2) * 1.001:
        return None
    # Reject if price is already deep below the neckline (pattern already played out)
    if current < neckline - 1.8 * atr:
        return None
    # Time freshness: right top should be relatively recent
    bars_since_right = len(df) - 1 - i2
    if bars_since_right > 25:
        return None

    equality_bonus = min(15, (1 - abs(_pct(h2, h1)) * 100) * 12)
    depth_bonus = min(12, (depth / atr - min_depth_atr) * 5)
    conf = 62 + equality_bonus + depth_bonus
    # Small bonus if second top is lower (more classic distribution)
    if h2 <= h1:
        conf += 3
    return Pattern(
        "Double Top", "reversal", "SELL",
        trigger_price=neckline,
        trigger_line=[(i1, neckline), (i2, neckline)],
        key_points=[(i1, h1, "Top 1"), (i2, h2, "Top 2"), (trough_i, neckline, "Neckline")],
        confidence=float(np.clip(conf, 65, 92)),
        note=(f"Clean Double Top: highs {h1:.5f} / {h2:.5f} ({abs(_pct(h2,h1))*100:.2f}% apart), "
              f"{i2 - i1} bars, neckline {neckline:.5f} (depth {depth / atr:.1f}×ATR). "
              f"Close below neckline confirms.")
    )


def detect_double_bottom(df, ph, pl, min_bars=12, min_depth_atr=1.4, max_peak_diff=0.0035):
    """
    High-quality Double Bottom only.
    Same strictness as Double Top (mirrored).
    """
    if len(pl) < 2:
        return None
    i2, i1 = pl[-1], pl[-2]
    if (i2 - i1) < min_bars:
        return None
    l1, l2 = float(df['Low'].iloc[i1]), float(df['Low'].iloc[i2])
    if abs(_pct(l2, l1)) > max_peak_diff:
        return None
    # Second bottom should not be significantly lower than the first
    if l2 < l1 * 0.998:
        return None
    between_highs = [p for p in ph if i1 < p < i2]
    if not between_highs:
        return None
    peak_i = max(between_highs, key=lambda p: df['High'].iloc[p])
    neckline = float(df['High'].iloc[peak_i])
    atr = _atr(df) or 1e-9
    height = neckline - min(l1, l2)
    if height < min_depth_atr * atr:
        return None
    current = float(df['Close'].iloc[-1])
    if current < min(l1, l2) * 0.999:
        return None
    if current > neckline + 1.8 * atr:
        return None
    bars_since_right = len(df) - 1 - i2
    if bars_since_right > 25:
        return None

    equality_bonus = min(15, (1 - abs(_pct(l2, l1)) * 100) * 12)
    depth_bonus = min(12, (height / atr - min_depth_atr) * 5)
    conf = 62 + equality_bonus + depth_bonus
    if l2 >= l1:
        conf += 3
    return Pattern(
        "Double Bottom", "reversal", "BUY",
        trigger_price=neckline,
        trigger_line=[(i1, neckline), (i2, neckline)],
        key_points=[(i1, l1, "Bottom 1"), (i2, l2, "Bottom 2"), (peak_i, neckline, "Neckline")],
        confidence=float(np.clip(conf, 65, 92)),
        note=(f"Clean Double Bottom: lows {l1:.5f} / {l2:.5f} ({abs(_pct(l2,l1))*100:.2f}% apart), "
              f"{i2 - i1} bars, neckline {neckline:.5f} (height {height / atr:.1f}×ATR). "
              f"Close above neckline confirms.")
    )


def detect_triple_top(df, ph, pl):
    if len(ph) < 3:
        return None
    i1, i2, i3 = ph[-3], ph[-2], ph[-1]
    h1, h2, h3 = df['High'].iloc[i1], df['High'].iloc[i2], df['High'].iloc[i3]
    tops = [h1, h2, h3]
    if (max(tops) - min(tops)) / max(tops) > 0.008:
        return None
    between = [p for p in pl if i1 < p < i3]
    if not between:
        return None
    trough_i = min(between, key=lambda p: df['Low'].iloc[p])
    neckline = float(df['Low'].iloc[trough_i])
    current = float(df['Close'].iloc[-1])
    if current > max(tops):
        return None
    return Pattern(
        "Triple Top", "reversal", "SELL",
        trigger_price=neckline,
        trigger_line=[(i1, neckline), (i3, neckline)],
        key_points=[(i1, h1, "Top 1"), (i2, h2, "Top 2"), (i3, h3, "Top 3"), (trough_i, neckline, "Neckline")],
        confidence=68,
        note=f"Three tests of resistance near {max(tops):.5f} rejected. Neckline at {neckline:.5f}."
    )


def detect_triple_bottom(df, ph, pl):
    if len(pl) < 3:
        return None
    i1, i2, i3 = pl[-3], pl[-2], pl[-1]
    l1, l2, l3 = df['Low'].iloc[i1], df['Low'].iloc[i2], df['Low'].iloc[i3]
    bots = [l1, l2, l3]
    if (max(bots) - min(bots)) / max(bots) > 0.008:
        return None
    between = [p for p in ph if i1 < p < i3]
    if not between:
        return None
    peak_i = max(between, key=lambda p: df['High'].iloc[p])
    neckline = float(df['High'].iloc[peak_i])
    current = float(df['Close'].iloc[-1])
    if current < min(bots):
        return None
    return Pattern(
        "Triple Bottom", "reversal", "BUY",
        trigger_price=neckline,
        trigger_line=[(i1, neckline), (i3, neckline)],
        key_points=[(i1, l1, "Bottom 1"), (i2, l2, "Bottom 2"), (i3, l3, "Bottom 3"), (peak_i, neckline, "Neckline")],
        confidence=68,
        note=f"Three tests of support near {min(bots):.5f} held. Neckline at {neckline:.5f}."
    )


def detect_head_shoulders(df, ph, pl):
    """
    High-probability Head & Shoulders only.
    Strict rules for clean, tradeable setups:
    - Clear head higher than both shoulders
    - Shoulders within ~2.2% of each other (symmetry)
    - Head must stand out meaningfully vs shoulders (at least ~0.4% or 0.7×ATR)
    - Both troughs (neckline points) must exist and form a sensible neckline
    - Right shoulder must be relatively recent
    - Price should not already be deep below the neckline (stale)
    - Prefer patterns where the right shoulder is complete and price is testing / near neckline
    """
    if len(ph) < 3:
        return None
    i1, i2, i3 = ph[-3], ph[-2], ph[-1]
    ls, head, rs = float(df['High'].iloc[i1]), float(df['High'].iloc[i2]), float(df['High'].iloc[i3])
    if not (head > ls and head > rs):
        return None
    # Tighter symmetry
    if abs(_pct(rs, ls)) > 0.022:
        return None
    atr = _atr(df) or 1e-9
    # Head must be a clear standout
    shoulder_avg = (ls + rs) / 2
    head_rise = head - shoulder_avg
    if head_rise < max(0.004 * head, 0.7 * atr):
        return None
    between1 = [p for p in pl if i1 < p < i2]
    between2 = [p for p in pl if i2 < p < i3]
    if not between1 or not between2:
        return None
    t1 = min(between1, key=lambda p: df['Low'].iloc[p])
    t2 = min(between2, key=lambda p: df['Low'].iloc[p])
    n1, n2 = float(df['Low'].iloc[t1]), float(df['Low'].iloc[t2])
    slope, intercept = _line_through((t1, n1), (t2, n2))
    neckline_now = slope * (len(df) - 1) + intercept
    current = float(df['Close'].iloc[-1])
    # Stale if already well below neckline
    if current < neckline_now - 1.5 * atr:
        return None
    # Right shoulder freshness
    if (len(df) - 1 - i3) > 22:
        return None
    # Minimum bars between key points for proper structure
    if (i2 - i1) < 6 or (i3 - i2) < 6:
        return None

    conf = 70
    # Symmetry bonus
    conf += min(8, (1 - abs(_pct(rs, ls)) * 100) * 6)
    # Head prominence bonus
    conf += min(8, (head_rise / atr) * 2.5)
    # Prefer when price is still above or near the neckline (setup not yet triggered hard)
    if current > neckline_now * 0.998:
        conf += 4
    return Pattern(
        "Head and Shoulders", "reversal", "SELL",
        trigger_price=float(neckline_now),
        trigger_line=[(t1, n1), (t2, n2)],
        key_points=[(i1, ls, "L Shoulder"), (i2, head, "Head"), (i3, rs, "R Shoulder")],
        confidence=float(np.clip(conf, 68, 93)),
        note=(f"Clean H&S: head {head:.5f} > shoulders {ls:.5f}/{rs:.5f} "
              f"(sym {_pct(rs,ls)*100:+.2f}%). Neckline ~{neckline_now:.5f}. "
              f"Close below confirms.")
    )


def detect_inverse_head_shoulders(df, ph, pl):
    """
    High-probability Inverse Head & Shoulders only.
    Same strictness as classic H&S (mirrored). This is the pattern quality
    standard the user wants (see clean XAUUSD example).
    """
    if len(pl) < 3:
        return None
    i1, i2, i3 = pl[-3], pl[-2], pl[-1]
    ls, head, rs = float(df['Low'].iloc[i1]), float(df['Low'].iloc[i2]), float(df['Low'].iloc[i3])
    if not (head < ls and head < rs):
        return None
    if abs(_pct(rs, ls)) > 0.022:
        return None
    atr = _atr(df) or 1e-9
    shoulder_avg = (ls + rs) / 2
    head_drop = shoulder_avg - head
    if head_drop < max(0.004 * head, 0.7 * atr):
        return None
    between1 = [p for p in ph if i1 < p < i2]
    between2 = [p for p in ph if i2 < p < i3]
    if not between1 or not between2:
        return None
    t1 = max(between1, key=lambda p: df['High'].iloc[p])
    t2 = max(between2, key=lambda p: df['High'].iloc[p])
    n1, n2 = float(df['High'].iloc[t1]), float(df['High'].iloc[t2])
    slope, intercept = _line_through((t1, n1), (t2, n2))
    neckline_now = slope * (len(df) - 1) + intercept
    current = float(df['Close'].iloc[-1])
    # Stale if already well above neckline
    if current > neckline_now + 1.5 * atr:
        return None
    if (len(df) - 1 - i3) > 22:
        return None
    if (i2 - i1) < 6 or (i3 - i2) < 6:
        return None

    conf = 70
    conf += min(8, (1 - abs(_pct(rs, ls)) * 100) * 6)
    conf += min(8, (head_drop / atr) * 2.5)
    if current < neckline_now * 1.002:
        conf += 4
    return Pattern(
        "Inverse Head and Shoulders", "reversal", "BUY",
        trigger_price=float(neckline_now),
        trigger_line=[(t1, n1), (t2, n2)],
        key_points=[(i1, ls, "L Shoulder"), (i2, head, "Head"), (i3, rs, "R Shoulder")],
        confidence=float(np.clip(conf, 68, 93)),
        note=(f"Clean Inverse H&S: head {head:.5f} < shoulders {ls:.5f}/{rs:.5f} "
              f"(sym {_pct(rs,ls)*100:+.2f}%). Neckline ~{neckline_now:.5f}. "
              f"Close above confirms.")
    )


def _fit_trend(points):
    """points: list of (x, y). Returns slope, intercept via least squares."""
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if len(xs) < 2 or np.all(xs == xs[0]):
        return 0.0, float(ys[-1]) if len(ys) else 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _touch_quality_score(points, slope, intercept, avg_price):
    """
    How tightly the given points actually hug their fitted line, as a
    confidence bonus (0-8). A triangle/wedge can have the "right" slope
    pattern by the numbers while price barely respects either boundary --
    this distinguishes a real, well-defended structure from a coincidental
    one. Tighter fit (lower deviation relative to price scale) -> higher bonus.
    """
    if len(points) < 2 or avg_price <= 0:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    fitted = slope * xs + intercept
    deviations = np.abs(ys - fitted)
    rms = float(np.sqrt(np.mean(deviations ** 2)))
    normalized_rms = rms / avg_price
    bonus = 8.0 - normalized_rms * 3000.0
    return float(np.clip(bonus, 0.0, 8.0))


def detect_triangle_or_wedge(df, ph, pl, lookback=60):
    """
    Uses the last several pivot highs (upper boundary) and pivot lows (lower
    boundary) within `lookback` bars to fit two trendlines, then classifies:
        - both flat-ish, converging  -> not used here (handled by rectangle)
        - upper flat, lower rising   -> Ascending Triangle (bullish continuation)
        - upper falling, lower flat  -> Descending Triangle (bearish continuation)
        - both converging, opposite slopes -> Symmetrical Triangle
        - both rising, converging    -> Rising Wedge (bearish reversal)
        - both falling, converging   -> Falling Wedge (bullish reversal)
    """
    n = len(df)
    start = max(0, n - lookback)
    recent_ph = [p for p in ph if p >= start][-4:]
    recent_pl = [p for p in pl if p >= start][-4:]
    if len(recent_ph) < 2 or len(recent_pl) < 2:
        return None

    upper_pts = [(p, float(df['High'].iloc[p])) for p in recent_ph]
    lower_pts = [(p, float(df['Low'].iloc[p])) for p in recent_pl]
    up_slope, up_intercept = _fit_trend(upper_pts)
    lo_slope, lo_intercept = _fit_trend(lower_pts)

    avg_price = float(df['Close'].tail(lookback).mean()) or 1.0
    touch_quality_bonus = (_touch_quality_score(upper_pts, up_slope, up_intercept, avg_price) +
                           _touch_quality_score(lower_pts, lo_slope, lo_intercept, avg_price)) / 2.0
    up_norm = (up_slope * lookback) / avg_price
    lo_norm = (lo_slope * lookback) / avg_price
    FLAT = 0.003   # ~0.3% drift over the window counts as "flat"

    x_now = n - 1
    upper_now = up_slope * x_now + up_intercept
    lower_now = lo_slope * x_now + lo_intercept
    if upper_now <= lower_now:
        return None  # lines already crossed, pattern played out

    current = float(df['Close'].iloc[-1])
    line = [(upper_pts[0][0], upper_pts[0][1]), (upper_pts[-1][0], upper_pts[-1][1])]
    lower_line = [(lower_pts[0][0], lower_pts[0][1]), (lower_pts[-1][0], lower_pts[-1][1])]

    # Ascending triangle: flat top, rising bottom -> bullish continuation
    if abs(up_norm) < FLAT and lo_norm > FLAT:
        return Pattern(
            "Ascending Triangle", "continuation", "BUY",
            trigger_price=float(upper_now),
            trigger_line=line,
            key_points=[(p, y, "Resistance") for p, y in upper_pts] + [(p, y, "Higher Low") for p, y in lower_pts],
            confidence=65 + touch_quality_bonus,
            note=f"Flat resistance near {upper_now:.5f} with rising higher-lows underneath — "
                 f"buyers stepping in earlier each time. Breakout above the flat top favors continuation up."
        )
    # Descending triangle: falling top, flat bottom -> bearish continuation
    if up_norm < -FLAT and abs(lo_norm) < FLAT:
        return Pattern(
            "Descending Triangle", "continuation", "SELL",
            trigger_price=float(lower_now),
            trigger_line=lower_line,
            key_points=[(p, y, "Support") for p, y in lower_pts] + [(p, y, "Lower High") for p, y in upper_pts],
            confidence=65 + touch_quality_bonus,
            note=f"Flat support near {lower_now:.5f} with falling lower-highs above — "
                 f"sellers stepping in earlier each time. Breakdown below flat support favors continuation down."
        )
    # Rising wedge: both rising, converging, upper rising slower -> bearish reversal
    if up_norm > FLAT and lo_norm > FLAT and lo_norm > up_norm:
        return Pattern(
            "Rising Wedge", "reversal", "SELL",
            trigger_price=float(lower_now),
            trigger_line=lower_line,
            key_points=[(p, y, "Upper") for p, y in upper_pts] + [(p, y, "Lower") for p, y in lower_pts],
            confidence=63 + touch_quality_bonus,
            note="Both boundaries rising but converging (upper line losing steam faster) — "
                 "classic exhaustion structure. Break of the rising lower trendline signals reversal down."
        )
    # Falling wedge: both falling, converging, lower falling slower -> bullish reversal
    if up_norm < -FLAT and lo_norm < -FLAT and up_norm < lo_norm:
        return Pattern(
            "Falling Wedge", "reversal", "BUY",
            trigger_price=float(upper_now),
            trigger_line=line,
            key_points=[(p, y, "Upper") for p, y in upper_pts] + [(p, y, "Lower") for p, y in lower_pts],
            confidence=63 + touch_quality_bonus,
            note="Both boundaries falling but converging (lower line losing steam faster) — "
                 "selling pressure fading. Break of the falling upper trendline signals reversal up."
        )
    # Symmetrical triangle: opposite slopes converging, roughly equal magnitude
    if up_norm < -FLAT and lo_norm > FLAT:
        bias = "BUY" if current >= (upper_now + lower_now) / 2 else "SELL"
        return Pattern(
            "Symmetrical Triangle", "continuation", bias,
            trigger_price=float(upper_now if bias == "BUY" else lower_now),
            trigger_line=line if bias == "BUY" else lower_line,
            key_points=[(p, y, "Upper") for p, y in upper_pts] + [(p, y, "Lower") for p, y in lower_pts],
            confidence=55 + touch_quality_bonus,
            note="Converging trendlines with contracting range (coiling price action). "
                 "Direction is set by whichever side breaks first — currently leaning "
                 + ("up." if bias == "BUY" else "down.")
        )
    return None


def detect_rectangle(df, ph, pl, lookback=50):
    n = len(df)
    start = max(0, n - lookback)
    recent_ph = [p for p in ph if p >= start]
    recent_pl = [p for p in pl if p >= start]
    if len(recent_ph) < 2 or len(recent_pl) < 2:
        return None
    highs = [df['High'].iloc[p] for p in recent_ph]
    lows = [df['Low'].iloc[p] for p in recent_pl]
    top = float(np.mean(highs))
    bottom = float(np.mean(lows))
    if (max(highs) - min(highs)) / top > 0.01 or (max(lows) - min(lows)) / bottom > 0.01:
        return None  # not flat enough to call a clean range
    if (top - bottom) / top < 0.003:
        return None
    current = float(df['Close'].iloc[-1])
    bias = "BUY" if current <= bottom * 1.003 else ("SELL" if current >= top * 0.997 else None)
    if bias is None:
        return None
    return Pattern(
        "Rectangle / Range", "continuation", bias,
        trigger_price=top if bias == "SELL" else bottom,
        trigger_line=[(recent_ph[0], top), (recent_ph[-1], top)] if bias == "SELL"
                     else [(recent_pl[0], bottom), (recent_pl[-1], bottom)],
        key_points=[(p, df['High'].iloc[p], "Range High") for p in recent_ph] +
                   [(p, df['Low'].iloc[p], "Range Low") for p in recent_pl],
        confidence=52,
        note=f"Price ranging between {bottom:.5f} and {top:.5f}. Currently testing the "
             f"{'top' if bias=='SELL' else 'bottom'} of the range."
    )


def detect_flag_or_pennant(df, lookback_pole=20, lookback_flag=15):
    """
    Bull/Bear flag & pennant detector — weighted highest per user requirement.

    Logic: look for a strong directional "flagpole" move over the last
    ~lookback_pole+lookback_flag bars, then a tight, shallow consolidation
    (the flag/pennant) over the most recent lookback_flag bars that retraces
    only a modest fraction of the pole and slopes counter to (or flat versus)
    the pole direction.
    """
    n = len(df)
    if n < lookback_pole + lookback_flag + 5:
        return None

    flag_start = n - lookback_flag
    pole_start = max(0, flag_start - lookback_pole)

    pole_df = df.iloc[pole_start:flag_start]
    flag_df = df.iloc[flag_start:]

    pole_move = float(pole_df['Close'].iloc[-1] - pole_df['Close'].iloc[0])
    pole_range = float(pole_df['High'].max() - pole_df['Low'].min()) or 1e-9
    atr = _atr(df) or 1e-9

    # Require a genuine impulsive pole: move at least ~3x ATR and directionally clean
    if abs(pole_move) < atr * 3.0:
        return None
    pole_up = pole_move > 0

    # directional cleanliness: fraction of bars closing in the pole's direction
    closes = pole_df['Close'].values
    diffs = np.diff(closes)
    if len(diffs) == 0:
        return None
    clean_frac = np.mean(diffs > 0) if pole_up else np.mean(diffs < 0)
    if clean_frac < 0.55:
        return None

    # Flag: shallow retracement, tight range, and (ideally) sloping against the pole
    flag_x = np.arange(len(flag_df))
    flag_slope, flag_intercept = np.polyfit(flag_x, flag_df['Close'].values, 1) if len(flag_df) >= 2 else (0, flag_df['Close'].iloc[-1])
    flag_range = float(flag_df['High'].max() - flag_df['Low'].min())
    retrace = flag_range / pole_range

    if retrace > 0.65:
        return None  # too deep a pullback to still call it a flag

    flag_norm_slope = (flag_slope * len(flag_df)) / (float(np.mean(flag_df['Close'])) or 1.0)
    counter_slope_ok = (flag_norm_slope < 0.004) if pole_up else (flag_norm_slope > -0.004)
    if not counter_slope_ok:
        return None

    upper = float(flag_df['High'].max())
    lower = float(flag_df['Low'].min())
    is_pennant = retrace < 0.35 and abs(flag_norm_slope) < 0.002  # tight converging = pennant

    name = ("Bull Flag" if pole_up else "Bear Flag") if not is_pennant else ("Bullish Pennant" if pole_up else "Bearish Pennant")
    bias = "BUY" if pole_up else "SELL"
    trigger_price = upper if pole_up else lower
    trigger_line = [(flag_start, upper if pole_up else lower), (n - 1, upper if pole_up else lower)]

    conf = 75 + min(15, clean_frac * 15) - min(10, retrace * 15)
    return Pattern(
        name, "continuation", bias,
        trigger_price=float(trigger_price),
        trigger_line=trigger_line,
        key_points=[(pole_start, float(pole_df['Close'].iloc[0]), "Pole Start"),
                    (flag_start, float(pole_df['Close'].iloc[-1]), "Pole End / Flag Start")],
        confidence=float(np.clip(conf, 55, 92)),
        note=(f"Strong {'bullish' if pole_up else 'bearish'} flagpole ({abs(pole_move):.5f}, "
              f"{clean_frac*100:.0f}% directional bars) followed by a tight "
              f"{retrace*100:.0f}%-retrace consolidation. Watch {trigger_price:.5f} — a break "
              f"in the pole's direction projects continuation roughly equal to the flagpole length.")
    )


# ----------------------------------------------------------------------------
# 4. TOP-LEVEL SCANNER
# ----------------------------------------------------------------------------
# Priority: flags/pennants first (highest-conviction continuation signal per
# user spec), then other continuation patterns, then classic reversals.
_PRIORITY = {
    "Bull Flag": 100, "Bear Flag": 100, "Bullish Pennant": 98, "Bearish Pennant": 98,
    "Ascending Channel": 92, "Descending Channel": 92,
    "Ascending Triangle": 85, "Descending Triangle": 85, "Symmetrical Triangle": 80,
    "Rising Wedge": 75, "Falling Wedge": 75,
    "Head and Shoulders": 72, "Inverse Head and Shoulders": 72,
    "Double Top": 68, "Double Bottom": 68,
    "Triple Top": 66, "Triple Bottom": 66,
    "Rectangle / Range": 50,
}


def scan_all_patterns(df, left=3, right=3, volume_profile=None):
    """
    Runs every detector against the given OHLC dataframe (must have a
    'Close'-indexed reset-friendly integer position order — pass df as-is,
    positions are derived internally).

    volume_profile: optional dict from volume_profile.compute_volume_profile().
    When provided, each detected pattern's confidence is adjusted based on
    whether its trigger/neckline sits at a volume-significant level (Point
    of Control / Value Area) or in a thin, low-activity gap. This is a
    post-detection adjustment layered on top -- it never changes whether a
    pattern is detected, only how much weight its trigger level deserves.

    Returns: (best_pattern_or_None, all_detected_list)
    """
    ph, pl = find_pivots(df, left=left, right=right)
    ph = _dedupe_adjacent(ph, min_gap=left + right)
    pl = _dedupe_adjacent(pl, min_gap=left + right)

    detected = []
    for fn in (detect_flag_or_pennant,):
        try:
            res = fn(df)
        except Exception as e:
            print(f"[patterns] {fn.__name__} raised: {e!r}")
            res = None
        if res:
            detected.append(res)

    for fn in (detect_double_top, detect_double_bottom, detect_triple_top,
               detect_triple_bottom, detect_head_shoulders,
               detect_inverse_head_shoulders):
        try:
            res = fn(df, ph, pl)
        except Exception as e:
            print(f"[patterns] {fn.__name__} raised: {e!r}")
            res = None
        if res:
            detected.append(res)

    for fn in (detect_triangle_or_wedge, detect_rectangle):
        try:
            res = fn(df, ph, pl)
        except Exception as e:
            print(f"[patterns] {fn.__name__} raised: {e!r}")
            res = None
        if res:
            detected.append(res)

    if not detected:
        return None, []

    if volume_profile is not None:
        for p in detected:
            bonus = level_volume_bonus(volume_profile, p.trigger_price)
            p.confidence = float(np.clip(p.confidence + bonus, 0.0, 100.0))
            if bonus > 0:
                p.note += " Trigger level sits at a high-volume node (POC/Value Area) -- reinforced."
            elif bonus < 0:
                p.note += " Trigger level sits in a thin, low-activity price gap -- treat with extra caution."

    # S/R zone clustering -- self-contained, reuses the pivots already
    # computed above. A trigger sitting on a level touched many times gets
    # weighted higher than one sitting on a level nobody's actually tested.
    touch_prices = [df['High'].iloc[i] for i in ph] + [df['Low'].iloc[i] for i in pl]
    zones = cluster_sr_zones(touch_prices)
    for p in detected:
        bonus = zone_strength_bonus(zones, p.trigger_price)
        if bonus > 0:
            p.confidence = float(np.clip(p.confidence + bonus, 0.0, 100.0))
            p.note += f" Trigger aligns with a well-defended S/R zone -- reinforced."

    # Hard quality floor: discard mediocre patterns so the bot stays silent
    # instead of pushing low-probability setups. Only high-conviction structures
    # (roughly matching the clean XAUUSD Inverse H&S standard) survive.
    MIN_CONFIDENCE = 68.0
    detected = [p for p in detected if p.confidence >= MIN_CONFIDENCE]
    if not detected:
        return None, []

    detected.sort(key=lambda p: (_PRIORITY.get(p.name, 40) + p.confidence), reverse=True)
    return detected[0], detected
