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
from __future__ import annotations

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
    return cleaned


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
    Decide whether a lower-TF pattern is allowed given HTF bias + structure.

    Returns:
      allowed: bool
      reason: str
      preferred_direction: 'BUY' | 'SELL' | 'NEUTRAL'
    """
    struct_bias = structure.get("bias", "NEUTRAL")
    event = structure.get("last_event")
    event_bias = structure.get("event_bias")

    # Map HTF BUY/SELL to structure language
    htf = htf_bias  # BUY / SELL / NEUTRAL

    if event == "MSS" and event_bias:
        # Confirmed shift — allow trading the new direction
        direction = "BUY" if event_bias == "BULLISH" else "SELL"
        return True, f"MSS confirmed ({event_bias}) — reversal permission granted", direction

    if event == "CHoCH" and event_bias:
        # Early warning only — do not fire full size yet
        direction = "BUY" if event_bias == "BULLISH" else "SELL"
        return False, f"CHoCH ({event_bias}) — watch zone, wait for MSS or BOS confirmation", direction

    if event == "BOS" and event_bias:
        direction = "BUY" if event_bias == "BULLISH" else "SELL"
        # BOS in direction of HTF = strong continuation
        if htf == "NEUTRAL" or (htf == "BUY" and direction == "BUY") or (htf == "SELL" and direction == "SELL"):
            return True, f"BOS ({event_bias}) aligned with HTF — continuation", direction
        # BOS against HTF without MSS = still counter, lower confidence
        return False, f"BOS ({event_bias}) against HTF {htf} — treat as internal structure only", direction

    # No clear event — fall back to structure bias vs HTF
    if struct_bias == "BULLISH" and htf in ("BUY", "NEUTRAL"):
        return True, "Bullish structure supports longs", "BUY"
    if struct_bias == "BEARISH" and htf in ("SELL", "NEUTRAL"):
        return True, "Bearish structure supports shorts", "SELL"

    return False, "No clear structure permission", "NEUTRAL"
"""
structure_engine.py
====================
Institutional Structure Engine (ISE)

Philosophy (per spec -- this replaces indicator-crossover thinking for the
Trendline strategy):

    Price -> Structure -> Liquidity -> Manipulation -> Acceptance -> Trade

    NOT:    RSI -> MA -> MACD -> Trade

Ten stages, run in strict order. Each stage's output feeds the next --
nothing downstream overrides an upstream veto. If Structure is unclear,
there is no Impulse to classify. If there's no Impulse, there's no
Pullback to type. If Liquidity was never swept, there's no Manipulation
to detect. If Manipulation isn't confirmed, there's no Acceptance test to
run. Only a setup that survives every gate reaches Stage 8 (Entry).

    STAGE 1  detect_market_state   -- HH+HL / LH+LL / mixed
    STAGE 2  detect_impulse        -- the strongest displacement leg
    STAGE 3  classify_pullback     -- what formed after the impulse
    STAGE 4  interpret_structure   -- what that combination means
    STAGE 5  liquidity_engine      -- was a level swept, and which kind
    STAGE 6  manipulation_detector -- did price reject the sweep immediately
    STAGE 7  acceptance_test       -- did the return hold, bar-close basis
    STAGE 8  entry_logic           -- combine everything into BUY/SELL/WAIT
    STAGE 9  trade_filter          -- explicit reject list
    STAGE 10 trade_management      -- BE at 1R, trail behind structure
"""


from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# [merged] was: from market_structure import find_swings


# ============================================================================
# STAGE 1 -- Detect Market State
# ============================================================================
def detect_market_state(df: pd.DataFrame, left: int = 3, right: int = 3,
                         lookback: int = 90) -> Dict[str, Any]:
    """
    Bullish  : most recent swing high is a Higher High AND swing low is a
               Higher Low.
    Bearish  : most recent swing high is a Lower High AND swing low is a
               Lower Low.
    Range    : anything mixed (e.g. HH but LL, or not enough swings yet).
    """
    n = len(df)
    swings = find_swings(df, left=left, right=right)
    swings = [s for s in swings if s["index"] >= max(0, n - lookback)]
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]

    if len(highs) < 2 or len(lows) < 2:
        return {"state": "RANGE", "reason": "Not enough confirmed swings yet",
                "swings": swings, "highs": highs, "lows": lows}

    hh = highs[-1]["price"] > highs[-2]["price"]
    hl = lows[-1]["price"] > lows[-2]["price"]
    lh = highs[-1]["price"] < highs[-2]["price"]
    ll = lows[-1]["price"] < lows[-2]["price"]

    if hh and hl:
        state, reason = "BULLISH", "Higher High + Higher Low"
    elif lh and ll:
        state, reason = "BEARISH", "Lower High + Lower Low"
    else:
        state, reason = "RANGE", "Mixed structure (no clean HH/HL or LH/LL)"

    return {
        "state": state, "reason": reason, "swings": swings,
        "highs": highs, "lows": lows,
        "last_high": highs[-1], "last_low": lows[-1],
        "prev_high": highs[-2], "prev_low": lows[-2],
    }


# ============================================================================
# STAGE 2 -- Detect the Impulse
# ============================================================================
def detect_impulse(df: pd.DataFrame, state_info: Dict[str, Any],
                    lookback: int = 50, min_strength_atr: float = 3.0) -> Optional[Dict[str, Any]]:
    """
    Find the strongest displacement leg in the direction implied by market
    state (bullish -> strongest up-leg, bearish -> strongest down-leg),
    anchored between two of the swings already found in Stage 1. This
    becomes the "parent leg" everything else is measured against.
    """
    state = state_info.get("state")
    if state == "RANGE":
        return None

    # Stage 1 already scoped its swing list to the market-state lookback --
    # re-filtering here by a second, shorter window can chop off the origin
    # point of the most recent impulse leg even though the leg itself (its
    # extreme) is still fully current. Use Stage 1's swings as-is.
    swings = state_info.get("swings") or []
    n = len(df)
    if len(swings) < 2:
        return None

    atr_col = df["ATR"] if "ATR" in df.columns else (df["High"] - df["Low"])
    atr = float(atr_col.iloc[-1]) if atr_col.iloc[-1] and atr_col.iloc[-1] > 0 else float((df["High"] - df["Low"]).tail(14).mean())
    atr = atr or 1e-9

    want = "low_to_high" if state == "BULLISH" else "high_to_low"
    best = None
    for i in range(len(swings) - 1):
        a, b = swings[i], swings[i + 1]
        if want == "low_to_high" and a["type"] == "low" and b["type"] == "high" and b["index"] > a["index"]:
            length = b["price"] - a["price"]
        elif want == "high_to_low" and a["type"] == "high" and b["type"] == "low" and b["index"] > a["index"]:
            length = a["price"] - b["price"]
        else:
            continue
        if length <= 0:
            continue
        length_atr = length / atr
        bars = max(1, b["index"] - a["index"])
        speed = length_atr / bars  # displacement, not a slow grind
        score = length_atr * 10 + speed * 20 + (b["index"] / max(n, 1)) * 10  # prefer recent, strong, fast legs
        if best is None or score > best["_score"]:
            best = {
                "origin_index": a["index"], "origin_price": a["price"],
                "extreme_index": b["index"], "extreme_price": b["price"],
                "length": length, "length_atr": round(length_atr, 2),
                "bars": bars, "speed_atr_per_bar": round(speed, 3),
                "direction": "BUY" if want == "low_to_high" else "SELL",
                "_score": score,
            }
    if best is None:
        return None
    if best["length_atr"] < min_strength_atr:
        best["weak"] = True
    else:
        best["weak"] = False
    best["strength"] = float(np.clip(best["length_atr"] * 8 + best["speed_atr_per_bar"] * 40, 0, 100))
    best.pop("_score", None)
    return best


# ============================================================================
# STAGE 3 -- Classify the Pullback
# ============================================================================
def classify_pullback(df: pd.DataFrame, impulse: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Everything after the impulse's extreme gets fit with two regression
    lines (highs, lows) over that segment, then classified by slope +
    width-over-time behaviour:

        Bull Flag        : impulse UP,   pullback slopes DOWN (or flat), tight
        Bear Flag        : impulse DOWN, pullback slopes UP (or flat), tight
        Rising Channel    : impulse UP,   pullback both lines rising, parallel
        Falling Channel   : impulse DOWN, pullback both lines falling, parallel
        Triangle          : converging lines, opposite slope signs
        Rectangle         : both lines ~flat, parallel
        Compression       : range narrowing bar over bar, no clean lines
        Expansion         : range widening bar over bar -- momentum continuing
    """
    n = len(df)
    seg_start = impulse["extreme_index"]
    if seg_start >= n - 6:
        return None  # not enough bars since the impulse ended yet

    seg = df.iloc[seg_start:]
    seg_n = len(seg)
    x = np.arange(seg_n)

    upper_slope, upper_b = np.polyfit(x, seg["High"].values, 1)
    lower_slope, lower_b = np.polyfit(x, seg["Low"].values, 1)

    avg_price = float(seg["Close"].mean()) or 1.0
    up_norm = (upper_slope * seg_n) / avg_price
    lo_norm = (lower_slope * seg_n) / avg_price
    FLAT = 0.003

    # Range/width trend: compare first-third average range to last-third
    third = max(2, seg_n // 3)
    early_width = float((seg["High"].iloc[:third] - seg["Low"].iloc[:third]).mean())
    late_width = float((seg["High"].iloc[-third:] - seg["Low"].iloc[-third:]).mean())
    width_ratio = (late_width / early_width) if early_width > 0 else 1.0

    impulse_dir = impulse["direction"]

    pattern = None
    bias_hint = None
    watch_for = None

    if width_ratio > 1.25 and abs(up_norm) > FLAT and abs(lo_norm) > FLAT and np.sign(up_norm) == np.sign(lo_norm):
        pattern = "EXPANSION"
        bias_hint = impulse_dir  # momentum still going the impulse's way
        watch_for = "Range widening bar over bar -- momentum continuing in the impulse direction"
    elif width_ratio < 0.75 and abs(up_norm) < FLAT * 2 and abs(lo_norm) < FLAT * 2:
        pattern = "COMPRESSION"
        watch_for = "Range narrowing with no clean boundary lines -- energy building, wait for the release"
    elif up_norm < -FLAT and lo_norm > FLAT:
        pattern = "TRIANGLE"
        watch_for = "Converging boundaries -- direction is set by whichever side breaks first"
    elif abs(up_norm) < FLAT and abs(lo_norm) < FLAT:
        pattern = "RECTANGLE"
        watch_for = "Flat, parallel boundaries -- treat as a range until one side breaks"
    elif impulse_dir == "BUY" and up_norm <= FLAT and lo_norm <= FLAT:
        pattern = "BULL_FLAG"
        bias_hint = "BUY"
        watch_for = "Shallow, controlled pullback against a strong up-impulse -- continuation bias"
    elif impulse_dir == "SELL" and up_norm >= -FLAT and lo_norm >= -FLAT:
        pattern = "BEAR_FLAG"
        bias_hint = "SELL"
        watch_for = "Shallow, controlled pullback against a strong down-impulse -- continuation bias"
    elif impulse_dir == "BUY" and up_norm > FLAT and lo_norm > FLAT:
        pattern = "RISING_CHANNEL"
        watch_for = "Rising channel after a bullish impulse -- possible distribution, watch for a liquidity sweep"
    elif impulse_dir == "SELL" and up_norm < -FLAT and lo_norm < -FLAT:
        pattern = "FALLING_CHANNEL"
        watch_for = "Falling channel after a bearish impulse -- possible accumulation, watch for a liquidity sweep"
    else:
        pattern = "COMPRESSION"
        watch_for = "No clean structure -- do not trade until this resolves"

    upper_now = float(upper_slope * (seg_n - 1) + upper_b)
    lower_now = float(lower_slope * (seg_n - 1) + lower_b)

    return {
        "pattern": pattern, "bias_hint": bias_hint, "watch_for": watch_for,
        "seg_start": seg_start, "upper_slope": float(upper_slope), "lower_slope": float(lower_slope),
        "upper_now": upper_now, "lower_now": lower_now,
        "width_ratio": round(width_ratio, 2),
        "channel_high": max(upper_now, float(seg["High"].max())),
        "channel_low": min(lower_now, float(seg["Low"].min())),
    }


# ============================================================================
# STAGE 4 -- Structure Interpretation
# ============================================================================
def interpret_structure(state_info: Dict[str, Any], impulse: Optional[Dict[str, Any]],
                         pullback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn (state, impulse, pullback) into a plain-language read + what to watch for next."""
    if impulse is None or pullback is None:
        return {"bias": "NEUTRAL", "note": "No impulse/pullback to interpret yet -- do not trade.",
                "expects_liquidity_sweep": False}

    pattern = pullback["pattern"]
    state = state_info["state"]

    if pattern in ("BULL_FLAG", "BEAR_FLAG"):
        bias = pullback["bias_hint"]
        note = f"{state.title()} trend, impulse then {pattern.replace('_', ' ').title()} -- continuation setup."
        expects_sweep = False
    elif pattern == "RISING_CHANNEL":
        bias = "WATCH"
        note = "Impulse up into a rising channel -- reads as possible distribution before a move down."
        expects_sweep = True
    elif pattern == "FALLING_CHANNEL":
        bias = "WATCH"
        note = "Impulse down into a falling channel -- reads as possible accumulation before a move up."
        expects_sweep = True
    elif pattern == "TRIANGLE":
        bias = "WATCH"
        note = "Coiling into a triangle -- direction unresolved until a boundary breaks with acceptance."
        expects_sweep = False
    elif pattern == "RECTANGLE":
        bias = "WATCH"
        note = "Ranging inside a rectangle -- fade the edges or wait for a break with acceptance."
        expects_sweep = False
    elif pattern == "EXPANSION":
        bias = impulse["direction"]
        note = "Range still expanding in the impulse direction -- momentum continuation, not exhaustion yet."
        expects_sweep = False
    else:  # COMPRESSION
        bias = "NEUTRAL"
        note = "Compression with no clean boundaries -- energy building, no read yet."
        expects_sweep = False

    return {"bias": bias, "note": note, "expects_liquidity_sweep": expects_sweep, "pattern": pattern}


# ============================================================================
# STAGE 5 -- Liquidity Engine
# ============================================================================
def liquidity_engine(df: pd.DataFrame, state_info: Dict[str, Any],
                      pullback: Optional[Dict[str, Any]], lookback: int = 10) -> Optional[Dict[str, Any]]:
    """
    Did price sweep a resting-liquidity level in the last `lookback` bars?
    Checks, in order: previous swing high/low, equal highs/equal lows
    (tolerance-based -- these are the classic "double top/bottom that never
    quite closes the trade" liquidity pools), then the channel high/low
    from Stage 3 if a channel/flag structure exists.

    Type is tagged Internal (inside the current structure/channel) vs
    External (beyond the last major swing) since institutional liquidity
    theory treats these differently -- external sweeps carry more weight.
    """
    n = len(df)
    recent = df.iloc[-lookback:]
    highs = recent["High"].values
    lows = recent["Low"].values
    closes = recent["Close"].values
    atr_col = df["ATR"] if "ATR" in df.columns else (df["High"] - df["Low"])
    atr = float(atr_col.iloc[-1]) if atr_col.iloc[-1] and atr_col.iloc[-1] > 0 else 1e-9
    tol = atr * 0.05

    candidates = []  # (level, side, label, kind)
    prev_high = state_info.get("prev_high")
    last_high = state_info.get("last_high")
    prev_low = state_info.get("prev_low")
    last_low = state_info.get("last_low")

    if last_high:
        candidates.append((last_high["price"], "high", "Previous swing high", "External"))
    if last_low:
        candidates.append((last_low["price"], "low", "Previous swing low", "External"))
    # Equal highs/lows: prev and last within tolerance of each other = a
    # liquidity pool resting exactly there.
    if prev_high and last_high and abs(prev_high["price"] - last_high["price"]) <= tol * 3:
        candidates.append((max(prev_high["price"], last_high["price"]), "high", "Equal highs (EQH)", "External"))
    if prev_low and last_low and abs(prev_low["price"] - last_low["price"]) <= tol * 3:
        candidates.append((min(prev_low["price"], last_low["price"]), "low", "Equal lows (EQL)", "External"))
    if pullback:
        candidates.append((pullback["channel_high"], "high", "Channel high", "Internal"))
        candidates.append((pullback["channel_low"], "low", "Channel low", "Internal"))

    for level, side, label, kind in candidates:
        if side == "high":
            for i in range(len(recent)):
                if highs[i] > level + tol and closes[i] < level:
                    return {"level": level, "side": "BSL", "label": label, "kind": kind,
                            "swept_index": recent.index[i], "swept_pos": n - lookback + i,
                            "direction_hint": "SELL",
                            "note": f"{label} swept at {level:.5f} ({kind} buy-side liquidity)"}
        else:
            for i in range(len(recent)):
                if lows[i] < level - tol and closes[i] > level:
                    return {"level": level, "side": "SSL", "label": label, "kind": kind,
                            "swept_index": recent.index[i], "swept_pos": n - lookback + i,
                            "direction_hint": "BUY",
                            "note": f"{label} swept at {level:.5f} ({kind} sell-side liquidity)"}
    return None


# ============================================================================
# STAGE 6 -- Manipulation Detection
# ============================================================================
def manipulation_detector(df: pd.DataFrame, sweep: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    After a sweep, did price reject immediately (manipulation -- the sweep
    was a stop-hunt, not real supply/demand), or did it keep going (real
    breakout, the level was genuinely absorbed)?

    Rejection = close back inside the level within 1-2 bars of the sweep,
    confirmed by a wick showing the rejection (opposite-side wick >= 40% of
    that candle's range).
    """
    if sweep is None:
        return {"confirmed": False, "type": None, "note": "No sweep to evaluate."}

    pos = sweep["swept_pos"]
    n = len(df)
    if pos >= n - 1:
        return {"confirmed": False, "type": "pending", "note": "Sweep just happened -- wait for the next bar to react."}

    look = df.iloc[pos:min(pos + 3, n)]
    level = sweep["level"]
    side = sweep["side"]  # BSL = swept a high, SSL = swept a low

    for i in range(len(look)):
        row = look.iloc[i]
        rng = max(row["High"] - row["Low"], 1e-9)
        if side == "BSL":
            wick = row["High"] - max(row["Open"], row["Close"])
            rejected = row["Close"] < level and (wick / rng) >= 0.35
        else:
            wick = min(row["Open"], row["Close"]) - row["Low"]
            rejected = row["Close"] > level and (wick / rng) >= 0.35
        if rejected:
            return {"confirmed": True, "type": "manipulation", "reject_index": look.index[i],
                    "note": f"Immediate rejection after the sweep (wick {wick/rng*100:.0f}% of range) — manipulation confirmed"}

    # No rejection within the check window -- price kept going past the level
    beyond = (df["Close"].iloc[min(pos + 2, n - 1)] > level) if side == "BSL" else (df["Close"].iloc[min(pos + 2, n - 1)] < level)
    if beyond:
        return {"confirmed": False, "type": "real_breakout",
                "note": "No rejection after the sweep — price kept going, treat as a real breakout, not manipulation."}
    return {"confirmed": False, "type": "undetermined", "note": "Reaction unclear yet — wait."}


# ============================================================================
# STAGE 7 -- Acceptance Test
# ============================================================================
def acceptance_test(df: pd.DataFrame, sweep: Optional[Dict[str, Any]],
                     manipulation: Dict[str, Any], hold_bars: int = 2) -> Dict[str, Any]:
    """
    This is the gate: never trade before this stage per the blueprint. After
    the sweep-and-reject, did price actually hold on the correct side of the
    swept level for `hold_bars` consecutive closes (bullish acceptance /
    bearish acceptance), or did it fall back through it (no acceptance yet)?
    """
    if sweep is None or not manipulation.get("confirmed"):
        return {"accepted": False, "side": None, "note": "No confirmed manipulation to test acceptance against."}

    level = sweep["level"]
    side = sweep["side"]
    reject_idx = manipulation.get("reject_index")
    if reject_idx is None:
        return {"accepted": False, "side": None, "note": "No rejection bar to measure acceptance from."}

    pos = df.index.get_loc(reject_idx)
    n = len(df)
    window = df.iloc[pos:min(pos + 1 + hold_bars, n)]
    if len(window) < 2:
        return {"accepted": False, "side": None, "note": "Not enough bars since rejection to confirm acceptance yet."}

    closes = window["Close"].values
    if side == "BSL":  # rejected a high -> want acceptance BELOW the level -> bearish acceptance
        holds = all(c < level for c in closes[1:])
        accept_side = "BEARISH" if holds else None
    else:  # rejected a low -> want acceptance ABOVE the level -> bullish acceptance
        holds = all(c > level for c in closes[1:])
        accept_side = "BULLISH" if holds else None

    if accept_side:
        return {"accepted": True, "side": accept_side,
                "note": f"{accept_side.title()} acceptance confirmed — {len(closes) - 1} consecutive closes holding the level"}
    return {"accepted": False, "side": None,
            "note": "Price has not held the level yet since rejection — wait for acceptance before entering"}


# ============================================================================
# STAGE 8 -- Entry Logic
# ============================================================================
def entry_logic(state_info, impulse, pullback, interpretation, sweep, manipulation, acceptance, df) -> Dict[str, Any]:
    """
    Two shapes, matching the blueprint's two worked examples:

    A) Reversal (distribution/accumulation path):
       Bullish trend -> impulse -> Rising Channel (distribution read) ->
       liquidity sweep -> manipulation confirmed -> bearish acceptance ->
       breaks channel low -> breaks horizontal level -> SELL.
       (Mirror for Falling Channel -> accumulation -> BUY.)

    B) Continuation (flag path):
       Bullish trend -> Bull Flag -> breakout -> retest -> BUY.
       (Mirror for Bear Flag -> SELL.)
    """
    reasons = []
    if impulse is None or pullback is None:
        return {"direction": "NEUTRAL", "reasons": ["No impulse/pullback -- nothing to trade yet."],
                "path": None, "score": 0}

    pattern = pullback["pattern"]
    reasons.append(f"Market state: {state_info['state']} ({state_info.get('reason', '')})")
    reasons.append(f"Impulse: {impulse['direction']} leg, {impulse['length_atr']}x ATR over {impulse['bars']} bars"
                    + (" (weak — below 3x ATR)" if impulse.get("weak") else ""))
    reasons.append(f"Pullback classified as {pattern.replace('_', ' ').title()} — {pullback['watch_for']}")

    # --- Path A: reversal via distribution/accumulation channel ---
    if pattern in ("RISING_CHANNEL", "FALLING_CHANNEL"):
        if sweep is None:
            return {"direction": "NEUTRAL", "path": "reversal", "score": 30,
                    "reasons": reasons + ["No liquidity sweep yet — continue waiting."]}
        reasons.append(sweep["note"])
        reasons.append(manipulation.get("note", ""))
        if not manipulation.get("confirmed"):
            return {"direction": "NEUTRAL", "path": "reversal", "score": 40, "reasons": reasons}
        reasons.append(acceptance.get("note", ""))
        if not acceptance.get("accepted"):
            return {"direction": "NEUTRAL", "path": "reversal", "score": 50, "reasons": reasons}

        want_dir = "SELL" if pattern == "RISING_CHANNEL" else "BUY"
        accept_matches = (acceptance["side"] == "BEARISH" and want_dir == "SELL") or \
                          (acceptance["side"] == "BULLISH" and want_dir == "BUY")
        if not accept_matches:
            reasons.append("Acceptance direction doesn't match the expected reversal — no trade.")
            return {"direction": "NEUTRAL", "path": "reversal", "score": 45, "reasons": reasons}

        # Final confirmation per blueprint: break the channel boundary, then
        # the horizontal (swept) level, in the acceptance direction.
        close = float(df["Close"].iloc[-1])
        channel_break = (close < pullback["channel_low"]) if want_dir == "SELL" else (close > pullback["channel_high"])
        horiz_break = (close < sweep["level"]) if want_dir == "SELL" else (close > sweep["level"])
        if channel_break and horiz_break:
            reasons.append("Broke channel boundary AND the swept horizontal level — reversal confirmed.")
            return {"direction": want_dir, "path": "reversal", "score": 85, "reasons": reasons}
        reasons.append("Acceptance confirmed but price hasn't broken the channel + horizontal level yet — wait.")
        return {"direction": "NEUTRAL", "path": "reversal", "score": 60, "reasons": reasons}

    # --- Path B: continuation via flag ---
    if pattern in ("BULL_FLAG", "BEAR_FLAG"):
        want_dir = "BUY" if pattern == "BULL_FLAG" else "SELL"
        close = float(df["Close"].iloc[-1])
        broke = (close > pullback["upper_now"]) if want_dir == "BUY" else (close < pullback["lower_now"])
        if not broke:
            reasons.append("Still inside the flag — no breakout yet.")
            return {"direction": "NEUTRAL", "path": "continuation", "score": 55, "reasons": reasons}
        reasons.append(f"Broke the flag boundary in the {want_dir} direction.")
        # Retest: did price come back to tag the broken boundary and hold,
        # or is this a fresh, unretested break (still tradeable, lower score)?
        boundary = pullback["upper_now"] if want_dir == "BUY" else pullback["lower_now"]
        recent = df.iloc[-4:]
        retested = ((recent["Low"] <= boundary).any() if want_dir == "BUY" else (recent["High"] >= boundary).any())
        if retested:
            reasons.append("Retest of the broken flag boundary held — continuation confirmed.")
            return {"direction": want_dir, "path": "continuation", "score": 82, "reasons": reasons}
        reasons.append("Fresh breakout, no retest yet — tradeable but lower conviction than a confirmed retest.")
        return {"direction": want_dir, "path": "continuation", "score": 62, "reasons": reasons}

    # --- Expansion: ride the impulse while range keeps widening ---
    if pattern == "EXPANSION":
        reasons.append("Range still expanding in the impulse direction — treat as live continuation.")
        return {"direction": impulse["direction"], "path": "expansion", "score": 66, "reasons": reasons}

    # Triangle / Rectangle / Compression: no directional read until it resolves
    reasons.append(f"{pattern.title()} — no directional edge until this resolves with acceptance.")
    return {"direction": "NEUTRAL", "path": "undecided", "score": 35, "reasons": reasons}


# ============================================================================
# STAGE 9 -- Trade Filter
# ============================================================================
def trade_filter(df: pd.DataFrame, state_info, impulse, pullback, entry_result) -> Dict[str, Any]:
    """Explicit reject list -- matches the blueprint's filter conditions."""
    reasons = []
    reject = False

    if state_info["state"] == "RANGE":
        reasons.append("Structure unclear (RANGE)"); reject = True
    if impulse is None:
        reasons.append("No impulse found"); reject = True
    elif impulse.get("weak"):
        reasons.append("Impulse too weak (< 3x ATR)"); reject = True
    if pullback is None:
        reasons.append("No classifiable pullback"); reject = True

    if pullback is not None:
        seg = df.iloc[pullback["seg_start"]:]
        if len(seg) >= 3:
            wick_ratio = float(((seg["High"] - seg[["Open", "Close"]].max(axis=1)) +
                                 (seg[["Open", "Close"]].min(axis=1) - seg["Low"])).sum() /
                                max((seg["High"] - seg["Low"]).sum(), 1e-9))
            if wick_ratio > 0.55:
                reasons.append("Too many wicks in the pullback (choppy, indecisive)"); reject = True
        # Trendline angle sanity + channel width sanity, same standard as
        # the Trendline-family module: an unrealistically steep line or an
        # overly wide channel isn't tradeable structure.
        avg_price = float(seg["Close"].mean()) or 1.0
        if abs(pullback["upper_slope"] * len(seg) / avg_price) > 0.06 or \
           abs(pullback["lower_slope"] * len(seg) / avg_price) > 0.06:
            reasons.append("Trendline angle too steep to be sustainable"); reject = True
        width_pct = (pullback["channel_high"] - pullback["channel_low"]) / avg_price
        if width_pct > 0.08:
            reasons.append("Channel too wide relative to price — low-conviction structure"); reject = True

    if entry_result.get("direction") == "NEUTRAL":
        reasons.append("No resolved direction yet (WAIT)"); reject = True

    return {"reject": reject, "reasons": reasons}


# ============================================================================
# STAGE 10 -- Trade Management
# ============================================================================
def compute_trade_management(entry: float, sl: float, direction: str, current_price: float,
                              recent_swings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Not indicator-based (no MA/RSI trail). Two rules only:
      1. Move to breakeven once price has moved 1R in your favour.
      2. After that, trail behind the most recent swing low (longs) /
         swing high (shorts) -- exit is a structure break, not an
         oscillator flipping.
    Returns the suggested new SL for the current bar; caller applies it to
    the live position.
    """
    risk = abs(entry - sl)
    if risk <= 0:
        return {"new_sl": sl, "at_be": False, "note": "Invalid risk distance."}

    if direction == "BUY":
        r_multiple = (current_price - entry) / risk
        if r_multiple < 1.0:
            return {"new_sl": sl, "at_be": False, "r_multiple": round(r_multiple, 2),
                    "note": "Below 1R -- keep original stop."}
        recent_lows = [s["price"] for s in recent_swings if s.get("type") == "low" and s["price"] < current_price]
        trail_sl = max([entry] + ([max(recent_lows)] if recent_lows else []))
        return {"new_sl": trail_sl, "at_be": True, "r_multiple": round(r_multiple, 2),
                "note": "At/above 1R -- stop moved to breakeven and trailing behind the latest swing low."}
    else:
        r_multiple = (entry - current_price) / risk
        if r_multiple < 1.0:
            return {"new_sl": sl, "at_be": False, "r_multiple": round(r_multiple, 2),
                    "note": "Below 1R -- keep original stop."}
        recent_highs = [s["price"] for s in recent_swings if s.get("type") == "high" and s["price"] > current_price]
        trail_sl = min([entry] + ([min(recent_highs)] if recent_highs else []))
        return {"new_sl": trail_sl, "at_be": True, "r_multiple": round(r_multiple, 2),
                "note": "At/above 1R -- stop moved to breakeven and trailing behind the latest swing high."}


# ============================================================================
# ORCHESTRATOR
# ============================================================================
def run_structure_engine(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or len(df) < 60:
        return {"error": "Not enough data for the Structure Engine (need 60+ bars)."}

    state_info = detect_market_state(df)
    impulse = detect_impulse(df, state_info)
    pullback = classify_pullback(df, impulse) if impulse else None
    interpretation = interpret_structure(state_info, impulse, pullback)
    sweep = liquidity_engine(df, state_info, pullback) if interpretation.get("expects_liquidity_sweep") or pullback else None
    manipulation = manipulation_detector(df, sweep)
    acceptance = acceptance_test(df, sweep, manipulation)
    entry = entry_logic(state_info, impulse, pullback, interpretation, sweep, manipulation, acceptance, df)
    filt = trade_filter(df, state_info, impulse, pullback, entry)

    valid = entry["direction"] in ("BUY", "SELL") and not filt["reject"]

    return {
        "state": state_info, "impulse": impulse, "pullback": pullback,
        "interpretation": interpretation, "sweep": sweep, "manipulation": manipulation,
        "acceptance": acceptance, "entry": entry, "filter": filt,
        "direction": entry["direction"] if valid else "NEUTRAL",
        "score": entry.get("score", 0),
        "valid": valid,
        "reasons": entry.get("reasons", []) + (filt["reasons"] if filt["reject"] else []),
        "df": df,
    }


def format_structure_report(result: Dict[str, Any], symbol: str) -> str:
    if result.get("error"):
        return result["error"]
    lines = [f"🏗 INSTITUTIONAL STRUCTURE ENGINE | {symbol}"]
    st = result["state"]
    lines.append(f"Stage 1 — State: {st['state']} ({st.get('reason', '')})")
    imp = result["impulse"]
    if imp:
        lines.append(f"Stage 2 — Impulse: {imp['direction']} · {imp['length_atr']}x ATR / {imp['bars']} bars"
                      + (" ⚠️ weak" if imp.get("weak") else ""))
    else:
        lines.append("Stage 2 — Impulse: none found")
    pb = result["pullback"]
    if pb:
        lines.append(f"Stage 3 — Pullback: {pb['pattern'].replace('_', ' ').title()}")
        lines.append(f"  {pb['watch_for']}")
    interp = result["interpretation"]
    lines.append(f"Stage 4 — Read: {interp['note']}")
    sw = result["sweep"]
    lines.append(f"Stage 5 — Liquidity: {sw['note'] if sw else 'No sweep detected yet'}")
    man = result["manipulation"]
    lines.append(f"Stage 6 — Manipulation: {man.get('note', '')}")
    acc = result["acceptance"]
    lines.append(f"Stage 7 — Acceptance: {acc.get('note', '')}")
    lines.append(f"Stage 8 — Entry: {result['direction']} (score {result['score']}/100, path={result['entry'].get('path')})")
    if result["filter"]["reject"]:
        lines.append("Stage 9 — Filter: ❌ REJECTED — " + "; ".join(result["filter"]["reasons"]))
    else:
        lines.append("Stage 9 — Filter: ✅ passed")
    lines.append(f"Verdict: {'TRADE — ' + result['direction'] if result['valid'] else 'WAIT'}")
    return "\n".join(lines)
