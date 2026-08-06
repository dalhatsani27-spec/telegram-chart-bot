"""
smc_zones.py
============
Smart Money Concepts zones:

  - Fair Value Gap (FVG) and Inverse FVG (IFVG / mitigated FVG that flips)
  - Order Block (OB) and Breaker Block (mitigated / inverted OB)
  - Inducement (IDM) zones — liquidity that traps traders before the true move
  - Rally-Base-Rally / Drop-Base-Drop / Rally-Base-Drop / Drop-Base-Rally
    (classic Supply & Demand base patterns)

Rules used (ICT-style + Sam Seiden style, practical for algo):

FVG (3-candle imbalance):
  Bullish FVG: candle[i-2].high < candle[i].low   (gap up)
  Bearish FVG: candle[i-2].low  > candle[i].high  (gap down)
  Minimum gap size: fraction of ATR (filters noise)

Order Block:
  Bullish OB: last down-close candle before a strong bullish displacement
              that creates BOS/CHoCH and preferably an FVG
  Bearish OB: last up-close candle before strong bearish displacement
  Body of displacement candle should be meaningful vs ATR

Inducement (IDM):
  Equal highs / equal lows (liquidity pools), or a minor internal swing
  that sits beyond a short-term range and is likely to be raided before
  the real expansion.

Base Patterns (RBR / DBD / RBD / DBR):
  Rally = strong bullish leg   Drop = strong bearish leg
  Base  = short consolidation (small range / overlapping candles)
  RBR / DBD = continuation zones (weaker)
  RBD / DBR = reversal zones (stronger)

Mitigation:
  Bullish zone mitigated when price body-closes through the far side
  After mitigation, zone can become Breaker / IFVG
"""


import numpy as np
import pandas as pd


def _atr_series(df, period=14):
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        return df['ATR']
    prev_c = df['Close'].shift(1)
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum((df['High'] - prev_c).abs(), (df['Low'] - prev_c).abs()))
    return tr.rolling(period).mean()


def detect_fvgs(df, min_gap_atr=0.18, max_zones=5):
    """
    Detect unfilled and recently mitigated FVGs.

    LOCKED RULE (clean chart):
      Prefer unmitigated FVGs. Keep only a small number of the most recent,
      meaningful gaps. Inverted FVGs (IFVG) are kept only when fully violated.
    """
    if df is None or len(df) < 5:
        return []

    atr = _atr_series(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    n = len(df)
    zones = []

    for i in range(2, n):
        a_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
        if a_atr <= 0:
            continue

        # Bullish FVG: gap between candle i-2 high and candle i low
        gap_low = float(highs[i - 2])
        gap_high = float(lows[i])
        if gap_high > gap_low:
            gap_size = gap_high - gap_low
            if gap_size >= min_gap_atr * a_atr:
                zones.append({
                    "type": "FVG",
                    "bias": "BULLISH",
                    "top": gap_high,
                    "bottom": gap_low,
                    "mid": (gap_high + gap_low) / 2.0,
                    "index": i,
                    "mitigated": False,
                    "inverted": False,
                })

        # Bearish FVG: gap between candle i-2 low and candle i high
        gap_high_b = float(lows[i - 2])
        gap_low_b = float(highs[i])
        if gap_high_b > gap_low_b:
            gap_size = gap_high_b - gap_low_b
            if gap_size >= min_gap_atr * a_atr:
                zones.append({
                    "type": "FVG",
                    "bias": "BEARISH",
                    "top": gap_high_b,
                    "bottom": gap_low_b,
                    "mid": (gap_high_b + gap_low_b) / 2.0,
                    "index": i,
                    "mitigated": False,
                    "inverted": False,
                })

    # Mark mitigation / inversion using later price action
    for z in zones:
        start = z["index"] + 1
        if start >= n:
            continue
        if z["bias"] == "BULLISH":
            later_lows = lows[start:]
            if len(later_lows) and later_lows.min() <= z["top"]:
                z["mitigated"] = True
                later_closes = closes[start:]
                # Full body close below the gap → IFVG
                if len(later_closes) and later_closes.min() < z["bottom"]:
                    z["inverted"] = True
                    z["type"] = "IFVG"
        else:
            later_highs = highs[start:]
            if len(later_highs) and later_highs.max() >= z["bottom"]:
                z["mitigated"] = True
                later_closes = closes[start:]
                if len(later_closes) and later_closes.max() > z["top"]:
                    z["inverted"] = True
                    z["type"] = "IFVG"

    # Quality score + prefer active (unmitigated) then recent IFVGs
    for z in zones:
        q = 40
        if not z.get("mitigated"):
            q += 30
        age = n - z.get("index", n)
        if age < 20:
            q += 15
        elif age < 50:
            q += 8
        gap_size = abs(z["top"] - z["bottom"])
        # Larger gaps relative to recent range get a small boost (more meaningful imbalance)
        q = min(100, q)
        z["quality"] = q
        z["mid"] = (z["top"] + z["bottom"]) / 2.0

    active = [z for z in zones if not z["mitigated"]]
    inverted = [z for z in zones if z.get("inverted")]
    active.sort(key=lambda z: (-z.get("quality", 0), -z["index"]))
    inverted.sort(key=lambda z: (-z.get("quality", 0), -z["index"]))
    combined = active + [z for z in inverted if z not in active]
    return combined[:max_zones]



def detect_order_blocks(df, structure=None, min_body_atr=0.45, max_zones=4, require_bos=True):
    """
    ICT-correct Order Blocks tied to ZigZag swings + displacement.

    LOCKED RULE (clean chart):
      Only map an OB when the displacement leg that created it also produced
      a BOS or CHoCH/MSS. Isolated OBs without structure break are discarded.

    Valid sequence:
      1. Swing high/low (ZigZag)
      2. Optional liquidity sweep of that swing
      3. Displacement that breaks structure (BOS/CHoCH)  ← REQUIRED when require_bos=True
      4. OB = last opposing candle BEFORE the displacement
      5. FVG often forms inside that same displacement (linked by impulse_index)
    """
    if df is None or len(df) < 20:
        return []

    from market_structure import zigzag_swings

    atr = _atr_series(df)
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    n = len(df)

    pivots = zigzag_swings(df, depth=4, deviation_atr=0.30)
    if len(pivots) < 3:
        pivots = []

    zones = []

    for k in range(1, len(pivots)):
        cur = pivots[k]
        prev = pivots[k - 1]
        leg_start = prev["index"]
        leg_end = cur["index"]
        if leg_end - leg_start < 2:
            continue

        # Did this leg break a prior swing of the same type? (BOS / CHoCH quality)
        broken = None
        for older in reversed(pivots[: k - 1]):
            if older["type"] == cur["type"]:
                if cur["type"] == "high" and cur["price"] > older["price"]:
                    broken = older
                elif cur["type"] == "low" and cur["price"] < older["price"]:
                    broken = older
                break

        # LOCKED: if require_bos is True, discard legs that did not break structure
        if broken is None:
            if require_bos:
                continue
            # Fallback (only when require_bos=False): allow very strong displacement
            leg_range = abs(cur["price"] - prev["price"])
            a_mid = float(atr.iloc[min(leg_end, n - 1)]) if not np.isnan(atr.iloc[min(leg_end, n - 1)]) else 0
            if a_mid <= 0 or leg_range < 1.5 * a_mid:
                continue

        # Find last opposing candle before the impulsive part of the leg
        if cur["type"] == "high":
            # Bullish displacement → last bearish candle = bullish OB
            ob_idx = None
            for j in range(leg_end - 1, leg_start - 1, -1):
                if j < 0:
                    break
                if c[j] < o[j]:
                    a = float(atr.iloc[j]) if not np.isnan(atr.iloc[j]) else 0
                    body = abs(c[j] - o[j])
                    if a > 0 and body >= min_body_atr * a * 0.5:
                        ob_idx = j
                        break
            if ob_idx is None:
                continue
            zones.append({
                "type": "OB",
                "bias": "BULLISH",
                "top": float(max(o[ob_idx], c[ob_idx])),
                "bottom": float(min(o[ob_idx], c[ob_idx])),
                "wick_top": float(h[ob_idx]),
                "wick_bottom": float(l[ob_idx]),
                "index": ob_idx,
                "impulse_index": leg_end,
                "swing_broken": broken["price"] if broken else None,
                "bos": broken is not None,
                "mitigated": False,
                "inverted": False,
            })
        else:
            # Bearish displacement → last bullish candle = bearish OB
            ob_idx = None
            for j in range(leg_end - 1, leg_start - 1, -1):
                if j < 0:
                    break
                if c[j] > o[j]:
                    a = float(atr.iloc[j]) if not np.isnan(atr.iloc[j]) else 0
                    body = abs(c[j] - o[j])
                    if a > 0 and body >= min_body_atr * a * 0.5:
                        ob_idx = j
                        break
            if ob_idx is None:
                continue
            zones.append({
                "type": "OB",
                "bias": "BEARISH",
                "top": float(max(o[ob_idx], c[ob_idx])),
                "bottom": float(min(o[ob_idx], c[ob_idx])),
                "wick_top": float(h[ob_idx]),
                "wick_bottom": float(l[ob_idx]),
                "index": ob_idx,
                "impulse_index": leg_end,
                "swing_broken": broken["price"] if broken else None,
                "bos": broken is not None,
                "mitigated": False,
                "inverted": False,
            })

    # Prefer structure-breaking OBs, then most recent
    zones.sort(key=lambda z: (not z.get("bos", False), -z["index"]))

    # Mitigation → Breaker (body-close test — ICT standard)
    # Wick through alone is a sweep, not full mitigation.
    # Only a candle BODY closing beyond the far side flips the zone to Breaker.
    for z in zones:
        start = z["impulse_index"] + 1
        if start >= n:
            continue
        if z["bias"] == "BULLISH":
            # Far side = bottom of the bullish OB. Body close below it = failed.
            later_closes = c[start:]
            if len(later_closes) and later_closes.min() < z["bottom"]:
                z["mitigated"] = True
                z["inverted"] = True
                z["type"] = "BREAKER"
        else:
            # Far side = top of the bearish OB. Body close above it = failed.
            later_closes = c[start:]
            if len(later_closes) and later_closes.max() > z["top"]:
                z["mitigated"] = True
                z["inverted"] = True
                z["type"] = "BREAKER"

    for z in zones:
        z["mid"] = (z["top"] + z["bottom"]) / 2.0
        # Simple quality score for downstream filtering / ranking
        # Higher = fresher + structure-breaking + unmitigated
        q = 50
        if z.get("bos"):
            q += 25
        if not z.get("mitigated"):
            q += 15
        # Recency bonus (more recent impulse = higher)
        age = n - z.get("impulse_index", n)
        if age < 15:
            q += 10
        elif age < 40:
            q += 5
        z["quality"] = min(100, q)

    # Prefer high-quality unmitigated first, then recent breakers
    active = [z for z in zones if not z["mitigated"]]
    breakers = [z for z in zones if z.get("inverted") and z not in active]
    active.sort(key=lambda z: (-z.get("quality", 0), -z["index"]))
    breakers.sort(key=lambda z: (-z.get("quality", 0), -z["index"]))
    combined = active + breakers

    # Deduplicate near-identical zones
    cleaned = []
    for z in combined:
        if any(abs(z["mid"] - c0["mid"]) / max(abs(z["mid"]), 1e-9) < 0.0008 for c0 in cleaned):
            continue
        cleaned.append(z)

    return cleaned[:max_zones]



def build_bos_events(df, max_events=8):
    """
    Build BOS / CHoCH events for chart drawing (dotted lines at broken levels).
    Uses ZigZag pivots.
    Returns list of {index, price, type: 'BOS'|'CHoCH', bias: 'BULLISH'|'BEARISH'}
    """
    if df is None or len(df) < 20:
        return []
    from market_structure import zigzag_swings

    pivots = zigzag_swings(df, depth=4, deviation_atr=0.30)
    if len(pivots) < 4:
        return []

    events = []
    bias = "NEUTRAL"
    for k in range(1, len(pivots)):
        cur = pivots[k]
        # Find previous same-type pivot
        older = None
        for j in range(k - 1, -1, -1):
            if pivots[j]["type"] == cur["type"]:
                older = pivots[j]
                break
        if older is None:
            continue

        if cur["type"] == "high" and cur["price"] > older["price"]:
            if bias == "BEARISH":
                events.append({
                    "index": cur["index"],
                    "price": older["price"],
                    "type": "CHoCH",
                    "bias": "BULLISH",
                })
                bias = "BULLISH"
            else:
                events.append({
                    "index": cur["index"],
                    "price": older["price"],
                    "type": "BOS",
                    "bias": "BULLISH",
                })
                bias = "BULLISH"
        elif cur["type"] == "low" and cur["price"] < older["price"]:
            if bias == "BULLISH":
                events.append({
                    "index": cur["index"],
                    "price": older["price"],
                    "type": "CHoCH",
                    "bias": "BEARISH",
                })
                bias = "BEARISH"
            else:
                events.append({
                    "index": cur["index"],
                    "price": older["price"],
                    "type": "BOS",
                    "bias": "BEARISH",
                })
                bias = "BEARISH"

    return events[-max_events:]


def detect_inducement_zones(df, equal_tol=0.0008, max_zones=8):
    """
    Inducement (IDM) = internal liquidity that sits BEFORE an extreme zone.

    Matches the OB + IDM + Confirmation model:
      - Equal highs / equal lows (classic liquidity pools)
      - Internal swing highs/lows (minor structure that traps breakout traders)
      - Explicit mitigated (swept) vs unmitigated status

    Buy-side IDM  → often raided before a sell into extreme bearish OB
    Sell-side IDM → often raided before a buy into extreme bullish OB
    """
    if df is None or len(df) < 20:
        return []

    from market_structure import find_swings

    swings = find_swings(df, left=2, right=2)
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    zones = []
    atr = _atr_series(df)
    last_atr = float(atr.iloc[-1]) if len(atr) and not np.isnan(atr.iloc[-1]) else 0.0
    highs_arr = df["High"].values
    lows_arr = df["Low"].values
    n = len(df)

    def _mitigated_buy_side(top, from_idx):
        """Buy-side IDM mitigated when a later high trades through it."""
        if from_idx + 1 >= n:
            return False
        return float(highs_arr[from_idx + 1:].max()) > top

    def _mitigated_sell_side(bottom, from_idx):
        if from_idx + 1 >= n:
            return False
        return float(lows_arr[from_idx + 1:].min()) < bottom

    # 1) Equal highs → buy-side inducement
    for i in range(1, len(highs)):
        a, b = highs[i - 1], highs[i]
        mid = (a["price"] + b["price"]) / 2.0
        if mid <= 0:
            continue
        if abs(a["price"] - b["price"]) / mid <= equal_tol:
            top = max(a["price"], b["price"])
            bottom = min(a["price"], b["price"])
            pad = last_atr * 0.05 if last_atr > 0 else mid * 0.0002
            mit = _mitigated_buy_side(top, b["index"])
            zones.append({
                "type": "IDM",
                "bias": "BUY_SIDE",
                "side": "buy_side_liquidity",
                "top": top + pad,
                "bottom": bottom - pad,
                "mid": mid,
                "index": b["index"],
                "mitigated": mit,
                "swept": mit,
                "note": "Equal highs IDM — internal buy-side liquidity before extreme",
            })

    # 2) Equal lows → sell-side inducement
    for i in range(1, len(lows)):
        a, b = lows[i - 1], lows[i]
        mid = (a["price"] + b["price"]) / 2.0
        if mid <= 0:
            continue
        if abs(a["price"] - b["price"]) / mid <= equal_tol:
            top = max(a["price"], b["price"])
            bottom = min(a["price"], b["price"])
            pad = last_atr * 0.05 if last_atr > 0 else mid * 0.0002
            mit = _mitigated_sell_side(bottom, b["index"])
            zones.append({
                "type": "IDM",
                "bias": "SELL_SIDE",
                "side": "sell_side_liquidity",
                "top": top + pad,
                "bottom": bottom - pad,
                "mid": mid,
                "index": b["index"],
                "mitigated": mit,
                "swept": mit,
                "note": "Equal lows IDM — internal sell-side liquidity before extreme",
            })

    # 3) Internal swing highs/lows as single-point inducement (not only equals)
    #    Take recent intermediate swings (skip the absolute extreme high/low)
    if len(highs) >= 3:
        sorted_h = sorted(highs, key=lambda s: s["price"], reverse=True)
        extreme_h = sorted_h[0]
        for sw in sorted_h[1:4]:  # next internal highs
            if sw["index"] >= extreme_h["index"]:
                continue  # only IDM that formed BEFORE the extreme
            pad = last_atr * 0.08 if last_atr > 0 else sw["price"] * 0.0003
            mit = _mitigated_buy_side(sw["price"], sw["index"])
            zones.append({
                "type": "IDM",
                "bias": "BUY_SIDE",
                "side": "buy_side_liquidity",
                "top": sw["price"] + pad,
                "bottom": sw["price"] - pad,
                "mid": sw["price"],
                "index": sw["index"],
                "mitigated": mit,
                "swept": mit,
                "note": "Internal swing high IDM — before extreme high/OB",
                "before_extreme": True,
            })

    if len(lows) >= 3:
        sorted_l = sorted(lows, key=lambda s: s["price"])
        extreme_l = sorted_l[0]
        for sw in sorted_l[1:4]:
            if sw["index"] >= extreme_l["index"]:
                continue
            pad = last_atr * 0.08 if last_atr > 0 else sw["price"] * 0.0003
            mit = _mitigated_sell_side(sw["price"], sw["index"])
            zones.append({
                "type": "IDM",
                "bias": "SELL_SIDE",
                "side": "sell_side_liquidity",
                "top": sw["price"] + pad,
                "bottom": sw["price"] - pad,
                "mid": sw["price"],
                "index": sw["index"],
                "mitigated": mit,
                "swept": mit,
                "note": "Internal swing low IDM — before extreme low/OB",
                "before_extreme": True,
            })

    # Prefer unmitigated, then before_extreme, then recent
    zones.sort(key=lambda z: (
        z.get("mitigated", False),
        not z.get("before_extreme", False),
        -z["index"],
    ))
    cleaned = []
    for z in zones:
        if any(abs(z["mid"] - c["mid"]) / max(abs(z["mid"]), 1e-9) < equal_tol for c in cleaned):
            continue
        cleaned.append(z)
    return cleaned[:max_zones]


def detect_base_zones(df, max_zones=6, min_leg_atr=0.8, max_base_bars=6):
    """
    Detect classic Supply & Demand base patterns:

      RBR  Rally-Base-Rally   → Demand (continuation)
      DBD  Drop-Base-Drop     → Supply (continuation)
      RBD  Rally-Base-Drop    → Supply (reversal)   [stronger]
      DBR  Drop-Base-Rally    → Demand (reversal)   [stronger]

    A "Rally" / "Drop" is a directional leg whose net body move is meaningful
    vs ATR. A "Base" is a short consolidation (small range relative to the
    preceding leg) that sits between two legs.

    Returns list of zone dicts with type, bias, top, bottom, quality, etc.
    """
    if df is None or len(df) < 20:
        return []

    atr = _atr_series(df)
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    n = len(df)

    # Classify each bar as rally / drop / base-ish using body direction + size
    body = c - o
    rng = np.maximum(h - l, 1e-12)
    body_ratio = np.abs(body) / rng

    def _leg_strength(start, end, direction):
        """Net directional move of a candidate leg in ATR units."""
        if end <= start:
            return 0.0
        a = float(atr.iloc[min(end, n - 1)]) if not np.isnan(atr.iloc[min(end, n - 1)]) else 0.0
        if a <= 0:
            return 0.0
        if direction == "up":
            move = c[end] - o[start]
        else:
            move = o[start] - c[end]
        return move / a

    zones = []
    i = 4
    while i < n - 4:
        # Look for a short base window
        for base_len in range(1, max_base_bars + 1):
            b0 = i
            b1 = i + base_len - 1
            if b1 >= n - 3:
                break

            base_high = float(h[b0:b1 + 1].max())
            base_low = float(l[b0:b1 + 1].min())
            base_range = base_high - base_low
            a_mid = float(atr.iloc[b1]) if not np.isnan(atr.iloc[b1]) else 0.0
            if a_mid <= 0 or base_range > a_mid * 1.4:
                continue  # base too wide

            # Left leg (before base)
            left_end = b0 - 1
            left_start = max(0, left_end - 8)
            # Right leg (after base)
            right_start = b1 + 1
            right_end = min(n - 1, right_start + 8)

            if left_end - left_start < 1 or right_end - right_start < 1:
                continue

            left_up = _leg_strength(left_start, left_end, "up")
            left_dn = _leg_strength(left_start, left_end, "down")
            right_up = _leg_strength(right_start, right_end, "up")
            right_dn = _leg_strength(right_start, right_end, "down")

            pattern = None
            bias = None
            strength = "continuation"

            # RBR: left up + right up
            if left_up >= min_leg_atr and right_up >= min_leg_atr * 0.7 and left_up > left_dn:
                pattern, bias, strength = "RBR", "BULLISH", "continuation"
            # DBD: left down + right down
            elif left_dn >= min_leg_atr and right_dn >= min_leg_atr * 0.7 and left_dn > left_up:
                pattern, bias, strength = "DBD", "BEARISH", "continuation"
            # RBD: left up + right down (reversal supply)
            elif left_up >= min_leg_atr and right_dn >= min_leg_atr * 0.8 and left_up > left_dn:
                pattern, bias, strength = "RBD", "BEARISH", "reversal"
            # DBR: left down + right up (reversal demand)
            elif left_dn >= min_leg_atr and right_up >= min_leg_atr * 0.8 and left_dn > left_up:
                pattern, bias, strength = "DBR", "BULLISH", "reversal"

            if pattern is None:
                continue

            # Quality: reversal patterns score higher; tighter base + stronger legs better
            q = 55 if strength == "reversal" else 40
            leg_score = max(left_up, left_dn) + max(right_up, right_dn)
            q += min(25, int(leg_score * 6))
            if base_range < a_mid * 0.6:
                q += 10  # tight base
            q = min(100, q)

            zones.append({
                "type": pattern,
                "bias": bias,
                "strength": strength,
                "top": base_high,
                "bottom": base_low,
                "mid": (base_high + base_low) / 2.0,
                "index": b0,
                "base_end": b1,
                "mitigated": False,
                "quality": q,
                "note": f"{pattern} {strength} base",
            })
            i = b1 + 2  # skip forward
            break
        else:
            i += 1

    # Mitigation: body close beyond far side after the right leg starts
    for z in zones:
        start = z.get("base_end", z["index"]) + 3
        if start >= n:
            continue
        if z["bias"] == "BULLISH":
            if c[start:].min() < z["bottom"]:
                z["mitigated"] = True
        else:
            if c[start:].max() > z["top"]:
                z["mitigated"] = True

    # Prefer unmitigated, high quality, recent
    zones.sort(key=lambda z: (z.get("mitigated", False), -z.get("quality", 0), -z["index"]))
    cleaned = []
    for z in zones:
        if any(abs(z["mid"] - c0["mid"]) / max(abs(z["mid"]), 1e-9) < 0.001 for c0 in cleaned):
            continue
        cleaned.append(z)
    return cleaned[:max_zones]


def pair_idm_with_extreme_ob(inducements, order_blocks):
    """
    Link IDM that sits in front of an extreme unmitigated OB.
    Returns list of setup dicts: {idm, ob, sequence_note}
    """
    setups = []
    active_obs = [o for o in (order_blocks or []) if not o.get("mitigated")]
    for idm in inducements or []:
        for ob in active_obs:
            # Buy-side IDM in front of bearish OB (IDM below the OB)
            if idm["bias"] == "BUY_SIDE" and ob["bias"] == "BEARISH":
                if idm["mid"] < ob["bottom"] and idm["index"] < ob.get("index", 10**9):
                    setups.append({
                        "idm": idm,
                        "ob": ob,
                        "direction": "SELL",
                        "sequence": "IDM (buy-side) → extreme bearish OB — wait sweep of IDM then confirmation into OB",
                    })
            # Sell-side IDM in front of bullish OB (IDM above the OB)
            if idm["bias"] == "SELL_SIDE" and ob["bias"] == "BULLISH":

                if idm["mid"] > ob["top"] and idm["index"] < ob.get("index", 10**9):
                    setups.append({
                        "idm": idm,
                        "ob": ob,
                        "direction": "BUY",
                        "sequence": "IDM (sell-side) → extreme bullish OB — wait sweep of IDM then confirmation into OB",
                    })
    return setups[:4]


def summarise_smc_zones(fvgs, obs, max_show=5, inducements=None, base_zones=None):
    """Short text lines — always show MITIGATED vs UNMITIGATED."""
    lines = []
    for z in (fvgs or [])[:max_show]:
        tag = z["type"]
        if z.get("mitigated"):
            status = "MITIGATED→IFVG" if z.get("inverted") else "MITIGATED"
        else:
            status = "UNMITIGATED"
        lines.append(
            f"  {tag} {z['bias']} [{status}]: {z['bottom']:.5f} – {z['top']:.5f}"
        )
    for z in (obs or [])[:max_show]:
        tag = z["type"]
        if z.get("mitigated"):
            status = "MITIGATED→BREAKER" if z.get("inverted") else "MITIGATED"
        else:
            status = "UNMITIGATED"
        lines.append(
            f"  {tag} {z['bias']} [{status}]: {z['bottom']:.5f} – {z['top']:.5f}"
        )
    if inducements:
        for z in inducements[:max_show]:
            status = "MITIGATED (swept)" if z.get("mitigated") or z.get("swept") else "UNMITIGATED"
            tag_extra = " before-extreme" if z.get("before_extreme") else ""
            lines.append(
                f"  IDM {z['bias']} [{status}]{tag_extra}: {z['bottom']:.5f} – {z['top']:.5f}"
            )
    if base_zones:
        for z in base_zones[:max_show]:
            status = "MITIGATED" if z.get("mitigated") else "UNMITIGATED"
            strength = z.get("strength", "")
            lines.append(
                f"  {z['type']} {z['bias']} [{status}] ({strength}): "
                f"{z['bottom']:.5f} – {z['top']:.5f}"
            )
    return lines

