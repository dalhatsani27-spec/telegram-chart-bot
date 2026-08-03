"""
smc_zones.py
============
Smart Money Concepts zones:

  - Fair Value Gap (FVG) and Inverse FVG (IFVG / mitigated FVG that flips)
  - Order Block (OB) and Breaker Block (mitigated / inverted OB)
  - Inducement (IDM) zones — liquidity that traps traders before the true move

Rules used (ICT-style, practical for algo):

FVG (3-candle imbalance):
  Bullish FVG: candle[i-2].high < candle[i].low   (gap up)
  Bearish FVG: candle[i-2].low  > candle[i].high  (gap down)
  Minimum gap size: fraction of ATR (filters noise)

Order Block:
  Bullish OB: last down-close (or bearish) candle before a strong bullish
              displacement that creates BOS/CHoCH and preferably an FVG
  Bearish OB: last up-close candle before strong bearish displacement
  Body of displacement candle should be meaningful vs ATR

Inducement (IDM):
  Equal highs / equal lows (liquidity pools), or a minor internal swing
  that sits beyond a short-term range and is likely to be raided before
  the real expansion (classic trap for breakout traders).

Mitigation:
  Bullish zone mitigated when price trades down through the zone
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


def detect_fvgs(df, min_gap_atr=0.15, max_zones=8):
    """
    Detect unfilled and recently mitigated FVGs.
    Returns list of zone dicts (most recent first).
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
            # Mitigated when a later low trades into/through the gap
            later_lows = lows[start:]
            if len(later_lows) and later_lows.min() <= z["top"]:
                z["mitigated"] = True
                # Fully filled + closed below → inverse FVG potential
                later_closes = closes[start:]
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

    # Prefer active (unmitigated) zones, then recent
    active = [z for z in zones if not z["mitigated"]]
    inverted = [z for z in zones if z.get("inverted")]
    combined = active + [z for z in inverted if z not in active]
    combined.sort(key=lambda z: z["index"], reverse=True)
    return combined[:max_zones]


def detect_order_blocks(df, structure=None, min_body_atr=0.6, max_zones=6):
    """
    Detect Order Blocks near displacement.

    Simplified ICT rule:
      - Find candles with strong body (>= min_body_atr * ATR)
      - Bullish OB: bearish candle immediately before a strong bullish impulse
      - Bearish OB: bullish candle immediately before a strong bearish impulse
      - Prefer OBs that align with recent BOS/CHoCH if structure is provided
    """
    if df is None or len(df) < 10:
        return []

    atr = _atr_series(df)
    o = df['Open'].values
    h = df['High'].values
    l = df['Low'].values
    c = df['Close'].values
    n = len(df)
    zones = []

    for i in range(1, n - 1):
        a = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
        if a <= 0:
            continue
        body = abs(c[i] - o[i])
        if body < min_body_atr * a:
            continue

        # Strong bullish candle → look for prior bearish candle as bullish OB
        if c[i] > o[i]:
            prev = i - 1
            if c[prev] < o[prev]:  # prior was bearish
                zones.append({
                    "type": "OB",
                    "bias": "BULLISH",
                    "top": float(max(o[prev], c[prev])),
                    "bottom": float(min(o[prev], c[prev])),
                    "wick_top": float(h[prev]),
                    "wick_bottom": float(l[prev]),
                    "index": prev,
                    "impulse_index": i,
                    "mitigated": False,
                    "inverted": False,
                })

        # Strong bearish candle → prior bullish candle as bearish OB
        if c[i] < o[i]:
            prev = i - 1
            if c[prev] > o[prev]:
                zones.append({
                    "type": "OB",
                    "bias": "BEARISH",
                    "top": float(max(o[prev], c[prev])),
                    "bottom": float(min(o[prev], c[prev])),
                    "wick_top": float(h[prev]),
                    "wick_bottom": float(l[prev]),
                    "index": prev,
                    "impulse_index": i,
                    "mitigated": False,
                    "inverted": False,
                })

    # Mitigation → Breaker
    for z in zones:
        start = z["impulse_index"] + 1
        if start >= n:
            continue
        if z["bias"] == "BULLISH":
            # Mitigated when price trades fully through the OB (low below bottom)
            if lows_min := l[start:].min() if start < n else None:
                if lows_min < z["bottom"]:
                    z["mitigated"] = True
                    z["inverted"] = True
                    z["type"] = "BREAKER"  # inverted order block
        else:
            if highs_max := h[start:].max() if start < n else None:
                if highs_max > z["top"]:
                    z["mitigated"] = True
                    z["inverted"] = True
                    z["type"] = "BREAKER"

    # Optional: boost zones near structure event
    if structure and structure.get("event_price") is not None:
        ep = structure["event_price"]
        for z in zones:
            if abs(z["mid"] if "mid" in z else (z["top"] + z["bottom"]) / 2 - ep) / max(ep, 1e-9) < 0.003:
                z["near_structure"] = True

    for z in zones:
        z["mid"] = (z["top"] + z["bottom"]) / 2.0

    active = [z for z in zones if not z["mitigated"]]
    breakers = [z for z in zones if z.get("inverted")]
    combined = active + [z for z in breakers if z not in active]
    combined.sort(key=lambda z: z["index"], reverse=True)
    return combined[:max_zones]


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


def summarise_smc_zones(fvgs, obs, max_show=5, inducements=None):
    """Short text lines — always show MITIGATED vs UNMITIGATED."""
    lines = []
    for z in fvgs[:max_show]:
        tag = z["type"]
        if z.get("mitigated"):
            status = "MITIGATED→IFVG" if z.get("inverted") else "MITIGATED"
        else:
            status = "UNMITIGATED"
        lines.append(
            f"  {tag} {z['bias']} [{status}]: {z['bottom']:.5f} – {z['top']:.5f}"
        )
    for z in obs[:max_show]:
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
    return lines
