"""
smc_engine.py
=============
Smart Money Concepts DETECTORS ONLY. No report formatting, no chart drawing,
no trade-model math -- that all lives in smc_strategy.py / chart_engine.py.

This file deliberately does NOT re-implement structure, order blocks, or
premium/discount -- those already exist and are reused as-is:
  - analyse_structure()          -> market_analysis.py   (BOS / CHoCH / MSS)
  - detect_order_blocks()        -> market_analysis.py   (OB + inducement)
  - fib_discount_premium_zone()  -> execution_engine.py  (premium/discount)

What's added here (did not exist anywhere in the bot before):
  1. detect_liquidity_pools()  -- buy-side / sell-side resting liquidity,
                                   with SWEPT vs INTACT status.
  2. detect_fair_value_gaps()  -- 3-candle imbalance (FVG), with
                                   FRESH / PARTIALLY_FILLED / FILLED status.
  3. select_smc_zone()         -- picks the single best OB/FVG confluence
                                   zone to report as "the" SMC zone, the way
                                   the sample report shows one zone.

Runs entirely on the SINGLE selected timeframe passed in -- no fixed
4H/1H/30M cascade. Callers may separately fetch topdown_engine.get_topdown_bias()
for optional HTF context, same as the Trendline/OTE strategies do.
"""

from typing import Any, Dict, List, Optional

from market_analysis import find_swings, analyse_structure, detect_order_blocks


# ============================================================
# 1. LIQUIDITY POOLS -- resting stops above swing highs / below swing lows.
#    A pool is SWEPT when a wick trades beyond it and price closes back on
#    the origin side (the classic stop-hunt / liquidity-grab shape from the
#    "BEARISH ENTRY" reference image: SWEEP -> reversal).
# ============================================================

def detect_liquidity_pools(df, left: int = 3, right: int = 3, lookback: int = 100,
                            sweep_lookahead: int = 15) -> Dict[str, Any]:
    """
    Returns:
      {
        "buy_side":  {"level": float, "index": int, "status": "SWEPT"|"INTACT", "swept_index": int|None},
        "sell_side": {"level": float, "index": int, "status": "SWEPT"|"INTACT", "swept_index": int|None},
        "pools": [ ... all recent pools, most recent first ... ]
      }

    buy_side  = nearest unbroken swing HIGH  (stop-liquidity for shorts, resting buy orders)
    sell_side = nearest unbroken swing LOW   (stop-liquidity for longs, resting sell orders)
    """
    empty = {"buy_side": None, "sell_side": None, "pools": []}
    if df is None or len(df) < left + right + 10:
        return empty

    n = len(df)
    start = max(0, n - lookback)
    swings = [s for s in find_swings(df, left=left, right=right) if s["index"] >= start]
    if not swings:
        return empty

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    pools = []
    for sw in swings:
        idx = sw["index"]
        level = sw["price"]
        status = "INTACT"
        swept_index = None
        end = min(idx + sweep_lookahead, n)
        if sw["type"] == "high":
            for k in range(idx + 1, end):
                if highs[k] > level:
                    # wick beyond the pool
                    if closes[k] < level:
                        status = "SWEPT"
                        swept_index = k
                    else:
                        status = "BROKEN"  # closed through -- no longer a pool, it's structure
                        swept_index = k
                    break
            pools.append({"type": "buy_side", "level": level, "index": idx,
                          "status": status, "swept_index": swept_index})
        else:
            for k in range(idx + 1, end):
                if lows[k] < level:
                    if closes[k] > level:
                        status = "SWEPT"
                        swept_index = k
                    else:
                        status = "BROKEN"
                        swept_index = k
                    break
            pools.append({"type": "sell_side", "level": level, "index": idx,
                          "status": status, "swept_index": swept_index})

    pools.sort(key=lambda p: -p["index"])

    def _nearest(kind):
        for p in pools:
            if p["type"] == kind and p["status"] != "BROKEN":
                return p
        return None

    return {
        "buy_side": _nearest("buy_side"),
        "sell_side": _nearest("sell_side"),
        "pools": pools[:12],
    }


# ============================================================
# 2. FAIR VALUE GAPS -- 3-candle imbalance left by a displacement move.
#    Bullish FVG: low[2] > high[0]  (gap between candle 1 and candle 3)
#    Bearish FVG: high[2] < low[0]
# ============================================================

def detect_fair_value_gaps(df, lookback: int = 100, min_gap_atr: float = 0.15,
                            max_per_side: int = 3) -> List[Dict[str, Any]]:
    """
    Returns list of dicts, most recent first:
      {
        "type": "bullish" | "bearish",
        "top": float, "bottom": float,
        "index": int,          # index of the 3rd (confirming) candle
        "status": "FRESH" | "PARTIALLY_FILLED" | "FILLED",
        "fill_pct": 0-100,
      }
    """
    if df is None or len(df) < 10:
        return []

    n = len(df)
    start = max(2, n - lookback)
    highs = df["High"].values
    lows = df["Low"].values
    atr = df["ATR"].values if "ATR" in df.columns else (df["High"] - df["Low"]).rolling(14, min_periods=1).mean().values

    gaps = []
    for i in range(start, n):
        a = float(atr[i]) if i < len(atr) and atr[i] > 0 else None
        # Bullish FVG: candle i-2 high < candle i low
        if lows[i] > highs[i - 2]:
            gap = lows[i] - highs[i - 2]
            if a is None or gap >= a * min_gap_atr:
                gaps.append({"type": "bullish", "top": float(lows[i]), "bottom": float(highs[i - 2]), "index": i})
        # Bearish FVG: candle i-2 low > candle i high
        if highs[i] < lows[i - 2]:
            gap = lows[i - 2] - highs[i]
            if a is None or gap >= a * min_gap_atr:
                gaps.append({"type": "bearish", "top": float(lows[i - 2]), "bottom": float(highs[i]), "index": i})

    # Fill status: how much of the gap has price traded back into since it formed
    for g in gaps:
        top, bottom = g["top"], g["bottom"]
        width = max(top - bottom, 1e-9)
        deepest = None
        for k in range(g["index"] + 1, n):
            if g["type"] == "bullish":
                # price returning DOWN into a bullish FVG fills it
                intrusion = top - lows[k]
            else:
                intrusion = highs[k] - bottom
            intrusion = max(0.0, min(width, intrusion))
            if deepest is None or intrusion > deepest:
                deepest = intrusion
        fill_pct = int(round((deepest or 0.0) / width * 100))
        if fill_pct <= 0:
            status = "FRESH"
        elif fill_pct >= 90:
            status = "FILLED"
        else:
            status = "PARTIALLY_FILLED"
        g["fill_pct"] = fill_pct
        g["status"] = status

    bullish = sorted([g for g in gaps if g["type"] == "bullish"], key=lambda g: -g["index"])[:max_per_side]
    bearish = sorted([g for g in gaps if g["type"] == "bearish"], key=lambda g: -g["index"])[:max_per_side]
    return sorted(bullish + bearish, key=lambda g: -g["index"])


# ============================================================
# 3. ZONE SELECTION -- pick the single OB/FVG confluence zone to report,
#    matching the sample's "ZONE" block (one Bearish OB + overlapping FVG).
# ============================================================

def select_smc_zone(order_blocks: List[Dict[str, Any]], fvgs: List[Dict[str, Any]],
                     bias: str) -> Dict[str, Any]:
    """
    Prefers an order block that OVERLAPS a same-direction FVG (highest
    quality confluence). Falls back to the best single OB, then best FVG.
    """
    want_type = "bearish" if bias == "SELL" else "bullish"
    obs = [ob for ob in (order_blocks or []) if ob.get("type") == want_type and not ob.get("is_inducement")]
    gaps = [g for g in (fvgs or []) if g.get("type") == want_type]

    def _overlaps(ob, g):
        return not (ob["top"] < g["bottom"] or ob["bottom"] > g["top"])

    for ob in obs:
        for g in gaps:
            if _overlaps(ob, g):
                return {
                    "ob": ob, "fvg": g,
                    "zone_top": max(ob["top"], g["top"]),
                    "zone_bottom": min(ob["bottom"], g["bottom"]),
                    "status": "FRESH" if g["status"] == "FRESH" and ob["freshness"] == "untested" else "PARTIAL",
                    "confluence": True,
                }

    if obs:
        ob = obs[0]
        return {
            "ob": ob, "fvg": None,
            "zone_top": ob["top"], "zone_bottom": ob["bottom"],
            "status": "FRESH" if ob["freshness"] == "untested" else "TESTED",
            "confluence": False,
        }

    if gaps:
        g = gaps[0]
        return {
            "ob": None, "fvg": g,
            "zone_top": g["top"], "zone_bottom": g["bottom"],
            "status": g["status"], "confluence": False,
        }

    return {"ob": None, "fvg": None, "zone_top": None, "zone_bottom": None,
            "status": "NONE", "confluence": False}
