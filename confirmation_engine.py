"""
confirmation_engine.py
================
Turns a detected chart pattern into an actual "fire or wait" decision.

Rule (agreed design):
  1. A pattern only fires on a MARUBOZU candle closing beyond the trigger
     (body >= 70% of range, reversal-side wick < 15% of range, range >= 0.8x ATR).
  2. If that marubozu closes within 2x ATR of the trigger -> fire at market,
     immediately.
  3. If it's already stretched beyond 2x ATR -> don't chase. Compute a
     Fibonacci discount/premium zone (50%-79% retracement) on the trigger->
     extreme leg, and fire a LIMIT order at the 61.8% anchor within that
     zone, with a 15-bar expiry.
  4. If no marubozu appears within 20 bars of a pattern becoming valid, the
     watch is abandoned (stale).

This module only decides WHEN and AT WHAT PRICE/ORDER-TYPE to act. SL/TP
math and what to actually DO with that decision (auto-fire / ask approval /
send a manual mobile ticket) live in the caller (bot.py's trade layer).

Known simplification: if price reverses hard against a still-WATCHING
pattern without a scan ever confirming it, the watch is cleared naturally
next time scan_all_patterns() stops returning that pattern (structure has
changed), rather than via an explicit "opposite invalidation" check here.
"""

STALE_BARS = 20
FIB_WAIT_BARS = 15
MARUBOZU_BODY_RATIO = 0.70
MARUBOZU_WICK_RATIO = 0.15
MARUBOZU_ATR_RATIO = 0.8
FAR_ATR_MULTIPLE = 2.0
FIB_ZONE_LOW = 0.50
FIB_ZONE_HIGH = 0.79
FIB_ENTRY_ANCHOR = 0.618

from candlestick_patterns import detect_confirmation_candle


def is_marubozu(o, h, l, c, atr):
    rng = h - l
    if rng <= 0 or atr is None or atr <= 0:
        return False
    if rng < MARUBOZU_ATR_RATIO * atr:
        return False
    body = abs(c - o)
    if body / rng < MARUBOZU_BODY_RATIO:
        return False
    reversal_wick = (h - c) if c >= o else (c - l)
    if reversal_wick / rng > MARUBOZU_WICK_RATIO:
        return False
    return True


def fib_discount_premium_zone(trigger_price, extreme_price, bias):
    """
    Returns (zone_low, zone_high, entry_anchor_price) for the pullback zone
    on the trigger->extreme breakout leg.
    """
    if bias == "BUY":
        leg = extreme_price - trigger_price
        zone_low = extreme_price - leg * FIB_ZONE_HIGH
        zone_high = extreme_price - leg * FIB_ZONE_LOW
        entry = extreme_price - leg * FIB_ENTRY_ANCHOR
    else:
        leg = trigger_price - extreme_price
        zone_high = extreme_price + leg * FIB_ZONE_HIGH
        zone_low = extreme_price + leg * FIB_ZONE_LOW
        entry = extreme_price + leg * FIB_ENTRY_ANCHOR
    return zone_low, zone_high, entry


def check_current_confirmation(df, trigger_price, bias):
    """
    One-off check of the LATEST candle against a trigger -- used by the
    Telegram informational display, which doesn't need the stateful
    bars_watched/stale-timeout tracking the live polling engine uses. Just
    answers: "as of right now, is this confirmed, and by what?"

    Returns (confirmed: bool, confirmation_type: str or None).
    """
    if len(df) < 3:
        return False, None
    o = float(df['Open'].iloc[-1]); h = float(df['High'].iloc[-1])
    l = float(df['Low'].iloc[-1]);  c = float(df['Close'].iloc[-1])
    atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else None

    broke = (c > trigger_price) if bias == "BUY" else (c < trigger_price)
    if not broke:
        return False, None

    if is_marubozu(o, h, l, c, atr):
        return True, "Marubozu"

    candle_confirmed, candle_name = detect_confirmation_candle(df, bias)
    if candle_confirmed:
        return True, candle_name

    return False, None


class ConfirmationEngine:
    """Holds per (symbol, timeframe) watch state across successive polls."""

    def __init__(self):
        self._watches = {}  # (symbol, tf) -> dict

    def reset(self, symbol, tf):
        self._watches.pop((symbol, tf), None)

    def step(self, symbol, tf, df, best_pattern):
        """
        df: cleaned OHLC dataframe (with 'ATR' column), chronological, latest
            bar last.
        best_pattern: the top result from patterns.scan_all_patterns(df), or
            None if nothing currently qualifies.

        Returns a dict:
          {"action": "NONE"|"FIRE_MARKET"|"FIRE_LIMIT",
           "pattern": DetectedPattern or None,
           "fire_price": float or None,
           "order_type": "MARKET"|"LIMIT"|None,
           "expiry_bars": int or None,
           "reason": str}
        """
        key = (symbol, tf)

        if best_pattern is None:
            self._watches.pop(key, None)
            return {"action": "NONE", "pattern": None, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "no_pattern"}

        watch = self._watches.get(key)
        if watch is None or watch["pattern_name"] != best_pattern.name or watch["bias"] != best_pattern.bias:
            watch = {"pattern_name": best_pattern.name, "bias": best_pattern.bias,
                      "trigger_price": best_pattern.trigger_price, "bars_watched": 0, "state": "WATCHING"}
            self._watches[key] = watch
        else:
            watch["trigger_price"] = best_pattern.trigger_price  # keep fresh for sloped necklines (H&S)

        if watch["state"] != "WATCHING":
            return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "already_resolved"}

        watch["bars_watched"] += 1
        if watch["bars_watched"] > STALE_BARS:
            self._watches.pop(key, None)
            return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "stale_pattern_timeout"}

        o = float(df['Open'].iloc[-1]); h = float(df['High'].iloc[-1])
        l = float(df['Low'].iloc[-1]);  c = float(df['Close'].iloc[-1])
        atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else None
        trigger = watch["trigger_price"]
        bias = watch["bias"]

        broke = (c > trigger) if bias == "BUY" else (c < trigger)
        if not broke:
            return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                    "order_type": None, "expiry_bars": None, "reason": "not_broken_yet"}

        if not is_marubozu(o, h, l, c, atr):
            candle_confirmed, candle_name = detect_confirmation_candle(df, bias)
            if not candle_confirmed:
                return {"action": "NONE", "pattern": best_pattern, "fire_price": None,
                        "order_type": None, "expiry_bars": None, "reason": "broke_but_not_confirmed"}
            confirmation_label = candle_name
        else:
            confirmation_label = "Marubozu"

        # Confirmed (either a marubozu close or a qualifying candlestick pattern) -- resolve this watch (one-shot fire).
        watch["state"] = "DONE"
        distance = abs(c - trigger)

        if atr and distance <= FAR_ATR_MULTIPLE * atr:
            return {"action": "FIRE_MARKET", "pattern": best_pattern, "fire_price": c,
                    "order_type": "MARKET", "expiry_bars": None,
                    "reason": f"{confirmation_label}_confirmed_near_trigger"}

        extreme = h if bias == "BUY" else l
        _, _, entry_anchor = fib_discount_premium_zone(trigger, extreme, bias)
        return {"action": "FIRE_LIMIT", "pattern": best_pattern, "fire_price": entry_anchor,
                "order_type": "LIMIT", "expiry_bars": FIB_WAIT_BARS,
                "reason": f"{confirmation_label}_confirmed_stretched_fib_pullback"}
