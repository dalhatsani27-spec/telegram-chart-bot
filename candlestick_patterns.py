"""
candlestick_patterns.py
================
Single/double/triple candle patterns, used as an ALTERNATIVE confirmation
trigger alongside the marubozu rule in confirmation_engine.py -- not a
replacement. A clean bullish engulfing or hammer at a breakout is just as
much "conviction" as a marubozu candle; this widens what counts as
confirmation without loosening the underlying standard (each pattern here
still requires a real, well-formed shape -- not just "any candle").

Only high-probability, clearly-directional patterns are included. Doji is
deliberately excluded from confirmation (it signals indecision, not
conviction) -- it's noted separately as a caution flag, not a trigger.
"""


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
