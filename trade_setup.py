"""
trade_setup.py
================
Takes a confirmation_engine fire decision + the detected pattern and computes
the final entry/SL/TP1/TP2 numbers. Kept separate from confirmation_engine.py
so "when do we act" and "what are the numbers" stay independently testable.

Rules carried over from the earlier design:
  - SL is structural: bound to the pattern's own footprint (its trigger_line
    span), not an arbitrary fixed lookback -- avoids grabbing a flagpole's
    origin and producing an oversized stop.
  - TP for flags/pennants uses the measured-move (flagpole height), ANCHORED
    TO THE TRIGGER PRICE (not the actual fill price) -- so a stretched/Fib
    entry doesn't get an inflated target just because the fill was late.
  - TP for everything else uses a 1.5R / 3R risk-multiple off the SL distance.
"""


def _pattern_atr(df):
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        return float(df['ATR'].iloc[-1])
    return float((df['High'] - df['Low']).tail(14).mean())


def build_trade_setup(df, pattern, fire_decision):
    """
    df: OHLC dataframe used for the scan (with ATR column).
    pattern: the DetectedPattern that was confirmed.
    fire_decision: dict from ConfirmationEngine.step() with action in
                   {"FIRE_MARKET","FIRE_LIMIT"}.

    Returns dict: entry, order_type, sl, tp1, tp2, trigger_price, bias, pattern_name
    """
    entry = fire_decision["fire_price"]
    order_type = fire_decision["order_type"]
    bias = pattern.bias
    trigger = pattern.trigger_price
    atr = _pattern_atr(df)

    span_xs = [p[0] for p in (pattern.trigger_line or [])]
    n = len(df)
    if span_xs:
        window_start = max(0, min(span_xs) - 3)
    else:
        window_start = max(0, n - 60)
    local_window = df.iloc[window_start:]
    resistance_level = float(local_window['High'].max())
    support_level = float(local_window['Low'].min())

    if bias == "BUY":
        sl = min(entry, support_level) - atr * 0.5
        risk = max(abs(entry - sl), atr * 0.25)
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 3.0
    else:
        sl = max(entry, resistance_level) + atr * 0.5
        risk = max(abs(sl - entry), atr * 0.25)
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 3.0

    if "Flag" in pattern.name or "Pennant" in pattern.name:
        pole_pts = [p for p in (pattern.key_points or []) if "Pole" in p[2]]
        if len(pole_pts) >= 2:
            pole_height = abs(pole_pts[1][1] - pole_pts[0][1])
            if pole_height > 0:
                # Anchored to the TRIGGER, not the fill price.
                if bias == "BUY":
                    tp1 = trigger + pole_height * 0.618
                    tp2 = trigger + pole_height * 1.0
                else:
                    tp1 = trigger - pole_height * 0.618
                    tp2 = trigger - pole_height * 1.0

    return {
        "entry": entry, "order_type": order_type, "bias": bias,
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "trigger_price": trigger, "pattern_name": pattern.name,
        "category": pattern.category, "confidence": pattern.confidence,
        "note": pattern.note, "expiry_bars": fire_decision.get("expiry_bars"),
    }
