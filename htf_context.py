"""
htf_context.py
================
Top-down analysis: for a lower-timeframe entry signal, check the higher
timeframe's own structure/trend first. Scalping timeframes (1m/3m/5m/15m)
are for entry TIMING -- the higher timeframe establishes the actual
directional bias. A lower-timeframe pattern that aligns with the higher
timeframe gets reinforced; one that fights it gets flagged as counter-trend.

HTF bias is read two ways, preferring the stronger signal:
  1. If the higher timeframe itself has a valid detected chart pattern,
     that pattern's bias IS the HTF bias (strongest signal).
  2. Otherwise, fall back to a simple EMA50 trend read on the HTF.
"""

from patterns import scan_all_patterns
import mt5_data

LTF_TO_HTF = {
    "1min": "15min",
    "3min": "30min",
    "5min": "1h",
    "15min": "4h",
}

HTF_ALIGN_BONUS = 8.0
HTF_COUNTER_PENALTY = -10.0


def get_htf_bias(symbol, ltf_timeframe):
    """
    Returns (bias, description) or (None, None) if no HTF mapping exists
    for this timeframe (e.g. already a high timeframe) or data is unavailable.
    """
    htf_tf = LTF_TO_HTF.get(ltf_timeframe)
    if htf_tf is None:
        return None, None

    df = mt5_data.fetch_candles(symbol, htf_tf, count=150)
    if df is None or df.empty or len(df) < 40:
        return None, None

    best, _ = scan_all_patterns(df)
    if best is not None:
        return best.bias, f"{best.name} on {htf_tf}"

    if 'EMA50' in df.columns:
        current_close = float(df['Close'].iloc[-1])
        ema50 = float(df['EMA50'].iloc[-1])
        bias = "BUY" if current_close > ema50 else "SELL"
        return bias, f"price {'above' if bias=='BUY' else 'below'} EMA50 trend on {htf_tf}"

    return None, None


def htf_alignment_adjustment(ltf_bias, htf_bias, htf_desc):
    """
    Returns (confidence_delta, note_text). Zero delta and a neutral note
    if no HTF context is available at all -- never silently penalizes
    when we simply don't know.
    """
    if htf_bias is None:
        return 0.0, "No higher-timeframe context available."
    if ltf_bias == htf_bias:
        return HTF_ALIGN_BONUS, f"Aligned with higher-timeframe bias: {htf_desc}."
    return HTF_COUNTER_PENALTY, f"⚠️ COUNTER-TREND: against higher-timeframe bias ({htf_desc}) -- treat with extra caution."
