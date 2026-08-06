"""
volume_profile.py
================
Fixed Range Volume Profile, computed from MT5 tick_volume (number of price
updates per bar -- a real, commonly-used proxy in forex/CFD trading, NOT
true traded volume, since that doesn't exist in retail forex. If tick
volume isn't available (e.g. Twelve Data fallback with no Volume column),
degrades to a "time spent at price" profile -- still useful, just weaker.

Produces:
  - Point of Control (POC): the price level with the most activity.
  - Value Area: the price band containing ~68% of total activity around the POC.

Used two ways:
  1. Visual overlay on the chart (a horizontal histogram beside price).
  2. A confidence signal: a pattern's trigger/neckline sitting at the POC or
     inside the Value Area gets a confidence boost (that level has proven
     significance); one sitting in a low-volume gap gets a penalty (thin,
     less-respected price, more prone to a violent, unreliable move).
"""

import numpy as np


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
