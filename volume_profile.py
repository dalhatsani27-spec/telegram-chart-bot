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
  - Value Area (VAH / VAL): the band containing ~70% of total activity around the POC.
  - High Volume Nodes (HVN): price levels with significantly above-average activity.

Used two ways:
  1. Visual overlay on the chart (horizontal histogram + POC / VA levels).
  2. Confidence signal: a pattern or zone sitting at the POC / HVN / inside
     the Value Area gets a boost; thin low-volume areas get a penalty.
"""


import numpy as np


def compute_volume_profile(df, bins=32, value_area_pct=0.70):
    """
    Fixed Range Volume Profile from OHLC (+ optional tick Volume).

    Returns dict with:
      bin_edges, bin_volumes, poc_price,
      value_area_low, value_area_high,
      hvn_prices (high-volume node mid prices),
      total_volume
    or None if range is too small.
    """
    if df is None or df.empty:
        return None
    price_min = float(df['Low'].min())
    price_max = float(df['High'].max())
    if price_max <= price_min:
        return None

    bins = max(12, min(int(bins), 80))
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

    total = float(bin_volumes.sum())
    if total <= 0:
        return None

    poc_idx = int(np.argmax(bin_volumes))
    poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

    lo_i = hi_i = poc_idx
    included_vol = bin_volumes[poc_idx]
    while included_vol / total < value_area_pct and (lo_i > 0 or hi_i < bins - 1):
        left_vol = bin_volumes[lo_i - 1] if lo_i > 0 else -1.0
        right_vol = bin_volumes[hi_i + 1] if hi_i < bins - 1 else -1.0
        if right_vol >= left_vol and hi_i < bins - 1:
            hi_i += 1
            included_vol += bin_volumes[hi_i]
        elif lo_i > 0:
            lo_i -= 1
            included_vol += bin_volumes[lo_i]
        else:
            break

    # High Volume Nodes (significantly above average activity)
    avg_vol = total / bins
    hvn_prices = []
    for i, v in enumerate(bin_volumes):
        if v >= avg_vol * 1.6:
            mid = float((bin_edges[i] + bin_edges[i + 1]) / 2)
            hvn_prices.append(mid)
    hvn_prices = sorted(hvn_prices, key=lambda p: abs(p - poc_price))[:6]

    return {
        "bin_edges": bin_edges,
        "bin_volumes": bin_volumes,
        "poc_price": poc_price,
        "value_area_low": float(bin_edges[lo_i]),
        "value_area_high": float(bin_edges[hi_i + 1]),
        "hvn_prices": hvn_prices,
        "total_volume": total,
    }



def level_volume_bonus(profile, price_level, tolerance_frac=0.0015):
    """
    Confidence delta for a pattern's trigger/neckline or SMC zone sitting at a
    volume-significant level (+) or in a thin, low-activity gap (-).
    Range: roughly -5 to +10. Returns 0.0 if no profile is available.
    """
    if profile is None or price_level is None:
        return 0.0

    bin_edges = profile["bin_edges"]
    avg_price = float((bin_edges[0] + bin_edges[-1]) / 2) or 1.0
    tol = avg_price * tolerance_frac

    poc = profile["poc_price"]
    if abs(price_level - poc) <= tol:
        return 10.0

    # High Volume Nodes
    for hvn in profile.get("hvn_prices") or []:
        if abs(price_level - hvn) <= tol:
            return 7.0

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

