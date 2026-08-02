"""
sr_zones.py
================
Right now a pattern's "resistance_level"/"support_level" is a single price
taken from a local high/low -- that treats a level touched once the same as
one defended four times. This clusters nearby pivot touches (highs and lows
together) into zones and scores them by touch count, so a trigger sitting
on a heavily-touched zone can be weighted higher than one sitting on a
level nobody's ever really tested.

Self-contained: patterns.py already computes pivot highs/lows internally
for its own detectors, so this just reuses those same arrays -- no new
external data source needed (unlike volume_profile.py, which needs real
tick data).
"""

import numpy as np


def cluster_sr_zones(prices, tolerance_frac=0.0015):
    """
    prices: flat list/array of price levels (pivot highs + pivot lows combined).
    Greedily clusters values within tolerance_frac of each other.
    Returns list of {"level": float, "touch_count": int}, sorted by touch_count desc.
    """
    if prices is None or len(prices) == 0:
        return []
    sorted_prices = sorted(float(p) for p in prices)
    zones = []
    current_cluster = [sorted_prices[0]]

    for p in sorted_prices[1:]:
        cluster_center = sum(current_cluster) / len(current_cluster)
        tol = cluster_center * tolerance_frac
        if abs(p - cluster_center) <= tol:
            current_cluster.append(p)
        else:
            zones.append({"level": sum(current_cluster) / len(current_cluster), "touch_count": len(current_cluster)})
            current_cluster = [p]
    zones.append({"level": sum(current_cluster) / len(current_cluster), "touch_count": len(current_cluster)})

    zones.sort(key=lambda z: z["touch_count"], reverse=True)
    return zones


def zone_strength_bonus(zones, price_level, tolerance_frac=0.0015, max_bonus=10.0):
    """
    Confidence delta for a trigger sitting at/near a well-touched S/R zone.
    +2 per touch beyond the first 2 (i.e. a 4-touch zone -> +4, capped at max_bonus).
    Returns 0.0 if no zone is nearby.
    """
    if not zones or price_level is None:
        return 0.0
    for z in zones:
        tol = z["level"] * tolerance_frac
        if abs(price_level - z["level"]) <= tol:
            bonus = max(0, z["touch_count"] - 2) * 2.0
            return float(min(max_bonus, bonus))
    return 0.0
