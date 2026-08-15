"""Modern Trendline pullback-entry adapter.

Loaded after sitecustomize. It keeps the protected structural trendline/master
trend logic intact and upgrades the entry layer from break-only to:

1. Trendline intact -> wait for a pullback into the master rail.
2. Pullback touches the rail within an ATR-normalized zone.
3. Rejection/continuation candle confirms the pullback.
4. Enter at the confirmation close.
5. If the rail breaks, fall back to the existing break/retest lifecycle.

No SMA/RSI/volume filter is used as an entry trigger. SMA remains a regime
compass in sitecustomize; the trendline + price reaction are the edge.
"""

import strategies
from market_analysis import (
    is_bullish_marubozu,
    is_bearish_marubozu,
    is_bullish_engulfing,
    is_bearish_engulfing,
)

_ORIGINAL_BUILD_TRENDLINE = getattr(strategies, "build_trendline_family", None)
_ORIGINAL_BUILD_POSITION = getattr(strategies, "build_position_container", None)
_ORIGINAL_FORMAT_REPORT = getattr(strategies, "format_trendline_report", None)

PULLBACK_ZONE_ATR = 0.35
PULLBACK_INVALIDATION_ATR = 0.15
MIN_BODY_RATIO = 0.45
MAX_REJECTION_WICK_BODY = 1.25


def _line_at(df, line, index=None):
    if not line or df is None or df.empty:
        return None
    i = len(df) - 1 if index is None else int(index)
    return float(strategies._line_value(line["x0"], line["y0"], line["x1"], line["y1"], i))


def _atr_at(df, index=None):
    i = len(df) - 1 if index is None else int(index)
    if "ATR" in df.columns:
        try:
            v = float(df["ATR"].iloc[i])
            if v > 0:
                return v
        except Exception:
            pass
    return max(float(df["High"].iloc[i]) - float(df["Low"].iloc[i]), 1e-9)


def _bar(df, i):
    r = df.iloc[int(i)]
    return float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])


def _rejection_confirmation(df, index, direction, line_price, atr):
    """Return (True, label) for a clean pullback rejection candle."""
    if index < 0 or index >= len(df):
        return False, None

    o, h, l, c = _bar(df, index)
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    body_ratio = body / rng
    prior = _bar(df, index - 1) if index > 0 else None

    if direction == "BUY":
        if is_bullish_marubozu((o, h, l, c), atr):
            return True, "Bullish Marubozu"
        if prior is not None and is_bullish_engulfing(prior, (o, h, l, c)):
            return True, "Bullish Engulfing"

        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        clean_rejection = (
            c > o
            and body_ratio >= MIN_BODY_RATIO
            and lower_wick >= body * 0.60
            and upper_wick <= max(body * MAX_REJECTION_WICK_BODY, atr * 0.12)
            and c >= line_price + atr * 0.10
        )
        if clean_rejection:
            return True, "Bullish Trendline Rejection"

    if direction == "SELL":
        if is_bearish_marubozu((o, h, l, c), atr):
            return True, "Bearish Marubozu"
        if prior is not None and is_bearish_engulfing(prior, (o, h, l, c)):
            return True, "Bearish Engulfing"

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        clean_rejection = (
            c < o
            and body_ratio >= MIN_BODY_RATIO
            and upper_wick >= body * 0.60
            and lower_wick <= max(body * MAX_REJECTION_WICK_BODY, atr * 0.12)
            and c <= line_price - atr * 0.10
        )
        if clean_rejection:
            return True, "Bearish Trendline Rejection"

    return False, None


def _pullback_zone(line_price, atr, direction):
    if direction == "BUY":
        return {
            "low": line_price - atr * PULLBACK_INVALIDATION_ATR,
            "high": line_price + atr * PULLBACK_ZONE_ATR,
            "anchor": line_price,
            "width_atr": PULLBACK_ZONE_ATR + PULLBACK_INVALIDATION_ATR,
            "side": "support",
        }
    return {
        "low": line_price - atr * PULLBACK_ZONE_ATR,
        "high": line_price + atr * PULLBACK_INVALIDATION_ATR,
        "anchor": line_price,
        "width_atr": PULLBACK_ZONE_ATR + PULLBACK_INVALIDATION_ATR,
        "side": "resistance",
    }


def _trendline_pullback_state(family):
    """Evaluate the current pullback to the master trendline."""
    df = family.get("df") if family else None
    master = family.get("master_trendline") if family else None
    direction = str((family or {}).get("direction") or "NEUTRAL").upper()
    role = str((family or {}).get("master_role") or "none").lower()

    out = {
        "state": "WAIT_PULLBACK",
        "confirmed": False,
        "confirmation": None,
        "entry_mode": None,
        "entry_price": None,
        "line_price": None,
        "zone": None,
        "touch": False,
        "distance_atr": None,
        "reason": "Wait for price to pull back into the master trendline zone.",
    }

    if df is None or df.empty or not master or direction not in ("BUY", "SELL"):
        return out
    if direction == "BUY" and role != "support":
        return out
    if direction == "SELL" and role != "resistance":
        return out

    i = len(df) - 1
    line_price = _line_at(df, master, i)
    atr = _atr_at(df, i)
    zone = _pullback_zone(line_price, atr, direction)
    close = float(df["Close"].iloc[i])
    high = float(df["High"].iloc[i])
    low = float(df["Low"].iloc[i])
    distance_atr = abs(close - line_price) / atr if atr > 0 else 999.0

    out.update(line_price=line_price, zone=zone, distance_atr=round(distance_atr, 2))

    # A real close through the rail is no longer a pullback. The master
    # breakout/retest lifecycle takes over instead.
    if direction == "BUY" and close < line_price - atr * PULLBACK_INVALIDATION_ATR:
        out.update(state="INVALIDATED", reason="Price closed materially below rising support; pullback setup invalidated.")
        return out
    if direction == "SELL" and close > line_price + atr * PULLBACK_INVALIDATION_ATR:
        out.update(state="INVALIDATED", reason="Price closed materially above falling resistance; pullback setup invalidated.")
        return out

    touch = high >= zone["low"] and low <= zone["high"]
    out["touch"] = bool(touch)

    if not touch:
        out["state"] = "WAIT_PULLBACK"
        out["reason"] = (
            f"Wait for pullback into master {role} {zone['low']:.5f}–{zone['high']:.5f} "
            f"({zone['anchor']:.5f} trendline)."
        )
        return out

    confirmed, name = _rejection_confirmation(df, i, direction, line_price, atr)
    if confirmed:
        out.update(
            state="PULLBACK_ENTRY_CONFIRMED",
            confirmed=True,
            confirmation=name,
            entry_mode="MARKET",
            entry_price=close,
            reason=f"{name} rejected the master {role} and closed back in the trend direction.",
        )
    else:
        out.update(
            state="WAIT_PULLBACK_CONFIRMATION",
            reason=f"Price touched the master {role}; wait for a directional rejection/continuation candle before entry.",
        )
    return out


def _legacy_break_retest_entry(family):
    """Preserve the existing post-break retest entry, with no extra filters."""
    df = family.get("df") if family else None
    direction = str((family or {}).get("direction") or "NEUTRAL").upper()
    retest = (family or {}).get("trendline_retest") or {}
    base = {"confirmed": False, "state": "WAIT_BREAK", "confirmation": None, "entry_mode": None, "reason": "Wait for trendline setup."}
    if df is None or len(df) < 3 or direction not in ("BUY", "SELL"):
        return base
    status = str(retest.get("status") or "INTACT")
    if status == "FAKEOUT":
        return {**base, "state": "INVALIDATED", "reason": "Trendline break was reclaimed; wait for a fresh setup."}
    if status != "BREAK_RETEST_CONFIRMED":
        if status in ("BREAK_CONFIRMED", "BREAK_DEVELOPING"):
            return {**base, "state": "WAIT_RETEST", "reason": "Break detected; wait for the broken trendline to retest and hold."}
        return base
    try:
        ri = int(retest.get("retest_index"))
    except Exception:
        ri = None
    if ri is None or ri >= len(df) - 1:
        return {**base, "state": "WAIT_CONTINUATION", "reason": "Retest confirmed; wait for the first directional continuation candle."}

    latest = len(df) - 1
    ok, name = _rejection_confirmation(df, latest, direction, float(retest.get("retest_level") or df["Close"].iloc[ri]), _atr_at(df, latest))
    if ok:
        return {"confirmed": True, "state": "BREAK_RETEST_ENTRY_CONFIRMED", "confirmation": name, "entry_mode": "MARKET", "reason": f"{name} confirmed continuation after trendline break/retest."}

    return {**base, "state": "WAIT_CONTINUATION", "reason": "Retest held; wait for a directional rejection/continuation candle."}


def _wrap_family(original):
    if original is None or getattr(original, "_modern_pullback_wrapped", False):
        return original

    def wrapped(df, max_lines=4, lookback_bars=60):
        family = original(df, max_lines=max_lines, lookback_bars=lookback_bars)
        if not family or family.get("error"):
            return family

        master = family.get("master_trendline")

        pullback = _trendline_pullback_state(family)
        family["pullback_entry"] = pullback
        family["pullback_zone"] = pullback.get("zone")
        family["pullback_entry_price"] = pullback.get("entry_price")
        family["pullback_distance_atr"] = pullback.get("distance_atr")

        # The pullback is the preferred entry while the master trendline is
        # intact. A break must invalidate it and hand control to break/retest.
        if pullback.get("confirmed"):
            family["entry_rules"] = {
                "checks": {"pullback": (True, pullback.get("confirmation"))},
                "passed": 1, "required": 1, "confirmed": True,
                "state": "PULLBACK_ENTRY_CONFIRMED",
                "wait_reason": pullback.get("reason"),
                "confirmation": pullback.get("confirmation"),
                "entry_mode": "MARKET",
                "entry_price": pullback.get("entry_price"),
                "pullback_zone": pullback.get("zone"),
            }
            family["master_entry_ready"] = True
            family["reasons"] = list(family.get("reasons") or [])
            family["reasons"].append(f"✅ PULLBACK ENTRY CONFIRMED — {pullback['confirmation']} at master trendline.")
            return family

        # If the master line has broken, use the established break/retest
        # lifecycle rather than treating a broken rail as support/resistance.
        legacy = _legacy_break_retest_entry(family) if master else {"confirmed": False, "state": "WAIT_BREAK", "reason": "No master trendline."}
        family["entry_rules"] = {
            "checks": {"trendline": (bool(legacy.get("confirmed")), legacy.get("reason"))},
            "passed": 1 if legacy.get("confirmed") else 0,
            "required": 1,
            "confirmed": bool(legacy.get("confirmed")),
            "state": legacy.get("state"),
            "wait_reason": legacy.get("reason"),
            "confirmation": legacy.get("confirmation"),
            "entry_mode": legacy.get("entry_mode"),
            "pullback_zone": pullback.get("zone"),
        }
        family["master_entry_ready"] = bool(legacy.get("confirmed"))
        family["reasons"] = list(family.get("reasons") or [])
        family["reasons"].append(("✅ " if legacy.get("confirmed") else "⏳ ") + legacy.get("reason", ""))
        return family

    wrapped._modern_pullback_wrapped = True
    wrapped._original = original
    return wrapped


if _ORIGINAL_BUILD_TRENDLINE is not None:
    strategies.build_trendline_family = _wrap_family(_ORIGINAL_BUILD_TRENDLINE)


if _ORIGINAL_BUILD_POSITION is not None and not getattr(_ORIGINAL_BUILD_POSITION, "_modern_pullback_wrapped", False):
    def _position(family, *args, **kwargs):
        is_trendline = bool(family and (family.get("master_trendline") is not None or family.get("trendline_retest") is not None))
        if is_trendline and not (family.get("entry_rules") or {}).get("confirmed"):
            return None

        pos = _ORIGINAL_BUILD_POSITION(family, *args, **kwargs)
        if not pos or not is_trendline:
            return pos

        rules = family.get("entry_rules") or {}
        df = family.get("df")
        if rules.get("confirmed") and df is not None and not df.empty:
            new_entry = rules.get("entry_price") or float(df["Close"].iloc[-1])
            old_entry = pos.get("entry")
            if old_entry is not None:
                delta = float(new_entry) - float(old_entry)
                for key in ("sl", "tp1", "tp2", "tp3"):
                    if pos.get(key) is not None:
                        pos[key] = float(pos[key]) + delta
            pos["entry"] = float(new_entry)
            pos["order_type"] = "MARKET"
            pos["entry_confirmation"] = rules.get("confirmation")
            pos["entry_confirmation_state"] = rules.get("state")
            pos["pullback_zone"] = rules.get("pullback_zone")
            pos["confirmed"] = True
            pos["entry_rules"] = rules
        return pos

    _position._modern_pullback_wrapped = True
    _position._original = _ORIGINAL_BUILD_POSITION
    strategies.build_position_container = _position


if _ORIGINAL_FORMAT_REPORT is not None and not getattr(_ORIGINAL_FORMAT_REPORT, "_modern_pullback_wrapped", False):
    def _report(family, symbol):
        report = _ORIGINAL_FORMAT_REPORT(family, symbol)
        rules = (family or {}).get("entry_rules") or {}
        reason = rules.get("wait_reason")
        if reason and "WAIT FOR:" in report and not rules.get("confirmed"):
            head, tail = report.split("WAIT FOR:", 1)
            suffix = ""
            marker = "\nNo trade yet."
            if marker in tail:
                _, end = tail.split(marker, 1)
                suffix = marker + end
            return head + "WAIT FOR:\n1. " + reason + suffix
        return report

    _report._modern_pullback_wrapped = True
    _report._original = _ORIGINAL_FORMAT_REPORT
    strategies.format_trendline_report = _report

print("[trendline_pullback] modern pullback-entry adapter installed")
