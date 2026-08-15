"""
Entry confirmation adapter for the existing Trendline-master engine.

IMPORTANT: this file is intentionally additive.  The working chart/trendline
engine remains untouched.  Python loads usercustomize after sitecustomize,
so we can add the finalized entry rules without rewriting the protected
trendline implementation.

Final Trendline entry rule:
  1. Master trendline break is confirmed.
  2. Retest is confirmed.
  3. First continuation candle after the retest is either:
       - Marubozu, or
       - Engulfing
     -> ENTER immediately. No SMA/RSI/volume/BOS filter is added.
  4. If that first entry was missed, wait for a 50%-79% Fibonacci golden-zone
     pullback and then the next Marubozu/Engulfing continuation candle.
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

FIB_GOLDEN_LOW = 0.50
FIB_GOLDEN_HIGH = 0.79


def _bar(df, index):
    row = df.iloc[index]
    return (
        float(row["Open"]),
        float(row["High"]),
        float(row["Low"]),
        float(row["Close"]),
    )


def _continuation_for_bar(df, index, direction):
    """Only the two finalized continuation candles are accepted."""
    if index < 0 or index >= len(df):
        return False, None

    atr = None
    if "ATR" in df.columns:
        try:
            atr = float(df["ATR"].iloc[index])
        except Exception:
            atr = None

    current = _bar(df, index)
    prior = _bar(df, index - 1) if index > 0 else None

    if direction == "BUY":
        if is_bullish_marubozu(current, atr):
            return True, "Bullish Marubozu"
        if prior is not None and is_bullish_engulfing(prior, current):
            return True, "Bullish Engulfing"
    elif direction == "SELL":
        if is_bearish_marubozu(current, atr):
            return True, "Bearish Marubozu"
        if prior is not None and is_bearish_engulfing(prior, current):
            return True, "Bearish Engulfing"

    return False, None


def _golden_zone(trigger, extreme, direction):
    """Return the 50%-79% retracement zone after the missed continuation."""
    leg = abs(float(extreme) - float(trigger))
    if leg <= 0:
        return None

    if direction == "BUY":
        zone_low = float(extreme) - leg * FIB_GOLDEN_HIGH
        zone_high = float(extreme) - leg * FIB_GOLDEN_LOW
        anchor = float(extreme) - leg * 0.618
    else:
        zone_low = float(extreme) + leg * FIB_GOLDEN_LOW
        zone_high = float(extreme) + leg * FIB_GOLDEN_HIGH
        anchor = float(extreme) + leg * 0.618

    return {
        "low": min(zone_low, zone_high),
        "high": max(zone_low, zone_high),
        "anchor": anchor,
        "extreme": float(extreme),
        "leg": leg,
    }


def _price_in_zone(price, zone):
    if not zone:
        return False
    return float(zone["low"]) <= float(price) <= float(zone["high"])


def _master_entry_confirmation(family):
    """Return the finalized Trendline entry state used by both Telegram and EA."""
    df = family.get("df") if family else None
    direction = str((family or {}).get("direction") or "NEUTRAL").upper()
    retest = (family or {}).get("trendline_retest") or {}

    base = {
        "checks": {"continuation": (False, "waiting for a continuation candle")},
        "passed": 0,
        "required": 1,
        "confirmed": False,
        "state": "WAIT_RETEST",
        "wait_reason": "Wait for confirmed trendline break + retest.",
        "confirmation": None,
        "retest_confirmed": False,
        "missed_entry": False,
        "fib_zone": None,
        "entry_mode": None,
    }

    if df is None or len(df) < 3 or direction not in ("BUY", "SELL"):
        return base

    status = str(retest.get("status") or "INTACT")
    if status == "FAKEOUT":
        base["state"] = "INVALIDATED"
        base["wait_reason"] = "Break was reclaimed — wait for a fresh setup."
        base["checks"]["continuation"] = (False, "fakeout / reclaimed trendline")
        return base

    if status != "BREAK_RETEST_CONFIRMED":
        if status in ("BREAK_CONFIRMED", "BREAK_DEVELOPING"):
            base["state"] = "WAIT_RETEST"
            base["wait_reason"] = "Break confirmed — wait for the retest to hold."
        else:
            base["state"] = "WAIT_BREAK"
            base["wait_reason"] = "Wait for a confirmed break of the master trendline."
        base["checks"]["continuation"] = (False, base["wait_reason"])
        return base

    try:
        retest_index = int(retest.get("retest_index"))
    except Exception:
        retest_index = None

    if retest_index is None or retest_index >= len(df) - 1:
        base["state"] = "WAIT_CONTINUATION"
        base["retest_confirmed"] = True
        base["wait_reason"] = "Retest confirmed — wait for the first Marubozu or Engulfing continuation candle."
        base["checks"]["continuation"] = (False, "retest confirmed; continuation candle not formed yet")
        return base

    base["retest_confirmed"] = True
    candidates = []
    for i in range(retest_index + 1, len(df)):
        ok, name = _continuation_for_bar(df, i, direction)
        if ok:
            candidates.append((i, name))

    latest = len(df) - 1

    # The first continuation after the confirmed retest is the immediate
    # entry. No extra filters are allowed here.
    if candidates:
        first_index, first_name = candidates[0]
        if first_index == latest:
            base.update(
                checks={"continuation": (True, first_name)},
                passed=1,
                confirmed=True,
                state="ENTRY_CONFIRMED",
                wait_reason="Entry confirmed — fire immediately.",
                confirmation=first_name,
                continuation_index=latest,
                entry_mode="MARKET",
            )
            return base

        # A continuation candle happened on an earlier closed bar. If the
        # user/EA did not enter it, the setup is now a missed entry and the
        # next allowed opportunity is a golden-zone pullback.
        base["missed_entry"] = True
        first_after_retest = first_index
    else:
        first_after_retest = None

    # Build the retracement leg from the confirmed retest to the strongest
    # excursion after it.  The golden zone is 50%-79%, with 61.8% as the
    # reference anchor.
    start = retest_index + 1
    if start >= len(df):
        start = retest_index

    if direction == "BUY":
        extreme = float(df["High"].iloc[start:].max())
    else:
        extreme = float(df["Low"].iloc[start:].min())

    trigger = retest.get("retest_level")
    if trigger is None:
        trigger = float(df["Close"].iloc[retest_index])
    trigger = float(trigger)

    zone = _golden_zone(trigger, extreme, direction)
    close = float(df["Close"].iloc[-1])

    if base["missed_entry"]:
        latest_ok, latest_name = _continuation_for_bar(df, latest, direction)
        if latest_ok and _price_in_zone(close, zone):
            base.update(
                checks={"continuation": (True, latest_name)},
                passed=1,
                confirmed=True,
                state="FIB_PULLBACK_ENTRY_CONFIRMED",
                wait_reason="Golden-zone pullback confirmed with Marubozu/Engulfing — fire immediately.",
                confirmation=latest_name,
                continuation_index=latest,
                entry_mode="MARKET",
            )
            base["fib_zone"] = zone
            return base

        base["state"] = "WAIT_FIB_PULLBACK"
        base["wait_reason"] = (
            f"Original entry missed — wait for pullback into the 50%-79% golden zone "
            f"({zone['low']:.5f}–{zone['high']:.5f}) and then the first Marubozu or Engulfing candle."
            if zone else
            "Original entry missed — wait for a fresh Fibonacci golden-zone pullback."
        )
        base["fib_zone"] = zone
        base["checks"]["continuation"] = (False, "waiting for golden-zone pullback + continuation candle")
        return base

    # No continuation has happened yet. Do not manufacture a pullback state;
    # the first valid continuation after the retest remains the entry.
    base["state"] = "WAIT_CONTINUATION"
    base["wait_reason"] = "Retest confirmed — wait for the first Marubozu or Engulfing continuation candle."
    base["fib_zone"] = zone
    base["checks"]["continuation"] = (False, "retest confirmed; continuation candle not formed yet")
    return base


def _strip_legacy_entry_reasons(reasons):
    out = []
    for reason in reasons or []:
        text = str(reason)
        if text.startswith("Entry confirmed (") or "Entry NOT confirmed yet" in text:
            continue
        out.append(reason)
    return out


def _wrap_trendline_family(original):
    if original is None or getattr(original, "_entry_adapter_wrapped", False):
        return original

    def wrapped(df, max_lines=4, lookback_bars=60):
        family = original(df, max_lines=max_lines, lookback_bars=lookback_bars)
        if not family or family.get("error"):
            return family

        # Only replace the Trendline master entry gate. OTE/SMC and the
        # protected chart geometry are left alone.
        if not family.get("trendline_retest") and not family.get("master_trendline"):
            return family

        old_rules = family.get("entry_rules") or {}
        # The old engine added +/- confidence for its obsolete 3-of-4 filter.
        # Remove that effect so confidence remains a geometry/structure score.
        try:
            strength = int(family.get("strength") or 0)
            if old_rules.get("confirmed"):
                strength -= 6
            else:
                strength += 15
            family["strength"] = max(0, min(100, strength))
        except Exception:
            pass

        rules = _master_entry_confirmation(family)
        family["entry_rules"] = rules
        family["entry_confirmation_state"] = rules["state"]
        family["entry_wait_reason"] = rules["wait_reason"]
        family["entry_confirmation"] = rules.get("confirmation")
        family["entry_fib_zone"] = rules.get("fib_zone")
        family["master_entry_ready"] = bool(rules.get("confirmed"))
        family["retest_entry_confirmed"] = bool(rules.get("retest_confirmed"))
        family["missed_entry"] = bool(rules.get("missed_entry"))
        family["reasons"] = _strip_legacy_entry_reasons(family.get("reasons"))

        if rules["confirmed"]:
            family["reasons"].append(
                f"✅ ENTRY CONFIRMED — {rules['confirmation']} after confirmed trendline retest."
            )
        elif rules["state"] == "WAIT_FIB_PULLBACK":
            family["reasons"].append(f"⏳ {rules['wait_reason']}")
        else:
            family["reasons"].append(f"⏳ {rules['wait_reason']}")

        return family

    wrapped._entry_adapter_wrapped = True
    wrapped._original = original
    return wrapped


if _ORIGINAL_BUILD_TRENDLINE is not None:
    strategies.build_trendline_family = _wrap_trendline_family(_ORIGINAL_BUILD_TRENDLINE)


if _ORIGINAL_BUILD_POSITION is not None and not getattr(_ORIGINAL_BUILD_POSITION, "_entry_adapter_wrapped", False):
    def _build_position_after_confirmation(family, *args, **kwargs):
        """Do not create a trade ticket until the finalized entry gate passes."""
        is_trendline = bool(family and (family.get("trendline_retest") is not None or family.get("master_trendline") is not None))
        if is_trendline:
            rules = family.get("entry_rules") or {}
            if not rules.get("confirmed"):
                return None

        pos = _ORIGINAL_BUILD_POSITION(family, *args, **kwargs)
        if not pos:
            return pos

        if is_trendline:
            rules = family.get("entry_rules") or {}
            df = family.get("df")
            if df is not None and not df.empty and rules.get("confirmed"):
                # Immediate confirmation means market entry at the confirmed
                # candle close. Preserve the original structural risk/reward
                # distances by shifting the existing price box to the actual
                # confirmation price instead of changing the risk model here.
                old_entry = pos.get("entry")
                new_entry = float(df["Close"].iloc[-1])
                if old_entry is not None and new_entry != float(old_entry):
                    delta = new_entry - float(old_entry)
                    for key in ("sl", "tp1", "tp2", "tp3"):
                        if pos.get(key) is not None:
                            pos[key] = float(pos[key]) + delta
                pos["entry"] = new_entry
                pos["order_type"] = "MARKET"
                pos["entry_confirmation"] = rules.get("confirmation")
                pos["entry_confirmation_state"] = rules.get("state")
                pos["fib_zone"] = rules.get("fib_zone")
                pos["missed_entry"] = rules.get("missed_entry", False)
                pos["confirmed"] = True
                pos["entry_rules"] = rules
        return pos

    _build_position_after_confirmation._entry_adapter_wrapped = True
    _build_position_after_confirmation._original = _ORIGINAL_BUILD_POSITION
    strategies.build_position_container = _build_position_after_confirmation


if _ORIGINAL_FORMAT_REPORT is not None and not getattr(_ORIGINAL_FORMAT_REPORT, "_entry_adapter_wrapped", False):
    def _format_report_with_entry_wait(family, symbol):
        report = _ORIGINAL_FORMAT_REPORT(family, symbol)
        rules = (family or {}).get("entry_rules") or {}
        if rules.get("confirmed"):
            return report

        wait_reason = rules.get("wait_reason")
        if not wait_reason or "WAIT FOR:" not in report:
            return report

        head, tail = report.split("WAIT FOR:", 1)
        if "\nNo trade yet." in tail:
            _, end = tail.split("\nNo trade yet.", 1)
            tail_end = "\nNo trade yet." + end
        else:
            tail_end = ""

        items = [wait_reason]
        # Keep the report deterministic and do not add the old 3-of-4 gate.
        replacement = "WAIT FOR:\n" + "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
        return head + replacement + tail_end

    _format_report_with_entry_wait._entry_adapter_wrapped = True
    _format_report_with_entry_wait._original = _ORIGINAL_FORMAT_REPORT
    strategies.format_trendline_report = _format_report_with_entry_wait

print("[entry_adapter] finalized Trendline break/retest + Marubozu/Engulfing + Fib golden-zone entry installed")
