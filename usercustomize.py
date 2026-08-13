"""SMC runtime state patch loaded by Python after sitecustomize."""
from __future__ import annotations


def _patch_smc() -> None:
    try:
        import smc_engine
    except Exception as exc:
        print(f"[usercustomize] SMC patch unavailable: {exc!r}")
        return

    def fvg_with_state(df, direction, start_i=0):
        if df is None or len(df) < 3 or direction not in ("BUY", "SELL"):
            return None
        candidates = []
        for i in range(max(2, int(start_i) + 2), len(df)):
            h2 = float(df.High.iloc[i - 2]); l2 = float(df.Low.iloc[i - 2])
            h = float(df.High.iloc[i]); l = float(df.Low.iloc[i])
            a = max(float(df.ATR.iloc[i]), 1e-9)
            if direction == "BUY" and l > h2 and l - h2 >= 0.10 * a:
                candidates.append({"low": h2, "high": l, "direction": "BUY", "i": i})
            elif direction == "SELL" and h < l2 and l2 - h >= 0.10 * a:
                candidates.append({"low": h, "high": l2, "direction": "SELL", "i": i})

        if not candidates:
            return None

        for zone in reversed(candidates):
            start = int(zone["i"]) + 1
            low = float(zone["low"]); high = float(zone["high"])
            touched_i = None; filled_i = None
            for j in range(start, len(df)):
                bar_hi = float(df.High.iloc[j]); bar_lo = float(df.Low.iloc[j])
                if bar_lo <= high and bar_hi >= low:
                    touched_i = j
                    if direction == "BUY" and bar_lo <= low:
                        filled_i = j
                    elif direction == "SELL" and bar_hi >= high:
                        filled_i = j
                    break
            zone["status"] = "FILLED" if filled_i is not None else "MITIGATED" if touched_i is not None else "ACTIVE"
            zone["mitigated"] = zone["status"] != "ACTIVE"
            zone["mitigated_index"] = touched_i
            zone["filled_index"] = filled_i
            return zone
        return None

    smc_engine._fvg = fvg_with_state

    # ---------------------------------------------------------------
    # Missed-entry recovery with Fibonacci pullback.
    # ---------------------------------------------------------------
    def _fib_pullback(df, direction, trigger_i, original_zone=None, confirmation_i=None):
        """Build a recovery entry after the original entry has been missed.

        The impulse is measured from the structural trigger/confirmation area
        to the latest directional extreme.  The recovery zone is 50%-61.8%
        retracement, with 61.8% as the preferred entry level.  A new rejection
        candle is required once price actually reaches the recovery zone.
        """
        out = {
            "status": "NOT_ACTIVE",
            "missed_entry": False,
            "direction": direction,
            "anchor_low": None,
            "anchor_high": None,
            "levels": {},
            "zone_low": None,
            "zone_high": None,
            "preferred": None,
            "touched": False,
            "rejection_confirmed": False,
            "entry": None,
            "reason": "Original entry is still considered available.",
        }
        if df is None or len(df) < 10 or direction not in ("BUY", "SELL"):
            return out

        n = len(df)
        start = max(0, min(int(trigger_i), n - 2))
        # Include the confirmation event if it predates the structural trigger.
        if confirmation_i is not None:
            start = max(0, min(start, int(confirmation_i)))
        a = max(float(df.ATR.iloc[-1]), 1e-9)
        close = float(df.Close.iloc[-1])

        zl = zh = None
        if original_zone:
            zl = float(original_zone.get("low")); zh = float(original_zone.get("high"))
        if zl is None or zh is None or zh <= zl:
            return out

        # The original entry is missed only after price has cleanly displaced
        # away from the original zone. A small move outside the zone is still
        # treated as an ordinary entry/retest, not a missed trade.
        if direction == "BUY":
            missed = close > zh + 0.75 * a
        else:
            missed = close < zl - 0.75 * a
        if not missed:
            return out

        # Find the directional impulse beginning at the trigger.
        if direction == "BUY":
            low_i = int(df.Low.iloc[start:].idxmin())
            high_i = int(df.High.iloc[low_i:].idxmax()) if low_i < n - 1 else low_i
            # If the low occurs unusually late, fall back to the full trigger window.
            if high_i <= low_i:
                high_i = int(df.High.iloc[start:].idxmax())
                low_i = int(df.Low.iloc[start:high_i + 1].idxmin()) if high_i >= start else start
            lo = float(df.Low.iloc[low_i]); hi = float(df.High.iloc[high_i])
            if hi <= lo or hi - lo < 1.0 * a:
                return out
            levels = {"38.2": hi - 0.382 * (hi - lo), "50.0": hi - 0.500 * (hi - lo),
                      "61.8": hi - 0.618 * (hi - lo), "70.5": hi - 0.705 * (hi - lo)}
            zone_low, zone_high = levels["61.8"], levels["50.0"]
            touched = float(df.Low.iloc[-1]) <= zone_high and float(df.High.iloc[-1]) >= zone_low
            rejection = False
            if touched:
                o = float(df.Open.iloc[-1]); h = float(df.High.iloc[-1]); l = float(df.Low.iloc[-1]); c = close
                rng = max(h - l, 1e-9)
                rejection = c > o and c > zone_low and (c - l) / rng >= 0.55
            entry = levels["61.8"]
        else:
            high_i = int(df.High.iloc[start:].idxmax())
            low_i = int(df.Low.iloc[high_i:].idxmin()) if high_i < n - 1 else high_i
            if low_i <= high_i:
                low_i = int(df.Low.iloc[start:].idxmin())
                high_i = int(df.High.iloc[start:low_i + 1].idxmax()) if low_i >= start else start
            hi = float(df.High.iloc[high_i]); lo = float(df.Low.iloc[low_i])
            if hi <= lo or hi - lo < 1.0 * a:
                return out
            levels = {"38.2": lo + 0.382 * (hi - lo), "50.0": lo + 0.500 * (hi - lo),
                      "61.8": lo + 0.618 * (hi - lo), "70.5": lo + 0.705 * (hi - lo)}
            zone_low, zone_high = levels["50.0"], levels["61.8"]
            touched = float(df.Low.iloc[-1]) <= zone_high and float(df.High.iloc[-1]) >= zone_low
            rejection = False
            if touched:
                o = float(df.Open.iloc[-1]); h = float(df.High.iloc[-1]); l = float(df.Low.iloc[-1]); c = close
                rng = max(h - l, 1e-9)
                rejection = c < o and c < zone_high and (h - c) / rng >= 0.55
            entry = levels["61.8"]

        out.update({
            "status": "PULLBACK_ENTRY_READY" if rejection else "MISSED_ENTRY_PULLBACK",
            "missed_entry": True,
            "anchor_low": lo,
            "anchor_high": hi,
            "anchor_low_i": low_i,
            "anchor_high_i": high_i,
            "levels": levels,
            "zone_low": zone_low,
            "zone_high": zone_high,
            "preferred": 61.8,
            "touched": touched,
            "rejection_confirmed": rejection,
            "entry": entry,
            "reason": (
                "Original entry was missed; price reached the 50%-61.8% Fibonacci pullback zone and rejected."
                if rejection else
                "Original entry was missed; wait for price to pull back into the 50%-61.8% Fibonacci zone."
            ),
        })
        return out

    original_run = smc_engine.run_smc_analysis

    def run_with_recovery(symbol, count=260):
        a = original_run(symbol, count)
        if not isinstance(a, dict) or a.get("error"):
            return a
        direction = a.get("direction")
        if direction not in ("BUY", "SELL"):
            return a

        df = a.get("df")
        confirmation_i = a.get("confirmation_i")
        structure_event = a.get("structure_event") or {}
        trigger_i = max([int(x.get("i")) for x in [a.get("sweep"), a.get("choch"), a.get("bos"), a.get("displacement"), structure_event] if isinstance(x, dict) and x.get("i") is not None] or [max(0, len(df) - 30)])

        original_zone = a.get("order_block") or a.get("fvg")
        recovery = _fib_pullback(df, direction, trigger_i, original_zone, confirmation_i)
        a["fib_pullback"] = recovery

        if recovery.get("missed_entry"):
            if recovery.get("rejection_confirmed"):
                a["status"] = "PULLBACK_ENTRY_READY"
                a["waiting_for"] = "FIB 50%-61.8% PULLBACK REJECTION"
            else:
                a["status"] = "MISSED_ENTRY_PULLBACK"
                a["waiting_for"] = "FIB 50%-61.8% PULLBACK"
            a["entry"] = recovery.get("entry") or a.get("entry")
            a["pullback_entry"] = recovery.get("entry")
            a["pullback_zone"] = [recovery.get("zone_low"), recovery.get("zone_high")]
            a["requirements"] = {**(a.get("requirements") or {}), "fib_pullback": True}
        return a

    smc_engine.run_smc_analysis = run_with_recovery

    original_report = smc_engine.format_smc_report

    def report_with_state(a):
        text = original_report(a)
        f = a.get("fvg") or {}
        if f:
            state = f.get("status", "ACTIVE")
            text = text.replace(
                f"FVG: {smc_engine._p(f['low'])} — {smc_engine._p(f['high'])}",
                f"FVG: {smc_engine._p(f['low'])} — {smc_engine._p(f['high'])} | {state}",
            )
            if state in ("MITIGATED", "FILLED"):
                text += f"\n\nIMBALANCE STATE: {state} — do not wait for a fresh first touch of this FVG."
            else:
                text += "\n\nIMBALANCE STATE: ACTIVE — fresh directional FVG remains available."

        conf_i = a.get("confirmation_i")
        if conf_i is not None and a.get("df") is not None:
            age = len(a["df"]) - 1 - int(conf_i)
            text += f"\nCONFIRMATION STATE: RETAINED ({age} bars ago)"

        fib = a.get("fib_pullback") or {}
        if fib.get("missed_entry"):
            lv = fib.get("levels") or {}
            text += (
                "\n\nMISSED ENTRY: YES"
                f"\nFIB PULLBACK: 50%={smc_engine._p(lv.get('50.0'))} | "
                f"61.8%={smc_engine._p(lv.get('61.8'))}"
                f"\nPREFERRED PULLBACK ENTRY: {smc_engine._p(fib.get('entry'))}"
                f"\nPULLBACK STATUS: {fib.get('status')}"
            )
            if fib.get("rejection_confirmed"):
                text += "\nPULLBACK CONFIRMATION: REJECTION CONFIRMED — ENTRY READY"
            else:
                text += "\nPULLBACK CONFIRMATION: WAIT FOR REJECTION IN FIB ZONE"
        return text

    smc_engine.format_smc_report = report_with_state


_patch_smc()
