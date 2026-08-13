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

        # Prefer the newest FVG, but explicitly calculate whether later price
        # action touched or completely filled it. A mitigated imbalance is not
        # a fresh entry requirement and must not make the bot wait for a zone
        # that has already been used.
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
        return text
    smc_engine.format_smc_report = report_with_state


_patch_smc()
