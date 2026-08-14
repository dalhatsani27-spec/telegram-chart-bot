"""Mandatory Trendline V4 runtime hook.

This file is intentionally small and defensive. It loads the V4 decision layer
at interpreter startup and patches the existing Trendline entry/report path so
Render cannot keep using the old report/entry gate while the bot is running.
"""
from __future__ import annotations

import os
import sys
import importlib


def _install():
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        if root not in sys.path:
            sys.path.insert(0, root)
        importlib.invalidate_caches()
        import strategies
        from trendline_confidence import calculate_confidence, evaluate_final_decision, format_v4_report

        if getattr(strategies.run_trendline_analysis, "_trendline_v4", False):
            print("[trendline_v4] already installed")
            return

        original_run = strategies.run_trendline_analysis
        original_build = strategies.build_position_container

        def run_v4(symbol, *args, **kwargs):
            family = original_run(symbol, *args, **kwargs)
            if isinstance(family, dict) and not family.get("error"):
                try:
                    confidence = calculate_confidence(family)
                    family["confidence_breakdown"] = confidence
                    family["confidence"] = confidence["score"]
                    family["confidence_grade"] = confidence["grade"]
                    decision = evaluate_final_decision(family)
                    family["final_decision"] = decision
                    family["entry_confirmed_v4"] = bool(decision["confirmed"])
                except Exception as exc:
                    print(f"[trendline_v4] evaluation failed for {symbol}: {exc!r}")
            return family

        def format_v4(family, symbol, *args, **kwargs):
            try:
                if isinstance(family, dict) and not family.get("error"):
                    return format_v4_report(family, symbol)
            except Exception as exc:
                print(f"[trendline_v4] report failed for {symbol}: {exc!r}")
            return strategies.format_trendline_report.__wrapped__(family, symbol, *args, **kwargs) if hasattr(strategies.format_trendline_report, "__wrapped__") else "Trendline V4 report unavailable."

        def build_v4(family, *args, **kwargs):
            if isinstance(family, dict) and not family.get("error"):
                decision = family.get("final_decision")
                if decision is None:
                    confidence = family.get("confidence_breakdown") or calculate_confidence(family)
                    decision = evaluate_final_decision({**family, "confidence_breakdown": confidence})
                if not decision.get("confirmed"):
                    return None
            return original_build(family, *args, **kwargs)

        run_v4._trendline_v4 = True
        format_v4.__wrapped__ = getattr(strategies, "format_trendline_report", None)
        strategies.run_trendline_analysis = run_v4
        strategies.format_trendline_report = format_v4
        strategies.build_position_container = build_v4
        print("[trendline_v4] INSTALLED — final confidence + mandatory entry gate active")
    except Exception as exc:
        print(f"[trendline_v4] startup hook FAILED: {exc!r}")


_install()

# Visual Pattern Engine: pattern names are derived from fitted chart geometry
# and the chart renderer draws those rails directly. It deliberately does not
# replace the V4 entry gate; it replaces the old pattern-drawing path.
try:
    from visual_pattern_runtime import install_visual_pattern_engine
    install_visual_pattern_engine()
except Exception as exc:
    print(f"[visual_pattern] activation failed: {exc!r}")
