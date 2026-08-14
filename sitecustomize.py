"""
Project startup hook.

The existing Trendline engine already contains geometry, classic pattern
scanning, breakout grading, entry confirmation, order blocks and the 4H/1H
cascade. This hook adds the independent final confidence layer without
rewriting those mature components or changing execution rules.
"""


def _install():
    try:
        import strategies
        from trendline_confidence import calculate_confidence, format_confidence_block

        original_run = strategies.run_trendline_analysis
        original_format = strategies.format_trendline_report

        if getattr(original_run, "_confidence_v3", False):
            return

        def run_with_confidence(symbol, *args, **kwargs):
            family = original_run(symbol, *args, **kwargs)
            if isinstance(family, dict) and not family.get("error"):
                try:
                    result = calculate_confidence(family)
                    family["confidence"] = result["score"]
                    family["confidence_grade"] = result["grade"]
                    family["confidence_breakdown"] = result
                    # Keep the existing 'strength' untouched. It remains the
                    # strategy/execution score; confidence is an independent
                    # evidence-weighted quality score for the final readout.
                except Exception as exc:
                    print(f"[confidence_v3] calculation failed for {symbol}: {exc!r}")
            return family

        run_with_confidence._confidence_v3 = True

        def format_with_confidence(family, symbol, *args, **kwargs):
            report = original_format(family, symbol, *args, **kwargs)
            result = family.get("confidence_breakdown") if isinstance(family, dict) else None
            if result:
                report = report.rstrip() + "\n\n" + format_confidence_block(result)
            return report

        strategies.run_trendline_analysis = run_with_confidence
        strategies.format_trendline_report = format_with_confidence
        print("[confidence_v3] Trendline confidence layer installed.")
    except Exception as exc:
        # Never prevent the existing bot from starting if the optional layer
        # cannot load.
        print(f"[confidence_v3] startup hook skipped: {exc!r}")


_install()
