"""Runtime activation for the visual geometric Trendline engine."""
from __future__ import annotations


def install_visual_pattern_engine() -> None:
    try:
        import strategies
        import chart_engine
        from visual_pattern_engine import detect_visual_pattern, render_trendline_map

        original_run = strategies.run_trendline_analysis

        if getattr(original_run, "_visual_geometry", False):
            return

        def run_with_visual_geometry(symbol, *args, **kwargs):
            family = original_run(symbol, *args, **kwargs)
            if isinstance(family, dict) and not family.get("error"):
                try:
                    df = family.get("df")
                    if df is None:
                        df = family.get("df_30m")
                    if df is not None and len(df) >= 30:
                        visual = detect_visual_pattern(df.tail(min(140, len(df))).copy())
                        family["visual_pattern"] = visual
                        family["visual_pattern_name"] = visual.get("name")
                        family["visual_pattern_confidence"] = visual.get("confidence", 0)
                        # The geometric detector is authoritative for the
                        # pattern shown on the chart. Existing market
                        # structure remains available for entry confirmation,
                        # but it is no longer used to draw/name the pattern.
                except Exception as exc:
                    print(f"[visual_pattern] detection failed for {symbol}: {exc!r}")
            return family

        def render_visual(df, symbol, setup, title_suffix=""):
            try:
                return render_trendline_map(df, symbol, setup, title_suffix)
            except Exception as exc:
                print(f"[visual_pattern] renderer failed for {symbol}: {exc!r}")
                # Fall back to the existing renderer so a chart is never lost.
                return original_renderer(df, symbol, setup, title_suffix)

        original_renderer = chart_engine.generate_trendline_map
        run_with_visual_geometry._visual_geometry = True
        strategies.run_trendline_analysis = run_with_visual_geometry
        chart_engine.generate_trendline_map = render_visual
        print("[visual_pattern] INSTALLED — geometric pattern drawing is active")
    except Exception as exc:
        print(f"[visual_pattern] startup hook failed: {exc!r}")
