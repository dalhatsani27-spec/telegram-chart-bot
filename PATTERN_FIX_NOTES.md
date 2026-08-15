# Pattern Detection / S-R Zone Fix

## Root cause
`build_trendline_family()` (drives the HH/HL/LH/LL structure + trendlines on
the chart) and `scan_all_patterns()` (drives Double Top/Bottom, H&S, etc.)
were each running their **own independent swing-pivot detector** with
different parameters. They could -- and did -- disagree about which points
on the chart were the "real" swings, which is why a Double Top could get
drawn on two minor pullback highs while the chart's own HH label sat on a
bigger, untested swing high right next to them.

## Changes

**market_analysis.py**
- `scan_all_patterns(df, pivots=...)` -- new optional param. When given a
  pre-computed pivot list, pattern detection uses those swings instead of
  running a second, separate zigzag pass.
- `_has_dominant_rival(...)` -- new helper. Rejects a double/triple
  top-or-bottom candidate if a more extreme swing sits in the lookback
  window that neither chosen peak/trough actually reached. Wired into
  `detect_double_top`, `detect_double_bottom`, `detect_triple_top`,
  `detect_triple_bottom`.
- `analyse_structure()` now uses the ATR-filtered ZigZag swings
  (`zigzag_swings`) instead of the raw 3-bar fractal (`find_swings`), so
  BOS/CHoCH/MSS reads are also noise-filtered, not just chart drawing.
- `cluster_sr_zones()` now returns `low`/`high` (the actual touch range) in
  addition to `level`, so a caller can draw a real zone/band instead of a
  single infinitely-precise price.

**strategies.py**
- `family["pivots_full"]` -- exposes the full (untrimmed) structural pivot
  list build_trendline_family already computes, so pattern detection has
  enough history to run the dominance check.
- `run_trendline_analysis()` now calls
  `scan_all_patterns(df_tf, pivots=family.get("pivots_full"))` so the
  pattern scanner shares the exact swings the chart is built from.
- `_detect_horizontal_levels()` now also returns `zone_low`/`zone_high` per
  level (the real touch range, padded by the clustering tolerance).

**chart_engine.py** (`generate_trendline_educational_map`, the function
`bot.py` actually calls)
- Added `_claim_label_y(...)`, a pixel-space label-collision helper. Any
  text placed near the chart (4H level labels, trendline labels, pattern
  trigger label) now checks for a nearby already-placed label and nudges
  vertically instead of overlapping (fixes "RISING SUPPORT" printing on top
  of "Double Top trigger").
- The pattern's neckline/trigger price is now drawn as an ATR-scaled shaded
  band (`axhspan`) plus a center line, instead of a single hairline --
  visually represents it as a zone, not an exact price.

## Verified (logic-level; no chart render -- see below)
Built a synthetic OHLC series replicating the reported bug (a dominant
untested HH followed by two minor pullback highs):
- **Before fix**: `detect_double_top` flagged "Double Top" on the two minor
  highs at 85% confidence, ignoring the HH.
- **After fix**: `detect_double_top` correctly returns `None`.

Built a second synthetic series with a genuine retest of the same real
swing high:
- **Before and after fix**: still correctly detected, 89% confidence.

## Update: TRENDLINE vs PATTERN conflation (strategies.py)

**Bug:** The "SETUP SCAN" report treated TRENDLINE and PATTERN as two
independent, competing setups and picked whichever scored higher. But a
trendline is just a rising support / falling resistance line -- and several
named patterns (Head & Shoulders, Wedge, Triangle...) are *built out of*
exactly that kind of rail (an H&S's shoulder-to-shoulder slope, a wedge's
converging rails). So a detected H&S could have its own Head -> Right
Shoulder slope scored a second time as an unrelated "TRENDLINE" setup,
outrank the pattern purely because the two scoring functions measure
different things (channel touch-count/quality vs. pattern
confidence-by-stage), and the report would say "BEST SETUP: TRENDLINE 63%"
while the chart was unambiguously showing a Head and Shoulders.

**Fix:** `_pattern_uses_trendline_geometry(sp, pv, primary_line)` in
`strategies.py` detects when the "trendline" rail being scored is really
the pattern's own skeleton:
- Any wedge/triangle (`pattern_visual`) is *always* disqualified -- it's
  two trendlines by definition, never an independent trendline setup.
- Any classic pattern (`scanned_pattern`, e.g. H&S/Double Top) is
  disqualified when the primary trendline's own two anchor points are
  literally two of the pattern's labelled key points (e.g. Head and R
  Shoulder), within a small index/price tolerance.

When true, `select_best_setup()` no longer lets TRENDLINE win BEST SETUP
over the pattern it's part of, and the report now annotates the TRENDLINE
line with "(this rail IS the pattern below, not a separate setup)" so it's
explicit rather than silently reordering.

**Verified:** reproduced the exact screenshot numbers (TRENDLINE 63% /
PATTERN 39%) with synthetic H&S key points sitting on the fitted resistance
rail's anchors -- confirmed `active_setup` flips from `TRENDLINE` to
`PATTERN` after the fix, with `trendline_is_pattern_rail: True`.

## Not verified
`mplfinance` isn't available in this sandbox (no network access to
install), so the actual chart image (label placement, shaded zone
rendering) hasn't been visually rendered. The code is syntax-checked and
the underlying detection logic is confirmed via the tests above, but you
should render one real chart on your own environment to sanity-check the
visual layout before deploying.
