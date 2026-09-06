# What changed — full merge into one decision engine

Previously there were two separate brains: `unified_strategy.analyze()`
(trendline+SMC+OTE+fundamentals+HTF confluence, used only for Telegram
reports) and `execution_engine.py`'s own pattern-scanner + confirmation
state machine (the only thing that actually fired trades). This pass
merges them into one path used by both.

## 1. `unified_strategy.py`
- (Carried over from the previous pass) weighted, source-confidence-based
  direction instead of a raw vote count; `ready` requires 2+ independent
  engines; every call logs to `signal_log.jsonl`.
- `analyze()` now takes an optional `df=` argument. Pass in candles you
  already have (e.g. an EA's freshly-pushed live MT5 bars) and every
  sub-engine (trendline, SMC, OTE) reads that exact data instead of each
  independently re-fetching — this is what let live trading and Telegram
  reports drift onto different bars before.
- Trendline-only setups can now actually produce a ticket: `tl["position"]`
  was referenced by the ticket-building code but never populated by
  `run_trendline_analysis` (it only existed inside a report-formatting
  function). `analyze()` now calls `strategies.build_position_container()`
  itself so a pure-trendline confluence isn't `ready=True` with no ticket.
- `strategies.run_trendline_analysis()`, `smc_strategy.run_smc_analysis()`
  now also accept an optional `df=` for the same reason.

## 2. `execution_engine.py` — `poll()` rewritten around one decision
- `poll()` no longer runs its own `scan_all_patterns()` + confirmation
  vote as the decision. It calls `unified_strategy.analyze()` once, and:
  - Pulls the **shared** classic-pattern read (`scanned_pattern`) out of
    `analyze()`'s trendline intelligence — this is the same pattern
    detector as before, but now run once, on shared pivots, instead of
    twice on two different data pulls. Still used purely so the EA has
    something to draw (name/category/boundary lines).
  - Fires only when **both** hold: the pattern's own trigger actually
    breaks and gets a confirmation candle (unchanged `ConfirmationEngine`
    timing/anti-duplicate-fire logic, preserved as-is), **and**
    `analyze()`'s multi-engine confluence says `ready` in the same
    direction. Either one saying no is enough to block — there's no
    second opinion left to overrule it, because there's only one now.
  - Entry/SL/TP now come from `analyze()`'s own `ticket` (SMC zone, OTE
    zone, or the newly-populated trendline position — whichever backed
    the ready decision) instead of a separately-computed pattern-based
    setup, so the numbers that fire match the numbers that justified
    firing.
- Removed: the old `_trendline_fallback` path, the separate HTF
  confidence-boost (`get_htf_bias`/`htf_alignment_adjustment` — HTF
  confluence is now scored once, inside `analyze()`), and last pass's
  "second opinion / downgrade AUTO to APPROVAL on disagreement" patch —
  no longer needed since there's only one opinion to begin with.
  `get_htf_bias`/`htf_alignment_adjustment`/`_trendline_fallback` are left
  in the file but unused/dead, marked as such, in case something else
  calls them directly (nothing in this repo does).

## Tested
Ran `poll()` end-to-end (not just compiled) against synthetic OHLC data
with `market_data.fetch_candles`, HTF bias, and fundamentals stubbed out
(no network in this environment) — 8 consecutive calls across two
different synthetic price series, zero exceptions, correct WAIT/ready/
evidence output each time. I could **not** synthesize data that actually
trips the classic-pattern detector (its geometry thresholds are picky by
design), so the fire path (`ConfirmationEngine.step` → ticket →
AUTO/APPROVAL/COPY_TRADE dispatch) is hand-verified against the real
function signatures but not exercised by an actual live fire in testing.

## Still outstanding
- No max-daily-loss / max-concurrent-trades cap.
- EA doesn't echo `signal_id` back on `report_event`, so outcome logging
  is still unlinked to the signal that caused it (`.mq5` source wasn't in
  your upload).
- Run this on a demo account and watch `signal_log.jsonl` plus a few real
  AUTO/APPROVAL cycles before trusting it live — nothing here has touched
  real market data or MT5.

---

# What changed — always draw a trendline

`strategies.py` / `build_trendline_family()`: the chart used to draw
nothing when no line cleared the trading bar (shallow slope rejected, or
too few pivots) — that was `uptrends`/`downtrends` staying empty with no
fallback. Now:

- `support`/`resistance` (the trading-quality lines that feed
  `direction`/`strength`/the trading decision) are **completely
  unchanged** — still `None` exactly when they were before. Nothing about
  what the bot trades on is different.
- When both end up empty, the chart now falls back to a visual-only line:
  first choice is the actual rejected-for-shallow-slope line if one was
  fit; if there wasn't even that, it connects the most recent swing
  points from a much looser pivot detector (or the last swing to the
  current bar if only one swing exists). These are tagged
  `quality: "visual_only"` / `tradeable: False` and only ever populate
  `family_lines` (the chart's own existing empty-uptrends/downtrends
  fallback path in `chart_engine.py` — no chart_engine changes needed).
- A reason string is added ("no trendline cleared the trading bar here —
  showing the nearest swing-to-swing reference line for visual context
  only; not a trade signal") so the report/chart both make clear it's
  reference geometry, not a signal.

Tested directly: built a synthetic tight-range (non-trending) series and
confirmed `uptrends`/`downtrends` stay empty (unchanged trading behavior)
while `family_lines` now contains the visual-only line(s) that
`chart_engine.py` will draw.

---

# What changed — 7-file consolidation, SMC/OTE removed, trendline upgraded

13 files -> 7: `bot.py`, `execution_engine.py`, `unified_strategy.py`,
`strategies.py`, `market_analysis.py`, `market_data.py`, `chart_engine.py`.

**Removed entirely:** `smc_strategy.py`, `smc_engine.py` (SMC dropped as a
decision source). **Folded into other files as real code (deleted as
separate files):** `topdown_engine.py` and `fundamental_analysis.py` ->
now live inside `unified_strategy.py`; `sitecustomize.py` and
`usercustomize.py` -> now live inside `strategies.py`.

### Important thing found along the way
`sitecustomize.py` / `usercustomize.py` were **monkey-patching
`strategies.build_trendline_family` at Python interpreter startup** —
silently replacing it with a 20-SMA-slope master-trendline method, then
layering a pullback-entry adapter on top. Whether this was actually
running depended on Python's site-customization file discovery on your
specific deploy platform, which is exactly the kind of thing that's easy
to lose track of. That logic was real and valuable (see below) — it's
now permanent code in `strategies.py`, not a runtime patch.

### `unified_strategy.py`
- SMC and OTE removed from the decision engine. Direction now comes from
  Trendline + Alligator regime + HTF bias + Fundamentals only.
- `topdown_engine.py`'s and `fundamental_analysis.py`'s functions are
  inlined directly (renamed only where they collided with existing names:
  `fundamental_analysis.analyze` -> `_fundamental_analyze`,
  its `format_report` -> `format_fundamental_report`).
- `ready` still requires 2+ independent sources agreeing (now out of 4
  possible: alligator/trendline/htf/fundamental, was 6 with smc/ote).
- `format_report()` (the Telegram text report) no longer prints SMC/OTE
  lines.

### `strategies.py`
- OTE block removed (`run_ote_analysis` and its helpers — was a clean,
  self-contained ~180-line section).
- The 20-SMA master-trendline method and the pullback-entry confirmation
  adapter (marubozu/engulfing/clean-rejection candle required before an
  entry is "confirmed") are now real functions
  (`_apply_trendline_upgrades`, called at the end of
  `build_trendline_family()`), not monkey-patches. `build_position_container`
  and `format_trendline_report` keep the same behavior they had under the
  old patches (gate the ticket on `entry_rules.confirmed`, rebase entry to
  the confirmation candle's close, substitute the more specific wait
  reason into the report) via thin wrappers around the original logic
  (`_build_position_container_base`, `_format_trendline_report_base`).
- One deliberate behavior change from the old patch: a flat 20-SMA no
  longer wipes the chart's lines to empty. That fought directly with the
  "always draw a reference trendline" fix from earlier in this
  conversation. Now a flat SMA just skips setting a master-line reading;
  whatever the pivot-fit engine (plus its own visual-fallback line) built
  is left alone.

### `bot.py`
- `import smc_strategy`, `generate_ote_map`/`generate_smc_map` imports,
  `send_ote_analysis()`, `send_smc_analysis()` all removed.
- `send_unified_analysis()` no longer reads `analysis["smc_intelligence"]`
  (doesn't exist anymore) — charts off the trendline intelligence instead.
- Dead OTE/SMC menu callback branches removed. The Trendline-only report
  was **already unreachable dead code** in your deployed bot (nothing in
  the home menu linked to it) — it's now wired back into the home menu
  as "📐 TRENDLINE ONLY", alongside "🧠 UNIFIED MARKET INTELLIGENCE".
- Help text and a few UI strings updated to drop OTE/SMC mentions.

### `execution_engine.py`
- Legacy `STRATEGY_OTE`/`STRATEGY_SMC` constants removed;
  `VALID_STRATEGIES` now `{UNIFIED, TRENDLINE}`.

## Tested
- All 7 files compile together (`python3 -m py_compile *.py`).
- Import order checked three ways: `bot.py`'s actual order, importing
  `execution_engine` standalone first, and `unified_strategy` standalone
  first — all resolve cleanly. (The old circular-import fragility from
  `smc_strategy.py` is gone now that the file doesn't exist.)
- Ran `unified_strategy.analyze()` and 4 consecutive `execution_engine.poll()`
  cycles against synthetic OHLC data with `market_data.fetch_candles`,
  HTF bias, and the fundamentals HTTP call all stubbed out (no network in
  this environment) — zero exceptions, correct WAIT/ready/evidence output,
  the merged 20-SMA/pullback-adapter fields (`master_trendline`,
  `entry_rules`, `pullback_entry`) all present and behaving.
- Did **not** get a live fire through `ConfirmationEngine` in testing (same
  limitation as the earlier merge — the classic-pattern detector's
  thresholds are picky and I couldn't synthesize data that trips it).

## Still outstanding
- No max-daily-loss / max-concurrent-trades cap.
- EA doesn't echo `signal_id` back on `report_event` (unrelated to this
  pass, unresolved from before).
- Run this on a demo account and watch `signal_log.jsonl` and a few real
  AUTO/APPROVAL cycles before trusting it live.
