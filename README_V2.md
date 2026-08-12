
# Telegram Chart Bot — Market Intelligence V2

This upgrade adds a deterministic price-action intelligence layer without
destroying the existing Trendline/OTE engines.

## Added

- Market-state classification
- Range/base detection
- Liquidity pool mapping
- Equal highs/lows
- Liquidity sweep detection
- Displacement confirmation
- FVG detection and mitigation state
- Order Block detection
- OB → Breaker lifecycle
- Structure context
- A/B/C setup grading
- Explicit execution permission
- Telegram-ready report
- AI-ready JSON output

## Philosophy

The engine is confirmation-based, not predictive.

It should answer:

> What has price confirmed?

rather than:

> What do I think price will do next?

## Install

Copy `market_intelligence.py` into the repository root.

Follow `INTEGRATION_PATCH.md` to connect it to `bot.py` and `strategies.py`.

## Safety

The V2 module does not place trades. It only produces analysis and an
execution-permission flag. Keep the existing execution engine as the final
trade-control layer while testing.
