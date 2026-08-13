# Smart Money Concepts (SMC)

The repository now contains `smc_strategy.py`, a standalone institutional-style SMC analysis engine.

## Analysis sequence

The engine does not treat an order block or FVG as an entry by itself. The preferred sequence is:

1. **Market structure** — structural swing highs/lows and directional bias.
2. **Liquidity mapping** — buy-side liquidity above highs and sell-side liquidity below lows.
3. **Liquidity sweep** — price takes external liquidity and closes back through the level.
4. **Displacement** — a decisive expansion candle in the intended direction.
5. **BOS / CHOCH** — structure confirms the displacement.
6. **Order Block** — last opposing candle before displacement.
7. **FVG** — displacement imbalance; only fresh/unmitigated zones are preferred.
8. **Premium / Discount** — longs prefer discount; shorts prefer premium.
9. **HTF alignment** — higher-timeframe structure can reinforce or penalize the setup.
10. **Liquidity targets** — targets are mapped to the next opposing liquidity pool; if no clean pool exists, a risk-based fallback is used.

## Conservative entry rule

A BUY/SELL signal requires the core SMC sequence to be present: liquidity sweep + displacement + structure confirmation + fresh OB/FVG confluence. Otherwise the engine returns **WAIT** rather than forcing a signal.

## Important distinction

- **OB/FVG are context zones, not standalone entries.**
- A liquidity sweep is not automatically a reversal.
- Displacement must be meaningful relative to ATR.
- Structure confirmation must follow the liquidity event.
- A fresh zone is preferred over a repeatedly mitigated zone.
- The engine is analysis-only and does not place broker orders by itself.

This file documents the SMC engine committed to `main` and is intended to be the reference for the Telegram/UI wiring that can be layered on top of it.
