# Refinement update

- Trendline pivots now use a filtered close/line-chart structural pivot engine, then map pivots to candle High/Low.
- Primary trendlines prefer a longer validated structural span; local pattern geometry remains separate.
- OTE uses the refined structural pivots and a stricter 1.75 ATR minimum impulse.
- OTE now distinguishes ACTIVE from MITIGATED/PASSED zones so an old, already-touched zone is not presented as a fresh setup.
- Trendline educational pattern labels are deduplicated.
- Deriv public WebSocket endpoint corrected to the documented public endpoint.
- One-shot Deriv tick/candle requests no longer send `subscribe: 0`, improving compatibility with Deriv API validation.
- Volatility indices such as Volatility 75 (1s) remain mapped to their Deriv symbols and can be analyzed on M30/H1/H4 using the same engines.
