# Refinement update

- Trendline pivots use a filtered close/line-chart structural pivot engine, then map accepted pivots to candle High/Low.
- Primary trendlines use longer validated structural spans; local pattern geometry remains separate.
- OTE uses the refined structural pivots and a stricter 1.75 ATR minimum impulse.
- OTE distinguishes ACTIVE from MITIGATED / PASSED zones so an already-touched zone is not presented as a fresh setup.
- Trendline educational pattern labels are deduplicated and staggered to reduce overlap.
- OTE Swing Origin and Impulse Extreme markers now use high-contrast colors and dark label boxes so they remain readable on the dark chart.
- Deriv market data uses public WebSocket access with endpoint fallback. Authenticated demo/real OTP URLs are ignored for analysis, preventing InvalidAppID / 401 authentication failures from breaking public market-data analysis.
- Volatility indices such as Volatility 75 (1s) and Volatility 10 remain mapped to their Deriv symbols and can be analyzed on M30/H1/H4 using the same Trendline/OTE engines.
- Personal Watch Price Level state is already present in bot.py; no extra file is required for that feature.

- Trendline state now distinguishes CONTINUATION, TRANSITION, and REVERSAL_CONFIRMED. A clean trendline rejection is treated as continuation evidence; a break alone is not a reversal.
- Top-down structure is now hierarchical: 4H macro swings + 4H horizontal key S/R -> 1H intermediate swings -> 30M execution trendline.
- 4H key S/R is exposed to both Trendline and OTE reports and drawn lightly on both charts.
- Compact 4H/1H swing anchors are mapped by timestamp onto the 30M Trendline chart without changing the existing 30M trendline engine.
- Deriv legacy fallback URLs now include the documented test app_id fallback (1089) while the current public endpoint remains the first choice. A production DERIV_APP_ID can be supplied in Render.
