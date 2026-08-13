SMC Analysis Fix - telegram-chart-bot
=====================================

Problem fixed:
  After selecting an asset in the SMC menu, analysis did not run.
  Cause: send_smc_analysis (and send_phase_analysis) depended on
  strategies.run_trendline_analysis() to get OHLC data. If Trendline
  failed or returned no dataframe, SMC silently failed.

What changed in bot.py:
  - send_smc_analysis now fetches 30M + 4H candles directly via
    market_data.fetch_candles and passes them to smc_strategy.analyse_smc
  - send_phase_analysis also uses direct market_data.fetch_candles
  - Better loading messages and error reporting

How to apply:
  1. Replace your existing bot.py with the one in this zip
  2. Commit and push to GitHub (or upload to Render)
  3. Restart / redeploy the service

Alternative (no code change needed):
  On Render, change Start Command from:
      python bot.py
  to:
      python bot_smc.py
  Then restart. bot_smc.py already uses the correct direct data path.

After the fix you should see:
  "⏳ Loading 30M + 4H market data for SYMBOL..."
  followed by the full SMC Market Map + Market Phase report.
