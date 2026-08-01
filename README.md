# telegram-chart-bot
# Institutional Price-Action Engine + MT5 Control Center

One brain (Python), two control surfaces: the MT5 chart (execution +
dashboard) and Telegram (mobile command center). Both talk to the same
pattern-detection + confirmation engine, so there's exactly one source of
truth for what counts as a valid, confirmed setup.

## How it fits together

```
MT5 Terminal (your PC)
 ├─ Market data, account info, broker info, trade history  →  read directly
 │  by Python via the MetaTrader5 package (free, local, no rate limits)
 │
 └─ PatternControlPanelEA.mq5 (attached to each chart you want traded)
      - polls Python's local API once per new bar
      - executes whatever command comes back (market or limit order)
      - manages TP1 partial-close + breakeven
      - draws the on-chart dashboard (broker/account/PnL/win-rate/mode)

Python (bot.py, runs locally on the same PC, started manually for now)
      - patterns.py: full chart-pattern scanner (flags, triangles, wedges,
        H&S, double/triple top/bottom, ranges)
      - confirmation_engine.py: marubozu breakout confirmation, with a
        Fibonacci discount/premium pullback path for stretched moves
      - trade_state.py: the Mode (OFF/AUTO/APPROVAL/COPY_TRADE), set from
        Telegram, plus the small command queue the EA polls
      - engine.py: ties it together, routes confirmed setups according to Mode
      - engine_api.py: the Flask routes the EA calls
      - Telegram: informational analysis buttons + the Mobile Control Panel
```

## One-time setup

### 1. Python environment
```
cd python
pip install -r requirements.txt
cp .env.example .env      # then fill in your actual values
```
`MetaTrader5` only works on Windows with a terminal installed -- if you're
setting this up on the same Windows PC as MT5 (which is the intended setup),
this just works. `TWELVE_DATA_API_KEY` is optional now -- it's a fallback for
symbols MT5 doesn't carry, not the primary data source anymore.

### 2. MT5 terminal
- Make sure you're logged into your account and the terminal is running
  before you start `bot.py` -- that's what `mt5_data.py` connects to.
- **Tools → Options → Expert Advisors** → check "Allow WebRequest for listed
  URL" and add: `http://127.0.0.1:5000` (match whatever `PORT` you set in `.env`).
- Enable **AutoTrading** (top toolbar) and allow algo trading for the EA
  in its properties.

### 3. Compile the EA
- Open MetaEditor, put `PatternControlPanelEA.mq5`, `HttpJson.mqh`,
  `TradeExecution.mqh`, and `ChartDashboard.mqh` all in the same
  `MQL5/Experts` folder.
- Open the `.mq5` and hit Compile (F7). Fix any errors it reports -- I
  couldn't compile-test this myself (no MQL5 toolchain available to me), so
  send me the exact error text if anything fails on first compile.

### 4. Start Python
```
python bot.py
```
This starts both the Telegram bot and the local Flask API in one process.
Message your bot `/start` on Telegram.

### 5. Attach the EA
- Drag `PatternControlPanelEA` onto a chart, set `InpApiKey` to match
  `EA_API_KEY` in your `.env`, confirm `InpApiBaseUrl` matches your port.
- Repeat for every symbol/timeframe you want it watching -- one EA instance
  per chart, each trading whatever timeframe that chart is set to.

## Using it

- **Master switch defaults to OFF.** Nothing trades until you turn it on
  from Telegram's Mobile Control Panel.
- **AUTO** — EA fires the moment Python confirms a setup.
- **APPROVAL** — Telegram sends you Approve/Reject buttons first; only
  fires after you tap Approve (expires after 3 minutes unanswered).
- **Copy Trade** — EA does not execute anything. Telegram sends you the
  full manual ticket (entry, order type, SL, TP1, TP2) to place yourself.
  Use this when you're away from your PC.
- **Confirmation rule**: a pattern only fires on a marubozu candle closing
  beyond its trigger. Near the trigger (within 2x ATR) it fires at market;
  stretched beyond that, it waits for a Fibonacci 50%-79% pullback instead
  of chasing, via a real MT5 limit order with a 15-bar expiry.

## Testing before real money

Everything in `python/` was tested with synthetic price data during
development (pattern detection, marubozu confirmation, Fib pullback
routing, all four Modes, the Flask API end-to-end). The MQL5 side could
only be checked for structural correctness (brace/paren balance, careful
manual review) since I have no MQL5 compiler -- **run this on a demo
account for a while before ever pointing it at live money**, and send me
compiler errors or runtime surprises as you hit them.
