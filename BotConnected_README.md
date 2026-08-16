# TrendlinePatternEA_BotConnected — live-linked to your Telegram bot

This is the version that actually talks to your running bot instead of
re-detecting patterns on its own. The bot is the single source of truth;
this EA is a thin client.

## How the connection works
```
EA  --POST candles-->  bot   /api/ea/poll/<symbol>/<tf>
EA  <--pattern + trade command--  bot
EA  --executes trade, reports fill/close-->  bot   /api/ea/report
```
Every `PollSeconds`, the EA:
1. Reads its own last N **closed** bars straight from MT5 and POSTs them to
   the bot.
2. The bot runs the exact same fixed pattern engine you already have
   (SMA anchor, ranging gate, triangle/wedge classification, confirmation-
   candle stage-gating) on those bars and answers with: the current
   pattern (name/bias/category/stage/confidence), the two boundary-line
   points to draw, and — if your Control Panel mode has fired a signal —
   a trade **command** to execute.
3. The EA draws the pattern exactly as the bot describes it (same
   cheat-sheet-style coloring/labels as the standalone EA) and, if a
   command came back, places the trade and reports the fill back.

Two real code changes were needed on the bot side to make this actually
work (the routes existed but weren't fully wired):
- `/api/ea/poll/<symbol>/<tf>` now accepts **POST** with your candles
  (previously GET-only, so it silently never marked the EA as "seen" or
  used your live MT5 data — see `execution_engine.py`).
- The response now includes the pattern's `category`, `stage`,
  `confidence`, `sma_ranging` state, and the `upper_line`/`lower_line`
  boundary points (converted to real timestamps) — previously it only
  returned bare name/bias/trigger_price, not enough to actually draw the
  pattern.

**Replace your deployed `execution_engine.py` with the updated one in this
delivery** — the EA depends on these fields existing in the response.

## Install
1. Update `execution_engine.py` on your bot (same file, same location) —
   included in this delivery.
2. Copy `TrendlinePatternEA_BotConnected.mq5` into `MQL5/Experts/`,
   compile it (F7 in MetaEditor), attach to a chart.
3. **Whitelist the bot's URL**: MT5 -> Tools -> Options -> Expert Advisors
   -> check "Allow WebRequest for listed URL" -> add your bot's base URL
   (e.g. `http://127.0.0.1:5000` if it's on the same PC, or
   `http://YOUR-SERVER:5000` if remote). WebRequest fails with error 4060
   until this is done — the EA will print this clearly on OnInit if it
   can't reach the bot.
4. Set the `ApiKey` input to match the `EA_API_KEY` environment variable
   your bot is running with (this is what `/api/ea/*` already checks via
   `X-API-KEY` — see `execution_engine.py`).
5. Set `BotBaseURL` to wherever your bot actually runs.

## Trading modes (controlled from your Telegram Control Panel, not the EA)
- **OFF** — bot does nothing, EA just... won't get commands.
- **AUTO** — bot queues a command the moment a signal fires; this EA
  executes it automatically.
- **APPROVAL** — bot asks you to Approve/Reject in Telegram first; only
  queues the command after you approve.
- **COPY_TRADE** — bot sends you a manual ticket in Telegram instead of
  commanding any EA (useful if you're not running this EA at all, or want
  to place trades by hand).

The EA doesn't choose the mode — that's entirely your Telegram Control
Panel. It just executes whatever the bot queues.

## Inputs worth knowing about
- `UseTP2AsFinalTarget` — the bot always computes two targets (tp1 closer,
  tp2 further). This EA uses a single-stage exit for simplicity: `false`
  (default) exits at tp1, `true` exits at tp2. (A two-stage partial-close-
  at-tp1-then-run-to-tp2 manager is a reasonable next step if you want it
  — just ask.)
- `MinLot` / `RiskPercent` — used depending on whether your Control Panel's
  lot mode is set to MIN or RISK.
- `CandlesToPush` — how many closed bars get sent per poll (250 default,
  matches what the bot's analysis expects).

## Before you trust it live
Same caveat as the standalone EA: I couldn't compile or run this against a
live bot from here. Compile it, run both sides on a demo account, and
watch a few poll cycles in the Experts/Journal log before enabling AUTO
mode.
