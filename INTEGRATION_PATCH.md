
# TELEGRAM CHART BOT — V2 INTEGRATION PATCH

## 1. Add this import to bot.py

from market_intelligence import analyze_market_intelligence, format_intelligence_report

## 2. Add this helper to bot.py

async def send_market_intelligence(context, chat_id, symbol):
    try:
        df = market_data.fetch_candles(symbol, "30min", count=250)
        if df is None or df.empty:
            await context.bot.send_message(chat_id=chat_id, text=f"⚠️ No market data for {symbol}.")
            return

        report = analyze_market_intelligence(df)
        text = format_intelligence_report(report, symbol)
        await context.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Market intelligence failed for {symbol}: {e}"
        )

## 3. Add a button to get_home_menu()

Add this row before CONTROL PANEL:

[InlineKeyboardButton(
    "🧠 MARKET INTELLIGENCE V2",
    callback_data="menu_intelligence"
)],

## 4. Add callback handling inside button_callback_handler()

Add:

elif data == "menu_intelligence":
    await query.edit_message_text(
        "🧠 MARKET INTELLIGENCE V2\n\n"
        "Deterministic market-state analysis:\n"
        "• Accumulation / range detection\n"
        "• Liquidity pools and sweeps\n"
        "• Manipulation confirmation\n"
        "• Displacement\n"
        "• FVG lifecycle\n"
        "• OB → Breaker lifecycle\n"
        "• Structure and execution permission\n\n"
        "Type the ticker to run it.",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ])
    )

## 5. Recommended typed-command behavior

If you want every typed ticker to include V2 intelligence, change
handle_text_message() to run the intelligence first:

    await send_market_intelligence(context, chat_id, symbol)

Then continue to the selected Trendline/OTE strategy.

For a cleaner architecture, use V2 intelligence as the common context
and let Trendline/OTE use it as a gate instead of replacing their geometry.

## 6. Strategy integration

In strategies.py add:

from market_intelligence import analyze_market_intelligence

Then, inside run_trendline_analysis() after df_30m is fetched:

    intelligence = analyze_market_intelligence(df_30m)
    family["market_intelligence"] = intelligence

Do the same in run_ote_analysis().

Then use:

    intelligence["execution"]["permission"]

as a confirmation/gating input, not as an automatic trade trigger.

IMPORTANT:
Do not replace the existing Trendline/OTE scoring yet. Run V2 in parallel
first so you can compare the bot's interpretation against your own chart
reading.

## 7. Execution rule

The V2 engine deliberately does NOT predict:
"price will go up" or "price will go down".

It waits for evidence:

    liquidity sweep
        ↓
    displacement
        ↓
    structure confirmation
        ↓
    POI / FVG / breaker
        ↓
    execution permission

This is the intended behavior.
