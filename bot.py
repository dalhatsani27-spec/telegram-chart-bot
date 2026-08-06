import os
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# Import custom MetaApi wrappers from mt5_data.py
from mt5_data import fetch_rates, execute_trade

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends welcome message and bot commands."""
    welcome_text = (
        "🤖 **MT5 Trading & Analysis Bot Online**\n\n"
        "Commands:\n"
        "• `/analyze <symbol>` - Fetch market analysis (e.g. `/analyze BTCUSD`)\n"
        "• `/trade <symbol> <action> <volume> <sl> <tp>` - Execute trade\n"
        "• `/status` - Check bot connection status"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Checks data connectivity to MT5 via MetaApi."""
    await update.message.reply_text("⏳ Checking MetaApi MT5 connection...")
    df = fetch_rates(symbol="EURUSD", timeframe="15m", count=10)
    
    if df is not None and not df.empty:
        await update.message.reply_text("✅ **Connected to MetaApi MT5 Cloud API**", parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ **Failed to reach MT5 via MetaApi.** Check credentials.", parse_mode="Markdown")


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fetches candle stats for a requested symbol."""
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Usage: `/analyze <symbol>` (e.g., `/analyze BTCUSD`)", parse_mode="Markdown")
        return

    symbol = args[0].upper()
    await update.message.reply_text(f"📊 Fetching 15m candle data for **{symbol}**...", parse_mode="Markdown")

    df = fetch_rates(symbol=symbol, timeframe="15m", count=50)

    if df is None or df.empty:
        await update.message.reply_text(f"❌ Failed to retrieve candles for {symbol}.")
        return

    latest_close = df["Close"].iloc[-1]
    highest = df["High"].max()
    lowest = df["Low"].min()

    reply = (
        f"📈 **Market Summary for {symbol} (15m)**\n"
        f"• **Current Close:** `{latest_close}`\n"
        f"• **50-Bar High:** `{highest}`\n"
        f"• **50-Bar Low:** `{lowest}`\n"
    )

    keyboard = [
        [
            InlineKeyboardButton(f"Buy {symbol}", callback_data=f"TRADE|BUY|{symbol}"),
            InlineKeyboardButton(f"Sell {symbol}", callback_data=f"TRADE|SELL|{symbol}")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(reply, reply_markup=reply_markup, parse_mode="Markdown")


async def trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Executes a market or pending order."""
    args = context.args
    if len(args) < 3:
        await update.message.reply_text(
            "⚠️ Usage: `/trade <symbol> <BUY|SELL|BUY_STOP|SELL_STOP> <volume> [SL] [TP]`",
            parse_mode="Markdown"
        )
        return

    symbol = args[0].upper()
    action = args[1].upper()
    volume = float(args[2])
    sl = float(args[3]) if len(args) > 3 else None
    tp = float(args[4]) if len(args) > 4 else None

    await update.message.reply_text(f"🔄 Executing **{action}** on **{symbol}** (Vol: {volume})...", parse_mode="Markdown")

    result = await execute_trade(symbol=symbol, action=action, volume=volume, sl=sl, tp=tp)

    if result:
        await update.message.reply_text(f"✅ Trade executed successfully!\nResult: `{result}`", parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ Trade execution failed. Check logs for details.")


async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles inline quick-action buttons."""
    query = update.callback_query
    await query.answer()

    data_parts = query.data.split("|")
    if data_parts[0] == "TRADE":
        action = data_parts[1]
        symbol = data_parts[2]

        await query.edit_message_text(f"⏳ Executing **{action}** order for **{symbol}**...", parse_mode="Markdown")
        result = await execute_trade(symbol=symbol, action=action, volume=0.01)

        if result:
            await query.message.reply_text(f"✅ **{action}** order placed for {symbol}!")
        else:
            await query.message.reply_text(f"❌ Failed to place **{action}** order for {symbol}.")


# ---------------------------------------------------------
# Main Application Launcher
# ---------------------------------------------------------
def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set!")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("trade", trade_command))
    app.add_handler(CallbackQueryHandler(button_callback_handler))

    # Detect if running on Render cloud environment
    port = int(os.getenv("PORT", 10000))
    render_url = os.getenv("RENDER_EXTERNAL_URL")

    if render_url:
        # WEBHOOK MODE (Automated on Render)
        logger.info(f"Starting bot with Webhook on Render (Port: {port})...")
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=TELEGRAM_BOT_TOKEN,
            webhook_url=f"{render_url}/{TELEGRAM_BOT_TOKEN}",
            drop_pending_updates=True
        )
    else:
        # POLLING MODE (Fallback for local testing)
        logger.info("Starting bot with Polling (Local mode)...")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
