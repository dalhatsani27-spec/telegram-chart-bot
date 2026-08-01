import os
import time
import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# Logging configuration
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==========================================
# IN-MEMORY STATE & MOCK DATA
# ==========================================
user_view_modes = {}          # {chat_id: True (Mobile) / False (Desktop)}
user_languages = {}           # {chat_id: "English" / "Hausa" / "Pidgin"}
active_subscribers = set()    # {chat_id}
pending_custom_ticker_tf = {} # {chat_id: timeframe_string}

# Telemetry data updated by MT5 EA webhooks
ea_telemetry = {
    "last_ping": 0,
    "account": {
        "login": "12345678",
        "server": "Broker-Live",
        "balance": 10500.50,
        "equity": 10720.10
    }
}

ASSET_CONTAINER = {
    "Forex Majors": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
    "Cryptocurrencies": ["BTCUSD", "ETHUSD", "SOLUSD"],
    "Commodities": ["XAUUSD", "USOIL"]
}

# ==========================================
# HELPER KEYBOARD GENERATORS
# ==========================================

def get_home_menu(chat_id: int) -> InlineKeyboardMarkup:
    """Main clean home dashboard."""
    keyboard = [
        [InlineKeyboardButton("📊 Run Institutional Analysis", callback_data="menu_analysis")],
        [InlineKeyboardButton("🚨 Generate Live Signal", callback_data="menu_signal")],
        [InlineKeyboardButton("⚡ Scalper Mode (1m/3m/5m/15m)", callback_data="menu_scalper")],
        [InlineKeyboardButton("🔍 Pattern Scanner (Choose Pair)", callback_data="menu_pattern_scanner")],
        [InlineKeyboardButton("✏️ Type Custom Ticker / Pair", callback_data="prompt_custom_ticker")],
        # SEPARATED DEDICATED HUBS
        [InlineKeyboardButton("🤖 EA & MT5 Hub", callback_data="menu_ea_hub")],
        [InlineKeyboardButton("📱 Mobile & Settings Hub", callback_data="menu_mobile_hub")]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_ea_menu() -> InlineKeyboardMarkup:
    """Dedicated Sub-menu for all MT5 & EA features."""
    is_mt5_connected = (time.time() - ea_telemetry.get("last_ping", 0)) < 45
    mt5_status_str = "🟢 Connected" if is_mt5_connected else "🔴 Disconnected"

    keyboard = [
        [InlineKeyboardButton(f"📡 Status: {mt5_status_str}", callback_data="view_mt5_status")],
        [InlineKeyboardButton("📊 EA Account & Balance Info", callback_data="view_mt5_account")],
        [InlineKeyboardButton("ℹ️ EA Setup Rules & Webhook", callback_data="menu_help")],
        [InlineKeyboardButton("« Back to Main Menu", callback_data="menu_home")]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_mobile_settings_menu(chat_id: int) -> InlineKeyboardMarkup:
    """Dedicated Sub-menu for Mobile UI and Bot Settings."""
    lang = user_languages.get(chat_id, "English")
    is_mobile = user_view_modes.get(chat_id, True)
    auto_active = chat_id in active_subscribers

    lang_flags = {"English": "🇬🇧 English", "Hausa": "🇳🇬 Hausa", "Pidgin": "🇳🇬 Pidgin"}
    current_flag = lang_flags.get(lang, "🇬🇧 English")
    
    view_str = "📱 Layout: MOBILE" if is_mobile else "💻 Layout: DESKTOP"
    scanner_str = "🟢 Auto-Scanner: ON" if auto_active else "🔴 Auto-Scanner: OFF"

    keyboard = [
        [InlineKeyboardButton(f"{view_str}", callback_data="toggle_view_mode")],
        [InlineKeyboardButton(f"{scanner_str}", callback_data="toggle_auto_scan")],
        [InlineKeyboardButton(f"🌐 Language: {current_flag}", callback_data="menu_lang_select")],
        [InlineKeyboardButton("« Back to Main Menu", callback_data="menu_home")]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_category_keyboard(prefix: str, back_target: str) -> InlineKeyboardMarkup:
    """Generates asset class selection menu."""
    keyboard = []
    for cat in ASSET_CONTAINER.keys():
        keyboard.append([InlineKeyboardButton(cat, callback_data=f"{prefix}|{cat}")])
    keyboard.append([InlineKeyboardButton("« Back", callback_data=back_target)])
    return InlineKeyboardMarkup(keyboard)


def get_pairs_keyboard(pairs: list, prefix: str, back_target: str) -> InlineKeyboardMarkup:
    """Generates pair list menu."""
    keyboard = []
    row = []
    for pair in pairs:
        row.append(InlineKeyboardButton(pair, callback_data=f"{prefix}|{pair}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("« Back", callback_data=back_target)])
    return InlineKeyboardMarkup(keyboard)


# ==========================================
# COMMAND & CALLBACK HANDLERS
# ==========================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /start command."""
    chat_id = update.message.chat_id
    await update.message.reply_text(
        text="🤖 **INSTITUTIONAL TRADING TERMINAL**\nSelect an option from the menu below:",
        reply_markup=get_home_menu(chat_id),
        parse_mode="Markdown"
    )


async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id

    # --- HOME NAVIGATION ---
    if data == "menu_home":
        await query.edit_message_text(
            text="🤖 **INSTITUTIONAL TRADING TERMINAL**\nSelect an option from the menu below:",
            reply_markup=get_home_menu(chat_id),
            parse_mode="Markdown"
        )

    # --- EA & MT5 HUB ---
    elif data == "menu_ea_hub":
        await query.edit_message_text(
            text="🤖 **EA & METATRADER 5 HUB**\nManage EA configuration, telemetry, and connection status:",
            reply_markup=get_ea_menu(),
            parse_mode="Markdown"
        )

    elif data in ["view_mt5_status", "view_mt5_account"]:
        is_connected = (time.time() - ea_telemetry.get("last_ping", 0)) < 45
        status_str = "🟢 ONLINE" if is_connected else "🔴 OFFLINE"
        acc = ea_telemetry.get("account", {})
        
        info_text = (
            f"📡 **METATRADER 5 EA TELEMETRY**\n"
            f"----------------------------------\n"
            f"• Connection: `{status_str}`\n"
            f"• Account ID: `{acc.get('login', 'N/A')}`\n"
            f"• Broker Server: `{acc.get('server', 'N/A')}`\n"
            f"• Balance: `${acc.get('balance', 0.0):,.2f}`\n"
            f"• Equity: `${acc.get('equity', 0.0):,.2f}`\n\n"
            f"Last Ping Received: `{time.strftime('%H:%M:%S', time.localtime(ea_telemetry.get('last_ping', 0)))}`"
        )
        kb = [[InlineKeyboardButton("« Back to EA Hub", callback_data="menu_ea_hub")]]
        await query.edit_message_text(text=info_text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    elif data == "menu_help":
        help_text = (
            "ℹ️ **EA SETUP & WEBHOOK RULES**\n"
            "----------------------------------\n"
            "1. Insert your Webhook URL into the MT5 EA Inputs.\n"
            "2. Allow WebRequests in MT5 Tools -> Options -> Expert Advisors.\n"
            "3. Ensure the EA interval matches your desired execution speed.\n"
            "4. Realignment trend filter validates macro structure automatically."
        )
        kb = [[InlineKeyboardButton("« Back to EA Hub", callback_data="menu_ea_hub")]]
        await query.edit_message_text(text=help_text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    # --- MOBILE & DISPLAY HUB ---
    elif data == "menu_mobile_hub":
        await query.edit_message_text(
            text="📱 **MOBILE & DISPLAY SETTINGS**\nConfigure chart presentation and preferences:",
            reply_markup=get_mobile_settings_menu(chat_id),
            parse_mode="Markdown"
        )

    elif data == "toggle_view_mode":
        user_view_modes[chat_id] = not user_view_modes.get(chat_id, True)
        await query.edit_message_text(
            text="📱 **MOBILE & DISPLAY SETTINGS**\nLayout preference updated!",
            reply_markup=get_mobile_settings_menu(chat_id),
            parse_mode="Markdown"
        )

    elif data == "toggle_auto_scan":
        if chat_id in active_subscribers:
            active_subscribers.remove(chat_id)
        else:
            active_subscribers.add(chat_id)
        await query.edit_message_text(
            text="📱 **MOBILE & DISPLAY SETTINGS**\nAuto-scanner status updated!",
            reply_markup=get_mobile_settings_menu(chat_id),
            parse_mode="Markdown"
        )

    elif data == "menu_lang_select":
        kb = [
            [InlineKeyboardButton("🇬🇧 English", callback_data="set_lang|English")],
            [InlineKeyboardButton("🇳🇬 Hausa", callback_data="set_lang|Hausa")],
            [InlineKeyboardButton("🇳🇬 Pidgin", callback_data="set_lang|Pidgin")],
            [InlineKeyboardButton("« Back to Settings", callback_data="menu_mobile_hub")]
        ]
        await query.edit_message_text(text="🌐 Select Interface Language:", reply_markup=InlineKeyboardMarkup(kb))

    elif data.startswith("set_lang|"):
        new_lang = data.split("|")[1]
        user_languages[chat_id] = new_lang
        await query.edit_message_text(
            text=f"Language updated to **{new_lang}**.",
            reply_markup=get_mobile_settings_menu(chat_id),
            parse_mode="Markdown"
        )

    # --- ANALYSIS & SCALPER NAVIGATION ---
    elif data in ["menu_pattern_scanner", "menu_analysis", "menu_signal"]:
        await query.edit_message_text(
            text="📊 **SELECT ASSET CATEGORY**:",
            reply_markup=get_category_keyboard("cat_select", "menu_home"),
            parse_mode="Markdown"
        )

    elif data.startswith("cat_select|"):
        _, cat_name = data.split("|", 1)
        pairs = ASSET_CONTAINER.get(cat_name, ASSET_CONTAINER["Forex Majors"])
        await query.edit_message_text(
            text=f"Select pair in **{cat_name}**:",
            reply_markup=get_pairs_keyboard(pairs, "pair_run", "menu_pattern_scanner"),
            parse_mode="Markdown"
        )

    elif data.startswith("pair_run|"):
        _, symbol = data.split("|", 1)
        await query.edit_message_text(text=f"🔄 Processing analysis for **{symbol}**...", parse_mode="Markdown")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"✅ Analysis for **{symbol}** complete.",
            reply_markup=get_home_menu(chat_id),
            parse_mode="Markdown"
        )

    elif data == "menu_scalper":
        kb = [
            [InlineKeyboardButton("1 Minute", callback_data="tf_scalp|1min"),
             InlineKeyboardButton("3 Minute", callback_data="tf_scalp|3min")],
            [InlineKeyboardButton("5 Minute", callback_data="tf_scalp|5min"),
             InlineKeyboardButton("15 Minute", callback_data="tf_scalp|15min")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="⚡ **SELECT SCALPER TIMEFRAME**:", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    elif data.startswith("tf_scalp|"):
        _, tf_code = data.split("|", 1)
        pending_custom_ticker_tf[chat_id] = tf_code
        await query.edit_message_text(
            text=f"Selected **{tf_code}**. Send any symbol in chat (e.g., GBPAUD, BTCUSD, XAUUSD) to execute scalper entry scan.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_scalper")]]),
            parse_mode="Markdown"
        )

    elif data == "prompt_custom_ticker":
        pending_custom_ticker_tf.pop(chat_id, None)
        await query.edit_message_text(
            text="✏️ Type any pair ticker directly into chat (e.g., EURUSD, XAUUSD):",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_home")]]),
            parse_mode="Markdown"
        )


async def text_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processes plain text user messages for custom ticker input."""
    chat_id = update.message.chat_id
    ticker = update.message.text.strip().upper()
    tf = pending_custom_ticker_tf.pop(chat_id, "5min")

    await update.message.reply_text(
        text=f"🔄 Running `{tf}` analysis for **{ticker}**...",
        parse_mode="Markdown"
    )
    await update.message.reply_text(
        text="Terminal Menu:",
        reply_markup=get_home_menu(chat_id)
    )

# ==========================================
# APPLICATION INITIALIZATION
# ==========================================

def main():
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
    app = Application.builder().token(BOT_TOKEN).build()

    # Register Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(button_callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_input_handler))

    logger.info("Bot starting polling loop...")
    app.run_polling()

if __name__ == "__main__":
    main()
