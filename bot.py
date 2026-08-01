import os
import time
import asyncio
import threading
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

from engine import (
    RENDER_EXTERNAL_URL, TELEGRAM_BOT_TOKEN, ea_telemetry,
    user_languages, user_view_modes, active_subscribers, pending_custom_ticker_tf,
    ASSET_CONTAINER, central_decision_engine, generate_execution_chart
)

# ==========================================
# 1. TELEGRAM KEYBOARDS & UI HELPERS
# ==========================================
def get_home_menu(chat_id):
    keyboard = [
        [InlineKeyboardButton("📊 Run Institutional Analysis", callback_data="menu_analysis")],
        [InlineKeyboardButton("🚨 Generate Live Signal", callback_data="menu_signal")],
        [InlineKeyboardButton("⚡ Scalper Mode (1m/3m/5m/15m)", callback_data="menu_scalper")],
        [InlineKeyboardButton("🔍 Pattern Scanner (Choose Pair)", callback_data="menu_pattern_scanner")],
        [InlineKeyboardButton("✏️ Type Custom Ticker / Pair", callback_data="prompt_custom_ticker")],
        # Dedicated Hubs
        [InlineKeyboardButton("🤖 EA & MT5 Hub", callback_data="menu_ea_hub")],
        [InlineKeyboardButton("📱 Mobile & Settings Hub", callback_data="menu_mobile_hub")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_ea_menu():
    is_mt5_connected = (time.time() - ea_telemetry.get("last_ping", 0)) < 45
    mt5_status_str = "🟢 Connected" if is_mt5_connected else "🔴 Disconnected"

    keyboard = [
        [InlineKeyboardButton(f"📡 Connection Status: {mt5_status_str}", callback_data="view_mt5_status")],
        [InlineKeyboardButton("📊 Account & Equity Telemetry", callback_data="view_mt5_account")],
        [InlineKeyboardButton("ℹ️ EA Setup Rules & Webhook", callback_data="menu_help")],
        [InlineKeyboardButton("« Back to Home Menu", callback_data="menu_home")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_mobile_settings_menu(chat_id):
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
        [InlineKeyboardButton("« Back to Home Menu", callback_data="menu_home")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_category_keyboard(prefix, back_callback):
    kb = []
    for cat in ASSET_CONTAINER.keys():
        kb.append([InlineKeyboardButton(f"📂 {cat}", callback_data=f"{prefix}|{cat}")])
    kb.append([InlineKeyboardButton("« Back to Home", callback_data=back_callback)])
    return InlineKeyboardMarkup(kb)

def get_pairs_keyboard(pairs, prefix, back_callback):
    kb, row = [], []
    for pair in pairs:
        row.append(InlineKeyboardButton(pair, callback_data=f"{prefix}|{pair}"))
        if len(row) == 2:
            kb.append(row)
            row = []
    if row: kb.append(row)
    kb.append([InlineKeyboardButton("« Back", callback_data=back_callback)])
    return InlineKeyboardMarkup(kb)

# ==========================================
# 2. TELEGRAM COMMANDS & HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_languages[chat_id] = "English"
    user_view_modes[chat_id] = True
    
    welcome_text = (
        "🤖 **INSTITUTIONAL TRADING TERMINAL**\n"
        "----------------------------------\n"
        "Full chart pattern recognition & EA Bridge active.\n\n"
        "Select an option from the menu below:"
    )
    await update.message.reply_text(welcome_text, reply_markup=get_home_menu(chat_id), parse_mode="Markdown")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    symbol = update.message.text.strip().upper()
    timeframe = pending_custom_ticker_tf.pop(chat_id, "30min")

    status_msg = await update.message.reply_text(f"Scanning `{symbol}` ({timeframe})...", parse_mode="Markdown")
    try:
        setup = central_decision_engine(symbol, timeframe=timeframe)
        chart_img = generate_execution_chart(setup, mobile_view=user_view_modes.get(chat_id, True))
        
        info_text = (
            f"📊 **{symbol}** ({setup['selected_tf']})\n"
            f"• Pattern: `{setup['pattern_name']}`\n"
            f"• Signal: `{setup['direction']}`\n"
            f"• Confidence: `{setup['confidence']:.1f}%`\n"
            f"• Validity: `{setup['trendline_status']}`"
        )
        await status_msg.delete()
        await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=info_text, parse_mode="Markdown")
        await context.bot.send_message(chat_id=chat_id, text="Terminal Menu:", reply_markup=get_home_menu(chat_id))
    except Exception as e:
        await status_msg.edit_text(f"⚠️ Error evaluating '{symbol}': {str(e)}", reply_markup=get_home_menu(chat_id))

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id

    # --- HOME NAVIGATION ---
    if data == "menu_home":
        await query.edit_message_text("🤖 **INSTITUTIONAL TRADING TERMINAL**\nSelect an option from the menu below:", reply_markup=get_home_menu(chat_id), parse_mode="Markdown")

    # --- EA & MT5 HUB ---
    elif data == "menu_ea_hub":
        await query.edit_message_text("🤖 **EA & METATRADER 5 HUB**\nManage EA configuration, telemetry, and connection status:", reply_markup=get_ea_menu(), parse_mode="Markdown")

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
        await query.edit_message_text(text=info_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back to EA Hub", callback_data="menu_ea_hub")]]), parse_mode="Markdown")

    elif data == "menu_help":
        help_text = (
            "ℹ️ **EA SETUP & WEBHOOK RULES**\n"
            "----------------------------------\n"
            "1. Insert your Webhook URL into the MT5 EA Inputs.\n"
            "2. Allow WebRequests in MT5 Tools -> Options -> Expert Advisors.\n"
            "3. Multi-timeframe trend alignment (1H Macro + Scalper Timeframes) is enforced automatically."
        )
        await query.edit_message_text(text=help_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back to EA Hub", callback_data="menu_ea_hub")]]), parse_mode="Markdown")

    # --- MOBILE & SETTINGS HUB ---
    elif data == "menu_mobile_hub":
        await query.edit_message_text("📱 **MOBILE & DISPLAY SETTINGS**\nConfigure chart presentation and preferences:", reply_markup=get_mobile_settings_menu(chat_id), parse_mode="Markdown")

    elif data == "toggle_view_mode":
        user_view_modes[chat_id] = not user_view_modes.get(chat_id, True)
        await query.edit_message_text("📱 **MOBILE & DISPLAY SETTINGS**\nLayout preference updated!", reply_markup=get_mobile_settings_menu(chat_id), parse_mode="Markdown")

    elif data == "toggle_auto_scan":
        if chat_id in active_subscribers: active_subscribers.remove(chat_id)
        else: active_subscribers.add(chat_id)
        await query.edit_message_text("📱 **MOBILE & DISPLAY SETTINGS**\nAuto-scanner status updated!", reply_markup=get_mobile_settings_menu(chat_id), parse_mode="Markdown")

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
        await query.edit_message_text(f"Language updated to **{new_lang}**.", reply_markup=get_mobile_settings_menu(chat_id), parse_mode="Markdown")

    # --- SCANNER & PAIRS NAVIGATION ---
    elif data in ["menu_pattern_scanner", "menu_analysis", "menu_signal"]:
        await query.edit_message_text("📊 **SELECT ASSET CATEGORY**:", reply_markup=get_category_keyboard("cat_select", "menu_home"), parse_mode="Markdown")

    elif data.startswith("cat_select|"):
        _, cat_name = data.split("|", 1)
        pairs = ASSET_CONTAINER.get(cat_name, ASSET_CONTAINER["Forex Majors"])
        await query.edit_message_text(f"Select pair in **{cat_name}**:", reply_markup=get_pairs_keyboard(pairs, "pair_run", "menu_pattern_scanner"), parse_mode="Markdown")

    elif data.startswith("pair_run|"):
        _, symbol = data.split("|", 1)
        await query.edit_message_text(text=f"🔄 Scanning **{symbol}**...", parse_mode="Markdown")
        try:
            setup = central_decision_engine(symbol, timeframe="30min")
            chart_img = generate_execution_chart(setup, mobile_view=user_view_modes.get(chat_id, True))
            await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"✅ {symbol} Scan Complete ({setup['direction']})")
            await context.bot.send_message(chat_id=chat_id, text="Main Menu:", reply_markup=get_home_menu(chat_id))
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Scan error: {str(e)}", reply_markup=get_home_menu(chat_id))

    elif data == "menu_scalper":
        kb = [
            [InlineKeyboardButton("1 Minute", callback_data="tf_scalp|1min"), InlineKeyboardButton("3 Minute", callback_data="tf_scalp|3min")],
            [InlineKeyboardButton("5 Minute", callback_data="tf_scalp|5min"), InlineKeyboardButton("15 Minute", callback_data="tf_scalp|15min")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text("⚡ **SELECT SCALPER TIMEFRAME**:", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    elif data.startswith("tf_scalp|"):
        _, tf_code = data.split("|", 1)
        pending_custom_ticker_tf[chat_id] = tf_code
        await query.edit_message_text(
            f"Selected **{tf_code}**. Type any pair ticker directly in chat (e.g., GBPAUD, BTCUSD, XAUUSD) to run execution scan.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_scalper")]]),
            parse_mode="Markdown"
        )

    elif data == "prompt_custom_ticker":
        pending_custom_ticker_tf.pop(chat_id, None)
        await query.edit_message_text("✏️ Type any pair ticker directly in chat (e.g., EURUSD, XAUUSD):", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_home")]]), parse_mode="Markdown")

# ==========================================
# 3. FLASK APP & EA BRIDGE
# ==========================================
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Institutional Structure & Price-Action Pattern Engine Active 24/7!", 200

@app.route('/api/ping', methods=['POST'])
def ea_ping():
    data = request.json or {}
    ea_telemetry["last_ping"] = time.time()
    ea_telemetry["connected"] = True
    if "account" in data: ea_telemetry["account"] = data["account"]
    if "positions" in data: ea_telemetry["positions"] = data["positions"]
    return jsonify({"status": "SUCCESS", "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}), 200

@app.route('/api/signal', methods=['GET'])
def ea_get_signal():
    symbol = request.args.get('symbol', 'XAUUSD')
    tf = request.args.get('tf', '15min')
    try:
        setup = central_decision_engine(symbol, timeframe=tf)
        if setup["direction"] != "NO_SIGNAL" and setup["is_valid"]:
            return jsonify({
                "symbol": symbol, "direction": setup["direction"],
                "entry": setup["entry"], "sl": setup["sl"],
                "tp1": setup["tp1"], "tp2": setup["tp2"],
                "confidence": setup["confidence"], "pattern": setup["pattern_name"],
                "valid": True
            }), 200
    except Exception: pass
    return jsonify({"symbol": symbol, "direction": "NO_SIGNAL", "valid": False}), 200

def keep_alive_ping():
    while True:
        time.sleep(600)
        if RENDER_EXTERNAL_URL:
            try: requests.get(RENDER_EXTERNAL_URL, timeout=10)
            except Exception: pass

# ==========================================
# 4. BOT RUNNER & ENTRY POINT
# ==========================================
def run_telegram_bot():
    if not TELEGRAM_BOT_TOKEN:
        print("⚠️ TELEGRAM_BOT_TOKEN missing. Telegram bot disabled.")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    app_bot = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start_command))
    app_bot.add_handler(CallbackQueryHandler(button_callback_handler))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    loop.run_until_complete(app_bot.initialize())
    loop.run_until_complete(app_bot.start())
    loop.run_until_complete(app_bot.updater.start_polling(drop_pending_updates=True))
    loop.run_forever()

if __name__ == "__main__":
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=run_telegram_bot, daemon=True).start()
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
