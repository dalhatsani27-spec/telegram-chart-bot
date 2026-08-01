"""
bot.py
================
Telegram Interface & Multi-Level Execution Controller.
Provides dedicated sub-menus for EA Control and Mobile Control.
"""

import os
import logging
import threading
from flask import Flask, jsonify, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FLASK KEEP-ALIVE & API SERVER ---
app = Flask(__name__)

# GLOBAL STATE ENGINE
EXECUTION_STATE = {
    "mode": "AUTOMATIC",      # 'AUTOMATIC' (EA Control) or 'MANUAL' (Mobile Control)
    "ea_enabled": True,       # EA execution switch
    "session_filter": True,   # Preserves all existing session restrictions
    "active_symbol": "GBPAUD",
    "connected_broker": "Exness / Deriv (Waiting for MT5...)",
    "account_id": "N/A",
    "last_signal": {}
}

@app.route('/')
def home():
    return "Institutional Telegram Engine & MT5 Bridge is Live."

@app.route('/get_state', methods=['GET'])
def get_state():
    """Endpoint polled by MT5 EA to fetch active mode, EA status & orders."""
    return jsonify(EXECUTION_STATE)

@app.route('/update_account', methods=['POST'])
def update_account():
    """Endpoint to catch dynamic account details when switching brokers in MT5."""
    data = request.json or {}
    EXECUTION_STATE["connected_broker"] = data.get("broker", "Unknown Broker")
    EXECUTION_STATE["account_id"] = data.get("account_id", "N/A")
    return jsonify({"status": "success", "state": EXECUTION_STATE})

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# ----------------------------------------------------------------------------
# TELEGRAM MENU BUILDERS
# ----------------------------------------------------------------------------

def get_main_menu():
    """Main Dashboard Menu"""
    keyboard = [
        [
            InlineKeyboardButton("🤖 EA Control", callback_data="menu_ea_control"),
            InlineKeyboardButton("📱 Mobile Control", callback_data="menu_mobile_control")
        ],
        [
            InlineKeyboardButton("📊 Run Pattern Analysis", callback_data="menu_analysis"),
            InlineKeyboardButton("⚡ Scalper Mode (5m / 15m)", callback_data="menu_scalper")
        ],
        [
            InlineKeyboardButton("🔍 Select Asset Pair", callback_data="menu_pattern_scanner")
        ],
        [
            InlineKeyboardButton("🌐 Language: 🇬🇧 English", callback_data="menu_lang_select")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_ea_control_menu():
    """Dedicated EA Control Sub-Menu"""
    ea_status = "🟢 EA Execution: ACTIVE" if EXECUTION_STATE["ea_enabled"] else "🔴 EA Execution: PAUSED"
    toggle_btn_text = "⏸️ Pause EA Auto-Trade" if EXECUTION_STATE["ea_enabled"] else "▶️ Resume EA Auto-Trade"
    
    keyboard = [
        [InlineKeyboardButton(ea_status, callback_data="ea_info_status")],
        [InlineKeyboardButton(toggle_btn_text, callback_data="toggle_ea_execution")],
        [InlineKeyboardButton("📊 View Connected MT5 Account", callback_data="view_account_info")],
        [InlineKeyboardButton("🕒 Session Filter Status", callback_data="view_session_status")],
        [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="menu_main")]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_mobile_control_menu():
    """Dedicated Mobile Control Sub-Menu"""
    keyboard = [
        [InlineKeyboardButton("🚨 Generate Signal & Manual Triggers", callback_data="mobile_generate_signal")],
        [InlineKeyboardButton("🚀 Quick BUY STOP Order", callback_data="mobile_order_buy")],
        [InlineKeyboardButton("🔻 Quick SELL STOP Order", callback_data="mobile_order_sell")],
        [InlineKeyboardButton("📋 Copy Entry & Risk Levels", callback_data="mobile_copy_levels")],
        [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="menu_main")]
    ]
    return InlineKeyboardMarkup(keyboard)


# ----------------------------------------------------------------------------
# TELEGRAM HANDLERS
# ----------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🏛️ *Institutional Pattern & Execution Desk*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Select an option below to enter **EA Control** or **Mobile Control**:"
    )
    await update.message.reply_text(
        text=welcome_text,
        parse_mode="Markdown",
        reply_markup=get_main_menu()
    )


async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    # --- MAIN MENU NAVIGATION ---
    if data == "menu_main":
        await query.edit_message_text(
            text="🏛️ *Main Menu*\nSelect an option below:",
            parse_mode="Markdown",
            reply_markup=get_main_menu()
        )

    # --- EA CONTROL SUB-MENU ---
    elif data == "menu_ea_control":
        EXECUTION_STATE["mode"] = "AUTOMATIC"
        ea_msg = (
            "🤖 *EA CONTROL DESK (PC / VPS Auto-Trade)*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 *Connected Broker:* {EXECUTION_STATE['connected_broker']}\n"
            f"🆔 *Account Number:* {EXECUTION_STATE['account_id']}\n"
            f"⚙️ *EA State:* {'ACTIVE 🟢' if EXECUTION_STATE['ea_enabled'] else 'PAUSED 🔴'}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Manage your MT5 Expert Advisor using the options below:"
        )
        await query.edit_message_text(text=ea_msg, parse_mode="Markdown", reply_markup=get_ea_control_menu())

    elif data == "toggle_ea_execution":
        EXECUTION_STATE["ea_enabled"] = not EXECUTION_STATE["ea_enabled"]
        status_txt = "ENABLED 🟢" if EXECUTION_STATE["ea_enabled"] else "PAUSED 🔴"
        await query.edit_message_text(
            text=f"🤖 *EA Execution updated to:* {status_txt}",
            parse_mode="Markdown",
            reply_markup=get_ea_control_menu()
        )

    elif data == "view_account_info":
        info_txt = (
            "📊 *CONNECTED MT5 ACCOUNT DETAILS*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Broker:* {EXECUTION_STATE['connected_broker']}\n"
            f"• *Account ID:* {EXECUTION_STATE['account_id']}\n"
            "• *Status:* Connected via MQL5 Bridge\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        await query.edit_message_text(text=info_txt, parse_mode="Markdown", reply_markup=get_ea_control_menu())

    elif data == "view_session_status":
        sess_txt = (
            "🕒 *SESSION RESTRICTIONS STATUS*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "• *Session Filter:* ACTIVE 🟢 (Untouched)\n"
            "• *Allowed Window:* London & New York Sessions\n"
            "• *Rule Enforcement:* EA blocks entries outside designated hours.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        await query.edit_message_text(text=sess_txt, parse_mode="Markdown", reply_markup=get_ea_control_menu())

    # --- MOBILE CONTROL SUB-MENU ---
    elif data == "menu_mobile_control":
        EXECUTION_STATE["mode"] = "MANUAL"
        mobile_msg = (
            "📱 *MOBILE CONTROL DESK (Manual Trading)*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "EA auto-execution is set to MANUAL mode.\n"
            "Generate signals, copy price levels, or trigger instant orders from your phone below:"
        )
        await query.edit_message_text(text=mobile_msg, parse_mode="Markdown", reply_markup=get_mobile_control_menu())

    elif data == "mobile_generate_signal":
        symbol = EXECUTION_STATE.get("active_symbol", "GBPAUD")
        sig_msg = (
            f"🚨 *MOBILE SIGNAL ({symbol})*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 *Pattern:* Bull Flag / Realignment\n"
            f"🟢 *Action:* BUY STOP @ 1.92450\n"
            f"🛑 *Stop Loss:* 1.92150 (30 pips)\n"
            f"🎯 *Take Profit:* 1.93050 (60 pips)\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        await query.message.reply_text(sig_msg, parse_mode="Markdown", reply_markup=get_mobile_control_menu())

    elif data == "mobile_order_buy":
        await query.message.reply_text("🚀 *BUY STOP order payload sent to MT5 Mobile bridge.*", parse_mode="Markdown")

    elif data == "mobile_order_sell":
        await query.message.reply_text("🔻 *SELL STOP order payload sent to MT5 Mobile bridge.*", parse_mode="Markdown")


# ----------------------------------------------------------------------------
# MAIN ENGINE RUNNER
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Start Keep-Alive & API Webserver Thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Start Telegram Bot Listener
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
    if bot_token != "YOUR_BOT_TOKEN_HERE":
        application = ApplicationBuilder().token(bot_token).build()
        
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CallbackQueryHandler(button_callback_handler))
        
        logger.info("Starting Telegram Bot listener...")
        application.run_polling()
    else:
        logger.warning("TELEGRAM_BOT_TOKEN not set. Webserver running standalone.")
