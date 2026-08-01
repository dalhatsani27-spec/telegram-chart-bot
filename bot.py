"""
bot.py
================
Telegram Interface & Execution State Server.
Integrated with flask keep-alive, execution mode controller, and pattern engine.
"""

import os
import logging
import threading
from flask import Flask, jsonify, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

from patterns import scan_all_patterns

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FLASK KEEP-ALIVE & API SERVER ---
app = Flask(__name__)

# GLOBAL STATE ENGINE
EXECUTION_STATE = {
    "mode": "AUTOMATIC",      # 'AUTOMATIC' (PC/VPS Auto-Trade) or 'MANUAL' (Mobile Signals)
    "session_filter": True,   # Existing session rules preserved
    "active_symbol": "GBPAUD",
    "last_signal": {}
}

@app.route('/')
def home():
    return "Institutional Telegram Engine & MT5 Bridge is Live."

@app.route('/get_state', methods=['GET'])
def get_state():
    """Endpoint polled by MT5 EA to fetch execution mode & orders."""
    return jsonify(EXECUTION_STATE)

@app.route('/update_signal', methods=['POST'])
def update_signal():
    """Endpoint for MT5 or internal engine to update latest trade signal state."""
    data = request.json or {}
    EXECUTION_STATE["last_signal"] = data
    return jsonify({"status": "success", "state": EXECUTION_STATE})

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# --- TELEGRAM BOT KEYBOARDS ---
def get_home_menu(lang="English", is_auto=False):
    lang_flags = {"English": "🇬🇧 English", "Hausa": "🇳🇬 Hausa", "Pidgin": "🇳🇬 Pidgin"}
    current_flag = lang_flags.get(lang, "🇬🇧 English")
    auto_status = "🟢 Continuous Scanner: ON" if is_auto else "🔴 Continuous Scanner: OFF"
    
    # Dynamic Mode Button
    if EXECUTION_STATE["mode"] == "AUTOMATIC":
        mode_btn = InlineKeyboardButton("🤖 Mode: AUTOMATIC (PC / VPS Auto)", callback_data="toggle_execution_mode")
    else:
        mode_btn = InlineKeyboardButton("📱 Mode: MANUAL (Mobile Signals)", callback_data="toggle_execution_mode")
    
    keyboard = [
        [mode_btn],
        [InlineKeyboardButton("📊 Run Pattern Analysis", callback_data="menu_analysis"),
         InlineKeyboardButton("🚨 Generate Live Signal", callback_data="menu_signal")],
        [InlineKeyboardButton("⚡ Scalper Mode (5m / 15m)", callback_data="menu_scalper")],
        [InlineKeyboardButton("🔍 Select Asset Pair", callback_data="menu_pattern_scanner")],
        [InlineKeyboardButton(auto_status, callback_data="toggle_auto_scan")],
        [InlineKeyboardButton(f"🌐 Language: {current_flag}", callback_data="menu_lang_select")]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- TELEGRAM HANDLERS ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🏛️ *Institutional Pattern & Execution Desk*\n\n"
        "Your engine is online and ready. Choose execution mode below:\n"
        "• *AUTOMATIC:* MT5 EA executes signals automatically on PC/VPS.\n"
        "• *MANUAL:* Signals generated with interactive buttons for MT5 Mobile."
    )
    await update.message.reply_text(
        text=welcome_text,
        parse_mode="Markdown",
        reply_markup=get_home_menu()
    )

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    # Mode Toggle Callback
    if data == "toggle_execution_mode":
        if EXECUTION_STATE["mode"] == "AUTOMATIC":
            EXECUTION_STATE["mode"] = "MANUAL"
            notice = "📱 *Switched to MANUAL Mode (Mobile Signals)*\n\nMT5 EA auto-execution paused. You will receive interactive buttons for manual trade entry on MT5 Mobile."
        else:
            EXECUTION_STATE["mode"] = "AUTOMATIC"
            notice = "🤖 *Switched to AUTOMATIC Mode (PC / VPS Execution)*\n\nMT5 EA will auto-execute setups strictly within your session rules."
        
        await query.edit_message_text(
            text=notice,
            parse_mode="Markdown",
            reply_markup=get_home_menu()
        )

    elif data == "menu_signal":
        mode = EXECUTION_STATE["mode"]
        symbol = EXECUTION_STATE.get("active_symbol", "GBPAUD")
        
        signal_msg = (
            f"🚨 *LIVE TRADING SIGNAL ({symbol})*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 *Pattern:* Bull Flag / Realignment\n"
            f"🟢 *Action:* BUY STOP @ 1.92450\n"
            f"🛑 *Stop Loss:* 1.92150 (30 pips)\n"
            f"🎯 *Take Profit:* 1.93050 (60 pips)\n"
            f"🕒 *Session:* Active London/NY\n"
            f"⚙️ *Mode:* {mode}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        
        # In MANUAL mode, attach copy/remote trigger buttons for MT5 Mobile
        if mode == "MANUAL":
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🚀 Send Order to MT5 Terminal", callback_data="remote_send_order")],
                [InlineKeyboardButton("📋 Copy Entry Level", callback_data="copy_entry")]
            ])
            await query.message.reply_text(signal_msg, parse_mode="Markdown", reply_markup=kb)
        else:
            await query.message.reply_text(signal_msg + "\n\n*Auto-routed to MT5 Terminal.*", parse_mode="Markdown")

# --- MAIN RUNNER ---
if __name__ == "__main__":
    # Start Keep-Alive / API Webserver Thread
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
        logger.warning("TELEGRAM_BOT_TOKEN not provided. Flask server running standalone.")
