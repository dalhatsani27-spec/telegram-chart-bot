import os
import io
import time
import threading
import asyncio
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# ==========================================
# 1. FLASK WEB SERVER & RENDER KEEP-ALIVE
# ==========================================
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Institutional Desk Engine Active!", 200

RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

def keep_alive_ping():
    while True:
        time.sleep(600)
        if RENDER_EXTERNAL_URL:
            try:
                requests.get(RENDER_EXTERNAL_URL, timeout=10)
            except Exception:
                pass

threading.Thread(target=keep_alive_ping, daemon=True).start()

# ==========================================
# 2. TICKER NORMALIZER & DATA FETCH
# ==========================================
def normalize_ticker(symbol):
    symbol = symbol.strip().upper()
    alias_map = {
        "GOLD": "GC=F", "XAUUSD": "GC=F",
        "SILVER": "SI=F", "XAGUSD": "SI=F",
        "OIL": "CL=F", "USOIL": "CL=F",
        "BTC": "BTC-USD", "BTCUSD": "BTC-USD"
    }
    if symbol in alias_map:
        return alias_map[symbol]
    if len(symbol) == 6 and not symbol.endswith("=X"):
        return f"{symbol}=X"
    return symbol

def fetch_data(symbol, entry_tf="15m"):
    ticker = normalize_ticker(symbol)
    tf_clean = entry_tf.lower().strip()
    period = "10d" if tf_clean in ["1m", "5m", "15m"] else "60d"

    df = yf.download(ticker, period=period, interval=tf_clean, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    if df.empty:
        raise ValueError(f"No market data retrieved for {symbol}.")

    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df, ticker

# ==========================================
# 3. STRUCTURE, SWINGS & TRENDLINE CALCULATOR
# ==========================================
def analyze_market_structure(df):
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    n = len(df)

    swing_highs = []
    swing_lows = []

    for i in range(2, n - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append({'index': i, 'price': highs[i]})
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append({'index': i, 'price': lows[i]})

    if not swing_highs:
        max_i = int(np.argmax(highs))
        swing_highs.append({'index': max_i, 'price': highs[max_i]})
    if not swing_lows:
        min_i = int(np.argmin(lows))
        swing_lows.append({'index': min_i, 'price': lows[min_i]})

    # Order Blocks (OB)
    obs = []
    avg_range = abs(closes - opens).mean()
    for i in range(2, n - 1):
        body = abs(closes[i] - opens[i])
        if body >= (0.6 * avg_range):
            if closes[i] > opens[i] and closes[i-1] < opens[i-1]:
                obs.append({'type': 'BULLISH', 'top': max(opens[i-1], closes[i-1]), 'bottom': lows[i-1]})
            elif closes[i] < opens[i] and closes[i-1] > opens[i-1]:
                obs.append({'type': 'BEARISH', 'top': highs[i-1], 'bottom': min(opens[i-1], closes[i-1])})

    return swing_highs, swing_lows, obs

# ==========================================
# 4. CHART GENERATOR WITH TRENDLINES & SMC
# ==========================================
def generate_chart_image(df, title_str):
    img_buf = io.BytesIO()
    chart_df = df.tail(80).copy()
    if not isinstance(chart_df.index, pd.DatetimeIndex):
        chart_df.index = pd.to_datetime(chart_df.index)

    swing_highs, swing_lows, obs = analyze_market_structure(chart_df)

    mc = mpf.make_marketcolors(
        up='#089981', down='#f23645',
        edge='inherit', wick='inherit', volume='in'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', 
        y_on_right=True, facecolor='#131722', figcolor='#131722'
    )

    addplots = [mpf.make_addplot(chart_df['EMA200'], color='#ffd700', width=1.5)]

    fig, axlist = mpf.plot(
        chart_df,
        type='candle',
        style=style,
        volume=False,
        addplot=addplots,
        returnfig=True
    )

    ax = axlist[0]
    first_time = chart_df.index[2]
    last_time = chart_df.index[-15] if len(chart_df) >= 15 else chart_df.index[0]

    # Draw Buy-Side / Sell-Side Liquidity Levels
    if swing_highs:
        bsl = swing_highs[-1]['price']
        ax.axhline(bsl, color='#00e676', linestyle=':', linewidth=1.2)
        ax.text(first_time, bsl, " BSL (Buy-Side Liquidity)", color='#00e676', fontsize=8, fontweight='bold', verticalalignment='bottom')

    if swing_lows:
        ssl = swing_lows[-1]['price']
        ax.axhline(ssl, color='#ff1744', linestyle=':', linewidth=1.2)
        ax.text(first_time, ssl, " SSL (Sell-Side Liquidity)", color='#ff1744', fontsize=8, fontweight='bold', verticalalignment='top')

    # Draw Resistance Trendline across Swing Highs
    if len(swing_highs) >= 2:
        sh1, sh2 = swing_highs[-2], swing_highs[-1]
        x_vals = [sh1['index'], sh2['index']]
        y_vals = [sh1['price'], sh2['price']]
        slope = (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0]) if (x_vals[1] - x_vals[0]) != 0 else 0
        
        x_ext = np.arange(sh1['index'], len(chart_df))
        y_ext = y_vals[0] + slope * (x_ext - sh1['index'])
        ax.plot(x_ext, y_ext, color='#00b0ff', linestyle='--', linewidth=1.5)

    # Draw Support Trendline across Swing Lows
    if len(swing_lows) >= 2:
        sl1, sl2 = swing_lows[-2], swing_lows[-1]
        x_vals = [sl1['index'], sl2['index']]
        y_vals = [sl1['price'], sl2['price']]
        slope = (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0]) if (x_vals[1] - x_vals[0]) != 0 else 0
        
        x_ext = np.arange(sl1['index'], len(chart_df))
        y_ext = y_vals[0] + slope * (x_ext - sl1['index'])
        ax.plot(x_ext, y_ext, color='#ff9100', linestyle='--', linewidth=1.5)

    # Draw Order Blocks (OB)
    for ob in obs[-2:]:
        color = '#089981' if ob['type'] == 'BULLISH' else '#f23645'
        label = " BULLISH OB" if ob['type'] == 'BULLISH' else " BEARISH OB"
        ax.axhspan(ob['bottom'], ob['top'], color=color, alpha=0.22)
        ax.text(last_time, ob['top'], label, color=color, fontsize=8, fontweight='bold')

    ax.set_title(title_str, color='#d1d4dc', fontsize=9, fontweight='bold')

    fig.savefig(img_buf, format='png', dpi=130, bbox_inches='tight', facecolor='#131722')
    plt.close(fig)

    img_buf.seek(0)
    return img_buf

# ==========================================
# 5. TELEGRAM HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🏛 *WELCOME TO THE INSTITUTIONAL DESK*\n\n"
        "_Precision market structure. Zero noise._\n\n"
        "I am configured to map Smart Money Concepts (SMC), key liquidity sweeps, "
        "Order Blocks, and automated trendline channels in real time.\n\n"
        "⚡ *QUICK START:*\n"
        "• `/analyze GOLD 15m`\n"
        "• `/analyze BTC 5m`\n"
        "• Or simply type: *'Check Gold on 15m'*\n\n"
        "Select an asset or send a query to begin."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    symbol = args[0].upper() if len(args) > 0 else "GOLD"
    tf = args[1].lower() if len(args) > 1 else "15m"

    keyboard = [[
        InlineKeyboardButton("📊 Structural & Trendline Analysis", callback_data=f"MODE_ANALYSIS|{symbol}|{tf}")
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"🎯 *TARGET: {symbol} ({tf.upper()})*\nClick below to fetch chart:",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data_parts = query.data.split("|")
    symbol = data_parts[1]
    tf = data_parts[2]

    await query.edit_message_text(f"⏳ Processing {symbol} ({tf.upper()})...")

    try:
        df, ticker_str = await asyncio.to_thread(fetch_data, symbol, tf)
        swing_highs, swing_lows, obs = analyze_market_structure(df)

        last_price = df['Close'].iloc[-1]
        ema_val = df['EMA200'].iloc[-1]
        bias = "BULLISH" if last_price > ema_val else "BEARISH"

        title_str = f"{symbol} ({tf.upper()}) | Trendlines & Market Structure"
        chart_img = generate_chart_image(df, title_str)

        report = (
            f"📊 *DIAGNOSTIC: {symbol} ({tf.upper()})*\n\n"
            f"• **Trend Bias (200 EMA):** `{bias}`\n"
            f"• **Buy-Side Liquidity (BSL):** `{swing_highs[-1]['price']:.2f}`\n"
            f"• **Sell-Side Liquidity (SSL):** `{swing_lows[-1]['price']:.2f}`\n"
            f"• **Detected Order Blocks:** `{len(obs)}` active zones."
        )

        await context.bot.send_photo(chat_id=query.message.chat_id, photo=chart_img, caption=report, parse_mode="Markdown")

    except Exception as e:
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"❌ **Error:** {str(e)}"
        )

async def natural_chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    words = text.split()
    symbols = ["GOLD", "XAUUSD", "BTC", "BTCUSD", "SILVER", "OIL", "EURUSD", "GBPUSD"]
    found_symbol = next((w.upper() for w in words if w.upper() in symbols), None)

    if found_symbol:
        tf = "15m"
        for w in words:
            if w in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                tf = w
                break
        keyboard = [[InlineKeyboardButton("📊 Generate Chart", callback_data=f"MODE_ANALYSIS|{found_symbol}|{tf}")]]
        await update.message.reply_text(
            f"Analyze *{found_symbol} ({tf.upper()})*?",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text("Send a ticker command like `/analyze GOLD 15m` to generate analysis.")

# ==========================================
# 6. ASYNC RUNNER (MAIN THREAD SAFE)
# ==========================================
def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)

async def main():
    if not TELEGRAM_BOT_TOKEN:
        print("[CRITICAL ERROR] TELEGRAM_BOT_TOKEN missing!")
        return

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, natural_chat_handler))

    async with application:
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        print("🟢 Bot Online & Running Cleanly.")
        await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
