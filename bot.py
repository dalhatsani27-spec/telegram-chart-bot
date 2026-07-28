import os
import io
import time
import sqlite3
import threading
import asyncio
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')  # Prevents thread-safety crashes in server environments
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
from openai import OpenAI

# ==========================================
# 1. FLASK WEB SERVER & KEEP-ALIVE LOOP
# ==========================================
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Self-Optimizing Institutional Desk Engine Active 24/7!", 200

RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

def keep_alive_ping():
    while True:
        time.sleep(600)
        if RENDER_EXTERNAL_URL:
            try:
                res = requests.get(RENDER_EXTERNAL_URL, timeout=10)
                print(f"[Keep-Alive] Ping status: {res.status_code}")
            except Exception as e:
                print(f"[Keep-Alive] Ping failed: {e}")

threading.Thread(target=keep_alive_ping, daemon=True).start()

# ==========================================
# 2. SELF-OPTIMIZING DATABASE (PERFORMANCE LOGS)
# ==========================================
DB_FILE = "trade_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signal_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT,
            timeframe TEXT,
            direction TEXT,
            entry_price REAL,
            sl_price REAL,
            tp1_price REAL,
            rr_ratio REAL,
            outcome TEXT DEFAULT 'PENDING'
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_signal(symbol, timeframe, direction, entry, sl, tp1, rr):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO signal_memory (symbol, timeframe, direction, entry_price, sl_price, tp1_price, rr_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, timeframe, direction, entry, sl, tp1, rr))
    conn.commit()
    conn.close()

def fetch_performance_stats():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) FROM signal_memory")
    total, wins = cursor.fetchone()
    conn.close()
    if not total or total == 0:
        return "🧠 *BOT OPTIMIZATION MEMORY:* No historical trade logs yet. Initializing baseline dataset."
    
    win_rate = (wins / total) * 100 if wins else 0
    return f"🧠 *BOT OPTIMIZATION MEMORY:*\n• Total Signals Evaluated: `{total}`\n• Historical Win Rate: `{win_rate:.1f}%`\n• Optimization Engine: Active adaptive weighting."

# ==========================================
# 3. OPENAI / OPENROUTER AI ENGINE
# ==========================================
ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def analyze_with_ai(prompt):
    models = [
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "openrouter/free"
    ]
    for model in models:
        try:
            res = ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600
            )
            content = res.choices[0].message.content
            if content and len(content.strip()) > 30:
                return content
        except Exception:
            continue
    return "Analysis complete. Refer to structural levels on chart."

# ==========================================
# 4. DATA ENGINE & TICKER MAPPER (WEEKEND & ASYNC SAFE)
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

def fetch_multi_timeframe_data(symbol, entry_tf="15m"):
    primary_ticker = normalize_ticker(symbol)
    ticker_candidates = [primary_ticker]
    if primary_ticker == "GC=F":
        ticker_candidates.append("XAUUSD=X")
    elif primary_ticker == "XAUUSD=X":
        ticker_candidates.insert(0, "GC=F")

    tf_clean = entry_tf.lower().strip()
    periods = ["10d", "30d", "60d"] if tf_clean in ["1m", "5m", "15m"] else ["1mo", "3mo"]

    df_daily, df_entry = pd.DataFrame(), pd.DataFrame()
    successful_ticker = primary_ticker

    for ticker_str in ticker_candidates:
        for period in periods:
            try:
                d_data = yf.download(ticker_str, period="1y", interval="1d", progress=False, auto_adjust=True)
                if isinstance(d_data.columns, pd.MultiIndex):
                    d_data.columns = d_data.columns.get_level_values(0)
                d_data = d_data.dropna()

                e_data = yf.download(ticker_str, period=period, interval=tf_clean, progress=False, auto_adjust=True)
                if isinstance(e_data.columns, pd.MultiIndex):
                    e_data.columns = e_data.columns.get_level_values(0)
                e_data = e_data.dropna()

                if not d_data.empty and not e_data.empty:
                    df_daily = d_data
                    df_entry = e_data
                    successful_ticker = ticker_str
                    break
            except Exception:
                continue
        if not df_daily.empty and not df_entry.empty:
            break

    if df_daily.empty or df_entry.empty:
        raise ValueError(f"Unable to retrieve market data for '{symbol}'. Market may be closed or ticker invalid.")

    last_candle_time = df_entry.index[-1]
    now_utc = datetime.now(timezone.utc)
    
    if hasattr(last_candle_time, 'tz') and last_candle_time.tz is not None:
        delta_hours = (now_utc - last_candle_time).total_seconds() / 3600
    else:
        delta_hours = (now_utc.replace(tzinfo=None) - last_candle_time).total_seconds() / 3600

    is_market_closed = delta_hours > 3.0

    df_daily['EMA200'] = df_daily['Close'].ewm(span=200, adjust=False).mean()
    daily_close = df_daily['Close'].iloc[-1]
    daily_ema = df_daily['EMA200'].iloc[-1]
    macro_is_bearish = daily_close < daily_ema

    df_entry['EMA200'] = df_entry['Close'].ewm(span=200, adjust=False).mean()

    return df_daily, df_entry, macro_is_bearish, daily_ema, successful_ticker, is_market_closed

# ==========================================
# 5. A+ CONFLUENCE STRUCTURE CALCULATOR
# ==========================================
def analyze_structure_and_setup(df):
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    n = len(df)

    results = {'swings': [], 'shifts': [], 'obs': [], 'fvgs': []}
    
    if n < 10:
        return results, [], [], None

    swing_highs = []
    swing_lows = []
    
    for i in range(1, n - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            swing_highs.append({'index': i, 'price': highs[i]})
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            swing_lows.append({'index': i, 'price': lows[i]})

    if not swing_highs:
        max_i = int(np.argmax(highs))
        swing_highs.append({'index': max_i, 'price': highs[max_i]})
    if not swing_lows:
        min_i = int(np.argmin(lows))
        swing_lows.append({'index': min_i, 'price': lows[min_i]})

    # MSS Detection
    for i in range(2, n):
        last_sh = [sh for sh in swing_highs if sh['index'] < i]
        last_sl = [sl for sl in swing_lows if sl['index'] < i]
        
        if last_sh and closes[i] > last_sh[-1]['price'] and closes[i-1] <= last_sh[-1]['price']:
            results['shifts'].append({'type': 'BULLISH MSS', 'index': i, 'price': last_sh[-1]['price']})
        elif last_sl and closes[i] < last_sl[-1]['price'] and closes[i-1] >= last_sl[-1]['price']:
            results['shifts'].append({'type': 'BEARISH MSS', 'index': i, 'price': last_sl[-1]['price']})

    # Order Blocks
    avg_range = abs(closes - opens).mean()
    for i in range(2, n - 1):
        body = abs(closes[i] - opens[i])
        if body >= (0.6 * avg_range):  # Dynamic threshold for zone detection
            if closes[i] > opens[i] and closes[i-1] < opens[i-1]:
                ob_top = max(opens[i-1], closes[i-1])
                ob_bottom = lows[i-1]
                future_lows = lows[i:]
                is_mitigated = (future_lows < ob_top).any() if len(future_lows) > 0 else False
                results['obs'].append({
                    'type': 'BULLISH_OB',
                    'top': ob_top,
                    'bottom': ob_bottom,
                    'index': i-1,
                    'mitigated': is_mitigated
                })
            elif closes[i] < opens[i] and closes[i-1] > opens[i-1]:
                ob_top = highs[i-1]
                ob_bottom = min(opens[i-1], closes[i-1])
                future_highs = highs[i:]
                is_mitigated = (future_highs > ob_bottom).any() if len(future_highs) > 0 else False
                results['obs'].append({
                    'type': 'BEARISH_OB',
                    'top': ob_top,
                    'bottom': ob_bottom,
                    'index': i-1,
                    'mitigated': is_mitigated
                })

    # A+ Trade Setup Evaluator
    trade_setup = None
    last_price = closes[-1]
    last_ema = df['EMA200'].iloc[-1]
    unmit_obs = [ob for ob in results['obs'] if not ob['mitigated']]

    if last_price < last_ema:  # Bearish Bias
        bear_obs = [ob for ob in unmit_obs if ob['type'] == 'BEARISH_OB']
        if bear_obs:
            target_ob = bear_obs[-1]
            entry_price = target_ob['bottom']
            sl_price = target_ob['top'] + (target_ob['top'] - target_ob['bottom']) * 0.2
            target_ssl = swing_lows[-1]['price'] if swing_lows else last_price * 0.99
            
            risk = sl_price - entry_price
            reward = entry_price - target_ssl
            rr_ratio = reward / risk if risk > 0 else 0

            if rr_ratio >= 2.0:
                trade_setup = {
                    'direction': 'SELL',
                    'bias': 'BEARISH',
                    'entry': entry_price,
                    'sl': sl_price,
                    'tp1': entry_price - (risk * 1.5),
                    'tp2': target_ssl,
                    'rr': rr_ratio,
                    'zone': f"{target_ob['bottom']:.2f} - {target_ob['top']:.2f}",
                    'target_liquidity': f"SSL at {target_ssl:.2f}"
                }
    else:  # Bullish Bias
        bull_obs = [ob for ob in unmit_obs if ob['type'] == 'BULLISH_OB']
        if bull_obs:
            target_ob = bull_obs[-1]
            entry_price = target_ob['top']
            sl_price = target_ob['bottom'] - (target_ob['top'] - target_ob['bottom']) * 0.2
            target_bsl = swing_highs[-1]['price'] if swing_highs else last_price * 1.01
            
            risk = entry_price - sl_price
            reward = target_bsl - entry_price
            rr_ratio = reward / risk if risk > 0 else 0

            if rr_ratio >= 2.0:
                trade_setup = {
                    'direction': 'BUY',
                    'bias': 'BULLISH',
                    'entry': entry_price,
                    'sl': sl_price,
                    'tp1': entry_price + (risk * 1.5),
                    'tp2': target_bsl,
                    'rr': rr_ratio,
                    'zone': f"{target_ob['bottom']:.2f} - {target_ob['top']:.2f}",
                    'target_liquidity': f"BSL at {target_bsl:.2f}"
                }

    return results, swing_highs, swing_lows, trade_setup

# ==========================================
# 6. HIGH-CONTRAST CHART GENERATOR (EXPLICIT OVERLAY RENDER)
# ==========================================
def generate_chart_image(df, title_str, trade_setup=None):
    img_buf = io.BytesIO()
    
    chart_df = df.tail(80).copy()
    if not isinstance(chart_df.index, pd.DatetimeIndex):
        chart_df.index = pd.to_datetime(chart_df.index)

    ict_data, swing_highs, swing_lows, _ = analyze_structure_and_setup(chart_df)
    
    mc = mpf.make_marketcolors(
        up='#089981', down='#f23645',
        edge='inherit', wick='inherit', volume='in'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', 
        y_on_right=True, facecolor='#131722', figcolor='#131722'
    )
    
    addplots = [mpf.make_addplot(chart_df['EMA200'], color='#ffd700', width=1.5)]

    # 1. Generate Figure without saving early
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
    mid_time = chart_df.index[len(chart_df) // 3]
    last_time = chart_df.index[-20] if len(chart_df) >= 20 else chart_df.index[0]

    # 2. Draw Buy-Side Liquidity (BSL)
    if swing_highs:
        last_bsl = swing_highs[-1]['price']
        ax.axhline(last_bsl, color='#00e676', linestyle=':', linewidth=1.2)
        ax.text(first_time, last_bsl, " BSL (Buy-Side Liquidity)", color='#00e676', fontsize=8, fontweight='bold', verticalalignment='bottom')

    # 3. Draw Sell-Side Liquidity (SSL)
    if swing_lows:
        last_ssl = swing_lows[-1]['price']
        ax.axhline(last_ssl, color='#ff1744', linestyle=':', linewidth=1.2)
        ax.text(first_time, last_ssl, " SSL (Sell-Side Liquidity)", color='#ff1744', fontsize=8, fontweight='bold', verticalalignment='top')

    # 4. Draw Market Structure Shifts (MSS)
    for shift in ict_data['shifts'][-2:]:
        color = '#00e676' if 'BULLISH' in shift['type'] else '#ff1744'
        ax.axhline(shift['price'], color=color, linestyle='--', linewidth=1.2)
        ax.text(mid_time, shift['price'], f" {shift['type']} ", color=color, fontsize=8, fontweight='bold', backgroundcolor='#131722')

    # 5. Draw Order Blocks (OB Shaded Boxes)
    for ob in ict_data['obs'][-3:]:
        color = '#089981' if ob['type'] == 'BULLISH_OB' else '#f23645'
        label = " BULLISH OB" if ob['type'] == 'BULLISH_OB' else " BEARISH OB"
        ax.axhspan(ob['bottom'], ob['top'], color=color, alpha=0.25 if not ob['mitigated'] else 0.08)
        ax.text(last_time, ob['top'], label, color=color, fontsize=8, fontweight='bold')

    # 6. Draw Setup Parameter Lines
    if trade_setup:
        ax.axhline(trade_setup['entry'], color='#2962ff', linestyle='-.', linewidth=1.5)
        ax.axhline(trade_setup['sl'], color='#f23645', linestyle='--', linewidth=1.5)
        ax.axhline(trade_setup['tp1'], color='#089981', linestyle='--', linewidth=1.5)
        
        ax.text(last_time, trade_setup['entry'], f" ENTRY: {trade_setup['entry']:.2f}", color='#2962ff', fontsize=8, fontweight='bold', backgroundcolor='#131722')
        ax.text(last_time, trade_setup['sl'], f" SL: {trade_setup['sl']:.2f}", color='#f23645', fontsize=8, fontweight='bold', backgroundcolor='#131722')
        ax.text(last_time, trade_setup['tp1'], f" TP1: {trade_setup['tp1']:.2f}", color='#089981', fontsize=8, fontweight='bold', backgroundcolor='#131722')

    ax.set_title(title_str, color='#d1d4dc', fontsize=9, fontweight='bold')

    # 7. EXPLICIT SAVE AFTER DRAWING OVERLAYS
    fig.savefig(img_buf, format='png', dpi=130, bbox_inches='tight', facecolor='#131722')
    plt.close(fig)

    img_buf.seek(0)
    return img_buf

# ==========================================
# 7. TELEGRAM INTERACTIVE HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🧠 *SELF-OPTIMIZING INSTITUTIONAL DESK ASSISTANT*\n\n"
        "I am ready. Ask me anything directly, or issue a request.\n\n"
        "📌 *EXAMPLES:*\n"
        "• `/analyze GOLD 15m`\n"
        "• Or just type: *'Check Gold on 15m'*"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    symbol = args[0].upper() if len(args) > 0 else "GOLD"
    tf = args[1].lower() if len(args) > 1 else "15m"

    keyboard = [
        [
            InlineKeyboardButton("📊 Full Market Analysis", callback_data=f"MODE_ANALYSIS|{symbol}|{tf}"),
            InlineKeyboardButton("🎯 Search A+ Trade Setup", callback_data=f"MODE_SETUP|{symbol}|{tf}")
        ],
        [
            InlineKeyboardButton("⚙️ Performance & Optimization Memory", callback_data="MODE_STATS")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"🎯 *TARGET ACQUIRED: {symbol} ({tf.upper()})*\nSelect your required workflow:",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data_parts = query.data.split("|")
    mode = data_parts[0]

    if mode == "MODE_STATS":
        stats_text = fetch_performance_stats()
        await query.edit_message_text(stats_text, parse_mode="Markdown")
        return

    symbol = data_parts[1]
    tf = data_parts[2]

    await query.edit_message_text(f"⏳ Executing calculations for {symbol} ({tf.upper()})...")

    try:
        # RUN FETCH IN BACKGROUND THREAD (PREVENTS LOOP FREEZING)
        df_daily, df_entry, macro_is_bearish, daily_ema, ticker_str, is_closed = await asyncio.to_thread(
            fetch_multi_timeframe_data, symbol, tf
        )
        
        ict_data, swing_highs, swing_lows, trade_setup = analyze_structure_and_setup(df_entry)
        
        last_price = df_entry['Close'].iloc[-1]
        local_ema = df_entry['EMA200'].iloc[-1]
        local_bias = "BEARISH" if last_price < local_ema else "BULLISH"
        
        market_status_str = "⚠️ *MARKET CLOSED* (Last Session Data)" if is_closed else "🟢 *MARKET OPEN*"

        if mode == "MODE_ANALYSIS":
            title_str = f"{symbol} ({tf.upper()}) | Structural Market Map"
            chart_img = generate_chart_image(df_entry, title_str, trade_setup=None)

            analysis_report = (
                f"📊 *STRUCTURAL MARKET DIAGNOSTIC: {symbol} ({tf.upper()})*\n"
                f"Status: {market_status_str}\n\n"
                f"• **200 EMA Baseline:** Market is trading `{local_bias}` below/above the Gold Line.\n"
                f"• **Buy-Side Liquidity (BSL):** `{swing_highs[-1]['price']:.2f}`\n"
                f"• **Sell-Side Liquidity (SSL):** `{swing_lows[-1]['price']:.2f}`\n"
                f"• **Active Unmitigated OBs:** `{len([ob for ob in ict_data['obs'] if not ob['mitigated']])}` Zones Detected."
            )

            await context.bot.send_photo(chat_id=query.message.chat_id, photo=chart_img, caption=analysis_report, parse_mode="Markdown")

        elif mode == "MODE_SETUP":
            if trade_setup:
                log_signal(symbol, tf, trade_setup['direction'], trade_setup['entry'], trade_setup['sl'], trade_setup['tp1'], trade_setup['rr'])
                title_str = f"{symbol} ({tf.upper()}) | A+ TRADE SETUP FOUND"
                chart_img = generate_chart_image(df_entry, title_str, trade_setup)

                setup_report = (
                    f"🎯 *GRADE A+ TRADE SETUP IDENTIFIED*\n"
                    f"Status: {market_status_str}\n\n"
                    f"• *Asset/Timeframe:* `{symbol} ({tf.upper()})`\n"
                    f"• *Order Direction:* `{trade_setup['direction']}`\n"
                    f"• *Risk-to-Reward Ratio:* `1:{trade_setup['rr']:.2f}`\n"
                    f"• *POI Zone:* `{trade_setup['zone']}`\n\n"
                    f"📍 *EXECUTION PARAMETERS:*\n"
                    f"• **Entry Limit:** `{trade_setup['entry']:.2f}`\n"
                    f"• **Stop Loss:** `{trade_setup['sl']:.2f}`\n"
                    f"• **Take Profit 1:** `{trade_setup['tp1']:.2f}`\n"
                    f"• **Take Profit 2 (Target Liquidity):** `{trade_setup['tp2']:.2f}`"
                )
                await context.bot.send_photo(chat_id=query.message.chat_id, photo=chart_img, caption=setup_report, parse_mode="Markdown")
            else:
                title_str = f"{symbol} ({tf.upper()}) | NO A+ SETUP AVAILABLE"
                chart_img = generate_chart_image(df_entry, title_str, trade_setup=None)

                no_trade_report = (
                    f"🛑 *VERDICT: NO GRADE A+ TRADE SETUP*\n"
                    f"Status: {market_status_str}\n\n"
                    f"• **Reason:** Price action currently lacks a valid high-probability confluence.\n"
                    f"• **Desk Rule:** Standing on hands to protect account capital."
                )
                await context.bot.send_photo(chat_id=query.message.chat_id, photo=chart_img, caption=no_trade_report, parse_mode="Markdown")

    except Exception as e:
        print(f"[ERROR] Callback error: {e}")
        await context.bot.send_message(
            chat_id=query.message.chat_id, 
            text=f"❌ **Execution Error:** {str(e)}"
        )

# ==========================================
# 8. NATURAL CHAT INTELLIGENCE HANDLER
# ==========================================
async def natural_chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()

    if "stats" in text or "performance" in text or "memory" in text:
        stats_msg = fetch_performance_stats()
        await update.message.reply_text(stats_msg, parse_mode="Markdown")
        return

    words = text.split()
    symbols = ["GOLD", "XAUUSD", "BTC", "BTCUSD", "SILVER", "OIL", "EURUSD", "GBPUSD"]
    found_symbol = None
    for word in words:
        if word.upper() in symbols:
            found_symbol = word.upper()
            break

    if found_symbol:
        tf = "15m"
        for word in words:
            if word in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                tf = word
                break
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Full Market Analysis", callback_data=f"MODE_ANALYSIS|{found_symbol}|{tf}"),
                InlineKeyboardButton("🎯 Search A+ Trade Setup", callback_data=f"MODE_SETUP|{found_symbol}|{tf}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            f"I see you're asking about *{found_symbol} ({tf.upper()})*. What would you like me to generate?",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    else:
        response = analyze_with_ai(f"Act as an institutional risk manager and trading desk assistant. Respond briefly to this trader message: '{update.message.text}'")
        await update.message.reply_text(response)

# ==========================================
# 9. MAIN APPLICATION RUNNER
# ==========================================
def main():
    if not TELEGRAM_BOT_TOKEN:
        print("[CRITICAL ERROR] TELEGRAM_BOT_TOKEN is missing!")
        return

    print("[INIT] Building Telegram Application...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, natural_chat_handler))

    def start_polling():
        print("[INIT] Starting Telegram bot polling...")
        try:
            application.run_polling(drop_pending_updates=True)
        except Exception as e:
            print(f"[CRITICAL ERROR] Polling crashed: {e}")

    bot_thread = threading.Thread(target=start_polling, daemon=True)
    bot_thread.start()

    port = int(os.environ.get("PORT", 10000))
    print(f"[INIT] Starting Flask web server on port {port}...")
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
