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
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from openai import OpenAI

# ==========================================
# 1. FLASK WEB SERVER & KEEP-ALIVE LOOP
# ==========================================
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Multi-Timeframe POI Chart Analyzer & Scalp Signal Bot is Active 24/7!", 200

# Environment variables
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MY_CHAT_ID = os.environ.get("MY_CHAT_ID")

def keep_alive_ping():
    """Pings Flask endpoint every 10 minutes to prevent server sleeping."""
    while True:
        time.sleep(600)
        if RENDER_EXTERNAL_URL:
            try:
                res = requests.get(RENDER_EXTERNAL_URL, timeout=10)
                print(f"[Keep-Alive] Ping status: {res.status_code}")
            except Exception as e:
                print(f"[Keep-Alive] Ping failed: {e}")

threading.Thread(target=keep_alive_ping, daemon=True).start()

# Track recent signals to avoid duplicate alert spam
SENT_SIGNALS_CACHE = set()

# ==========================================
# 2. OPENROUTER / AI VISION SETUP
# ==========================================
ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY or "dummy_key",
)

def fetch_ai_analysis(prompt):
    """Fallback model chain for high-reliability text analysis."""
    if not OPENROUTER_API_KEY:
        return (
            "🎯 *AI ANALYSIS & MARKET STORY*\n\n"
            "1. **Overall Bias:** Multi-Timeframe Structural Alignment Active.\n"
            "2. **Macro Context:** Reference the 200 EMA (Gold Line) for major trend direction.\n"
            "3. **Mapped POIs:** Green zones represent Bullish OB/FVG demand pools; Red zones represent Bearish OB/FVG supply pools."
        )

    vision_models = [
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "openrouter/free"
    ]
    
    for model in vision_models:
        try:
            response = ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600
            )
            content = response.choices[0].message.content
            if content and "User Safety:" not in content and len(content.strip()) > 30:
                return content
        except Exception:
            continue
            
    return (
        "🎯 *AI ANALYSIS & MARKET STORY*\n\n"
        "1. **Overall Bias:** Multi-Timeframe Structural Alignment Active.\n"
        "2. **Macro Context:** Reference the 200 EMA (Gold Line) for major trend direction.\n"
        "3. **Mapped POIs:** Green zones represent Bullish OB/FVG demand pools; Red zones represent Bearish OB/FVG supply pools."
    )

# ==========================================
# 3. RESILIENT TICKER ALIAS & DATA ENGINE
# ==========================================
def normalize_ticker(symbol):
    """Maps shorthand terms to clean Yahoo Finance tickers with GC=F gold feed."""
    symbol = symbol.strip().upper()
    
    alias_map = {
        "GOLD": "GC=F",      
        "XAUUSD": "GC=F",    
        "SILVER": "SI=F",
        "XAGUSD": "SI=F",
        "OIL": "CL=F",
        "USOIL": "CL=F",
        "BTC": "BTC-USD",
        "BTCUSD": "BTC-USD"
    }
    
    if symbol in alias_map:
        return alias_map[symbol]
    
    if len(symbol) == 6 and not symbol.endswith("=X"):
        return f"{symbol}=X"
        
    return symbol

def fetch_multi_timeframe_data(symbol, entry_tf="15m"):
    """Fetches Daily macro context alongside execution timeframe data cleanly with fallbacks."""
    primary_ticker = normalize_ticker(symbol)
    
    ticker_candidates = [primary_ticker]
    if primary_ticker == "GC=F":
        ticker_candidates.append("XAUUSD=X")
    elif primary_ticker == "XAUUSD=X":
        ticker_candidates.insert(0, "GC=F")

    tf_clean = entry_tf.lower().strip()
    entry_period = "5d" if tf_clean in ["1m", "5m"] else "10d"

    df_daily = pd.DataFrame()
    df_entry = pd.DataFrame()
    successful_ticker = primary_ticker

    for ticker_str in ticker_candidates:
        try:
            d_data = yf.download(ticker_str, period="1y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(d_data.columns, pd.MultiIndex):
                d_data.columns = d_data.columns.get_level_values(0)
            d_data = d_data.dropna()

            e_data = yf.download(ticker_str, period=entry_period, interval=tf_clean, progress=False, auto_adjust=True)
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

    if df_daily.empty or df_entry.empty:
        raise ValueError(f"Unable to retrieve market data for '{symbol}'.")

    df_daily['EMA200'] = df_daily['Close'].ewm(span=200, adjust=False).mean()
    daily_close = df_daily['Close'].iloc[-1]
    daily_ema = df_daily['EMA200'].iloc[-1]
    macro_is_bearish = daily_close < daily_ema

    df_entry['EMA200'] = df_entry['Close'].ewm(span=200, adjust=False).mean()

    return df_daily, df_entry, macro_is_bearish, daily_ema, successful_ticker

# ==========================================
# 4. ALGORITHMIC ICT POI DETECTORS (FVG / OB / BB)
# ==========================================
def detect_fvg(df):
    """Detects active Bullish and Bearish Fair Value Gaps (3-candle imbalance)."""
    fvgs = []
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            bottom = df['High'].iloc[i-2]
            top = df['Low'].iloc[i]
            fvgs.append({'type': 'BULLISH_FVG', 'top': top, 'bottom': bottom, 'index': i})
            
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            top = df['Low'].iloc[i-2]
            bottom = df['High'].iloc[i]
            fvgs.append({'type': 'BEARISH_FVG', 'top': top, 'bottom': bottom, 'index': i})
            
    return fvgs

def detect_order_blocks(df):
    """Detects recent Bullish and Bearish Order Blocks (OB) & Breaker Blocks (BB)."""
    obs = []
    
    for i in range(5, len(df) - 1):
        body_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        avg_body = abs(df['Close'].iloc[i-5:i] - df['Open'].iloc[i-5:i]).mean()
        
        if df['Close'].iloc[i] > df['Open'].iloc[i] and body_size > (1.5 * avg_body):
            for j in range(i-1, i-4, -1):
                if df['Close'].iloc[j] < df['Open'].iloc[j]:
                    ob_top = max(df['Open'].iloc[j], df['Close'].iloc[j])
                    ob_bottom = df['Low'].iloc[j]
                    
                    current_close = df['Close'].iloc[-1]
                    if current_close < ob_bottom:
                        obs.append({'type': 'BEARISH_BB', 'top': ob_top, 'bottom': ob_bottom, 'index': j})
                    else:
                        obs.append({'type': 'BULLISH_OB', 'top': ob_top, 'bottom': ob_bottom, 'index': j})
                    break

        elif df['Close'].iloc[i] < df['Open'].iloc[i] and body_size > (1.5 * avg_body):
            for j in range(i-1, i-4, -1):
                if df['Close'].iloc[j] > df['Open'].iloc[j]:
                    ob_top = df['High'].iloc[j]
                    ob_bottom = min(df['Open'].iloc[j], df['Close'].iloc[j])
                    
                    current_close = df['Close'].iloc[-1]
                    if current_close > ob_top:
                        obs.append({'type': 'BULLISH_BB', 'top': ob_top, 'bottom': ob_bottom, 'index': j})
                    else:
                        obs.append({'type': 'BEARISH_OB', 'top': ob_top, 'bottom': ob_bottom, 'index': j})
                    break

    return obs

# ==========================================
# 5. DYNAMIC SIGNAL FORMATTER (KING STYLE)
# ==========================================
def format_scalp_signal(symbol, direction, entry_price):
    """Formats high-impact Telegram scalping signals with dynamic TP/SL levels."""
    direction_upper = direction.upper()
    emoji = "💥" if direction_upper == "BUY" else "🔻"

    # Determine asset step size based on price magnitude
    if "BTC" in symbol:
        tp_step = 250.0
        sl_dist = 800.0
        decimals = 1
    elif any(k in symbol for k in ["GOLD", "XAU", "GC=F"]):
        tp_step = 3.0
        sl_dist = 10.0
        decimals = 2
    elif "JPY" in symbol:
        tp_step = 0.20
        sl_dist = 0.60
        decimals = 3
    else:  # Standard Forex (EURUSD, GBPUSD)
        tp_step = 0.0010  # 10 pips
        sl_dist = 0.0030  # 30 pips
        decimals = 5

    if direction_upper == "BUY":
        tp1 = entry_price + (tp_step * 1)
        tp2 = entry_price + (tp_step * 2)
        tp3 = entry_price + (tp_step * 3)
        tp4 = entry_price + (tp_step * 4)
        tp5 = entry_price + (tp_step * 5)
        sl  = entry_price - sl_dist
    else:  # SELL
        tp1 = entry_price - (tp_step * 1)
        tp2 = entry_price - (tp_step * 2)
        tp3 = entry_price - (tp_step * 3)
        tp4 = entry_price - (tp_step * 4)
        tp5 = entry_price - (tp_step * 5)
        sl  = entry_price + sl_dist

    fmt = f"{{:.{decimals}f}}"

    signal_text = (
        f"🔜 *𝐆𝐄𝐓 𝐑𝐄𝐀𝐃𝐘 𝐒𝐈𝐆𝐍𝐀𝐋𝐒 𝐂𝐎𝐌𝐈𝐍𝐆 𝐒𝐎𝐎𝐍* 🔜\n"
        f"‼️‼️‼️‼️‼️‼️‼️\n\n"
        f"{emoji} *{symbol} {direction_upper}* {fmt.format(entry_price)}\n\n"
        f"✔️ *TP1.* {fmt.format(tp1)}\n"
        f"✔️ *TP2.* {fmt.format(tp2)}\n"
        f"✔️ *TP3.* {fmt.format(tp3)}\n"
        f"✔️ *TP4.* {fmt.format(tp4)}\n"
        f"✔️ *TP5.* {fmt.format(tp5)}\n\n"
        f"❌ *SL.*  {fmt.format(sl)}📌\n\n"
        f"⚠️ *Use Money Management*"
    )
    return signal_text

# ==========================================
# 6. CHART GENERATOR
# ==========================================
def generate_chart_image(df, title_str, is_long=True):
    """Renders dark-themed chart with 200 EMA (Gold Line) and mapped FVG, OB, & BB POI Zones."""
    img_buf = io.BytesIO()
    chart_df = df.tail(80).copy()
    
    mc = mpf.make_marketcolors(
        up='#089981', down='#f23645',
        edge='inherit', wick='inherit', volume='in'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', 
        y_on_right=True, facecolor='#131722', figcolor='#131722'
    )
    
    addplots = [
        mpf.make_addplot(chart_df['EMA200'], color='#ffd700', width=1.5)
    ]

    recent_high = chart_df['High'].max()
    recent_low = chart_df['Low'].min()
    price_range = recent_high - recent_low

    if is_long:
        entry = recent_low
        risk_dist = max((recent_high - recent_low) * 0.12, price_range * 0.04)
        sl = entry - risk_dist
        tp = recent_high
    else:
        entry = recent_high
        risk_dist = max((recent_high - recent_low) * 0.12, price_range * 0.04)
        sl = entry + risk_dist
        tp = recent_low

    hlines_dict = dict(
        hlines=[entry, sl, tp],
        colors=['#2962ff', '#e53935', '#43a047'],
        linestyle=['-.', '--', '--'],
        linewidths=[1.2, 1.0, 1.0]
    )

    fig, axlist = mpf.plot(
        chart_df,
        type='candle',
        style=style,
        volume=False,
        hlines=hlines_dict,
        addplot=addplots,
        savefig=dict(fname=img_buf, dpi=130),
        returnfig=True
    )

    ax = axlist[0]

    fvgs = detect_fvg(chart_df)
    obs = detect_order_blocks(chart_df)

    recent_fvgs = fvgs[-2:] if len(fvgs) >= 2 else fvgs
    for fvg in recent_fvgs:
        if fvg['type'] == 'BULLISH_FVG':
            ax.axhspan(fvg['bottom'], fvg['top'], color='#089981', alpha=0.25, hatch='//')
            ax.text(0, fvg['top'], " Bullish FVG", color='#089981', fontsize=7, verticalalignment='bottom')
        elif fvg['type'] == 'BEARISH_FVG':
            ax.axhspan(fvg['bottom'], fvg['top'], color='#f23645', alpha=0.25, hatch='\\\\')
            ax.text(0, fvg['bottom'], " Bearish FVG", color='#f23645', fontsize=7, verticalalignment='top')

    recent_obs = obs[-2:] if len(obs) >= 2 else obs
    for ob in recent_obs:
        if ob['type'] == 'BULLISH_OB':
            ax.axhspan(ob['bottom'], ob['top'], color='#00e676', alpha=0.30)
            ax.text(len(chart_df)-15, ob['top'], " Bullish OB", color='#00e676', fontsize=8, fontweight='bold')
        elif ob['type'] == 'BEARISH_OB':
            ax.axhspan(ob['bottom'], ob['top'], color='#ff1744', alpha=0.30)
            ax.text(len(chart_df)-15, ob['bottom'], " Bearish OB", color='#ff1744', fontsize=8, fontweight='bold')
        elif ob['type'] == 'BEARISH_BB':
            ax.axhspan(ob['bottom'], ob['top'], color='#ff9100', alpha=0.30)
            ax.text(len(chart_df)-15, ob['bottom'], " Breaker Block (BB)", color='#ff9100', fontsize=8, fontweight='bold')

    ax.set_title(title_str, color='#d1d4dc', fontsize=9, fontweight='bold')
    img_buf.seek(0)
    return img_buf

# ==========================================
# 7. TELEGRAM BOT HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🤖 *MULTI-PAIR AUTOMATED SCALPER BOT*\n\n"
        "I monitor multiple pairs (GOLD, EURUSD, GBPUSD, BTC, etc.) for high-probability 200 EMA realignment scalp setups.\n\n"
        "📌 *COMMANDS:*\n"
        "• `/analyze [SYMBOL] [TIMEFRAME]` - Request chart and AI analysis.\n"
        "  _Example:_ `/analyze XAUUSD 5m`\n"
        "• `/signal [SYMBOL] [BUY/SELL] [PRICE]` - Generate a manual signal prompt."
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    
    raw_symbol = args[0].upper() if len(args) > 0 else "GOLD"
    tf = args[1] if len(args) > 1 else "15m"
    
    status_msg = await update.message.reply_text(f"📊 Mapping POI Zones for {raw_symbol}...")
    
    try:
        df_daily, df_entry, macro_is_bearish, daily_ema, ticker_str = fetch_multi_timeframe_data(raw_symbol, tf)
        
        last_price = df_entry['Close'].iloc[-1]
        local_ema = df_entry['EMA200'].iloc[-1]
        local_is_bearish = last_price < local_ema
        
        macro_bias = "BEARISH" if macro_is_bearish else "BULLISH"
        local_bias = "BEARISH" if local_is_bearish else "BULLISH"

        fvgs = detect_fvg(df_entry)
        obs = detect_order_blocks(df_entry)
        
        fvg_summary = f"{len(fvgs)} Active FVG(s)" if fvgs else "No immediate FVG"
        ob_summary = f"{len(obs)} Active OB/BB Zone(s)" if obs else "No immediate Order Blocks"

        prompt = f"""
        You are an ICT price-action scalping trader.
        Context:
        - Symbol: {raw_symbol}
        - Macro Bias: {macro_bias} (Daily 200 EMA)
        - Execution Bias ({tf}): {local_bias}
        - Structural POIs: {fvg_summary} | {ob_summary}
        
        Format output:
        🎯 AI ANALYSIS & MARKET STORY
        1. Macro Context: {macro_bias}
        2. Execution Bias ({tf}): {local_bias}
        3. Structural POI Reaction: Detail active FVG or OB levels.
        4. Realignment Setup: Next probable scalp trigger.
        """

        analysis_text = fetch_ai_analysis(prompt)

        is_long = not local_is_bearish
        title_str = f"{raw_symbol} ({tf.upper()}) | Trend: {local_bias} | 200 EMA Gold Line"
        chart_img = generate_chart_image(df_entry, title_str, is_long=is_long)
        
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=chart_img,
            caption=f"🎯 *LIVE POI MAP: {raw_symbol} ({tf.upper()})*",
            parse_mode="Markdown"
        )
        
        await status_msg.delete()
        await context.bot.send_message(chat_id=chat_id, text=analysis_text, parse_mode="Markdown")

    except Exception as e:
        await status_msg.edit_text(f"❌ Error generating analysis: {str(e)}")

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manual signal command generator: /signal XAUUSD BUY 4028"""
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("⚠️ Usage: `/signal [SYMBOL] [BUY/SELL] [ENTRY_PRICE]`", parse_mode="Markdown")
        return

    symbol = args[0].upper()
    direction = args[1].upper()
    try:
        entry = float(args[2])
    except ValueError:
        await update.message.reply_text("❌ Invalid price format.")
        return

    signal_msg = format_scalp_signal(symbol, direction, entry)
    await update.message.reply_text(signal_msg, parse_mode="Markdown")

# ==========================================
# 8. AUTOMATED MULTI-PAIR SCALPING SIGNAL SCANNER
# ==========================================
async def auto_market_scanner(context: ContextTypes.DEFAULT_TYPE):
    """Monitors all pairs on 5m timeframe and broadcasts live scalping signals when price taps POI."""
    if not MY_CHAT_ID:
        return

    watchlist = ["GOLD", "EURUSD", "GBPUSD", "BTCUSD", "USOIL", "GBPAUD"]
    
    for symbol in watchlist:
        try:
            _, df_entry, local_is_bearish, _, ticker_str = fetch_multi_timeframe_data(symbol, "5m")
            if df_entry.empty:
                continue

            last_close = df_entry['Close'].iloc[-1]
            last_timestamp = str(df_entry.index[-1])
            
            fvgs = detect_fvg(df_entry)
            obs = detect_order_blocks(df_entry)

            setup_found = None
            direction = None

            # 1. Check Bearish Realignment (Price BELOW 200 EMA & inside Bearish POI)
            if local_is_bearish:
                for fvg in fvgs[-2:]:
                    if fvg['type'] == 'BEARISH_FVG' and fvg['bottom'] <= last_close <= fvg['top']:
                        setup_found = "BEARISH_FVG"
                        direction = "SELL"
                        break
                if not setup_found:
                    for ob in obs[-2:]:
                        if ob['type'] in ['BEARISH_OB', 'BEARISH_BB'] and ob['bottom'] <= last_close <= ob['top']:
                            setup_found = ob['type']
                            direction = "SELL"
                            break

            # 2. Check Bullish Realignment (Price ABOVE 200 EMA & inside Bullish POI)
            else:
                for fvg in fvgs[-2:]:
                    if fvg['type'] == 'BULLISH_FVG' and fvg['bottom'] <= last_close <= fvg['top']:
                        setup_found = "BULLISH_FVG"
                        direction = "BUY"
                        break
                if not setup_found:
                    for ob in obs[-2:]:
                        if ob['type'] in ['BULLISH_OB', 'BULLISH_BB'] and ob['bottom'] <= last_close <= ob['top']:
                            setup_found = ob['type']
                            direction = "BUY"
                            break

            # If setup exists and hasn't been broadcasted recently
            if setup_found and direction:
                cache_key = f"{symbol}_{direction}_{last_timestamp}"
                if cache_key not in SENT_SIGNALS_CACHE:
                    SENT_SIGNALS_CACHE.add(cache_key)
                    
                    # Clean cache if too large
                    if len(SENT_SIGNALS_CACHE) > 200:
                        SENT_SIGNALS_CACHE.clear()

                    signal_msg = format_scalp_signal(symbol, direction, last_close)
                    await context.bot.send_message(chat_id=MY_CHAT_ID, text=signal_msg, parse_mode="Markdown")

        except Exception as e:
            print(f"[Scanner Error] {symbol}: {e}")

# ==========================================
# 9. APPLICATION INITIALIZATION & THREADING
# ==========================================
def run_telegram_bot():
    """Runs Telegram bot polling cleanly inside an independent AsyncIO loop."""
    if not TELEGRAM_BOT_TOKEN:
        print("[Error] TELEGRAM_BOT_TOKEN missing! Skipping bot startup.")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("signal", signal_command))

    # Automatic Scanner runs every 3 minutes (180s)
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(auto_market_scanner, interval=180, first=10)

    loop.run_until_complete(application.initialize())
    loop.run_until_complete(application.start())
    loop.run_until_complete(application.updater.start_polling(drop_pending_updates=True))
    
    print("[Telegram Bot] Live Scanner and Polling active...")
    loop.run_forever()

if __name__ == "__main__":
    telegram_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    telegram_thread.start()

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
