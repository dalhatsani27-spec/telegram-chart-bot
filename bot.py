import os
import io
import asyncio
import threading
from flask import Flask
import yfinance as yf
import mplfinance as mpf
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# --- 1. Flask Health Check Server for Render Free Tier ---
flask_app = Flask(__name__)

@flask_app.route('/')
def health_check():
    return "Bot is active!", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_flask, daemon=True).start()

# --- 2. Gemini Configuration & Rate-Limited Generator ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def resize_image_for_api(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Resizes high-res screenshots to reduce token footprint on Free Tier."""
    img = image.copy()
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return img

async def generate_content_async(prompt: str, chart_img: Image.Image, retries: int = 3, delay: int = 6):
    """Non-blocking generate content with backoff retry logic across multiple models."""
    optimized_img = resize_image_for_api(chart_img)
    models_to_try = ['gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-2.0-flash']
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            current_delay = delay
            for attempt in range(retries):
                try:
                    # Run generative call in executor to keep loop non-blocking
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None, lambda: model.generate_content([prompt, optimized_img])
                    )
                    return response
                except ResourceExhausted:
                    print(f"Rate limit on {model_name} (Attempt {attempt+1}/{retries}). Waiting {current_delay}s...")
                    await asyncio.sleep(current_delay)
                    current_delay *= 2
        except Exception as err:
            print(f"Model {model_name} unavailable: {err}")
            continue
            
    raise Exception("Free Tier rate limit reached across all models. Please wait 1 minute before trying another image.")

def format_ticker(symbol: str) -> str:
    """Formats user symbol inputs to standard Yahoo Finance tickers."""
    sym = symbol.upper().replace("/", "").strip()
    forex_pairs = [
        "GBPUSD", "EURUSD", "AUDUSD", "USDCAD", 
        "USDJPY", "USDCHF", "NZDUSD", "GBPAUD", 
        "EURGBP", "EURAUD", "GBPJPY", "EURJPY"
    ]
    if sym in forex_pairs:
        return f"{sym}=X"
    
    if sym in ["BTCUSD", "BTC", "BITCOIN"]:
        return "BTC-USD"
    elif sym in ["ETHUSD", "ETH", "ETHEREUM"]:
        return "ETH-USD"
    elif sym in ["SOLUSD", "SOL"]:
        return "SOL-USD"
    elif sym in ["XAUUSD", "GOLD"]:
        return "GC=F"
    elif sym in ["USOIL", "WTI", "CRUDE"]:
        return "CL=F"
    elif sym in ["US30", "DJI"]:
        return "^DJI"
    elif sym in ["NAS100", "NDX"]:
        return "^IXIC"
    elif sym in ["SPX500", "SP500"]:
        return "^GSPC"
        
    return sym

active_alerts = {}

# --- 3. Handlers & Commands ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Welcome to Market Vision Bot!*\n\n"
        "📌 *Analysis Commands:*\n"
        "• `/analyze GBPUSD` (Default 15m chart)\n"
        "• `/analyze BTCUSD 5m` (Timeframes: 1m, 5m, 15m, 1h, 4h, 1d)\n\n"
        "🔔 *Alert Commands:*\n"
        "• `/alert GBPUSD 1.3100` (Notifies you when price crosses target)\n"
        "• `/listalerts` (View active alerts)\n\n"
        "📸 Or simply upload a screenshot of your chart directly!",
        parse_mode="Markdown"
    )

async def analyze_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "⚠️ *Please specify a symbol.*\n\nExamples:\n• `/analyze GBPUSD`\n• `/analyze BTCUSD 5m`",
            parse_mode="Markdown"
        )
        return

    user_symbol = context.args[0]
    tf_input = context.args[1].lower() if len(context.args) > 1 else "15m"
    
    tf_map = {
        "1m": ("1m", "1d"), "5m": ("5m", "5d"), "15m": ("15m", "5d"),
        "30m": ("30m", "5d"), "1h": ("60m", "1mo"), "4h": ("60m", "1mo"), "1d": ("1d", "3mo")
    }
    
    interval, period = tf_map.get(tf_input, ("15m", "5d"))
    ticker = format_ticker(user_symbol)

    await update.message.reply_text(
        f"📊 *Fetching market data & generating {tf_input.upper()} chart for {user_symbol.upper()}...*", 
        parse_mode="Markdown"
    )

    try:
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            await update.message.reply_text(f"❌ Could not retrieve market data for `{user_symbol}`.", parse_mode="Markdown")
            return

        if hasattr(df.columns, "levels"):
            df.columns = df.columns.droplevel(1)

        img_buf = io.BytesIO()
        mc = mpf.make_marketcolors(up='#00b050', down='#ff0000', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)
        
        mpf.plot(
            df.tail(80), type='candle', style=s, volume=False, 
            title=f"{user_symbol.upper()} ({tf_input.upper()} Chart)", 
            savefig=dict(fname=img_buf, dpi=120)
        )
        
        img_buf.seek(0)
        chart_img = Image.open(img_buf)

        prompt = f"""
        Perform a strict price action and market structure analysis on this {user_symbol.upper()} ({tf_input.upper()}) chart:
        1. **Overall Bias**: Bullish / Bearish / Consolidation.
        2. **Liquidity & Key Levels**: Identify liquidity high/low hunts, key support/resistance, or supply/demand zones.
        3. **Price Action Interrogation**: Highlight active wicks, rejections, or displacement candles.
        4. **Trade Setup**: Provide a clear Trigger Zone, Stop Loss, and Take Profit targets.
        Keep the output crisp, clear, and structured for a real-time trading alert.
        """
        
        response = await generate_content_async(prompt, chart_img)

        img_buf.seek(0)
        await update.message.reply_photo(
            photo=img_buf,
            caption=f"🎯 *LIVE AI ANALYSIS: {user_symbol.upper()} ({tf_input.upper()})*\n\n{response.text}",
            parse_mode="Markdown"
        )

    except Exception as e:
        await update.message.reply_text(f"❌ Error generating analysis: {str(e)}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles direct uploaded screenshots with token optimization."""
    await update.message.reply_text("📸 *Chart screenshot received! Optimizing & analyzing structure...*", parse_mode="Markdown")
    
    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        chart_img = Image.open(io.BytesIO(image_bytes))
        
        prompt = """
        Perform a strict price action and market structure analysis on this uploaded trading chart:
        1. **Overall Bias**: Bullish / Bearish / Consolidation.
        2. **Liquidity & Key Levels**: Identify liquidity high/low hunts, wicks, and rejection zones.
        3. **Trade Setup**: Provide a clear Trigger Zone, Stop Loss, and Take Profit targets.
        Keep the output crisp, clear, and structured for a real-time trading alert.
        """
        
        response = await generate_content_async(prompt, chart_img)
        await update.message.reply_text(f"🎯 *AI CHART ANALYSIS*\n\n{response.text}", parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error analyzing image: {str(e)}")

async def set_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("⚠️ Usage: `/alert <SYMBOL> <TARGET_PRICE>`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    try:
        target_price = float(context.args[1])
    except ValueError:
        await update.message.reply_text("❌ Invalid price number.")
        return

    ticker = format_ticker(symbol)
    df = yf.download(ticker, period="1d", interval="1m")
    if df.empty:
        await update.message.reply_text(f"❌ Could not fetch live price for `{symbol}`.", parse_mode="Markdown")
        return

    if hasattr(df.columns, "levels"):
        df.columns = df.columns.droplevel(1)

    current_price = float(df['Close'].iloc[-1])
    direction = "above" if target_price > current_price else "below"

    chat_id = update.effective_chat.id
    if chat_id not in active_alerts:
        active_alerts[chat_id] = []

    active_alerts[chat_id].append({
        'symbol': symbol, 'ticker': ticker, 'target': target_price, 'direction': direction
    })

    await update.message.reply_text(
        f"🔔 *Alert set for {symbol}!*\n• Current: `{current_price:.5f}`\n• Target: `{target_price:.5f}`\n• Condition: Price `{direction}` target.",
        parse_mode="Markdown"
    )

async def list_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    alerts = active_alerts.get(chat_id, [])
    if not alerts:
        await update.message.reply_text("ℹ️ No active price alerts set.")
        return

    msg = "🔔 *Your Active Alerts:*\n"
    for i, a in enumerate(alerts, 1):
        msg += f"{i}. `{a['symbol']}` Target: `{a['target']}` ({a['direction']})\n"
    await update.message.reply_text(msg, parse_mode="Markdown")

async def check_alerts_job(context: ContextTypes.DEFAULT_TYPE):
    for chat_id, alerts in list(active_alerts.items()):
        for alert in list(alerts):
            try:
                df = yf.download(alert['ticker'], period="1d", interval="1m")
                if df.empty:
                    continue
                if hasattr(df.columns, "levels"):
                    df.columns = df.columns.droplevel(1)

                current_price = float(df['Close'].iloc[-1])
                triggered = False

                if alert['direction'] == "above" and current_price >= alert['target']:
                    triggered = True
                elif alert['direction'] == "below" and current_price <= alert['target']:
                    triggered = True

                if triggered:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"🚨 *PRICE ALERT TRIGGERED!*\n\nSymbol: `{alert['symbol']}`\nTarget Level: `{alert['target']}`\nCurrent Price: `{current_price:.5f}`",
                        parse_mode="Markdown"
                    )
                    alerts.remove(alert)
            except Exception as e:
                print(f"Alert check error: {e}")

# --- 4. Main App Execution ---
if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN environment variable missing!")
    else:
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("analyze", analyze_pair))
        app.add_handler(CommandHandler("alert", set_alert))
        app.add_handler(CommandHandler("listalerts", list_alerts))
        app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        
        if app.job_queue:
            app.job_queue.run_repeating(check_alerts_job, interval=60, first=10)
        
        print("Telegram Vision Bot active...")
        app.run_polling()
