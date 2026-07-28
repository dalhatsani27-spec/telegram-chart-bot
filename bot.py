import os
import io
import asyncio
import base64
import threading
from flask import Flask
import yfinance as yf
import mplfinance as mpf
from openai import OpenAI
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# --- 1. Flask Health Check Server for Render ---
flask_app = Flask(__name__)

@flask_app.route('/')
def health_check():
    return "Bot is active!", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_flask, daemon=True).start()

# --- 2. OpenRouter Setup ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY or "dummy_key"
)

def image_to_base64(image: Image.Image) -> str:
    """Converts PIL Image to Base64 format for OpenRouter API."""
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def analyze_chart_async(prompt: str, chart_img: Image.Image) -> str:
    """Non-blocking vision request via OpenRouter free router."""
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY environment variable is missing on Render!")

    base64_image = image_to_base64(chart_img)
    
    models_to_try = [
        "openrouter/free",
        "google/gemini-2.5-flash:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free"
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            def _call():
                return client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=700
                )

            response = await asyncio.to_thread(_call)
            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            last_error = e
            continue

    raise Exception(f"OpenRouter API Error: {str(last_error)}")

def format_ticker(symbol: str) -> str:
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
    elif sym in ["XAUUSD", "GOLD"]:
        return "GC=F"
    return sym

def fetch_market_data(ticker_symbol: str, period: str, interval: str):
    ticker = yf.Ticker(ticker_symbol)
    return ticker.history(period=period, interval=interval)

def generate_chart_image(df, title_str):
    img_buf = io.BytesIO()
    mc = mpf.make_marketcolors(up='#00b050', down='#ff0000', edge='inherit', wick='inherit', volume='in')
    s  = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)
    
    mpf.plot(
        df.tail(80), type='candle', style=s, volume=False, 
        title=title_str, savefig=dict(fname=img_buf, dpi=120)
    )
    img_buf.seek(0)
    return img_buf

# --- 3. Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Welcome to Market Vision Bot!*\n\n"
        "📌 *Analysis Commands:*\n"
        "• `/analyze GBPUSD` (Default 15m chart)\n"
        "• `/analyze BTCUSD 5m` (Timeframes: 1m, 5m, 15m, 1h, 4h, 1d)\n\n"
        "📸 Or send a chart screenshot directly to analyze!",
        parse_mode="Markdown"
    )

async def analyze_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Usage: `/analyze GBPUSD`", parse_mode="Markdown")
        return

    user_symbol = context.args[0]
    tf_input = context.args[1].lower() if len(context.args) > 1 else "15m"
    
    tf_map = {
        "1m": ("1m", "1d"), "5m": ("5m", "5d"), "15m": ("15m", "5d"),
        "30m": ("30m", "5d"), "1h": ("60m", "1mo"), "4h": ("60m", "1mo"), "1d": ("1d", "3mo")
    }
    
    interval, period = tf_map.get(tf_input, ("15m", "5d"))
    ticker_sym = format_ticker(user_symbol)

    await update.message.reply_text(
        f"📊 *Fetching market data & analyzing {tf_input.upper()} chart for {user_symbol.upper()}...*", 
        parse_mode="Markdown"
    )

    try:
        df = await asyncio.to_thread(fetch_market_data, ticker_sym, period, interval)
        if df is None or df.empty:
            await update.message.reply_text(f"❌ Could not retrieve market data for `{user_symbol}`.", parse_mode="Markdown")
            return

        if hasattr(df.columns, "levels"):
            df.columns = df.columns.droplevel(1)

        img_buf = await asyncio.to_thread(generate_chart_image, df, f"{user_symbol.upper()} ({tf_input.upper()})")
        chart_img = Image.open(img_buf)

        prompt = f"""
        Perform a strict price action and market structure analysis on this {user_symbol.upper()} ({tf_input.upper()}) chart:
        1. **Overall Bias**: Bullish / Bearish / Consolidation.
        2. **Liquidity & Key Levels**: Identify liquidity high/low hunts, key support/resistance, or supply/demand zones.
        3. **Price Action**: Highlight active wicks, rejections, or displacement candles.
        4. **Trade Setup**: Provide a clear Trigger Zone, Stop Loss, and Take Profit targets.
        Keep the output crisp and structured for a real-time trading alert.
        """
        
        analysis_text = await analyze_chart_async(prompt, chart_img)

        # Send photo first, then full analysis as text message
        img_buf.seek(0)
        await update.message.reply_photo(photo=img_buf, caption=f"🎯 *LIVE CHART: {user_symbol.upper()} ({tf_input.upper()})*")
        await update.message.reply_text(f"🎯 *AI ANALYSIS*\n\n{analysis_text}", parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ Error generating analysis: {str(e)}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📸 *Chart screenshot received! Analyzing structure...*", parse_mode="Markdown")
    
    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        chart_img = Image.open(io.BytesIO(image_bytes))
        
        prompt = """
        Perform a strict price action and market structure analysis on this chart screenshot:
        1. **Overall Bias**: Bullish / Bearish / Consolidation.
        2. **Liquidity & Key Levels**: Identify liquidity high/low hunts, wicks, and rejection zones.
        3. **Trade Setup**: Provide a clear Trigger Zone, Stop Loss, and Take Profit targets.
        """
        
        analysis_text = await analyze_chart_async(prompt, chart_img)
        await update.message.reply_text(f"🎯 *AI CHART ANALYSIS*\n\n{analysis_text}", parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error analyzing image: {str(e)}")

# --- 4. Main Execution ---
if __name__ == '__main__':
    if TELEGRAM_BOT_TOKEN:
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("analyze", analyze_pair))
        app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        
        print("Telegram Vision Bot active...")
        app.run_polling()
