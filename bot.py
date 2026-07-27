import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is active!")

def start_health_check():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    server.serve_forever()

threading.Thread(target=start_health_check, daemon=True).start()

import os
import io
import yfinance as yf
import mplfinance as mpf
import google.generativeai as genai
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Retrieve environment variables configured in Render
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

genai.configure(api_key=GEMINI_API_KEY)

def format_ticker(symbol: str) -> str:
    """Formats user symbol inputs to standard Yahoo Finance tickers."""
    sym = symbol.upper().replace("/", "").strip()
    
    # Forex Pairs
    forex_pairs = [
        "GBPUSD", "EURUSD", "AUDUSD", "USDCAD", 
        "USDJPY", "USDCHF", "NZDUSD", "GBPAUD", 
        "EURGBP", "EURAUD", "GBPJPY", "EURJPY"
    ]
    if sym in forex_pairs:
        return f"{sym}=X"
    
    # Cryptocurrencies
    if sym in ["BTCUSD", "BTC", "BITCOIN"]:
        return "BTC-USD"
    elif sym in ["ETHUSD", "ETH", "ETHEREUM"]:
        return "ETH-USD"
    elif sym in ["SOLUSD", "SOL"]:
        return "SOL-USD"
        
    # Commodities
    elif sym in ["XAUUSD", "GOLD"]:
        return "GC=F"
    elif sym in ["USOIL", "WTI", "CRUDE"]:
        return "CL=F"
        
    # Major Indices
    elif sym in ["US30", "DJI"]:
        return "^DJI"
    elif sym in ["NAS100", "NDX"]:
        return "^IXIC"
    elif sym in ["SPX500", "SP500"]:
        return "^GSPC"
        
    return sym

async def analyze_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generates candle charts automatically and feeds them to Gemini AI."""
    if not context.args:
        await update.message.reply_text(
            "⚠️ *Please specify a symbol.*\n\n"
            "Examples:\n"
            "• `/analyze GBPUSD`\n"
            "• `/analyze BTCUSD`\n"
            "• `/analyze XAUUSD`\n\n"
            "_Note: For Deriv Synthetics (V75, Boom/Crash), send a screenshot directly!_",
            parse_mode="Markdown"
        )
        return

    user_symbol = context.args[0]
    ticker = format_ticker(user_symbol)
    await update.message.reply_text(f"📊 *Fetching market data & generating chart for {user_symbol.upper()}...*", parse_mode="Markdown")

    try:
        # Download 15-minute timeframe data over the last 5 days
        df = yf.download(ticker, period="5d", interval="15m")
        if df.empty:
            await update.message.reply_text(
                f"❌ Could not retrieve market data for `{user_symbol}`. "
                "Ensure symbol is typed correctly or send a manual screenshot.",
                parse_mode="Markdown"
            )
            return

        # Flatten multi-level dataframe columns if returned by yfinance
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.droplevel(1)

        # Plot clean Candlestick chart in memory
        img_buf = io.BytesIO()
        mc = mpf.make_marketcolors(up='#00b050', down='#ff0000', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)
        
        mpf.plot(
            df.tail(80), 
            type='candle', 
            style=s, 
            volume=False, 
            title=f"{user_symbol.upper()} (15M Chart)", 
            savefig=dict(fname=img_buf, dpi=150)
        )
        
        img_buf.seek(0)
        chart_img = Image.open(img_buf)

        prompt = f"""
        Perform a strict price action and market structure analysis on this {user_symbol.upper()} chart:
        1. **Overall Bias**: Bullish / Bearish / Consolidation.
        2. **Liquidity & Key Levels**: Identify liquidity high/low hunts, key support/resistance, or supply/demand zones.
        3. **Price Action Interrogation**: Highlight active wicks, rejections, or displacement candles.
        4. **Trade Setup**: Provide a clear Trigger Zone, Stop Loss, and Take Profit targets.
        Keep the output crisp, clear, and structured for a real-time trading alert.
        """
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content([prompt, chart_img])

        img_buf.seek(0)
        await update.message.reply_photo(
            photo=img_buf,
            caption=f"🎯 *LIVE AI ANALYSIS: {user_symbol.upper()}*\n\n{response.text}",
            parse_mode="Markdown"
        )

    except Exception as e:
        await update.message.reply_text(f"❌ Error generating chart: {str(e)}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles direct uploaded screenshots (MT5 Synthetics, custom indicators, etc.)."""
    await update.message.reply_text("📸 *Chart screenshot received! Analyzing structure...*", parse_mode="Markdown")
    
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
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content([prompt, chart_img])
        
        await update.message.reply_text(f"🎯 *AI CHART ANALYSIS*\n\n{response.text}", parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error analyzing image: {str(e)}")

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Handlers
    app.add_handler(CommandHandler("analyze", analyze_pair))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("Telegram Vision Bot active...")
    app.run_polling()
