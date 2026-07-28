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
    return "Bot Service is Alive & Running 24/7!", 200

# Fetch environment variables
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MY_CHAT_ID = os.environ.get("MY_CHAT_ID")  # Your personal Telegram Chat ID for alerts

def keep_alive_ping():
    """Pings Flask endpoint every 10 minutes to prevent Render from sleeping."""
    while True:
        time.sleep(600)  # 10 minute interval
        if RENDER_EXTERNAL_URL:
            try:
                res = requests.get(RENDER_EXTERNAL_URL, timeout=10)
                print(f"[Keep-Alive] Ping sent to {RENDER_EXTERNAL_URL} | Status: {res.status_code}")
            except Exception as e:
                print(f"[Keep-Alive] Ping failed: {e}")

# Start background keep-alive ping thread
threading.Thread(target=keep_alive_ping, daemon=True).start()

# ==========================================
# 2. OPENROUTER / AI VISION SETUP
# ==========================================
ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def fetch_ai_analysis(prompt):
    """Fallback chain ensuring reliable text analysis without guardrail outputs."""
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
            # Guard against guardrail responses (e.g. 'User Safety: safe')
            if content and "User Safety:" not in content and len(content.strip()) > 30:
                return content
        except Exception:
            continue
            
    return (
        "🎯 *AI ANALYSIS & MARKET STORY*\n\n"
        "1. **Overall Bias:** Dynamic Market Structure\n"
        "2. **Liquidity & Key POI:** Check chart overlay for dynamic Pivot Support/Resistance lines.\n"
        "3. **Trade Setup:** Refer to blue trigger line and green/red shaded position boxes on the generated chart."
    )

# ==========================================
# 3. CHART MAPPING: PIVOTS & TRADINGVIEW POIs
# ==========================================
def find_pivots(df, window=3):
    """Calculates local fractal Pivot Highs and Pivot Lows."""
    pivots_high = []
    pivots_low = []
    for i in range(window, len(df) - window):
        high_window = df['High'].iloc[i-window:i+window+1]
        low_window = df['Low'].iloc[i-window:i+window+1]
        
        if df['High'].iloc[i] == high_window.max():
            pivots_high.append((df.index[i], df['High'].iloc[i]))
            
        if df['Low'].iloc[i] == low_window.min():
            pivots_low.append((df.index[i], df['Low'].iloc[i]))
            
    return pivots_high, pivots_low

def generate_chart_image(df, title_str, entry=None, sl=None, tp=None, is_long=True):
    """Renders TradingView-style dark chart with dynamic POI boxes and Pivot trendlines."""
    img_buf = io.BytesIO()
    chart_df = df.tail(80)
    
    # TradingView Dark Color Scheme
    mc = mpf.make_marketcolors(
        up='#089981', down='#f23645',
        edge='inherit', wick='inherit', volume='in'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', 
        y_on_right=True, facecolor='#131722', figcolor='#131722'
    )
    
    # Dynamic Pivot Trendlines
    p_highs, p_lows = find_pivots(chart_df, window=3)
    alines_list = []
    
    if len(p_highs) >= 2:
        alines_list.append([p_highs[-2], p_highs[-1]])  # Dynamic Resistance
    if len(p_lows) >= 2:
        alines_list.append([p_lows[-2], p_lows[-1]])    # Dynamic Support
        
    alines_dict = dict(alines=alines_list, colors=['#f23645', '#089981'], linewidths=1.2) if alines_list else None

    # Proportional POI Box Logic
    last_close = chart_df['Close'].iloc[-1]
    range_offset = (chart_df['High'].max() - chart_df['Low'].min()) * 0.15  # 15% chart height
    
    if entry is None:
        entry = last_close
        if is_long:
            sl = entry - range_offset
            tp = entry + (range_offset * 1.5)
        else:
            sl = entry + range_offset
            tp = entry - (range_offset * 1.5)

    # Horizontal Level Lines
    hlines_dict = dict(
        hlines=[entry, sl, tp],
        colors=['#2962ff', '#f23645', '#089981'],  # Blue Entry, Red SL, Green TP
        linestyle=['-.', '--', '--'],
        linewidths=[1.2, 1.0, 1.0]
    )

    # Direction-Aware Shaded Position Boxes
    fill_tp = dict(y1=entry, y2=tp, color='#089981', alpha=0.18)
    fill_sl = dict(y1=entry, y2=sl, color='#f23645', alpha=0.18)

    # Plot Chart
    fig, axlist = mpf.plot(
        chart_df,
        type='candle',
        style=style,
        volume=False,
        hlines=hlines_dict,
        alines=alines_dict,
        fill_between=[fill_tp, fill_sl],
        savefig=dict(fname=img_buf, dpi=130),
        returnfig=True
    )
    
    axlist[0].set_title(title_str, color='#d1d4dc', fontsize=11, fontweight='bold')
    img_buf.seek(0)
    return img_buf

# ==========================================
# 4. TELEGRAM BOT HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a formatted welcome message with full instructions."""
    welcome_text = (
        "🤖 *WELCOME TO YOUR CHART & SIGNAL ANALYZER*\n\n"
        "I am configured to map institutional price action, liquidity sweeps, and POI zones directly onto chart setups.\n\n"
        "📌 *HOW TO USE ME:*\n"
        "• `/analyze [SYMBOL] [TIMEFRAME]`\n"
        "  _Examples:_\n"
        "  - `/analyze GBPUSD 15m`\n"
        "  - `/analyze EURUSD 1h`\n"
        "  - `/analyze BTCUSD 5m`\n\n"
        "📊 *WHAT YOU GET:*\n"
        "1. **TradingView Dark Chart:** Clean high-resolution chart mapping.\n"
        "2. **Dynamic Trendlines:** Connected fractal Pivot Highs & Lows.\n"
        "3. **Position Templates:** Direction-aligned Risk-Reward boxes (Green TP / Red SL).\n"
        "4. **AI Market Story:** Liquidity sweep & POI breakdown.\n"
        "5. **Automated Pop-Up Alerts:** Instant signals delivered straight to your phone.\n\n"
        "💡 *Get Started:* Try running `/analyze GBPUSD 15m` now!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    
    symbol = args[0].upper() if len(args) > 0 else "GBPUSD"
    tf = args[1] if len(args) > 1 else "15m"
    
    # Map Symbol for Yahoo Finance
    ticker_str = f"{symbol}=X" if len(symbol) == 6 else symbol
    if symbol == "BTCUSD": ticker_str = "BTC-USD"
    
    status_msg = await update.message.reply_text(f"📊 Fetching market data & analyzing {tf} chart for {symbol}...")
    
    try:
        df = yf.download(ticker_str, period="5d", interval=tf)
        if df.empty:
            await status_msg.edit_text(f"❌ Error: Unable to fetch market data for {symbol}.")
            return
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Structure Prompt
        last_price = df['Close'].iloc[-1]
        recent_high = df['High'].tail(30).max()
        recent_low = df['Low'].tail(30).min()
        
        prompt = f"""
        You are an elite quantitative price-action trader.
        Analyze this chart data for {symbol}:
        Current Price: {last_price}
        Recent High: {recent_high}
        Recent Low: {recent_low}
        
        Provide analysis formatted exactly as:
        🎯 AI ANALYSIS & MARKET STORY
        1. Overall Bias: (Bullish/Bearish)
        2. Liquidity & Key POI: Highlight liquidity sweeps or high/low hunts.
        3. Price Action Story: Rejection wicks, displacement, and volume context.
        4. Trade Setup:
           - Trigger Zone:
           - Stop Loss:
           - Take Profit:
        """

        # Fetch robust analysis text
        analysis_text = fetch_ai_analysis(prompt)

        # Determine directional bias to align chart position template correctly
        is_long_bias = "bearish" not in analysis_text.lower()

        # Generate Chart Image matching bias direction
        chart_img = generate_chart_image(
            df, 
            f"{symbol} ({tf.upper()}) - LIVE POI & STRUCTURE MAPPING",
            is_long=is_long_bias
        )
        
        # Send Chart Image
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=chart_img,
            caption=f"🎯 *LIVE CHART: {symbol} ({tf.upper()})*",
            parse_mode="Markdown"
        )
        
        # Delete status message and send text breakdown
        await status_msg.delete()
        await context.bot.send_message(chat_id=chat_id, text=analysis_text, parse_mode="Markdown")

    except Exception as e:
        await status_msg.edit_text(f"❌ Error generating analysis: {str(e)}")

# ==========================================
# 5. REAL-TIME SIGNAL & AUTOMATED SCANNER
# ==========================================
async def send_signal_alert(context: ContextTypes.DEFAULT_TYPE, symbol: str, signal_type: str, details: str):
    """Pushes instant signal pop-ups to your Telegram user ID."""
    if not MY_CHAT_ID:
        return
        
    alert_text = (
        f"🚨 *AUTOMATED SIGNAL ALERT: {symbol}*\n\n"
        f"📌 *Signal Type:* {signal_type}\n"
        f"📝 *Details:* {details}\n\n"
        f"⚡ *Action:* Open MT5 and confirm entry setup."
    )
    
    await context.bot.send_message(
        chat_id=MY_CHAT_ID,
        text=alert_text,
        parse_mode="Markdown"
    )

async def auto_market_scanner(context: ContextTypes.DEFAULT_TYPE):
    """Periodically scans liquidity levels and triggers signal pop-ups."""
    if not MY_CHAT_ID:
        return

    symbols_to_scan = ["GBPUSD", "EURUSD", "BTCUSD"]
    
    for symbol in symbols_to_scan:
        try:
            ticker_str = "BTC-USD" if symbol == "BTCUSD" else f"{symbol}=X"
            df = yf.download(ticker_str, period="1d", interval="5m", progress=False)
            
            if df.empty:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            last_close = df['Close'].iloc[-1]
            recent_high = df['High'].tail(20).max()
            recent_low = df['Low'].tail(20).min()

            # Trigger condition: price sweeps within 2 pips/points of recent liquidity high/low
            if abs(last_close - recent_high) < 0.0002:
                await send_signal_alert(
                    context, 
                    symbol, 
                    "BUY-SIDE LIQUIDITY HUNT", 
                    f"Price ({last_close:.5f}) is sweeping recent high at {recent_high:.5f}. Look for potential short realignment entry."
                )
            elif abs(last_close - recent_low) < 0.0002:
                await send_signal_alert(
                    context, 
                    symbol, 
                    "SELL-SIDE LIQUIDITY HUNT", 
                    f"Price ({last_close:.5f}) is sweeping recent low at {recent_low:.5f}. Look for potential long realignment entry."
                )
        except Exception as e:
            print(f"Scanner error for {symbol}: {e}")

# ==========================================
# 6. APPLICATION INITIALIZATION & THREADING
# ==========================================
def run_telegram_bot():
    """Runs the Telegram bot polling loop cleanly inside its own AsyncIO loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Register Command Handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("analyze", analyze_command))

    # Schedule Market Scanner Job Queue (Every 5 minutes)
    job_queue = application.job_queue
    job_queue.run_repeating(auto_market_scanner, interval=300, first=10)

    # Initialize and start polling
    loop.run_until_complete(application.initialize())
    loop.run_until_complete(application.start())
    loop.run_until_complete(application.updater.start_polling(drop_pending_updates=True))
    
    print("[Telegram Bot] Polling started successfully!")
    loop.run_forever()

if __name__ == "__main__":
    # 1. Start Telegram Bot Polling in Background Thread
    telegram_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    telegram_thread.start()

    # 2. Run Flask Web App on Main Thread (Render binds to PORT 10000 immediately)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
