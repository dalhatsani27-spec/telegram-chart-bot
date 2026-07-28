import os
import io
import time
import threading
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
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
MY_CHAT_ID = os.environ.get("MY_CHAT_ID")  # Set your Telegram Chat ID for alerts

def keep_alive_ping():
    """Pings Flask endpoint every 10 minutes to prevent Render from sleeping."""
    while True:
        time.sleep(600)  # 10 minutes interval
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
        alines_list.append([p_highs[-2], p_highs[-1]])  # Dynamic Resistance Trendline
    if len(p_lows) >= 2:
        alines_list.append([p_lows[-2], p_lows[-1]])    # Dynamic Support Trendline
        
    alines_dict = dict(alines=alines_list, colors=['#f23645', '#089981'], linewidths=1.5) if alines_list else None

    # Fallback POI logic if none parsed
    last_close = chart_df['Close'].iloc[-1]
    if entry is None:
        entry = last_close
        if is_long:
            sl = chart_df['Low'].min()
            tp = last_close + (last_close - sl) * 1.5
        else:
            sl = chart_df['High'].max()
            tp = last_close - (sl - last_close) * 1.5

    # Horizontal Level Lines
    hlines_dict = dict(
        hlines=[entry, sl, tp],
        colors=['#2962ff', '#f23645', '#089981'],  # Blue Entry, Red SL, Green TP
        linestyle=['-.', '-', '-'],
        linewidths=[1.5, 1.2, 1.2]
    )

    # Shaded Position Template Boxes
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
# 4. TELEGRAM BOT HANDLERS & ANALYSIS LOGIC
# ==========================================
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

        # Generate Chart Image
        chart_img = generate_chart_image(df, f"{symbol} ({tf.upper()}) - LIVE POI & STRUCTURE MAPPING")
        
        # Send Chart Image First
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=chart_img,
            caption=f"🎯 *LIVE CHART: {symbol} ({tf.upper()})*",
            parse_mode="Markdown"
        )
        
        # Structure OpenRouter Prompt
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

        response = ai_client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        
        analysis_text = response.choices[0].message.content
        
        # Delete loading status and send full breakdown
        await status_msg.delete()
        await context.bot.send_message(chat_id=chat_id, text=analysis_text)

    except Exception as e:
        await status_msg.edit_text(f"❌ Error generating analysis: {str(e)}")

# ==========================================
# 5. REAL-TIME SIGNAL & ALERT SYSTEM
# ==========================================
async def send_signal_alert(context: ContextTypes.DEFAULT_TYPE, symbol: str, signal_type: str, details: str):
    """Pushes instant signal pop-ups to your Telegram when offline or active."""
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

# ==========================================
# 6. APPLICATION INITIALIZATION
# ==========================================
def main():
    # Start Flask in separate thread
    port = int(os.environ.get("PORT", 5000))
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port), daemon=True).start()

    # Build Telegram Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Register Commands
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Bot active! Use /analyze GBPUSD 15m")))

    # Run Telegram Bot long-polling loop
    application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
