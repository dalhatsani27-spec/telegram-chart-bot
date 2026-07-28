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
    return "Multi-Timeframe Chart Analyzer Bot is Alive & Running 24/7!", 200

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
                print(f"[Keep-Alive] Ping sent | Status: {res.status_code}")
            except Exception as e:
                print(f"[Keep-Alive] Ping failed: {e}")

threading.Thread(target=keep_alive_ping, daemon=True).start()

# ==========================================
# 2. OPENROUTER / AI VISION SETUP
# ==========================================
ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def fetch_ai_analysis(prompt):
    """Fallback chain ensuring reliable text analysis."""
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
        "1. **Overall Bias:** Multi-Timeframe Alignment Active.\n"
        "2. **Macro Context:** Check 200 EMA (Yellow Line) for master trend direction.\n"
        "3. **Liquidity Footprints:** Reference BSL (▼) and SSL (▲) zones before entry."
    )

# ==========================================
# 3. MULTI-TIMEFRAME & INDICATOR LOGIC
# ==========================================
def fetch_multi_timeframe_data(symbol, entry_tf="15m"):
    """Fetches Daily macro context alongside execution timeframe data."""
    ticker_str = f"{symbol}=X" if len(symbol) == 6 else symbol
    if symbol == "BTCUSD": 
        ticker_str = "BTC-USD"

    # 1. Macro Trend Data (Daily)
    df_daily = yf.download(ticker_str, period="1y", interval="1d", progress=False)
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)
        
    df_daily['EMA200'] = df_daily['Close'].ewm(span=200, adjust=False).mean()
    daily_close = df_daily['Close'].iloc[-1]
    daily_ema = df_daily['EMA200'].iloc[-1]
    macro_is_bearish = daily_close < daily_ema

    # 2. Local Execution Data (e.g. 15m)
    df_entry = yf.download(ticker_str, period="10d", interval=entry_tf, progress=False)
    if isinstance(df_entry.columns, pd.MultiIndex):
        df_entry.columns = df_entry.columns.get_level_values(0)
        
    df_entry['EMA200'] = df_entry['Close'].ewm(span=200, adjust=False).mean()
    
    return df_daily, df_entry, macro_is_bearish, daily_ema

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

def generate_chart_image(df, title_str, is_long=True):
    """Renders TradingView dark chart with 200 EMA, POI marks, and localized position box."""
    img_buf = io.BytesIO()
    chart_df = df.tail(80).copy()
    
    # TradingView Dark Theme
    mc = mpf.make_marketcolors(
        up='#089981', down='#f23645',
        edge='inherit', wick='inherit', volume='in'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', 
        y_on_right=True, facecolor='#131722', figcolor='#131722'
    )
    
    # Pivot Marks
    p_highs, p_lows = find_pivots(chart_df, window=3)
    bsl_points = [np.nan] * len(chart_df)
    ssl_points = [np.nan] * len(chart_df)
    price_range = chart_df['High'].max() - chart_df['Low'].min()
    
    for idx_time, val in p_highs:
        if idx_time in chart_df.index:
            loc = chart_df.index.get_loc(idx_time)
            bsl_points[loc] = val + (price_range * 0.02)
            
    for idx_time, val in p_lows:
        if idx_time in chart_df.index:
            loc = chart_df.index.get_loc(idx_time)
            ssl_points[loc] = val - (price_range * 0.02)

    # Overlays: Yellow 200 EMA + BSL (▼ Red) + SSL (▲ Green)
    addplots = [
        mpf.make_addplot(chart_df['EMA200'], color='#ffd700', width=1.5)
    ]
    if any(~np.isnan(bsl_points)):
        addplots.append(mpf.make_addplot(bsl_points, type='scatter', marker='v', markersize=40, color='#f23645'))
    if any(~np.isnan(ssl_points)):
        addplots.append(mpf.make_addplot(ssl_points, type='scatter', marker='^', markersize=40, color='#089981'))

    recent_bsl = p_highs[-1][1] if len(p_highs) > 0 else chart_df['High'].tail(30).max()
    recent_ssl = p_lows[-1][1] if len(p_lows) > 0 else chart_df['Low'].tail(30).min()
    
    # Align position direction strictly with the 200 EMA filter
    if is_long:
        entry = recent_ssl
        risk_dist = max((recent_bsl - recent_ssl) * 0.12, price_range * 0.04)
        sl = entry - risk_dist
        tp = recent_bsl
    else:
        entry = recent_bsl
        risk_dist = max((recent_bsl - recent_ssl) * 0.12, price_range * 0.04)
        sl = entry + risk_dist
        tp = recent_ssl

    # Structural Lines
    hlines_dict = dict(
        hlines=[recent_bsl, recent_ssl, entry, sl, tp],
        colors=['#f23645', '#089981', '#2962ff', '#e53935', '#43a047'],
        linestyle=[':', ':', '-.', '--', '--'],
        linewidths=[1.0, 1.0, 1.2, 1.0, 1.0]
    )

    # Restrict shading strictly to rightmost 15 candles
    fill_tp_data = [np.nan] * len(chart_df)
    fill_sl_data = [np.nan] * len(chart_df)
    box_width = min(15, len(chart_df))
    
    for i in range(len(chart_df) - box_width, len(chart_df)):
        fill_tp_data[i] = tp
        fill_sl_data[i] = sl

    fig, axlist = mpf.plot(
        chart_df,
        type='candle',
        style=style,
        volume=False,
        hlines=hlines_dict,
        addplot=addplots,
        fill_between=dict(y1=fill_tp_data, y2=entry, color='#089981', alpha=0.18),
        savefig=dict(fname=img_buf, dpi=130),
        returnfig=True
    )
    
    axlist[0].fill_between(
        range(len(chart_df)), fill_sl_data, entry,
        where=~np.isnan(fill_sl_data), color='#f23645', alpha=0.18
    )

    axlist[0].set_title(title_str, color='#d1d4dc', fontsize=9, fontweight='bold')
    img_buf.seek(0)
    return img_buf

# ==========================================
# 4. TELEGRAM BOT HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🤖 *MULTI-TIMEFRAME CHART ANALYZER*\n\n"
        "I am hard-locked to enforce 200 EMA macro context across Daily to lower timeframes to prevent counter-trend traps.\n\n"
        "📌 *COMMANDS:*\n"
        "• `/analyze [SYMBOL] [TIMEFRAME]`\n"
        "  _Examples:_ `/analyze GBPUSD 15m`, `/analyze CHFJPY 15m`\n\n"
        "📊 *KEY FEATURES:*\n"
        "1. **Gold Line:** 200 EMA Macro Trend Filter.\n"
        "2. **POI Footprints:** ▼ BSL (Buy-Side Liquidity) & ▲ SSL (Sell-Side Liquidity).\n"
        "3. **Localized Position Box:** Clean setup shading restricted to recent candles."
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    
    symbol = args[0].upper() if len(args) > 0 else "GBPUSD"
    tf = args[1] if len(args) > 1 else "15m"
    
    status_msg = await update.message.reply_text(f"📊 Running Top-Down Multi-Timeframe Analysis for {symbol}...")
    
    try:
        df_daily, df_entry, macro_is_bearish, daily_ema = fetch_multi_timeframe_data(symbol, tf)
        
        if df_entry.empty:
            await status_msg.edit_text(f"❌ Error: Unable to fetch market data for {symbol}.")
            return

        last_price = df_entry['Close'].iloc[-1]
        local_ema = df_entry['EMA200'].iloc[-1]
        recent_high = df_entry['High'].tail(30).max()
        recent_low = df_entry['Low'].tail(30).min()
        
        local_is_bearish = last_price < local_ema
        
        # Determine strict bias: Must respect 200 EMA
        macro_bias = "BEARISH" if macro_is_bearish else "BULLISH"
        local_bias = "BEARISH" if local_is_bearish else "BULLISH"
        
        prompt = f"""
        You are a quantitative price-action trader.
        CRITICAL RULE:
        - Daily 200 EMA Macro Bias: {macro_bias} (Daily Close: {df_daily['Close'].iloc[-1]:.5f}, Daily EMA: {daily_ema:.5f})
        - {tf.upper()} 200 EMA Local Bias: {local_bias} (Current Price: {last_price:.5f}, Local EMA: {local_ema:.5f})
        
        If price is BELOW the 200 EMA, YOU ARE STRICTLY FORBIDDEN from recommending Buy setups. Any upward push is a RETRACEMENT / BULL TRAP into Buy-Side Liquidity (BSL).
        
        Analyze {symbol} ({tf}):
        - Recent BSL Pool (High): {recent_high:.5f}
        - Recent SSL Pool (Low): {recent_low:.5f}
        
        Format output:
        🎯 AI ANALYSIS & MARKET STORY
        1. Macro Bias: {macro_bias} (Daily 200 EMA Alignment)
        2. Execution Bias ({tf}): {local_bias}
        3. Structural Story: Explain BSL/SSL sweeps relative to the 200 EMA.
        4. Realignment Setup: Detail expected entry alignment.
        """

        analysis_text = fetch_ai_analysis(prompt)

        # Generate chart based on local 200 EMA trend
        is_long = not local_is_bearish
        title_str = f"{symbol} ({tf.upper()}) | Trend: {local_bias} (Gold Line = 200 EMA)"
        chart_img = generate_chart_image(df_entry, title_str, is_long=is_long)
        
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=chart_img,
            caption=f"🎯 *LIVE CHART: {symbol} ({tf.upper()})*",
            parse_mode="Markdown"
        )
        
        await status_msg.delete()
        await context.bot.send_message(chat_id=chat_id, text=analysis_text, parse_mode="Markdown")

    except Exception as e:
        await status_msg.edit_text(f"❌ Error generating analysis: {str(e)}")

# ==========================================
# 5. REAL-TIME SIGNAL SCANNER
# ==========================================
async def auto_market_scanner(context: ContextTypes.DEFAULT_TYPE):
    """Periodically scans liquidity sweeps aligned with the 200 EMA."""
    if not MY_CHAT_ID:
        return

    symbols = ["GBPUSD", "EURUSD", "BTCUSD"]
    for symbol in symbols:
        try:
            _, df_entry, local_is_bearish, _ = fetch_multi_timeframe_data(symbol, "5m")
            if df_entry.empty:
                continue

            last_close = df_entry['Close'].iloc[-1]
            recent_high = df_entry['High'].tail(20).max()
            recent_low = df_entry['Low'].tail(20).min()

            # Trigger only if sweep aligns with 200 EMA direction
            if local_is_bearish and abs(last_close - recent_high) < 0.0002:
                alert_text = (
                    f"🚨 *AUTOMATED BEARISH REALIGNMENT ALERT: {symbol}*\n\n"
                    f"Price ({last_close:.5f}) swept BSL High ({recent_high:.5f}) below the 200 EMA.\n"
                    f"⚡ *Action:* Check for potential short entry."
                )
                await context.bot.send_message(chat_id=MY_CHAT_ID, text=alert_text, parse_mode="Markdown")
                
            elif (not local_is_bearish) and abs(last_close - recent_low) < 0.0002:
                alert_text = (
                    f"🚨 *AUTOMATED BULLISH REALIGNMENT ALERT: {symbol}*\n\n"
                    f"Price ({last_close:.5f}) swept SSL Low ({recent_low:.5f}) above the 200 EMA.\n"
                    f"⚡ *Action:* Check for potential long entry."
                )
                await context.bot.send_message(chat_id=MY_CHAT_ID, text=alert_text, parse_mode="Markdown")

        except Exception as e:
            print(f"Scanner error for {symbol}: {e}")

# ==========================================
# 6. APPLICATION INITIALIZATION & THREADING
# ==========================================
def run_telegram_bot():
    """Runs Telegram bot polling cleanly inside an independent AsyncIO loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("analyze", analyze_command))

    # Run scanner every 5 minutes
    job_queue = application.job_queue
    job_queue.run_repeating(auto_market_scanner, interval=300, first=10)

    loop.run_until_complete(application.initialize())
    loop.run_until_complete(application.start())
    loop.run_until_complete(application.updater.start_polling(drop_pending_updates=True))
    
    print("[Telegram Bot] Polling running...")
    loop.run_forever()

if __name__ == "__main__":
    telegram_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    telegram_thread.start()

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
