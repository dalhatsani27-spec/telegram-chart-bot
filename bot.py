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
    return "Multi-Timeframe POI Chart Analyzer (FVG/OB/BB) is Active 24/7!", 200

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
        raise ValueError(f"Unable to retrieve market data for '{symbol}'. Please check symbol status or try again.")

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
# 5. CHART GENERATOR WITH MAPPED POI ZONES
# ==========================================
def generate_chart_image(df, title_str, is_long=True):
    """Renders dark-themed chart with 200 EMA (Gold Line) and mapped FVG, OB, & BB POI Zones."""
    img_buf = io.BytesIO()
    chart_df = df.tail(80).copy()
    n = len(chart_df)

    mc = mpf.make_marketcolors(
        up='#089981', down='#f23645',
        edge='inherit', wick='inherit', volume='in'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#2a2e39',
        y_on_right=True, facecolor='#131722', figcolor='#131722'
    )

    # Gold Line: 200 EMA Overlay
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

    # Map Structural POIs
    fvgs = detect_fvg(chart_df)
    obs = detect_order_blocks(chart_df)

    def idx_to_frac(i):
        """Convert a candle index into an axes-fraction x position (0-1),
        which is what axhspan's xmin/xmax actually expect."""
        return max(0.0, min(1.0, i / n))

    # 1. Render Fair Value Gaps (FVGs) — band now starts at the candle that formed it
    recent_fvgs = fvgs[-2:] if len(fvgs) >= 2 else fvgs
    for fvg in recent_fvgs:
        x0 = idx_to_frac(fvg['index'])
        if fvg['type'] == 'BULLISH_FVG':
            ax.axhspan(fvg['bottom'], fvg['top'], xmin=x0, xmax=1, color='#089981', alpha=0.25, hatch='//')
            ax.text(fvg['index'], fvg['top'], " Bullish FVG", color='#089981', fontsize=7, verticalalignment='bottom')
        elif fvg['type'] == 'BEARISH_FVG':
            ax.axhspan(fvg['bottom'], fvg['top'], xmin=x0, xmax=1, color='#f23645', alpha=0.25, hatch='\\\\')
            ax.text(fvg['index'], fvg['bottom'], " Bearish FVG", color='#f23645', fontsize=7, verticalalignment='top')

    # 2. Render Order Blocks & Breaker Blocks (OB / BB) — same fix, plus BULLISH_BB was missing entirely
    recent_obs = obs[-2:] if len(obs) >= 2 else obs
    for ob in recent_obs:
        x0 = idx_to_frac(ob['index'])
        label_x = ob['index']
        if ob['type'] == 'BULLISH_OB':
            ax.axhspan(ob['bottom'], ob['top'], xmin=x0, xmax=1, color='#00e676', alpha=0.30)
            ax.text(label_x, ob['top'], " Bullish OB", color='#00e676', fontsize=8, fontweight='bold')
        elif ob['type'] == 'BEARISH_OB':
            ax.axhspan(ob['bottom'], ob['top'], xmin=x0, xmax=1, color='#ff1744', alpha=0.30)
            ax.text(label_x, ob['bottom'], " Bearish OB", color='#ff1744', fontsize=8, fontweight='bold')
        elif ob['type'] == 'BEARISH_BB':
            ax.axhspan(ob['bottom'], ob['top'], xmin=x0, xmax=1, color='#ff9100', alpha=0.30)
            ax.text(label_x, ob['bottom'], " Breaker Block (BB)", color='#ff9100', fontsize=8, fontweight='bold')
        elif ob['type'] == 'BULLISH_BB':
            ax.axhspan(ob['bottom'], ob['top'], xmin=x0, xmax=1, color='#00b0ff', alpha=0.30)
            ax.text(label_x, ob['top'], " Breaker Block (BB)", color='#00b0ff', fontsize=8, fontweight='bold')

    ax.set_title(title_str, color='#d1d4dc', fontsize=9, fontweight='bold')
    img_buf.seek(0)
    return img_buf

# ==========================================
# 6. TELEGRAM BOT HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🤖 *MULTI-TIMEFRAME POI CHART ANALYZER*\n\n"
        "I map structural **Fair Value Gaps (FVG)**, **Order Blocks (OB)**, and **Breaker Blocks (BB)** directly onto your charts while enforcing 200 EMA macro context.\n\n"
        "📌 *COMMANDS:*\n"
        "• `/analyze [SYMBOL] [TIMEFRAME]`\n"
        "  _Examples:_ `/analyze GOLD 15m`, `/analyze XAUUSD 5m`, `/analyze GBPUSD 15m`\n\n"
        "📊 *ON-CHART POI LEGEND:*\n"
        "1. 🟡 **Gold Line:** 200 EMA Trend Filter.\n"
        "2. 🟩 **Green Zone:** Bullish Order Block (OB) / FVG Demand Zone.\n"
        "3. 🟥 **Red Zone:** Bearish Order Block (OB) / FVG Supply Zone.\n"
        "4. 🟧 **Orange Zone:** Breaker Block (BB) Support/Resistance Flip."
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args

    raw_symbol = args[0].upper() if len(args) > 0 else "GOLD"
    tf = args[1] if len(args) > 1 else "15m"

    status_msg = await update.message.reply_text(f"📊 Mapping FVG, OB & BB POI Zones for {raw_symbol}...")

    try:
        df_daily, df_entry, macro_is_bearish, daily_ema, ticker_str = fetch_multi_timeframe_data(raw_symbol, tf)

        if df_entry.empty:
            await status_msg.edit_text(f"❌ Error: Unable to fetch market data for {raw_symbol} ({ticker_str}).")
            return

        last_price = df_entry['Close'].iloc[-1]
        local_ema = df_entry['EMA200'].iloc[-1]
        local_is_bearish = last_price < local_ema

        macro_bias = "BEARISH" if macro_is_bearish else "BULLISH"
        local_bias = "BEARISH" if local_is_bearish else "BULLISH"

        fvgs = detect_fvg(df_entry)
        obs = detect_order_blocks(df_entry)

        fvg_summary = f"{len(fvgs)} Active FVG(s) detected" if fvgs else "No immediate FVG gaps"
        ob_summary = f"{len(obs)} Active OB/BB Zone(s) detected" if obs else "No immediate Order Blocks"

        prompt = f"""
        You are an ICT quantitative price-action trader.
        CRITICAL CONTEXT:
        - Symbol: {raw_symbol} ({ticker_str})
        - Daily 200 EMA Macro Bias: {macro_bias} (Daily Close: {df_daily['Close'].iloc[-1]:.2f}, Daily EMA: {daily_ema:.2f})
        - {tf.upper()} 200 EMA Local Bias: {local_bias} (Current Price: {last_price:.2f}, Local EMA: {local_ema:.2f})
        - Structural POI Map: {fvg_summary} | {ob_summary}

        RULE: If price is BELOW the 200 EMA (Gold Line), DO NOT recommend buys. Any pullbacks into Bearish OB or FVG supply zones are sell realignment setups.

        Analyze {raw_symbol} ({tf}):
        Format output:
        🎯 AI ANALYSIS & MARKET STORY
        1. Macro Bias: {macro_bias} (Daily 200 EMA)
        2. Execution Bias ({tf}): {local_bias}
        3. Structural POI Reaction: Detail how price is interacting with mapped FVG, OB, or Breaker Block zones.
        4. Realignment Setup: Detail expected entry trigger.
        """

        analysis_text = fetch_ai_analysis(prompt)

        is_long = not local_is_bearish
        title_str = f"{raw_symbol} ({tf.upper()}) | Trend: {local_bias} | Gold Line = 200 EMA"
        chart_img = generate_chart_image(df_entry, title_str, is_long=is_long)

        await context.bot.send_photo(
            chat_id=chat_id,
            photo=chart_img,
            caption=f"🎯 *LIVE POI MAP (FVG / OB / BB): {raw_symbol} ({tf.upper()})*",
            parse_mode="Markdown"
        )

        await status_msg.delete()
        await context.bot.send_message(chat_id=chat_id, text=analysis_text, parse_mode="Markdown")

    except Exception as e:
        await status_msg.edit_text(f"❌ Error generating analysis: {str(e)}")

# ==========================================
# 7. REAL-TIME SIGNAL SCANNER
# ==========================================
async def auto_market_scanner(context: ContextTypes.DEFAULT_TYPE):
    """Periodically scans liquidity sweeps and POI taps aligned with the 200 EMA."""
    if not MY_CHAT_ID:
        return

    symbols = ["GOLD", "GBPUSD", "EURUSD", "BTC"]
    for symbol in symbols:
        try:
            _, df_entry, local_is_bearish, _, ticker_str = fetch_multi_timeframe_data(symbol, "5m")
            if df_entry.empty:
                continue

            last_close = df_entry['Close'].iloc[-1]
            fvgs = detect_fvg(df_entry)

            for fvg in fvgs[-2:]:
                if local_is_bearish and fvg['type'] == 'BEARISH_FVG' and fvg['bottom'] <= last_close <= fvg['top']:
                    alert_text = (
                        f"🚨 *BEARISH POI ALERT: {symbol}*\n\n"
                        f"Price ({last_close:.2f}) tapped into a **Bearish FVG Zone** ({fvg['bottom']:.2f} - {fvg['top']:.2f}) below the 200 EMA.\n"
                        f"⚡ *Action:* Look for short realignment entries."
                    )
                    await context.bot.send_message(chat_id=MY_CHAT_ID, text=alert_text, parse_mode="Markdown")
                    break

        except Exception as e:
            print(f"Scanner error for {symbol}: {e}")

# ==========================================
# 8. APPLICATION INITIALIZATION & THREADING
# ==========================================
def run_telegram_bot():
    """Runs Telegram bot polling cleanly inside an independent AsyncIO loop."""
    if not TELEGRAM_BOT_TOKEN:
        print("[Error] TELEGRAM_BOT_TOKEN missing in environment variables! Telegram bot skipped.")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("analyze", analyze_command))

    job_queue = application.job_queue
    if job_queue:
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
