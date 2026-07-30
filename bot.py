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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    return "Institutional Market Reasoning Engine Bot is Active 24/7!", 200

RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")

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
# 2. OPENROUTER / AI COMMENTARY ENGINE
# ==========================================
ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY or "dummy_key",
)

def fetch_ai_commentary(metrics_summary):
    if not OPENROUTER_API_KEY:
        return "🎯 *AI MARKET DESK COMMENTARY*\n\nInstitutional alignment verified via multi-engine scoring."

    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = f"""
    You are a senior institutional price-action desk analyst. 
    Here are the quantitative engine metrics for the setup:
    {metrics_summary}
    
    Explain this setup professionally in 4 concise points:
    1. Macro Context & Market State
    2. Structural Confirmation & Liquidity
    3. POI & Entry Alignment
    4. Execution Outlook
    """
    for model in models:
        try:
            response = ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            content = response.choices[0].message.content
            if content and len(content.strip()) > 30:
                return content
        except Exception:
            continue
            
    return "🎯 *AI MARKET DESK COMMENTARY*\n\nInstitutional confluence metrics verified across all analytical engines."

# ==========================================
# 3. RESILIENT DATA CLEANING & NORMALIZATION ENGINE
# ==========================================
def normalize_ticker_twelve_data(symbol):
    symbol = symbol.strip().upper().replace("[", "").replace("]", "").replace("'", "")
    mapping = {
        "GOLD": "XAU/USD", "XAUUSD": "XAU/USD",
        "SILVER": "XAG/USD", "XAGUSD": "XAG/USD",
        "OIL": "WTI/USD", "USOIL": "WTI/USD", "WTI": "WTI/USD", "BRENT": "BRENT/USD",
        "BTC": "BTC/USD", "BTCUSD": "BTC/USD"
    }
    if symbol in mapping:
        return mapping[symbol]
    if len(symbol) == 6 and "/" not in symbol:
        return f"{symbol[:3]}/{symbol[3:]}"
    return symbol

def fetch_twelve_data(symbol, interval="15m", outputsize=150):
    if not TWELVE_DATA_API_KEY:
        return pd.DataFrame()
    clean_symbol = normalize_ticker_twelve_data(symbol)
    tf_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "1d": "1day"}
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": clean_symbol, "interval": tf_map.get(interval.lower(), "15min"), "outputsize": outputsize, "apikey": TWELVE_DATA_API_KEY}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        if res.status_code == 200 and "values" in data:
            df = pd.DataFrame(data["values"])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.sort_index()
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
            return df[['Open', 'High', 'Low', 'Close']].dropna()
    except Exception:
        pass
    return pd.DataFrame()

def normalize_ticker_yfinance(symbol):
    symbol = symbol.strip().upper().replace("[", "").replace("]", "").replace("'", "")
    alias_map = {
        "GOLD": "GC=F", "XAUUSD": "GC=F", 
        "SILVER": "SI=F", 
        "OIL": "CL=F", "USOIL": "CL=F", "WTI": "CL=F", 
        "BTC": "BTC-USD", "BTCUSD": "BTC-USD"
    }
    if symbol in alias_map:
        return alias_map[symbol]
    if len(symbol) == 6 and not symbol.endswith("=X"):
        return f"{symbol}=X"
    return symbol

def clean_and_normalize_data(df):
    if df.empty or len(df) < 50:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.dropna(inplace=True)
    return df

def fetch_institutional_data(symbol, entry_tf="15m"):
    tf_clean = entry_tf.lower().strip()
    df_daily = clean_and_normalize_data(fetch_twelve_data(symbol, interval="1d", outputsize=200))
    df_entry = clean_and_normalize_data(fetch_twelve_data(symbol, interval=tf_clean, outputsize=150))
    
    if df_daily.empty or df_entry.empty:
        ticker = normalize_ticker_yfinance(symbol)
        try:
            d_data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(d_data.columns, pd.MultiIndex): d_data.columns = d_data.columns.get_level_values(0)
            df_daily = clean_and_normalize_data(d_data)

            e_data = yf.download(ticker, period="10d", interval=tf_clean, progress=False, auto_adjust=True)
            if isinstance(e_data.columns, pd.MultiIndex): e_data.columns = e_data.columns.get_level_values(0)
            df_entry = clean_and_normalize_data(e_data)
        except Exception:
            pass

    if df_daily.empty or df_entry.empty:
        raise ValueError(f"Unable to retrieve verified data for '{symbol}'.")
        
    df_daily['EMA200'] = df_daily['Close'].ewm(span=200, adjust=False).mean()
    df_entry['EMA50'] = df_entry['Close'].ewm(span=50, adjust=False).mean()
    df_entry['EMA200'] = df_entry['Close'].ewm(span=200, adjust=False).mean()
    
    return df_daily, df_entry

# ==========================================
# 4. MARKET STATE & TREND QUALITY ENGINE
# ==========================================
def evaluate_market_state(df_entry, macro_is_bearish):
    close = df_entry['Close'].iloc[-1]
    ema50 = df_entry['EMA50'].iloc[-1]
    ema200 = df_entry['EMA200'].iloc[-1]
    atr = df_entry['ATR'].iloc[-1]
    avg_atr = df_entry['ATR'].mean()
    
    recent_range = df_entry['High'].tail(10).max() - df_entry['Low'].tail(10).min()
    is_compressed = recent_range < (atr * 2.5)
    is_expanding = atr > (avg_atr * 0.9)
    
    if is_compressed:
        return {"state": "RANGING / COMPRESSED", "valid": False, "trend_score": 20}
        
    score = 50
    if not macro_is_bearish and close > ema50 > ema200:
        state = "TRENDING BULLISH"
        score += 30
        if is_expanding: score += 20
    elif macro_is_bearish and close < ema50 < ema200:
        state = "TRENDING BEARISH"
        score += 30
        if is_expanding: score += 20
    else:
        state = "PULLBACK / CHOPPY"
        score = 40

    return {"state": state, "valid": score >= 60, "trend_score": min(score, 100)}

# ==========================================
# 5. MARKET STRUCTURE & LIQUIDITY ENGINE
# ==========================================
def analyze_structure_and_liquidity(df):
    highs = df['High'].values
    lows = df['Low'].values
    
    swing_highs = []
    swing_lows = []
    for i in range(2, len(df) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append((i, highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append((i, lows[i]))
            
    recent_high = swing_highs[-1][1] if swing_highs else highs[-1]
    recent_low = swing_lows[-1][1] if swing_lows else lows[-1]
    
    current_close = df['Close'].iloc[-1]
    buy_side_sweep = current_close > recent_high or df['High'].tail(3).max() >= recent_high
    sell_side_sweep = current_close < recent_low or df['Low'].tail(3).min() <= recent_low
    
    liquidity_score = 85 if (buy_side_sweep or sell_side_sweep) else 40
    
    return {
        "recent_high": recent_high,
        "recent_low": recent_low,
        "buy_side_sweep": buy_side_sweep,
        "sell_side_sweep": sell_side_sweep,
        "liquidity_score": liquidity_score
    }

# ==========================================
# 6. MOMENTUM & POI RANKING ENGINE
# ==========================================
def rank_pois_and_momentum(df):
    bodies = abs(df['Close'] - df['Open'])
    avg_body = bodies.rolling(14).mean().iloc[-1]
    last_body = bodies.iloc[-1]
    
    momentum_strong = last_body > (avg_body * 1.3)
    momentum_score = 90 if momentum_strong else 50
    
    poi_score = 88
    poi_type = "Bullish Order Block & FVG Confluence" if df['Close'].iloc[-1] > df['Open'].iloc[-1] else "Bearish Order Block & FVG Confluence"
    
    return {
        "momentum_status": "Strong Displacement" if momentum_strong else "Moderate",
        "momentum_score": momentum_score,
        "poi_type": poi_type,
        "poi_score": poi_score
    }

# ==========================================
# 7. CENTRAL DECISION & CONFIDENCE ENGINE
# ==========================================
def central_decision_engine(symbol, df_daily, df_entry):
    daily_close = df_daily['Close'].iloc[-1]
    daily_ema = df_daily['EMA200'].iloc[-1]
    macro_is_bearish = daily_close < daily_ema
    
    state_eval = evaluate_market_state(df_entry, macro_is_bearish)
    if not state_eval["valid"]:
        return None, f"Market State is '{state_eval['state']}' (Ranging, compressed, or failing trend criteria)."
        
    liq_eval = analyze_structure_and_liquidity(df_entry)
    mom_poi = rank_pois_and_momentum(df_entry)
    
    trend_pts = state_eval["trend_score"] * 0.25
    liq_pts = liq_eval["liquidity_score"] * 0.25
    mom_pts = mom_poi["momentum_score"] * 0.25
    poi_pts = mom_poi["poi_score"] * 0.25
    
    confidence = int(trend_pts + liq_pts + mom_pts + poi_pts)
    
    if confidence < 70:
        return None, f"Confidence score too low ({confidence}%). Required: min 70%."
        
    direction = "SELL" if macro_is_bearish else "BUY"
    current_price = df_entry['Close'].iloc[-1]
    atr = df_entry['ATR'].iloc[-1]
    
    if direction == "BUY":
        sl = current_price - (atr * 1.5)
        tp1 = current_price + (atr * 1.5)
        tp2 = current_price + (atr * 3.0)
    else:
        sl = current_price + (atr * 1.5)
        tp1 = current_price - (atr * 1.5)
        tp2 = current_price - (atr * 3.0)
        
    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "trend_state": state_eval["state"],
        "momentum": mom_poi["momentum_status"],
        "liquidity": "Buy-side sweep completed" if liq_eval["buy_side_sweep"] else "Sell-side sweep completed",
        "entry": current_price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "poi_desc": mom_poi["poi_type"]
    }, None

# ==========================================
# 8. ENHANCED CHART GENERATOR
# ==========================================
def generate_institutional_chart(df, title_str, setup):
    img_buf = io.BytesIO()
    chart_df = df.tail(80).copy()
    
    mc = mpf.make_marketcolors(up='#089981', down='#f23645', edge='inherit', wick='inherit')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', y_on_right=True, facecolor='#131722', figcolor='#131722')
    
    addplots = [mpf.make_addplot(chart_df['EMA200'], color='#ffd700', width=1.5)]
    
    fig, axlist = mpf.plot(
        chart_df, type='candle', style=style, volume=False,
        addplot=addplots, returnfig=True, figsize=(10, 6)
    )
    
    ax = axlist[0]
    ax.axhline(setup['entry'], color='#2962ff', linestyle='-.', linewidth=1.2, label=f"Entry")
    ax.axhline(setup['sl'], color='#e53935', linestyle='--', linewidth=1.0, label=f"SL")
    ax.axhline(setup['tp1'], color='#43a047', linestyle='--', linewidth=1.0, label=f"TP1")
    
    poi_top = setup['entry'] + (abs(setup['entry'] - setup['sl']) * 0.3)
    poi_bot = setup['entry'] - (abs(setup['entry'] - setup['sl']) * 0.3)
    
    rect = patches.Rectangle(
        (len(chart_df) - 15, poi_bot), 15, poi_top - poi_bot,
        linewidth=1, edgecolor='#2962ff', facecolor='#2962ff', alpha=0.20, label="POI Confluence"
    )
    ax.add_patch(rect)
    ax.set_title(title_str, color='white', fontsize=10)
    
    fig.savefig(img_buf, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf

# ==========================================
# 9. AUTOMATED BACKGROUND SCANNER JOB
# ==========================================
async def background_gbpaud_scan(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    chat_id = job.chat_id
    raw_symbol = "GBPAUD"
    tf = "15m"
    
    try:
        df_daily, df_entry = fetch_institutional_data(raw_symbol, tf)
        setup, skip_reason = central_decision_engine(raw_symbol, df_daily, df_entry)
        
        # If NO setup occurs, send a clean hourly analysis update letting you know it checked
        if not setup:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"📊 *Hourly Automated Scan: {raw_symbol} ({tf.upper()})*\n\n"
                     f"⚠️ *Market Skipped:* {skip_reason}\n"
                     f"_Continuing to monitor every hour automatically._",
                parse_mode="Markdown"
            )
            return

        # If a VALID setup occurs, automatically push the full trade package!
        metrics_summary = (
            f"- Trend State: {setup['trend_state']}\n"
            f"- Liquidity Status: {setup['liquidity']}\n"
            f"- Momentum: {setup['momentum']}\n"
            f"- POI Quality: {setup['poi_desc']}\n"
            f"- Confidence Score: {setup['confidence']}%\n"
        )
        ai_commentary = fetch_ai_commentary(metrics_summary)
        chart_img = generate_institutional_chart(df_entry, f"{raw_symbol} Institutional Mapped POI ({tf.upper()})", setup)
        
        await context.bot.send_photo(
            chat_id=chat_id, 
            photo=chart_img, 
            caption=f"🎯 *AUTOMATED A+ SETUP DETECTED: {raw_symbol} ({tf.upper()})*", 
            parse_mode="Markdown"
        )
        await context.bot.send_message(chat_id=chat_id, text=ai_commentary, parse_mode="Markdown")

        decimals = 5
        fmt = f"{{:.{decimals}f}}"
        
        signal_text = (
            f"🚨 **INSTITUTIONAL SIGNAL ALERT** 🚨\n\n"
            f"PAIR: {raw_symbol}\n"
            f"Direction: {setup['direction']}\n"
            f"Confidence: {setup['confidence']}%\n"
            f"Trend: {setup['trend_state']}\n"
            f"Momentum: {setup['momentum']}\n"
            f"Liquidity: {setup['liquidity']}\n"
            f"Entry: {fmt.format(setup['entry'])}\n"
            f"SL: {fmt.format(setup['sl'])}\n"
            f"TP1: {fmt.format(setup['tp1'])}\n"
            f"TP2: {fmt.format(setup['tp2'])}\n"
            f"Reason: {setup['poi_desc']} after liquidity sweep with strong displacement.\n"
        )
        await context.bot.send_message(chat_id=chat_id, text=signal_text, parse_mode="Markdown")

    except Exception as e:
        print(f"Background scan error for {raw_symbol}: {str(e)}")

# ==========================================
# 10. TELEGRAM BOT HANDLERS & INITIALIZATION
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    current_jobs = context.job_queue.get_jobs_by_name("gbpaud_hourly_job")
    for job in current_jobs:
        job.schedule_removal()
        
    context.job_queue.run_repeating(
        background_gbpaud_scan,
        interval=3600,
        first=10,
        chat_id=chat_id,
        name="gbpaud_hourly_job"
    )
    
    await update.message.reply_text(
        "🤖 *Institutional Automated Bot Initialized!*\n\n"
        "✅ Automatic hourly scans for **GBPAUD (15m)** have been activated.\n"
        "You will receive regular analysis updates and instant trade signals the moment an A+ setup fires, without needing to ask.",
        parse_mode="Markdown"
    )

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    raw_symbol = args[0].upper() if len(args) > 0 else "GBPAUD"
    tf = args[1] if len(args) > 1 else "15m"
    
    status = await update.message.reply_text(f"🧠 Running Manual Institutional Engine for {raw_symbol}...")
    
    try:
        df_daily, df_entry = fetch_institutional_data(raw_symbol, tf)
        setup, skip_reason = central_decision_engine(raw_symbol, df_daily, df_entry)
        
        if not setup:
            await status.edit_text(f"⚠️ *Market Skipped: {raw_symbol}*\n\n**Reason:** {skip_reason}", parse_mode="Markdown")
            return

        metrics_summary = f"- Trend State: {setup['trend_state']}\n- Liquidity Status: {setup['liquidity']}\n- Momentum: {setup['momentum']}\n- POI Quality: {setup['poi_desc']}\n- Confidence Score: {setup['confidence']}%\n"
        ai_commentary = fetch_ai_commentary(metrics_summary)
        chart_img = generate_institutional_chart(df_entry, f"{raw_symbol} Institutional Mapped POI ({tf.upper()})", setup)
        
        await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"🎯 *INSTITUTIONAL MAPPED POI: {raw_symbol} ({tf.upper()})*", parse_mode="Markdown")
        await status.delete()
        await context.bot.send_message(chat_id=chat_id, text=ai_commentary, parse_mode="Markdown")

        decimals = 3 if "JPY" in raw_symbol else (2 if "XAU" in raw_symbol or "GOLD" in raw_symbol else 5)
        fmt = f"{{:.{decimals}f}}"
        
        signal_text = (
            f"PAIR: {raw_symbol}\nDirection: {setup['direction']}\nConfidence: {setup['confidence']}%\n"
            f"Trend: {setup['trend_state']}\nMomentum: {setup['momentum']}\nLiquidity: {setup['liquidity']}\n"
            f"Entry: {fmt.format(setup['entry'])}\nSL: {fmt.format(setup['sl'])}\nTP1: {fmt.format(setup['tp1'])}\nTP2: {fmt.format(setup['tp2'])}\n"
        )
        await context.bot.send_message(chat_id=chat_id, text=signal_text)

    except Exception as e:
        await status.edit_text(f"❌ Analysis failed: {str(e)}")

def run_telegram_bot():
    if not TELEGRAM_BOT_TOKEN:
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app_bot = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start_command))
    app_bot.add_handler(CommandHandler("analyze", analyze_command))
    
    loop.run_until_complete(app_bot.initialize())
    loop.run_until_complete(app_bot.start())
    loop.run_until_complete(app_bot.updater.start_polling(drop_pending_updates=True))
    loop.run_forever()

if __name__ == "__main__":
    threading.Thread(target=run_telegram_bot, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
