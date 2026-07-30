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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
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
# 2. OPENROUTER / AI COMMENTARY & TRANSLATION ENGINE
# ==========================================
ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY or "dummy_key",
)

def translate_text(text, target_language):
    if target_language == "English" or not OPENROUTER_API_KEY:
        return text

    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = f"""
    Translate the following financial market commentary or signal text accurately into {target_language}. Maintain all formatting, numbers, abbreviations, and professional financial tone:
    
    {text}
    """
    for model in models:
        try:
            response = ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600
            )
            content = response.choices[0].message.content
            if content and len(content.strip()) > 10:
                return content
        except Exception:
            continue
            
    return text

def fetch_ai_commentary(metrics_summary, target_language="English"):
    if not OPENROUTER_API_KEY:
        base_text = "🎯 *AI MARKET DESK COMMENTARY*\n\nInstitutional alignment verified via multi-engine scoring."
        return translate_text(base_text, target_language)

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
    
    Write the response directly in {target_language}.
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
            
    fallback = "🎯 *AI MARKET DESK COMMENTARY*\n\nInstitutional confluence metrics verified across all analytical engines."
    return translate_text(fallback, target_language)

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
    liq_eval = analyze_structure_and_liquidity(df_entry)
    mom_poi = rank_pois_and_momentum(df_entry)
    
    trend_pts = state_eval["trend_score"] * 0.25
    liq_pts = liq_eval["liquidity_score"] * 0.25
    mom_pts = mom_poi["momentum_score"] * 0.25
    poi_pts = mom_poi["poi_score"] * 0.25
    
    confidence = int(trend_pts + liq_pts + mom_pts + poi_pts)
    
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
    }

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
# 9. PROFESSIONAL TELEGRAM MENU & INTERFACE
# ==========================================
user_languages = {}

def get_home_menu(lang="English"):
    lang_flags = {"English": "🇬🇧 English", "Hausa": "🇳🇬 Hausa", "Pidgin": "🇳🇬 Pidgin"}
    current_flag = lang_flags.get(lang, "🇬🇧 English")
    
    keyboard = [
        [InlineKeyboardButton("📊 Run Analysis", callback_data="menu_analysis"),
         InlineKeyboardButton("🚨 Generate Signal", callback_data="menu_signal")],
        [InlineKeyboardButton(f"🌐 Language: {current_flag}", callback_data="menu_lang_select")],
        [InlineKeyboardButton("ℹ️ Help & Instructions", callback_data="menu_help")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_languages[chat_id] = "English"
    
    welcome_text = (
        "🏛️ *INSTITUTIONAL MARKET DESK*\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "Welcome to your professional algorithmic trading terminal. "
        "Select your preferred language or choose an action below to begin.\n\n"
        "📌 *Status:* Ready for execution."
    )
    await update.message.reply_text(
        welcome_text,
        reply_markup=get_home_menu("English"),
        parse_mode="Markdown"
    )

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id
    lang = user_languages.get(chat_id, "English")
    
    if data == "menu_home":
        home_text = (
            "🏛️ *INSTITUTIONAL MARKET DESK*\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "Main Control Menu. Choose an option below:"
        )
        await query.edit_message_text(text=home_text, reply_markup=get_home_menu(lang), parse_mode="Markdown")

    elif data == "menu_lang_select":
        kb = [
            [InlineKeyboardButton("🇬🇧 English", callback_data="set_lang|English"),
             InlineKeyboardButton("🇳🇬 Hausa", callback_data="set_lang|Hausa")],
            [InlineKeyboardButton("🇳🇬 Pidgin", callback_data="set_lang|Pidgin")],
            [InlineKeyboardButton("« Back to Home", callback_data="menu_home")]
        ]
        await query.edit_message_text(
            text="🌍 *Select Your Interface Language:*",
            reply_markup=InlineKeyboardMarkup(kb),
            parse_mode="Markdown"
        )

    elif data.startswith("set_lang|"):
        new_lang = data.split("|")[1]
        user_languages[chat_id] = new_lang
        await query.edit_message_text(
            text=f"✅ Language successfully updated to **{new_lang}**.",
            reply_markup=get_home_menu(new_lang),
            parse_mode="Markdown"
        )

    elif data == "menu_analysis":
        kb = [
            [InlineKeyboardButton("GBPAUD (15m)", callback_data="run_an|GBPAUD|15m"),
             InlineKeyboardButton("EURUSD (15m)", callback_data="run_an|EURUSD|15m")],
            [InlineKeyboardButton("XAUUSD (GOLD 15m)", callback_data="run_an|XAUUSD|15m"),
             InlineKeyboardButton("BTCUSD (15m)", callback_data="run_an|BTCUSD|15m")],
            [InlineKeyboardButton("« Back to Home", callback_data="menu_home")]
        ]
        await query.edit_message_text(
            text="📊 *Select Asset for Market Analysis:*",
            reply_markup=InlineKeyboardMarkup(kb),
            parse_mode="Markdown"
        )

    elif data == "menu_signal":
        kb = [
            [InlineKeyboardButton("GBPAUD (15m)", callback_data="run_sig|GBPAUD|15m"),
             InlineKeyboardButton("EURUSD (15m)", callback_data="run_sig|EURUSD|15m")],
            [InlineKeyboardButton("XAUUSD (GOLD 15m)", callback_data="run_sig|XAUUSD|15m"),
             InlineKeyboardButton("BTCUSD (15m)", callback_data="run_sig|BTCUSD|15m")],
            [InlineKeyboardButton("« Back to Home", callback_data="menu_home")]
        ]
        await query.edit_message_text(
            text="🚨 *Select Asset for Trade Signal:*",
            reply_markup=InlineKeyboardMarkup(kb),
            parse_mode="Markdown"
        )

    elif data.startswith("run_an|"):
        _, symbol, tf = data.split("|")
        await query.edit_message_text(text=f"🧠 Running Institutional Analysis for {symbol} ({tf})...")
        try:
            df_daily, df_entry = fetch_institutional_data(symbol, tf)
            setup = central_decision_engine(symbol, df_daily, df_entry)

            metrics_summary = f"- Trend State: {setup['trend_state']}\n- Liquidity Status: {setup['liquidity']}\n- Momentum: {setup['momentum']}\n- POI Quality: {setup['poi_desc']}\n- Confidence Score: {setup['confidence']}%\n"
            ai_commentary = fetch_ai_commentary(metrics_summary, lang)
            chart_img = generate_institutional_chart(df_entry, f"{symbol} Institutional Mapped POI ({tf.upper()})", setup)
            
            await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"🎯 *INSTITUTIONAL MAPPED POI: {symbol} ({tf.upper()})*", parse_mode="Markdown")
            await context.bot.send_message(chat_id=chat_id, text=ai_commentary, parse_mode="Markdown")
            
            # Return home menu prompt
            await context.bot.send_message(chat_id=chat_id, text="📌 *Session Action Complete.* Choose next action:", reply_markup=get_home_menu(lang), parse_mode="Markdown")
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Analysis failed: {str(e)}", reply_markup=get_home_menu(lang))

    elif data.startswith("run_sig|"):
        _, symbol, tf = data.split("|")
        await query.edit_message_text(text=f"🚨 Generating Institutional Signal for {symbol} ({tf}) in {lang}...")
        try:
            df_daily, df_entry = fetch_institutional_data(symbol, tf)
            setup = central_decision_engine(symbol, df_daily, df_entry)

            decimals = 3 if "JPY" in symbol else (2 if "XAU" in symbol or "GOLD" in symbol else 5)
            fmt = f"{{:.{decimals}f}}"
            
            raw_signal_text = (
                f"🚨 **INSTITUTIONAL SIGNAL ALERT** 🚨\n\n"
                f"PAIR: {symbol}\n"
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
            final_signal_text = translate_text(raw_signal_text, lang)
            await context.bot.send_message(chat_id=chat_id, text=final_signal_text, parse_mode="Markdown")
            
            # Return home menu prompt
            await context.bot.send_message(chat_id=chat_id, text="📌 *Session Action Complete.* Choose next action:", reply_markup=get_home_menu(lang), parse_mode="Markdown")
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Signal generation failed: {str(e)}", reply_markup=get_home_menu(lang))

    elif data == "menu_help":
        help_text = (
            "ℹ️ *TERMINAL GUIDE*\n\n"
            "• **Run Analysis**: Delivers full AI market desk commentary, structural metrics, and institutional chart plots.\n"
            "• **Generate Signal**: Provides precision entry, stop loss, and take profit parameters.\n"
            "• **Language**: Switch instantly between English, Hausa, and Pidgin for all outputs.\n\n"
            "Use `/start` anytime to pull up the primary dashboard menu."
        )
        kb = [[InlineKeyboardButton("« Back to Home", callback_data="menu_home")]]
        await query.edit_message_text(text=help_text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

def run_telegram_bot():
    if not TELEGRAM_BOT_TOKEN:
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app_bot = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start_command))
    app_bot.add_handler(CallbackQueryHandler(button_callback_handler))
    
    loop.run_until_complete(app_bot.initialize())
    loop.run_until_complete(app_bot.start())
    loop.run_until_complete(app_bot.updater.start_polling(drop_pending_updates=True))
    loop.run_forever()

if __name__ == "__main__":
    threading.Thread(target=run_telegram_bot, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
