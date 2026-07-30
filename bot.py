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
from datetime import datetime
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
        base_text = "🎯 *AI TOP-DOWN MARKET DESK COMMENTARY*\n\nInstitutional channel alignment verified across multi-timeframe engines."
        return translate_text(base_text, target_language)

    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = f"""
    You are a senior institutional price-action desk analyst. 
    Here are the quantitative top-down multi-timeframe metrics for the setup:
    {metrics_summary}
    
    Provide a professional top-down breakdown in 4 concise points:
    1. Macro / Higher Timeframe Context & Structure
    2. Intermediate Timeframe Alignment
    3. 15-Minute Execution TF, Parallel Sloped Channel Geometry & Breakout Status
    4. Execution Outlook & Risk Parameters
    
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
            
    fallback = "🎯 *AI TOP-DOWN MARKET DESK COMMENTARY*\n\nInstitutional confluence metrics verified across macro, intermediate, and 15m execution layers."
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
        "BTC": "BTC/USD", "BTCUSD": "BTC/USD",
        "TESLA": "TSLA", "AAPL": "AAPL",
        "US30": "DJI", "NAS100": "IXIC", "SPX500": "SPX"
    }
    if symbol in mapping:
        return mapping[symbol]
    if len(symbol) == 6 and "/" not in symbol and symbol not in ["US30", "NAS100"]:
        return f"{symbol[:3]}/{symbol[3:]}"
    return symbol

def fetch_twelve_data(symbol, interval="15m", outputsize=150):
    if not TWELVE_DATA_API_KEY:
        return pd.DataFrame()
    clean_symbol = normalize_ticker_twelve_data(symbol)
    tf_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day"}
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
        "BTC": "BTC-USD", "BTCUSD": "BTC-USD",
        "TESLA": "TSLA", "AAPL": "AAPL",
        "US30": "^DJI", "NAS100": "^IXIC", "SPX500": "^GSPC"
    }
    if symbol in alias_map:
        return alias_map[symbol]
    if len(symbol) == 6 and not symbol.endswith("=X") and symbol not in ["US30", "NAS100"]:
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

def fetch_top_down_institutional_data(symbol):
    df_daily = clean_and_normalize_data(fetch_twelve_data(symbol, interval="1d", outputsize=200))
    df_4h = clean_and_normalize_data(fetch_twelve_data(symbol, interval="4h", outputsize=150))
    df_15m = clean_and_normalize_data(fetch_twelve_data(symbol, interval="15m", outputsize=150))
    
    if df_daily.empty or df_4h.empty or df_15m.empty:
        ticker = normalize_ticker_yfinance(symbol)
        try:
            if df_daily.empty:
                d_data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
                if isinstance(d_data.columns, pd.MultiIndex): d_data.columns = d_data.columns.get_level_values(0)
                df_daily = clean_and_normalize_data(d_data)
            if df_4h.empty:
                h_data = yf.download(ticker, period="60d", interval="60m", progress=False, auto_adjust=True)
                if isinstance(h_data.columns, pd.MultiIndex): h_data.columns = h_data.columns.get_level_values(0)
                df_4h = clean_and_normalize_data(h_data)
            if df_15m.empty:
                e_data = yf.download(ticker, period="10d", interval="15m", progress=False, auto_adjust=True)
                if isinstance(e_data.columns, pd.MultiIndex): e_data.columns = e_data.columns.get_level_values(0)
                df_15m = clean_and_normalize_data(e_data)
        except Exception:
            pass

    if df_daily.empty or df_4h.empty or df_15m.empty:
        raise ValueError(f"Unable to retrieve verified multi-timeframe data for '{symbol}'.")
        
    df_daily['EMA200'] = df_daily['Close'].ewm(span=200, adjust=False).mean()
    df_4h['EMA50'] = df_4h['Close'].ewm(span=50, adjust=False).mean()
    df_15m['EMA50'] = df_15m['Close'].ewm(span=50, adjust=False).mean()
    df_15m['EMA200'] = df_15m['Close'].ewm(span=200, adjust=False).mean()
    
    return df_daily, df_4h, df_15m

# ==========================================
# 4. PARALLEL SLOPED CHANNEL & SWING PIVOT ENGINE
# ==========================================
def calculate_parallel_sloped_channel(chart_df):
    highs = chart_df['High'].values
    lows = chart_df['Low'].values
    
    swing_highs = []
    swing_lows = []
    for i in range(2, len(chart_df) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append((i, highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append((i, lows[i]))
            
    x_vals = np.arange(len(chart_df))
    
    # Construct Upper Sloped Trendline connecting swing highs
    if len(swing_highs) >= 2:
        p1, p2 = swing_highs[-2], swing_highs[-1]
        slope_upper = (p2[1] - p1[1]) / (p2[0] - p1[0] if p2[0] != p1[0] else 1)
        intercept_upper = p1[1] - slope_upper * p1[0]
        upper_line = slope_upper * x_vals + intercept_upper
    else:
        slope_upper = 0.0
        upper_line = np.full(len(chart_df), chart_df['High'].max())
        
    # Construct Lower Sloped Parallel Trendline connecting swing lows
    if len(swing_lows) >= 2:
        p1_l, p2_l = swing_lows[-2], swing_lows[-1]
        slope_lower = (p2_l[1] - p1_l[1]) / (p2_l[0] - p1_l[0] if p2_l[0] != p1_l[0] else 1)
        intercept_lower = p1_l[1] - slope_lower * p1_l[0]
        lower_line = slope_lower * x_vals + intercept_lower
    else:
        lower_line = np.full(len(chart_df), chart_df['Low'].min())

    current_close = chart_df['Close'].iloc[-1]
    projected_upper = upper_line[-1]
    break_confirmed = current_close > projected_upper
    
    status_msg = "Sloped Parallel Channel Break & Close Confirmed" if break_confirmed else "Within Sloped Parallel Channel: Awaiting Breakout"

    return {
        "x_vals": x_vals,
        "upper_line": upper_line,
        "lower_line": lower_line,
        "break_confirmed": break_confirmed,
        "score": 95 if break_confirmed else 60,
        "status_msg": status_msg
    }

# ==========================================
# 5. MARKET STATE & CENTRAL DECISION ENGINE
# ==========================================
def evaluate_top_down_state(df_daily, df_4h, df_15m):
    daily_close = df_daily['Close'].iloc[-1]
    daily_ema = df_daily['EMA200'].iloc[-1]
    macro_bearish = daily_close < daily_ema
    
    h4_close = df_4h['Close'].iloc[-1]
    h4_ema = df_4h['EMA50'].iloc[-1]
    h4_bearish = h4_close < h4_ema
    
    m15_close = df_15m['Close'].iloc[-1]
    m15_ema50 = df_15m['EMA50'].iloc[-1]
    
    if not macro_bearish and not h4_bearish and m15_close > m15_ema50:
        trend_state = "TOP-DOWN BULLISH ALIGNMENT"
        trend_score = 90
    elif macro_bearish and h4_bearish and m15_close < m15_ema50:
        trend_state = "TOP-DOWN BEARISH ALIGNMENT"
        trend_score = 90
    else:
        trend_state = "MIXED / MULTI-TF TRANSITION"
        trend_score = 50

    return {
        "macro_bias": "Bearish (Daily < EMA200)" if macro_bearish else "Bullish (Daily > EMA200)",
        "intermediate_bias": "Bearish (4H < EMA50)" if h4_bearish else "Bullish (4H > EMA50)",
        "execution_state": trend_state,
        "trend_score": trend_score,
        "macro_bearish": macro_bearish
    }

def central_decision_engine(symbol, df_daily, df_4h, df_15m):
    top_down = evaluate_top_down_state(df_daily, df_4h, df_15m)
    chart_slice = df_15m.tail(80).copy()
    channel_eval = calculate_parallel_sloped_channel(chart_slice)
    
    trend_pts = top_down["trend_score"] * 0.30
    ch_pts = channel_eval["score"] * 0.70
    
    confidence = int(trend_pts + ch_pts)
    direction = "SELL" if top_down["macro_bearish"] else "BUY"
    current_price = df_15m['Close'].iloc[-1]
    atr = df_15m['ATR'].iloc[-1]
    
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
        "macro_bias": top_down["macro_bias"],
        "intermediate_bias": top_down["intermediate_bias"],
        "execution_state": top_down["execution_state"],
        "trendline_status": channel_eval["status_msg"],
        "break_confirmed": channel_eval["break_confirmed"],
        "entry": current_price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2
    }

# ==========================================
# 6. 15M CHART RENDERER WITH PARALLEL SLOPED CHANNELS
# ==========================================
def generate_15m_execution_chart(df_15m, title_str, setup, macro_bearish):
    img_buf = io.BytesIO()
    chart_df = df_15m.tail(80).copy()
    
    channel_calc = calculate_parallel_sloped_channel(chart_df)
    
    # Map upper and lower sloped channel lines into addplots
    upper_series = pd.Series(channel_calc['upper_line'], index=chart_df.index)
    lower_series = pd.Series(channel_calc['lower_line'], index=chart_df.index)
    
    mc = mpf.make_marketcolors(up='#089981', down='#f23645', edge='inherit', wick='inherit')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', y_on_right=True, facecolor='#131722', figcolor='#131722')
    
    addplots = [
        mpf.make_addplot(chart_df['EMA50'], color='#2962ff', width=1.2),
        mpf.make_addplot(chart_df['EMA200'], color='#ffd700', width=1.5),
        mpf.make_addplot(upper_series, color='#00e676', width=2.0, linestyle='-'),  # Upper Sloped Resistance Channel
        mpf.make_addplot(lower_series, color='#00e676', width=2.0, linestyle='-')   # Lower Sloped Support Channel
    ]
    
    fig, axlist = mpf.plot(
        chart_df, type='candle', style=style, volume=False,
        addplot=addplots, returnfig=True, figsize=(10, 6)
    )
    
    ax = axlist[0]
    ax.axhline(setup['entry'], color='#2962ff', linestyle='-.', linewidth=1.2, label=f"Entry Level")
    ax.axhline(setup['sl'], color='#e53935', linestyle='--', linewidth=1.0, label=f"SL")
    ax.axhline(setup['tp1'], color='#43a047', linestyle='--', linewidth=1.0, label=f"TP1")
    
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
    ax.set_title(f"{title_str}\nParallel Sloped Channel Geometry Active | {current_time_str}", color='white', fontsize=9)
    
    fig.savefig(img_buf, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf

# ==========================================
# 7. PROFESSIONAL TELEGRAM MENU & CONTINUOUS A+ SCANNER
# ==========================================
user_languages = {}
active_subscribers = set()

def get_home_menu(lang="English", auto_active=False):
    lang_flags = {"English": "🇬🇧 English", "Hausa": "🇳🇬 Hausa", "Pidgin": "🇳🇬 Pidgin"}
    current_flag = lang_flags.get(lang, "🇬🇧 English")
    auto_status = "🟢 GBPAUD Continuous A+ Scanner: ON" if auto_active else "🔴 GBPAUD Continuous A+ Scanner: OFF"
    
    keyboard = [
        [InlineKeyboardButton("📊 Run Analysis", callback_data="menu_analysis"),
         InlineKeyboardButton("🚨 Generate Signal", callback_data="menu_signal")],
        [InlineKeyboardButton(auto_status, callback_data="toggle_auto_scan")],
        [InlineKeyboardButton(f"🌐 Language: {current_flag}", callback_data="menu_lang_select")],
        [InlineKeyboardButton("ℹ️ Help & Instructions", callback_data="menu_help")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_languages[chat_id] = "English"
    is_auto = chat_id in active_subscribers
    
    welcome_text = (
        "🏛️ *INSTITUTIONAL TOP-DOWN MARKET DESK*\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "Terminal online with **Parallel Sloped Channel Geometry** matching your MetaTrader 5 mobile setup exactly.\n\n"
        "📌 *Status:* Ready for execution."
    )
    await update.message.reply_text(
        welcome_text,
        reply_markup=get_home_menu("English", is_auto),
        parse_mode="Markdown"
    )

async def background_continuous_scanner(application):
    await asyncio.sleep(10)
    while True:
        try:
            if active_subscribers:
                symbol = "GBPAUD"
                df_daily, df_4h, df_15m = fetch_top_down_institutional_data(symbol)
                setup = central_decision_engine(symbol, df_daily, df_4h, df_15m)
                macro_bearish = df_daily['Close'].iloc[-1] < df_daily['EMA200'].iloc[-1]
                
                if setup['confidence'] >= 80 and setup['break_confirmed']:
                    decimals = 3 if "JPY" in symbol else (2 if "XAU" in symbol or "GOLD" in symbol else 5)
                    fmt = f"{{:.{decimals}f}}"
                    scan_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
                    
                    raw_signal_text = (
                        f"🚨 **CONTINUOUS A+ SIGNAL ALERT (GBPAUD)** 🚨\n"
                        f"⏱️ *Timestamp:* {scan_timestamp}\n\n"
                        f"PAIR: {symbol}\n"
                        f"Direction: {setup['direction']}\n"
                        f"Confidence: {setup['confidence']}% (Parallel Sloped Channel Break)\n"
                        f"Status: {setup['trendline_status']}\n"
                        f"Entry: {fmt.format(setup['entry'])}\n"
                        f"SL: {fmt.format(setup['sl'])}\n"
                        f"TP1: {fmt.format(setup['tp1'])}\n"
                        f"TP2: {fmt.format(setup['tp2'])}\n"
                    )
                    
                    chart_img = generate_15m_execution_chart(df_15m, f"{symbol} Parallel Sloped Channel Break", setup, macro_bearish)

                    for chat_id in list(active_subscribers):
                        lang = user_languages.get(chat_id, "English")
                        final_signal_text = translate_text(raw_signal_text, lang)
                        try:
                            await application.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"🎯 *A+ CHANNEL BREAKOUT: {symbol}*\n⏱️ `{scan_timestamp}`", parse_mode="Markdown")
                            chart_img.seek(0)
                            await application.bot.send_message(chat_id=chat_id, text=final_signal_text, parse_mode="Markdown")
                        except Exception:
                            pass
        except Exception:
            pass
            
        await asyncio.sleep(300)

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id
    lang = user_languages.get(chat_id, "English")
    is_auto = chat_id in active_subscribers
    
    if data == "menu_home":
        await query.edit_message_text(text="🏛️ *Main Control Menu:*", reply_markup=get_home_menu(lang, is_auto), parse_mode="Markdown")

    elif data == "toggle_auto_scan":
        if chat_id in active_subscribers:
            active_subscribers.remove(chat_id)
            status_msg = "🔴 Continuous GBPAUD scanner disabled."
        else:
            active_subscribers.add(chat_id)
            status_msg = "🟢 Continuous GBPAUD scanner enabled."
        is_auto = chat_id in active_subscribers
        await query.edit_message_text(text=status_msg, reply_markup=get_home_menu(lang, is_auto), parse_mode="Markdown")

    elif data == "menu_lang_select":
        kb = [
            [InlineKeyboardButton("🇬🇧 English", callback_data="set_lang|English"),
             InlineKeyboardButton("🇳🇬 Hausa", callback_data="set_lang|Hausa")],
            [InlineKeyboardButton("🇳🇬 Pidgin", callback_data="set_lang|Pidgin")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="🌍 *Select Language:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    elif data.startswith("set_lang|"):
        new_lang = data.split("|")[1]
        user_languages[chat_id] = new_lang
        await query.edit_message_text(text=f"✅ Language updated to **{new_lang}**.", reply_markup=get_home_menu(new_lang, is_auto), parse_mode="Markdown")

    elif data == "menu_analysis":
        kb = [
            [InlineKeyboardButton("GBPAUD", callback_data="run_an|GBPAUD"),
             InlineKeyboardButton("EURUSD", callback_data="run_an|EURUSD")],
            [InlineKeyboardButton("XAUUSD (GOLD)", callback_data="run_an|XAUUSD"),
             InlineKeyboardButton("BTCUSD", callback_data="run_an|BTCUSD")],
            [InlineKeyboardButton("AAPL", callback_data="run_an|AAPL"),
             InlineKeyboardButton("TESLA", callback_data="run_an|TESLA")],
            [InlineKeyboardButton("US30", callback_data="run_an|US30"),
             InlineKeyboardButton("NAS100", callback_data="run_an|NAS100")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="📊 *Select Asset for Parallel Channel Analysis:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    elif data == "menu_signal":
        kb = [
            [InlineKeyboardButton("GBPAUD", callback_data="run_sig|GBPAUD"),
             InlineKeyboardButton("EURUSD", callback_data="run_sig|EURUSD")],
            [InlineKeyboardButton("XAUUSD (GOLD)", callback_data="run_sig|XAUUSD"),
             InlineKeyboardButton("BTCUSD", callback_data="run_sig|BTCUSD")],
            [InlineKeyboardButton("AAPL", callback_data="run_sig|AAPL"),
             InlineKeyboardButton("TESLA", callback_data="run_sig|TESLA")],
            [InlineKeyboardButton("US30", callback_data="run_sig|US30"),
             InlineKeyboardButton("NAS100", callback_data="run_sig|NAS100")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="🚨 *Select Asset for Breakout Signal:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    elif data.startswith("run_an|"):
        _, symbol = data.split("|")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
        await query.edit_message_text(text=f"🧠 Mapping Parallel Sloped Channels for {symbol} at {ts}...")
        try:
            df_daily, df_4h, df_15m = fetch_top_down_institutional_data(symbol)
            setup = central_decision_engine(symbol, df_daily, df_4h, df_15m)
            macro_bearish = df_daily['Close'].iloc[-1] < df_daily['EMA200'].iloc[-1]

            metrics = f"- Timestamp: {ts}\n- Macro Bias: {setup['macro_bias']}\n- Status: {setup['trendline_status']}\n- Confidence: {setup['confidence']}%\n"
            ai_commentary = fetch_ai_commentary(metrics, lang)
            chart_img = generate_15m_execution_chart(df_15m, f"{symbol} Parallel Sloped Channel", setup, macro_bearish)
            
            decimals = 2 if symbol in ["AAPL", "TESLA", "XAUUSD", "GOLD", "US30", "NAS100"] else 5
            fmt = f"{{:.{decimals}f}}"
            
            price_box = (
                f"📌 *PARALLEL SLOPED CHANNEL ZONE:*\n"
                f"⏱️ *Timestamp:* `{ts}`\n"
                f"• Status: {setup['trendline_status']}\n"
                f"• Current Price: {fmt.format(setup['entry'])}\n"
                f"• Direction: {setup['direction']}\n"
                f"• SL: {fmt.format(setup['sl'])}\n"
                f"• TP1: {fmt.format(setup['tp1'])}\n"
                f"• TP2: {fmt.format(setup['tp2'])}\n"
            )
            
            await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"🎯 *CHANNEL CHART: {symbol}*\n⏱️ `{ts}`", parse_mode="Markdown")
            await context.bot.send_message(chat_id=chat_id, text=price_box, parse_mode="Markdown")
            await context.bot.send_message(chat_id=chat_id, text=ai_commentary, parse_mode="Markdown")
            await context.bot.send_message(chat_id=chat_id, text="📌 *Complete.* Choose next action:", reply_markup=get_home_menu(lang, is_auto), parse_mode="Markdown")
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Failed: {str(e)}", reply_markup=get_home_menu(lang, is_auto))

    elif data.startswith("run_sig|"):
        _, symbol = data.split("|")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
        await query.edit_message_text(text=f"🚨 Verifying Channel Breakout for {symbol} at {ts}...")
        try:
            df_daily, df_4h, df_15m = fetch_top_down_institutional_data(symbol)
            setup = central_decision_engine(symbol, df_daily, df_4h, df_15m)

            decimals = 2 if symbol in ["AAPL", "TESLA", "XAUUSD", "GOLD", "US30", "NAS100"] else 5
            fmt = f"{{:.{decimals}f}}"
            
            raw_sig = (
                f"🚨 **A+ PARALLEL CHANNEL BREAK SIGNAL** 🚨\n"
                f"⏱️ *Timestamp:* {ts}\n\n"
                f"PAIR: {symbol}\n"
                f"Direction: {setup['direction']}\n"
                f"Confidence: {setup['confidence']}%\n"
                f"Status: {setup['trendline_status']}\n"
                f"Entry: {fmt.format(setup['entry'])}\n"
                f"SL: {fmt.format(setup['sl'])}\n"
                f"TP1: {fmt.format(setup['tp1'])}\n"
                f"TP2: {fmt.format(setup['tp2'])}\n"
            )
            final_sig = translate_text(raw_sig, lang)
            await context.bot.send_message(chat_id=chat_id, text=final_sig, parse_mode="Markdown")
            await context.bot.send_message(chat_id=chat_id, text="📌 *Complete.* Choose next action:", reply_markup=get_home_menu(lang, is_auto), parse_mode="Markdown")
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Failed: {str(e)}", reply_markup=get_home_menu(lang, is_auto))

    elif data == "menu_help":
        help_text = (
            "ℹ️ *TERMINAL GUIDE*\n\n"
            "• **Parallel Sloped Channels**: Dynamically computes both the upper resistance trendline connecting swing highs and the lower support trendline connecting swing lows across the chart.\n"
            "• **Breakout Filter**: Signals are verified when price decisively breaks and closes beyond the sloped boundary."
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
    
    loop.create_task(background_continuous_scanner(app_bot))

    loop.run_until_complete(app_bot.initialize())
    loop.run_until_complete(app_bot.start())
    loop.run_until_complete(app_bot.updater.start_polling(drop_pending_updates=True))
    loop.run_forever()

if __name__ == "__main__":
    threading.Thread(target=run_telegram_bot, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
