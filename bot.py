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
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI

# ==========================================
# 1. FLASK WEB SERVER & KEEP-ALIVE LOOP
# ==========================================
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Dynamic Institutional Trend Filter & Dealing Range Engine Active 24/7!", 200

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
    Translate the following market analysis or signal text accurately into {target_language}. Maintain all formatting, numbers, abbreviations, and professional tone:
    
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
        base_text = "🎯 DYNAMIC DESK SUMMARY: Live market narrative and dynamic structure evaluated."
        return translate_text(base_text, target_language)

    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = f"""
    You are a professional price action desk analyst. 
    Here are the live dynamic market metrics:
    {metrics_summary}
    
    Provide a concise 3-line institutional summary focusing on the live structural narrative and dynamic shift.
    
    Write the response directly in {target_language}.
    """
    for model in models:
        try:
            response = ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            content = response.choices[0].message.content
            if content and len(content.strip()) > 20:
                return content
        except Exception:
            continue
            
    fallback = "🎯 DYNAMIC DESK SUMMARY: Live structural mapping updated based on recent price action."
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
        "ETH": "ETH/USD", "ETHUSD": "ETH/USD",
        "AAPL": "AAPL", "TSLA": "TSLA", "NVDA": "NVDA", "MSFT": "MSFT", "AMZN": "AMZN", "GOOGL": "GOOGL", "META": "META",
        "US30": "DJI", "NAS100": "IXIC", "SPX500": "SPX"
    }
    if symbol in mapping:
        return mapping[symbol]
    if len(symbol) == 6 and "/" not in symbol and symbol not in ["US30", "NAS100", "SPX500"]:
        return f"{symbol[:3]}/{symbol[3:]}"
    return symbol

def fetch_twelve_data(symbol, interval="30min", outputsize=150):
    if not TWELVE_DATA_API_KEY:
        return pd.DataFrame()
    clean_symbol = normalize_ticker_twelve_data(symbol)
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": clean_symbol, "interval": interval, "outputsize": outputsize, "apikey": TWELVE_DATA_API_KEY}
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
        "ETH": "ETH-USD", "ETHUSD": "ETH-USD",
        "AAPL": "AAPL", "TSLA": "TSLA", "NVDA": "NVDA", "MSFT": "MSFT", "AMZN": "AMZN", "GOOGL": "GOOGL", "META": "META",
        "US30": "^DJI", "NAS100": "^IXIC", "SPX500": "^GSPC"
    }
    if symbol in alias_map:
        return alias_map[symbol]
    if len(symbol) == 6 and not symbol.endswith("=X") and symbol not in ["US30", "NAS100", "SPX500"]:
        return f"{symbol}=X"
    return symbol

def clean_and_normalize_data(df):
    if df.empty or len(df) < 30:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.dropna(inplace=True)
    return df

def fetch_institutional_multi_tf_data(symbol):
    df_30m = clean_and_normalize_data(fetch_twelve_data(symbol, interval="30min", outputsize=150))
    df_4h = clean_and_normalize_data(fetch_twelve_data(symbol, interval="4h", outputsize=150))
    
    if df_30m.empty or df_4h.empty:
        ticker = normalize_ticker_yfinance(symbol)
        try:
            if df_30m.empty:
                m30_data = yf.download(ticker, period="30d", interval="30m", progress=False, auto_adjust=True)
                if isinstance(m30_data.columns, pd.MultiIndex): m30_data.columns = m30_data.columns.get_level_values(0)
                df_30m = clean_and_normalize_data(m30_data)
            if df_4h.empty:
                h4_data = yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=True)
                if isinstance(h4_data.columns, pd.MultiIndex): h4_data.columns = h4_data.columns.get_level_values(0)
                df_4h = clean_and_normalize_data(h4_data)
        except Exception:
            pass

    if df_30m.empty:
        raise ValueError(f"Unable to retrieve verified market data for '{symbol}'.")
        
    if not df_4h.empty and len(df_4h) >= 50:
        df_4h['EMA200'] = df_4h['Close'].ewm(span=min(200, len(df_4h)), adjust=False).mean()
        macro_ema = df_4h['EMA200'].iloc[-1]
        macro_close = df_4h['Close'].iloc[-1]
    else:
        df_30m['EMA200'] = df_30m['Close'].ewm(span=min(200, len(df_30m)), adjust=False).mean()
        macro_ema = df_30m['EMA200'].iloc[-1]
        macro_close = df_30m['Close'].iloc[-1]

    df_30m['EMA50'] = df_30m['Close'].ewm(span=50, adjust=False).mean()
    return df_30m, macro_close > macro_ema, macro_ema

# ==========================================
# 4. DYNAMIC DEALING RANGE & MULTI-LINE PATTERN ENGINE
# ==========================================
def analyze_m30_dealing_range_pattern(df_30m):
    lows = df_30m['Low'].values
    highs = df_30m['High'].values
    x_vals = np.arange(len(df_30m))

    swing_lows = []
    for i in range(2, len(df_30m) - 2):
        if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            swing_lows.append((i, lows[i]))

    swing_highs = []
    for i in range(2, len(df_30m) - 2):
        if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
            swing_highs.append((i, highs[i]))

    if len(swing_lows) >= 2:
        p1_l, p2_l = swing_lows[0], swing_lows[-1]
        slope_lower = (p2_l[1] - p1_l[1]) / (p2_l[0] - p1_l[0] if p2_l[0] != p1_l[0] else 1)
        intercept_lower = p1_l[1] - slope_lower * p1_l[0]
        lower_line = slope_lower * x_vals + intercept_lower

        if len(swing_highs) >= 2:
            p1_h, p2_h = swing_highs[0], swing_highs[-1]
            slope_upper = (p2_h[1] - p1_h[1]) / (p2_h[0] - p1_h[0] if p2_h[0] != p1_h[0] else 1)
            intercept_upper = p1_h[1] - slope_upper * p1_h[0]
            upper_line = slope_upper * x_vals + intercept_upper
            
            # Dynamic Pattern Classification
            channel_width = np.mean(upper_line - lower_line)
            if abs(slope_upper - slope_lower) < 0.0001:
                pattern_name = "Parallel Dealing Range Channel"
            elif slope_upper <= 0.00005 and slope_lower > 0:
                pattern_name = "Ascending Triangle Compression"
            elif channel_width > (df_30m['ATR'].iloc[-1] * 6):
                pattern_name = "Multi-Line Parallel Pitchfork Channel"
            else:
                pattern_name = "Dynamic Channel Structure"
        else:
            upper_line = lower_line + (df_30m['High'].max() - df_30m['Low'].min())
            pattern_name = "Dynamic Ascending Channel"
    else:
        min_l, max_h = df_30m['Low'].min(), df_30m['High'].max()
        lower_line = np.full(len(df_30m), min_l)
        upper_line = np.full(len(df_30m), max_h)
        pattern_name = "Dynamic Consolidation Range"

    # Multi-line sub-channel subdivisions for live structural storytelling
    middle_line = (lower_line + upper_line) / 2.0
    sub_upper = lower_line + (upper_line - lower_line) * 0.75
    sub_lower = lower_line + (upper_line - lower_line) * 0.25

    return {
        "df": df_30m,
        "x_vals": x_vals,
        "lower_line": lower_line,
        "sub_lower": sub_lower,
        "middle_line": middle_line,
        "sub_upper": sub_upper,
        "upper_line": upper_line,
        "current_lower": lower_line[-1],
        "current_middle": middle_line[-1],
        "current_upper": upper_line[-1],
        "pattern_name": pattern_name,
        "status_msg": f"Dynamic Structure: {pattern_name}"
    }

def evaluate_dealing_range_signals(channel_eval, macro_bullish):
    df = channel_eval['df']
    current_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    middle_limit = channel_eval['current_middle']
    lower_limit = channel_eval['current_lower']
    tolerance = 0.0003
    
    if current_close < (lower_limit - tolerance):
        return {
            "signal": "SELL",
            "action_type": "DYNAMIC_CHANNEL_BREAK_SELL",
            "rationale": "Live price broke and closed below dynamic channel lower boundary. Bearish momentum active."
        }
    elif prev_close <= middle_limit and current_close > middle_limit:
        if macro_bullish:
            return {
                "signal": "BUY",
                "action_type": "MIDDLE_PIVOT_CROSS_BUY",
                "rationale": "Healthy candle close crossing middle pivot in alignment with macro 200 EMA trend filter."
            }
        else:
            return {
                "signal": "BUY",
                "action_type": "SHORT_TERM_CORRECTION_BUY",
                "rationale": "Healthy candle close crossing middle pivot capturing short-term dynamic correction entry."
            }
    elif current_close > middle_limit:
        return {
            "signal": "BUY",
            "action_type": "DYNAMIC_CONTINUATION_BUY",
            "rationale": "Price operating dynamically above middle range pivot. Continuation active."
        }
    else:
        return {
            "signal": "SELL",
            "action_type": "LOWER_RANGE_REACTION_SELL",
            "rationale": "Price operating below middle range pivot within lower sub-channel zone."
        }

def central_decision_engine(symbol):
    df_30m, macro_bullish, macro_ema = fetch_institutional_multi_tf_data(symbol)
    channel_eval = analyze_m30_dealing_range_pattern(df_30m)
    action_eval = evaluate_dealing_range_signals(channel_eval, macro_bullish)
    
    direction = action_eval["signal"]
    confidence = 90
    current_close = df_30m['Close'].iloc[-1]
    tail_df = df_30m.tail(15).copy()
    
    if direction == "BUY":
        entry_price = current_close
        sweet_spot_sl = tail_df['Low'].min() - (tail_df['ATR'].iloc[-1] * 0.4)
        tp1 = entry_price + (abs(entry_price - sweet_spot_sl) * 1.5)
        tp2 = entry_price + (abs(entry_price - sweet_spot_sl) * 3.0)
    else:
        entry_price = current_close
        sweet_spot_sl = tail_df['High'].max() + (tail_df['ATR'].iloc[-1] * 0.4)
        tp1 = entry_price - (abs(sweet_spot_sl - entry_price) * 1.5)
        tp2 = entry_price - (abs(sweet_spot_sl - entry_price) * 3.0)
        
    return {
        "symbol": symbol,
        "selected_tf": "30M",
        "direction": action_eval["signal"],
        "action_type": action_eval["action_type"],
        "rationale": action_eval["rationale"],
        "confidence": confidence,
        "macro_bias": f"Bullish (Macro 200 EMA Filter Active at {macro_ema:.5f})" if macro_bullish else f"Bearish / Correction Mode (Macro 200 EMA at {macro_ema:.5f})",
        "trendline_status": channel_eval["status_msg"],
        "pattern_name": channel_eval["pattern_name"],
        "channel_data": channel_eval,
        "entry": entry_price,
        "current_market_price": current_close,
        "sl": sweet_spot_sl,
        "tp1": tp1,
        "tp2": tp2
    }

def generate_institutional_memorandum(asset_symbol, setup):
    c = setup["channel_data"]
    decimals = 2 if asset_symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else 5
    fmt = f"{{:.{decimals}f}}"
    
    memo = (
        f"DYNAMIC MAPPING MEMORANDUM ({asset_symbol})\n"
        f"Structure Identified: {setup['pattern_name']}\n"
        f"Timeframe: 30M | Signal: {setup['direction']} [{setup['action_type']}]\n\n"
        f"LIVE STRUCTURAL BOUNDARIES:\n"
        f"- Upper Channel Boundary: {fmt.format(c['current_upper'])}\n"
        f"- Middle Pivot Line: {fmt.format(c['current_middle'])}\n"
        f"- Lower Channel Boundary: {fmt.format(c['current_lower'])}\n\n"
        f"LIVE MARKET STORY & RATIONALE:\n"
        f"{setup['rationale']}"
    )
    return memo

# ==========================================
# 5. HIGH-RESOLUTION DYNAMIC CHART RENDERER
# ==========================================
def generate_execution_chart(setup):
    img_buf = io.BytesIO()
    channel_calc = setup['channel_data']
    chart_df = channel_calc['df'].tail(80).copy()
    
    upper_series = pd.Series(channel_calc['upper_line'][-len(chart_df):], index=chart_df.index)
    sub_upper_series = pd.Series(channel_calc['sub_upper'][-len(chart_df):], index=chart_df.index)
    middle_series = pd.Series(channel_calc['middle_line'][-len(chart_df):], index=chart_df.index)
    sub_lower_series = pd.Series(channel_calc['sub_lower'][-len(chart_df):], index=chart_df.index)
    lower_series = pd.Series(channel_calc['lower_line'][-len(chart_df):], index=chart_df.index)
    
    mc = mpf.make_marketcolors(up='#089981', down='#f23645', edge='inherit', wick='inherit')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', y_on_right=True, facecolor='#131722', figcolor='#131722')
    
    addplots = [
        mpf.make_addplot(chart_df['EMA50'], color='#2962ff', width=1.5),
        mpf.make_addplot(upper_series, color='#00e676', width=2.0, linestyle='-'),
        mpf.make_addplot(sub_upper_series, color='#00e676', width=1.0, linestyle='--'),
        mpf.make_addplot(middle_series, color='#ff9800', width=2.0, linestyle='--'),
        mpf.make_addplot(sub_lower_series, color='#00e676', width=1.0, linestyle='--'),
        mpf.make_addplot(lower_series, color='#00e676', width=2.0, linestyle='-')
    ]
    
    all_visible_values = pd.concat([
        chart_df['High'], chart_df['Low'], 
        pd.Series([setup['tp1'], setup['tp2'], setup['entry']])
    ])
    ymin = all_visible_values.min()
    ymax = all_visible_values.max()
    padding = (ymax - ymin) * 0.08
    if padding == 0:
        padding = 0.001
        
    fig, axlist = mpf.plot(
        chart_df, type='candle', style=style, volume=False,
        addplot=addplots, returnfig=True, figsize=(12, 7),
        ylim=(ymin - padding, ymax + padding)
    )
    
    ax = axlist[0]
    ax.axhline(setup['tp2'], color='#e53935', linestyle='--', linewidth=1.2)
    ax.axhline(setup['tp1'], color='#2962ff', linestyle='--', linewidth=1.2)
    ax.axhline(setup['entry'], color='#00e676', linestyle='--', linewidth=1.2)
    
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
    ax.set_title(f"{setup['symbol']} 30M Dynamic {setup['pattern_name']}\n{setup['trendline_status']} | {current_time_str}", color='white', fontsize=11, fontweight='bold', pad=12)
    
    fig.savefig(img_buf, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf

# ==========================================
# 6. PROFESSIONAL TELEGRAM MENU & ASSET CONTAINER
# ==========================================
user_languages = {}
active_subscribers = set()

ASSET_CONTAINER = {
    "Forex Majors": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"],
    "Forex Crosses": ["EURGBP", "EURJPY", "GBPJPY", "GBPAUD", "AUDJPY", "EURAUD"],
    "Commodities": ["XAUUSD", "XAGUSD", "OIL"],
    "Crypto": ["BTCUSD", "ETHUSD"],
    "Indices": ["US30", "NAS100", "SPX500"],
    "Stocks": ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META"]
}

def get_home_menu(lang="English", auto_active=False):
    lang_flags = {"English": "🇬🇧 English", "Hausa": "🇳🇬 Hausa", "Pidgin": "🇳🇬 Pidgin"}
    current_flag = lang_flags.get(lang, "🇬🇧 English")
    auto_status = "🟢 GBPAUD Continuous Scanner: ON" if auto_active else "🔴 GBPAUD Continuous Scanner: OFF"
    
    keyboard = [
        [InlineKeyboardButton("📊 Run Dynamic Mapping", callback_data="menu_analysis"),
         InlineKeyboardButton("🚨 Generate Live Signal", callback_data="menu_signal")],
        [InlineKeyboardButton("🔍 Type Custom Ticker / Pair", callback_data="prompt_custom_ticker")],
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
        "DYNAMIC INSTITUTIONAL MAPPING TERMINAL\n"
        "-----------------------------------\n"
        "Terminal active with live dynamic channel recognition, multi-line sub-channels, and adaptive market storytelling.\n\n"
        "Status: Ready. Select an asset category or type any ticker symbol directly into chat."
    )
    await update.message.reply_text(
        welcome_text,
        reply_markup=get_home_menu("English", is_auto)
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    lang = user_languages.get(chat_id, "English")
    is_auto = chat_id in active_subscribers
    
    symbol = text.upper()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
    
    await update.message.reply_text(f"Running analysis for {symbol}")
    try:
        setup = central_decision_engine(symbol)

        metrics = f"- Timestamp: {ts}\n- TF: 30M\n- Structure: {setup['pattern_name']}\n- Signal: {setup['direction']} ({setup['action_type']})\n"
        ai_commentary = fetch_ai_commentary(metrics, lang)
        memo_text = generate_institutional_memorandum(symbol, setup)
        chart_img = generate_execution_chart(setup)
        
        decimals = 2 if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else 5
        fmt = f"{{:.{decimals}f}}"
        
        summary_box = (
            f"DYNAMIC SUMMARY BOX ({symbol}):\n"
            f"- TF: 30M | Structure: {setup['pattern_name']}\n"
            f"- Macro Bias: {setup['macro_bias']}\n"
            f"- Signal: {setup['direction']} [{setup['action_type']}]\n"
            f"- Price: {fmt.format(setup['current_market_price'])} | Entry: {fmt.format(setup['entry'])}\n"
            f"- SL: {fmt.format(setup['sl'])} | TP1: {fmt.format(setup['tp1'])} | TP2: {fmt.format(setup['tp2'])}\n"
        )
        
        await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"CHART MAPPING: {symbol} (30M) | {ts}")
        await context.bot.send_message(chat_id=chat_id, text=summary_box)
        await context.bot.send_message(chat_id=chat_id, text=memo_text)
        await context.bot.send_message(chat_id=chat_id, text=ai_commentary)
        await context.bot.send_message(chat_id=chat_id, text="Complete. Choose next action:", reply_markup=get_home_menu(lang, is_auto))
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Failed to analyze '{symbol}': {str(e)}", reply_markup=get_home_menu(lang, is_auto))

async def background_continuous_scanner(application):
    await asyncio.sleep(10)
    CHANNEL_ID = "@InstitutionalQuantDesk"
    
    while True:
        try:
            if active_subscribers:
                symbol = "GBPAUD"
                setup = central_decision_engine(symbol)
                
                if setup['confidence'] >= 80:
                    decimals = 3 if "JPY" in symbol else (2 if "XAU" in symbol or "GOLD" in symbol else 5)
                    fmt = f"{{:.{decimals}f}}"
                    scan_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
                    
                    raw_signal_text = (
                        f"DYNAMIC MAPPING SIGNAL ({symbol})\n"
                        f"Structure: {setup['pattern_name']}\n"
                        f"Timestamp: {scan_timestamp}\n"
                        f"Timeframe: 30M\n"
                        f"Signal: {setup['direction']} ({setup['action_type']})\n"
                        f"Entry: {fmt.format(setup['entry'])}\n"
                        f"SL: {fmt.format(setup['sl'])}\n"
                        f"TP1: {fmt.format(setup['tp1'])}\n"
                        f"TP2: {fmt.format(setup['tp2'])}\n"
                    )
                    
                    chart_img = generate_execution_chart(setup)
                    lang = "English"
                    final_signal_text = translate_text(raw_signal_text, lang)
                    try:
                        await application.bot.send_photo(chat_id=CHANNEL_ID, photo=chart_img, caption=f"SIGNAL: {symbol} | {scan_timestamp}")
                        chart_img.seek(0)
                        await application.bot.send_message(chat_id=CHANNEL_ID, text=final_signal_text)
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
        await query.edit_message_text(text="Main Control Menu:", reply_markup=get_home_menu(lang, is_auto))

    elif data == "prompt_custom_ticker":
        await query.edit_message_text(
            text="Type Any Ticker / Pair:\n\nSend any symbol (e.g., EURUSD, XAUUSD, BTCUSD) in chat to run live mapping.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_home")]]))

    elif data == "toggle_auto_scan":
        if chat_id in active_subscribers:
            active_subscribers.remove(chat_id)
            status_msg = "Continuous GBPAUD scanner disabled."
        else:
            active_subscribers.add(chat_id)
            status_msg = "Continuous GBPAUD scanner enabled."
        is_auto = chat_id in active_subscribers
        await query.edit_message_text(text=status_msg, reply_markup=get_home_menu(lang, is_auto))

    elif data == "menu_lang_select":
        kb = [
            [InlineKeyboardButton("🇬🇧 English", callback_data="set_lang|English"),
             InlineKeyboardButton("🇳🇬 Hausa", callback_data="set_lang|Hausa")],
            [InlineKeyboardButton("🇳🇬 Pidgin", callback_data="set_lang|Pidgin")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="Select Language:", reply_markup=InlineKeyboardMarkup(kb))

    elif data.startswith("set_lang|"):
        new_lang = data.split("|")[1]
        user_languages[chat_id] = new_lang
        await query.edit_message_text(text=f"Language updated to {new_lang}.", reply_markup=get_home_menu(new_lang, is_auto))

    elif data == "menu_analysis":
        kb = [
            [InlineKeyboardButton("💱 Forex Majors", callback_data="cat_an|Forex Majors"),
             InlineKeyboardButton("💱 Forex Crosses", callback_data="cat_an|Forex Crosses")],
            [InlineKeyboardButton("🥇 Commodities", callback_data="cat_an|Commodities"),
             InlineKeyboardButton("🪙 Crypto", callback_data="cat_an|Crypto")],
            [InlineKeyboardButton("📈 Indices", callback_data="cat_an|Indices"),
             InlineKeyboardButton("📈 Stocks", callback_data="cat_an|Stocks")],
            [InlineKeyboardButton("🔍 Type Custom Ticker", callback_data="prompt_custom_ticker")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="Select Asset Category:", reply_markup=InlineKeyboardMarkup(kb))

    elif data == "menu_signal":
        kb = [
            [InlineKeyboardButton("💱 Forex Majors", callback_data="cat_sig|Forex Majors"),
             InlineKeyboardButton("💱 Forex Crosses", callback_data="cat_sig|Forex Crosses")],
            [InlineKeyboardButton("🥇 Commodities", callback_data="cat_sig|Commodities"),
             InlineKeyboardButton("🪙 Crypto", callback_data="cat_sig|Crypto")],
            [InlineKeyboardButton("📈 Indices", callback_data="cat_sig|Indices"),
             InlineKeyboardButton("📈 Stocks", callback_data="cat_sig|Stocks")],
            [InlineKeyboardButton("🔍 Type Custom Ticker", callback_data="prompt_custom_ticker")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="Select Asset Category for Signal:", reply_markup=InlineKeyboardMarkup(kb))

    elif data.startswith("cat_an|") or data.startswith("cat_sig|"):
        action_type, cat_name = data.split("|")
        pairs = ASSET_CONTAINER.get(cat_name, [])
        prefix = "run_an" if action_type == "cat_an" else "run_sig"
        
        kb = []
        row = []
        for pair in pairs:
            row.append(InlineKeyboardButton(pair, callback_data=f"{prefix}|{pair}"))
            if len(row) == 2:
                kb.append(row)
                row = []
        if row:
            kb.append(row)
        kb.append([InlineKeyboardButton("« Back", callback_data="menu_analysis" if prefix=="run_an" else "menu_signal")])
        
        await query.edit_message_text(text=f"{cat_name} Assets:", reply_markup=InlineKeyboardMarkup(kb))

    elif data.startswith("run_an|"):
        _, symbol = data.split("|")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
        await query.edit_message_text(text=f"Running analysis for {symbol}")
        try:
            setup = central_decision_engine(symbol)

            metrics = f"- Timestamp: {ts}\n- TF: 30M\n- Structure: {setup['pattern_name']}\n- Signal: {setup['direction']}\n"
            ai_commentary = fetch_ai_commentary(metrics, lang)
            memo_text = generate_institutional_memorandum(symbol, setup)
            chart_img = generate_execution_chart(setup)
            
            decimals = 2 if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else 5
            fmt = f"{{:.{decimals}f}}"
            
            summary_box = (
                f"DYNAMIC SUMMARY BOX ({symbol}):\n"
                f"- TF: 30M | Structure: {setup['pattern_name']}\n"
                f"- Macro Bias: {setup['macro_bias']}\n"
                f"- Signal: {setup['direction']} [{setup['action_type']}]\n"
                f"- Price: {fmt.format(setup['current_market_price'])} | Entry: {fmt.format(setup['entry'])}\n"
                f"- SL: {fmt.format(setup['sl'])} | TP1: {fmt.format(setup['tp1'])} | TP2: {fmt.format(setup['tp2'])}\n"
            )
            
            await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"CHART MAPPING: {symbol} (30M) | {ts}")
            await context.bot.send_message(chat_id=chat_id, text=summary_box)
            await context.bot.send_message(chat_id=chat_id, text=memo_text)
            await context.bot.send_message(chat_id=chat_id, text=ai_commentary)
            await context.bot.send_message(chat_id=chat_id, text="Complete. Choose next action:", reply_markup=get_home_menu(lang, is_auto))
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"Failed: {str(e)}", reply_markup=get_home_menu(lang, is_auto))

    elif data.startswith("run_sig|"):
        _, symbol = data.split("|")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
        await query.edit_message_text(text=f"Running analysis for {symbol}")
        try:
            setup = central_decision_engine(symbol)

            decimals = 2 if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else 5
            fmt = f"{{:.{decimals}f}}"
            
            raw_sig = (
                f"DYNAMIC MAPPING SIGNAL ({symbol})\n"
                f"Structure: {setup['pattern_name']}\n"
                f"Timestamp: {ts}\n"
                f"Timeframe: 30M\n"
                f"Signal: {setup['direction']} ({setup['action_type']})\n"
                f"Entry: {fmt.format(setup['entry'])}\n"
                f"SL: {fmt.format(setup['sl'])}\n"
                f"TP1: {fmt.format(setup['tp1'])}\n"
                f"TP2: {fmt.format(setup['tp2'])}\n"
            )
            final_sig = translate_text(raw_sig, lang)
            await context.bot.send_message(chat_id=chat_id, text=final_sig)
            await context.bot.send_message(chat_id=chat_id, text="Complete. Choose next action:", reply_markup=get_home_menu(lang, is_auto))
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"Failed: {str(e)}", reply_markup=get_home_menu(lang, is_auto))

    elif data == "menu_help":
        help_text = (
            "DYNAMIC MAPPING TERMINAL GUIDE:\n\n"
            "- Dynamic Structural Mapping: Automatically adapts channels and multi-line sub-divisions based on live swing behavior.\n"
            "- Non-Static Storytelling: Live narratives update dynamically with every scan.\n"
            "- 1. Mapped Chart Image: High-resolution visual output with all sub-channels.\n"
            "- 2. Detailed Institutional Analysis: Quantitative narrative outlining active boundaries and pivots."
        )
        kb = [[InlineKeyboardButton("« Back", callback_data="menu_home")]]
        await query.edit_message_text(text=help_text, reply_markup=InlineKeyboardMarkup(kb))

def run_telegram_bot():
    if not TELEGRAM_BOT_TOKEN:
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app_bot = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start_command))
    app_bot.add_handler(CallbackQueryHandler(button_callback_handler))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    loop.create_task(background_continuous_scanner(app_bot))

    loop.run_until_complete(app_bot.initialize())
    loop.run_until_complete(app_bot.start())
    loop.run_until_complete(app_bot.updater.start_polling(drop_pending_updates=True))
    loop.run_forever()

if __name__ == "__main__":
    threading.Thread(target=run_telegram_bot, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
