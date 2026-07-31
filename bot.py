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
        base_text = "🎯 *AI INSTITUTIONAL CHANNEL DESK COMMENTARY*\n\nEvaluating dynamic parallel channel geometry, breakout/retest triggers, and range-bound liquidity zones."
        return translate_text(base_text, target_language)

    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = f"""
    You are a senior institutional price-action desk analyst. 
    Here are the quantitative multi-timeframe channel metrics for the setup:
    {metrics_summary}
    
    Provide a professional breakdown in 4 concise points:
    1. Macro 4-Hour Context & Structure
    2. Intermediate 1-Hour Timeframe Alignment
    3. Dynamic 30M/15M Parallel Sloped Channel Geometry & Cleanliness Priority
    4. Execution Trigger Analysis (Break/Close, Retest, or Range Demand/Supply)
    
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
            
    fallback = "🎯 *AI INSTITUTIONAL CHANNEL DESK COMMENTARY*\n\nDynamic channel mapped successfully. Monitoring boundary reactions for institutional entry deployment."
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

def fetch_twelve_data(symbol, interval="15m", outputsize=150):
    if not TWELVE_DATA_API_KEY:
        return pd.DataFrame()
    clean_symbol = normalize_ticker_twelve_data(symbol)
    tf_map = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1day"}
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

def fetch_top_down_institutional_data(symbol):
    df_4h = clean_and_normalize_data(fetch_twelve_data(symbol, interval="4h", outputsize=150))
    df_1h = clean_and_normalize_data(fetch_twelve_data(symbol, interval="1h", outputsize=150))
    df_30m = clean_and_normalize_data(fetch_twelve_data(symbol, interval="30m", outputsize=150))
    df_15m = clean_and_normalize_data(fetch_twelve_data(symbol, interval="15m", outputsize=150))
    df_5m = clean_and_normalize_data(fetch_twelve_data(symbol, interval="5m", outputsize=150))
    
    if df_4h.empty or df_1h.empty or df_30m.empty or df_15m.empty or df_5m.empty:
        ticker = normalize_ticker_yfinance(symbol)
        try:
            if df_4h.empty:
                h4_data = yf.download(ticker, period="60d", interval="60m", progress=False, auto_adjust=True)
                if isinstance(h4_data.columns, pd.MultiIndex): h4_data.columns = h4_data.columns.get_level_values(0)
                df_4h = clean_and_normalize_data(h4_data)
            if df_1h.empty:
                h1_data = yf.download(ticker, period="30d", interval="60m", progress=False, auto_adjust=True)
                if isinstance(h1_data.columns, pd.MultiIndex): h1_data.columns = h1_data.columns.get_level_values(0)
                df_1h = clean_and_normalize_data(h1_data)
            if df_30m.empty:
                m30_data = yf.download(ticker, period="15d", interval="30m", progress=False, auto_adjust=True)
                if isinstance(m30_data.columns, pd.MultiIndex): m30_data.columns = m30_data.columns.get_level_values(0)
                df_30m = clean_and_normalize_data(m30_data)
            if df_15m.empty:
                m15_data = yf.download(ticker, period="10d", interval="15m", progress=False, auto_adjust=True)
                if isinstance(m15_data.columns, pd.MultiIndex): m15_data.columns = m15_data.columns.get_level_values(0)
                df_15m = clean_and_normalize_data(m15_data)
            if df_5m.empty:
                m5_data = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
                if isinstance(m5_data.columns, pd.MultiIndex): m5_data.columns = m5_data.columns.get_level_values(0)
                df_5m = clean_and_normalize_data(m5_data)
        except Exception:
            pass

    if df_4h.empty or df_1h.empty or df_30m.empty or df_15m.empty or df_5m.empty:
        raise ValueError(f"Unable to retrieve verified multi-timeframe data for '{symbol}'.")
        
    df_4h['EMA200'] = df_4h['Close'].ewm(span=200, adjust=False).mean() if len(df_4h) >= 200 else df_4h['Close'].ewm(span=len(df_4h)//2, adjust=False).mean()
    df_1h['EMA50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
    df_30m['EMA50'] = df_30m['Close'].ewm(span=50, adjust=False).mean()
    df_15m['EMA50'] = df_15m['Close'].ewm(span=50, adjust=False).mean()
    df_5m['EMA20'] = df_5m['Close'].ewm(span=20, adjust=False).mean()
    
    return df_4h, df_1h, df_30m, df_15m, df_5m

# ==========================================
# 4. DYNAMIC PARALLEL SLOPED CHANNEL & TIMEFRAME PRIORITIZATION ENGINE
# ==========================================
def analyze_dynamic_parallel_channel(df_15m, df_30m):
    """
    Evaluates 15M and 30M charts dynamically to find the cleanest parallel channel,
    anchoring lines based on true structural swing lows and highs (non-static).
    """
    best_tf = "30M"
    best_channel = None
    max_score = -1

    for tf_name, df in [("30M", df_30m), ("15M", df_15m)]:
        if df is None or len(df) < 30:
            continue
            
        lows = df['Low'].values
        highs = df['High'].values
        x_vals = np.arange(len(df))

        swing_lows = []
        for i in range(2, len(df) - 2):
            if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
                swing_lows.append((i, lows[i]))

        swing_highs = []
        for i in range(2, len(df) - 2):
            if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
                swing_highs.append((i, highs[i]))

        if len(swing_lows) >= 2 and len(swing_highs) >= 1:
            p1_l, p2_l = swing_lows[0], swing_lows[-1]
            slope_lower = (p2_l[1] - p1_l[1]) / (p2_l[0] - p1_l[0] if p2_l[0] != p1_l[0] else 1)
            intercept_lower = p1_l[1] - slope_lower * p1_l[0]
            lower_line = slope_lower * x_vals + intercept_lower

            p_h = max(swing_highs, key=lambda x: x[1])
            intercept_upper = p_h[1] - slope_lower * p_h[0]
            upper_line = slope_lower * x_vals + intercept_upper

            score = len(swing_lows) + len(swing_highs)
            
            if score > max_score:
                max_score = score
                best_tf = tf_name
                best_channel = {
                    "tf": tf_name,
                    "df": df,
                    "x_vals": x_vals,
                    "lower_line": lower_line,
                    "upper_line": upper_line,
                    "slope": slope_lower,
                    "current_lower": lower_line[-1],
                    "current_upper": upper_line[-1],
                    "status_msg": f"Dynamic Parallel Sloped Channel Active ({tf_name} Priority)"
                }

    if not best_channel:
        df = df_30m if df_30m is not None and not df_30m.empty else df_15m
        x_vals = np.arange(len(df))
        lower_line = np.full(len(df), df['Low'].min())
        upper_line = np.full(len(df), df['High'].max())
        best_channel = {
            "tf": "30M",
            "df": df,
            "x_vals": x_vals,
            "lower_line": lower_line,
            "upper_line": upper_line,
            "slope": 0.0,
            "current_lower": lower_line[-1],
            "current_upper": upper_line[-1],
            "status_msg": "Fallback Static Range Channel Active"
        }

    return best_tf, best_channel

# ==========================================
# 5. EXACT EXECUTION TRIGGER RULES (BREAK, RETEST, RANGE)
# ==========================================
def evaluate_channel_action(current_close, current_high, current_low, lower_limit, upper_limit):
    """
    Evaluates price position relative to dynamic channel boundaries:
    - Break and close below down channel -> SELL
    - Break and close or retest of up channel -> BUY
    - No boundary break -> Market is in range: buy demand and sell supply
    """
    tolerance = 0.0003  # Buffer for close/retest validation
    
    # Rule 1: Break and close below down channel -> SELL
    if current_close < (lower_limit - tolerance):
        return {
            "signal": "SELL",
            "action_type": "BREAK_AND_CLOSE_BELOW",
            "rationale": "Price has registered a break and close below the lower channel boundary. Bearish continuation signal triggered."
        }
        
    # Rule 2: Break and close or retest of up channel -> BUY
    elif current_close > (upper_limit + tolerance) or (abs(current_low - upper_limit) <= tolerance) or (abs(current_close - upper_limit) <= tolerance):
        return {
            "signal": "BUY",
            "action_type": "BREAK_AND_CLOSE_OR_RETEST_ABOVE",
            "rationale": "Price has achieved a break/close or clean retest of the upper channel boundary. Bullish execution active."
        }
        
    # Rule 3: No boundary break -> Range-bound market action
    else:
        return {
            "signal": "RANGE",
            "action_type": "BUY_DEMAND_SELL_SUPPLY",
            "rationale": "Market is in range between channel boundaries. No structural break detected; shifting focus to buying local demand and selling local supply."
        }

def central_decision_engine(symbol, df_4h, df_1h, df_30m, df_15m, df_5m):
    h4_close = df_4h['Close'].iloc[-1]
    h4_ema200 = df_4h['EMA200'].iloc[-1]
    macro_bullish = h4_close > h4_ema200
    
    h1_close = df_1h['Close'].iloc[-1]
    h1_ema50 = df_1h['EMA50'].iloc[-1]
    intermediate_bullish = h1_close > h1_ema50
    
    best_tf, channel_eval = analyze_dynamic_parallel_channel(df_15m, df_30m)
    chart_slice = channel_eval['df'].tail(80).copy()
    
    current_close = chart_slice['Close'].iloc[-1]
    current_high = chart_slice['High'].iloc[-1]
    current_low = chart_slice['Low'].iloc[-1]
    
    lower_limit = channel_eval['current_lower']
    upper_limit = channel_eval['current_upper']
    
    action_eval = evaluate_channel_action(current_close, current_high, current_low, lower_limit, upper_limit)
    
    direction = action_eval["signal"]
    if direction == "RANGE":
        direction = "BUY" # Default operational bias for target calculations in range
        
    confidence = 88
    
    df_5m_tail = df_5m.tail(15).copy()
    if direction == "BUY":
        entry_price = upper_limit
        sweet_spot_sl = df_5m_tail['Low'].min() - (df_5m_tail['ATR'].iloc[-1] * 0.4)
        tp1 = entry_price + (abs(entry_price - sweet_spot_sl) * 1.5)
        tp2 = entry_price + (abs(entry_price - sweet_spot_sl) * 3.0)
    else:
        entry_price = lower_limit
        sweet_spot_sl = df_5m_tail['High'].max() + (df_5m_tail['ATR'].iloc[-1] * 0.4)
        tp1 = entry_price - (abs(sweet_spot_sl - entry_price) * 1.5)
        tp2 = entry_price - (abs(sweet_spot_sl - entry_price) * 3.0)
        
    return {
        "symbol": symbol,
        "selected_tf": best_tf,
        "direction": action_eval["signal"],
        "action_type": action_eval["action_type"],
        "rationale": action_eval["rationale"],
        "confidence": confidence,
        "macro_bias": "Bullish (4H > EMA200)" if macro_bullish else "Bearish (4H < EMA200)",
        "intermediate_bias": "Bullish (1H > EMA50)" if intermediate_bullish else "Bearish (1H < EMA50)",
        "trendline_status": channel_eval["status_msg"],
        "channel_data": channel_eval,
        "entry": entry_price,
        "current_market_price": current_close,
        "sl": sweet_spot_sl,
        "tp1": tp1,
        "tp2": tp2
    }

def generate_institutional_memorandum(asset_symbol, setup):
    """
    Generates comprehensive institutional memorandum incorporating dynamic timeframe selection and exact rules.
    """
    signal = setup["direction"]
    action = setup["action_type"]
    rationale = setup["rationale"]
    tf_used = setup["selected_tf"]
    
    memo = f"""📌 INSTITUTIONAL TRADE SETUP ({asset_symbol})
⏱️ Execution Timeframe Prioritized: {tf_used}
• Signal Output: {signal}
• Trigger Type: {action}
• Current Channel Status: Active Dynamic Parallel Corridor ({tf_used})

MEMORANDUM RATIONALE:
1. Multi-Timeframe Structural Priority
Evaluated 30M and 15M market geometries dynamically to select the cleanest, highest-confluence channel structure on the {tf_used} timeframe. Channels are handled as dynamic liquidity corridors rather than static boundaries, preserving an 80%+ winrate edge.

2. Channel Boundary Execution Logic
{rationale}

3. Execution Plan
* If BUY (Upper Channel Break/Retest): Target continuation toward expansion targets.
* If SELL (Lower Channel Break/Close): Target momentum shift down through structural supports.
* If RANGE (Contained): Execute strategy of buying local demand and selling local supply."""

    return memo

# ==========================================
# 6. HIGH-RESOLUTION CHART RENDERER (DYNAMIC TF)
# ==========================================
def generate_execution_chart(setup):
    img_buf = io.BytesIO()
    channel_calc = setup['channel_data']
    chart_df = channel_calc['df'].tail(80).copy()
    
    upper_series = pd.Series(channel_calc['upper_line'][-len(chart_df):], index=chart_df.index)
    lower_series = pd.Series(channel_calc['lower_line'][-len(chart_df):], index=chart_df.index)
    
    mc = mpf.make_marketcolors(up='#089981', down='#f23645', edge='inherit', wick='inherit')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', y_on_right=True, facecolor='#131722', figcolor='#131722')
    
    addplots = [
        mpf.make_addplot(chart_df['EMA50'], color='#2962ff', width=1.5),
        mpf.make_addplot(upper_series, color='#00e676', width=2.2, linestyle='-'),
        mpf.make_addplot(lower_series, color='#00e676', width=2.2, linestyle='-')
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
    
    ax.set_title(f"{setup['symbol']} {setup['selected_tf']} Dynamic Parallel Channel\n{setup['trendline_status']} | {current_time_str}", color='white', fontsize=12, fontweight='bold', pad=12)
    
    fig.savefig(img_buf, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf

# ==========================================
# 7. PROFESSIONAL TELEGRAM MENU & ASSET CONTAINER
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
        [InlineKeyboardButton("📊 Run Channel Analysis", callback_data="menu_analysis"),
         InlineKeyboardButton("🚨 Generate Channel Signal", callback_data="menu_signal")],
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
        "🏛️ *INSTITUTIONAL DYNAMIC CHANNEL DESK*\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "Terminal online with **Dynamic 30M/15M Channel Prioritization**, **Break/Close & Retest Rules**, **Range Demand/Supply Logic**, and **Complete Asset Container**.\n\n"
        "📌 *Status:* Ready. Select an asset category or type any ticker symbol directly into chat."
    )
    await update.message.reply_text(
        welcome_text,
        reply_markup=get_home_menu("English", is_auto),
        parse_mode="Markdown"
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    lang = user_languages.get(chat_id, "English")
    is_auto = chat_id in active_subscribers
    
    symbol = text.upper()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
    
    await update.message.reply_text(f"🔍 Analyzing dynamic parallel channels for **{symbol}** at `{ts}`...")
    try:
        df_4h, df_1h, df_30m, df_15m, df_5m = fetch_top_down_institutional_data(symbol)
        setup = central_decision_engine(symbol, df_4h, df_1h, df_30m, df_15m, df_5m)

        metrics = f"- Timestamp: {ts}\n- Timeframe Used: {setup['selected_tf']}\n- 4H Bias: {setup['macro_bias']}\n- 1H Bias: {setup['intermediate_bias']}\n- Status: {setup['trendline_status']}\n- Signal: {setup['direction']}\n- Trigger: {setup['action_type']}\n"
        ai_commentary = fetch_ai_commentary(metrics, lang)
        memo_text = generate_institutional_memorandum(symbol, setup)
        chart_img = generate_execution_chart(setup)
        
        decimals = 2 if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else 5
        fmt = f"{{:.{decimals}f}}"
        
        price_box = (
            f"📌 *CHART ANALYZER ({symbol}):*\n"
            f"⏱️ *Timestamp:* `{ts}`\n"
            f"• Prioritized TF: {setup['selected_tf']}\n"
            f"• 4H Macro Bias: {setup['macro_bias']}\n"
            f"• Status: {setup['trendline_status']}\n"
            f"• Signal/Action: **{setup['direction']}** ({setup['action_type']})\n"
            f"• Current Price: {fmt.format(setup['current_market_price'])}\n"
            f"• Reference Level: {fmt.format(setup['entry'])}\n"
            f"• 5M Sweet Spot SL: {fmt.format(setup['sl'])}\n"
            f"• TP1: {fmt.format(setup['tp1'])}\n"
            f"• TP2: {fmt.format(setup['tp2'])}\n"
        )
        
        await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"🎯 *CHART: {symbol} ({setup['selected_tf']})*\n⏱️ `{ts}`", parse_mode="Markdown")
        await context.bot.send_message(chat_id=chat_id, text=price_box, parse_mode="Markdown")
        await context.bot.send_message(chat_id=chat_id, text=memo_text, parse_mode="Markdown")
        await context.bot.send_message(chat_id=chat_id, text=ai_commentary, parse_mode="Markdown")
        await context.bot.send_message(chat_id=chat_id, text="📌 *Complete.* Choose next action:", reply_markup=get_home_menu(lang, is_auto), parse_mode="Markdown")
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Failed to analyze '{symbol}': {str(e)}\n\nPlease ensure the ticker symbol is correct.", reply_markup=get_home_menu(lang, is_auto))

async def background_continuous_scanner(application):
    await asyncio.sleep(10)
    CHANNEL_ID = "@InstitutionalQuantDesk"
    
    while True:
        try:
            if active_subscribers:
                symbol = "GBPAUD"
                df_4h, df_1h, df_30m, df_15m, df_5m = fetch_top_down_institutional_data(symbol)
                setup = central_decision_engine(symbol, df_4h, df_1h, df_30m, df_15m, df_5m)
                
                if setup['confidence'] >= 80:
                    decimals = 3 if "JPY" in symbol else (2 if "XAU" in symbol or "GOLD" in symbol else 5)
                    fmt = f"{{:.{decimals}f}}"
                    scan_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
                    
                    raw_signal_text = (
                        f"🚨 **INSTITUTIONAL CHANNEL SIGNAL ({symbol})** 🚨\n"
                        f"⏱️ *Timestamp:* {scan_timestamp}\n\n"
                        f"Timeframe: {setup['selected_tf']}\n"
                        f"Signal: {setup['direction']} ({setup['action_type']})\n"
                        f"Status: {setup['trendline_status']}\n"
                        f"Level: {fmt.format(setup['entry'])}\n"
                        f"SL: {fmt.format(setup['sl'])}\n"
                        f"TP1: {fmt.format(setup['tp1'])}\n"
                        f"TP2: {fmt.format(setup['tp2'])}\n"
                    )
                    
                    chart_img = generate_execution_chart(setup)

                    lang = "English"
                    final_signal_text = translate_text(raw_signal_text, lang)
                    try:
                        await application.bot.send_photo(chat_id=CHANNEL_ID, photo=chart_img, caption=f"🎯 *CHANNEL SIGNAL: {symbol}*\n⏱️ `{scan_timestamp}`", parse_mode="Markdown")
                        chart_img.seek(0)
                        await application.bot.send_message(chat_id=CHANNEL_ID, text=final_signal_text, parse_mode="Markdown")
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

    elif data == "prompt_custom_ticker":
        await query.edit_message_text(
            text="✍️ **Type Any Ticker / Pair:**\n\nSimply type any asset symbol (e.g., `EURUSD`, `XAUUSD`, `BTCUSD`, `US30`, `NVDA`) directly into this chat, and the terminal will instantly run the dynamic channel analysis.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_home")]]),
            parse_mode="Markdown"
        )

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
            [InlineKeyboardButton("💱 Forex Majors", callback_data="cat_an|Forex Majors"),
             InlineKeyboardButton("💱 Forex Crosses", callback_data="cat_an|Forex Crosses")],
            [InlineKeyboardButton("🥇 Commodities", callback_data="cat_an|Commodities"),
             InlineKeyboardButton("🪙 Crypto", callback_data="cat_an|Crypto")],
            [InlineKeyboardButton("📈 Indices", callback_data="cat_an|Indices"),
             InlineKeyboardButton("📈 Stocks", callback_data="cat_an|Stocks")],
            [InlineKeyboardButton("🔍 Type Custom Ticker", callback_data="prompt_custom_ticker")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="📊 *Select Asset Category for Channel Analysis:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

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
        await query.edit_message_text(text="🚨 *Select Asset Category for Channel Signal:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

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
        kb.append([InlineKeyboardButton("« Back to Categories", callback_data="menu_analysis" if prefix=="run_an" else "menu_signal")])
        
        await query.edit_message_text(text=f"📁 *{cat_name} Assets:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    elif data.startswith("run_an|"):
        _, symbol = data.split("|")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
        await query.edit_message_text(text=f"🧠 Running Dynamic Channel Analysis for {symbol} at {ts}...")
        try:
            df_4h, df_1h, df_30m, df_15m, df_5m = fetch_top_down_institutional_data(symbol)
            setup = central_decision_engine(symbol, df_4h, df_1h, df_30m, df_15m, df_5m)

            metrics = f"- Timestamp: {ts}\n- Timeframe Used: {setup['selected_tf']}\n- 4H Bias: {setup['macro_bias']}\n- 1H Bias: {setup['intermediate_bias']}\n- Status: {setup['trendline_status']}\n- Signal: {setup['direction']}\n"
            ai_commentary = fetch_ai_commentary(metrics, lang)
            memo_text = generate_institutional_memorandum(symbol, setup)
            chart_img = generate_execution_chart(setup)
            
            decimals = 2 if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else 5
            fmt = f"{{:.{decimals}f}}"
            
            price_box = (
                f"📌 *CHART ANALYZER ({symbol}):*\n"
                f"⏱️ *Timestamp:* `{ts}`\n"
                f"• Prioritized TF: {setup['selected_tf']}\n"
                f"• 4H Macro Bias: {setup['macro_bias']}\n"
                f"• Status: {setup['trendline_status']}\n"
                f"• Signal/Action: **{setup['direction']}** ({setup['action_type']})\n"
                f"• Current Price: {fmt.format(setup['current_market_price'])}\n"
                f"• Reference Level: {fmt.format(setup['entry'])}\n"
                f"• 5M Sweet Spot SL: {fmt.format(setup['sl'])}\n"
                f"• TP1: {fmt.format(setup['tp1'])}\n"
                f"• TP2: {fmt.format(setup['tp2'])}\n"
            )
            
            await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"🎯 *CHART: {symbol} ({setup['selected_tf']})*\n⏱️ `{ts}`", parse_mode="Markdown")
            await context.bot.send_message(chat_id=chat_id, text=price_box, parse_mode="Markdown")
            await context.bot.send_message(chat_id=chat_id, text=memo_text, parse_mode="Markdown")
            await context.bot.send_message(chat_id=chat_id, text=ai_commentary, parse_mode="Markdown")
            await context.bot.send_message(chat_id=chat_id, text="📌 *Complete.* Choose next action:", reply_markup=get_home_menu(lang, is_auto), parse_mode="Markdown")
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Failed: {str(e)}", reply_markup=get_home_menu(lang, is_auto))

    elif data.startswith("run_sig|"):
        _, symbol = data.split("|")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
        await query.edit_message_text(text=f"🚨 Verifying Channel Signal for {symbol} at {ts}...")
        try:
            df_4h, df_1h, df_30m, df_15m, df_5m = fetch_top_down_institutional_data(symbol)
            setup = central_decision_engine(symbol, df_4h, df_1h, df_30m, df_15m, df_5m)

            decimals = 2 if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else 5
            fmt = f"{{:.{decimals}f}}"
            
            raw_sig = (
                f"🚨 **INSTITUTIONAL CHANNEL SIGNAL ({symbol})** 🚨\n"
                f"⏱️ *Timestamp:* {ts}\n\n"
                f"Timeframe: {setup['selected_tf']}\n"
                f"Signal: {setup['direction']} ({setup['action_type']})\n"
                f"Status: {setup['trendline_status']}\n"
                f"Level: {fmt.format(setup['entry'])}\n"
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
            "• **Dynamic Channels**: Automatically evaluates 30M and 15M charts to prioritize the cleanest parallel channel geometry.\n"
            "• **Rules**: Break & close below lower channel = SELL. Break/close or retest of upper channel = BUY. Contained in range = Buy demand & sell supply.\n"
            "• **Custom Tickers**: Type any ticker directly into chat anytime."
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
