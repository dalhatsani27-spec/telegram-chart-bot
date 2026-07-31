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
    return "Institutional Structure & Geometry Analysis Engine Active 24/7!", 200

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
                max_tokens=600,
                timeout=8
            )
            content = response.choices[0].message.content
            if content and len(content.strip()) > 10:
                return content
        except Exception:
            continue
            
    return text

def fetch_ai_commentary(metrics_summary, target_language="English"):
    if not OPENROUTER_API_KEY:
        base_text = "🎯 INSTITUTIONAL DESK SUMMARY: Dynamic price geometry and multi-factor confidence evaluated."
        return translate_text(base_text, target_language)

    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = f"""
    You are a professional price action desk analyst. 
    Here are the live structural and geometric metrics derived from actual price data:
    {metrics_summary}
    
    Provide a concise 3-line institutional summary focusing on pattern recognition, validity status, and breakout validation.
    
    Write the response directly in {target_language}.
    """
    for model in models:
        try:
            response = ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                timeout=8
            )
            content = response.choices[0].message.content
            if content and len(content.strip()) > 20:
                return content
        except Exception:
            continue
            
    fallback = "🎯 INSTITUTIONAL DESK SUMMARY: Dynamic market geometry verified through calculated price structures."
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
# 4. TRUE DYNAMIC GEOMETRY & PATTERN ENGINE
# ==========================================
def analyze_backend_line_geometry(df_30m):
    close_prices = df_30m['Close'].values
    high_prices = df_30m['High'].values
    low_prices = df_30m['Low'].values
    x_vals = np.arange(len(df_30m))

    resistance_level = float(high_prices.max())
    support_level = float(low_prices.min())
    current_close = float(close_prices[-1])
    
    span = len(df_30m)
    slope, intercept = np.polyfit(x_vals, close_prices, 1)
    middle_line = slope * x_vals + intercept
    
    upper_intercept = high_prices.max() - slope * (np.argmax(high_prices))
    lower_intercept = low_prices.min() - slope * (np.argmin(low_prices))
    upper_line = slope * x_vals + upper_intercept
    lower_line = slope * x_vals + lower_intercept

    if slope > 0.0001 and current_close >= middle_line[-1]:
        pattern_name = "Ascending Channel / Bullish Continuation Structure"
        bias = "BUY"
    elif slope < -0.0001 and current_close <= middle_line[-1]:
        pattern_name = "Descending Channel / Bearish Impulsive Trend"
        bias = "SELL"
    else:
        mid_idx = span // 2
        left_shoulder = high_prices[:mid_idx].max()
        head = high_prices.max()
        right_shoulder = high_prices[mid_idx:].max()
        
        if head > left_shoulder and head > right_shoulder and current_close < middle_line[-1]:
            pattern_name = "Normal Head and Shoulders (Resistance Rejection)"
            bias = "SELL"
        else:
            pattern_name = "Inverted Head and Shoulders (Support Accumulation / Breaker Block)"
            bias = "BUY"

    atr = df_30m['ATR'].iloc[-1] if 'ATR' in df_30m.columns else (resistance_level - support_level) * 0.05
    if atr == 0: atr = 0.001
    
    channel_width = upper_line[-1] - lower_line[-1]
    position_in_channel = (current_close - lower_line[-1]) / channel_width if channel_width != 0 else 0.5
    position_in_channel = np.clip(position_in_channel, 0.0, 1.0)
    
    base_confidence = 60.0
    trend_strength_bonus = min(20.0, abs(slope) * 10000)
    position_bonus = (1.0 - abs(position_in_channel - 0.5)) * 14.5
    confidence_score = float(np.clip(base_confidence + trend_strength_bonus + position_bonus, 45.0, 94.5))

    if bias == "BUY":
        invalidation_threshold = support_level - (atr * 0.5)
        is_valid = current_close >= invalidation_threshold
    else:
        invalidation_threshold = resistance_level + (atr * 0.5)
        is_valid = current_close <= invalidation_threshold

    status_msg = "VALID Confirmed Structure" if is_valid else "INVALID (Structure Broken / Threshold Breached)"
    
    return {
        "df": df_30m,
        "x_vals": x_vals,
        "lower_line": lower_line,
        "middle_line": middle_line,
        "upper_line": upper_line,
        "resistance_level": resistance_level,
        "support_level": support_level,
        "current_lower": lower_line[-1],
        "current_middle": middle_line[-1],
        "current_upper": upper_line[-1],
        "pattern_name": pattern_name,
        "confidence_score": confidence_score,
        "status_msg": status_msg,
        "is_valid": is_valid,
        "invalidation_threshold": invalidation_threshold,
        "calculated_bias": bias
    }

def evaluate_geometry_signals(geom_eval, macro_bullish):
    bias = geom_eval['calculated_bias']
    current_close = geom_eval['df']['Close'].iloc[-1]
    lower_line_val = geom_eval['current_lower']
    upper_line_val = geom_eval['current_upper']
    channel_range = upper_line_val - lower_line_val
    
    if channel_range == 0:
        channel_range = 0.001
        
    relative_position = (current_close - lower_line_val) / channel_range
    
    if not geom_eval['is_valid']:
        return {
            "signal": "SELL" if bias == "BUY" else "BUY",
            "action_type": "PATTERN_INVALIDATED_REVERSAL",
            "rationale": "Pattern is INVALID. Price breached structural invalidation threshold."
        }
    
    # BOUNDARY REVERSAL OVERRIDES: Respect floor and ceiling extremes
    if relative_position <= 0.20:
        return {
            "signal": "BUY",
            "action_type": "LOWER_CHANNEL_SUPPORT_BOUNCE",
            "rationale": "Price is testing/sweeping the lower channel boundary floor. High probability mean-reversion or support reaction."
        }
    elif relative_position >= 0.80:
        return {
            "signal": "SELL",
            "action_type": "UPPER_CHANNEL_RESISTANCE_REJECTION",
            "rationale": "Price is testing the upper channel boundary ceiling. High probability resistance rejection."
        }
    elif bias == "BUY":
        return {
            "signal": "BUY",
            "action_type": "SUPPORT_BOUNCE_OR_CONTINUATION",
            "rationale": "Pattern is VALID. Bullish structural geometry and support reaction holding."
        }
    else:
        return {
            "signal": "SELL",
            "action_type": "RESISTANCE_REJECTION_OR_IMPULSE",
            "rationale": "Pattern is VALID. Bearish structural geometry and resistance pressure active."
        }

def central_decision_engine(symbol):
    df_30m, macro_bullish, macro_ema = fetch_institutional_multi_tf_data(symbol)
    geom_eval = analyze_backend_line_geometry(df_30m)
    action_eval = evaluate_geometry_signals(geom_eval, macro_bullish)
    
    direction = action_eval["signal"]
    confidence = geom_eval["confidence_score"]
    current_close = df_30m['Close'].iloc[-1]
    tail_df = df_30m.tail(15).copy()
    atr_val = tail_df['ATR'].iloc[-1] if 'ATR' in tail_df.columns else 0.001
    
    if direction == "BUY":
        entry_price = current_close
        sweet_spot_sl = geom_eval['support_level'] - (atr_val * 0.5)
        tp1 = entry_price + (abs(entry_price - sweet_spot_sl) * 1.5)
        tp2 = entry_price + (abs(entry_price - sweet_spot_sl) * 3.0)
    else:
        entry_price = current_close
        sweet_spot_sl = geom_eval['resistance_level'] + (atr_val * 0.5)
        tp1 = entry_price - (abs(sweet_spot_sl - entry_price) * 1.5)
        tp2 = entry_price - (abs(sweet_spot_sl - entry_price) * 3.0)
        
    return {
        "symbol": symbol,
        "selected_tf": "30M (Multi-TF 4H/30M)",
        "direction": action_eval["signal"],
        "action_type": action_eval["action_type"],
        "rationale": action_eval["rationale"],
        "confidence": confidence,
        "macro_bias": f"Bullish (Macro Filter Active at {macro_ema:.2f})" if macro_bullish else f"Bearish / Correction Mode (Macro Filter at {macro_ema:.2f})",
        "trendline_status": geom_eval["status_msg"],
        "is_valid": geom_eval["is_valid"],
        "pattern_name": geom_eval["pattern_name"],
        "geometry_data": geom_eval,
        "entry": entry_price,
        "current_market_price": current_close,
        "sl": sweet_spot_sl,
        "tp1": tp1,
        "tp2": tp2
    }

def generate_institutional_memorandum(asset_symbol, setup):
    g = setup["geometry_data"]
    decimals = 2 if asset_symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else (3 if "JPY" in asset_symbol else 5)
    fmt = f"{{:.{decimals}f}}"
    
    memo = (
        f"INSTITUTIONAL STRUCTURAL MEMORANDUM ({asset_symbol})\n"
        f"Pattern Type: {setup['pattern_name']}\n"
        f"Confidence Score: {setup['confidence']:.1f}% | Validity: {setup['trendline_status']}\n"
        f"Timeframe: 30M | Signal: {setup['direction']} [{setup['action_type']}]\n\n"
        f"KEY GEOMETRIC LEVELS:\n"
        f"- Upper Resistance / Ceiling: {fmt.format(g['resistance_level'])}\n"
        f"- Support / Floor: {fmt.format(g['support_level'])}\n"
        f"- Invalidation Threshold: {fmt.format(g['invalidation_threshold'])}\n\n"
        f"LIVE MARKET RATIONALE:\n"
        f"{setup['rationale']}"
    )
    return memo

# ==========================================
# 5. FRONT-END CANDLESTICK & BACKEND GEOMETRY CHART RENDERER
# ==========================================
def generate_execution_chart(setup):
    img_buf = io.BytesIO()
    geom_calc = setup['geometry_data']
    chart_df = geom_calc['df'].tail(80).copy()
    
    upper_series = pd.Series(geom_calc['upper_line'][-len(chart_df):], index=chart_df.index)
    middle_series = pd.Series(geom_calc['middle_line'][-len(chart_df):], index=chart_df.index)
    lower_series = pd.Series(geom_calc['lower_line'][-len(chart_df):], index=chart_df.index)
    
    mc = mpf.make_marketcolors(up='#089981', down='#f23645', edge='inherit', wick='inherit')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', y_on_right=True, facecolor='#131722', figcolor='#131722')
    
    addplots = [
        mpf.make_addplot(chart_df['EMA50'], color='#2962ff', width=1.5),
        mpf.make_addplot(upper_series, color='#00e676', width=2.0, linestyle='-'),
        mpf.make_addplot(middle_series, color='#ff9800', width=1.5, linestyle='--'),
        mpf.make_addplot(lower_series, color='#00e676', width=2.0, linestyle='-')
    ]
    
    price_min = chart_df['Low'].min()
    price_max = chart_df['High'].max()
    padding = (price_max - price_min) * 0.15
    if padding == 0: padding = 0.001
        
    ymin = price_min - padding
    ymax = price_max + padding
        
    fig, axlist = mpf.plot(
        chart_df, type='candle', style=style, volume=False,
        addplot=addplots, returnfig=True, figsize=(12, 7),
        ylim=(ymin, ymax)
    )
    
    ax = axlist[0]
    ax.axhline(geom_calc['resistance_level'], color='#ffeb3b', linestyle='-', linewidth=1.5, label='Structural Resistance')
    ax.axhline(geom_calc['support_level'], color='#ffeb3b', linestyle='-', linewidth=1.5, label='Structural Support')
    ax.axhline(setup['entry'], color='#00e676', linestyle='--', linewidth=1.2, label='Entry')
    
    validity_tag = "VALID" if setup['is_valid'] else "INVALID"
    status_color = "#00e676" if setup['is_valid'] else "#f23645"
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
    
    ax.set_title(f"{setup['symbol']} Institutional Geometry Map [{validity_tag}]\nPattern: {setup['pattern_name']} | Confidence: {setup['confidence']:.1f}% | {current_time_str}", color=status_color, fontsize=9, fontweight='bold', pad=12)
    
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
        [InlineKeyboardButton("📊 Run Institutional Analysis", callback_data="menu_analysis"),
         InlineKeyboardButton("🚨 Generate Live Signal", callback_data="menu_signal")],
        [InlineKeyboardButton("🔍 Pattern Scanner (Choose Pair)", callback_data="menu_pattern_scanner")],
        [InlineKeyboardButton("🔍 Type Custom Ticker / Pair", callback_data="prompt_custom_ticker")],
        [InlineKeyboardButton(auto_status, callback_data="toggle_auto_scan")],
        [InlineKeyboardButton(f"🌐 Language: {current_flag}", callback_data="menu_lang_select")],
        [InlineKeyboardButton("ℹ️ Help & Validation Rules", callback_data="menu_help")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_languages[chat_id] = "English"
    is_auto = chat_id in active_subscribers
    
    welcome_text = (
        "INSTITUTIONAL TRADING CO-PILOT TERMINAL\n"
        "---------------------------------------\n"
        "Frontend: Candlesticks | Backend: Dynamic Independent Geometry\n\n"
        "Status: Ready. Select an asset category, use the Pattern Scanner to check specific pairs, or type any ticker."
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
    
    status_msg = await update.message.reply_text(f"Running dynamic multi-timeframe geometric analysis for {symbol}...")
    try:
        setup = central_decision_engine(symbol)

        metrics = f"- Timestamp: {ts}\n- Pattern: {setup['pattern_name']}\n- Confidence: {setup['confidence']:.1f}%\n- Validity: {setup['trendline_status']}\n- Signal: {setup['direction']} ({setup['action_type']})\n"
        ai_commentary = fetch_ai_commentary(metrics, lang)
        memo_text = generate_institutional_memorandum(symbol, setup)
        chart_img = generate_execution_chart(setup)
        
        decimals = 2 if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else (3 if "JPY" in symbol else 5)
        fmt = f"{{:.{decimals}f}}"
        
        summary_box = (
            f"SUMMARY BOX ({symbol}):\n"
            f"- Pattern: {setup['pattern_name']}\n"
            f"- Confidence Score: {setup['confidence']:.1f}%\n"
            f"- Validity Status: {'VALID 🟢' if setup['is_valid'] else 'INVALID 🔴'}\n"
            f"- Macro Bias: {setup['macro_bias']}\n"
            f"- Signal: {setup['direction']} [{setup['action_type']}]\n"
            f"- Price: {fmt.format(setup['current_market_price'])} | Entry: {fmt.format(setup['entry'])}\n"
            f"- SL: {fmt.format(setup['sl'])} | TP1: {fmt.format(setup['tp1'])} | TP2: {fmt.format(setup['tp2'])}\n"
        )
        
        await status_msg.delete()
        await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"CHART MAPPING: {symbol} (30M) | {ts}")
        await context.bot.send_message(chat_id=chat_id, text=summary_box)
        await context.bot.send_message(chat_id=chat_id, text=memo_text)
        await context.bot.send_message(chat_id=chat_id, text=ai_commentary)
        await context.bot.send_message(chat_id=chat_id, text="Complete. Choose next action:", reply_markup=get_home_menu(lang, is_auto))
    except Exception as e:
        try:
            await status_msg.delete()
        except Exception:
            pass
        await context.bot.send_message(chat_id=chat_id, text=f"Failed to analyze '{symbol}': {str(e)}", reply_markup=get_home_menu(lang, is_auto))

async def background_continuous_scanner(application):
    await asyncio.sleep(10)
    CHANNEL_ID = "@InstitutionalQuantDesk"
    
    while True:
        try:
            if active_subscribers:
                symbol = "GBPAUD"
                setup = central_decision_engine(symbol)
                
                if setup['confidence'] >= 75 and setup['is_valid']:
                    decimals = 3 if "JPY" in symbol else (2 if "XAU" in symbol or "GOLD" in symbol else 5)
                    fmt = f"{{:.{decimals}f}}"
                    scan_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
                    
                    raw_signal_text = (
                        f"INSTITUTIONAL GEOMETRY SIGNAL ({symbol})\n"
                        f"Pattern: {setup['pattern_name']}\n"
                        f"Status: VALID Confirmed Structure\n"
                        f"Confidence: {setup['confidence']:.1f}%\n"
                        f"Timestamp: {scan_timestamp}\n"
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

    elif data == "menu_pattern_scanner":
        kb = [
            [InlineKeyboardButton("💱 Forex Majors", callback_data="scan_cat|Forex Majors"),
             InlineKeyboardButton("💱 Forex Crosses", callback_data="scan_cat|Forex Crosses")],
            [InlineKeyboardButton("🥇 Commodities", callback_data="scan_cat|Commodities"),
             InlineKeyboardButton("🪙 Crypto", callback_data="scan_cat|Crypto")],
            [InlineKeyboardButton("📈 Indices", callback_data="scan_cat|Indices"),
             InlineKeyboardButton("📈 Stocks", callback_data="scan_cat|Stocks")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="🔍 Pattern Scanner:\nSelect an asset category to choose a specific pair and run independent validation:", reply_markup=InlineKeyboardMarkup(kb))

    elif data.startswith("scan_cat|"):
        _, cat_name = data.split("|")
        pairs = ASSET_CONTAINER.get(cat_name, [])
        
        kb = []
        row = []
        for pair in pairs:
            row.append(InlineKeyboardButton(pair, callback_data=f"scan_run|{pair}"))
            if len(row) == 2:
                kb.append(row)
                row = []
        if row:
            kb.append(row)
        kb.append([InlineKeyboardButton("« Back", callback_data="menu_pattern_scanner")])
        
        await query.edit_message_text(text=f"Select specific pair in {cat_name} to scan independent pattern geometry:", reply_markup=InlineKeyboardMarkup(kb))

    elif data.startswith("scan_run|"):
        _, symbol = data.split("|")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
        await query.edit_message_text(text=f"Scanning dynamic pattern for {symbol}...")
        try:
            setup = central_decision_engine(symbol)
            chart_img = generate_execution_chart(setup)
            
            decimals = 2 if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else (3 if "JPY" in symbol else 5)
            fmt = f"{{:.{decimals}f}}"
            
            validity_str = "VALID 🟢" if setup['is_valid'] else "INVALID 🔴"
            scanner_result_box = (
                f"PATTERN SCANNER RESULT ({symbol}):\n"
                f"- Pattern: {setup['pattern_name']}\n"
                f"- Status: {validity_str}\n"
                f"- Confidence Score: {setup['confidence']:.1f}%\n"
                f"- Signal: {setup['direction']} [{setup['action_type']}]\n"
                f"- Price: {fmt.format(setup['current_market_price'])} | Entry: {fmt.format(setup['entry'])}\n"
                f"- SL: {fmt.format(setup['sl'])} | Invalidation: {fmt.format(setup['geometry_data']['invalidation_threshold'])}\n"
            )
            
            await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"SCANNER MAPPING: {symbol} | {ts}")
            await context.bot.send_message(chat_id=chat_id, text=scanner_result_box)
            await context.bot.send_message(chat_id=chat_id, text="Scan complete. Choose next action:", reply_markup=get_home_menu(lang, is_auto))
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"Failed to scan '{symbol}': {str(e)}", reply_markup=get_home_menu(lang, is_auto))

    elif data == "prompt_custom_ticker":
        await query.edit_message_text(
            text="Type Any Ticker / Pair:\n\nSend any symbol (e.g., AUDUSD, XAUUSD, BTCUSD) in chat to run calculated geometry and pattern validation.",
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
        await query.edit_message_text(text=f"Running institutional analysis for {symbol}...")
        try:
            setup = central_decision_engine(symbol)

            metrics = f"- Timestamp: {ts}\n- Pattern: {setup['pattern_name']}\n- Confidence: {setup['confidence']:.1f}%\n- Validity: {setup['trendline_status']}\n- Signal: {setup['direction']}\n"
            ai_commentary = fetch_ai_commentary(metrics, lang)
            memo_text = generate_institutional_memorandum(symbol, setup)
            chart_img = generate_execution_chart(setup)
            
            decimals = 2 if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else (3 if "JPY" in symbol else 5)
            fmt = f"{{:.{decimals}f}}"
            
            summary_box = (
                f"SUMMARY BOX ({symbol}):\n"
                f"- Pattern: {setup['pattern_name']}\n"
                f"- Confidence Score: {setup['confidence']:.1f}%\n"
                f"- Validity Status: {'VALID 🟢' if setup['is_valid'] else 'INVALID 🔴'}\n"
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
        await query.edit_message_text(text=f"Running analysis for {symbol}...")
        try:
            setup = central_decision_engine(symbol)

            decimals = 2 if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else (3 if "JPY" in symbol else 5)
            fmt = f"{{:.{decimals}f}}"
            
            raw_sig = (
                f"INSTITUTIONAL GEOMETRY SIGNAL ({symbol})\n"
                f"Pattern: {setup['pattern_name']}\n"
                f"Status: {'VALID 🟢' if setup['is_valid'] else 'INVALID 🔴'}\n"
                f"Confidence: {setup['confidence']:.1f}%\n"
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
            "HOW DYNAMIC PATTERN RECOGNITION WORKS:\n\n"
            "1. Independent Slope & Structural Classification:\n"
            "   - The bot calculates linear regression slopes and swing structures unique to each selected asset.\n"
            "   - Patterns dynamically shift between Ascending Channels, Descending Channels, Normal Head & Shoulders, and Inverted Head & Shoulders based on actual market data.\n\n"
            "2. Mathematical Confidence Scoring & Boundary Filters:\n"
            "   - Confidence is computed dynamically from trend momentum and price position within channels.\n"
            "   - Boundary override filters prevent incorrect continuation signals when price interacts directly with structural support floors or resistance ceilings."
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
