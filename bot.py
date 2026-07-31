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
    If the pattern is marked as unclear/no-signal, say so plainly instead of inventing a directional bias.
    
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
# Minimum normalized drift (as a fraction of average price, over the analysis window)
# required before we're willing to call something a "clear" ascending/descending channel.
TREND_CLARITY_THRESHOLD = 0.015
# Minimum swing separation (as a fraction of the resistance-support range) required
# before we call a shoulder pattern "clear" rather than noise.
HS_CLARITY_FRACTION = 0.03

def analyze_backend_line_geometry(df_30m):
    close_prices = df_30m['Close'].values
    high_prices = df_30m['High'].values
    low_prices = df_30m['Low'].values
    x_vals = np.arange(len(df_30m))
    span = len(df_30m)

    resistance_level = float(high_prices.max())
    support_level = float(low_prices.min())
    current_close = float(close_prices[-1])

    slope, intercept = np.polyfit(x_vals, close_prices, 1)
    middle_line = slope * x_vals + intercept

    # --- FIX: parallel channel anchoring bug ---
    # The old code anchored the upper/lower channel lines at a fixed x-offset
    # (x_vals[-30]) but took the price extreme over the *whole* last-30-bar window.
    # Since the extreme rarely occurs exactly 30 bars back, the line was anchored
    # at the wrong point and, combined with the trend slope, drifted away from
    # where price actually was -- which silently corrupted "position in channel"
    # and made support bounces get misread as mid-channel or resistance zones.
    # Fix: anchor each line at the actual index where that extreme occurred.
    recent_window = min(30, span)
    window_start = span - recent_window
    window_x = x_vals[window_start:]
    window_high = high_prices[window_start:]
    window_low = low_prices[window_start:]

    upper_idx = int(np.argmax(window_high))
    lower_idx = int(np.argmin(window_low))
    upper_intercept = window_high[upper_idx] - slope * window_x[upper_idx]
    lower_intercept = window_low[lower_idx] - slope * window_x[lower_idx]
    upper_line = slope * x_vals + upper_intercept
    lower_line = slope * x_vals + lower_intercept

    # --- FIX: slope wasn't normalized by price scale, so trend "clarity" and the
    # confidence bonus derived from it were wildly inconsistent across assets
    # (e.g. BTC's slope in raw price units dwarfs EURUSD's). Normalize as % drift.
    avg_price = float(np.mean(close_prices)) or 1.0
    normalized_slope = (slope * span) / avg_price

    if normalized_slope > TREND_CLARITY_THRESHOLD and current_close >= middle_line[-1]:
        pattern_name = "Ascending Channel / Bullish Continuation Structure"
        bias = "BUY"
    elif normalized_slope < -TREND_CLARITY_THRESHOLD and current_close <= middle_line[-1]:
        pattern_name = "Descending Channel / Bearish Impulsive Trend"
        bias = "SELL"
    else:
        mid_idx = span // 2
        left_shoulder = float(high_prices[:mid_idx].max())
        head_high = float(high_prices.max())
        right_shoulder = float(high_prices[mid_idx:].max())

        left_trough = float(low_prices[:mid_idx].min())
        head_low = float(low_prices.min())
        right_trough = float(low_prices[mid_idx:].min())

        swing_margin = (resistance_level - support_level) * HS_CLARITY_FRACTION

        if head_high > left_shoulder + swing_margin and head_high > right_shoulder + swing_margin and current_close < middle_line[-1]:
            pattern_name = "Normal Head and Shoulders (Resistance Rejection)"
            bias = "SELL"
        elif head_low < left_trough - swing_margin and head_low < right_trough - swing_margin and current_close > middle_line[-1]:
            # FIX: the old fallback labeled ANYTHING that wasn't a normal H&S as an
            # "Inverted Head and Shoulders" bullish setup, using the *highs* array,
            # without ever checking the lows for an actual inverted structure.
            # That's a fabricated pattern claim. Now it's checked against the lows.
            pattern_name = "Inverted Head and Shoulders (Support Accumulation)"
            bias = "BUY"
        else:
            # FIX (per user requirement): if none of the above genuinely apply,
            # say so instead of forcing a directional label.
            pattern_name = "No Clear Pattern (Range-Bound / Insufficient Structure)"
            bias = "NONE"

    atr = df_30m['ATR'].iloc[-1] if 'ATR' in df_30m.columns else (resistance_level - support_level) * 0.05
    if atr == 0: atr = 0.001

    channel_width = upper_line[-1] - lower_line[-1]
    position_in_channel = (current_close - lower_line[-1]) / channel_width if channel_width != 0 else 0.5
    position_in_channel = float(np.clip(position_in_channel, 0.0, 1.0))

    # --- FIX: momentum confirmation. The user wants confidence to reflect
    # buyer/seller dominance building as the move continues, not just static slope.
    recent_closes = close_prices[-6:]
    diffs = np.diff(recent_closes)
    if bias == "BUY":
        momentum_bonus = float(min(8.0, np.sum(diffs > 0) * 1.6))
    elif bias == "SELL":
        momentum_bonus = float(min(8.0, np.sum(diffs < 0) * 1.6))
    else:
        momentum_bonus = 0.0

    if bias == "NONE":
        # Deliberately capped low -- an unclear pattern should never score
        # in "high confidence" territory, regardless of how flat/noisy price is.
        confidence_score = float(np.clip(35.0 + (1.0 - abs(position_in_channel - 0.5)) * 10.0, 25.0, 50.0))
        invalidation_threshold = None
        is_valid = False
        status_msg = "NO CLEAR STRUCTURE (Range-Bound / Ambiguous)"
    else:
        base_confidence = 55.0
        trend_strength_bonus = min(20.0, abs(normalized_slope) * 500.0)
        position_bonus = (1.0 - abs(position_in_channel - 0.5)) * 12.0
        confidence_score = float(np.clip(
            base_confidence + trend_strength_bonus + position_bonus + momentum_bonus, 45.0, 96.0
        ))

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
        "calculated_bias": bias,
        "position_in_channel": position_in_channel
    }

def evaluate_geometry_signals(geom_eval, macro_bullish):
    relative_position = geom_eval.get('position_in_channel', 0.5)
    current_close = geom_eval['df']['Close'].iloc[-1]
    support_floor = geom_eval['support_level']

    # FIX: no clear pattern -> say so, don't force a side.
    if geom_eval['calculated_bias'] == "NONE":
        return {
            "signal": "NO_SIGNAL",
            "action_type": "NO_CLEAR_PATTERN",
            "rationale": "No clear structural pattern detected -- trend is too weak/noisy and no defined channel or head-and-shoulders structure is present. Waiting for clearer price action before issuing a signal."
        }

    # FIX: an invalidated pattern used to auto-flip to the opposite direction,
    # which fabricates directional confidence out of a broken setup. Now it
    # correctly reports "no reliable signal" instead.
    if not geom_eval['is_valid']:
        return {
            "signal": "NO_SIGNAL",
            "action_type": "PATTERN_INVALIDATED_NO_SIGNAL",
            "rationale": "Pattern is INVALID: price breached the structural invalidation threshold. No reliable directional signal until the structure reforms."
        }

    if relative_position <= 0.25 or current_close <= support_floor * 1.003:
        return {
            "signal": "BUY",
            "action_type": "LOWER_CHANNEL_SUPPORT_BOUNCE",
            "rationale": f"Price tested structural support floor ({support_floor:.2f}) and held without a breakdown close. Support bounce buy setup."
        }
    elif relative_position >= 0.75:
        return {
            "signal": "SELL",
            "action_type": "UPPER_CHANNEL_RESISTANCE_REJECTION",
            "rationale": f"Price is at the structural channel ceiling (relative position: {relative_position:.2f}). Resistance rejection sell setup."
        }
    elif geom_eval['calculated_bias'] == "BUY":
        return {
            "signal": "BUY",
            "action_type": "TREND_CONTINUATION",
            "rationale": "Pattern is VALID. Bullish structural geometry holding mid-channel."
        }
    else:
        return {
            "signal": "SELL",
            "action_type": "TREND_CONTINUATION",
            "rationale": "Pattern is VALID. Bearish structural geometry holding mid-channel."
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
    elif direction == "SELL":
        entry_price = current_close
        sweet_spot_sl = geom_eval['resistance_level'] + (atr_val * 0.5)
        tp1 = entry_price - (abs(sweet_spot_sl - entry_price) * 1.5)
        tp2 = entry_price - (abs(sweet_spot_sl - entry_price) * 3.0)
    else:
        # NO_SIGNAL: don't fabricate trade levels for a setup we're not calling
        entry_price = current_close
        sweet_spot_sl = None
        tp1 = None
        tp2 = None
        
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

def format_trade_block(setup, fmt):
    """Central place that renders the SL/TP block, or an honest 'no trade' line
    instead of numbers when direction is NO_SIGNAL."""
    if setup["direction"] == "NO_SIGNAL" or setup["sl"] is None:
        return f"- No trade: {setup['rationale']}\n"
    return (
        f"- Price: {fmt.format(setup['current_market_price'])} | Entry: {fmt.format(setup['entry'])}\n"
        f"- SL: {fmt.format(setup['sl'])} | TP1: {fmt.format(setup['tp1'])} | TP2: {fmt.format(setup['tp2'])}\n"
    )

def generate_institutional_memorandum(asset_symbol, setup):
    g = setup["geometry_data"]
    decimals = 2 if asset_symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"] else (3 if "JPY" in asset_symbol else 5)
    fmt = f"{{:.{decimals}f}}"
    invalidation_str = fmt.format(g['invalidation_threshold']) if g['invalidation_threshold'] is not None else "N/A"

    memo = (
        f"INSTITUTIONAL STRUCTURAL MEMORANDUM ({asset_symbol})\n"
        f"Pattern Type: {setup['pattern_name']}\n"
        f"Confidence Score: {setup['confidence']:.1f}% | Validity: {setup['trendline_status']}\n"
        f"Timeframe: 30M | Signal: {setup['direction']} [{setup['action_type']}]\n\n"
        f"KEY GEOMETRIC LEVELS:\n"
        f"- Upper Resistance / Ceiling: {fmt.format(g['resistance_level'])}\n"
        f"- Support / Floor: {fmt.format(g['support_level'])}\n"
        f"- Invalidation Threshold: {invalidation_str}\n\n"
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
    
    if setup['direction'] == "NO_SIGNAL":
        validity_tag = "NO SIGNAL"
        status_color = "#ffb300"
    else:
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
            f"- Validity Status: {'VALID 🟢' if setup['is_valid'] else ('NO SIGNAL 🟡' if setup['direction']=='NO_SIGNAL' else 'INVALID 🔴')}\n"
            f"- Macro Bias: {setup['macro_bias']}\n"
            f"- Signal: {setup['direction']} [{setup['action_type']}]\n"
            + format_trade_block(setup, fmt)
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
                
                if setup['direction'] != "NO_SIGNAL" and setup['confidence'] >= 75 and setup['is_valid']:
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
            
            validity_str = 'VALID 🟢' if setup['is_valid'] else ('NO SIGNAL 🟡' if setup['direction']=='NO_SIGNAL' else 'INVALID 🔴')
            g = setup['geometry_data']
            invalidation_str = fmt.format(g['invalidation_threshold']) if g['invalidation_threshold'] is not None else "N/A"
            scanner_result_box = (
                f"PATTERN SCANNER RESULT ({symbol}):\n"
                f"- Pattern: {setup['pattern_name']}\n"
                f"- Status: {validity_str}\n"
                f"- Confidence Score: {setup['confidence']:.1f}%\n"
                f"- Signal: {setup['direction']} [{setup['action_type']}]\n"
                + format_trade_block(setup, fmt)
                + f"- Invalidation: {invalidation_str}\n"
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
                f"- Validity Status: {'VALID 🟢' if setup['is_valid'] else ('NO SIGNAL 🟡' if setup['direction']=='NO_SIGNAL' else 'INVALID 🔴')}\n"
                f"- Macro Bias: {setup['macro_bias']}\n"
                f"- Signal: {setup['direction']} [{setup['action_type']}]\n"
                + format_trade_block(setup, fmt)
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

            if setup['direction'] == "NO_SIGNAL":
                raw_sig = (
                    f"INSTITUTIONAL GEOMETRY SIGNAL ({symbol})\n"
                    f"Pattern: {setup['pattern_name']}\n"
                    f"Status: NO SIGNAL 🟡\n"
                    f"Confidence: {setup['confidence']:.1f}%\n"
                    f"Timestamp: {ts}\n"
                    f"Timeframe: 30M\n\n"
                    f"No trade: {setup['rationale']}"
                )
            else:
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
            "   - The bot calculates linear regression slopes (normalized against price scale) and swing structures unique to each selected asset.\n"
            "   - Patterns dynamically shift between Ascending Channels, Descending Channels, Normal Head & Shoulders, and Inverted Head & Shoulders based on actual market data.\n"
            "   - If none of these are clearly present, the bot reports 'No Clear Pattern' instead of forcing a label.\n\n"
            "2. Mathematical Confidence Scoring & Boundary Filters:\n"
            "   - Confidence is computed dynamically from trend momentum, price position within the channel, and short-term directional momentum.\n"
            "   - Boundary override filters correct the label when price is testing the support floor or resistance ceiling, so a support bounce is never reported as a resistance rejection.\n"
            "   - If a pattern's invalidation threshold is breached, the bot reports 'no reliable signal' rather than auto-flipping to the opposite trade."
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
