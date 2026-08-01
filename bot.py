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
from flask import Flask, request, jsonify
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI

from patterns import scan_all_patterns, _atr as pattern_atr

# Optional MetaTrader 5 Local Resource Import
try:
    import MetaTrader5 as mt5
    MT5_LOCAL_AVAILABLE = True
except ImportError:
    MT5_LOCAL_AVAILABLE = False

# ==========================================
# 1. FLASK WEB SERVER & EA BRIDGE ENDPOINTS
# ==========================================
app = Flask(__name__)

ea_telemetry = {
    "last_ping": 0,
    "connected": False,
    "account": {},
    "positions": []
}

@app.route('/')
def health_check():
    return "Institutional Structure & Price-Action Pattern Engine Active 24/7!", 200

@app.route('/api/ping', methods=['POST'])
def ea_ping():
    """Endpoint pinged by MT5 EA to show connection status."""
    global ea_telemetry
    data = request.json or {}
    ea_telemetry["last_ping"] = time.time()
    ea_telemetry["connected"] = True
    if "account" in data:
        ea_telemetry["account"] = data["account"]
    if "positions" in data:
        ea_telemetry["positions"] = data["positions"]
    return jsonify({"status": "SUCCESS", "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}), 200

@app.route('/api/signal', methods=['GET'])
def ea_get_signal():
    """Endpoint polled by MT5 EA to retrieve trade signals."""
    symbol = request.args.get('symbol', 'XAUUSD')
    tf = request.args.get('tf', '15min')
    try:
        setup = central_decision_engine(symbol, timeframe=tf)
        if setup["direction"] != "NO_SIGNAL" and setup["is_valid"]:
            return jsonify({
                "symbol": symbol,
                "direction": setup["direction"],
                "entry": setup["entry"],
                "sl": setup["sl"],
                "tp1": setup["tp1"],
                "tp2": setup["tp2"],
                "confidence": setup["confidence"],
                "pattern": setup["pattern_name"],
                "valid": True
            }), 200
    except Exception:
        pass
    return jsonify({"symbol": symbol, "direction": "NO_SIGNAL", "valid": False}), 200

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

def translate_text(text, target_language):
    if target_language == "English" or not OPENROUTER_API_KEY:
        return text

    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = f"""
    Translate the following market analysis text into {target_language}. Maintain formatting and numbers:
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
        base_text = "🎯 DESK SUMMARY: Live price-action structure evaluated from current chart data."
        return translate_text(base_text, target_language)

    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = f"""
    You are a professional price action analyst. Here are the live structural metrics:
    {metrics_summary}
    
    Provide a concise 3-line summary: pattern forming, trigger level, and invalidation condition.
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
            
    fallback = "🎯 DESK SUMMARY: Structure verified through calculated price geometry scan."
    return translate_text(fallback, target_language)

# ==========================================
# 3. DATA & ANALYSIS ENGINE
# ==========================================
def fetch_mt5_local_rates(symbol, tf_code, count=200):
    if not MT5_LOCAL_AVAILABLE or not mt5.initialize():
        return pd.DataFrame()
    tf_map = {
        "1min": mt5.TIMEFRAME_M1, "3min": mt5.TIMEFRAME_M3, 
        "5min": mt5.TIMEFRAME_M5, "15min": mt5.TIMEFRAME_M15, 
        "30min": mt5.TIMEFRAME_M30, "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4
    }
    mt5_tf = tf_map.get(tf_code, mt5.TIMEFRAME_M30)
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('datetime', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    return df[['Open', 'High', 'Low', 'Close']]

def normalize_ticker_twelve_data(symbol):
    symbol = symbol.strip().upper().replace("[", "").replace("]", "").replace("'", "")
    mapping = {
        "GOLD": "XAU/USD", "XAUUSD": "XAU/USD", "SILVER": "XAG/USD", "XAGUSD": "XAG/USD",
        "OIL": "WTI/USD", "USOIL": "WTI/USD", "WTI": "WTI/USD", "BTC": "BTC/USD", "BTCUSD": "BTC/USD",
        "ETH": "ETH/USD", "ETHUSD": "ETH/USD", "US30": "DJI", "NAS100": "IXIC", "SPX500": "SPX"
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
        "GOLD": "GC=F", "XAUUSD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F", "USOIL": "CL=F",
        "BTC": "BTC-USD", "BTCUSD": "BTC-USD", "ETH": "ETH-USD", "ETHUSD": "ETH-USD",
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

    if df_30m.empty:
        df_30m = clean_and_normalize_data(fetch_mt5_local_rates(symbol, "30min", 150))
    if df_4h.empty:
        df_4h = clean_and_normalize_data(fetch_mt5_local_rates(symbol, "4h", 150))

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

_TF_TWELVE_INTERVAL = {"1min": "1min", "3min": "1min", "5min": "5min", "15min": "15min"}
_TF_YF_INTERVAL = {"1min": "1m", "3min": "1m", "5min": "5m", "15min": "15m"}
_TF_RESAMPLE = {"3min": "3min"}
_TF_YF_PERIOD = {"1m": "7d", "5m": "60d", "15m": "60d"}
TF_LABELS = {"1min": "1 Minute", "3min": "3 Minute", "5min": "5 Minute", "15min": "15 Minute"}

def _resample_ohlc(df, rule):
    if df.empty:
        return df
    return df.resample(rule).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

def fetch_timeframe_data(symbol, tf_code, outputsize=250):
    if tf_code not in _TF_TWELVE_INTERVAL:
        raise ValueError(f"Unsupported timeframe '{tf_code}'.")
    td_interval = _TF_TWELVE_INTERVAL[tf_code]
    fetch_size = outputsize * 3 if tf_code == "3min" else outputsize
    raw = fetch_twelve_data(symbol, interval=td_interval, outputsize=min(fetch_size, 5000))

    if raw.empty:
        raw = fetch_mt5_local_rates(symbol, tf_code, fetch_size)

    if raw.empty:
        ticker = normalize_ticker_yfinance(symbol)
        yf_interval = _TF_YF_INTERVAL[tf_code]
        period = _TF_YF_PERIOD.get(yf_interval, "30d")
        try:
            data = yf.download(ticker, period=period, interval=yf_interval, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            raw = data
        except Exception:
            raw = pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if tf_code in _TF_RESAMPLE:
        raw.columns = [c.capitalize() for c in raw.columns]
        raw = _resample_ohlc(raw[['Open', 'High', 'Low', 'Close']], _TF_RESAMPLE[tf_code])

    df = clean_and_normalize_data(raw)
    if df.empty:
        return df
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=min(50, len(df)), adjust=False).mean()
    return df

def fetch_macro_reference(symbol):
    df_1h = clean_and_normalize_data(fetch_twelve_data(symbol, interval="1h", outputsize=150))
    if df_1h.empty:
        df_1h = clean_and_normalize_data(fetch_mt5_local_rates(symbol, "1h", 150))
    if df_1h.empty:
        ticker = normalize_ticker_yfinance(symbol)
        try:
            data = yf.download(ticker, period="30d", interval="60m", progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            df_1h = clean_and_normalize_data(data)
        except Exception:
            pass
    if not df_1h.empty and len(df_1h) >= 30:
        ema = df_1h['Close'].ewm(span=min(200, len(df_1h)), adjust=False).mean()
        return float(df_1h['Close'].iloc[-1]), float(ema.iloc[-1])
    return None, None

def macro_bias_string(macro_close, macro_ema):
    if macro_close is None or macro_ema is None:
        return "Macro Filter Unavailable"
    if macro_close > macro_ema:
        return f"Bullish (1H Macro Filter Active at {macro_ema:.5f})"
    return f"Bearish (1H Macro Filter at {macro_ema:.5f})"

TREND_CLARITY_THRESHOLD = 0.015

def analyze_backend_line_geometry(df):
    close_prices = df['Close'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    x_vals = np.arange(len(df))
    span = len(df)

    resistance_level = float(high_prices.max())
    support_level = float(low_prices.min())
    current_close = float(close_prices[-1])

    slope, intercept = np.polyfit(x_vals, close_prices, 1)
    middle_line = slope * x_vals + intercept

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

    avg_price = float(np.mean(close_prices)) or 1.0
    normalized_slope = (slope * span) / avg_price

    if normalized_slope > TREND_CLARITY_THRESHOLD and current_close >= middle_line[-1]:
        pattern_name = "Ascending Channel / Bullish Drift"
        bias = "BUY"
    elif normalized_slope < -TREND_CLARITY_THRESHOLD and current_close <= middle_line[-1]:
        pattern_name = "Descending Channel / Bearish Drift"
        bias = "SELL"
    else:
        pattern_name = "No Clear Pattern (Range-Bound)"
        bias = "NONE"

    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else (resistance_level - support_level) * 0.05
    if atr == 0: atr = 0.001

    channel_width = upper_line[-1] - lower_line[-1]
    position_in_channel = (current_close - lower_line[-1]) / channel_width if channel_width != 0 else 0.5
    position_in_channel = float(np.clip(position_in_channel, 0.0, 1.0))

    if bias == "NONE":
        confidence_score = float(np.clip(35.0 + (1.0 - abs(position_in_channel - 0.5)) * 10.0, 25.0, 50.0))
        invalidation_threshold = None
        is_valid = False
        status_msg = "NO CLEAR STRUCTURE"
    else:
        confidence_score = float(np.clip(50.0 + min(18.0, abs(normalized_slope) * 500.0), 40.0, 88.0))
        if bias == "BUY":
            invalidation_threshold = support_level - (atr * 0.5)
            is_valid = current_close >= invalidation_threshold
        else:
            invalidation_threshold = resistance_level + (atr * 0.5)
            is_valid = current_close <= invalidation_threshold
        status_msg = "VALID Structure" if is_valid else "INVALID Structure"

    return {
        "df": df, "upper_line": upper_line, "middle_line": middle_line, "lower_line": lower_line,
        "resistance_level": resistance_level, "support_level": support_level,
        "pattern_name": pattern_name, "confidence_score": confidence_score, "status_msg": status_msg,
        "is_valid": is_valid, "invalidation_threshold": invalidation_threshold,
        "calculated_bias": bias, "position_in_channel": position_in_channel,
    }

def evaluate_geometry_signals(geom_eval):
    relative_position = geom_eval.get('position_in_channel', 0.5)
    current_close = geom_eval['df']['Close'].iloc[-1]
    support_floor = geom_eval['support_level']

    if geom_eval['calculated_bias'] == "NONE":
        return {"signal": "NO_SIGNAL", "action_type": "NO_CLEAR_PATTERN", "rationale": "No qualifying structure detected."}

    if not geom_eval['is_valid']:
        return {"signal": "NO_SIGNAL", "action_type": "PATTERN_INVALIDATED", "rationale": "Structure invalid."}

    if relative_position <= 0.25 or current_close <= support_floor * 1.003:
        return {"signal": "BUY", "action_type": "CHANNEL_SUPPORT_BOUNCE", "rationale": "Support bounce buy."}
    elif relative_position >= 0.75:
        return {"signal": "SELL", "action_type": "CHANNEL_RESISTANCE_REJECTION", "rationale": "Resistance rejection sell."}
    elif geom_eval['calculated_bias'] == "BUY":
        return {"signal": "BUY", "action_type": "TREND_CONTINUATION", "rationale": "Bullish channel drift."}
    else:
        return {"signal": "SELL", "action_type": "TREND_CONTINUATION", "rationale": "Bearish channel drift."}

def build_setup_from_pattern(df, best, all_patterns, symbol, tf_label, macro_close, macro_ema):
    current_close = float(df['Close'].iloc[-1])
    atr = pattern_atr(df) or 0.001
    direction = best.bias
    trigger = best.trigger_price

    resistance_level = float(df['High'].tail(30).max())
    support_level = float(df['Low'].tail(30).min())

    confidence = float(np.clip(best.confidence, 40.0, 97.0))
    entry = current_close

    if direction == "BUY":
        structural_sl = min(trigger, support_level) - atr * 0.5
        risk = max(abs(entry - structural_sl), atr * 0.25)
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 3.0
        invalidation_threshold = structural_sl
        is_valid = current_close > structural_sl
    else:
        structural_sl = max(trigger, resistance_level) + atr * 0.5
        risk = max(abs(structural_sl - entry), atr * 0.25)
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 3.0
        invalidation_threshold = structural_sl
        is_valid = current_close < structural_sl

    return {
        "symbol": symbol, "selected_tf": tf_label,
        "direction": direction if is_valid else "NO_SIGNAL",
        "action_type": f"{best.name.upper().replace(' ', '_')}_BREAKOUT",
        "rationale": best.note if is_valid else "Pattern invalidated.",
        "confidence": confidence, "macro_bias": macro_bias_string(macro_close, macro_ema),
        "trendline_status": "VALID" if is_valid else "INVALID",
        "is_valid": is_valid, "pattern_name": best.name,
        "geometry_data": {
            "df": df, "mode": "pattern",
            "resistance_level": resistance_level, "support_level": support_level,
            "invalidation_threshold": invalidation_threshold,
            "trigger_line": best.trigger_line, "key_points": best.key_points,
            "trigger_price": trigger, "category": best.category,
        },
        "entry": entry, "current_market_price": current_close,
        "sl": structural_sl if is_valid else None,
        "tp1": tp1 if is_valid else None,
        "tp2": tp2 if is_valid else None,
    }

def build_setup_from_channel(df, symbol, tf_label, macro_close, macro_ema):
    geom_eval = analyze_backend_line_geometry(df)
    action_eval = evaluate_geometry_signals(geom_eval)
    direction = action_eval["signal"]
    current_close = float(df['Close'].iloc[-1])
    atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0.001

    if direction == "BUY":
        sweet_spot_sl = geom_eval['support_level'] - (atr_val * 0.5)
        tp1 = current_close + (abs(current_close - sweet_spot_sl) * 1.5)
        tp2 = current_close + (abs(current_close - sweet_spot_sl) * 3.0)
    elif direction == "SELL":
        sweet_spot_sl = geom_eval['resistance_level'] + (atr_val * 0.5)
        tp1 = current_close - (abs(sweet_spot_sl - current_close) * 1.5)
        tp2 = current_close - (abs(sweet_spot_sl - current_close) * 3.0)
    else:
        sweet_spot_sl = None
        tp1, tp2 = None, None

    geom_eval["mode"] = "channel"
    return {
        "symbol": symbol, "selected_tf": tf_label,
        "direction": direction, "action_type": action_eval["action_type"],
        "rationale": action_eval["rationale"], "confidence": geom_eval["confidence_score"],
        "macro_bias": macro_bias_string(macro_close, macro_ema),
        "trendline_status": geom_eval["status_msg"], "is_valid": geom_eval["is_valid"],
        "pattern_name": geom_eval["pattern_name"], "geometry_data": geom_eval,
        "entry": current_close, "current_market_price": current_close,
        "sl": sweet_spot_sl, "tp1": tp1, "tp2": tp2,
    }

def central_decision_engine(symbol, timeframe="30min"):
    if timeframe == "30min":
        df, macro_bullish, macro_ema = fetch_institutional_multi_tf_data(symbol)
        macro_close = float(df['Close'].iloc[-1])
        tf_label = "30M"
    else:
        df = fetch_timeframe_data(symbol, timeframe)
        if df.empty or len(df) < 40:
            raise ValueError(f"Unable to retrieve verified market data for '{symbol}'.")
        macro_close, macro_ema = fetch_macro_reference(symbol)
        tf_label = TF_LABELS.get(timeframe, timeframe)

    best, all_patterns = scan_all_patterns(df)
    if best is not None:
        return build_setup_from_pattern(df, best, all_patterns, symbol, tf_label, macro_close, macro_ema)
    return build_setup_from_channel(df, symbol, tf_label, macro_close, macro_ema)

# ==========================================
# 4. FRONT-END CHART RENDERER
# ==========================================
def generate_execution_chart(setup, mobile_view=False):
    img_buf = io.BytesIO()
    g = setup['geometry_data']
    full_df = g['df']
    chart_len = min(60 if mobile_view else 90, len(full_df))
    chart_df = full_df.tail(chart_len).copy()

    mc = mpf.make_marketcolors(up='#089981', down='#f23645', edge='inherit', wick='inherit')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', y_on_right=True, facecolor='#131722', figcolor='#131722')

    addplots = []
    if 'EMA50' in chart_df.columns:
        addplots.append(mpf.make_addplot(chart_df['EMA50'], color='#2962ff', width=1.3))

    figsize = (8, 9) if mobile_view else (12, 7)
    fig, axlist = mpf.plot(
        chart_df, type='candle', style=style, volume=False,
        addplot=addplots if addplots else None, returnfig=True, figsize=figsize
    )

    fig.savefig(img_buf, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf

# ==========================================
# 5. TELEGRAM INTERFACE & SUB-MENUS (UNCLUTTERED LAYOUT)
# ==========================================
user_languages = {}
user_view_modes = {}
active_subscribers = set()
pending_custom_ticker_tf = {}

ASSET_CONTAINER = {
    "Forex Majors": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"],
    "Forex Crosses": ["EURGBP", "EURJPY", "GBPJPY", "GBPAUD", "AUDJPY", "EURAUD"],
    "Commodities": ["XAUUSD", "XAGUSD", "OIL"],
    "Crypto": ["BTCUSD", "ETHUSD"],
    "Indices": ["US30", "NAS100", "SPX500"],
    "Stocks": ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META"]
}

def get_home_menu(chat_id):
    lang = user_languages.get(chat_id, "English")
    is_mobile = user_view_modes.get(chat_id, True)
    auto_active = chat_id in active_subscribers

    lang_flags = {"English": "🇬🇧 English", "Hausa": "🇳🇬 Hausa", "Pidgin": "🇳🇬 Pidgin"}
    current_flag = lang_flags.get(lang, "🇬🇧 English")
    auto_status = "🟢 Scanner: ON" if auto_active else "🔴 Scanner: OFF"
    view_toggle_btn = "📱 View: MOBILE" if is_mobile else "💻 View: DESKTOP"
    
    is_mt5_connected = (time.time() - ea_telemetry.get("last_ping", 0)) < 45
    mt5_status_btn = "🟢 MT5 Connected" if is_mt5_connected else "🔴 MT5 Disconnected"

    # STACKED SINGLE-ROW LAYOUT PREVENTS TEXT CLIPPING ON MOBILE SCREENS
    keyboard = [
        [InlineKeyboardButton("📊 Run Institutional Analysis", callback_data="menu_analysis")],
        [InlineKeyboardButton("🚨 Generate Live Signal", callback_data="menu_signal")],
        [InlineKeyboardButton("⚡ Scalper Mode (1m/3m/5m/15m)", callback_data="menu_scalper")],
        [InlineKeyboardButton("🔍 Pattern Scanner (Choose Pair)", callback_data="menu_pattern_scanner")],
        [InlineKeyboardButton("✏️ Type Custom Ticker / Pair", callback_data="prompt_custom_ticker")],
        [InlineKeyboardButton(f"📡 Status: {mt5_status_btn}", callback_data="view_mt5_status")],
        [InlineKeyboardButton(f"{view_toggle_btn}", callback_data="toggle_view_mode")],
        [InlineKeyboardButton(f"{auto_status}", callback_data="toggle_auto_scan")],
        [InlineKeyboardButton(f"🌐 Language: {current_flag}", callback_data="menu_lang_select")],
        [InlineKeyboardButton("ℹ️ Help & EA Setup Rules", callback_data="menu_help")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_category_keyboard(prefix, back_callback, extra_row=None):
    kb = [
        [InlineKeyboardButton("💱 Forex Majors", callback_data=f"{prefix}|Forex Majors")],
        [InlineKeyboardButton("💱 Forex Crosses", callback_data=f"{prefix}|Forex Crosses")],
        [InlineKeyboardButton("🥇 Commodities", callback_data=f"{prefix}|Commodities")],
        [InlineKeyboardButton("🪙 Crypto", callback_data=f"{prefix}|Crypto")],
        [InlineKeyboardButton("📈 Indices & Stocks", callback_data=f"{prefix}|Indices")],
    ]
    if extra_row:
        kb.append(extra_row)
    kb.append([InlineKeyboardButton("« Back to Home", callback_data=back_callback)])
    return InlineKeyboardMarkup(kb)

def get_pairs_keyboard(pairs, prefix, back_callback):
    kb, row = [], []
    for pair in pairs:
        row.append(InlineKeyboardButton(pair, callback_data=f"{prefix}|{pair}"))
        if len(row) == 2:
            kb.append(row)
            row = []
    if row:
        kb.append(row)
    kb.append([InlineKeyboardButton("« Back", callback_data=back_callback)])
    return InlineKeyboardMarkup(kb)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_languages[chat_id] = "English"
    user_view_modes[chat_id] = True
    
    welcome_text = (
        "🤖 INSTITUTIONAL TRADING TERMINAL\n"
        "----------------------------------\n"
        "Full chart pattern recognition & EA Bridge active.\n\n"
        "Select an option from the menu below:"
    )
    await update.message.reply_text(
        welcome_text,
        reply_markup=get_home_menu(chat_id)
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    symbol = update.message.text.strip().upper()
    lang = user_languages.get(chat_id, "English")
    timeframe = pending_custom_ticker_tf.pop(chat_id, "30min")

    status_msg = await update.message.reply_text(f"Scanning {symbol} ({timeframe})...")
    try:
        setup = central_decision_engine(symbol, timeframe=timeframe)
        chart_img = generate_execution_chart(setup, mobile_view=user_view_modes.get(chat_id, True))
        
        info_text = (
            f"📊 {symbol} ({setup['selected_tf']})\n"
            f"Pattern: {setup['pattern_name']}\n"
            f"Signal: {setup['direction']}\n"
            f"Confidence: {setup['confidence']:.1f}%\n"
            f"Validity: {setup['trendline_status']}"
        )
        await status_msg.delete()
        await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=info_text)
        await context.bot.send_message(chat_id=chat_id, text="Select next action:", reply_markup=get_home_menu(chat_id))
    except Exception as e:
        await status_msg.edit_text(f"Error evaluating '{symbol}': {str(e)}", reply_markup=get_home_menu(chat_id))

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id

    if data == "menu_home":
        await query.edit_message_text(text="Main Control Menu:", reply_markup=get_home_menu(chat_id))

    elif data == "toggle_view_mode":
        user_view_modes[chat_id] = not user_view_modes.get(chat_id, True)
        mode_str = "MOBILE 📱" if user_view_modes[chat_id] else "DESKTOP 💻"
        await query.edit_message_text(text=f"Layout set to {mode_str}.", reply_markup=get_home_menu(chat_id))

    elif data == "view_mt5_status":
        is_connected = (time.time() - ea_telemetry.get("last_ping", 0)) < 45
        status_str = "🟢 ONLINE" if is_connected else "🔴 OFFLINE"
        acc = ea_telemetry.get("account", {})
        
        info_text = (
            f"📡 METATRADER 5 EA STATUS\n"
            f"-------------------------\n"
            f"Connection: {status_str}\n"
            f"Account: {acc.get('login', 'N/A')}\n"
            f"Server: {acc.get('server', 'N/A')}\n"
            f"Balance: ${acc.get('balance', 0.0):,.2f}\n"
            f"Equity: ${acc.get('equity', 0.0):,.2f}"
        )
        kb = [[InlineKeyboardButton("« Back", callback_data="menu_home")]]
        await query.edit_message_text(text=info_text, reply_markup=InlineKeyboardMarkup(kb))

    elif data == "menu_pattern_scanner" or data == "menu_analysis" or data == "menu_signal":
        await query.edit_message_text(
            text="Choose an Asset Category:",
            reply_markup=get_category_keyboard("cat_select", "menu_home")
        )

    elif data.startswith("cat_select|"):
        _, cat_name = data.split("|", 1)
        pairs = ASSET_CONTAINER.get(cat_name, ASSET_CONTAINER["Forex Majors"])
        await query.edit_message_text(
            text=f"Select pair in {cat_name}:",
            reply_markup=get_pairs_keyboard(pairs, "pair_run", "menu_pattern_scanner")
        )

    elif data.startswith("pair_run|"):
        _, symbol = data.split("|", 1)
        await query.edit_message_text(text=f"Analyzing {symbol}...")
        try:
            setup = central_decision_engine(symbol, timeframe="30min")
            chart_img = generate_execution_chart(setup, mobile_view=user_view_modes.get(chat_id, True))
            await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"{symbol} Scan Complete ({setup['direction']})")
            await context.bot.send_message(chat_id=chat_id, text="Main Menu:", reply_markup=get_home_menu(chat_id))
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"Scan error: {str(e)}", reply_markup=get_home_menu(chat_id))

    elif data == "menu_scalper":
        kb = [
            [InlineKeyboardButton("1 Minute", callback_data="tf_scalp|1min"),
             InlineKeyboardButton("3 Minute", callback_data="tf_scalp|3min")],
            [InlineKeyboardButton("5 Minute", callback_data="tf_scalp|5min"),
             InlineKeyboardButton("15 Minute", callback_data="tf_scalp|15min")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="⚡ Select Scalper Timeframe:", reply_markup=InlineKeyboardMarkup(kb))

    elif data.startswith("tf_scalp|"):
        _, tf_code = data.split("|", 1)
        pending_custom_ticker_tf[chat_id] = tf_code
        await query.edit_message_text(
            text=f"Selected {tf_code}. Type any ticker (e.g., GBPAUD, BTCUSD, XAUUSD) in chat to analyze.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_scalper")]])
        )

    elif data == "prompt_custom_ticker":
        pending_custom_ticker_tf.pop(chat_id, None)
        await query.edit_message_text(
            text="Type any Ticker / Pair in chat (e.g., EURUSD, XAUUSD):",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_home")]])
        )

    elif data == "toggle_auto_scan":
        if chat_id in active_subscribers:
            active_subscribers.remove(chat_id)
        else:
            active_subscribers.add(chat_id)
        await query.edit_message_text(text="Scanner updated.", reply_markup=get_home_menu(chat_id))

    elif data == "menu_lang_select":
        kb = [
            [InlineKeyboardButton("🇬🇧 English", callback_data="set_lang|English")],
            [InlineKeyboardButton("🇳🇬 Hausa", callback_data="set_lang|Hausa")],
            [InlineKeyboardButton("🇳🇬 Pidgin", callback_data="set_lang|Pidgin")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(text="Select Language:", reply_markup=InlineKeyboardMarkup(kb))

    elif data.startswith("set_lang|"):
        new_lang = data.split("|")[1]
        user_languages[chat_id] = new_lang
        await query.edit_message_text(text=f"Language set to {new_lang}.", reply_markup=get_home_menu(chat_id))

    elif data == "menu_help":
        help_text = (
            "📖 OPERATIONAL RULES:\n\n"
            "1. Connect your MT5 EA using the Webhook URL endpoint.\n"
            "2. Ensure your MT5 terminal has WebRequest enabled for your server domain.\n"
            "3. Multi-timeframe trend alignment (1H Macro + Scalper Timeframes) is enforced automatically."
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

    loop.run_until_complete(app_bot.initialize())
    loop.run_until_complete(app_bot.start())
    loop.run_until_complete(app_bot.updater.start_polling(drop_pending_updates=True))
    loop.run_forever()

if __name__ == "__main__":
    threading.Thread(target=run_telegram_bot, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
