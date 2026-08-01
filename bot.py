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

# ==========================================
# 1. FLASK WEB SERVER & EA BRIDGE ENDPOINTS
# ==========================================
app = Flask(__name__)

# State cache for MT5 connection & telemetry
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
    except Exception as e:
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
        base_text = "🎯 DESK SUMMARY: Live price-action structure and pattern confluence evaluated from actual candles."
        return translate_text(base_text, target_language)

    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = f"""
    You are a professional price action desk analyst (day trader / scalper mindset).
    Here are the live structural and pattern metrics derived from actual price data:
    {metrics_summary}
    
    Provide a concise 3-line institutional summary focusing on: which chart pattern is forming,
    the neckline/trigger level to watch, and what would invalidate the setup.
    If no clean pattern is present, say so plainly instead of inventing a directional bias.
    
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
            
    fallback = "🎯 DESK SUMMARY: Structure verified through calculated price geometry and pattern-recognition scan."
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

_TF_TWELVE_INTERVAL = {"1min": "1min", "3min": "1min", "5min": "5min", "15min": "15min"}
_TF_YF_INTERVAL = {"1min": "1m", "3min": "1m", "5min": "5m", "15min": "15m"}
_TF_RESAMPLE = {"3min": "3min"}
_TF_YF_PERIOD = {"1m": "7d", "5m": "60d", "15m": "60d"}
TF_LABELS = {"1min": "1 Minute", "3min": "3 Minute", "5min": "5 Minute", "15min": "15 Minute"}

def _resample_ohlc(df, rule):
    if df.empty:
        return df
    out = df.resample(rule).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    return out

def fetch_timeframe_data(symbol, tf_code, outputsize=250):
    if tf_code not in _TF_TWELVE_INTERVAL:
        raise ValueError(f"Unsupported timeframe '{tf_code}'.")

    td_interval = _TF_TWELVE_INTERVAL[tf_code]
    fetch_size = outputsize * 3 if tf_code == "3min" else outputsize
    raw = fetch_twelve_data(symbol, interval=td_interval, outputsize=min(fetch_size, 5000))

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
        return "Macro Filter Unavailable (insufficient higher-timeframe data)"
    if macro_close > macro_ema:
        return f"Bullish (1H Macro Filter Active at {macro_ema:.5f})"
    return f"Bearish / Correction Mode (1H Macro Filter at {macro_ema:.5f})"

# ==========================================
# 4. UNIFIED PRICE-ACTION DECISION ENGINE
# ==========================================
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
        pattern_name = "No Clear Chart Pattern (Range-Bound / Insufficient Structure)"
        bias = "NONE"

    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else (resistance_level - support_level) * 0.05
    if atr == 0: atr = 0.001

    channel_width = upper_line[-1] - lower_line[-1]
    position_in_channel = (current_close - lower_line[-1]) / channel_width if channel_width != 0 else 0.5
    position_in_channel = float(np.clip(position_in_channel, 0.0, 1.0))

    recent_closes = close_prices[-6:]
    diffs = np.diff(recent_closes)
    if bias == "BUY":
        momentum_bonus = float(min(8.0, np.sum(diffs > 0) * 1.6))
    elif bias == "SELL":
        momentum_bonus = float(min(8.0, np.sum(diffs < 0) * 1.6))
    else:
        momentum_bonus = 0.0

    if bias == "NONE":
        confidence_score = float(np.clip(35.0 + (1.0 - abs(position_in_channel - 0.5)) * 10.0, 25.0, 50.0))
        invalidation_threshold = None
        is_valid = False
        status_msg = "NO CLEAR STRUCTURE (Range-Bound / Ambiguous)"
    else:
        base_confidence = 50.0
        trend_strength_bonus = min(18.0, abs(normalized_slope) * 500.0)
        position_bonus = (1.0 - abs(position_in_channel - 0.5)) * 12.0
        confidence_score = float(np.clip(
            base_confidence + trend_strength_bonus + position_bonus + momentum_bonus, 40.0, 88.0
        ))
        if bias == "BUY":
            invalidation_threshold = support_level - (atr * 0.5)
            is_valid = current_close >= invalidation_threshold
        else:
            invalidation_threshold = resistance_level + (atr * 0.5)
            is_valid = current_close <= invalidation_threshold
        status_msg = "VALID Confirmed Structure" if is_valid else "INVALID (Structure Broken / Threshold Breached)"

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
        return {"signal": "NO_SIGNAL", "action_type": "NO_CLEAR_PATTERN",
                "rationale": "No qualifying chart pattern and no clear channel drift -- price action is too "
                             "noisy/range-bound right now. Waiting for a defined structure before issuing a signal."}

    if not geom_eval['is_valid']:
        return {"signal": "NO_SIGNAL", "action_type": "PATTERN_INVALIDATED_NO_SIGNAL",
                "rationale": "Structure INVALID: price breached the invalidation threshold. No reliable "
                             "directional signal until the structure reforms."}

    if relative_position <= 0.25 or current_close <= support_floor * 1.003:
        return {"signal": "BUY", "action_type": "CHANNEL_SUPPORT_BOUNCE",
                "rationale": f"Price tested the channel floor ({support_floor:.5f}) and held. Support bounce buy."}
    elif relative_position >= 0.75:
        return {"signal": "SELL", "action_type": "CHANNEL_RESISTANCE_REJECTION",
                "rationale": f"Price is at the channel ceiling (relative position {relative_position:.2f}). Resistance rejection sell."}
    elif geom_eval['calculated_bias'] == "BUY":
        return {"signal": "BUY", "action_type": "TREND_CONTINUATION",
                "rationale": "Bullish channel drift holding mid-structure."}
    else:
        return {"signal": "SELL", "action_type": "TREND_CONTINUATION",
                "rationale": "Bearish channel drift holding mid-structure."}

def build_setup_from_pattern(df, best, all_patterns, symbol, tf_label, macro_close, macro_ema):
    current_close = float(df['Close'].iloc[-1])
    atr = pattern_atr(df) or 0.001
    direction = best.bias
    trigger = best.trigger_price

    span_xs = [p[0] for p in (best.trigger_line or [])]
    if span_xs:
        window_start = max(0, min(span_xs) - 3)
    else:
        window_start = max(0, len(df) - 60)
    local_window = df.iloc[window_start:]
    resistance_level = float(local_window['High'].max())
    support_level = float(local_window['Low'].min())

    recent_closes = df['Close'].values[-6:]
    diffs = np.diff(recent_closes) if len(recent_closes) > 1 else np.array([])
    if direction == "BUY":
        momentum_bonus = float(min(8.0, np.sum(diffs > 0) * 1.6)) if len(diffs) else 0.0
    else:
        momentum_bonus = float(min(8.0, np.sum(diffs < 0) * 1.6)) if len(diffs) else 0.0

    confidence = float(np.clip(best.confidence + momentum_bonus, 40.0, 97.0))

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

    if "Flag" in best.name or "Pennant" in best.name:
        pole_pts = [p for p in (best.key_points or []) if "Pole" in p[2]]
        if len(pole_pts) >= 2:
            pole_height = abs(pole_pts[1][1] - pole_pts[0][1])
            if pole_height > 0:
                if direction == "BUY":
                    tp1 = entry + pole_height * 0.618
                    tp2 = entry + pole_height * 1.0
                else:
                    tp1 = entry - pole_height * 0.618
                    tp2 = entry - pole_height * 1.0

    status_msg = (f"VALID -- {best.category.title()} pattern forming, awaiting breakout confirmation at "
                  f"{trigger:.5f}") if is_valid else "INVALID (price already broke through the structural invalidation level)"

    secondary = [p.name for p in all_patterns if p is not best][:2]
    rationale = best.note
    if secondary:
        rationale += f" Also developing on this chart: {', '.join(secondary)} (secondary, lower priority)."

    macro_bias = macro_bias_string(macro_close, macro_ema)

    return {
        "symbol": symbol, "selected_tf": tf_label,
        "direction": direction if is_valid else "NO_SIGNAL",
        "action_type": f"{best.name.upper().replace(' ', '_')}_BREAKOUT_WATCH" if is_valid else "PATTERN_INVALIDATED_NO_SIGNAL",
        "rationale": rationale if is_valid else
                     f"{best.name} was identified but price already closed through the structural invalidation "
                     f"level ({invalidation_threshold:.5f}) -- the setup is no longer clean. Waiting for a fresh structure.",
        "confidence": confidence,
        "macro_bias": macro_bias,
        "trendline_status": status_msg,
        "is_valid": is_valid,
        "pattern_name": best.name,
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
    confidence = geom_eval["confidence_score"]
    current_close = float(df['Close'].iloc[-1])
    atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0.001
    macro_bias = macro_bias_string(macro_close, macro_ema)

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
        entry_price = current_close
        sweet_spot_sl = None
        tp1 = None
        tp2 = None

    geom_eval["mode"] = "channel"
    return {
        "symbol": symbol, "selected_tf": tf_label,
        "direction": action_eval["signal"], "action_type": action_eval["action_type"],
        "rationale": action_eval["rationale"], "confidence": confidence, "macro_bias": macro_bias,
        "trendline_status": geom_eval["status_msg"], "is_valid": geom_eval["is_valid"],
        "pattern_name": geom_eval["pattern_name"], "geometry_data": geom_eval,
        "entry": entry_price, "current_market_price": current_close,
        "sl": sweet_spot_sl, "tp1": tp1, "tp2": tp2,
    }

def central_decision_engine(symbol, timeframe="30min"):
    if timeframe == "30min":
        df, macro_bullish, macro_ema = fetch_institutional_multi_tf_data(symbol)
        macro_close = float(df['Close'].iloc[-1])
        tf_label = "30M (Multi-TF 4H/30M)"
    else:
        df = fetch_timeframe_data(symbol, timeframe)
        if df.empty or len(df) < 40:
            raise ValueError(f"Unable to retrieve verified {TF_LABELS.get(timeframe, timeframe)} market data for '{symbol}'.")
        macro_close, macro_ema = fetch_macro_reference(symbol)
        tf_label = f"{TF_LABELS.get(timeframe, timeframe)} (Scalper Mode, 1H Macro Filter)"

    best, all_patterns = scan_all_patterns(df)
    if best is not None:
        setup = build_setup_from_pattern(df, best, all_patterns, symbol, tf_label, macro_close, macro_ema)
    else:
        setup = build_setup_from_channel(df, symbol, tf_label, macro_close, macro_ema)
    return setup

def format_trade_block(setup, fmt):
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
    invalidation_str = fmt.format(g['invalidation_threshold']) if g.get('invalidation_threshold') is not None else "N/A"
    trigger_line = ""
    if g.get("mode") == "pattern" and g.get("trigger_price") is not None:
        trigger_line = f"- Neckline / Trigger Level to Watch: {fmt.format(g['trigger_price'])}\n"

    memo = (
        f"INSTITUTIONAL STRUCTURAL MEMORANDUM ({asset_symbol})\n"
        f"Pattern Type: {setup['pattern_name']}\n"
        f"Confidence Score: {setup['confidence']:.1f}% | Validity: {setup['trendline_status']}\n"
        f"Timeframe: {setup['selected_tf']} | Signal: {setup['direction']} [{setup['action_type']}]\n\n"
        f"KEY GEOMETRIC LEVELS:\n"
        f"- Upper Resistance / Ceiling: {fmt.format(g['resistance_level'])}\n"
        f"- Support / Floor: {fmt.format(g['support_level'])}\n"
        + trigger_line +
        f"- Invalidation Threshold: {invalidation_str}\n\n"
        f"LIVE MARKET RATIONALE:\n"
        f"{setup['rationale']}"
    )
    return memo

# ==========================================
# 5. FRONT-END CANDLESTICK & PATTERN/GEOMETRY CHART RENDERER
# ==========================================
def generate_execution_chart(setup):
    img_buf = io.BytesIO()
    g = setup['geometry_data']
    full_df = g['df']
    chart_len = min(90, len(full_df))
    chart_df = full_df.tail(chart_len).copy()
    offset = len(full_df) - chart_len

    mc = mpf.make_marketcolors(up='#089981', down='#f23645', edge='inherit', wick='inherit')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', y_on_right=True, facecolor='#131722', figcolor='#131722')

    addplots = []
    if 'EMA50' in chart_df.columns:
        addplots.append(mpf.make_addplot(chart_df['EMA50'], color='#2962ff', width=1.3))
    if 'EMA20' in chart_df.columns:
        addplots.append(mpf.make_addplot(chart_df['EMA20'], color='#ab47bc', width=1.0))

    if g.get("mode") == "channel":
        n = len(chart_df)
        upper_series = pd.Series(g['upper_line'][-n:], index=chart_df.index)
        middle_series = pd.Series(g['middle_line'][-n:], index=chart_df.index)
        lower_series = pd.Series(g['lower_line'][-n:], index=chart_df.index)
        addplots.append(mpf.make_addplot(upper_series, color='#00e676', width=2.0, linestyle='-'))
        addplots.append(mpf.make_addplot(middle_series, color='#ff9800', width=1.5, linestyle='--'))
        addplots.append(mpf.make_addplot(lower_series, color='#00e676', width=2.0, linestyle='-'))

    price_min = chart_df['Low'].min()
    price_max = chart_df['High'].max()
    padding = (price_max - price_min) * 0.15
    if padding == 0: padding = 0.001
    ymin = price_min - padding
    ymax = price_max + padding

    fig, axlist = mpf.plot(
        chart_df, type='candle', style=style, volume=False,
        addplot=addplots if addplots else None, returnfig=True, figsize=(12, 7),
        ylim=(ymin, ymax)
    )
    ax = axlist[0]

    if g.get("mode") == "pattern":
        trigger_line = g.get("trigger_line") or []
        if len(trigger_line) >= 2:
            xs = [pt[0] - offset for pt in trigger_line]
            ys = [pt[1] for pt in trigger_line]
            if len(xs) >= 2 and xs[-1] != xs[0]:
                slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
                extended_x = chart_len - 1
                extended_y = ys[-1] + slope * (extended_x - xs[-1])
                xs.append(extended_x)
                ys.append(extended_y)
            xs_clipped = [max(0, min(chart_len - 1, x)) for x in xs]
            ax.plot(xs_clipped, ys, color='#ffeb3b', linewidth=2.0, linestyle='--', zorder=5, label='Neckline / Trigger')

        for (px, py, label) in (g.get("key_points") or []):
            cx = px - offset
            if 0 <= cx <= chart_len - 1:
                ax.scatter([cx], [py], color='#ffffff', edgecolor='#000000', s=45, zorder=6)
                ax.annotate(label, (cx, py), textcoords="offset points", xytext=(0, 10),
                            fontsize=7, color='#ffffff', ha='center', zorder=6)

        if g.get("trigger_price") is not None:
            ax.axhline(g['trigger_price'], color='#ffeb3b', linestyle=':', linewidth=1.0, alpha=0.6)

    ax.axhline(g['resistance_level'], color='#ffb300', linestyle='-', linewidth=1.0, alpha=0.5)
    ax.axhline(g['support_level'], color='#ffb300', linestyle='-', linewidth=1.0, alpha=0.5)
    if setup['sl'] is not None:
        ax.axhline(setup['entry'], color='#00e676', linestyle='--', linewidth=1.2, label='Entry')

    if setup['direction'] == "NO_SIGNAL":
        validity_tag = "NO SIGNAL"
        status_color = "#ffb300"
    else:
        validity_tag = "VALID" if setup['is_valid'] else "INVALID"
        status_color = "#00e676" if setup['is_valid'] else "#f23645"

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
    ax.set_title(
        f"{setup['symbol']} Price-Action Map [{validity_tag}] | TF: {setup['selected_tf']}\n"
        f"Pattern: {setup['pattern_name']} | Confidence: {setup['confidence']:.1f}% | {current_time_str}",
        color=status_color, fontsize=9, fontweight='bold', pad=12
    )

    fig.savefig(img_buf, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf

# ==========================================
# 6. PROFESSIONAL TELEGRAM MENU & ASSET CONTAINER
# ==========================================
user_languages = {}
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

SCALP_TIMEFRAMES = [("1min", "1 Minute"), ("3min", "3 Minute"), ("5min", "5 Minute"), ("15min", "15 Minute")]

def get_home_menu(lang="English", auto_active=False):
    lang_flags = {"English": "🇬🇧 English", "Hausa": "🇳🇬 Hausa", "Pidgin": "🇳🇬 Pidgin"}
    current_flag = lang_flags.get(lang, "🇬🇧 English")
    auto_status = "🟢 GBPAUD Continuous Scanner: ON" if auto_active else "🔴 GBPAUD Continuous Scanner: OFF"
    
    # Check MT5 connection status
    is_mt5_connected = (time.time() - ea_telemetry.get("last_ping", 0)) < 45
    mt5_status_btn = "🟢 MT5 CONNECTED" if is_mt5_connected else "🔴 MT5 DISCONNECTED"

    keyboard = [
        [InlineKeyboardButton("📊 Run Institutional Analysis", callback_data="menu_analysis"),
         InlineKeyboardButton("🚨 Generate Live Signal", callback_data="menu_signal")],
        [InlineKeyboardButton("⚡ Scalper Mode (1m/3m/5m/15m)", callback_data="menu_scalper")],
        [InlineKeyboardButton("🔍 Pattern Scanner (Choose Pair)", callback_data="menu_pattern_scanner")],
        [InlineKeyboardButton("🔍 Type Custom Ticker / Pair", callback_data="prompt_custom_ticker")],
        [InlineKeyboardButton(auto_status, callback_data="toggle_auto_scan")],
        [InlineKeyboardButton(f"📡 Status: {mt5_status_btn}", callback_data="view_mt5_status")],
        [InlineKeyboardButton(f"🌐 Language: {current_flag}", callback_data="menu_lang_select")],
        [InlineKeyboardButton("ℹ️ Help & Validation Rules", callback_data="menu_help")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_category_keyboard(prefix, back_callback, extra_row=None):
    kb = [
        [InlineKeyboardButton("💱 Forex Majors", callback_data=f"{prefix}|Forex Majors"),
         InlineKeyboardButton("💱 Forex Crosses", callback_data=f"{prefix}|Forex Crosses")],
        [InlineKeyboardButton("🥇 Commodities", callback_data=f"{prefix}|Commodities"),
         InlineKeyboardButton("🪙 Crypto", callback_data=f"{prefix}|Crypto")],
        [InlineKeyboardButton("📈 Indices", callback_data=f"{prefix}|Indices"),
         InlineKeyboardButton("📈 Stocks", callback_data=f"{prefix}|Stocks")],
    ]
    if extra_row:
        kb.append(extra_row)
    kb.append([InlineKeyboardButton("« Back", callback_data=back_callback)])
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
    is_auto = chat_id in active_subscribers
    
    welcome_text = (
        "INSTITUTIONAL TRADING CO-PILOT TERMINAL\n"
        "---------------------------------------\n"
        "Frontend: Candlesticks | Backend: Full Chart-Pattern Recognition Engine\n"
        "(flags, pennants, triangles, wedges, head & shoulders, double/triple tops & bottoms, ranges)\n\n"
        "Status: Ready. Select an asset category, use Scalper Mode for 1m/3m/5m/15m signals, "
        "use the Pattern Scanner, or type any ticker."
    )
    await update.message.reply_text(
        welcome_text,
        reply_markup=get_home_menu("English", is_auto)
    )

def _decimals_for(symbol):
    if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"]:
        return 2
    if "JPY" in symbol:
        return 3
    return 5

async def send_full_analysis(context, chat_id, symbol, timeframe, lang, is_auto):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
    try:
        setup = central_decision_engine(symbol, timeframe=timeframe)

        metrics = (f"- Timestamp: {ts}\n- Pattern: {setup['pattern_name']}\n- Confidence: {setup['confidence']:.1f}%\n"
                   f"- Validity: {setup['trendline_status']}\n- Signal: {setup['direction']} ({setup['action_type']})\n"
                   f"- Timeframe: {setup['selected_tf']}\n")
        ai_commentary = fetch_ai_commentary(metrics, lang)
        memo_text = generate_institutional_memorandum(symbol, setup)
        chart_img = generate_execution_chart(setup)

        fmt = f"{{:.{_decimals_for(symbol)}f}}"
        summary_box = (
            f"SUMMARY BOX ({symbol} | {setup['selected_tf']}):\n"
            f"- Pattern: {setup['pattern_name']}\n"
            f"- Confidence Score: {setup['confidence']:.1f}%\n"
            f"- Validity Status: {'VALID 🟢' if setup['is_valid'] else ('NO SIGNAL 🟡' if setup['direction']=='NO_SIGNAL' else 'INVALID 🔴')}\n"
            f"- Macro Bias: {setup['macro_bias']}\n"
            f"- Signal: {setup['direction']} [{setup['action_type']}]\n"
            + format_trade_block(setup, fmt)
        )

        await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"CHART MAPPING: {symbol} ({setup['selected_tf']}) | {ts}")
        await context.bot.send_message(chat_id=chat_id, text=summary_box)
        await context.bot.send_message(chat_id=chat_id, text=memo_text)
        await context.bot.send_message(chat_id=chat_id, text=ai_commentary)
        await context.bot.send_message(chat_id=chat_id, text="Complete. Choose next action:", reply_markup=get_home_menu(lang, is_auto))
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Failed to analyze '{symbol}': {str(e)}", reply_markup=get_home_menu(lang, is_auto))

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    lang = user_languages.get(chat_id, "English")
    is_auto = chat_id in active_subscribers
    symbol = text.upper()

    timeframe = pending_custom_ticker_tf.pop(chat_id, "30min")
    tf_note = TF_LABELS.get(timeframe, "30M") if timeframe != "30min" else "30M"

    status_msg = await update.message.reply_text(f"Running {tf_note} price-action pattern scan for {symbol}...")
    try:
        await status_msg.delete()
    except Exception:
        pass
    await send_full_analysis(context, chat_id, symbol, timeframe, lang, is_auto)

async def background_continuous_scanner(application):
    await asyncio.sleep(10)
    CHANNEL_ID = "@InstitutionalQuantDesk"
    
    while True:
        try:
            if active_subscribers:
                symbol = "GBPAUD"
                setup = central_decision_engine(symbol)
                
                if setup['direction'] != "NO_SIGNAL" and setup['confidence'] >= 75 and setup['is_valid']:
                    fmt = f"{{:.{_decimals_for(symbol)}f}}"
                    scan_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
                    
                    raw_signal_text = (
                        f"INSTITUTIONAL PRICE-ACTION SIGNAL ({symbol})\n"
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

    elif data == "view_mt5_status":
        is_connected = (time.time() - ea_telemetry.get("last_ping", 0)) < 45
        status_str = "🟢 ONLINE" if is_connected else "🔴 OFFLINE"
        acc = ea_telemetry.get("account", {})
        
        info_text = (
            f"📡 METATRADER 5 BRIDGE STATUS\n"
            f"----------------------------\n"
            f"Terminal Connection: {status_str}\n"
            f"Account #: {acc.get('login', 'N/A')}\n"
            f"Server: {acc.get('server', 'N/A')}\n"
            f"Balance: ${acc.get('balance', 0.0):,.2f}\n"
            f"Equity: ${acc.get('equity', 0.0):,.2f}\n"
            f"Free Margin: ${acc.get('margin_free', 0.0):,.2f}\n"
            f"Active Orders: {len(ea_telemetry.get('positions', []))}"
        )
        kb = [[InlineKeyboardButton("« Back", callback_data="menu_home")]]
        await query.edit_message_text(text=info_text, reply_markup=InlineKeyboardMarkup(kb))

    elif data == "menu_pattern_scanner":
        await query.edit_message_text(
            text="🔍 Pattern Scanner:\nSelect an asset category to choose a specific pair and run full "
                 "pattern-recognition validation (flags, triangles, wedges, H&S, double/triple tops, ranges):",
            reply_markup=get_category_keyboard("scan_cat", "menu_home"))

    elif data.startswith("scan_cat|"):
        _, cat_name = data.split("|", 1)
        pairs = ASSET_CONTAINER.get(cat_name, [])
        await query.edit_message_text(
            text=f"Select specific pair in {cat_name} to scan pattern geometry:",
            reply_markup=get_pairs_keyboard(pairs, "scan_run", "menu_pattern_scanner"))

    elif data.startswith("scan_run|"):
        _, symbol = data.split("|", 1)
        await query.edit_message_text(text=f"Scanning full pattern set for {symbol}...")
        await send_full_analysis(context, chat_id, symbol, "30min", lang, is_auto)

    elif data == "menu_scalper":
        kb = [
            [InlineKeyboardButton("1️⃣ 1 Minute", callback_data="tf_scalp|1min"),
             InlineKeyboardButton("3️⃣ 3 Minute", callback_data="tf_scalp|3min")],
            [InlineKeyboardButton("5️⃣ 5 Minute", callback_data="tf_scalp|5min"),
             InlineKeyboardButton("🔟 15 Minute", callback_data="tf_scalp|15min")],
            [InlineKeyboardButton("« Back", callback_data="menu_home")]
        ]
        await query.edit_message_text(
            text="⚡ SCALPER MODE\nChoose a lower timeframe. The engine scans that timeframe for flags, "
                 "pennants, triangles, wedges, head & shoulders, double/triple tops & bottoms, and ranges, "
                 "using the 1H trend as directional context.",
            reply_markup=InlineKeyboardMarkup(kb))

    elif data.startswith("tf_scalp|"):
        _, tf_code = data.split("|", 1)
        extra = [InlineKeyboardButton("🔍 Type Custom Ticker", callback_data=f"prompt_scalp_ticker|{tf_code}")]
        await query.edit_message_text(
            text=f"{TF_LABELS.get(tf_code, tf_code)} chart selected. Choose an asset category:",
            reply_markup=get_category_keyboard(f"cat_scalp|{tf_code}", "menu_scalper", extra_row=extra))

    elif data.startswith("cat_scalp|"):
        _, tf_code, cat_name = data.split("|", 2)
        pairs = ASSET_CONTAINER.get(cat_name, [])
        await query.edit_message_text(
            text=f"{cat_name} — {TF_LABELS.get(tf_code, tf_code)}:",
            reply_markup=get_pairs_keyboard(pairs, f"run_scalp|{tf_code}", f"tf_scalp|{tf_code}"))

    elif data.startswith("run_scalp|"):
        _, tf_code, symbol = data.split("|", 2)
        await query.edit_message_text(text=f"Running {TF_LABELS.get(tf_code, tf_code)} scalper pattern scan for {symbol}...")
        await send_full_analysis(context, chat_id, symbol, tf_code, lang, is_auto)

    elif data.startswith("prompt_scalp_ticker|"):
        _, tf_code = data.split("|", 1)
        pending_custom_ticker_tf[chat_id] = tf_code
        await query.edit_message_text(
            text=f"Type Any Ticker / Pair for {TF_LABELS.get(tf_code, tf_code)} scalper analysis:\n\n"
                 f"Send any symbol (e.g., AUDUSD, XAUUSD, BTCUSD) in chat.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_scalper")]]))

    elif data == "prompt_custom_ticker":
        pending_custom_ticker_tf.pop(chat_id, None)
        await query.edit_message_text(
            text="Type Any Ticker / Pair:\n\nSend any symbol (e.g., AUDUSD, XAUUSD, BTCUSD) in chat to run "
                 "full pattern-recognition analysis on the 30M timeframe.",
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
        extra = [InlineKeyboardButton("🔍 Type Custom Ticker", callback_data="prompt_custom_ticker")]
        await query.edit_message_text(text="Select Asset Category:", reply_markup=get_category_keyboard("cat_an", "menu_home", extra_row=extra))

    elif data == "menu_signal":
        extra = [InlineKeyboardButton("🔍 Type Custom Ticker", callback_data="prompt_custom_ticker")]
        await query.edit_message_text(text="Select Asset Category for Signal:", reply_markup=get_category_keyboard("cat_sig", "menu_home", extra_row=extra))

    elif data.startswith("cat_an|") or data.startswith("cat_sig|"):
        action_type, cat_name = data.split("|", 1)
        pairs = ASSET_CONTAINER.get(cat_name, [])
        prefix = "run_an" if action_type == "cat_an" else "run_sig"
        back_cb = "menu_analysis" if prefix == "run_an" else "menu_signal"
        await query.edit_message_text(text=f"{cat_name} Assets:", reply_markup=get_pairs_keyboard(pairs, prefix, back_cb))

    elif data.startswith("run_an|"):
        _, symbol = data.split("|", 1)
        await query.edit_message_text(text=f"Running institutional analysis for {symbol}...")
        await send_full_analysis(context, chat_id, symbol, "30min", lang, is_auto)

    elif data.startswith("run_sig|"):
        _, symbol = data.split("|", 1)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
        await query.edit_message_text(text=f"Running signal analysis for {symbol}...")
        try:
            setup = central_decision_engine(symbol, timeframe="30min")
            fmt = f"{{:.{_decimals_for(symbol)}f}}"

            if setup['direction'] == "NO_SIGNAL":
                raw_sig = (
                    f"INSTITUTIONAL PRICE-ACTION SIGNAL ({symbol})\n"
                    f"Pattern: {setup['pattern_name']}\n"
                    f"Status: NO SIGNAL 🟡\n"
                    f"Confidence: {setup['confidence']:.1f}%\n"
                    f"Timestamp: {ts}\n"
                    f"Timeframe: {setup['selected_tf']}\n\n"
                    f"No trade: {setup['rationale']}"
                )
            else:
                raw_sig = (
                    f"INSTITUTIONAL PRICE-ACTION SIGNAL ({symbol})\n"
                    f"Pattern: {setup['pattern_name']}\n"
                    f"Status: {'VALID 🟢' if setup['is_valid'] else 'INVALID 🔴'}\n"
                    f"Confidence: {setup['confidence']:.1f}%\n"
                    f"Timestamp: {ts}\n"
                    f"Timeframe: {setup['selected_tf']}\n"
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
            "HOW PRICE-ACTION PATTERN RECOGNITION WORKS:\n\n"
            "1. Full Pattern Scan (every analysis run):\n"
            "   - The engine detects swing highs/lows, then checks for ALL major chart patterns: "
            "Bull/Bear Flags, Pennants, Ascending/Descending/Symmetrical Triangles, Rising/Falling Wedges, "
            "Head & Shoulders (+ Inverse), Double/Triple Tops & Bottoms, and Rectangles/Ranges.\n"
            "   - Flags, pennants, and other continuation/reversal patterns are weighted highest, since "
            "they tend to give the cleanest directional read.\n"
            "   - Only ONE pattern is mapped on the chart at a time -- the highest-conviction one currently "
            "forming -- with its neckline/trigger trendline drawn so you know exactly what to watch.\n"
            "   - If nothing genuinely qualifies, the bot reports 'No Clear Chart Pattern' and falls back to "
            "a basic channel read, or says so plainly, instead of forcing a label.\n\n"
            "2. Confidence & Validity:\n"
            "   - Confidence blends the pattern's own structural quality with short-term momentum.\n"
            "   - 'VALID' means price hasn't yet closed through the pattern's invalidation level; "
            "'INVALID' means it has, and the bot will not auto-flip to the opposite trade -- it waits for "
            "a fresh structure.\n\n"
            "3. Scalper Mode:\n"
            "   - Runs the identical pattern engine on 1m/3m/5m/15m candles, using the 1H trend as directional context."
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
