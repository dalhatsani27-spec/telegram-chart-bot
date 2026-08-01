import os
import io
import time
import threading
import asyncio
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI

from patterns import scan_all_patterns, _atr as pattern_atr
import mt5_data
import legacy_data
import trade_state as ts
import engine
import engine_api
import ai_throttle

# ==========================================
# 1. FLASK WEB SERVER (local only -- EA talks to 127.0.0.1)
# ==========================================
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Price-Action Engine + MT5 Control Center Active", 200

engine_api.register_routes(app)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# ==========================================
# 2. AI COMMENTARY / TRANSLATION (throttled -- see ai_throttle.py)
# ==========================================
ai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY or "dummy_key")

def _raw_translate(text, target_language):
    if target_language == "English" or not OPENROUTER_API_KEY:
        return text
    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = f"Translate the following trading text into {target_language}, preserving numbers/formatting:\n\n{text}"
    for model in models:
        try:
            response = ai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=600, timeout=8)
            content = response.choices[0].message.content
            if content and len(content.strip()) > 10:
                return content
        except Exception:
            continue
    return text

def translate_text(text, target_language):
    return ai_throttle.throttled_call(text, target_language, _raw_translate, fallback_text=text)

def _raw_commentary(metrics_summary, target_language):
    models = ["google/gemma-4-31b-it:free", "openrouter/free"]
    prompt = (f"You are a professional price-action desk analyst. Metrics:\n{metrics_summary}\n\n"
              f"Give a concise 3-line summary: pattern, trigger/neckline to watch, invalidation. "
              f"Write it in {target_language}.")
    for model in models:
        try:
            response = ai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=300, timeout=8)
            content = response.choices[0].message.content
            if content and len(content.strip()) > 20:
                return content
        except Exception:
            continue
    return "DESK SUMMARY: Structure verified through calculated price geometry and pattern-recognition scan."

def fetch_ai_commentary(metrics_summary, target_language="English"):
    if not OPENROUTER_API_KEY:
        return translate_text("DESK SUMMARY: Pattern and confluence evaluated from live price data.", target_language)
    fallback = "DESK SUMMARY: Structure verified through calculated price geometry and pattern-recognition scan."
    return ai_throttle.throttled_call(metrics_summary, target_language, _raw_commentary, fallback_text=fallback)

# ==========================================
# 3. INFORMATIONAL ANALYSIS ENGINE (for on-demand Telegram buttons --
#    separate from the EA-driven confirmation/execution pipeline in engine.py)
# ==========================================
TREND_CLARITY_THRESHOLD = 0.015
TF_LABELS = {"1min": "1 Minute", "3min": "3 Minute", "5min": "5 Minute", "15min": "15 Minute", "30min": "30 Minute"}

def analyze_backend_line_geometry(df):
    close_prices = df['Close'].values; high_prices = df['High'].values; low_prices = df['Low'].values
    x_vals = np.arange(len(df)); span = len(df)
    resistance_level = float(high_prices.max()); support_level = float(low_prices.min())
    current_close = float(close_prices[-1])
    slope, intercept = np.polyfit(x_vals, close_prices, 1)
    middle_line = slope * x_vals + intercept
    recent_window = min(30, span); window_start = span - recent_window
    window_x = x_vals[window_start:]; window_high = high_prices[window_start:]; window_low = low_prices[window_start:]
    upper_idx = int(np.argmax(window_high)); lower_idx = int(np.argmin(window_low))
    upper_intercept = window_high[upper_idx] - slope * window_x[upper_idx]
    lower_intercept = window_low[lower_idx] - slope * window_x[lower_idx]
    upper_line = slope * x_vals + upper_intercept; lower_line = slope * x_vals + lower_intercept
    avg_price = float(np.mean(close_prices)) or 1.0
    normalized_slope = (slope * span) / avg_price

    if normalized_slope > TREND_CLARITY_THRESHOLD and current_close >= middle_line[-1]:
        pattern_name = "Ascending Channel / Bullish Drift"; bias = "BUY"
    elif normalized_slope < -TREND_CLARITY_THRESHOLD and current_close <= middle_line[-1]:
        pattern_name = "Descending Channel / Bearish Drift"; bias = "SELL"
    else:
        pattern_name = "No Clear Chart Pattern (Range-Bound / Insufficient Structure)"; bias = "NONE"

    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else (resistance_level - support_level) * 0.05
    if atr == 0: atr = 0.001
    channel_width = upper_line[-1] - lower_line[-1]
    position_in_channel = (current_close - lower_line[-1]) / channel_width if channel_width != 0 else 0.5
    position_in_channel = float(np.clip(position_in_channel, 0.0, 1.0))
    recent_closes = close_prices[-6:]; diffs = np.diff(recent_closes)
    if bias == "BUY": momentum_bonus = float(min(8.0, np.sum(diffs > 0) * 1.6))
    elif bias == "SELL": momentum_bonus = float(min(8.0, np.sum(diffs < 0) * 1.6))
    else: momentum_bonus = 0.0

    if bias == "NONE":
        confidence_score = float(np.clip(35.0 + (1.0 - abs(position_in_channel - 0.5)) * 10.0, 25.0, 50.0))
        invalidation_threshold = None; is_valid = False; status_msg = "NO CLEAR STRUCTURE (Range-Bound / Ambiguous)"
    else:
        base_confidence = 50.0
        trend_strength_bonus = min(18.0, abs(normalized_slope) * 500.0)
        position_bonus = (1.0 - abs(position_in_channel - 0.5)) * 12.0
        confidence_score = float(np.clip(base_confidence + trend_strength_bonus + position_bonus + momentum_bonus, 40.0, 88.0))
        if bias == "BUY":
            invalidation_threshold = support_level - (atr * 0.5); is_valid = current_close >= invalidation_threshold
        else:
            invalidation_threshold = resistance_level + (atr * 0.5); is_valid = current_close <= invalidation_threshold
        status_msg = "VALID Confirmed Structure" if is_valid else "INVALID (Structure Broken / Threshold Breached)"

    return {"df": df, "upper_line": upper_line, "middle_line": middle_line, "lower_line": lower_line,
            "resistance_level": resistance_level, "support_level": support_level, "pattern_name": pattern_name,
            "confidence_score": confidence_score, "status_msg": status_msg, "is_valid": is_valid,
            "invalidation_threshold": invalidation_threshold, "calculated_bias": bias, "position_in_channel": position_in_channel}

def evaluate_geometry_signals(geom_eval):
    relative_position = geom_eval.get('position_in_channel', 0.5)
    current_close = geom_eval['df']['Close'].iloc[-1]; support_floor = geom_eval['support_level']
    if geom_eval['calculated_bias'] == "NONE":
        return {"signal": "NO_SIGNAL", "action_type": "NO_CLEAR_PATTERN",
                "rationale": "No qualifying chart pattern and no clear channel drift. Waiting for a defined structure."}
    if not geom_eval['is_valid']:
        return {"signal": "NO_SIGNAL", "action_type": "PATTERN_INVALIDATED_NO_SIGNAL",
                "rationale": "Structure INVALID: price breached the invalidation threshold."}
    if relative_position <= 0.25 or current_close <= support_floor * 1.003:
        return {"signal": "BUY", "action_type": "CHANNEL_SUPPORT_BOUNCE", "rationale": f"Price tested the channel floor ({support_floor:.5f}) and held."}
    elif relative_position >= 0.75:
        return {"signal": "SELL", "action_type": "CHANNEL_RESISTANCE_REJECTION", "rationale": f"Price is at the channel ceiling (position {relative_position:.2f})."}
    elif geom_eval['calculated_bias'] == "BUY":
        return {"signal": "BUY", "action_type": "TREND_CONTINUATION", "rationale": "Bullish channel drift holding mid-structure."}
    else:
        return {"signal": "SELL", "action_type": "TREND_CONTINUATION", "rationale": "Bearish channel drift holding mid-structure."}

def _decimals_for(symbol):
    if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"]:
        return 2
    if "JPY" in symbol: return 3
    return 5

def build_display_setup(symbol, timeframe):
    """Informational (non-executing) analysis for the Telegram 'Run Analysis'/'Scanner' buttons."""
    df = mt5_data.fetch_candles(symbol, timeframe, count=250)
    if df is None or df.empty or len(df) < 40:
        raise ValueError(f"Unable to retrieve market data for '{symbol}' on {timeframe}.")
    tf_label = TF_LABELS.get(timeframe, timeframe)
    best, all_patterns = scan_all_patterns(df)
    current_close = float(df['Close'].iloc[-1])

    if best is not None:
        atr = pattern_atr(df) or 0.001
        direction = best.bias; trigger = best.trigger_price
        span_xs = [p[0] for p in (best.trigger_line or [])]
        window_start = max(0, min(span_xs) - 3) if span_xs else max(0, len(df) - 60)
        local_window = df.iloc[window_start:]
        resistance_level = float(local_window['High'].max()); support_level = float(local_window['Low'].min())
        if direction == "BUY":
            structural_sl = min(trigger, support_level) - atr * 0.5
            risk = max(abs(current_close - structural_sl), atr * 0.25)
            tp1 = current_close + risk * 1.5; tp2 = current_close + risk * 3.0
            is_valid = current_close > structural_sl
        else:
            structural_sl = max(trigger, resistance_level) + atr * 0.5
            risk = max(abs(structural_sl - current_close), atr * 0.25)
            tp1 = current_close - risk * 1.5; tp2 = current_close - risk * 3.0
            is_valid = current_close < structural_sl
        secondary = [p.name for p in all_patterns if p is not best][:2]
        rationale = best.note + (f" Also developing: {', '.join(secondary)}." if secondary else "")
        return {
            "symbol": symbol, "selected_tf": tf_label, "direction": direction if is_valid else "NO_SIGNAL",
            "action_type": f"{best.name.upper().replace(' ', '_')}_WATCH" if is_valid else "AWAITING_MARUBOZU_CONFIRMATION",
            "rationale": rationale, "confidence": float(np.clip(best.confidence, 40, 97)),
            "trendline_status": f"Watching for marubozu confirmation at {trigger:.5f}" if is_valid else "Structure invalidated",
            "is_valid": is_valid, "pattern_name": best.name,
            "geometry_data": {"df": df, "mode": "pattern", "resistance_level": resistance_level, "support_level": support_level,
                              "invalidation_threshold": structural_sl, "trigger_line": best.trigger_line,
                              "key_points": best.key_points, "trigger_price": trigger, "category": best.category},
            "entry": current_close, "current_market_price": current_close,
            "sl": structural_sl if is_valid else None, "tp1": tp1 if is_valid else None, "tp2": tp2 if is_valid else None,
        }
    else:
        geom_eval = analyze_backend_line_geometry(df)
        action_eval = evaluate_geometry_signals(geom_eval)
        direction = action_eval["signal"]; atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0.001
        if direction == "BUY":
            sl = geom_eval['support_level'] - atr_val * 0.5
            tp1 = current_close + abs(current_close - sl) * 1.5; tp2 = current_close + abs(current_close - sl) * 3.0
        elif direction == "SELL":
            sl = geom_eval['resistance_level'] + atr_val * 0.5
            tp1 = current_close - abs(sl - current_close) * 1.5; tp2 = current_close - abs(sl - current_close) * 3.0
        else:
            sl = tp1 = tp2 = None
        geom_eval["mode"] = "channel"
        return {
            "symbol": symbol, "selected_tf": tf_label, "direction": direction, "action_type": action_eval["action_type"],
            "rationale": action_eval["rationale"], "confidence": geom_eval["confidence_score"],
            "trendline_status": geom_eval["status_msg"], "is_valid": geom_eval["is_valid"],
            "pattern_name": geom_eval["pattern_name"], "geometry_data": geom_eval,
            "entry": current_close, "current_market_price": current_close, "sl": sl, "tp1": tp1, "tp2": tp2,
        }

# ==========================================
# 4. CHART RENDERER
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
    if 'EMA50' in chart_df.columns: addplots.append(mpf.make_addplot(chart_df['EMA50'], color='#2962ff', width=1.3))
    if 'EMA20' in chart_df.columns: addplots.append(mpf.make_addplot(chart_df['EMA20'], color='#ab47bc', width=1.0))

    if g.get("mode") == "channel":
        n = len(chart_df)
        addplots.append(mpf.make_addplot(pd.Series(g['upper_line'][-n:], index=chart_df.index), color='#00e676', width=2.0))
        addplots.append(mpf.make_addplot(pd.Series(g['middle_line'][-n:], index=chart_df.index), color='#ff9800', width=1.5, linestyle='--'))
        addplots.append(mpf.make_addplot(pd.Series(g['lower_line'][-n:], index=chart_df.index), color='#00e676', width=2.0))

    price_min = chart_df['Low'].min(); price_max = chart_df['High'].max()
    padding = (price_max - price_min) * 0.15 or 0.001
    fig, axlist = mpf.plot(chart_df, type='candle', style=style, volume=False, addplot=addplots if addplots else None,
                           returnfig=True, figsize=(12, 7), ylim=(price_min - padding, price_max + padding))
    ax = axlist[0]

    if g.get("mode") == "pattern":
        trigger_line = g.get("trigger_line") or []
        if len(trigger_line) >= 2:
            xs = [pt[0] - offset for pt in trigger_line]; ys = [pt[1] for pt in trigger_line]
            if xs[-1] != xs[0]:
                slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
                extended_x = chart_len - 1
                xs.append(extended_x); ys.append(ys[-1] + slope * (extended_x - xs[-2]))
            xs_clipped = [max(0, min(chart_len - 1, x)) for x in xs]
            ax.plot(xs_clipped, ys, color='#ffeb3b', linewidth=2.0, linestyle='--', zorder=5)
        for (px, py, label) in (g.get("key_points") or []):
            cx = px - offset
            if 0 <= cx <= chart_len - 1:
                ax.scatter([cx], [py], color='#ffffff', edgecolor='#000000', s=45, zorder=6)
                ax.annotate(label, (cx, py), textcoords="offset points", xytext=(0, 10), fontsize=7, color='#ffffff', ha='center', zorder=6)
        if g.get("trigger_price") is not None:
            ax.axhline(g['trigger_price'], color='#ffeb3b', linestyle=':', linewidth=1.0, alpha=0.6)

    ax.axhline(g['resistance_level'], color='#ffb300', linestyle='-', linewidth=1.0, alpha=0.5)
    ax.axhline(g['support_level'], color='#ffb300', linestyle='-', linewidth=1.0, alpha=0.5)
    if setup['sl'] is not None:
        ax.axhline(setup['entry'], color='#00e676', linestyle='--', linewidth=1.2)

    validity_tag = "NO SIGNAL" if setup['direction'] == "NO_SIGNAL" else ("VALID" if setup['is_valid'] else "INVALID")
    status_color = "#ffb300" if setup['direction'] == "NO_SIGNAL" else ("#00e676" if setup['is_valid'] else "#f23645")
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"{setup['symbol']} Price-Action Map [{validity_tag}] | TF: {setup['selected_tf']}\n"
                f"Pattern: {setup['pattern_name']} | Confidence: {setup['confidence']:.1f}% | {ts_str}",
                color=status_color, fontsize=9, fontweight='bold', pad=12)
    fig.savefig(img_buf, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    img_buf.seek(0); plt.close(fig)
    return img_buf

def generate_memorandum(asset_symbol, setup):
    g = setup["geometry_data"]
    decimals = _decimals_for(asset_symbol); fmt = f"{{:.{decimals}f}}"
    invalidation_str = fmt.format(g['invalidation_threshold']) if g.get('invalidation_threshold') is not None else "N/A"
    trigger_line = f"- Neckline / Trigger Level to Watch: {fmt.format(g['trigger_price'])}\n" if g.get("mode") == "pattern" and g.get("trigger_price") is not None else ""
    return (f"STRUCTURAL MEMORANDUM ({asset_symbol})\nPattern: {setup['pattern_name']}\n"
            f"Confidence: {setup['confidence']:.1f}% | {setup['trendline_status']}\n"
            f"Timeframe: {setup['selected_tf']} | Signal: {setup['direction']} [{setup['action_type']}]\n\n"
            f"- Resistance: {fmt.format(g['resistance_level'])}\n- Support: {fmt.format(g['support_level'])}\n"
            + trigger_line + f"- Invalidation: {invalidation_str}\n\n{setup['rationale']}")

# ==========================================
# 5. TELEGRAM: STATE, MENUS, HANDLERS
# ==========================================
user_languages = {}
primary_chat_id = None  # this is a personal single-user bot; captured on first contact
TELEGRAM_LOOP = None
TELEGRAM_APP = None

ASSET_CONTAINER = {
    "Forex Majors": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"],
    "Forex Crosses": ["EURGBP", "EURJPY", "GBPJPY", "GBPAUD", "AUDJPY", "EURAUD"],
    "Commodities": ["XAUUSD", "XAGUSD", "OIL"],
    "Crypto": ["BTCUSD", "ETHUSD"],
    "Indices": ["US30", "NAS100", "SPX500"],
    "Stocks": ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META"],
}
SCALP_TIMEFRAMES = [("1min", "1 Minute"), ("3min", "3 Minute"), ("5min", "5 Minute"), ("15min", "15 Minute")]

def push_telegram_message(text, reply_markup=None):
    """Thread-safe way to send a Telegram message from Flask/engine callback threads."""
    if TELEGRAM_LOOP is None or TELEGRAM_APP is None or primary_chat_id is None:
        return
    coro = TELEGRAM_APP.bot.send_message(chat_id=primary_chat_id, text=text, reply_markup=reply_markup)
    asyncio.run_coroutine_threadsafe(coro, TELEGRAM_LOOP)

def push_telegram_photo(photo, caption=""):
    if TELEGRAM_LOOP is None or TELEGRAM_APP is None or primary_chat_id is None:
        return
    coro = TELEGRAM_APP.bot.send_photo(chat_id=primary_chat_id, photo=photo, caption=caption)
    asyncio.run_coroutine_threadsafe(coro, TELEGRAM_LOOP)

def get_home_menu():
    keyboard = [
        [InlineKeyboardButton("📊 Run Institutional Analysis", callback_data="menu_analysis"),
         InlineKeyboardButton("⚡ Scalper Mode (1m/3m/5m/15m)", callback_data="menu_scalper")],
        [InlineKeyboardButton("🔍 Pattern Scanner", callback_data="menu_pattern_scanner"),
         InlineKeyboardButton("🔍 Custom Ticker", callback_data="prompt_custom_ticker")],
        [InlineKeyboardButton("📱 Mobile Control Panel", callback_data="menu_mobile_panel")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="menu_help")],
    ]
    return InlineKeyboardMarkup(keyboard)

def get_mobile_panel_menu():
    mode = ts.state.get_mode()
    master_label = "🟢 Master: ON" if mode != ts.MODE_OFF else "🔴 Master: OFF"
    trade_mode_label = f"Manual Trade Mode: {'AUTO' if mode == ts.MODE_AUTO else 'APPROVAL'}"
    copy_active = (mode == ts.MODE_COPY_TRADE)
    copy_label = "🟢 Copy Trade (Away Mode): ON" if copy_active else "⚪ Copy Trade (Away Mode): OFF"
    lot_label = f"Lot Mode: {ts.state.lot_mode}"

    keyboard = [
        [InlineKeyboardButton(master_label, callback_data="toggle_master")],
        [InlineKeyboardButton(trade_mode_label, callback_data="toggle_trade_mode")],
        [InlineKeyboardButton(copy_label, callback_data="toggle_copy_trade")],
        [InlineKeyboardButton(lot_label, callback_data="toggle_lot_mode")],
        [InlineKeyboardButton("💰 Account & PnL", callback_data="show_account_pnl"),
         InlineKeyboardButton("📈 Open Positions", callback_data="show_open_positions")],
        [InlineKeyboardButton("« Back", callback_data="menu_home")],
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
    if extra_row: kb.append(extra_row)
    kb.append([InlineKeyboardButton("« Back", callback_data=back_callback)])
    return InlineKeyboardMarkup(kb)

def get_pairs_keyboard(pairs, prefix, back_callback):
    kb, row = [], []
    for pair in pairs:
        row.append(InlineKeyboardButton(pair, callback_data=f"{prefix}|{pair}"))
        if len(row) == 2: kb.append(row); row = []
    if row: kb.append(row)
    kb.append([InlineKeyboardButton("« Back", callback_data=back_callback)])
    return InlineKeyboardMarkup(kb)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global primary_chat_id
    primary_chat_id = update.effective_chat.id
    user_languages[primary_chat_id] = "English"
    await update.message.reply_text(
        "INSTITUTIONAL TRADING CO-PILOT\n---------------------------------------\n"
        "Full pattern-recognition engine + marubozu/Fib confirmation, MT5 EA execution, "
        "and this mobile control panel.\n\nMaster switch is OFF by default -- turn it on "
        "from the Mobile Control Panel when you're ready to trade.",
        reply_markup=get_home_menu())

async def send_full_analysis(context, chat_id, symbol, timeframe):
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        setup = build_display_setup(symbol, timeframe)
        chart_img = generate_execution_chart(setup)
        memo_text = generate_memorandum(symbol, setup)
        fmt = f"{{:.{_decimals_for(symbol)}f}}"
        trade_block = (f"- Price: {fmt.format(setup['current_market_price'])}\n"
                       f"- SL: {fmt.format(setup['sl'])} | TP1: {fmt.format(setup['tp1'])} | TP2: {fmt.format(setup['tp2'])}\n"
                       if setup['sl'] is not None else f"- No trade levels yet: {setup['rationale']}\n")
        summary = (f"SUMMARY ({symbol} | {setup['selected_tf']}):\n- Pattern: {setup['pattern_name']}\n"
                   f"- Confidence: {setup['confidence']:.1f}%\n- Status: {setup['trendline_status']}\n"
                   f"- Signal: {setup['direction']} [{setup['action_type']}]\n" + trade_block)
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🤖 AI Commentary", callback_data=f"ai_commentary|{symbol}|{timeframe}")]])
        await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"{symbol} ({setup['selected_tf']}) | {ts_str}")
        await context.bot.send_message(chat_id=chat_id, text=summary)
        await context.bot.send_message(chat_id=chat_id, text=memo_text, reply_markup=kb)
        await context.bot.send_message(chat_id=chat_id, text="Choose next action:", reply_markup=get_home_menu())
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Failed to analyze '{symbol}': {str(e)}", reply_markup=get_home_menu())

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global primary_chat_id
    chat_id = update.effective_chat.id
    primary_chat_id = chat_id
    symbol = update.message.text.strip().upper()
    status_msg = await update.message.reply_text(f"Scanning {symbol}...")
    try:
        await status_msg.delete()
    except Exception:
        pass
    await send_full_analysis(context, chat_id, symbol, "30min")

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global primary_chat_id
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id
    primary_chat_id = chat_id
    lang = user_languages.get(chat_id, "English")

    if data == "menu_home":
        await query.edit_message_text("Main Menu:", reply_markup=get_home_menu())

    # ---------------- Mobile Control Panel ----------------
    elif data == "menu_mobile_panel":
        await query.edit_message_text("📱 MOBILE CONTROL PANEL", reply_markup=get_mobile_panel_menu())

    elif data == "toggle_master":
        ts.state.set_mode(ts.MODE_OFF if ts.state.get_mode() != ts.MODE_OFF else ts.MODE_AUTO)
        await query.edit_message_text("📱 MOBILE CONTROL PANEL", reply_markup=get_mobile_panel_menu())

    elif data == "toggle_trade_mode":
        cur = ts.state.get_mode()
        if cur == ts.MODE_AUTO: ts.state.set_mode(ts.MODE_APPROVAL)
        elif cur == ts.MODE_APPROVAL: ts.state.set_mode(ts.MODE_AUTO)
        # if currently OFF or COPY_TRADE, this button just sets AUTO as the baseline execution mode
        else: ts.state.set_mode(ts.MODE_AUTO)
        await query.edit_message_text("📱 MOBILE CONTROL PANEL", reply_markup=get_mobile_panel_menu())

    elif data == "toggle_copy_trade":
        if ts.state.get_mode() == ts.MODE_COPY_TRADE:
            ts.state.set_mode(ts.MODE_AUTO)
        else:
            ts.state.set_mode(ts.MODE_COPY_TRADE)
        await query.edit_message_text("📱 MOBILE CONTROL PANEL", reply_markup=get_mobile_panel_menu())

    elif data == "toggle_lot_mode":
        ts.state.set_lot_mode("RISK" if ts.state.lot_mode == "MIN" else "MIN")
        await query.edit_message_text("📱 MOBILE CONTROL PANEL", reply_markup=get_mobile_panel_menu())

    elif data == "show_account_pnl":
        acc = mt5_data.get_account_summary()
        if acc is None:
            text = "MT5 not reachable right now. Open the terminal and make sure it's logged in."
        else:
            pnl_day = mt5_data.get_pnl_summary(magic_number=778899, days=1)
            pnl_week = mt5_data.get_pnl_summary(magic_number=778899, days=7)
            text = (f"BROKER: {acc['broker']} ({acc['server']})\nAccount: {acc['login']} | {acc['currency']}\n"
                   f"Leverage: 1:{acc['leverage']}\nBalance: {acc['balance']:.2f} | Equity: {acc['equity']:.2f}\n"
                   f"Free Margin: {acc['margin_free']:.2f}\n\n"
                   f"Daily: {pnl_day['trades']} trades | PnL {pnl_day['pnl']:.2f} | Win rate: "
                   f"{pnl_day['win_rate']:.0f}%\n" if pnl_day['win_rate'] is not None else f"Daily: no closed trades yet\n")
            text += (f"Weekly: {pnl_week['trades']} trades | PnL {pnl_week['pnl']:.2f} | Win rate: "
                    f"{pnl_week['win_rate']:.0f}%" if pnl_week['win_rate'] is not None else "Weekly: no closed trades yet")
        await query.edit_message_text(text, reply_markup=get_mobile_panel_menu())

    elif data == "show_open_positions":
        positions = mt5_data.get_open_positions(magic_number=778899)
        if not positions:
            text = "No open positions."
        else:
            lines = [f"{p['symbol']} {p['type']} {p['volume']} lots | Entry {p['price_open']:.5f} | PnL {p['profit']:.2f}" for p in positions]
            text = "OPEN POSITIONS:\n" + "\n".join(lines)
        await query.edit_message_text(text, reply_markup=get_mobile_panel_menu())

    # ---------------- Approval flow ----------------
    elif data.startswith("approve|") or data.startswith("reject|"):
        action, approval_id = data.split("|", 1)
        req, status = ts.state.resolve_approval(approval_id, approved=(action == "approve"))
        if req is None:
            await query.edit_message_text("This approval request no longer exists.")
        elif status == "expired":
            await query.edit_message_text(f"⏱️ Setup expired before you responded:\n{req['summary']}")
        elif status == "approved":
            await query.edit_message_text(f"✅ Approved -- sending to EA:\n{req['summary']}")
        else:
            await query.edit_message_text(f"❌ Rejected:\n{req['summary']}")

    # ---------------- On-demand AI commentary ----------------
    elif data.startswith("ai_commentary|"):
        _, symbol, timeframe = data.split("|", 2)
        try:
            setup = build_display_setup(symbol, timeframe)
            metrics = (f"Pattern: {setup['pattern_name']}\nConfidence: {setup['confidence']:.1f}%\n"
                      f"Status: {setup['trendline_status']}\nSignal: {setup['direction']}")
            commentary = fetch_ai_commentary(metrics, lang)
            await query.message.reply_text(commentary)
        except Exception as e:
            await query.message.reply_text(f"Couldn't generate commentary: {e}")

    # ---------------- Pattern Scanner / Analysis / Scalper (existing style) ----------------
    elif data == "menu_pattern_scanner":
        await query.edit_message_text("🔍 Pattern Scanner -- pick a category:", reply_markup=get_category_keyboard("scan_cat", "menu_home"))
    elif data.startswith("scan_cat|"):
        _, cat = data.split("|", 1)
        await query.edit_message_text(f"{cat}:", reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), "scan_run", "menu_pattern_scanner"))
    elif data.startswith("scan_run|"):
        _, symbol = data.split("|", 1)
        await query.edit_message_text(f"Scanning {symbol}...")
        await send_full_analysis(context, chat_id, symbol, "30min")

    elif data == "menu_analysis":
        extra = [InlineKeyboardButton("🔍 Custom Ticker", callback_data="prompt_custom_ticker")]
        await query.edit_message_text("Select category:", reply_markup=get_category_keyboard("cat_an", "menu_home", extra_row=extra))
    elif data.startswith("cat_an|"):
        _, cat = data.split("|", 1)
        await query.edit_message_text(f"{cat}:", reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), "run_an", "menu_analysis"))
    elif data.startswith("run_an|"):
        _, symbol = data.split("|", 1)
        await query.edit_message_text(f"Running analysis for {symbol}...")
        await send_full_analysis(context, chat_id, symbol, "30min")

    elif data == "menu_scalper":
        kb = [[InlineKeyboardButton("1️⃣ 1 Minute", callback_data="tf_scalp|1min"), InlineKeyboardButton("3️⃣ 3 Minute", callback_data="tf_scalp|3min")],
              [InlineKeyboardButton("5️⃣ 5 Minute", callback_data="tf_scalp|5min"), InlineKeyboardButton("🔟 15 Minute", callback_data="tf_scalp|15min")],
              [InlineKeyboardButton("« Back", callback_data="menu_home")]]
        await query.edit_message_text("⚡ SCALPER MODE -- choose timeframe:", reply_markup=InlineKeyboardMarkup(kb))
    elif data.startswith("tf_scalp|"):
        _, tf = data.split("|", 1)
        await query.edit_message_text(f"{TF_LABELS.get(tf)}:", reply_markup=get_category_keyboard(f"cat_scalp|{tf}", "menu_scalper"))
    elif data.startswith("cat_scalp|"):
        _, tf, cat = data.split("|", 2)
        await query.edit_message_text(f"{cat} — {TF_LABELS.get(tf)}:", reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), f"run_scalp|{tf}", f"tf_scalp|{tf}"))
    elif data.startswith("run_scalp|"):
        _, tf, symbol = data.split("|", 2)
        await query.edit_message_text(f"Running {TF_LABELS.get(tf)} scan for {symbol}...")
        await send_full_analysis(context, chat_id, symbol, tf)

    elif data == "prompt_custom_ticker":
        await query.edit_message_text("Type any ticker (e.g. AUDUSD, XAUUSD, BTCUSD) in chat.",
                                       reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_home")]]))

    elif data == "menu_help":
        text = ("How this works:\nFull pattern scan every request (flags, triangles, wedges, H&S, "
               "double/triple tops, ranges), weighted toward flags/continuation patterns.\n\n"
               "The EA only fires on a marubozu candle confirming the breakout -- near the trigger fires "
               "at market, stretched moves wait for a Fibonacci pullback instead of chasing.\n\n"
               "Mobile Control Panel: Master ON/OFF, AUTO vs APPROVAL execution, Copy Trade (away mode -- "
               "EA halts, you get manual tickets instead), and live account/PnL.")
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_home")]]))

# ==========================================
# 6. NOTIFIER CALLBACKS (wired into engine.py so it can push Telegram
#    messages without importing this file directly)
# ==========================================
def on_auto_fired(setup):
    fmt = f"{{:.{_decimals_for(setup['symbol'])}f}}"
    text = (f"🚨 AUTO-FIRED: {setup['symbol']} {setup['bias']} [{setup['order_type']}]\n"
           f"Pattern: {setup['pattern_name']}\nEntry: {fmt.format(setup['entry'])}\n"
           f"SL: {fmt.format(setup['sl'])} | TP1: {fmt.format(setup['tp1'])} | TP2: {fmt.format(setup['tp2'])}")
    push_telegram_message(text)

def on_approval_request(approval_id, setup):
    fmt = f"{{:.{_decimals_for(setup['symbol'])}f}}"
    text = (f"⏳ APPROVAL NEEDED: {setup['symbol']} {setup['bias']} [{setup['order_type']}]\n"
           f"Pattern: {setup['pattern_name']}\nEntry: {fmt.format(setup['entry'])}\n"
           f"SL: {fmt.format(setup['sl'])} | TP1: {fmt.format(setup['tp1'])} | TP2: {fmt.format(setup['tp2'])}\n"
           f"Expires in 3 minutes.")
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Approve", callback_data=f"approve|{approval_id}"),
                                InlineKeyboardButton("❌ Reject", callback_data=f"reject|{approval_id}")]])
    push_telegram_message(text, reply_markup=kb)

def on_manual_ticket(setup):
    fmt = f"{{:.{_decimals_for(setup['symbol'])}f}}"
    order_note = "Place as a LIMIT order at this exact price." if setup['order_type'] == "LIMIT" else "Place as a MARKET order now."
    text = (f"📋 MANUAL SETUP (Copy Trade -- you're away, EA is halted):\n"
           f"{setup['symbol']} {setup['bias']} | {setup['pattern_name']}\n{order_note}\n\n"
           f"Entry: {fmt.format(setup['entry'])}\nSL: {fmt.format(setup['sl'])}\n"
           f"TP1: {fmt.format(setup['tp1'])} | TP2: {fmt.format(setup['tp2'])}")
    push_telegram_message(text)

def on_report_event(symbol, event_type, details):
    push_telegram_message(f"ℹ️ {symbol}: {event_type}\n{details}")

engine.register_notifiers(auto_fired=on_auto_fired, approval_request=on_approval_request, manual_ticket=on_manual_ticket)
engine._notify_callbacks["report_event"] = on_report_event

# ==========================================
# 7. ENTRYPOINT -- runs Flask (for the EA) and Telegram polling together,
#    all local, no keep-alive ping needed since nothing sleeps on your PC.
# ==========================================
def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, use_reloader=False)

def run_telegram_bot():
    global TELEGRAM_LOOP, TELEGRAM_APP
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set -- Telegram bot will not start.")
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    TELEGRAM_LOOP = loop

    app_bot = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    TELEGRAM_APP = app_bot
    app_bot.add_handler(CommandHandler("start", start_command))
    app_bot.add_handler(CallbackQueryHandler(button_callback_handler))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    loop.run_until_complete(app_bot.initialize())
    loop.run_until_complete(app_bot.start())
    loop.run_until_complete(app_bot.updater.start_polling(drop_pending_updates=True))
    loop.run_forever()

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    run_telegram_bot()
