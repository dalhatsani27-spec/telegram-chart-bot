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
from volume_profile import compute_volume_profile
from confirmation_engine import check_current_confirmation
import htf_context
import mt5_data
import legacy_data
import trade_state as ts
import engine
import engine_api
import ai_throttle
from direction_banner import direction_banner
from institutional_analysis import run_topdown_analysis, format_institutional_report
from amd_analysis import run_amd_analysis, format_amd_report
from market_structure import structure_trade_permission
from chart_engine import (
    generate_smc_map,
    generate_amd_map,
    generate_trendline_map,
    generate_ticket_chart,
)
from silver_bullet import run_silver_bullet_analysis, format_silver_bullet_report
from strategy_engine import run_strategy_for_symbol, format_trade_ticket

# ==========================================
# 1. FLASK WEB SERVER (local only -- EA talks to 127.0.0.1)
# ==========================================
app = Flask(__name__)

@app.route('/')
@app.route('/health')
@app.route('/ping')
def health_check():
    return {
        "status": "ok",
        "service": "Price-Action Engine + MT5 Control Center",
        "ts": datetime.utcnow().isoformat() + "Z",
    }, 200

def _public_base_url():
    explicit = (os.environ.get("RENDER_EXTERNAL_URL")
                or os.environ.get("KEEP_ALIVE_URL")
                or "").strip().rstrip("/")
    if explicit:
        return explicit
    host = (os.environ.get("RENDER_EXTERNAL_HOSTNAME") or "").strip()
    if host:
        return f"https://{host}"
    return None

def keep_alive_ping():
    """Ping /health every 4 min so Render free-tier stays awake while process is up.
    Also set an external monitor (UptimeRobot) on /health every 5 min to wake from full sleep."""
    import requests
    interval = int(os.environ.get("KEEP_ALIVE_INTERVAL_SEC", "240"))
    time.sleep(20)
    while True:
        base = _public_base_url()
        targets = []
        if base:
            targets.extend([f"{base}/health", f"{base}/", f"{base}/ping"])
        port = int(os.environ.get("PORT", "5000"))
        targets.append(f"http://127.0.0.1:{port}/health")
        for url in targets:
            try:
                requests.get(url, timeout=12)
            except Exception:
                pass
        time.sleep(interval)

threading.Thread(target=keep_alive_ping, daemon=True, name="keep-alive").start()

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
    """
    Informational analysis for Telegram Pattern Scanner / Run Analysis.

    ALWAYS builds a pivot-anchored parallel channel family (the same structure
    you draw by hand on MT5: swing highs/lows → rails). Pattern scanner
    results (flag, H&S, triangle, etc.) layer on top when they qualify.
    Charts must never come back as candles-only.
    """
    df = mt5_data.fetch_candles(symbol, timeframe, count=250)
    if df is None or df.empty or len(df) < 40:
        raise ValueError(f"Unable to retrieve market data for '{symbol}' on {timeframe}.")
    tf_label = TF_LABELS.get(timeframe, timeframe)
    volume_profile = compute_volume_profile(df.iloc[:-1])
    current_close = float(df['Close'].iloc[-1])
    atr = pattern_atr(df) or 0.001

    # --- 1) Pivot-anchored parallel channel family (primary chart structure) ---
    family = {}
    try:
        from trendline_family import build_trendline_family
        family = build_trendline_family(df, max_lines=4)
        if family.get("error"):
            family = {}
    except Exception:
        family = {}

    # --- 2) Classic pattern scan (flags, H&S, triangles, double top/bottom…) ---
    best, all_patterns = scan_all_patterns(df.iloc[:-1], volume_profile=volume_profile)

    # If no classic pattern, promote the channel family itself as the "pattern"
    if best is None and family and family.get("family_kind") in ("ascending", "descending"):
        kind = family["family_kind"]
        direction = "BUY" if kind == "ascending" else "SELL"
        ch = family.get("channel") or {}
        upper = ch.get("upper") or {}
        lower = ch.get("lower") or {}
        # Build a synthetic Pattern-like payload so the rest of the pipeline is uniform
        from patterns import Pattern
        y_hi = float(upper.get("y_end", current_close * 1.002))
        y_lo = float(lower.get("y_end", current_close * 0.998))
        trigger = y_hi if direction == "BUY" else y_lo
        # Trigger line along the far rail of the channel
        rail = upper if direction == "BUY" else lower
        if rail and "x0" in rail:
            trigger_line = [
                (int(rail["x0"]), float(rail["y0"])),
                (int(rail.get("x1", rail["x0"])), float(rail.get("y1", rail["y0"]))),
            ]
        else:
            trigger_line = [(max(0, len(df) - 40), trigger), (len(df) - 1, trigger)]
        conf = float(family.get("strength") or 60)
        # Key points = pivot anchors
        key_points = []
        for p in (family.get("pivots") or [])[-8:]:
            key_points.append((int(p["index"]), float(p["price"]),
                               "HH" if p.get("type") == "high" else "LL"))
        name = "Ascending Channel" if kind == "ascending" else "Descending Channel"
        best = Pattern(
            name, "continuation", direction,
            trigger_price=trigger,
            trigger_line=trigger_line,
            key_points=key_points,
            confidence=float(np.clip(conf, 50, 88)),
            note=(f"Pivot-anchored {name.lower()} from swing structure. "
                  f"Price is {'above' if direction == 'BUY' else 'below'} the primary rail — "
                  f"direction follows the channel until a break with acceptance."),
        )
        all_patterns = [best] + list(all_patterns or [])

    if best is not None:
        htf_bias, htf_desc = htf_context.get_htf_bias(symbol, timeframe)
        htf_delta, htf_note = htf_context.htf_alignment_adjustment(best.bias, htf_bias, htf_desc)
        if htf_delta != 0.0:
            best.confidence = float(np.clip(best.confidence + htf_delta, 0.0, 100.0))
            best.note += f" {htf_note}"

        direction = best.bias
        trigger = best.trigger_price
        span_xs = [p[0] for p in (best.trigger_line or [])]
        window_start = max(0, min(span_xs) - 3) if span_xs else max(0, len(df) - 60)
        local_window = df.iloc[window_start:]
        resistance_level = float(local_window['High'].max())
        support_level = float(local_window['Low'].min())
        if direction == "BUY":
            structural_sl = min(trigger, support_level) - atr * 0.5
            risk = max(abs(current_close - structural_sl), atr * 0.25)
            tp1 = current_close + risk * 1.5
            tp2 = current_close + risk * 3.0
            is_valid = current_close > structural_sl
        else:
            structural_sl = max(trigger, resistance_level) + atr * 0.5
            risk = max(abs(structural_sl - current_close), atr * 0.25)
            tp1 = current_close - risk * 1.5
            tp2 = current_close - risk * 3.0
            is_valid = current_close < structural_sl
        secondary = [p.name for p in all_patterns if p is not best][:2]
        rationale = best.note + (f" Also developing: {', '.join(secondary)}." if secondary else "")

        confirmed, confirmation_type = check_current_confirmation(df, trigger, direction) if is_valid else (False, None)
        if confirmed:
            trendline_status = f"✅ CONFIRMED via {confirmation_type} at {trigger:.5f}"
            action_type = f"{best.name.upper().replace(' ', '_')}_{confirmation_type.upper().replace(' ', '_')}_CONFIRMED"
        elif is_valid:
            trendline_status = f"Awaiting confirmation (marubozu OR a qualifying candlestick pattern) at {trigger:.5f}"
            action_type = f"{best.name.upper().replace(' ', '_')}_WATCH"
        else:
            trendline_status = "Structure invalidated"
            action_type = "AWAITING_CONFIRMATION"

        aggressive_entry = aggressive_label = aggressive_sl = None
        if best.key_points:
            latest_kp = max(best.key_points, key=lambda p: p[0])
            aggressive_entry = float(latest_kp[1])
            aggressive_label = latest_kp[2]
            buf = atr * 0.3
            aggressive_sl = aggressive_entry - buf if direction == "BUY" else aggressive_entry + buf

        # Merge family rails + pivots into geometry so the chart ALWAYS draws them
        geometry = {
            "df": df,
            "mode": "pattern",
            "resistance_level": resistance_level,
            "support_level": support_level,
            "invalidation_threshold": structural_sl,
            "trigger_line": best.trigger_line,
            "key_points": best.key_points,
            "trigger_price": trigger,
            "category": best.category,
            "volume_profile": volume_profile,
            # Pivot-anchored channel payload for generate_trendline_map
            "pivots": family.get("pivots") or [],
            "family_lines": family.get("family_lines") or [],
            "family_kind": family.get("family_kind"),
            "channel": family.get("channel"),
            "upper_line": family.get("upper_line"),
            "middle_line": family.get("middle_line"),
            "lower_line": family.get("lower_line"),
            "projections": family.get("projections") or [],
            "mw_pattern": family.get("mw_pattern"),
        }
        return {
            "symbol": symbol, "selected_tf": tf_label,
            "direction": direction if is_valid else "NO_SIGNAL",
            "action_type": action_type,
            "rationale": rationale,
            "confidence": float(np.clip(best.confidence, 40, 97)),
            "trendline_status": trendline_status,
            "is_valid": is_valid,
            "pattern_name": best.name,
            "aggressive_entry": aggressive_entry,
            "aggressive_entry_label": aggressive_label,
            "aggressive_sl": aggressive_sl,
            "geometry_data": geometry,
            "family": geometry,
            "entry": current_close,
            "current_market_price": current_close,
            "sl": structural_sl if is_valid else None,
            "tp1": tp1 if is_valid else None,
            "tp2": tp2 if is_valid else None,
        }

    # --- 3) Last resort: regression geometry (still attach any pivots we have) ---
    geom_eval = analyze_backend_line_geometry(df)
    action_eval = evaluate_geometry_signals(geom_eval)
    direction = action_eval["signal"]
    atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0.001
    if direction == "BUY":
        sl = geom_eval['support_level'] - atr_val * 0.5
        tp1 = current_close + abs(current_close - sl) * 1.5
        tp2 = current_close + abs(current_close - sl) * 3.0
    elif direction == "SELL":
        sl = geom_eval['resistance_level'] + atr_val * 0.5
        tp1 = current_close - abs(sl - current_close) * 1.5
        tp2 = current_close - abs(sl - current_close) * 3.0
    else:
        sl = tp1 = tp2 = None
    geom_eval["mode"] = "channel"
    geom_eval["volume_profile"] = volume_profile
    geom_eval["pivots"] = family.get("pivots") or []
    geom_eval["family_lines"] = family.get("family_lines") or []
    geom_eval["family_kind"] = family.get("family_kind")
    geom_eval["channel"] = family.get("channel")
    geom_eval["projections"] = family.get("projections") or []
    return {
        "symbol": symbol, "selected_tf": tf_label, "direction": direction,
        "action_type": action_eval["action_type"],
        "rationale": action_eval["rationale"],
        "confidence": geom_eval["confidence_score"],
        "trendline_status": geom_eval["status_msg"],
        "is_valid": geom_eval["is_valid"],
        "pattern_name": geom_eval["pattern_name"],
        "geometry_data": geom_eval,
        "family": geom_eval,
        "entry": current_close, "current_market_price": current_close,
        "sl": sl, "tp1": tp1, "tp2": tp2,
    }

# ==========================================
# 4. CHART RENDERER  (moved to chart_engine.py)
# ==========================================
# Old generate_smc_chart / generate_execution_chart removed.
# New institutional maps:
#   generate_smc_map()       – SMC zones + sessions
#   generate_amd_map()       – AMD 5-phase + session separators
#   generate_trendline_map() – Trendline / HH-HL style
#   generate_ticket_chart()  – Mobile Manual Trade ticket
# Imported from chart_engine at top of file.


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
    """Professional institutional-style home interface."""
    strat_label = ts.state.strategy_label()
    keyboard = [
        [InlineKeyboardButton(f"🧠 STRATEGY: {strat_label}", callback_data="menu_strategy")],
        [InlineKeyboardButton("🏛  DESK (Strict Decision)", callback_data="menu_desk")],
        [InlineKeyboardButton("🏛  SMC (Top-Down)", callback_data="menu_analysis"),
         InlineKeyboardButton("🕯  AMD Cycle", callback_data="menu_amd")],
        [InlineKeyboardButton("⚡  Silver Bullet", callback_data="menu_silver_bullet"),
         InlineKeyboardButton("📐  Trendline", callback_data="menu_trendline")],
        [InlineKeyboardButton("📐  OTE (Fib Fan+Exp)", callback_data="menu_ote"),
         InlineKeyboardButton("🔍  Pattern Scanner", callback_data="menu_pattern_scanner")],
        [InlineKeyboardButton("🎯  Custom Ticker", callback_data="prompt_custom_ticker")],
        [InlineKeyboardButton("📱  CONTROL PANEL", callback_data="menu_mobile_panel")],
        [InlineKeyboardButton("ℹ️  HELP & GUIDE", callback_data="menu_help")],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_strategy_menu():
    mode = ts.state.get_strategy_mode()
    selected = ts.state.get_selected_strategy()
    mode_label = "🟢 SINGLE" if mode == ts.STRATEGY_MODE_SINGLE else "🟣 HYBRID"
    keyboard = [
        [InlineKeyboardButton(f"Mode: {mode_label} (tap to switch)", callback_data="toggle_strategy_mode")],
        [InlineKeyboardButton(
            f"{'✅' if selected == ts.STRATEGY_SMC else '⚪'} SMC",
            callback_data="set_strategy|SMC"),
         InlineKeyboardButton(
            f"{'✅' if selected == ts.STRATEGY_AMD else '⚪'} AMD",
            callback_data="set_strategy|AMD")],
        [InlineKeyboardButton(
            f"{'✅' if selected == ts.STRATEGY_SILVER_BULLET else '⚪'} Silver Bullet",
            callback_data="set_strategy|SILVER_BULLET"),
         InlineKeyboardButton(
            f"{'✅' if selected == ts.STRATEGY_TRENDLINE else '⚪'} Trendline",
            callback_data="set_strategy|TRENDLINE")],
        [InlineKeyboardButton(
            f"{'✅' if selected == ts.STRATEGY_OTE else '⚪'} OTE (Fib Fan+Exp)",
            callback_data="set_strategy|OTE"),
         InlineKeyboardButton(
            f"{'✅' if selected == ts.STRATEGY_DESK else '⚪'} DESK (Strict)",
            callback_data="set_strategy|DESK")],
        [InlineKeyboardButton("« Back", callback_data="menu_home")],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_mobile_panel_menu():
    mode = ts.state.get_mode()
    master_label = "🟢 Master: ON" if mode != ts.MODE_OFF else "🔴 Master: OFF"
    if mode == ts.MODE_AUTO:
        trade_mode_label = "Trade Mode: AUTO (EA executes)"
    elif mode == ts.MODE_APPROVAL:
        trade_mode_label = "Trade Mode: APPROVAL"
    elif mode == ts.MODE_COPY_TRADE:
        trade_mode_label = "Trade Mode: MOBILE MANUAL (tickets only)"
    else:
        trade_mode_label = "Trade Mode: OFF"
    copy_active = (mode == ts.MODE_COPY_TRADE)
    copy_label = "🟢 Mobile Manual Tickets: ON" if copy_active else "⚪ Mobile Manual Tickets: OFF"
    lot_label = f"Lot Mode: {ts.state.lot_mode}"
    watched = ts.state.get_watched_symbol()
    watch_label = f"🎯 Watching: {watched} (tap to change)" if watched else "🎯 Select Asset to Watch"
    tf_label = f"⏱ Entry Timeframe: {ts.state.watch_timeframe_label()}"
    strat_label = ts.state.strategy_label()

    keyboard = [
        [InlineKeyboardButton(master_label, callback_data="toggle_master")],
        [InlineKeyboardButton(trade_mode_label, callback_data="toggle_trade_mode")],
        [InlineKeyboardButton(copy_label, callback_data="toggle_copy_trade")],
        [InlineKeyboardButton(f"🧠 {strat_label}", callback_data="menu_strategy")],
        [InlineKeyboardButton(lot_label, callback_data="toggle_lot_mode")],
        [InlineKeyboardButton(watch_label, callback_data="menu_watch_asset")],
        [InlineKeyboardButton(tf_label, callback_data="menu_watch_tf")],
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
    welcome = (
        "══════════════════════════════════\n"
        "  INSTITUTIONAL PRICE-ACTION ENGINE\n"
        "══════════════════════════════════\n\n"
        "Top-Down Analysis  •  Trendline Families\n"
        "200 EMA Regime  •  VWAP  •  Volume Profile\n"
        "Pattern Confirmation  •  MT5 Execution\n\n"
        "──────────────────────────────\n"
        "Master switch is OFF by default.\n"
        "Enable trading from the Control Panel\n"
        "when you are ready.\n"
        "──────────────────────────────"
    )
    await update.message.reply_text(welcome, reply_markup=get_home_menu())

# When no EA has checked in recently for the watched symbol (you're away
# from the PC), this loop keeps scanning it so Copy Trade tickets keep
# coming -- throttled to respect the Twelve Data free tier. The symbol
# AND the timeframe are chosen from the Mobile Control Panel, not
# hardcoded -- if no symbol is selected, this loop does nothing until you
# pick one. Any error is logged (not swallowed) so a broken scan is visible
# instead of just silently never producing a ticket.
WATCHLIST_SCAN_INTERVAL_SECONDS = 300  # 5 minutes -- keeps well under free-tier rate limits
_last_scan_error_notified = {"symbol": None, "at": 0}

async def background_watchlist_scanner():
    await asyncio.sleep(15)
    while True:
        symbol = None
        try:
            symbol = ts.state.get_watched_symbol()
            tf = ts.state.get_watch_timeframe()
            if symbol and ts.state.get_mode() != ts.MODE_OFF and not ts.state.is_ea_available(symbol):
                try:
                    result = engine.poll(symbol, tf)
                    if result and result.get("error"):
                        print(f"[watchlist_scanner] {symbol} ({tf}) poll returned error: {result['error']}")
                except Exception as e:
                    print(f"[watchlist_scanner] poll failed for {symbol} ({tf}): {e!r}")
                    # Also surface it in Telegram once per 30 min per symbol, so a
                    # persistently broken scan doesn't just go silent forever.
                    now = time.time()
                    if (_last_scan_error_notified["symbol"] != symbol
                            or now - _last_scan_error_notified["at"] > 1800):
                        _last_scan_error_notified["symbol"] = symbol
                        _last_scan_error_notified["at"] = now
                        push_telegram_message(
                            f"⚠️ Background scanner error on {symbol} ({tf}): {e}\n"
                            f"Copy Trade tickets paused for this symbol until it recovers."
                        )
        except Exception as e:
            print(f"[watchlist_scanner] outer loop failure (symbol={symbol}): {e!r}")
        await asyncio.sleep(WATCHLIST_SCAN_INTERVAL_SECONDS)

async def send_amd_analysis(context, chat_id, symbol):
    """AMD — 5-phase cycle map with TradingView-style session separators."""
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        analysis = run_amd_analysis(symbol)
        report = format_amd_report(analysis)

        try:
            df_1h = analysis.get("df_1h")
            if df_1h is not None and not df_1h.empty:
                chart_img = generate_amd_map(
                    df_1h,
                    symbol,
                    analysis,
                    title_suffix=f"1H | {ts_str}",
                )
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=chart_img,
                    caption=f"{symbol} AMD Cycle Map | Phase: {analysis.get('phase', 'n/a')} | {ts_str}",
                )
        except Exception as chart_err:
            # Fallback: still send text even if chart fails
            pass

        await context.bot.send_message(chat_id=chat_id, text=report)
        await context.bot.send_message(
            chat_id=chat_id, text="Choose next action:", reply_markup=get_home_menu()
        )
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"AMD analysis failed for '{symbol}': {str(e)}",
            reply_markup=get_home_menu()
        )


async def send_institutional_topdown(context, chat_id, symbol):
    """SMC Top-Down — zones, structure, liquidity mapped like the educational charts."""
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        analysis = run_topdown_analysis(symbol)
        report = format_institutional_report(analysis)
        overall_bias = analysis.get("overall_bias", "NEUTRAL")

        try:
            frames = analysis.get("frames") or []
            chart_frame = None
            for f in frames:
                if f.get("tf") == "1h":
                    chart_frame = f
                    break
            if chart_frame is None:
                chart_frame = analysis.get("chart_frame") or (frames[0] if frames else None)

            if chart_frame and chart_frame.get("df") is not None:
                zones = {
                    "fvgs": chart_frame.get("fvgs") or [],
                    "order_blocks": chart_frame.get("order_blocks") or [],
                    "inducements": chart_frame.get("inducements") or [],
                    "structure": chart_frame.get("structure") or {},
                    "volume_profile": chart_frame.get("volume_profile") or analysis.get("volume_profile"),
                    "bos_events": chart_frame.get("bos_events") or [],
                    "swings": (chart_frame.get("structure") or {}).get("swings") or analysis.get("swings"),
                    "projections": analysis.get("projections") or [],
                    "position": analysis.get("position"),
                }
                tf_lab = chart_frame.get("tf_label", "1H")
                chart_img = generate_smc_map(
                    chart_frame["df"],
                    symbol,
                    title_suffix=f"Institutional {tf_lab} | {ts_str}",
                    zones=zones,
                    bias_label=overall_bias,
                    show_sessions=True,
                )
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=chart_img,
                    caption=f"{symbol} SMC Map | Bias: {overall_bias} | {ts_str}",
                )
        except Exception:
            pass

        await context.bot.send_message(chat_id=chat_id, text=report)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Choose next action:",
            reply_markup=get_home_menu()
        )
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Institutional analysis failed for '{symbol}': {str(e)}",
            reply_markup=get_home_menu()
        )


async def send_full_analysis(context, chat_id, symbol, timeframe):
    """Single-timeframe analysis (used by Scalper Mode & Pattern Scanner)."""
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        setup = build_display_setup(symbol, timeframe)
        # generate_trendline_map() reads chart-drawing keys (upper_line/
        # channel/pivots for channel mode, trigger_line/key_points for
        # pattern mode) from setup["family"]/setup["analysis"], falling back
        # to setup itself only if neither is present. build_display_setup()
        # nests all of that under setup["geometry_data"] instead -- without
        # this line the chart silently rendered candles only, with none of
        # the actual detected pattern/channel ever drawn.
        setup["family"] = setup["geometry_data"]
        # Use new trendline / execution style map
        chart_img = generate_trendline_map(
            setup["geometry_data"]["df"],
            symbol,
            setup,
            title_suffix=f"{timeframe} | {ts_str}",
        )
        memo_text = generate_memorandum(symbol, setup)
        fmt = f"{{:.{_decimals_for(symbol)}f}}"
        trade_block = (f"- Price: {fmt.format(setup['current_market_price'])}\n"
                       f"- SL: {fmt.format(setup['sl'])} | TP1: {fmt.format(setup['tp1'])} | TP2: {fmt.format(setup['tp2'])}\n"
                       if setup['sl'] is not None else f"- No trade levels yet: {setup['rationale']}\n")
        entry_styles_block = ""
        if setup.get('aggressive_entry') is not None and setup['sl'] is not None:
            entry_styles_block = (
                f"\nENTRY STYLES (for manual trades -- the EA/AUTO pipeline only ever takes Confirmation):\n"
                f"- Aggressive: {fmt.format(setup['aggressive_entry'])} (at {setup['aggressive_entry_label']}) "
                f"| SL {fmt.format(setup['aggressive_sl'])} -- smaller size, best R:R, lower probability\n"
                f"- Confirmation: close beyond {fmt.format(setup['geometry_data']['trigger_price'])} "
                f"-- higher probability, standard size (this is what the EA waits for)\n"
            )
        summary = (f"SUMMARY ({symbol} | {setup['selected_tf']}):\n"
                   + direction_banner(setup['direction'], extra=setup['action_type']) + "\n"
                   f"- Pattern: {setup['pattern_name']}\n"
                   f"- Confidence: {setup['confidence']:.1f}%\n- Status: {setup['trendline_status']}\n"
                   + trade_block + entry_styles_block)
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🤖 AI Commentary", callback_data=f"ai_commentary|{symbol}|{timeframe}")]])
        await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"{symbol} ({setup['selected_tf']}) | {ts_str}")
        await context.bot.send_message(chat_id=chat_id, text=summary)
        await context.bot.send_message(chat_id=chat_id, text=memo_text, reply_markup=kb)
        await context.bot.send_message(chat_id=chat_id, text="Choose next action:", reply_markup=get_home_menu())
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Failed to analyze '{symbol}': {str(e)}", reply_markup=get_home_menu())


async def send_silver_bullet_analysis(context, chat_id, symbol):
    """ICT Silver Bullet analysis + optional ticket in Mobile Manual mode."""
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        analysis = run_silver_bullet_analysis(symbol)
        report = format_silver_bullet_report(analysis)

        # Chart: reuse SMC map with FVGs + structure
        try:
            df = analysis.get("df")
            if df is not None and not df.empty:
                zones = {
                    "fvgs": analysis.get("fvgs") or [],
                    "order_blocks": analysis.get("order_blocks") or [],
                    "structure": analysis.get("structure") or {},
                }
                chart_img = generate_smc_map(
                    df, symbol,
                    title_suffix=f"Silver Bullet | {ts_str}",
                    zones=zones,
                    bias_label=analysis.get("direction", ""),
                    show_sessions=True,
                )
                await context.bot.send_photo(
                    chat_id=chat_id, photo=chart_img,
                    caption=f"{symbol} Silver Bullet | {ts_str}",
                )
        except Exception:
            pass

        await context.bot.send_message(chat_id=chat_id, text=report)

        # Mobile Manual ticket when in COPY_TRADE mode and setup is valid
        if ts.state.get_mode() == ts.MODE_COPY_TRADE and analysis.get("valid"):
            from silver_bullet import build_silver_bullet_ticket
            ticket = build_silver_bullet_ticket(analysis)
            if ticket:
                ticket_text = format_trade_ticket(
                    {
                        "direction": ticket["direction"],
                        "strategy": "ICT Silver Bullet",
                        "score": analysis.get("score", 0),
                        "reasons": analysis.get("reasons") or [],
                        "ticket": ticket,
                    },
                    symbol,
                )
                await context.bot.send_message(chat_id=chat_id, text=ticket_text)

        await context.bot.send_message(chat_id=chat_id, text="Choose next action:", reply_markup=get_home_menu())
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Silver Bullet analysis failed for '{symbol}': {str(e)}",
            reply_markup=get_home_menu(),
        )


async def send_smart_analysis(context, chat_id, symbol):
    """
    Runs Single or Hybrid strategy engine and returns analysis + ticket
    (ticket only when Mobile Manual / COPY_TRADE is active).
    """
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        result = run_strategy_for_symbol(symbol)
        strategy = result.get("strategy", "")
        report = result.get("report") or "No report"

        # Chart by strategy
        try:
            analysis = result.get("analysis") or {}
            if strategy == ts.STRATEGY_AMD and analysis.get("df_1h") is not None:
                chart_img = generate_amd_map(analysis["df_1h"], symbol, analysis, title_suffix=ts_str)
                await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"{symbol} AMD | {ts_str}")
            elif strategy in (ts.STRATEGY_SMC, ts.STRATEGY_SILVER_BULLET):
                df = None
                zones = {}
                if strategy == ts.STRATEGY_SILVER_BULLET:
                    df = analysis.get("df")
                    zones = {
                        "fvgs": analysis.get("fvgs") or [],
                        "order_blocks": analysis.get("order_blocks") or [],
                        "structure": analysis.get("structure") or {},
                        "volume_profile": analysis.get("volume_profile"),
                        "bos_events": analysis.get("bos_events") or [],
                        "swings": analysis.get("swings"),
                        "projections": result.get("projections") or analysis.get("projections") or [],
                        "position": result.get("position") or analysis.get("position"),
                    }
                else:
                    frames = analysis.get("frames") or []
                    frame = next((f for f in frames if f.get("tf") == "1h"), frames[0] if frames else None)
                    if frame:
                        df = frame.get("df")
                        zones = {
                            "fvgs": frame.get("fvgs") or [],
                            "order_blocks": frame.get("order_blocks") or [],
                            "inducements": frame.get("inducements") or [],
                            "structure": frame.get("structure") or {},
                            "volume_profile": frame.get("volume_profile") or analysis.get("volume_profile"),
                            "bos_events": frame.get("bos_events") or [],
                            "swings": (frame.get("structure") or {}).get("swings") or analysis.get("swings"),
                            "projections": result.get("projections") or analysis.get("projections") or [],
                            "position": result.get("position") or analysis.get("position"),
                        }
                if df is not None:
                    chart_img = generate_smc_map(
                        df, symbol, title_suffix=f"{strategy} | {ts_str}",
                        zones=zones, bias_label=result.get("direction", ""), show_sessions=True,
                    )
                    await context.bot.send_photo(chat_id=chat_id, photo=chart_img, caption=f"{symbol} {strategy} | {ts_str}")
            elif strategy == ts.STRATEGY_TRENDLINE:
                family = result.get("family") or analysis or {}
                df_tl = family.get("df")
                if df_tl is not None and not getattr(df_tl, "empty", True):
                    chart_payload = dict(family)
                    chart_payload["position"] = result.get("position") or family.get("position")
                    chart_img = generate_trendline_map(df_tl, symbol, chart_payload, title_suffix=ts_str)
                    await context.bot.send_photo(
                        chat_id=chat_id, photo=chart_img,
                        caption=f"{symbol} Trendline Family | {ts_str}",
                    )
            elif strategy == ts.STRATEGY_OTE:
                from chart_engine import generate_ote_map
                df_ote = analysis.get("df")
                if df_ote is not None and not getattr(df_ote, "empty", True):
                    chart_img = generate_ote_map(df_ote, symbol, analysis, title_suffix=ts_str)
                    await context.bot.send_photo(
                        chat_id=chat_id, photo=chart_img,
                        caption=f"{symbol} OTE (Fib Fan + Expansion) | {ts_str}",
                    )
            elif strategy == ts.STRATEGY_DESK:
                from chart_engine import generate_ote_map
                # DESK reuses OTE visual (Fan + Expansion) because that is the entry model
                ote_payload = (analysis.get("ote") or analysis) if isinstance(analysis, dict) else {}
                df_desk = analysis.get("df") or ote_payload.get("df")
                chart_data = {
                    "impulse": ote_payload.get("impulse") or analysis.get("impulse"),
                    "fans": ote_payload.get("fans") or analysis.get("fans") or [],
                    "expansions": ote_payload.get("expansions") or analysis.get("expansions") or [],
                    "position": result.get("position") or analysis.get("position"),
                    "ticket": result.get("ticket") or analysis.get("ticket"),
                    "direction": result.get("direction"),
                    "score": result.get("score"),
                }
                if df_desk is not None and not getattr(df_desk, "empty", True):
                    chart_img = generate_ote_map(df_desk, symbol, chart_data, title_suffix=f"DESK | {ts_str}")
                    await context.bot.send_photo(
                        chat_id=chat_id, photo=chart_img,
                        caption=f"{symbol} DESK (Strict) | {ts_str}",
                    )
        except Exception:
            pass

        header = (
            f"Strategy: {strategy}\n"
            f"Mode: {ts.state.strategy_label()}\n"
            f"Direction: {result.get('direction')} | Score: {result.get('score', 0)}/100\n"
        )
        if result.get("chosen_reason"):
            header += f"Why: {result['chosen_reason']}\n"
        await context.bot.send_message(chat_id=chat_id, text=header + "\n" + report)

        # Mobile Manual ticket
        if ts.state.get_mode() == ts.MODE_COPY_TRADE and result.get("valid"):
            ticket_text = format_trade_ticket(result, symbol)
            await context.bot.send_message(chat_id=chat_id, text=ticket_text)

        await context.bot.send_message(chat_id=chat_id, text="Choose next action:", reply_markup=get_home_menu())
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Smart analysis failed for '{symbol}': {str(e)}",
            reply_markup=get_home_menu(),
        )


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global primary_chat_id
    chat_id = update.effective_chat.id
    primary_chat_id = chat_id
    symbol = update.message.text.strip().upper()
    status_msg = await update.message.reply_text(
        f"Scanning {symbol} with {ts.state.strategy_label()}..."
    )
    try:
        await status_msg.delete()
    except Exception:
        pass
    await send_smart_analysis(context, chat_id, symbol)

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global primary_chat_id
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id
    primary_chat_id = chat_id
    lang = user_languages.get(chat_id, "English")

    if data == "menu_home":
        await query.edit_message_text(
            "══════════════════════════════════\n"
            "  INSTITUTIONAL PRICE-ACTION ENGINE\n"
            "══════════════════════════════════\n"
            f"Strategy: {ts.state.strategy_label()}\n"
            "Select a command:",
            reply_markup=get_home_menu()
        )

    # ---------------- Strategy Selector ----------------
    elif data == "menu_strategy":
        await query.edit_message_text(
            "🧠 STRATEGY SELECTOR\n\n"
            "SINGLE = run only the selected strategy\n"
            "HYBRID = scan all, pick best confluence + explain why\n\n"
            f"Current: {ts.state.strategy_label()}",
            reply_markup=get_strategy_menu(),
        )

    elif data == "toggle_strategy_mode":
        if ts.state.get_strategy_mode() == ts.STRATEGY_MODE_SINGLE:
            ts.state.set_strategy_mode(ts.STRATEGY_MODE_HYBRID)
        else:
            ts.state.set_strategy_mode(ts.STRATEGY_MODE_SINGLE)
        await query.edit_message_text(
            f"Mode switched → {ts.state.strategy_label()}",
            reply_markup=get_strategy_menu(),
        )

    elif data.startswith("set_strategy|"):
        name = data.split("|", 1)[1]
        mapping = {
            "SMC": ts.STRATEGY_SMC,
            "AMD": ts.STRATEGY_AMD,
            "SILVER_BULLET": ts.STRATEGY_SILVER_BULLET,
            "TRENDLINE": ts.STRATEGY_TRENDLINE,
            "OTE": ts.STRATEGY_OTE,
            "DESK": ts.STRATEGY_DESK,
        }
        if name in mapping:
            ts.state.set_selected_strategy(mapping[name])
            ts.state.set_strategy_mode(ts.STRATEGY_MODE_SINGLE)
        await query.edit_message_text(
            f"Strategy set → {ts.state.strategy_label()}",
            reply_markup=get_strategy_menu(),
        )

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

    # ---------------- Watched asset selection (away-mode background scanner) ----------------
    elif data == "menu_watch_asset":
        watched = ts.state.get_watched_symbol()
        extra = [InlineKeyboardButton("⛔ Clear Selection (stop watching)", callback_data="watch_clear")] if watched else None
        text = (f"🎯 Currently watching: {watched}\n\nThis is the single asset the background scanner "
               f"focuses on for Copy Trade tickets when you're away and no EA is connected. "
               f"Pick a new one to replace it, or clear it to stop." if watched else
               "🎯 No asset selected yet.\n\nThis is what the background scanner watches for Copy Trade "
               "tickets when you're away and no EA is connected. Pick one below.")
        kb = get_category_keyboard("watch_cat", "menu_mobile_panel", extra_row=extra)
        await query.edit_message_text(text, reply_markup=kb)

    elif data.startswith("watch_cat|"):
        _, cat = data.split("|", 1)
        await query.edit_message_text(f"{cat} — select the asset to watch:",
                                       reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), "watch_set", "menu_watch_asset"))

    elif data.startswith("watch_set|"):
        _, symbol = data.split("|", 1)
        ts.state.set_watched_symbol(symbol)
        await query.edit_message_text(f"🎯 Now watching {symbol}. It'll stay focused on this until you change "
                                      f"or clear it.", reply_markup=get_mobile_panel_menu())

    elif data == "watch_clear":
        ts.state.clear_watched_symbol()
        await query.edit_message_text("🎯 Watch cleared. Background scanner is idle until you pick a new asset.",
                                       reply_markup=get_mobile_panel_menu())

    # ---------------- Entry timeframe selection (background scanner + on-demand trendline) ----------------
    elif data == "menu_watch_tf":
        current = ts.state.get_watch_timeframe()
        rows = []
        row = []
        for tf in ts.WATCH_TIMEFRAMES:
            label = f"{'✅' if tf == current else '⚪'} {ts.WATCH_TIMEFRAME_LABELS[tf]}"
            row.append(InlineKeyboardButton(label, callback_data=f"watch_tf_set|{tf}"))
            if len(row) == 3:
                rows.append(row); row = []
        if row:
            rows.append(row)
        rows.append([InlineKeyboardButton("« Back", callback_data="menu_mobile_panel")])
        await query.edit_message_text(
            f"⏱ Entry Timeframe (currently {ts.state.watch_timeframe_label()})\n\n"
            "This is the candle timeframe used for:\n"
            "• The background scanner (Copy Trade tickets while away)\n"
            "• On-demand Trendline analysis\n\n"
            "Pick the timeframe your entries should be based on:",
            reply_markup=InlineKeyboardMarkup(rows),
        )

    elif data.startswith("watch_tf_set|"):
        _, tf_code = data.split("|", 1)
        ts.state.set_watch_timeframe(tf_code)
        await query.edit_message_text(
            f"⏱ Entry timeframe set to {ts.state.watch_timeframe_label()}.",
            reply_markup=get_mobile_panel_menu(),
        )

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

    elif data == "show_pattern_stats":
        stats = mt5_data.get_pattern_win_rates(magic_number=778899, days=30)
        if not stats:
            text = "No closed trades yet (or MT5 not reachable) -- pattern win-rates will show up here once you have trade history."
        else:
            lines = ["PATTERN WIN RATES (last 30 days):"]
            for name, s in stats.items():
                wr = f"{s['win_rate']:.0f}%" if s['win_rate'] is not None else "n/a"
                lines.append(f"- {name}: {s['trades']} trades | {s['wins']} wins ({wr}) | PnL {s['pnl']:.2f}")
            text = "\n".join(lines)
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
        ts.state.set_selected_strategy(ts.STRATEGY_SMC)
        ts.state.set_strategy_mode(ts.STRATEGY_MODE_SINGLE)
        await query.edit_message_text(f"Running SMC Top-Down for {symbol}...")
        await send_institutional_topdown(context, chat_id, symbol)

    elif data == "menu_amd":
        extra = [InlineKeyboardButton("🔍 Custom Ticker", callback_data="prompt_custom_ticker_amd")]
        await query.edit_message_text(
            "AMD (5-Phase Cycle)\nAccumulation → Manipulation → Displacement → Reversion → Continuation\nPrimary: 1 HOUR + session separators",
            reply_markup=get_category_keyboard("cat_amd", "menu_home", extra_row=extra)
        )
    elif data.startswith("cat_amd|"):
        _, cat = data.split("|", 1)
        await query.edit_message_text(f"{cat}:", reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), "run_amd", "menu_amd"))
    elif data.startswith("run_amd|"):
        _, symbol = data.split("|", 1)
        ts.state.set_selected_strategy(ts.STRATEGY_AMD)
        ts.state.set_strategy_mode(ts.STRATEGY_MODE_SINGLE)
        await query.edit_message_text(f"Running AMD (1H) analysis for {symbol}...")
        await send_amd_analysis(context, chat_id, symbol)
    elif data == "prompt_custom_ticker_amd":
        await query.edit_message_text(
            "Type any ticker for AMD analysis (e.g. XAUUSD, EURUSD).",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_amd")]])
        )

    # ---------------- ICT Silver Bullet ----------------
    elif data == "menu_silver_bullet":
        extra = [InlineKeyboardButton("🔍 Custom Ticker", callback_data="prompt_custom_ticker_sb")]
        await query.edit_message_text(
            "⚡ ICT SILVER BULLET\nWindows (NY): 03:00–04:00 | 10:00–11:00 | 14:00–15:00\nLiquidity sweep → Displacement → FVG entry",
            reply_markup=get_category_keyboard("cat_sb", "menu_home", extra_row=extra)
        )
    elif data.startswith("cat_sb|"):
        _, cat = data.split("|", 1)
        await query.edit_message_text(f"{cat}:", reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), "run_sb", "menu_silver_bullet"))
    elif data.startswith("run_sb|"):
        _, symbol = data.split("|", 1)
        ts.state.set_selected_strategy(ts.STRATEGY_SILVER_BULLET)
        ts.state.set_strategy_mode(ts.STRATEGY_MODE_SINGLE)
        await query.edit_message_text(f"Running Silver Bullet analysis for {symbol}...")
        await send_silver_bullet_analysis(context, chat_id, symbol)
    elif data == "prompt_custom_ticker_sb":
        await query.edit_message_text(
            "Type any ticker for Silver Bullet (e.g. EURUSD, NAS100).",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_silver_bullet")]])
        )

    # ---------------- Trendline Strategy ----------------
    elif data == "menu_trendline":
        extra = [InlineKeyboardButton("🔍 Custom Ticker", callback_data="prompt_custom_ticker_tl")]
        await query.edit_message_text(
            "📐 TRENDLINE STRATEGY\nValid trendlines, HH/HL structure, BOS, entry on retest/break",
            reply_markup=get_category_keyboard("cat_tl", "menu_home", extra_row=extra)
        )
    elif data.startswith("cat_tl|"):
        _, cat = data.split("|", 1)
        await query.edit_message_text(f"{cat}:", reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), "run_tl", "menu_trendline"))
    elif data.startswith("run_tl|"):
        _, symbol = data.split("|", 1)
        ts.state.set_selected_strategy(ts.STRATEGY_TRENDLINE)
        ts.state.set_strategy_mode(ts.STRATEGY_MODE_SINGLE)
        await query.edit_message_text(f"Running Trendline analysis for {symbol}...")
        await send_smart_analysis(context, chat_id, symbol)
    elif data == "prompt_custom_ticker_tl":
        await query.edit_message_text(
            "Type any ticker for Trendline analysis.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_trendline")]])
        )

    # ---------------- DESK (Strict Institutional Decision) ----------------
    elif data == "menu_desk":
        extra = [InlineKeyboardButton("🔍 Custom Ticker", callback_data="prompt_custom_ticker_desk")]
        await query.edit_message_text(
            "🏛 INSTITUTIONAL DESK\n"
            "Strict gates: HTF bias → Structure → OTE entry zone → R:R\n"
            "Only VALID setups are shown. Everything else is WAIT.",
            reply_markup=get_category_keyboard("cat_desk", "menu_home", extra_row=extra)
        )
    elif data.startswith("cat_desk|"):
        _, cat = data.split("|", 1)
        await query.edit_message_text(
            f"{cat}:",
            reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), "run_desk", "menu_desk")
        )
    elif data.startswith("run_desk|"):
        _, symbol = data.split("|", 1)
        ts.state.set_selected_strategy(ts.STRATEGY_DESK)
        ts.state.set_strategy_mode(ts.STRATEGY_MODE_SINGLE)
        await query.edit_message_text(f"Running DESK analysis for {symbol}...")
        await send_smart_analysis(context, chat_id, symbol)
    elif data == "prompt_custom_ticker_desk":
        ts.state.set_selected_strategy(ts.STRATEGY_DESK)
        ts.state.set_strategy_mode(ts.STRATEGY_MODE_SINGLE)
        await query.edit_message_text(
            "Type any ticker for DESK analysis (e.g. XAUUSD, NAS100, EURUSD).",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_desk")]])
        )

    # ---------------- OTE Strategy (Fib Fan + Expansion) ----------------
    elif data == "menu_ote":
        extra = [InlineKeyboardButton("🔍 Custom Ticker", callback_data="prompt_custom_ticker_ote")]
        await query.edit_message_text(
            "📐 OTE STRATEGY\nFibonacci Fan (38.2 / 50 / 61.8) for entry\n"
            "Fibonacci Expansion (127.2 / 161.8 / 200 / 261.8) for targets",
            reply_markup=get_category_keyboard("cat_ote", "menu_home", extra_row=extra)
        )
    elif data.startswith("cat_ote|"):
        _, cat = data.split("|", 1)
        await query.edit_message_text(
            f"{cat}:",
            reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), "run_ote", "menu_ote")
        )
    elif data.startswith("run_ote|"):
        _, symbol = data.split("|", 1)
        ts.state.set_selected_strategy(ts.STRATEGY_OTE)
        ts.state.set_strategy_mode(ts.STRATEGY_MODE_SINGLE)
        await query.edit_message_text(f"Running OTE (Fib Fan + Expansion) for {symbol}...")
        await send_smart_analysis(context, chat_id, symbol)
    elif data == "prompt_custom_ticker_ote":
        ts.state.set_selected_strategy(ts.STRATEGY_OTE)
        ts.state.set_strategy_mode(ts.STRATEGY_MODE_SINGLE)
        await query.edit_message_text(
            "Type any ticker for OTE analysis (e.g. XAUUSD, NAS100, EURUSD).",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_ote")]])
        )

    # ---------------- Smart Strategy Run (uses selected / hybrid) ----------------
    elif data.startswith("run_smart|"):
        _, symbol = data.split("|", 1)
        await query.edit_message_text(f"Running {ts.state.strategy_label()} analysis for {symbol}...")
        await send_smart_analysis(context, chat_id, symbol)

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
    text = (f"🚨 AUTO-FIRED: {setup['symbol']} [{setup['order_type']}]\n"
           + direction_banner(setup['bias'], extra=setup['symbol']) + "\n"
           f"Pattern: {setup['pattern_name']}\nEntry: {fmt.format(setup['entry'])}\n"
           f"SL: {fmt.format(setup['sl'])} | TP1: {fmt.format(setup['tp1'])} | TP2: {fmt.format(setup['tp2'])}")
    push_telegram_message(text)

def on_approval_request(approval_id, setup):
    fmt = f"{{:.{_decimals_for(setup['symbol'])}f}}"
    text = (f"⏳ APPROVAL NEEDED: {setup['symbol']} [{setup['order_type']}]\n"
           + direction_banner(setup['bias'], extra=setup['symbol']) + "\n"
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
           + direction_banner(setup['bias'], extra=setup['symbol']) + "\n"
           f"{setup['pattern_name']}\n{order_note}\n\n"
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
    app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)

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
    loop.create_task(background_watchlist_scanner())
    loop.run_forever()

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    run_telegram_bot()
