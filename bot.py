"""
bot.py
======
Telegram front-end. Two chart-analysis strategies (Trendline, OTE — both
full 4H -> 1H -> 30M top-down reads). Trendline follows classic
educational rules (wicks, retest, candlestick confirmation, TradingView position template). plus the live-trading Control Panel
(Master switch, Auto/Approval/Mobile-Manual modes, EA heartbeat, lot
sizing, Account & PnL, Open Positions).
"""

import os
import time
import threading
import asyncio
from datetime import datetime

from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

import market_data
from market_analysis import direction_banner
from chart_engine import generate_trendline_map, generate_ote_map
import strategies
import execution_engine as engine
from execution_engine import state as ts_state

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# ==========================================
# 1. FLASK WEB SERVER (local only -- EA talks to 127.0.0.1)
# ==========================================
app = Flask(__name__)
engine.register_routes(app)


@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok"}, 200


# ==========================================
# 2. HELPERS
# ==========================================
def _decimals_for(symbol):
    if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"]:
        return 2
    if "JPY" in symbol:
        return 3
    return 5


def format_trade_ticket(symbol, strategy_name, direction, score, ticket, reasons=None):
    """Human-readable Mobile Manual ticket."""
    lines = [
        "══════════════════════════",
        f"  TRADE TICKET  |  {symbol}",
        "══════════════════════════",
        direction_banner(direction),
        f"Strategy  : {strategy_name}",
        f"Score     : {score}/100",
    ]
    if ticket:
        lines += [
            f"Entry     : {ticket.get('entry', '—')}",
            f"SL        : {ticket.get('sl', '—')}",
            f"TP1       : {ticket.get('tp1', '—')}",
            f"TP2       : {ticket.get('tp2', '—')}",
            f"Order     : {ticket.get('order_type', 'MARKET')}",
        ]
    if reasons:
        lines.append("Why       :")
        for r in reasons[:6]:
            lines.append(f"  • {r}")
    lines.append("══════════════════════════")
    lines.append("Enter this trade manually on your phone.")
    return "\n".join(lines)


# ==========================================
# 3. TELEGRAM: STATE, MENUS, HANDLERS
# ==========================================
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
    """Home interface -- only the two remaining chart strategies."""
    strat_label = ts_state.strategy_label()
    keyboard = [
        [InlineKeyboardButton(f"🧠 STRATEGY: {strat_label}", callback_data="menu_strategy")],
        [InlineKeyboardButton("📐  Trendline (4H→1H→30M)", callback_data="menu_trendline")],
        [InlineKeyboardButton("🎯  OTE (Fib Fan+Exp, 30M)", callback_data="menu_ote")],
        [InlineKeyboardButton("🔎  Custom Ticker", callback_data="prompt_custom_ticker")],
        [InlineKeyboardButton("📱  CONTROL PANEL", callback_data="menu_mobile_panel")],
        [InlineKeyboardButton("ℹ️  HELP & GUIDE", callback_data="menu_help")],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_strategy_menu():
    selected = ts_state.get_selected_strategy()
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if selected == engine.STRATEGY_TRENDLINE else '⚪'} Trendline",
            callback_data="set_strategy|TRENDLINE")],
        [InlineKeyboardButton(
            f"{'✅' if selected == engine.STRATEGY_OTE else '⚪'} OTE (Fib Fan+Exp)",
            callback_data="set_strategy|OTE")],
        [InlineKeyboardButton("« Back", callback_data="menu_home")],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_mobile_panel_menu():
    mode = ts_state.get_mode()
    master_label = "🟢 Master: ON" if mode != engine.MODE_OFF else "🔴 Master: OFF"
    if mode == engine.MODE_AUTO:
        trade_mode_label = "Trade Mode: AUTO (EA executes)"
    elif mode == engine.MODE_APPROVAL:
        trade_mode_label = "Trade Mode: APPROVAL"
    elif mode == engine.MODE_COPY_TRADE:
        trade_mode_label = "Trade Mode: MOBILE MANUAL (tickets only)"
    else:
        trade_mode_label = "Trade Mode: OFF"
    copy_active = (mode == engine.MODE_COPY_TRADE)
    copy_label = "🟢 Mobile Manual Tickets: ON" if copy_active else "⚪ Mobile Manual Tickets: OFF"
    lot_label = f"Lot Mode: {ts_state.lot_mode}"
    watched = ts_state.get_watched_symbol()
    watch_label = f"🎯 Watching: {watched} (tap to change)" if watched else "🎯 Select Asset to Watch"
    tf_label = f"⏱ Entry Timeframe: {ts_state.watch_timeframe_label()}"
    strat_label = ts_state.strategy_label()

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
    if extra_row:
        kb.append(extra_row)
    kb.append([InlineKeyboardButton("« Back", callback_data=back_callback)])
    return InlineKeyboardMarkup(kb)


def get_pairs_keyboard(pairs, prefix, back_callback):
    kb, row = [], []
    for pair in pairs:
        row.append(InlineKeyboardButton(pair, callback_data=f"{prefix}|{pair}"))
        if len(row) == 2:
            kb.append(row); row = []
    if row:
        kb.append(row)
    kb.append([InlineKeyboardButton("« Back", callback_data=back_callback)])
    return InlineKeyboardMarkup(kb)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global primary_chat_id
    primary_chat_id = update.effective_chat.id
    welcome = (
        "══════════════════════════════════\n"
        "  TOP-DOWN PRICE-ACTION ENGINE\n"
        "══════════════════════════════════\n\n"
        "Classic Trendline  •  OTE Fib Fan+Expansion\n"
        "4H → 1H → 30M Top-Down Bias  •  200 EMA Regime\n"
        "Structure Permission  •  MT5 Execution\n\n"
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
            symbol = ts_state.get_watched_symbol()
            tf = ts_state.get_watch_timeframe()
            if symbol and ts_state.get_mode() != engine.MODE_OFF and not ts_state.is_ea_available(symbol):
                try:
                    result = engine.poll(symbol, tf)
                    if result and result.get("error"):
                        print(f"[watchlist_scanner] {symbol} ({tf}) poll returned error: {result['error']}")
                except Exception as e:
                    print(f"[watchlist_scanner] poll failed for {symbol} ({tf}): {e!r}")
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


async def send_trendline_analysis(context, chat_id, symbol):
    """Trendline strategy: full 4H -> 1H -> 30M top-down cascade."""
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        family = strategies.run_trendline_analysis(symbol)
        report = strategies.format_trendline_report(family, symbol)

        try:
            df_tl = family.get("df")
            if df_tl is not None and not getattr(df_tl, "empty", True):
                pos = strategies.build_position_container(family)
                chart_payload = dict(family)
                chart_payload["position"] = pos
                chart_img = generate_trendline_map(df_tl, symbol, chart_payload, title_suffix=ts_str)
                await context.bot.send_photo(
                    chat_id=chat_id, photo=chart_img,
                    caption=f"{symbol} Trendline (30M) | {ts_str}",
                )
        except Exception:
            pos = None

        await context.bot.send_message(chat_id=chat_id, text=report)

        if ts_state.get_mode() == engine.MODE_COPY_TRADE and family.get("direction") in ("BUY", "SELL"):
            pos = strategies.build_position_container(family)
            if pos:
                ticket_text = format_trade_ticket(
                    symbol, "TRENDLINE", family.get("direction"), family.get("strength", 0),
                    pos, reasons=family.get("gating_notes"),
                )
                await context.bot.send_message(chat_id=chat_id, text=ticket_text)

        await context.bot.send_message(chat_id=chat_id, text="Choose next action:", reply_markup=get_home_menu())
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Trendline analysis failed for '{symbol}': {str(e)}",
            reply_markup=get_home_menu(),
        )


async def send_ote_analysis(context, chat_id, symbol):
    """OTE strategy: 4H -> 1H top-down bias, Fib Fan+Expansion entry on 30M."""
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        analysis = strategies.run_ote_analysis(symbol)
        report = strategies.format_ote_report(analysis)

        try:
            df_ote = analysis.get("df")
            if df_ote is not None and not getattr(df_ote, "empty", True):
                chart_img = generate_ote_map(df_ote, symbol, analysis, title_suffix=ts_str)
                await context.bot.send_photo(
                    chat_id=chat_id, photo=chart_img,
                    caption=f"{symbol} OTE (Fib Fan + Expansion, 30M) | {ts_str}",
                )
        except Exception:
            pass

        await context.bot.send_message(chat_id=chat_id, text=report)

        if ts_state.get_mode() == engine.MODE_COPY_TRADE and analysis.get("valid"):
            ticket_text = format_trade_ticket(
                symbol, "OTE", analysis.get("direction"), analysis.get("score", 0),
                analysis.get("ticket"), reasons=analysis.get("reasons"),
            )
            await context.bot.send_message(chat_id=chat_id, text=ticket_text)

        await context.bot.send_message(chat_id=chat_id, text="Choose next action:", reply_markup=get_home_menu())
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"OTE analysis failed for '{symbol}': {str(e)}",
            reply_markup=get_home_menu(),
        )


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """A typed symbol is routed to whichever strategy is currently selected."""
    global primary_chat_id
    chat_id = update.effective_chat.id
    primary_chat_id = chat_id
    symbol = (update.message.text or "").strip().upper()
    if not symbol:
        return
    await update.message.reply_text(f"Running {ts_state.strategy_label()} analysis for {symbol}...")
    if ts_state.get_selected_strategy() == engine.STRATEGY_OTE:
        await send_ote_analysis(context, chat_id, symbol)
    else:
        await send_trendline_analysis(context, chat_id, symbol)


async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global primary_chat_id
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id
    primary_chat_id = chat_id

    if data == "menu_home":
        await query.edit_message_text(
            "══════════════════════════════════\n"
            "  TOP-DOWN PRICE-ACTION ENGINE\n"
            "══════════════════════════════════\n"
            f"Strategy: {ts_state.strategy_label()}\n"
            "Select a command:",
            reply_markup=get_home_menu()
        )

    # ---------------- Strategy Selector ----------------
    elif data == "menu_strategy":
        await query.edit_message_text(
            "🧠 STRATEGY SELECTOR\n\n"
            "Both strategies run the same 4H → 1H top-down bias read, "
            "then execute on the 30M chart.\n\n"
            f"Current: {ts_state.strategy_label()}",
            reply_markup=get_strategy_menu(),
        )

    elif data.startswith("set_strategy|"):
        name = data.split("|", 1)[1]
        mapping = {"TRENDLINE": engine.STRATEGY_TRENDLINE, "OTE": engine.STRATEGY_OTE}
        if name in mapping:
            ts_state.set_selected_strategy(mapping[name])
        await query.edit_message_text(
            f"Strategy set → {ts_state.strategy_label()}",
            reply_markup=get_strategy_menu(),
        )

    # ---------------- Trendline ----------------
    elif data == "menu_trendline":
        extra = [InlineKeyboardButton("🔎 Custom Ticker", callback_data="prompt_custom_ticker_tl")]
        await query.edit_message_text(
            "📐 TRENDLINE (4H → 1H → 30M top-down):",
            reply_markup=get_category_keyboard("cat_tl", "menu_home", extra_row=extra),
        )
    elif data.startswith("cat_tl|"):
        _, cat = data.split("|", 1)
        await query.edit_message_text(f"{cat}:", reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), "run_tl", "menu_trendline"))
    elif data.startswith("run_tl|"):
        _, symbol = data.split("|", 1)
        ts_state.set_selected_strategy(engine.STRATEGY_TRENDLINE)
        await query.edit_message_text(f"Running Trendline analysis for {symbol}...")
        await send_trendline_analysis(context, chat_id, symbol)
    elif data == "prompt_custom_ticker_tl":
        ts_state.set_selected_strategy(engine.STRATEGY_TRENDLINE)
        await query.edit_message_text(
            "Type any ticker for Trendline analysis.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_trendline")]])
        )

    # ---------------- OTE ----------------
    elif data == "menu_ote":
        extra = [InlineKeyboardButton("🔎 Custom Ticker", callback_data="prompt_custom_ticker_ote")]
        await query.edit_message_text(
            "🎯 OTE (Fib Fan + Expansion, 4H → 1H → 30M top-down):",
            reply_markup=get_category_keyboard("cat_ote", "menu_home", extra_row=extra),
        )
    elif data.startswith("cat_ote|"):
        _, cat = data.split("|", 1)
        await query.edit_message_text(f"{cat}:", reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat, []), "run_ote", "menu_ote"))
    elif data.startswith("run_ote|"):
        _, symbol = data.split("|", 1)
        ts_state.set_selected_strategy(engine.STRATEGY_OTE)
        await query.edit_message_text(f"Running OTE analysis for {symbol}...")
        await send_ote_analysis(context, chat_id, symbol)
    elif data == "prompt_custom_ticker_ote":
        ts_state.set_selected_strategy(engine.STRATEGY_OTE)
        await query.edit_message_text(
            "Type any ticker for OTE analysis.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_ote")]])
        )

    # ---------------- Generic custom ticker (from home menu) ----------------
    elif data == "prompt_custom_ticker":
        await query.edit_message_text(
            f"Type any ticker for {ts_state.strategy_label()} analysis.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_home")]])
        )

    # ---------------- Mobile Control Panel ----------------
    elif data == "menu_mobile_panel":
        await query.edit_message_text("📱 MOBILE CONTROL PANEL", reply_markup=get_mobile_panel_menu())

    elif data == "toggle_master":
        ts_state.set_mode(engine.MODE_OFF if ts_state.get_mode() != engine.MODE_OFF else engine.MODE_AUTO)
        await query.edit_message_text("📱 MOBILE CONTROL PANEL", reply_markup=get_mobile_panel_menu())

    elif data == "toggle_trade_mode":
        cur = ts_state.get_mode()
        if cur == engine.MODE_AUTO:
            ts_state.set_mode(engine.MODE_APPROVAL)
        elif cur == engine.MODE_APPROVAL:
            ts_state.set_mode(engine.MODE_AUTO)
        else:
            ts_state.set_mode(engine.MODE_AUTO)
        await query.edit_message_text("📱 MOBILE CONTROL PANEL", reply_markup=get_mobile_panel_menu())

    elif data == "toggle_copy_trade":
        if ts_state.get_mode() == engine.MODE_COPY_TRADE:
            ts_state.set_mode(engine.MODE_AUTO)
        else:
            ts_state.set_mode(engine.MODE_COPY_TRADE)
        await query.edit_message_text("📱 MOBILE CONTROL PANEL", reply_markup=get_mobile_panel_menu())

    elif data == "toggle_lot_mode":
        ts_state.set_lot_mode("RISK" if ts_state.lot_mode == "MIN" else "MIN")
        await query.edit_message_text("📱 MOBILE CONTROL PANEL", reply_markup=get_mobile_panel_menu())

    # ---------------- Watched asset selection (away-mode background scanner) ----------------
    elif data == "menu_watch_asset":
        watched = ts_state.get_watched_symbol()
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
        ts_state.set_watched_symbol(symbol)
        await query.edit_message_text(f"🎯 Now watching {symbol}. It'll stay focused on this until you change "
                                      f"or clear it.", reply_markup=get_mobile_panel_menu())

    elif data == "watch_clear":
        ts_state.clear_watched_symbol()
        await query.edit_message_text("🎯 Watch cleared. Background scanner is idle until you pick a new asset.",
                                       reply_markup=get_mobile_panel_menu())

    # ---------------- Entry timeframe selection (background scanner) ----------------
    elif data == "menu_watch_tf":
        current = ts_state.get_watch_timeframe()
        rows = []
        row = []
        for tf in engine.WATCH_TIMEFRAMES:
            label = f"{'✅' if tf == current else '⚪'} {engine.WATCH_TIMEFRAME_LABELS[tf]}"
            row.append(InlineKeyboardButton(label, callback_data=f"watch_tf_set|{tf}"))
            if len(row) == 3:
                rows.append(row); row = []
        if row:
            rows.append(row)
        rows.append([InlineKeyboardButton("« Back", callback_data="menu_mobile_panel")])
        await query.edit_message_text(
            f"⏱ Entry Timeframe (currently {ts_state.watch_timeframe_label()})\n\n"
            "This is the candle timeframe used for the background scanner "
            "(Copy Trade tickets while away). Trendline and OTE always run "
            "their own fixed 4H → 1H → 30M cascade regardless of this setting.\n\n"
            "Pick the timeframe the background scanner's entries should be based on:",
            reply_markup=InlineKeyboardMarkup(rows),
        )

    elif data.startswith("watch_tf_set|"):
        _, tf_code = data.split("|", 1)
        ts_state.set_watch_timeframe(tf_code)
        await query.edit_message_text(
            f"⏱ Entry timeframe set to {ts_state.watch_timeframe_label()}.",
            reply_markup=get_mobile_panel_menu(),
        )

    elif data == "show_account_pnl":
        acc = market_data.get_account_summary()
        if acc is None:
            text = "MT5 not reachable right now. Open the terminal and make sure it's logged in."
        else:
            pnl_day = market_data.get_pnl_summary(magic_number=778899, days=1)
            pnl_week = market_data.get_pnl_summary(magic_number=778899, days=7)
            text = (f"BROKER: {acc['broker']} ({acc['server']})\nAccount: {acc['login']} | {acc['currency']}\n"
                   f"Leverage: 1:{acc['leverage']}\nBalance: {acc['balance']:.2f} | Equity: {acc['equity']:.2f}\n"
                   f"Free Margin: {acc['margin_free']:.2f}\n\n"
                   f"Daily: {pnl_day['trades']} trades | PnL {pnl_day['pnl']:.2f} | Win rate: "
                   f"{pnl_day['win_rate']:.0f}%\n" if pnl_day['win_rate'] is not None else f"Daily: no closed trades yet\n")
            text += (f"Weekly: {pnl_week['trades']} trades | PnL {pnl_week['pnl']:.2f} | Win rate: "
                    f"{pnl_week['win_rate']:.0f}%" if pnl_week['win_rate'] is not None else "Weekly: no closed trades yet")
        await query.edit_message_text(text, reply_markup=get_mobile_panel_menu())

    elif data == "show_open_positions":
        positions = market_data.get_open_positions(magic_number=778899)
        if not positions:
            text = "No open positions."
        else:
            lines = [f"{p['symbol']} {p['type']} {p['volume']} lots | Entry {p['price_open']:.5f} | PnL {p['profit']:.2f}" for p in positions]
            text = "OPEN POSITIONS:\n" + "\n".join(lines)
        await query.edit_message_text(text, reply_markup=get_mobile_panel_menu())

    elif data == "show_pattern_stats":
        stats = market_data.get_pattern_win_rates(magic_number=778899, days=30)
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
        req, status = ts_state.resolve_approval(approval_id, approved=(action == "approve"))
        if req is None:
            await query.edit_message_text("This approval request no longer exists.")
        elif status == "expired":
            await query.edit_message_text(f"⏱️ Setup expired before you responded:\n{req['summary']}")
        elif status == "approved":
            await query.edit_message_text(f"✅ Approved -- sending to EA:\n{req['summary']}")
        else:
            await query.edit_message_text(f"❌ Rejected:\n{req['summary']}")

    # ---------------- Help ----------------
    elif data == "menu_help":
        await query.edit_message_text(
            "ℹ️ HELP & GUIDE\n\n"
            "📐 Trendline -- classic rules (wicks, 2-3+ touches, retest, confirmation) "
            "wedge/triangle), entry/SL/TP from liquidity + measured-move "
            "targets. Full 4H → 1H → 30M top-down cascade: 4H sets the macro "
            "regime (200 EMA + structure), 1H grants/denies structure "
            "permission, 30M supplies the actual entry.\n\n"
            "🎯 OTE -- Fibonacci Fan + Expansion off the most recent clean "
            "impulse leg on the 30M chart, gated by the same 4H → 1H "
            "top-down bias.\n\n"
            "📱 Control Panel -- Master switch, Trade Mode (Auto/Approval/"
            "Mobile Manual), lot sizing, the watched-asset background "
            "scanner (for when you're away and no EA is connected), Account "
            "& PnL, Open Positions.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="menu_home")]])
        )


# ==========================================
# 4. NOTIFIER CALLBACKS (wired into execution_engine.py so it can push
#    Telegram messages without importing bot.py directly)
# ==========================================
def on_auto_fired(setup):
    fmt = f"{{:.{_decimals_for(setup['symbol'])}f}}"
    text = (f"🤖 AUTO-FIRED  |  {setup['symbol']}\n{direction_banner(setup['bias'])}\n"
            f"{setup['pattern_name']}\nEntry {fmt.format(setup['entry'])}  SL {fmt.format(setup['sl'])}  "
            f"TP1 {fmt.format(setup['tp1'])}  TP2 {fmt.format(setup['tp2'])}")
    push_telegram_message(text)


def on_approval_request(approval_id, setup):
    fmt = f"{{:.{_decimals_for(setup['symbol'])}f}}"
    text = (f"⏳ APPROVAL NEEDED  |  {setup['symbol']}\n{direction_banner(setup['bias'])}\n"
            f"{setup['pattern_name']}\nEntry {fmt.format(setup['entry'])}  SL {fmt.format(setup['sl'])}  "
            f"TP1 {fmt.format(setup['tp1'])}  TP2 {fmt.format(setup['tp2'])}")
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Approve", callback_data=f"approve|{approval_id}"),
        InlineKeyboardButton("❌ Reject", callback_data=f"reject|{approval_id}"),
    ]])
    push_telegram_message(text, reply_markup=kb)


def on_manual_ticket(setup):
    fmt = f"{{:.{_decimals_for(setup['symbol'])}f}}"
    text = (f"📲 MANUAL TICKET  |  {setup['symbol']}\n{direction_banner(setup['bias'])}\n"
            f"{setup['pattern_name']}\nEntry {fmt.format(setup['entry'])}  SL {fmt.format(setup['sl'])}  "
            f"TP1 {fmt.format(setup['tp1'])}  TP2 {fmt.format(setup['tp2'])}\n"
            f"{setup.get('note', '')}".rstrip())
    push_telegram_message(text)


def on_report_event(symbol, event_type, details):
    push_telegram_message(f"📣 {symbol} -- {event_type}: {details}")


engine.register_notifiers(auto_fired=on_auto_fired, approval_request=on_approval_request, manual_ticket=on_manual_ticket)
engine._notify_callbacks["report_event"] = on_report_event


# ==========================================
# 5. ENTRYPOINT -- runs Flask (for the EA) and Telegram polling together,
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
