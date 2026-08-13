"""SMC-enabled launcher for telegram-chart-bot.

Keeps the existing bot.py intact and layers SMC + market-phase analysis into
its Telegram UI. Run this file instead of bot.py in deployment/start commands.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime

import bot as legacy
import market_data
import smc_strategy
import market_phases
from chart_engine import generate_smc_map


SMC_SELECTED = False


def _smc_home_menu():
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 SMC — Smart Money Concepts", callback_data="menu_smc")],
        [InlineKeyboardButton("📐 Trendline (4H→1H→30M)", callback_data="menu_trendline")],
        [InlineKeyboardButton("🎯 OTE (Fib Fan+Exp, 30M)", callback_data="menu_ote")],
        [InlineKeyboardButton("📊 Market Phase & Sentiment", callback_data="menu_phase")],
        [InlineKeyboardButton("🔎 Custom Ticker", callback_data="prompt_custom_ticker")],
        [InlineKeyboardButton("📱 CONTROL PANEL", callback_data="menu_mobile_panel")],
        [InlineKeyboardButton("ℹ️ HELP & GUIDE", callback_data="menu_help")],
    ])


def _smc_strategy_menu():
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 SMC", callback_data="set_strategy|SMC")],
        [InlineKeyboardButton("📐 Trendline", callback_data="set_strategy|TRENDLINE")],
        [InlineKeyboardButton("🎯 OTE", callback_data="set_strategy|OTE")],
        [InlineKeyboardButton("« Back", callback_data="menu_home")],
    ])


legacy.get_home_menu = _smc_home_menu
legacy.get_strategy_menu = _smc_strategy_menu


async def send_smc_analysis(context, chat_id: int, symbol: str):
    """Run SMC on 30M with 4H context and send a complete market map."""
    global SMC_SELECTED
    SMC_SELECTED = True
    try:
        df = await asyncio.to_thread(market_data.fetch_candles, symbol, "30min", 300)
        htf = await asyncio.to_thread(market_data.fetch_candles, symbol, "4h", 200)
        if df is None or df.empty:
            raise RuntimeError("No market data returned for this symbol.")
        result = await asyncio.to_thread(smc_strategy.analyse_smc, df, htf)
        phase = await asyncio.to_thread(market_phases.analyze_market_phase, df)
        report = smc_strategy.format_smc_report(result, symbol, "30M")
        phase_report = market_phases.format_phase_report(phase)

        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            if not result.get("error"):
                chart_img = await asyncio.to_thread(generate_smc_map, df, symbol, result, ts_str)
                await context.bot.send_photo(
                    chat_id=chat_id, photo=chart_img,
                    caption=f"{symbol} SMC (30M) | {ts_str}",
                )
        except Exception:
            await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Chart failed to render for {symbol} (sending text analysis only).")

        await context.bot.send_message(chat_id=chat_id, text=report)
        await context.bot.send_message(
            chat_id=chat_id,
            text=phase_report,
            reply_markup=_smc_home_menu(),
        )
    except Exception as exc:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"SMC analysis failed for '{symbol}': {exc}",
            reply_markup=_smc_home_menu(),
        )


async def send_phase_analysis(context, chat_id: int, symbol: str):
    try:
        df = await asyncio.to_thread(market_data.fetch_candles, symbol, "30min", 300)
        if df is None or df.empty:
            raise RuntimeError("No market data returned for this symbol.")
        phase = await asyncio.to_thread(market_phases.analyze_market_phase, df)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"📊 {symbol} — 30M\n\n{market_phases.format_phase_report(phase)}",
            reply_markup=_smc_home_menu(),
        )
    except Exception as exc:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Market phase analysis failed for '{symbol}': {exc}",
            reply_markup=_smc_home_menu(),
        )


_original_callback = legacy.button_callback_handler
_original_text = legacy.handle_text_message


async def smc_callback(update, context):
    global SMC_SELECTED
    data = update.callback_query.data
    query = update.callback_query
    chat_id = query.message.chat_id
    if data == "menu_smc":
        SMC_SELECTED = True
        await query.answer()
        await query.edit_message_text(
            "🧠 SMART MONEY CONCEPTS\n\n"
            "Institutional sequence — not an isolated indicator.\n\n"
            "Liquidity → Sweep → Displacement → BOS/CHOCH → "
            "OB/FVG → Premium/Discount → Liquidity targets.\n\n"
            "Choose an asset:",
            reply_markup=legacy.get_category_keyboard("cat_smc", "menu_home", extra_row=[
                __import__('telegram').InlineKeyboardButton("🔎 Custom Ticker", callback_data="prompt_custom_ticker_smc")
            ]),
        )
        return
    if data.startswith("cat_smc|"):
        await query.answer()
        cat = data.split("|", 1)[1]
        await query.edit_message_text(
            f"🧠 SMC — {cat}",
            reply_markup=legacy.get_pairs_keyboard(legacy.ASSET_CONTAINER.get(cat, []), "run_smc", "menu_smc"),
        )
        return
    if data.startswith("run_smc|"):
        await query.answer()
        symbol = data.split("|", 1)[1]
        await query.edit_message_text(f"🧠 Running SMC + Market Phase analysis for {symbol}...")
        await send_smc_analysis(context, chat_id, symbol)
        return
    if data == "menu_phase":
        SMC_SELECTED = True
        await query.answer()
        await query.edit_message_text(
            "📊 MARKET PHASE & SENTIMENT\n\n"
            "Detects observable accumulation, distribution, markup, markdown, "
            "liquidity sweeps, compression and expansion.\n\nChoose an asset:",
            reply_markup=legacy.get_category_keyboard("cat_phase", "menu_home"),
        )
        return
    if data.startswith("cat_phase|"):
        await query.answer()
        cat = data.split("|", 1)[1]
        await query.edit_message_text(
            f"📊 Phase Engine — {cat}",
            reply_markup=legacy.get_pairs_keyboard(legacy.ASSET_CONTAINER.get(cat, []), "run_phase", "menu_phase"),
        )
        return
    if data.startswith("run_phase|"):
        await query.answer()
        symbol = data.split("|", 1)[1]
        await query.edit_message_text(f"📊 Reading market phase for {symbol}...")
        await send_phase_analysis(context, chat_id, symbol)
        return
    if data == "set_strategy|SMC":
        SMC_SELECTED = True
        await query.answer()
        await query.edit_message_text("Strategy set → 🧠 SMC", reply_markup=_smc_strategy_menu())
        return
    if data == "prompt_custom_ticker_smc":
        SMC_SELECTED = True
        await query.answer()
        await query.edit_message_text("Type any ticker for SMC analysis.")
        return
    await _original_callback(update, context)


async def smc_text(update, context):
    global SMC_SELECTED
    if SMC_SELECTED:
        symbol = (update.message.text or "").strip().upper()
        if symbol:
            await update.message.reply_text(f"🧠 Running SMC + Market Phase analysis for {symbol}...")
            await send_smc_analysis(context, update.effective_chat.id, symbol)
            return
    await _original_text(update, context)


legacy.button_callback_handler = smc_callback
legacy.handle_text_message = smc_text


if __name__ == "__main__":
    threading.Thread(target=legacy.run_flask, daemon=True).start()
    threading.Thread(target=legacy._keepalive_pinger, daemon=True).start()
    legacy.run_telegram_bot()
