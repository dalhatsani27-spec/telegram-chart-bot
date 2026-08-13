"""
bot.py
======
Telegram front-end. Two chart-analysis strategies (Trendline, OTE — both
full 4H -> 1H -> 30M top-down reads) plus the live-trading Control Panel
(Master switch, Auto/Approval/Mobile-Manual modes, EA heartbeat, lot
sizing, Account & PnL, Open Positions).
"""

import os
import time
import traceback
import threading
import asyncio
from datetime import datetime

from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

import market_data
from market_analysis import direction_banner
from chart_engine import generate_trendline_educational_map, generate_ote_map
import strategies
import smc_engine
import execution_engine as engine
from execution_engine import state as ts_state

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

app = Flask(__name__)
engine.register_routes(app)

@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok"}, 200


def _keepalive_pinger():
    import requests as _requests
    base_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not base_url:
        return
    ping_url = base_url.rstrip("/") + "/health"
    while True:
        time.sleep(600)
        try:
            _requests.get(ping_url, timeout=10)
        except Exception as e:
            print(f"[keepalive] ping failed: {e!r}")


def _decimals_for(symbol):
    if symbol in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "XAUUSD", "GOLD", "US30", "NAS100", "SPX500"]:
        return 2
    if "JPY" in symbol:
        return 3
    return 5


def format_trade_ticket(symbol, strategy_name, direction, score, ticket, reasons=None):
    lines=["══════════════════════════",f"  TRADE TICKET  |  {symbol}","══════════════════════════",direction_banner(direction),f"Strategy  : {strategy_name}",f"Score     : {score}/100"]
    if ticket:
        lines += [f"Entry     : {ticket.get('entry','—')}",f"SL        : {ticket.get('sl','—')}",f"TP1       : {ticket.get('tp1','—')}",f"TP2       : {ticket.get('tp2','—')}",f"TP3       : {ticket.get('tp3','—')}" + (f"  ({ticket['tp3_basis']})" if ticket.get('tp3_basis') else ""),f"Order     : {ticket.get('order_type','MARKET')}"]
    if reasons:
        lines.append("Why       :"); lines.extend([f"  • {r}" for r in reasons[:6]])
    lines += ["══════════════════════════","Enter this trade manually on your phone."]
    return "\n".join(lines)

primary_chat_id=None; _analysis_mode={}; TELEGRAM_LOOP=None; TELEGRAM_APP=None; _pending_price_watch={}

ASSET_CONTAINER={"Forex Majors":["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","NZDUSD"],"Forex Crosses":["EURGBP","EURJPY","GBPJPY","GBPAUD","AUDJPY","EURAUD"],"Commodities":["XAUUSD","XAGUSD","OIL"],"Crypto":["BTCUSD","ETHUSD"],"Indices":["US30","NAS100","SPX500"],"Stocks":["AAPL","TSLA","NVDA","MSFT","AMZN","GOOGL","META"],"Deriv Synthetic":["Volatility 10 (1s)","Volatility 25 (1s)","Volatility 50 (1s)","Volatility 75 (1s)","Volatility 100 (1s)","Volatility 10","Volatility 25","Volatility 50","Volatility 75","Volatility 100"]}


def push_telegram_message(text, reply_markup=None):
    if TELEGRAM_LOOP is None or TELEGRAM_APP is None or primary_chat_id is None: return
    coro=TELEGRAM_APP.bot.send_message(chat_id=primary_chat_id,text=text,reply_markup=reply_markup); asyncio.run_coroutine_threadsafe(coro,TELEGRAM_LOOP)


def push_telegram_photo(photo, caption=""):
    if TELEGRAM_LOOP is None or TELEGRAM_APP is None or primary_chat_id is None: return
    coro=TELEGRAM_APP.bot.send_photo(chat_id=primary_chat_id,photo=photo,caption=caption); asyncio.run_coroutine_threadsafe(coro,TELEGRAM_LOOP)


def get_home_menu():
    strat_label=ts_state.strategy_label(); keyboard=[[InlineKeyboardButton(f"🧠 STRATEGY: {strat_label}",callback_data="menu_strategy")],[InlineKeyboardButton("📐  Trendline (4H→1H→30M)",callback_data="menu_trendline")],[InlineKeyboardButton("🎯  OTE (62–79%, 30M)",callback_data="menu_ote")],[InlineKeyboardButton("🏦  SMC Strategy",callback_data="menu_smc"),InlineKeyboardButton("🔀  Hybrid",callback_data="menu_hybrid")],[InlineKeyboardButton("👁  Watch Price Level",callback_data="menu_price_watch"),InlineKeyboardButton("📋  My Watch Levels",callback_data="show_price_watches")],[InlineKeyboardButton("🔎  Custom Ticker",callback_data="prompt_custom_ticker")],[InlineKeyboardButton("📱  CONTROL PANEL",callback_data="menu_mobile_panel")],[InlineKeyboardButton("ℹ️  HELP & GUIDE",callback_data="menu_help")]]; return InlineKeyboardMarkup(keyboard)


def get_strategy_menu():
    selected=ts_state.get_selected_strategy(); keyboard=[[InlineKeyboardButton(f"{'✅' if selected==engine.STRATEGY_TRENDLINE else '⚪'} Trendline",callback_data="set_strategy|TRENDLINE")],[InlineKeyboardButton(f"{'✅' if selected==engine.STRATEGY_OTE else '⚪'} OTE (62–79%)",callback_data="set_strategy|OTE")],[InlineKeyboardButton("« Back",callback_data="menu_home")]]; return InlineKeyboardMarkup(keyboard)


def get_mobile_panel_menu():
    mode=ts_state.get_mode(); master_label="🟢 Master: ON" if mode!=engine.MODE_OFF else "🔴 Master: OFF"; trade_mode_label="Trade Mode: AUTO (EA executes)" if mode==engine.MODE_AUTO else "Trade Mode: APPROVAL" if mode==engine.MODE_APPROVAL else "Trade Mode: MOBILE MANUAL (tickets only)" if mode==engine.MODE_COPY_TRADE else "Trade Mode: OFF"; copy_active=mode==engine.MODE_COPY_TRADE; copy_label="🟢 Mobile Manual Tickets: ON" if copy_active else "⚪ Mobile Manual Tickets: OFF"; lot_label=f"Lot Mode: {ts_state.lot_mode}"; watched=ts_state.get_watched_symbol(); watch_label=f"🎯 Watching: {watched} (tap to change)" if watched else "🎯 Select Asset to Watch"; tf_label=f"⏱ Entry Timeframe: {ts_state.watch_timeframe_label()}"; strat_label=ts_state.strategy_label()
    keyboard=[[InlineKeyboardButton(master_label,callback_data="toggle_master")],[InlineKeyboardButton(trade_mode_label,callback_data="toggle_trade_mode")],[InlineKeyboardButton(copy_label,callback_data="toggle_copy_trade")],[InlineKeyboardButton(f"🧠 {strat_label}",callback_data="menu_strategy")],[InlineKeyboardButton(lot_label,callback_data="toggle_lot_mode")],[InlineKeyboardButton(watch_label,callback_data="menu_watch_asset")],[InlineKeyboardButton(tf_label,callback_data="menu_watch_tf")],[InlineKeyboardButton("💰 Account & PnL",callback_data="show_account_pnl"),InlineKeyboardButton("📈 Open Positions",callback_data="show_open_positions")],[InlineKeyboardButton("« Back",callback_data="menu_home")]]; return InlineKeyboardMarkup(keyboard)


def get_category_keyboard(prefix,back_callback,extra_row=None):
    kb=[[InlineKeyboardButton("🏠 Dashboard",callback_data="menu_home")],[InlineKeyboardButton("💱 Forex Majors",callback_data=f"{prefix}|Forex Majors"),InlineKeyboardButton("💱 Forex Crosses",callback_data=f"{prefix}|Forex Crosses")],[InlineKeyboardButton("🥇 Commodities",callback_data=f"{prefix}|Commodities"),InlineKeyboardButton("🪙 Crypto",callback_data=f"{prefix}|Crypto")],[InlineKeyboardButton("📈 Indices",callback_data=f"{prefix}|Indices"),InlineKeyboardButton("📈 Stocks",callback_data=f"{prefix}|Stocks")],[InlineKeyboardButton("⚡ Deriv Synthetic",callback_data=f"{prefix}|Deriv Synthetic")]]
    if extra_row: kb.append(extra_row)
    kb.append([InlineKeyboardButton("« Back",callback_data=back_callback)]); return InlineKeyboardMarkup(kb)


def get_pairs_keyboard(pairs,prefix,back_callback):
    kb=[[InlineKeyboardButton("🏠 Dashboard",callback_data="menu_home")]]; row=[]
    for pair in pairs:
        row.append(InlineKeyboardButton(pair,callback_data=f"{prefix}|{pair}"))
        if len(row)==2: kb.append(row); row=[]
    if row: kb.append(row)
    kb.append([InlineKeyboardButton("« Back",callback_data=back_callback)]); return InlineKeyboardMarkup(kb)


async def start_command(update:Update,context:ContextTypes.DEFAULT_TYPE):
    global primary_chat_id; primary_chat_id=update.effective_chat.id
    welcome=("══════════════════════════════════\n  TOP-DOWN PRICE-ACTION ENGINE\n══════════════════════════════════\n\nTrendline Structure  •  OTE 62–79% Retracement\n4H → 1H → 30M Top-Down Bias  •  200 EMA Regime\nStructure Permission  •  MT5 Execution\n\n──────────────────────────────\nMaster switch is OFF by default.\nEnable trading from the Control Panel\nwhen you are ready.\n──────────────────────────────")
    await update.message.reply_text(welcome,reply_markup=get_home_menu())


async def send_trendline_analysis(context,chat_id,symbol):
    ts_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        family=strategies.run_trendline_analysis(symbol)
        if family.get("error"):
            await context.bot.send_message(chat_id=chat_id,text=strategies.format_trendline_report(family,symbol)); await context.bot.send_message(chat_id=chat_id,text="Choose next action:",reply_markup=get_home_menu()); return
        report=strategies.format_trendline_report(family,symbol)
        try:
            df_tl=family.get("df")
            if df_tl is not None and not getattr(df_tl,"empty",True):
                pos=strategies.build_position_container(family); payload=dict(family); payload["position"]=pos; chart_img=generate_trendline_educational_map(df_tl,symbol,payload,title_suffix=ts_str)
                await context.bot.send_photo(chat_id=chat_id,photo=chart_img,caption=f"{symbol} Trendline (30M) | {ts_str}")
        except Exception:
            traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"⚠️ Chart failed to render for {symbol}. Check server logs.")
        await context.bot.send_message(chat_id=chat_id,text=report)
        if ts_state.get_mode()==engine.MODE_COPY_TRADE and family.get("direction") in ("BUY","SELL"):
            pos=strategies.build_position_container(family)
            if pos: await context.bot.send_message(chat_id=chat_id,text=format_trade_ticket(symbol,"TRENDLINE",family.get("direction"),family.get("strength",0),pos,reasons=family.get("gating_notes")))
        await context.bot.send_message(chat_id=chat_id,text="Choose next action:",reply_markup=get_home_menu())
    except Exception as e: await context.bot.send_message(chat_id=chat_id,text=f"Trendline analysis failed for '{symbol}': {str(e)}",reply_markup=get_home_menu())


async def send_ote_analysis(context,chat_id,symbol):
    ts_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        analysis=strategies.run_ote_analysis(symbol)
        if analysis.get("error"):
            await context.bot.send_message(chat_id=chat_id,text=strategies.format_ote_report(analysis)); await context.bot.send_message(chat_id=chat_id,text="Choose next action:",reply_markup=get_home_menu()); return
        report=strategies.format_ote_report(analysis)
        try:
            df_ote=analysis.get("df")
            if df_ote is not None and not getattr(df_ote,"empty",True): await context.bot.send_photo(chat_id=chat_id,photo=generate_ote_map(df_ote,symbol,analysis,title_suffix=ts_str),caption=f"{symbol} OTE (62–79% OTE, 30M) | {ts_str}")
        except Exception:
            traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"⚠️ Chart failed to render for {symbol}. Check server logs.")
        await context.bot.send_message(chat_id=chat_id,text=report)
        if ts_state.get_mode()==engine.MODE_COPY_TRADE and analysis.get("valid"):
            await context.bot.send_message(chat_id=chat_id,text=format_trade_ticket(symbol,"OTE",analysis.get("direction"),analysis.get("score",0),analysis.get("ticket"),reasons=analysis.get("reasons")))
        await context.bot.send_message(chat_id=chat_id,text="Choose next action:",reply_markup=get_home_menu())
    except Exception as e: await context.bot.send_message(chat_id=chat_id,text=f"OTE analysis failed for '{symbol}': {str(e)}",reply_markup=get_home_menu())


async def send_smc_analysis(context,chat_id,symbol):
    try:
        a=smc_engine.run_smc_analysis(symbol)
        if not a.get("error"):
            try:
                await context.bot.send_photo(chat_id=chat_id,photo=smc_engine.generate_smc_chart(a),caption=f"{symbol} SMC M30 | {a.get('setup_type','—')} | {a.get('status','—')}")
            except Exception as chart_error:
                traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"⚠️ SMC chart failed for {symbol}: {chart_error}")
        await context.bot.send_message(chat_id=chat_id,text=smc_engine.format_smc_report(a),reply_markup=get_home_menu())
    except Exception as e:
        traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"SMC analysis failed: {e}",reply_markup=get_home_menu())


async def send_hybrid_analysis(context,chat_id,symbol):
    try:
        a=smc_engine.run_hybrid_analysis(symbol)
        if not a.get("error"):
            try:
                chart=smc_engine.generate_hybrid_chart(a)
                if chart is not None:
                    await context.bot.send_photo(chat_id=chat_id,photo=chart,caption=f"{symbol} HYBRID M30 | {a.get('direction','WAIT')} | {a.get('status','WAIT')}")
            except Exception as chart_error:
                traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"⚠️ Hybrid chart failed for {symbol}: {chart_error}")
        await context.bot.send_message(chat_id=chat_id,text=smc_engine.format_hybrid_report(a),reply_markup=get_home_menu())
    except Exception as e:
        traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"Hybrid analysis failed: {e}",reply_markup=get_home_menu())


async def handle_text_message(update:Update,context:ContextTypes.DEFAULT_TYPE):
    global primary_chat_id; chat_id=update.effective_chat.id; primary_chat_id=chat_id; raw_text=(update.message.text or "").strip()
    if not raw_text:return
    if chat_id in _pending_price_watch:
        symbol=_pending_price_watch.pop(chat_id)
        try:
            level=float(raw_text.replace(",",""))
            if level<=0: raise ValueError
            ts_state.add_watch_level(symbol,level); await update.message.reply_text(f"👁 WATCH CREATED\n\n{symbol}\nLevel: {level:.10f}".rstrip("0").rstrip(".")+"\n\nI will alert you when price touches or crosses this level.",reply_markup=get_home_menu())
        except ValueError:
            _pending_price_watch[chat_id]=symbol; await update.message.reply_text("❌ Invalid price. Send a positive numeric price, e.g. 4385.50.")
        return
    symbol=raw_text.upper()
    if _analysis_mode.get(chat_id)=="SMC": await send_smc_analysis(context,chat_id,symbol); return
    if _analysis_mode.get(chat_id)=="HYBRID": await send_hybrid_analysis(context,chat_id,symbol); return
    await update.message.reply_text(f"Running {ts_state.strategy_label()} analysis for {symbol}...")
    if ts_state.get_selected_strategy()==engine.STRATEGY_OTE: await send_ote_analysis(context,chat_id,symbol)
    else: await send_trendline_analysis(context,chat_id,symbol)


async def button_callback_handler(update:Update,context:ContextTypes.DEFAULT_TYPE):
    global primary_chat_id; query=update.callback_query; await query.answer(); data=query.data; chat_id=query.message.chat_id; primary_chat_id=chat_id
    if data=="menu_home":
        await query.edit_message_text("══════════════════════════════════\n  TOP-DOWN PRICE-ACTION ENGINE\n══════════════════════════════════\n"+f"Strategy: {ts_state.strategy_label()}\nSelect a command:",reply_markup=get_home_menu())
    elif data=="menu_strategy":
        await query.edit_message_text("🧠 STRATEGY SELECTOR\n\nBoth strategies run the same 4H → 1H top-down bias read, then execute on the 30M chart.\n\n"+f"Current: {ts_state.strategy_label()}",reply_markup=get_strategy_menu())
    elif data.startswith("set_strategy|"):
        name=data.split("|",1)[1]; mapping={"TRENDLINE":engine.STRATEGY_TRENDLINE,"OTE":engine.STRATEGY_OTE}
        if name in mapping: ts_state.set_selected_strategy(mapping[name])
        await query.edit_message_text(f"Strategy set → {ts_state.strategy_label()}",reply_markup=get_strategy_menu())
    elif data=="menu_smc":
        _analysis_mode[chat_id]="SMC"; await query.edit_message_text("🏦 SMC STRATEGY\n\n4H → 1H → 30M\nLiquidity → Sweep → Inducement → CHoCH/MSS → BOS → Displacement → OB/FVG → Confirmation → Entry",reply_markup=get_category_keyboard("cat_smc","menu_home"))
    elif data.startswith("cat_smc|"):
        _,cat=data.split("|",1); await query.edit_message_text(f"🏦 SMC — {cat}",reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat,[]),"run_smc","menu_smc"))
    elif data.startswith("run_smc|"):
        _,symbol=data.split("|",1); _analysis_mode[chat_id]="SMC"; await query.edit_message_text(f"Running SMC analysis for {symbol}..."); await send_smc_analysis(context,chat_id,symbol)
    elif data=="menu_hybrid":
        _analysis_mode[chat_id]="HYBRID"; await query.edit_message_text("🔀 HYBRID\n\nSMC + Trendline. Agreement can confirm; conflict always means WAIT.",reply_markup=get_category_keyboard("cat_hybrid","menu_home"))
    elif data.startswith("cat_hybrid|"):
        _,cat=data.split("|",1); await query.edit_message_text(f"🔀 Hybrid — {cat}",reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat,[]),"run_hybrid","menu_hybrid"))
    elif data.startswith("run_hybrid|"):
        _,symbol=data.split("|",1); _analysis_mode[chat_id]="HYBRID"; await query.edit_message_text(f"Running Hybrid analysis for {symbol}..."); await send_hybrid_analysis(context,chat_id,symbol)
    elif data=="menu_trendline":
        extra=[InlineKeyboardButton("🔎 Custom Ticker",callback_data="prompt_custom_ticker_tl")]; await query.edit_message_text("📐 TRENDLINE (4H → 1H → 30M top-down):",reply_markup=get_category_keyboard("cat_tl","menu_home",extra_row=extra))
    elif data.startswith("cat_tl|"):
        _,cat=data.split("|",1); await query.edit_message_text(f"{cat}:",reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat,[]),"run_tl","menu_trendline"))
    elif data.startswith("run_tl|"):
        _,symbol=data.split("|",1); ts_state.set_selected_strategy(engine.STRATEGY_TRENDLINE); await query.edit_message_text(f"Running Trendline analysis for {symbol}..."); await send_trendline_analysis(context,chat_id,symbol)
    elif data=="menu_ote":
        extra=[InlineKeyboardButton("🔎 Custom Ticker",callback_data="prompt_custom_ticker_ote")]; await query.edit_message_text("🎯 OTE (62–79% retracement, 30M):",reply_markup=get_category_keyboard("cat_ote","menu_home",extra_row=extra))
    elif data.startswith("cat_ote|"):
        _,cat=data.split("|",1); await query.edit_message_text(f"{cat}:",reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat,[]),"run_ote","menu_ote"))
    elif data.startswith("run_ote|"):
        _,symbol=data.split("|",1); ts_state.set_selected_strategy(engine.STRATEGY_OTE); await query.edit_message_text(f"Running OTE analysis for {symbol}..."); await send_ote_analysis(context,chat_id,symbol)
    elif data=="prompt_custom_ticker":
        _analysis_mode[chat_id]="TRENDLINE"; await query.edit_message_text("Send a ticker symbol, e.g. EURUSD, XAUUSD, BTCUSD.")
    elif data=="prompt_custom_ticker_tl":
        _analysis_mode[chat_id]="TRENDLINE"; await query.edit_message_text("Send a ticker symbol for Trendline analysis.")
    elif data=="prompt_custom_ticker_ote":
        _analysis_mode[chat_id]="OTE"; await query.edit_message_text("Send a ticker symbol for OTE analysis.")
    else:
        await query.edit_message_text("Choose an available command from the dashboard.",reply_markup=get_home_menu())


def main():
    global TELEGRAM_LOOP,TELEGRAM_APP
    if not TELEGRAM_BOT_TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    application=Application.builder().token(TELEGRAM_BOT_TOKEN).build(); TELEGRAM_APP=application
    application.add_handler(CommandHandler("start",start_command)); application.add_handler(CallbackQueryHandler(button_callback_handler)); application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,handle_text_message))
    TELEGRAM_LOOP=asyncio.get_event_loop(); application.run_polling(drop_pending_updates=True)

if __name__=="__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0",port=int(os.environ.get("PORT","10000")),debug=False,use_reloader=False),daemon=True).start()
    threading.Thread(target=_keepalive_pinger,daemon=True).start(); main()
