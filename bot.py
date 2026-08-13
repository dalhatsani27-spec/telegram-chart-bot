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
def health_check(): return {"status": "ok"}, 200

def _keepalive_pinger():
    import requests as _requests
    base_url=os.environ.get("RENDER_EXTERNAL_URL")
    if not base_url: return
    ping_url=base_url.rstrip("/")+"/health"
    while True:
        time.sleep(600)
        try: _requests.get(ping_url,timeout=10)
        except Exception as e: print(f"[keepalive] ping failed: {e!r}")

def _decimals_for(symbol):
    if symbol in ["AAPL","TSLA","NVDA","MSFT","AMZN","GOOGL","META","XAUUSD","GOLD","US30","NAS100","SPX500"]: return 2
    if "JPY" in symbol: return 3
    return 5

def format_trade_ticket(symbol,strategy_name,direction,score,ticket,reasons=None):
    lines=["══════════════════════════",f"  TRADE TICKET  |  {symbol}","══════════════════════════",direction_banner(direction),f"Strategy  : {strategy_name}",f"Score     : {score}/100"]
    if ticket: lines += [f"Entry     : {ticket.get('entry','—')}",f"SL        : {ticket.get('sl','—')}",f"TP1       : {ticket.get('tp1','—')}",f"TP2       : {ticket.get('tp2','—')}",f"TP3       : {ticket.get('tp3','—')}"+(f"  ({ticket['tp3_basis']})" if ticket.get('tp3_basis') else ""),f"Order     : {ticket.get('order_type','MARKET')}"]
    if reasons: lines += ["Why       :"]+[f"  • {r}" for r in reasons[:6]]
    return "\n".join(lines+["══════════════════════════","Enter this trade manually on your phone."])

primary_chat_id=None; _analysis_mode={}; TELEGRAM_LOOP=None; TELEGRAM_APP=None; _pending_price_watch={}
ASSET_CONTAINER={"Forex Majors":["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","NZDUSD"],"Forex Crosses":["EURGBP","EURJPY","GBPJPY","GBPAUD","AUDJPY","EURAUD"],"Commodities":["XAUUSD","XAGUSD","OIL"],"Crypto":["BTCUSD","ETHUSD"],"Indices":["US30","NAS100","SPX500"],"Stocks":["AAPL","TSLA","NVDA","MSFT","AMZN","GOOGL","META"],"Deriv Synthetic":["Volatility 10 (1s)","Volatility 25 (1s)","Volatility 50 (1s)","Volatility 75 (1s)","Volatility 100 (1s)","Volatility 10","Volatility 25","Volatility 50","Volatility 75","Volatility 100"]}

def push_telegram_message(text,reply_markup=None):
    if TELEGRAM_LOOP is None or TELEGRAM_APP is None or primary_chat_id is None:return
    asyncio.run_coroutine_threadsafe(TELEGRAM_APP.bot.send_message(chat_id=primary_chat_id,text=text,reply_markup=reply_markup),TELEGRAM_LOOP)

def push_telegram_photo(photo,caption=""):
    if TELEGRAM_LOOP is None or TELEGRAM_APP is None or primary_chat_id is None:return
    asyncio.run_coroutine_threadsafe(TELEGRAM_APP.bot.send_photo(chat_id=primary_chat_id,photo=photo,caption=caption),TELEGRAM_LOOP)

def get_home_menu():
    s=ts_state.strategy_label(); return InlineKeyboardMarkup([[InlineKeyboardButton(f"🧠 STRATEGY: {s}",callback_data="menu_strategy")],[InlineKeyboardButton("📐  Trendline (4H→1H→30M)",callback_data="menu_trendline")],[InlineKeyboardButton("🎯  OTE (62–79%, 30M)",callback_data="menu_ote")],[InlineKeyboardButton("🏦  SMC Strategy",callback_data="menu_smc"),InlineKeyboardButton("🔀  Hybrid",callback_data="menu_hybrid")],[InlineKeyboardButton("👁  Watch Price Level",callback_data="menu_price_watch"),InlineKeyboardButton("📋  My Watch Levels",callback_data="show_price_watches")],[InlineKeyboardButton("🔎  Custom Ticker",callback_data="prompt_custom_ticker")],[InlineKeyboardButton("📱  CONTROL PANEL",callback_data="menu_mobile_panel")],[InlineKeyboardButton("ℹ️  HELP & GUIDE",callback_data="menu_help")]])

def get_strategy_menu():
    s=ts_state.get_selected_strategy(); return InlineKeyboardMarkup([[InlineKeyboardButton(f"{'✅' if s==engine.STRATEGY_TRENDLINE else '⚪'} Trendline",callback_data="set_strategy|TRENDLINE")],[InlineKeyboardButton(f"{'✅' if s==engine.STRATEGY_OTE else '⚪'} OTE (62–79%)",callback_data="set_strategy|OTE")],[InlineKeyboardButton("« Back",callback_data="menu_home")]])

def get_mobile_panel_menu():
    mode=ts_state.get_mode(); master="🟢 Master: ON" if mode!=engine.MODE_OFF else "🔴 Master: OFF"; trade="Trade Mode: AUTO (EA executes)" if mode==engine.MODE_AUTO else "Trade Mode: APPROVAL" if mode==engine.MODE_APPROVAL else "Trade Mode: MOBILE MANUAL (tickets only)" if mode==engine.MODE_COPY_TRADE else "Trade Mode: OFF"; copy="🟢 Mobile Manual Tickets: ON" if mode==engine.MODE_COPY_TRADE else "⚪ Mobile Manual Tickets: OFF"; watched=ts_state.get_watched_symbol(); watch=f"🎯 Watching: {watched} (tap to change)" if watched else "🎯 Select Asset to Watch"; tf=f"⏱ Entry Timeframe: {ts_state.watch_timeframe_label()}"
    return InlineKeyboardMarkup([[InlineKeyboardButton(master,callback_data="toggle_master")],[InlineKeyboardButton(trade,callback_data="toggle_trade_mode")],[InlineKeyboardButton(copy,callback_data="toggle_copy_trade")],[InlineKeyboardButton(f"🧠 {ts_state.strategy_label()}",callback_data="menu_strategy")],[InlineKeyboardButton(f"Lot Mode: {ts_state.lot_mode}",callback_data="toggle_lot_mode")],[InlineKeyboardButton(watch,callback_data="menu_watch_asset")],[InlineKeyboardButton(tf,callback_data="menu_watch_tf")],[InlineKeyboardButton("💰 Account & PnL",callback_data="show_account_pnl"),InlineKeyboardButton("📈 Open Positions",callback_data="show_open_positions")],[InlineKeyboardButton("« Back",callback_data="menu_home")]])

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
    await update.message.reply_text("══════════════════════════════════\n  TOP-DOWN PRICE-ACTION ENGINE\n══════════════════════════════════\n\nTrendline Structure  •  OTE 62–79% Retracement\n4H → 1H → 30M Top-Down Bias  •  200 EMA Regime\nStructure Permission  •  MT5 Execution\n\n──────────────────────────────\nMaster switch is OFF by default.\nEnable trading from the Control Panel\nwhen you are ready.\n──────────────────────────────",reply_markup=get_home_menu())

PRICE_WATCH_SCAN_INTERVAL_SECONDS=2
async def background_price_watch_scanner():
    await asyncio.sleep(5)
    while True:
        try:
            watches=ts_state.get_watch_levels(); symbols=sorted({w["symbol"] for w in watches if not w.get("triggered")})
            for symbol in symbols:
                try:
                    price=await asyncio.to_thread(market_data.fetch_watch_price,symbol)
                    if price is None: continue
                    for w in watches:
                        if w["symbol"]!=symbol or w.get("triggered"): continue
                        event=ts_state.update_watch_level(w,price)
                        if event: push_telegram_message(f"🔔 WATCH LEVEL ALERT — {symbol}\nLevel: {w['level']}\nPrice: {price}\nEvent: {event.replace('_',' ')}")
                except Exception as e: print(f"[price_watch] {symbol}: {e!r}")
        except Exception as e: print(f"[price_watch] scanner failure: {e!r}")
        await asyncio.sleep(PRICE_WATCH_SCAN_INTERVAL_SECONDS)

WATCHLIST_SCAN_INTERVAL_SECONDS=300; _last_scan_error_notified={"symbol":None,"at":0}
async def background_watchlist_scanner():
    await asyncio.sleep(15)
    while True:
        symbol=None
        try:
            symbol=ts_state.get_watched_symbol(); tf=ts_state.get_watch_timeframe()
            if symbol and ts_state.get_mode()!=engine.MODE_OFF and not ts_state.is_ea_available(symbol):
                try: engine.poll(symbol,tf)
                except Exception as e:
                    print(f"[watchlist_scanner] poll failed for {symbol} ({tf}): {e!r}"); now=time.time()
                    if _last_scan_error_notified["symbol"]!=symbol or now-_last_scan_error_notified["at"]>1800: _last_scan_error_notified.update(symbol=symbol,at=now); push_telegram_message(f"⚠️ Background scanner error on {symbol} ({tf}): {e}")
        except Exception as e: print(f"[watchlist_scanner] outer loop failure: {e!r}")
        await asyncio.sleep(WATCHLIST_SCAN_INTERVAL_SECONDS)

async def send_trendline_analysis(context,chat_id,symbol):
    try:
        family=strategies.run_trendline_analysis(symbol)
        if family.get("error"): await context.bot.send_message(chat_id=chat_id,text=strategies.format_trendline_report(family,symbol)); return
        try:
            df=family.get("df")
            if df is not None and not getattr(df,"empty",True):
                pos=strategies.build_position_container(family); payload=dict(family); payload["position"]=pos; await context.bot.send_photo(chat_id=chat_id,photo=generate_trendline_educational_map(df,symbol,payload,title_suffix=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),caption=f"{symbol} Trendline (30M)")
        except Exception: traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"⚠️ Chart failed to render for {symbol}.")
        await context.bot.send_message(chat_id=chat_id,text=strategies.format_trendline_report(family,symbol))
        await context.bot.send_message(chat_id=chat_id,text="Choose next action:",reply_markup=get_home_menu())
    except Exception as e: await context.bot.send_message(chat_id=chat_id,text=f"Trendline analysis failed for '{symbol}': {e}",reply_markup=get_home_menu())

async def send_ote_analysis(context,chat_id,symbol):
    try:
        a=strategies.run_ote_analysis(symbol)
        if a.get("error"): await context.bot.send_message(chat_id=chat_id,text=strategies.format_ote_report(a)); return
        try:
            df=a.get("df")
            if df is not None and not getattr(df,"empty",True): await context.bot.send_photo(chat_id=chat_id,photo=generate_ote_map(df,symbol,a,title_suffix=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),caption=f"{symbol} OTE (30M)")
        except Exception: traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"⚠️ Chart failed to render for {symbol}.")
        await context.bot.send_message(chat_id=chat_id,text=strategies.format_ote_report(a)); await context.bot.send_message(chat_id=chat_id,text="Choose next action:",reply_markup=get_home_menu())
    except Exception as e: await context.bot.send_message(chat_id=chat_id,text=f"OTE analysis failed for '{symbol}': {e}",reply_markup=get_home_menu())

async def send_smc_analysis(context,chat_id,symbol):
    try:
        a=smc_engine.run_smc_analysis(symbol)
        if not a.get("error"):
            try: await context.bot.send_photo(chat_id=chat_id,photo=smc_engine.generate_smc_chart(a),caption=f"{symbol} SMC M30 | {a.get('setup_type','—')} | {a.get('status','—')}")
            except Exception as chart_error: traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"⚠️ SMC chart failed for {symbol}: {chart_error}")
        await context.bot.send_message(chat_id=chat_id,text=smc_engine.format_smc_report(a),reply_markup=get_home_menu())
    except Exception as e: traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"SMC analysis failed: {e}",reply_markup=get_home_menu())

async def send_hybrid_analysis(context,chat_id,symbol):
    try:
        a=smc_engine.run_hybrid_analysis(symbol)
        if not a.get("error"):
            try:
                chart=smc_engine.generate_hybrid_chart(a)
                if chart is not None: await context.bot.send_photo(chat_id=chat_id,photo=chart,caption=f"{symbol} HYBRID M30 | {a.get('direction','WAIT')} | {a.get('status','WAIT')}")
            except Exception as chart_error: traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"⚠️ Hybrid chart failed for {symbol}: {chart_error}")
        await context.bot.send_message(chat_id=chat_id,text=smc_engine.format_hybrid_report(a),reply_markup=get_home_menu())
    except Exception as e: traceback.print_exc(); await context.bot.send_message(chat_id=chat_id,text=f"Hybrid analysis failed: {e}",reply_markup=get_home_menu())

async def handle_text_message(update:Update,context:ContextTypes.DEFAULT_TYPE):
    global primary_chat_id; chat_id=update.effective_chat.id; primary_chat_id=chat_id; raw=(update.message.text or "").strip()
    if not raw:return
    symbol=raw.upper()
    if _analysis_mode.get(chat_id)=="SMC": return await send_smc_analysis(context,chat_id,symbol)
    if _analysis_mode.get(chat_id)=="HYBRID": return await send_hybrid_analysis(context,chat_id,symbol)
    await update.message.reply_text(f"Running {ts_state.strategy_label()} analysis for {symbol}...")
    if ts_state.get_selected_strategy()==engine.STRATEGY_OTE: await send_ote_analysis(context,chat_id,symbol)
    else: await send_trendline_analysis(context,chat_id,symbol)

async def button_callback_handler(update:Update,context:ContextTypes.DEFAULT_TYPE):
    global primary_chat_id; q=update.callback_query; await q.answer(); data=q.data; chat_id=q.message.chat_id; primary_chat_id=chat_id
    if data=="menu_home": await q.edit_message_text(f"Strategy: {ts_state.strategy_label()}\nSelect a command:",reply_markup=get_home_menu())
    elif data=="menu_strategy": await q.edit_message_text(f"🧠 STRATEGY SELECTOR\n\nCurrent: {ts_state.strategy_label()}",reply_markup=get_strategy_menu())
    elif data.startswith("set_strategy|"):
        name=data.split("|",1)[1]; mapping={"TRENDLINE":engine.STRATEGY_TRENDLINE,"OTE":engine.STRATEGY_OTE};
        if name in mapping: ts_state.set_selected_strategy(mapping[name])
        await q.edit_message_text(f"Strategy set → {ts_state.strategy_label()}",reply_markup=get_strategy_menu())
    elif data=="menu_smc": _analysis_mode[chat_id]="SMC"; await q.edit_message_text("🏦 SMC\n\n4H → 1H → 30M\nLiquidity → Sweep → Inducement → CHoCH/MSS → BOS → Displacement → OB/FVG → Confirmation → Entry",reply_markup=get_category_keyboard("cat_smc","menu_home"))
    elif data.startswith("cat_smc|"):
        _,cat=data.split("|",1); await q.edit_message_text(f"🏦 SMC — {cat}",reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat,[]),"run_smc","menu_smc"))
    elif data.startswith("run_smc|"):
        _,symbol=data.split("|",1); _analysis_mode[chat_id]="SMC"; await q.edit_message_text(f"Running SMC analysis for {symbol}..."); await send_smc_analysis(context,chat_id,symbol)
    elif data=="menu_hybrid": _analysis_mode[chat_id]="HYBRID"; await q.edit_message_text("🔀 HYBRID\n\nSMC + Trendline. Agreement can confirm; conflict always means WAIT.",reply_markup=get_category_keyboard("cat_hybrid","menu_home"))
    elif data.startswith("cat_hybrid|"):
        _,cat=data.split("|",1); await q.edit_message_text(f"🔀 Hybrid — {cat}",reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat,[]),"run_hybrid","menu_hybrid"))
    elif data.startswith("run_hybrid|"):
        _,symbol=data.split("|",1); _analysis_mode[chat_id]="HYBRID"; await q.edit_message_text(f"Running Hybrid analysis for {symbol}..."); await send_hybrid_analysis(context,chat_id,symbol)
    elif data=="menu_trendline": await q.edit_message_text("📐 TRENDLINE (4H → 1H → 30M):",reply_markup=get_category_keyboard("cat_tl","menu_home"))
    elif data.startswith("cat_tl|"):
        _,cat=data.split("|",1); await q.edit_message_text(cat,reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat,[]),"run_tl","menu_trendline"))
    elif data.startswith("run_tl|"):
        _,symbol=data.split("|",1); ts_state.set_selected_strategy(engine.STRATEGY_TRENDLINE); await q.edit_message_text(f"Running Trendline analysis for {symbol}..."); await send_trendline_analysis(context,chat_id,symbol)
    elif data=="menu_ote": await q.edit_message_text("🎯 OTE (62–79%):",reply_markup=get_category_keyboard("cat_ote","menu_home"))
    elif data.startswith("cat_ote|"):
        _,cat=data.split("|",1); await q.edit_message_text(cat,reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat,[]),"run_ote","menu_ote"))
    elif data.startswith("run_ote|"):
        _,symbol=data.split("|",1); ts_state.set_selected_strategy(engine.STRATEGY_OTE); await q.edit_message_text(f"Running OTE analysis for {symbol}..."); await send_ote_analysis(context,chat_id,symbol)
    elif data=="menu_price_watch": await q.edit_message_text("👁 WATCH PRICE LEVEL\n\nChoose the market and asset.",reply_markup=get_category_keyboard("price_watch_cat","menu_home"))
    elif data.startswith("price_watch_cat|"):
        _,cat=data.split("|",1); await q.edit_message_text(f"{cat}",reply_markup=get_pairs_keyboard(ASSET_CONTAINER.get(cat,[]),"price_watch_set","menu_price_watch"))
    elif data.startswith("price_watch_set|"):
        _,symbol=data.split("|",1); _pending_price_watch[chat_id]=symbol.upper(); await q.edit_message_text(f"👁 Send the price level for {symbol}.",reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Cancel",callback_data="menu_price_watch")]]))
    elif data=="show_price_watches": await q.edit_message_text("📋 MY WATCH LEVELS\n\n"+"\n".join([f"• {w['symbol']} — {w['level']}" for w in ts_state.get_watch_levels()]) if ts_state.get_watch_levels() else "📋 MY WATCH LEVELS\n\nNone",reply_markup=get_home_menu())
    elif data=="clear_price_watches": ts_state.clear_watch_levels(); await q.edit_message_text("🗑 All personal price watches cleared.",reply_markup=get_home_menu())
    elif data=="menu_mobile_panel": await q.edit_message_text("📱 MOBILE CONTROL PANEL",reply_markup=get_mobile_panel_menu())
    elif data=="toggle_master": ts_state.set_mode(engine.MODE_OFF if ts_state.get_mode()!=engine.MODE_OFF else engine.MODE_AUTO); await q.edit_message_text("📱 MOBILE CONTROL PANEL",reply_markup=get_mobile_panel_menu())
    elif data=="toggle_trade_mode": ts_state.set_mode(engine.MODE_APPROVAL if ts_state.get_mode()==engine.MODE_AUTO else engine.MODE_AUTO); await q.edit_message_text("📱 MOBILE CONTROL PANEL",reply_markup=get_mobile_panel_menu())
    elif data=="toggle_copy_trade": ts_state.set_mode(engine.MODE_AUTO if ts_state.get_mode()==engine.MODE_COPY_TRADE else engine.MODE_COPY_TRADE); await q.edit_message_text("📱 MOBILE CONTROL PANEL",reply_markup=get_mobile_panel_menu())
    elif data=="toggle_lot_mode": ts_state.set_lot_mode("RISK" if ts_state.lot_mode=="MIN" else "MIN"); await q.edit_message_text("📱 MOBILE CONTROL PANEL",reply_markup=get_mobile_panel_menu())
    elif data=="show_account_pnl":
        acc=market_data.get_account_summary(); text="MT5 not reachable right now." if acc is None else f"BROKER: {acc['broker']} ({acc['server']})\nAccount: {acc['login']} | {acc['currency']}\nBalance: {acc['balance']:.2f} | Equity: {acc['equity']:.2f}\nFree Margin: {acc['margin_free']:.2f}"; await q.edit_message_text(text,reply_markup=get_mobile_panel_menu())
    elif data=="show_open_positions":
        pos=market_data.get_open_positions(magic_number=778899); await q.edit_message_text("No open positions." if not pos else "OPEN POSITIONS:\n"+"\n".join([f"{p['symbol']} {p['type']} {p['volume']} lots | PnL {p['profit']:.2f}" for p in pos]),reply_markup=get_mobile_panel_menu())
    elif data=="menu_help": await q.edit_message_text("ℹ️ HELP & GUIDE\n\nSMC uses liquidity sweep → inducement → CHoCH/MSS → BOS → displacement → OB/FVG → confirmation. Hybrid requires SMC + Trendline agreement.",reply_markup=get_home_menu())
    elif data=="prompt_custom_ticker": await q.edit_message_text(f"Type any ticker for {ts_state.strategy_label()} analysis.",reply_markup=get_home_menu())
    else: await q.edit_message_text("Choose an available command.",reply_markup=get_home_menu())

async def smc_command(update:Update,context:ContextTypes.DEFAULT_TYPE):
    global primary_chat_id; primary_chat_id=update.effective_chat.id; _analysis_mode[primary_chat_id]="SMC"; await update.message.reply_text("🏦 SMC mode selected. Send a symbol.",reply_markup=get_home_menu())

async def hybrid_command(update:Update,context:ContextTypes.DEFAULT_TYPE):
    global primary_chat_id; primary_chat_id=update.effective_chat.id; _analysis_mode[primary_chat_id]="HYBRID"; await update.message.reply_text("🔀 Hybrid mode selected. Send a symbol.",reply_markup=get_home_menu())

def on_auto_fired(setup): push_telegram_message(f"🤖 AUTO-FIRED | {setup['symbol']} | {setup['pattern_name']}")
def on_approval_request(approval_id,setup): push_telegram_message(f"⏳ APPROVAL NEEDED | {setup['symbol']} | {setup['pattern_name']}",reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("✅ Approve",callback_data=f"approve|{approval_id}"),InlineKeyboardButton("❌ Reject",callback_data=f"reject|{approval_id}")]]))
def on_manual_ticket(setup): push_telegram_message(f"📲 MANUAL TICKET | {setup['symbol']} | {setup['pattern_name']}")
def on_report_event(symbol,event_type,details): push_telegram_message(f"📣 {symbol} -- {event_type}: {details}")
engine.register_notifiers(auto_fired=on_auto_fired,approval_request=on_approval_request,manual_ticket=on_manual_ticket); engine._notify_callbacks["report_event"]=on_report_event

def run_flask(): app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),use_reloader=False,threaded=True)
def run_telegram_bot():
    global TELEGRAM_LOOP,TELEGRAM_APP
    if not TELEGRAM_BOT_TOKEN: print("TELEGRAM_BOT_TOKEN not set -- Telegram bot will not start."); return
    loop=asyncio.new_event_loop(); asyncio.set_event_loop(loop); TELEGRAM_LOOP=loop; app_bot=Application.builder().token(TELEGRAM_BOT_TOKEN).build(); TELEGRAM_APP=app_bot
    app_bot.add_handler(CommandHandler("start",start_command)); app_bot.add_handler(CommandHandler("smc",smc_command)); app_bot.add_handler(CommandHandler("hybrid",hybrid_command)); app_bot.add_handler(CallbackQueryHandler(button_callback_handler)); app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,handle_text_message)); loop.run_until_complete(app_bot.initialize()); loop.run_until_complete(app_bot.bot.set_my_commands([('start','Open Dashboard'),('smc','SMC analysis'),('hybrid','SMC + Trendline')])); loop.run_until_complete(app_bot.start()); loop.run_until_complete(app_bot.updater.start_polling(drop_pending_updates=True)); loop.create_task(background_watchlist_scanner()); loop.create_task(background_price_watch_scanner()); loop.run_forever()

if __name__=="__main__": threading.Thread(target=run_flask,daemon=True).start(); threading.Thread(target=_keepalive_pinger,daemon=True).start(); run_telegram_bot()
