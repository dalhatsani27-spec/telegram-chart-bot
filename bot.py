import asyncio
from telegram import Update
from telegram.ext import ContextTypes
from mt5_data import fetch_rates
from structure_engine import analyze_smc_realignment
import MetaTrader5 as mt5

def run_strategy_scan(symbol: str = "EURUSD"):
    """Synchronous worker function that executes analysis in a thread."""
    # Fetch M15 chart data for structural analysis
    df = fetch_rates(symbol, mt5.TIMEFRAME_M15, count=300)
    if df is None:
        return f"❌ Failed to fetch MT5 market data for {symbol}."
    
    return analyze_smc_realignment(df)


async def handle_strategy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Async Telegram handler with thread offloading and safety timeout."""
    query = update.callback_query
    if query:
        await query.answer()
        message_func = query.message.reply_text
    else:
        message_func = update.message.reply_text

    symbol = "EURUSD"
    status_msg = await message_func(f"🔎 Scanning {symbol} (SMC Structure)...")

    try:
        # Offload heavy calculation to background thread with 120s timeout window
        result = await asyncio.wait_for(
            asyncio.to_thread(run_strategy_scan, symbol),
            timeout=120.0
        )
        await status_msg.edit_text(result, parse_mode="Markdown")

    except asyncio.TimeoutError:
        await status_msg.edit_text(
            f"⏱️ Analysis timed out for {symbol} (data provider slow). Try again."
        )
    except Exception as e:
        await status_msg.edit_text(f"⚠️ Error running analysis: {str(e)}")
