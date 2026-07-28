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
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from openai import OpenAI

# ==========================================
# 1. FLASK WEB SERVER & KEEP-ALIVE LOOP
# ==========================================
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Institutional ICT Engine (Adaptive Liquidity & Structure) Active 24/7", 200

RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MY_CHAT_ID = os.environ.get("MY_CHAT_ID")

def keep_alive_ping():
    """Keeps the service alive continuously without sleeping."""
    while True:
        time.sleep(600)
        if RENDER_EXTERNAL_URL:
            try:
                res = requests.get(RENDER_EXTERNAL_URL, timeout=10)
                print(f"[Keep-Alive] Ping status: {res.status_code}")
            except Exception as e:
                print(f"[Keep-Alive] Ping failed: {e}")

threading.Thread(target=keep_alive_ping, daemon=True).start()

# ==========================================
# 2. OPENROUTER / AI ENGINE SETUP
# ==========================================
ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def fetch_ai_analysis(prompt):
    """Executes precision model chain for strict market delivery analysis."""
    vision_models = [
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "openrouter/free"
    ]
    
    for model in vision_models:
        try:
            response = ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700
            )
            content = response.choices[0].message.content
            if content and "User Safety:" not in content and len(content.strip()) > 30:
                return content
        except Exception:
            continue
            
    return (
        "⚠️ *INSTITUTIONAL MARKET REPORT*\n\n"
        "1. **Trend & Macro Filter:** Daily 200 EMA Baseline Active.\n"
        "2. **Structure:** Reference mapped Unmitigated High-Probability OBs and MSS/CHoCH lines on chart.\n"
        "3. **Execution Rule:** Trade strictly in direction of local and daily 200 EMA alignment."
    )

# ==========================================
# 3. TICKER ALIAS & DATA ENGINE
# ==========================================
def normalize_ticker(symbol):
    """Maps shorthand terms to reliable yfinance futures/spot feeds."""
    symbol = symbol.strip().upper()
    alias_map = {
        "GOLD": "GC=F",      
        "XAUUSD": "GC=F",    
        "SILVER": "SI=F",
        "XAGUSD": "SI=F",
        "OIL": "CL=F",
        "USOIL": "CL=F",
        "BTC": "BTC-USD",
        "BTCUSD": "BTC-USD"
    }
    if symbol in alias_map:
        return alias_map[symbol]
    if len(symbol) == 6 and not symbol.endswith("=X"):
        return f"{symbol}=X"
    return symbol

def fetch_multi_timeframe_data(symbol, entry_tf="15m"):
    """Fetches Daily macro baseline alongside execution timeframe data cleanly with fallbacks."""
    primary_ticker = normalize_ticker(symbol)
    ticker_candidates = [primary_ticker]
    if primary_ticker == "GC=F":
        ticker_candidates.append("XAUUSD=X")
    elif primary_ticker == "XAUUSD=X":
        ticker_candidates.insert(0, "GC=F")

    tf_clean = entry_tf.lower().strip()
    entry_period = "5d" if tf_clean in ["1m", "5m"] else "10d"

    df_daily, df_entry = pd.DataFrame(), pd.DataFrame()
    successful_ticker = primary_ticker

    for ticker_str in ticker_candidates:
        try:
            d_data = yf.download(ticker_str, period="1y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(d_data.columns, pd.MultiIndex):
                d_data.columns = d_data.columns.get_level_values(0)
            d_data = d_data.dropna()

            e_data = yf.download(ticker_str, period=entry_period, interval=tf_clean, progress=False, auto_adjust=True)
            if isinstance(e_data.columns, pd.MultiIndex):
                e_data.columns = e_data.columns.get_level_values(0)
            e_data = e_data.dropna()

            if not d_data.empty and not e_data.empty:
                df_daily = d_data
                df_entry = e_data
                successful_ticker = ticker_str
                break
        except Exception:
            continue

    if df_daily.empty or df_entry.empty:
        raise ValueError(f"Data feed dropped for symbol '{symbol}'. Verify ticker or market hours.")

    df_daily['EMA200'] = df_daily['Close'].ewm(span=200, adjust=False).mean()
    daily_close = df_daily['Close'].iloc[-1]
    daily_ema = df_daily['EMA200'].iloc[-1]
    macro_is_bearish = daily_close < daily_ema

    df_entry['EMA200'] = df_entry['Close'].ewm(span=200, adjust=False).mean()

    return df_daily, df_entry, macro_is_bearish, daily_ema, successful_ticker

# ==========================================
# 4. ADAPTIVE ICT STRUCTURAL ENGINE
# ==========================================
def analyze_ict_structure(df):
    """
    Calculates Liquidity Sweeps, MSS/CHoCH, FVGs, and Unmitigated High-Probability OBs.
    """
    results = {
        'swings': [],
        'shifts': [],      # MSS / CHoCH
        'sweeps': [],      # BSL/SSL Sweeps
        'obs': [],         # Unmitigated / Mitigated Order Blocks
        'fvgs': []         # Fair Value Gaps
    }
    
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    
    swing_highs = []
    swing_lows = []
    
    # Identify Swing Fractality
    for i in range(2, len(df) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append({'index': i, 'price': highs[i]})
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append({'index': i, 'price': lows[i]})
            
    # Detect Sweeps vs. Structure Shifts
    last_sh = swing_highs[-1] if swing_highs else None
    last_sl = swing_lows[-1] if swing_lows else None
    
    for i in range(5, len(df)):
        # Bearish Structure Check
        if last_sl and i > last_sl['index']:
            # Liquidity Sweep: Wick pokes below swing low, but body closes inside
            if lows[i] < last_sl['price'] and closes[i] >= last_sl['price']:
                results['sweeps'].append({'type': 'SSL SWEEP', 'index': i, 'price': last_sl['price']})
            # True MSS/CHoCH: Full candle body displacement close below swing low
            elif closes[i] < last_sl['price'] and closes[i-1] >= last_sl['price']:
                results['shifts'].append({'type': 'BEARISH MSS', 'index': i, 'price': last_sl['price']})
                
        # Bullish Structure Check
        if last_sh and i > last_sh['index']:
            # Liquidity Sweep
            if highs[i] > last_sh['price'] and closes[i] <= last_sh['price']:
                results['sweeps'].append({'type': 'BSL SWEEP', 'index': i, 'price': last_sh['price']})
            # True MSS/CHoCH
            elif closes[i] > last_sh['price'] and closes[i-1] <= last_sh['price']:
                results['shifts'].append({'type': 'BULLISH MSS', 'index': i, 'price': last_sh['price']})

    # Fair Value Gap Detection
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            results['fvgs'].append({'type': 'BULLISH_FVG', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'index': i})
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            results['fvgs'].append({'type': 'BEARISH_FVG', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'index': i})

    # High-Displacement Order Blocks + Mitigation Logic
    for i in range(5, len(df) - 1):
        body_size = abs(closes[i] - opens[i])
        avg_body = abs(closes[i-5:i] - opens[i-5:i]).mean()
        
        if body_size > (1.35 * avg_body):
            # Bullish Order Block
            if closes[i] > opens[i]:
                for j in range(i-1, max(0, i-4), -1):
                    if closes[j] < opens[j]:
                        ob_top = max(opens[j], closes[j])
                        ob_bottom = lows[j]
                        
                        future_lows = df['Low'].iloc[j+1:]
                        is_mitigated = (future_lows < ob_top).any()
                        has_fvg = any(f['type'] == 'BULLISH_FVG' and f['index'] >= j for f in results['fvgs'])
                        
                        results['obs'].append({
                            'type': 'BULLISH_OB',
                            'top': ob_top,
                            'bottom': ob_bottom,
                            'index': j,
                            'mitigated': is_mitigated,
                            'high_prob': not is_mitigated and has_fvg
                        })
                        break

            # Bearish Order Block
            elif closes[i] < opens[i]:
                for j in range(i-1, max(0, i-4), -1):
                    if closes[j] > opens[j]:
                        ob_top = highs[j]
                        ob_bottom = min(opens[j], closes[j])
                        
                        future_highs = df['High'].iloc[j+1:]
                        is_mitigated = (future_highs > ob_bottom).any()
                        has_fvg = any(f['type'] == 'BEARISH_FVG' and f['index'] >= j for f in results['fvgs'])
                        
                        results['obs'].append({
                            'type': 'BEARISH_OB',
                            'top': ob_top,
                            'bottom': ob_bottom,
                            'index': j,
                            'mitigated': is_mitigated,
                            'high_prob': not is_mitigated and has_fvg
                        })
                        break

    return results, swing_highs, swing_lows

# ==========================================
# 5. HIGH-PRECISION CHART MAPPER
# ==========================================
def generate_chart_image(df, title_str, ict_data, swing_highs, swing_lows):
    img_buf = io.BytesIO()
    chart_df = df.tail(80).copy()
    
    mc = mpf.make_marketcolors(
        up='#089981', down='#f23645',
        edge='inherit', wick='inherit', volume='in'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', 
        y_on_right=True, facecolor='#131722', figcolor='#131722'
    )
    
    # Gold Line: 200 EMA Baseline
    addplots = [
        mpf.make_addplot(chart_df['EMA200'], color='#ffd700', width=1.5)
    ]

    fig, axlist = mpf.plot(
        chart_df,
        type='candle',
        style=style,
        volume=False,
        addplot=addplots,
        savefig=dict(fname=img_buf, dpi=130),
        returnfig=True
    )

    ax = axlist[0]

    # A. Draw Buy-Side / Sell-Side Liquidity Lines
    if swing_highs:
        last_bsl = swing_highs[-1]['price']
        ax.axhline(last_bsl, color='#00e676', linestyle=':', linewidth=1.0)
        ax.text(0, last_bsl, " BSL (Buy-Side Liquidity)", color='#00e676', fontsize=7, verticalalignment='bottom')

    if swing_lows:
        last_ssl = swing_lows[-1]['price']
        ax.axhline(last_ssl, color='#ff1744', linestyle=':', linewidth=1.0)
        ax.text(0, last_ssl, " SSL (Sell-Side Liquidity)", color='#ff1744', fontsize=7, verticalalignment='top')

    # B. Plot Structural Shifts (MSS)
    for shift in ict_data['shifts'][-1:]:
        color = '#00e676' if 'BULLISH' in shift['type'] else '#ff1744'
        ax.axhline(shift['price'], color=color, linestyle='--', linewidth=1.2)
        ax.text(len(chart_df)//2, shift['price'], f" {shift['type']} ", color=color, fontsize=8, fontweight='bold', backgroundcolor='#131722')

    # C. Plot UNMITIGATED High-Probability OBs
    unmitigated_obs = [ob for ob in ict_data['obs'] if not ob['mitigated']]
    display_obs = unmitigated_obs[-2:] if unmitigated_obs else ict_data['obs'][-2:]

    for ob in display_obs:
        if ob['type'] == 'BULLISH_OB':
            color = '#00e676' if ob['high_prob'] else '#089981'
            label = " HIGH-PROB UNMITIGATED OB" if ob['high_prob'] else " BULLISH OB"
            ax.axhspan(ob['bottom'], ob['top'], color=color, alpha=0.35 if not ob['mitigated'] else 0.10)
            ax.text(len(chart_df)-22, ob['top'], label, color=color, fontsize=8, fontweight='bold')
            
        elif ob['type'] == 'BEARISH_OB':
            color = '#ff1744' if ob['high_prob'] else '#f23645'
            label = " HIGH-PROB UNMITIGATED OB" if ob['high_prob'] else " BEARISH OB"
            ax.axhspan(ob['bottom'], ob['top'], color=color, alpha=0.35 if not ob['mitigated'] else 0.10)
            ax.text(len(chart_df)-22, ob['bottom'], label, color=color, fontsize=8, fontweight='bold')

    ax.set_title(title_str, color='#d1d4dc', fontsize=9, fontweight='bold')
    img_buf.seek(0)
    return img_buf

# ==========================================
# 6. TELEGRAM COMMAND HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🎯 *INSTITUTIONAL ICT CHART ANALYZER*\n\n"
        "Engine Active with High-Precision Algorithmic Detection:\n"
        "• **BSL / SSL Pools:** Live Buy-Side & Sell-Side Stop Pools\n"
        "• **MSS / CHoCH:** True Body Displacement Structural Breaks\n"
        "• **Unmitigated OBs:** Filtered High-Probability Entry Zones\n"
        "• **200 EMA Filter:** Mandatory Gold Baseline Overlay\n\n"
        "📌 *COMMAND:* `/analyze [SYMBOL] [TIMEFRAME]`\n"
        "_Example:_ `/analyze GOLD 15m`"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    
    raw_symbol = args[0].upper() if len(args) > 0 else "GOLD"
    tf = args[1] if len(args) > 1 else "15m"
    
    status_msg = await update.message.reply_text(f"⏳ Running Institutional Calculation for {raw_symbol}...")
    
    try:
        df_daily, df_entry, macro_is_bearish, daily_ema, ticker_str = fetch_multi_timeframe_data(raw_symbol, tf)
        
        last_price = df_entry['Close'].iloc[-1]
        local_ema = df_entry['EMA200'].iloc[-1]
        local_is_bearish = last_price < local_ema
        
        ict_data, swing_highs, swing_lows = analyze_ict_structure(df_entry)

        macro_bias = "BEARISH" if macro_is_bearish else "BULLISH"
        local_bias = "BEARISH" if local_is_bearish else "BULLISH"

        unmitigated_count = len([ob for ob in ict_data['obs'] if not ob['mitigated']])
        shift_status = ict_data['shifts'][-1]['type'] if ict_data['shifts'] else "No Structural Shift"
        sweep_status = ict_data['sweeps'][-1]['type'] if ict_data['sweeps'] else "No Recent Liquidity Sweep"

        # Direct, zero-fluff institutional prompt
        prompt = f"""
        Act as an uncompromising quantitative institutional ICT trader. Provide a precise, non-negotiable market breakdown.

        MARKET METRICS ({raw_symbol} - {tf}):
        - Daily 200 EMA Bias: {macro_bias} (Daily EMA: {daily_ema:.2f})
        - Execution Timeframe ({tf}) 200 EMA: {local_bias} (Current Price: {last_price:.2f}, Local EMA: {local_ema:.2f})
        - Structure State: {shift_status}
        - Liquidity State: {sweep_status}
        - Unmitigated High-Prob OBs: {unmitigated_count} Active

        CRITICAL EXECUTION RULE:
        If Price is BELOW 200 EMA, absolute sell bias only. Ignore long signals. Target short realignments at Bearish Unmitigated OBs.
        If Price is ABOVE 200 EMA, absolute buy bias only.

        Write structured report:
        🎯 INSTITUTIONAL ANALYSIS & MARKET STORY
        1. **Macro Baseline:** State Daily and Local 200 EMA Alignment.
        2. **Liquidity Target:** Identify target BSL or SSL liquidity pools.
        3. **High-Probability Zone:** Pinpoint exact Unmitigated OB or FVG price level.
        4. **Trade Execution Plan:** Explicit entry trigger requirement based on 200 EMA alignment.
        """

        analysis_text = fetch_ai_analysis(prompt)

        title_str = f"{raw_symbol} ({tf.upper()}) | Shift: {shift_status} | Gold Line: 200 EMA"
        chart_img = generate_chart_image(df_entry, title_str, ict_data, swing_highs, swing_lows)
        
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=chart_img,
            caption=f"🎯 *INSTITUTIONAL STRUCTURE MAP: {raw_symbol} ({tf.upper()})*",
            parse_mode="Markdown"
        )
        
        await status_msg.delete()
        await context.bot.send_message(chat_id=chat_id, text=analysis_text, parse_mode="Markdown")

    except Exception as e:
        await status_msg.edit_text(f"❌ Calculation error: {str(e)}")

# ==========================================
# 7. MAIN ENGINE RUNNER
# ==========================================
def run_telegram_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("analyze", analyze_command))

    loop.run_until_complete(application.initialize())
    loop.run_until_complete(application.start())
    loop.run_until_complete(application.updater.start_polling(drop_pending_updates=True))
    loop.run_forever()

if __name__ == "__main__":
    telegram_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    telegram_thread.start()

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
