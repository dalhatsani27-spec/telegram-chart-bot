import numpy as np
import pandas as pd

def find_fractal_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Detects structural fractal highs and lows."""
    highs, lows = df['High'].values, df['Low'].values
    n = len(df)
    p_highs, p_lows = [], []

    for i in range(left, n - right):
        w_h = highs[i - left : i + right + 1]
        w_l = lows[i - left : i + right + 1]
        if highs[i] == w_h.max() and np.argmax(w_h) == left:
            p_highs.append((i, highs[i]))
        if lows[i] == w_l.min() and np.argmin(w_l) == left:
            p_lows.append((i, lows[i]))

    return p_highs, p_lows


def analyze_smc_realignment(df: pd.DataFrame) -> str:
    """
    Scans for institutional SMC setups:
    1. Liquidity Sweep (High/Low hunt)
    2. Change of Character (CHoCH)
    3. Inducement Retracement Entry
    4. Target Opposing Liquidity
    """
    if df is None or len(df) < 50:
        return "⚠️ Insufficient price data for structure analysis."

    p_highs, p_lows = find_fractal_pivots(df, left=3, right=3)
    if len(p_highs) < 2 or len(p_lows) < 2:
        return "ℹ️ Building market structure... No pivot sequence detected yet."

    atr = float((df['High'] - df['Low']).tail(14).mean())

    # =========================================================================
    # 1. BEARISH ENTRY SCAN (Buy-Side Sweep -> Bearish CHoCH -> Sell)
    # =========================================================================
    major_high_idx, major_high_price = p_highs[-2] if len(p_highs) >= 2 else p_highs[-1]
    post_high_df = df.iloc[major_high_idx + 1:]

    if not post_high_df.empty and post_high_df['High'].max() > major_high_price:
        sweep_max = post_high_df['High'].max()
        sweep_bar = post_high_df[post_high_df['High'] == sweep_max].iloc[0]

        # Liquidity Sweep condition: Wick pierces high, but body closes below/near
        if sweep_bar['Close'] < major_high_price + (0.25 * atr):
            post_sweep_lows = [l for idx, l in p_lows if idx > major_high_idx]

            if post_sweep_lows:
                choch_level = min(post_sweep_lows)

                if post_high_df['Low'].min() < choch_level:
                    # Inducement Premium Entry (50% retracement of sweep move)
                    entry_zone = float(choch_level + (sweep_max - choch_level) * 0.50)
                    stop_loss = float(sweep_max + (0.1 * atr))
                    target_tp = float(min([l for _, l in p_lows[-4:]]))

                    return (
                        f"🎯 **BEARISH REVERSAL SETUP**\n"
                        f"----------------------------------------\n"
                        f"• **Buy-Side Sweep:** {sweep_max:.5f}\n"
                        f"• **CHoCH Shift:** {choch_level:.5f} (BEARISH)\n"
                        f"• **Inducement Entry Zone:** {entry_zone:.5f}\n"
                        f"• **Stop Loss:** {stop_loss:.5f}\n"
                        f"• **Target TP (Liquidity):** {target_tp:.5f}\n"
                        f"----------------------------------------\n"
                        f"📌 **Action:** Buy-side liquidity swept. Wait for price "
                        f"to retrace into {entry_zone:.5f} for sell entry."
                    )

    # =========================================================================
    # 2. BULLISH ENTRY SCAN (Sell-Side Sweep -> Bullish CHoCH -> Buy)
    # =========================================================================
    major_low_idx, major_low_price = p_lows[-2] if len(p_lows) >= 2 else p_lows[-1]
    post_low_df = df.iloc[major_low_idx + 1:]

    if not post_low_df.empty and post_low_df['Low'].min() < major_low_price:
        sweep_min = post_low_df['Low'].min()
        sweep_bar_b = post_low_df[post_low_df['Low'] == sweep_min].iloc[0]

        if sweep_bar_b['Close'] > major_low_price - (0.25 * atr):
            post_sweep_highs = [h for idx, h in p_highs if idx > major_low_idx]

            if post_sweep_highs:
                choch_level_b = max(post_sweep_highs)

                if post_low_df['High'].max() > choch_level_b:
                    entry_zone_b = float(sweep_min + (choch_level_b - sweep_min) * 0.50)
                    stop_loss_b = float(sweep_min - (0.1 * atr))
                    target_tp_b = float(max([h for _, h in p_highs[-4:]]))

                    return (
                        f"🚀 **BULLISH REVERSAL SETUP**\n"
                        f"----------------------------------------\n"
                        f"• **Sell-Side Sweep:** {sweep_min:.5f}\n"
                        f"• **CHoCH Shift:** {choch_level_b:.5f} (BULLISH)\n"
                        f"• **Inducement Entry Zone:** {entry_zone_b:.5f}\n"
                        f"• **Stop Loss:** {stop_loss_b:.5f}\n"
                        f"• **Target TP (Liquidity):** {target_tp_b:.5f}\n"
                        f"----------------------------------------\n"
                        f"📌 **Action:** Sell-side liquidity swept. Wait for pullback "
                        f"into {entry_zone_b:.5f} for buy entry."
                    )

    return "SB / SMC Analysis: No completed Sweep + CHoCH sequence detected. Setup status: WAIT."
