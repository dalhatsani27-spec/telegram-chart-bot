"""
smc_strategy.py
================
Assembles smc_engine.py detectors + existing market_analysis/execution_engine
functions into the SMC trade model and report text, on the SINGLE selected
timeframe (no fixed 4H/1H/30M cascade -- matches the Trendline/OTE update).

Public entry points (mirror strategies.py's pattern):
  run_smc_analysis(symbol, tf_code)  -> dict
  format_smc_report(analysis)        -> str   (matches the sample schema)
  build_smc_ticket(analysis)         -> dict | None   (for Copy-Trade tickets)

Optional HTF context: call topdown_engine.get_topdown_bias(symbol) separately
and pass it in as `topdown=` -- this file does not fetch it automatically.
"""

from typing import Any, Dict, Optional

import market_data
from market_analysis import analyse_structure, detect_order_blocks, detect_confirmation_candle
from execution_engine import fib_discount_premium_zone
from smc_engine import detect_liquidity_pools, detect_fair_value_gaps, select_smc_zone

_TF_LABELS = {"1min": "M1", "3min": "M3", "5min": "M5", "15min": "M15",
              "30min": "M30", "1h": "H1", "4h": "H4"}


def run_smc_analysis(symbol: str, tf_code: str = "30min", topdown: Optional[Dict[str, Any]] = None,
                      df=None) -> Dict[str, Any]:
    if df is None:
        df = market_data.fetch_candles(symbol, tf_code, count=250)
    tf_label = _TF_LABELS.get(tf_code, tf_code)

    if df is None or df.empty or len(df) < 40:
        return {"error": f"Insufficient {tf_label} data for SMC analysis",
                "symbol": symbol, "timeframe": tf_code, "bias": "NEUTRAL"}

    structure = analyse_structure(df, left=3, right=3, lookback=100)
    bias_word = structure.get("bias", "NEUTRAL")           # BULLISH | BEARISH | NEUTRAL
    bias = "BUY" if bias_word == "BULLISH" else ("SELL" if bias_word == "BEARISH" else "NEUTRAL")

    liquidity = detect_liquidity_pools(df)
    fvgs = detect_fair_value_gaps(df)
    order_blocks = detect_order_blocks(df)

    zone = select_smc_zone(order_blocks, fvgs, bias) if bias != "NEUTRAL" else \
        {"ob": None, "fvg": None, "zone_top": None, "zone_bottom": None, "status": "NONE", "confluence": False}

    # Premium / discount location: use the same swept-range the liquidity
    # pools sit in (nearest buy-side high vs nearest sell-side low) as the
    # range, reusing fib_discount_premium_zone's math for consistency with OTE.
    close = float(df["Close"].iloc[-1])
    location = "EQUILIBRIUM"
    range_high = liquidity.get("buy_side", {}).get("level") if liquidity.get("buy_side") else None
    range_low = liquidity.get("sell_side", {}).get("level") if liquidity.get("sell_side") else None
    if range_high and range_low and range_high > range_low:
        pct = (close - range_low) / (range_high - range_low)
        location = "PREMIUM" if pct > 0.55 else ("DISCOUNT" if pct < 0.45 else "EQUILIBRIUM")

    # Candle confirmation at the zone
    candle_confirmed, candle_name = (False, None)
    price_in_zone = False
    if zone.get("zone_top") is not None:
        price_in_zone = zone["zone_bottom"] <= close <= zone["zone_top"] * 1.002 or \
                         abs(close - zone["zone_top"]) / max(close, 1e-9) < 0.003
        if bias != "NEUTRAL":
            candle_confirmed, candle_name = detect_confirmation_candle(df, bias)

    # Sweep status feeding the LIQUIDITY block (opposite-side pool swept
    # = the "trap" that precedes this bias, mirrors the reference image)
    if bias == "SELL":
        trap_pool = liquidity.get("buy_side")
        target_pool = liquidity.get("sell_side")
    elif bias == "BUY":
        trap_pool = liquidity.get("sell_side")
        target_pool = liquidity.get("buy_side")
    else:
        trap_pool, target_pool = None, None

    # ---- Trade model ----
    entry_ready = bool(zone.get("zone_top")) and price_in_zone and candle_confirmed
    status = "WAIT"
    sl = tp1 = tp2 = None
    if zone.get("zone_top") is not None and bias in ("BUY", "SELL"):
        if bias == "SELL":
            sl = zone["zone_top"] * 1.0015
            tp1 = target_pool["level"] if target_pool else None
            tp2 = structure.get("structure_low")
        else:
            sl = zone["zone_bottom"] * 0.9985
            tp1 = target_pool["level"] if target_pool else None
            tp2 = structure.get("structure_high")
        status = "SELL: WAITING FOR RETEST" if bias == "SELL" and not entry_ready else \
                 "BUY: WAITING FOR RETEST" if bias == "BUY" and not entry_ready else \
                 f"{bias} CONFIRMED"

    return {
        "symbol": symbol, "timeframe": tf_code, "timeframe_label": tf_label,
        "df": df, "close": close,
        "bias": bias, "bias_word": bias_word, "structure": structure,
        "location": location,
        "liquidity": liquidity, "trap_pool": trap_pool, "target_pool": target_pool,
        "fvgs": fvgs, "order_blocks": order_blocks, "zone": zone,
        "price_in_zone": price_in_zone,
        "candle_confirmed": candle_confirmed, "candle_name": candle_name,
        "entry": close if entry_ready else None,
        "entry_ready": entry_ready,
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "status": status, "topdown": topdown,
        "error": None,
    }


def _decimals_for(price):
    if price is None:
        return 2
    return 5 if price < 50 else (2 if price >= 500 else 4)


def _fmt(price, symbol=""):
    if price is None:
        return "—"
    return f"{price:.{_decimals_for(price)}f}"


def format_smc_report(analysis: Dict[str, Any]) -> str:
    if analysis.get("error"):
        return f"🧠 SMC ANALYSIS — {analysis.get('symbol', '?')}\n\n⚠️ {analysis['error']}"

    sym = analysis["symbol"]
    tf = analysis["timeframe_label"]
    bias = analysis["bias"]
    structure = analysis["structure"]
    liquidity = analysis["liquidity"]
    trap_pool = analysis["trap_pool"]
    target_pool = analysis["target_pool"]
    zone = analysis["zone"]
    topdown = analysis.get("topdown")

    bias_emoji = "🟢" if bias == "BUY" else ("🔴" if bias == "SELL" else "⚪")
    lines = [f"🧠 SMC ANALYSIS — {sym} ({tf})"]

    if topdown:
        td_dir = topdown.get("direction", "NEUTRAL")
        td_emoji = "🟢" if td_dir == "BUY" else ("🔴" if td_dir == "SELL" else "⚪")
        lines += [
            "",
            f"HTF BIAS: {td_emoji} {topdown.get('bias_4h', 'NEUTRAL')}",
        ]

    lines += [
        "",
        "STRUCTURE",
        f"{tf}: {bias_emoji} {structure.get('bias', 'NEUTRAL')}",
    ]
    if structure.get("last_event"):
        lines.append(f"Last event: {structure['last_event']} ({structure.get('event_bias')})")

    lines += ["", "LIQUIDITY"]
    if trap_pool:
        side_label = "BUY-SIDE" if trap_pool["type"] == "buy_side" else "SELL-SIDE"
        status_icon = "✅" if trap_pool["status"] == "SWEPT" else "⏳"
        lines.append(f"{side_label}: {trap_pool['status']} {status_icon} @ {_fmt(trap_pool['level'])}")
    if target_pool:
        side_label = "BUY-SIDE" if target_pool["type"] == "buy_side" else "SELL-SIDE"
        lines.append(f"TARGET ({side_label}): {_fmt(target_pool['level'])}")

    lines += [
        "",
        "STRUCTURAL EVENT",
        f"MSS: {'✅' if structure.get('last_event') == 'MSS' else '—'}",
        f"CHoCH: {'✅' if structure.get('last_event') == 'CHoCH' else '—'}",
        f"BOS: {'CONFIRMED ✅' if structure.get('last_event') == 'BOS' else '—'}",
    ]

    lines += ["", "ZONE"]
    if zone.get("ob") is not None:
        ob = zone["ob"]
        lines.append(f"{ob['type'].capitalize()} OB: {_fmt(ob['bottom'])}–{_fmt(ob['top'])}")
    if zone.get("fvg") is not None:
        g = zone["fvg"]
        lines.append(f"FVG: {_fmt(g['bottom'])}–{_fmt(g['top'])} ({g['fill_pct']}% filled)")
    if zone.get("zone_top") is None:
        lines.append("No qualifying OB/FVG zone found")
    else:
        lines.append(f"STATUS: {zone['status']}")
        if zone.get("confluence"):
            lines.append("Confluence: OB + FVG overlap ✅")

    lines += [
        "",
        "CANDLE CONFIRMATION",
        f"{bias.capitalize() if bias != 'NEUTRAL' else 'Bias'} rejection: "
        f"{'CONFIRMED ✅ (' + analysis['candle_name'] + ')' if analysis['candle_confirmed'] else 'NOT YET'}",
    ]

    lines += ["", "─" * 20, "🎯 TRADE MODEL", ""]
    if bias == "NEUTRAL" or zone.get("zone_top") is None:
        lines += ["STATUS: ⏳ WAIT", "", "NO TRADE", "",
                  "REASON:", "No clear structure/zone confluence on this timeframe yet."]
    else:
        lines += [f"{bias}: {'ENTRY CONFIRMED' if analysis['entry_ready'] else 'WAITING FOR RETEST'}", ""]
        lines.append(f"ENTRY: {_fmt(analysis['entry']) if analysis['entry'] else 'at zone ' + _fmt(zone['zone_top'])}")
        lines.append(f"SL: {_fmt(analysis['sl'])}")
        lines.append(f"TP1: {_fmt(analysis['tp1'])}")
        lines.append(f"TP2: {_fmt(analysis['tp2'])}")
        lines.append("")
        lines.append(f"STATUS: {'🔥 ' + bias + ' CONFIRMED' if analysis['entry_ready'] else '⏳ WAIT — price not yet at zone / candle unconfirmed'}")

    return "\n".join(lines)


def build_smc_ticket(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not analysis or not analysis.get("entry_ready"):
        return None
    return {
        "entry": analysis["entry"], "sl": analysis["sl"],
        "tp1": analysis["tp1"], "tp2": analysis["tp2"],
        "direction": analysis["bias"],
    }
