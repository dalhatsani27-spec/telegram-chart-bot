"""Compatibility API for the bot's SMC strategy.

SMC is now powered by strategy_upgrade.smc_analysis so it shares the same
market-regime and volatility model as Trendline and OTE.
"""
from strategy_upgrade import smc_analysis, format_smc_report


def run_smc_analysis(symbol: str, tf_code: str = "30min", topdown=None):
    return smc_analysis(symbol, tf_code=tf_code, topdown=topdown)


def build_smc_ticket(analysis):
    if not analysis or not analysis.get("entry_ready"):
        return None
    return {
        "entry": analysis.get("entry"),
        "sl": analysis.get("sl"),
        "tp1": analysis.get("tp1"),
        "tp2": analysis.get("tp2"),
        "tp3": analysis.get("tp3"),
        "direction": analysis.get("direction"),
        "order_type": "MARKET",
    }
