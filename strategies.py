"""Compatibility API for the bot's Trendline and OTE strategies.

The heavy legacy strategy implementation is intentionally replaced by the
shared strategy_upgrade engine. Keeping these public function names means
bot.py and existing chart/execution integrations continue to work.
"""
from strategy_upgrade import trendline_analysis, ote_analysis, format_trendline_report, format_ote_report, build_position_container


def run_trendline_analysis(symbol: str, tf_code: str = "30min", topdown=None):
    return trendline_analysis(symbol, tf_code=tf_code, topdown=topdown)


def run_ote_analysis(symbol: str, tf_code: str = "30min", topdown=None):
    return ote_analysis(symbol, tf_code=tf_code, topdown=topdown)
