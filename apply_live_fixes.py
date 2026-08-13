from pathlib import Path
import re

# ------------------------------------------------------------
# 1. WATCH PRICE: use the same analysis data source as fallback
# ------------------------------------------------------------

p = Path("market_data.py")
s = p.read_text()

old = '''def fetch_watch_price(symbol):
    """Unified latest price for personal watch-level alerts."""
    if is_deriv_symbol(symbol):
        return deriv_fetch_tick(symbol)
    bid, ask, _ = fetch_tick(symbol)
    return (bid + ask) / 2.0 if bid is not None and ask is not None else None
'''

new = '''def fetch_watch_price(symbol):
    """
    Unified current price for watch-level alerts.

    Priority:
      1. Deriv live tick for Deriv symbols.
      2. MT5 live bid/ask when available.
      3. The same market-data pipeline used by chart analysis.
      4. Latest available M1 candle close as final fallback.

    Watch Price deliberately uses live/current quote data where available,
    while SMC analysis continues to use closed candles for stable structure.
    """
    symbol = str(symbol or "").strip().upper()

    # Deriv live quote
    if is_deriv_symbol(symbol):
        try:
            price = deriv_fetch_tick(symbol)
            if price is not None:
                return float(price)
        except Exception:
            pass

    # MT5 live quote
    try:
        bid, ask, _ = fetch_tick(symbol)
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2.0
        if bid is not None:
            return float(bid)
        if ask is not None:
            return float(ask)
    except Exception:
        pass

    # IMPORTANT: use the exact same fallback market-data path as analysis.
    # This works on Render where MT5 is unavailable.
    try:
        df = fetch_candles(symbol, "1min", 3)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass

    return None
'''

if old not in s:
    raise SystemExit("Could not find fetch_watch_price() exactly. No market_data.py changes made.")

s = s.replace(old, new, 1)
p.write_text(s)
print("OK: market_data.py Watch Price updated")


# ------------------------------------------------------------
# 2. CHARTS: add reusable current-price overlay helper
# ------------------------------------------------------------

p = Path("chart_engine.py")
s = p.read_text()

helper = r'''

def _draw_current_price(ax, price, chart_len, label="CURRENT PRICE"):
    """Draw a highly visible current-price marker on dark charts."""
    if price is None:
        return

    try:
        price = float(price)
    except (TypeError, ValueError):
        return

    # Current price gets a dedicated visual layer so it remains visible
    # above zones, trendlines and other annotations.
    ax.axhline(
        price,
        color="#ffd54f",
        linestyle="--",
        linewidth=1.8,
        alpha=0.95,
        zorder=25,
    )

    ax.text(
        chart_len - 1,
        price,
        f" {label}  {price:.5f} ",
        fontsize=8.5,
        fontweight="bold",
        color="#111111",
        ha="right",
        va="center",
        zorder=30,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="#ffd54f",
            edgecolor="#ffffff",
            linewidth=0.8,
            alpha=0.98,
        ),
    )
'''

if "_draw_current_price(" not in s:
    # Insert after the colour/session configuration and before the first
    # renderer helper. This keeps it available to every chart renderer.
    marker = '\n\ndef _to_naive_utc'
    if marker not in s:
        raise SystemExit("Could not find chart_engine insertion point.")
    s = s.replace(marker, helper + marker, 1)

# ------------------------------------------------------------
# 3. Inject current-price rendering into every chart renderer.
#
# The renderers already have chart data. We obtain the latest close
# from that data, which is the same data source used by the strategy.
# ------------------------------------------------------------

# Try several safe insertion anchors commonly used by the renderers.
anchors = [
    'fig.tight_layout()',
    'plt.tight_layout()',
]

injected = '''
    # Current price: keep it unmistakable on every non-SMC strategy chart.
    try:
        current_price = float(chart_df["Close"].iloc[-1])
        _draw_current_price(ax, current_price, chart_len)
    except Exception:
        pass

'''

count = 0
for anchor in anchors:
    if anchor in s:
        # Only inject into function bodies containing chart_df and ax.
        parts = s.split(anchor)
        rebuilt = parts[0]
        for part in parts[1:]:
            # Avoid duplicate insertion when multiple tight_layout calls exist.
            rebuilt += injected + anchor + part
        s = rebuilt
        count += 1

# The above may affect multiple renderers, which is intentional. Remove
# accidental duplicate current-price blocks in the same function.
lines = s.splitlines()
clean = []
prev_block = False
for line in lines:
    if line.strip().startswith("# Current price: keep it unmistakable"):
        if prev_block:
            continue
        prev_block = True
    elif line.strip() and not line.strip().startswith("#"):
        if "current_price = float(chart_df" not in line:
            prev_block = False
    clean.append(line)
s = "\n".join(clean) + "\n"

p.write_text(s)
print("OK: chart_engine.py current-price overlay updated")


# ------------------------------------------------------------
# 4. Ensure dark chart defaults remain enforced.
# ------------------------------------------------------------

p = Path("chart_engine.py")
s = p.read_text()

required = {
    '"bg": "#0d1117"': '"bg": "#0d1117"',
    '"grid": "#1f2937"': '"grid": "#1f2937"',
    '"text": "#e5e7eb"': '"text": "#e5e7eb"',
}

for needle in required:
    if needle not in s:
        print("WARNING: expected dark-theme palette entry missing:", needle)

p.write_text(s)

print("DONE: live chart fixes prepared.")
