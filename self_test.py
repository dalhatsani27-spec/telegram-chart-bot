
import numpy as np, pandas as pd
from market_intelligence import analyze_market_intelligence

np.random.seed(7)
n = 160
close = 100 + np.cumsum(np.random.normal(0, .25, n))
open_ = close + np.random.normal(0, .08, n)
high = np.maximum(open_, close) + np.random.uniform(.05, .25, n)
low = np.minimum(open_, close) - np.random.uniform(.05, .25, n)
df = pd.DataFrame({"Open":open_, "High":high, "Low":low, "Close":close})
r = analyze_market_intelligence(df)
assert "market_state" in r and "execution" in r
print("V2 self-test OK:", r["market_state"]["state"], r["setup"]["score"])
