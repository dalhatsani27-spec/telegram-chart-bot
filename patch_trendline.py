from pathlib import Path

p = Path('strategies.py')
s = p.read_text(encoding='utf-8')

helper = '''\n\ndef _find_impulse_anchor_pair(pivots: List[Dict], df: pd.DataFrame, kind: str) -> Optional[Tuple[Dict, Dict]]:\n    """Select the hand-drawn structural trendline anchors.\n\n    BUY/uptrend: last confirmed HL before a meaningful bullish impulse\n    (HL -> HH), then the latest confirmed HL after that impulse.\n\n    SELL/downtrend: last confirmed HH before a meaningful bearish impulse\n    (HH -> LL), then the latest confirmed LH after that impulse.\n    """\n    if df is None or len(pivots) < 4:\n        return None\n\n    ordered = sorted(pivots, key=lambda p: p.get('index', 0))\n    atr = pd.to_numeric(df['ATR'], errors='coerce').to_numpy(float) if 'ATR' in df.columns else None\n\n    def _atr_at(i: int) -> float:\n        if atr is not None and 0 <= i < len(atr) and np.isfinite(atr[i]) and atr[i] > 0:\n            return float(atr[i])\n        return max(float(df['High'].iloc[i] - df['Low'].iloc[i]), 1e-9)\n\n    lows = [p for p in ordered if p.get('type') == 'low']\n    highs = [p for p in ordered if p.get('type') == 'high']\n\n    if kind == 'support':\n        if len(lows) < 3 or len(highs) < 2:\n            return None\n        candidates = []\n        for i in range(1, len(lows) - 1):\n            anchor = lows[i]\n            if float(anchor['price']) <= float(lows[i - 1]['price']):\n                continue\n            next_highs = [h for h in highs if h['index'] > anchor['index']]\n            if not next_highs:\n                continue\n            impulse_high = next_highs[0]\n            prior_highs = [h for h in highs if h['index'] < impulse_high['index']]\n            if not prior_highs or float(impulse_high['price']) <= float(prior_highs[-1]['price']):\n                continue\n            move = float(impulse_high['price']) - float(anchor['price'])\n            impulse_atr = move / max((_atr_at(anchor['index']) + _atr_at(impulse_high['index'])) / 2.0, 1e-9)\n            if impulse_atr < 1.25:\n                continue\n            candidates.append((impulse_high['index'], impulse_atr, anchor))\n        if not candidates:\n            return None\n        _, _, anchor = max(candidates, key=lambda x: (x[0], x[1]))\n        endpoints = [p for p in lows if p['index'] > anchor['index'] and float(p['price']) > float(anchor['price'])]\n        if not endpoints:\n            return None\n        endpoint = endpoints[-1]\n        return (anchor, endpoint) if endpoint['index'] - anchor['index'] >= 4 else None\n\n    if kind == 'resistance':\n        if len(highs) < 3 or len(lows) < 2:\n            return None\n        candidates = []\n        for i in range(1, len(highs) - 1):\n            anchor = highs[i]\n            if float(anchor['price']) >= float(highs[i - 1]['price']):\n                continue\n            next_lows = [l for l in lows if l['index'] > anchor['index']]\n            if not next_lows:\n                continue\n            impulse_low = next_lows[0]\n            prior_lows = [l for l in lows if l['index'] < impulse_low['index']]\n            if not prior_lows or float(impulse_low['price']) >= float(prior_lows[-1]['price']):\n                continue\n            move = float(anchor['price']) - float(impulse_low['price'])\n            impulse_atr = move / max((_atr_at(anchor['index']) + _atr_at(impulse_low['index'])) / 2.0, 1e-9)\n            if impulse_atr < 1.25:\n                continue\n            candidates.append((impulse_low['index'], impulse_atr, anchor))\n        if not candidates:\n            return None\n        _, _, anchor = max(candidates, key=lambda x: (x[0], x[1]))\n        endpoints = [p for p in highs if p['index'] > anchor['index'] and float(p['price']) < float(anchor['price'])]\n        if not endpoints:\n            return None\n        endpoint = endpoints[-1]\n        return (anchor, endpoint) if endpoint['index'] - anchor['index'] >= 4 else None\n\n    return None\n'''

marker = "\ndef _fit_primary(pivots: List[Dict], kind: str, n: int, df: pd.DataFrame,\n"
if marker not in s:
    raise SystemExit('_fit_primary marker not found')
s = s.replace(marker, helper + marker, 1)

old = '''    # Prefer anchoring the line at the pivot that actually started the
    # current leg -- the low/high price broke away from and then closed
    # back through the 20 SMA. This beats picking whichever pivot merely
    # maximizes span: a trendline that starts at the SMA-reclaim point is
    # the one a trader would actually draw by hand, and it gives a single
    # unambiguous line to watch for a break/flip instead of several
    # candidate spans quietly competing underneath.
    sma_anchor = _find_sma_reclaim_anchor(pivots, sma, df, kind)
    if sma_anchor is not None:
        later_pts = [p for p in pts if p["index"] > sma_anchor["index"]]
        b = later_pts[-1] if later_pts else pts[-1]
        if b["index"] > sma_anchor["index"]:
            a = sma_anchor
        else:
            a = None
    else:
        a = None
'''
new = '''    # Hand-drawn structural rule: anchor at the last HL/HH that launched
    # the current impulse, then connect it to the latest same-side pivot.
    # The older SMA-reclaim method remains only as a fallback when the
    # structural sequence is not mature enough.
    impulse_pair = _find_impulse_anchor_pair(pivots, df, kind)
    if impulse_pair is not None:
        a, b = impulse_pair
        sma_anchor = None
    else:
        sma_anchor = _find_sma_reclaim_anchor(pivots, sma, df, kind)
        if sma_anchor is not None:
            later_pts = [p for p in pts if p["index"] > sma_anchor["index"]]
            b = later_pts[-1] if later_pts else pts[-1]
            if b["index"] > sma_anchor["index"]:
                a = sma_anchor
            else:
                a = None
        else:
            a = None
'''
if old not in s:
    raise SystemExit('primary anchor block not found')
s = s.replace(old, new, 1)

old_method = '"method": "sma_reclaim" if (sma_anchor is not None and a is sma_anchor) else "classic_sequential",'
new_method = '"method": "impulse_structural" if impulse_pair is not None else ("sma_reclaim" if (sma_anchor is not None and a is sma_anchor) else "classic_sequential"),'
if old_method not in s:
    raise SystemExit('method field not found')
s = s.replace(old_method, new_method, 1)

p.write_text(s, encoding='utf-8')
