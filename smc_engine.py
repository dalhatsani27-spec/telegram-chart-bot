from __future__ import annotations
from io import BytesIO
import mplfinance as mpf
import numpy as np
import pandas as pd
import market_data
from market_analysis import detect_confirmation_candle
from topdown_engine import get_topdown_bias


def _atr(df, n=14):
    p = df.Close.shift(1)
    tr = pd.concat([df.High-df.Low, (df.High-p).abs(), (df.Low-p).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _pivots(df, left=3, right=3, min_leg_atr=.55):
    if len(df) < left + right + 10:
        return []
    a = _atr(df)
    c = df.Close.to_numpy(float)
    h = df.High.to_numpy(float)
    l = df.Low.to_numpy(float)
    out = []
    for i in range(left, len(df)-right):
        w = c[i-left:i+right+1]
        typ = 'high' if c[i] >= w.max() and c[i] > w.min() else 'low' if c[i] <= w.min() and c[i] < w.max() else None
        if not typ:
            continue
        if out and out[-1]['type'] == typ:
            better = c[i] > out[-1]['close'] if typ == 'high' else c[i] < out[-1]['close']
            if better:
                out[-1] = {'i': i, 'price': h[i] if typ == 'high' else l[i], 'close': c[i], 'type': typ}
            continue
        if out and abs(c[i]-out[-1]['close']) / max(float(a.iloc[i]), 1e-9) < min_leg_atr:
            continue
        out.append({'i': i, 'price': h[i] if typ == 'high' else l[i], 'close': c[i], 'type': typ})
    return out[-32:]


def _bias(p):
    hs = [x for x in p if x['type'] == 'high']
    ls = [x for x in p if x['type'] == 'low']
    if len(hs) < 2 or len(ls) < 2:
        return 'NEUTRAL'
    if hs[-1]['close'] > hs[-2]['close'] and ls[-1]['close'] > ls[-2]['close']:
        return 'BULLISH'
    if hs[-1]['close'] < hs[-2]['close'] and ls[-1]['close'] < ls[-2]['close']:
        return 'BEARISH'
    return 'TRANSITION'


def _liquidity(p, atr):
    tol = max(atr * .20, 1e-9)
    out = []
    for typ, side in [('high', 'BSL'), ('low', 'SSL')]:
        q = [x for x in p if x['type'] == typ]
        for x in q[-14:]:
            out.append({'side': side, 'price': x['price'], 'kind': 'SWING', 'i': x['i']})
        for a, b in zip(q[-14:-1], q[-13:]):
            if abs(a['price']-b['price']) <= tol:
                out.append({'side': side, 'price': (a['price']+b['price'])/2, 'kind': 'EQUAL', 'i': b['i']})
    return sorted(out, key=lambda x: x['i'], reverse=True)


def _find_sweep(df, pools, lookback=40):
    start = max(0, len(df)-lookback)
    found = None
    for i in range(start, len(df)):
        atr = max(float(df.ATR.iloc[i]), 1e-9)
        row = df.iloc[i]
        eligible = [p for p in pools if p['i'] < i]
        for p in eligible:
            if p['side'] == 'SSL' and float(row.Low) < p['price']-atr*.03 and float(row.Close) > p['price']:
                found = {**p, 'event': 'SSL_SWEEP', 'i': i}
            elif p['side'] == 'BSL' and float(row.High) > p['price']+atr*.03 and float(row.Close) < p['price']:
                found = {**p, 'event': 'BSL_SWEEP', 'i': i}
    return found


def _first_break(df, level, direction, start, end=None):
    end = len(df)-1 if end is None else min(end, len(df)-1)
    if level is None:
        return None
    for i in range(max(0, start), end+1):
        c = float(df.Close.iloc[i])
        if direction == 'BUY' and c > level:
            return {'i': i, 'level': float(level), 'direction': 'BUY'}
        if direction == 'SELL' and c < level:
            return {'i': i, 'level': float(level), 'direction': 'SELL'}
    return None


def _sequence(df, pivots, sweep, forced_direction=None):
    if not sweep:
        return {'model': 'NONE', 'direction': forced_direction or 'WAIT', 'choch': None, 'bos': None, 'inducement': None}
    direction = 'BUY' if sweep['event'] == 'SSL_SWEEP' else 'SELL'
    if forced_direction in ('BUY', 'SELL') and forced_direction != direction:
        return {'model': 'LIQUIDITY_CONFLICT', 'direction': forced_direction, 'choch': None, 'bos': None, 'inducement': None}
    si = sweep['i']
    before = [p for p in pivots if p['i'] < si]
    hs = [p for p in before if p['type'] == 'high']
    ls = [p for p in before if p['type'] == 'low']
    pre_bias = _bias(before)
    anchor = hs[-1] if direction == 'BUY' and hs else ls[-1] if direction == 'SELL' and ls else None
    reversal = (direction == 'BUY' and pre_bias == 'BEARISH') or (direction == 'SELL' and pre_bias == 'BULLISH')
    continuation = (direction == 'BUY' and pre_bias == 'BULLISH') or (direction == 'SELL' and pre_bias == 'BEARISH')
    if anchor is None:
        return {'model': 'UNCONFIRMED', 'direction': direction, 'choch': None, 'bos': None, 'inducement': None, 'pre_bias': pre_bias}
    choch = _first_break(df, anchor['price'], direction, si+1)
    if not reversal:
        bos = choch
        label = 'CONTINUATION' if continuation and bos else 'UNCONFIRMED'
        end_i = bos['i'] if bos else len(df)-1
    else:
        if not choch:
            return {'model': 'REVERSAL_PENDING', 'direction': direction, 'choch': None, 'bos': None, 'inducement': None, 'pre_bias': pre_bias}
        post = [p for p in pivots if p['i'] > choch['i']]
        post_same = [p for p in post if p['type'] == ('high' if direction == 'BUY' else 'low')]
        next_anchor = post_same[-1] if post_same else None
        bos = _first_break(df, next_anchor['price'] if next_anchor else None, direction, choch['i']+1)
        label = 'REVERSAL' if bos else 'REVERSAL_PENDING'
        end_i = bos['i'] if bos else len(df)-1
    opp_type = 'low' if direction == 'BUY' else 'high'
    inducement = [p for p in pivots if si < p['i'] <= end_i and p['type'] == opp_type]
    inducement = inducement[-1] if inducement else None
    return {'model': label, 'direction': direction, 'pre_bias': pre_bias, 'choch': choch if reversal else None, 'bos': bos, 'inducement': inducement}


def _displacement(df, direction, start_i=0):
    for i in range(len(df)-1, max(start_i, len(df)-10)-1, -1):
        o, h, l, c = map(float, [df.Open.iloc[i], df.High.iloc[i], df.Low.iloc[i], df.Close.iloc[i]])
        rng = max(h-l, 1e-9)
        body = abs(c-o)
        a = max(float(df.ATR.iloc[i]), 1e-9)
        d = 'BUY' if c > o else 'SELL'
        if d == direction and rng >= 1.25*a and body/rng >= .65:
            return {'i': i, 'direction': d, 'atr_multiple': rng/a}
    return None


def _ob(df, disp):
    if not disp:
        return None
    a = max(float(df.ATR.iloc[disp['i']]), 1e-9)
    for i in range(disp['i']-1, max(-1, disp['i']-9), -1):
        o, h, l, c = map(float, [df.Open.iloc[i], df.High.iloc[i], df.Low.iloc[i], df.Close.iloc[i]])
        if h-l < .25*a:
            continue
        if (disp['direction'] == 'BUY' and c < o) or (disp['direction'] == 'SELL' and c > o):
            return {'type': 'Bullish OB' if disp['direction'] == 'BUY' else 'Bearish OB', 'low': l, 'high': h, 'i': i, 'fresh': True, 'direction': disp['direction']}
    return None


def _fvg(df, direction, start_i=0):
    for i in range(len(df)-1, max(2, start_i+2)-1, -1):
        h2, l2 = float(df.High.iloc[i-2]), float(df.Low.iloc[i-2])
        h, l = float(df.High.iloc[i]), float(df.Low.iloc[i])
        a = max(float(df.ATR.iloc[i]), 1e-9)
        if direction == 'BUY' and l > h2 and l-h2 >= .10*a:
            return {'low': h2, 'high': l, 'direction': 'BUY', 'i': i}
        if direction == 'SELL' and h < l2 and l2-h >= .10*a:
            return {'low': h, 'high': l2, 'direction': 'SELL', 'i': i}
    return None


def _range(df, p):
    hs = [x['price'] for x in p if x['type'] == 'high']
    ls = [x['price'] for x in p if x['type'] == 'low']
    hi = max(hs[-3:]) if hs else float(df.High.tail(80).max())
    lo = min(ls[-3:]) if ls else float(df.Low.tail(80).min())
    eq = (hi+lo)/2
    px = float(df.Close.iloc[-1])
    return {'high': hi, 'low': lo, 'equilibrium': eq, 'location': 'PREMIUM' if px > eq else 'DISCOUNT' if px < eq else 'EQUILIBRIUM'}


def _targets(pools, price, direction, risk):
    side = 'BSL' if direction == 'BUY' else 'SSL'
    q = [p for p in pools if p['side'] == side and (p['price'] > price if direction == 'BUY' else p['price'] < price)]
    q.sort(key=lambda x: abs(x['price']-price))
    return [{**p, 'rr': abs(p['price']-price)/max(risk, 1e-9)} for p in q if abs(p['price']-price)/max(risk, 1e-9) >= 1.2][:3]


def run_smc_analysis(symbol, count=260):
    symbol = str(symbol).strip().upper()
    try:
        h4 = market_data.fetch_candles(symbol, '4h', count)
        h1 = market_data.fetch_candles(symbol, '1h', count)
        m30 = market_data.fetch_candles(symbol, '30min', count)
        if any(x is None or len(x) < 50 for x in (h4, h1, m30)):
            return {'error': f'Insufficient data for {symbol}.'}
        for x in (h4, h1, m30):
            x['ATR'] = _atr(x)
        td = get_topdown_bias(symbol, count_4h=count, count_1h=count)
        p4, p1, p30 = _pivots(h4), _pivots(h1), _pivots(m30)
        b4, b1, b30 = _bias(p4), _bias(p1), _bias(p30)
        atr = float(m30.ATR.iloc[-1])
        pools = _liquidity(p4+p1+p30, atr)
        sweep = _find_sweep(m30, pools)
        td_dir = td.get('direction') if td.get('direction') in ('BUY', 'SELL') else None
        seq = _sequence(m30, p30, sweep, None)
        direction = seq.get('direction') if seq.get('direction') in ('BUY', 'SELL') else td_dir or 'WAIT'
        conflict = bool(sweep and td_dir and direction != td_dir and seq.get('model') == 'LIQUIDITY_CONFLICT')
        disp_start = sweep['i'] if sweep else max(0, len(m30)-12)
        disp = _displacement(m30, direction, disp_start) if direction in ('BUY', 'SELL') else None
        ob = _ob(m30, disp)
        fvg = _fvg(m30, direction, disp_start) if direction in ('BUY', 'SELL') else None
        dr = _range(h4, p4)
        entry = float(m30.Close.iloc[-1])
        sl = ob['low']-atr*.15 if direction == 'BUY' and ob else ob['high']+atr*.15 if direction == 'SELL' and ob else entry-atr*1.5 if direction == 'BUY' else entry+atr*1.5 if direction == 'SELL' else entry
        risk = abs(entry-sl)
        targets = _targets(pools, entry, direction, risk) if direction in ('BUY','SELL') else []
        ok, candle = detect_confirmation_candle(m30, direction) if direction in ('BUY','SELL') else (False, None)
        sweep_ok = bool(sweep and sweep['event'] == ('SSL_SWEEP' if direction == 'BUY' else 'BSL_SWEEP'))
        inducement_ok = bool(seq.get('inducement'))
        choch_ok = bool(seq.get('choch'))
        bos_ok = bool(seq.get('bos'))
        structure_ok = bos_ok and (seq.get('model') == 'CONTINUATION' or choch_ok)
        displacement_ok = bool(disp and disp['direction'] == direction and (not bos_ok or disp['i'] >= bos_ok['i']))
        if bos_ok and disp and disp['i'] < bos_ok['i']:
            displacement_ok = False
        ob_ok = bool(ob and ob['direction'] == direction)
        fv_ok = bool(fvg and fvg['direction'] == direction)
        a4 = (direction == 'BUY' and b4 == 'BULLISH') or (direction == 'SELL' and b4 == 'BEARISH')
        a1 = (direction == 'BUY' and b1 == 'BULLISH') or (direction == 'SELL' and b1 == 'BEARISH')
        score = sum([15 if a4 else 0, 10 if a1 else 0, 20 if sweep_ok else 0, 10 if inducement_ok else 0, 15 if displacement_ok else 0, 15 if structure_ok else 0, 8 if ob_ok else 0, 4 if fv_ok else 0, 3 if ok else 0])
        confirmed = direction in ('BUY','SELL') and not conflict and sweep_ok and inducement_ok and structure_ok and displacement_ok and ob_ok and score >= 70
        status = 'CONFIRMED' if confirmed else 'WAIT'
        return {'symbol': symbol, 'df': m30, 'direction': direction, 'status': status, 'score': min(score, 100), 'bias_4h': b4, 'bias_1h': b1, 'bias_30m': b30, 'topdown': td, 'dealing_range': dr, 'liquidity': pools, 'sweep': sweep, 'inducement': seq.get('inducement'), 'displacement': disp, 'structure_event': seq.get('bos') or seq.get('choch'), 'choch': seq.get('choch'), 'bos': seq.get('bos'), 'setup_type': seq.get('model', 'NONE'), 'pre_sweep_bias': seq.get('pre_bias'), 'order_block': ob, 'fvg': fvg, 'candle_confirmation': candle if ok else None, 'entry': entry, 'sl': sl, 'risk': risk, 'targets': targets, 'key_levels_4h': td.get('key_levels_4h', []), 'requirements': {'sweep': sweep_ok, 'inducement': inducement_ok, 'choch': choch_ok, 'bos': bos_ok, 'structure': structure_ok, 'displacement': displacement_ok, 'order_block': ob_ok, 'fvg': fv_ok}}
    except Exception as e:
        return {'error': f'SMC analysis failed for {symbol}: {e}'}


def _p(x):
    return f'{float(x):.5f}' if isinstance(x, (int, float, np.floating)) else '—'


def format_smc_report(a):
    if a.get('error'):
        return '🏦 SMC ANALYSIS\n\n❌ ' + a['error']
    dr, s, o, f = a['dealing_range'], a.get('sweep'), a.get('order_block'), a.get('fvg')
    ch, bos, ind, d = a.get('choch'), a.get('bos'), a.get('inducement'), a.get('displacement')
    r = a.get('requirements', {})
    lines = [f"🏦 SMC ANALYSIS — {a['symbol']} M30", '', 'TOP-DOWN CONTEXT', f"4H structure: {a['bias_4h']}", f"1H structure: {a['bias_1h']}", f"30M structure: {a['bias_30m']}", f"4H dealing range: {_p(dr['low'])} — {_p(dr['high'])}", f"Equilibrium: {_p(dr['equilibrium'])}", f"Price location: {dr['location']}", '', 'LIQUIDITY / SETUP', f"Sweep: {s['event'] if s else '—'}", f"Inducement: {_p(ind['price']) if ind else '—'}", f"Setup type: {a.get('setup_type','—')}", '', 'STRUCTURE / ORDER FLOW', f"CHoCH/MSS: {_p(ch['level']) if ch else '—'}", f"BOS: {_p(bos['level']) if bos else '—'}", f"Displacement: {d['direction']} ({d['atr_multiple']:.2f} ATR)" if d else 'Displacement: —', f"Order Block: {o['type']} {_p(o['low'])} — {_p(o['high'])}" if o else 'Order Block: —', f"FVG: {_p(f['low'])} — {_p(f['high'])}" if f else 'FVG: —', f"Candle confirmation: {a.get('candle_confirmation') or '—'}", '', 'SEQUENCE CHECK', f"Sweep {'✅' if r.get('sweep') else '❌'} | Inducement {'✅' if r.get('inducement') else '❌'}", f"CHoCH {'✅' if r.get('choch') else '—'} | BOS {'✅' if r.get('bos') else '❌'}", f"Displacement {'✅' if r.get('displacement') else '❌'} | OB {'✅' if r.get('order_block') else '❌'} | FVG {'✅' if r.get('fvg') else '—'}", '', '━━━━━━━━━━━━━━━━', '🎯 DECISION', f"BIAS: {a['direction']}", f"SETUP: {a.get('setup_type','—')}", f"STATUS: {a['status']}", f"SMC SCORE: {a['score']}/100", f"ENTRY: {_p(a['entry'])}", f"SL: {_p(a['sl'])}"]
    for i, t in enumerate(a.get('targets', [])[:3], 1):
        lines.append(f"TP{i}: {_p(t['price'])} ({t['kind']} liquidity, {t['rr']:.2f}R)")
    if not a.get('targets'):
        lines.append('TP: No qualifying liquidity target ≥1.2R')
    return '\n'.join(lines)


def generate_smc_chart(a):
    df = a['df'].tail(120).copy(); df['ATR'] = _atr(df)
    fig, axes = mpf.plot(df[['Open','High','Low','Close']], type='candle', style='charles', volume=False, returnfig=True, figsize=(12,7), warn_too_much_data=10000)
    ax = axes[0]; atr = float(df.ATR.iloc[-1]); n = len(df)
    for q in a.get('key_levels_4h', []):
        px = float(q['price'])
        if df.Low.min()-5*atr <= px <= df.High.max()+5*atr:
            ax.axhline(px, linestyle='--', linewidth=.9, alpha=.55); ax.text(n-1, px, f"  4H {q.get('side','LIQ')} ({q.get('touches','')})", fontsize=6.5, ha='right')
    for q in a.get('liquidity', [])[:8]:
        px = float(q['price'])
        if df.Low.min()-4*atr <= px <= df.High.max()+4*atr:
            ax.axhline(px, linestyle=':', linewidth=.7, alpha=.28)
    if a.get('dealing_range'): ax.axhline(a['dealing_range']['equilibrium'], linestyle='-.', linewidth=.7, alpha=.35)
    if a.get('order_block'):
        o=a['order_block']; ax.axhspan(o['low'], o['high'], alpha=.18); ax.text(2,o['high'],o['type'],fontsize=7,fontweight='bold')
    if a.get('fvg'):
        f=a['fvg']; ax.axhspan(f['low'],f['high'],alpha=.12); ax.text(2,f['high'],'FVG',fontsize=7,fontweight='bold')
    if a.get('inducement'):
        ax.axhline(a['inducement']['price'], linestyle='--', linewidth=.8, alpha=.35); ax.text(2,a['inducement']['price'],'Inducement',fontsize=7)
    ax.set_title(f"{a['symbol']} | M30 | SMC {a['direction']} | {a.get('setup_type','—')} | {a['status']}")
    fig.tight_layout(); out=BytesIO(); fig.savefig(out,format='png',dpi=150,bbox_inches='tight'); out.seek(0); return out


def run_hybrid_analysis(symbol):
    import strategies
    s=run_smc_analysis(symbol); t=strategies.run_trendline_analysis(symbol)
    if s.get('error') or t.get('error'): return {'error':s.get('error') or t.get('error'),'smc':s,'trendline':t}
    agree=s.get('direction') in ('BUY','SELL') and s.get('direction')==t.get('direction')
    score=min(100,int((s.get('score',0)+t.get('strength',0))/2+(15 if agree else 0)))
    return {'symbol':symbol,'direction':s['direction'] if agree else 'WAIT','status':'CONFIRMED' if agree and s['status']=='CONFIRMED' else 'WAIT','score':score,'agreement':agree,'smc':s,'trendline':t}


def format_hybrid_report(h):
    if h.get('error'): return '🔀 HYBRID ANALYSIS\n\n❌ '+h['error']
    s,t=h['smc'],h['trendline']; sw=s.get('sweep'); o=s.get('order_block')
    return '\n'.join([f"🔀 HYBRID ANALYSIS — {h['symbol']} M30",'',f"SMC: {s.get('direction')} | {s.get('status')} | {s.get('score')}/100",f"Setup: {s.get('setup_type','—')}",f"Trendline: {t.get('direction','—')} | {t.get('strength',0)}/100",f"Liquidity sweep: {sw['event'] if sw else '—'}",f"Inducement: {_p(s['inducement']['price']) if s.get('inducement') else '—'}",f"Order Block: {o['type'] if o else '—'}",'', '━━━━━━━━━━━━━━━━','🎯 DECISION',f"SMC: {s.get('direction')}",f"Trendline: {t.get('direction')}",f"CONFLUENCE: {'AGREE' if h['agreement'] else 'CONFLICT'}",f"BIAS: {h['direction']}",f"STATUS: {h['status']}",f"HYBRID SCORE: {h['score']}/100",'', 'SMC and Trendline remain independent; conflict = WAIT.'])
