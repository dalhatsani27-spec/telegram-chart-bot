from __future__ import annotations
from io import BytesIO

import mplfinance as mpf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import market_data
from market_analysis import detect_confirmation_candle, detect_order_blocks
from topdown_engine import get_topdown_bias


def _atr(df, n=14):
    p = df.Close.shift(1)
    tr = pd.concat([df.High-df.Low, (df.High-p).abs(), (df.Low-p).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _pivots(df, left=3, right=3, min_leg_atr=.55):
    if len(df) < left + right + 10:
        return []
    a = _atr(df)
    c = df.Close.to_numpy(float); h = df.High.to_numpy(float); l = df.Low.to_numpy(float)
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
    hs = [x for x in p if x['type'] == 'high']; ls = [x for x in p if x['type'] == 'low']
    if len(hs) < 2 or len(ls) < 2: return 'NEUTRAL'
    if hs[-1]['close'] > hs[-2]['close'] and ls[-1]['close'] > ls[-2]['close']: return 'BULLISH'
    if hs[-1]['close'] < hs[-2]['close'] and ls[-1]['close'] < ls[-2]['close']: return 'BEARISH'
    return 'TRANSITION'


def _liquidity(p, atr):
    tol = max(atr * .20, 1e-9); out = []
    for typ, side in [('high', 'BSL'), ('low', 'SSL')]:
        q = [x for x in p if x['type'] == typ]
        for x in q[-14:]: out.append({'side': side, 'price': x['price'], 'kind': 'SWING', 'i': x['i']})
        for a, b in zip(q[-14:-1], q[-13:]):
            if abs(a['price']-b['price']) <= tol:
                out.append({'side': side, 'price': (a['price']+b['price'])/2, 'kind': 'EQUAL', 'i': b['i']})
    return sorted(out, key=lambda x: x['i'], reverse=True)


def _find_sweep(df, pools, lookback=60):
    start = max(0, len(df)-lookback); found = None
    for i in range(start, len(df)):
        atr = max(float(df.ATR.iloc[i]), 1e-9); row = df.iloc[i]
        for p in [p for p in pools if p['i'] < i]:
            if p['side'] == 'SSL' and float(row.Low) < p['price']-atr*.03 and float(row.Close) > p['price']:
                found = {**p, 'event': 'SSL_SWEEP', 'i': i}
            elif p['side'] == 'BSL' and float(row.High) > p['price']+atr*.03 and float(row.Close) < p['price']:
                found = {**p, 'event': 'BSL_SWEEP', 'i': i}
    return found


def _first_break(df, level, direction, start, end=None):
    end = len(df)-1 if end is None else min(end, len(df)-1)
    if level is None: return None
    for i in range(max(0, start), end+1):
        c = float(df.Close.iloc[i])
        if direction == 'BUY' and c > level: return {'i': i, 'level': float(level), 'direction': 'BUY'}
        if direction == 'SELL' and c < level: return {'i': i, 'level': float(level), 'direction': 'SELL'}
    return None


def _sequence(df, pivots, sweep, forced_direction=None):
    if not sweep:
        return {'model': 'NONE', 'direction': forced_direction or 'WAIT', 'choch': None, 'bos': None, 'inducement': None}
    direction = 'BUY' if sweep['event'] == 'SSL_SWEEP' else 'SELL'
    if forced_direction in ('BUY','SELL') and forced_direction != direction:
        return {'model': 'LIQUIDITY_CONFLICT', 'direction': forced_direction, 'choch': None, 'bos': None, 'inducement': None}
    si = sweep['i']; before = [p for p in pivots if p['i'] < si]
    hs = [p for p in before if p['type'] == 'high']; ls = [p for p in before if p['type'] == 'low']
    pre_bias = _bias(before)
    anchor = hs[-1] if direction == 'BUY' and hs else ls[-1] if direction == 'SELL' and ls else None
    reversal = (direction == 'BUY' and pre_bias == 'BEARISH') or (direction == 'SELL' and pre_bias == 'BULLISH')
    continuation = (direction == 'BUY' and pre_bias == 'BULLISH') or (direction == 'SELL' and pre_bias == 'BEARISH')
    if anchor is None:
        return {'model':'UNCONFIRMED','direction':direction,'choch':None,'bos':None,'inducement':None,'pre_bias':pre_bias}
    choch = _first_break(df, anchor['price'], direction, si+1)
    if not reversal:
        bos = choch; label = 'CONTINUATION' if continuation and bos else 'UNCONFIRMED'; end_i = bos['i'] if bos else len(df)-1
    else:
        if not choch:
            return {'model':'REVERSAL_PENDING','direction':direction,'choch':None,'bos':None,'inducement':None,'pre_bias':pre_bias}
        post = [p for p in pivots if p['i'] > choch['i']]
        post_same = [p for p in post if p['type'] == ('high' if direction == 'BUY' else 'low')]
        next_anchor = post_same[-1] if post_same else None
        bos = _first_break(df, next_anchor['price'] if next_anchor else None, direction, choch['i']+1)
        label = 'REVERSAL' if bos else 'REVERSAL_PENDING'; end_i = bos['i'] if bos else len(df)-1
    opp_type = 'low' if direction == 'BUY' else 'high'
    inducement = [p for p in pivots if si < p['i'] <= end_i and p['type'] == opp_type]
    inducement = inducement[-1] if inducement else None
    return {'model':label,'direction':direction,'pre_bias':pre_bias,'choch':choch if reversal else None,'bos':bos,'inducement':inducement}


def _displacement(df, direction, start_i=0):
    for i in range(len(df)-1, max(start_i, len(df)-15)-1, -1):
        o,h,l,c = map(float,[df.Open.iloc[i],df.High.iloc[i],df.Low.iloc[i],df.Close.iloc[i]])
        rng=max(h-l,1e-9); body=abs(c-o); a=max(float(df.ATR.iloc[i]),1e-9); d='BUY' if c>o else 'SELL'
        if d == direction and rng >= 1.25*a and body/rng >= .65:
            return {'i':i,'direction':d,'atr_multiple':rng/a}
    return None


def _fvg(df, direction, start_i=0):
    for i in range(len(df)-1, max(2,start_i+2)-1, -1):
        h2,l2=float(df.High.iloc[i-2]),float(df.Low.iloc[i-2]); h,l=float(df.High.iloc[i]),float(df.Low.iloc[i]); a=max(float(df.ATR.iloc[i]),1e-9)
        if direction == 'BUY' and l > h2 and l-h2 >= .10*a: return {'low':h2,'high':l,'direction':'BUY','i':i}
        if direction == 'SELL' and h < l2 and l2-h >= .10*a: return {'low':h,'high':l2,'direction':'SELL','i':i}
    return None


def _range(df,p):
    hs=[x['price'] for x in p if x['type']=='high']; ls=[x['price'] for x in p if x['type']=='low']; hi=max(hs[-3:]) if hs else float(df.High.tail(80).max()); lo=min(ls[-3:]) if ls else float(df.Low.tail(80).min()); eq=(hi+lo)/2; px=float(df.Close.iloc[-1])
    return {'high':hi,'low':lo,'equilibrium':eq,'location':'PREMIUM' if px>eq else 'DISCOUNT' if px<eq else 'EQUILIBRIUM'}


def _targets(pools,price,direction,risk):
    side='BSL' if direction=='BUY' else 'SSL'; q=[p for p in pools if p['side']==side and (p['price']>price if direction=='BUY' else p['price']<price)]; q.sort(key=lambda x:abs(x['price']-price))
    return [{**p,'rr':abs(p['price']-price)/max(risk,1e-9)} for p in q if abs(p['price']-price)/max(risk,1e-9)>=1.2][:3]


def _confirmation_history(df, direction, start_i):
    """Find the most recent valid confirmation after the structural trigger.
    Confirmation is stateful: once it occurred, it remains confirmed until the
    setup is invalidated, instead of disappearing because the latest candle is quiet.
    """
    if direction not in ('BUY','SELL'): return None, None
    start=max(20,int(start_i)); found_i=None; found=None
    for i in range(start,len(df)):
        try:
            ok,candle=detect_confirmation_candle(df.iloc[:i+1].copy(),direction)
        except Exception:
            ok,candle=False,None
        if ok:
            found_i=i; found=candle
    return found_i,found


def _zone_touch(df, zone, direction):
    if not zone: return False
    px=float(df.Close.iloc[-1]); lo=float(zone['low']); hi=float(zone['high'])
    return lo <= px <= hi


def run_smc_analysis(symbol,count=260):
    symbol=str(symbol).strip().upper()
    try:
        h4=market_data.fetch_candles(symbol,'4h',count); h1=market_data.fetch_candles(symbol,'1h',count); m30=market_data.fetch_candles(symbol,'30min',count)
        if any(x is None or len(x)<50 for x in (h4,h1,m30)): return {'error':f'Insufficient data for {symbol}.'}
        for x in (h4,h1,m30): x['ATR']=_atr(x)
        td=get_topdown_bias(symbol,count_4h=count,count_1h=count); p4,p1,p30=_pivots(h4),_pivots(h1),_pivots(m30); b4,b1,b30=_bias(p4),_bias(p1),_bias(p30)
        atr=float(m30.ATR.iloc[-1]); pools=_liquidity(p4+p1+p30,atr); sweep=_find_sweep(m30,pools)
        td_dir=td.get('direction') if td.get('direction') in ('BUY','SELL') else None; seq=_sequence(m30,p30,sweep,None); direction=seq.get('direction') if seq.get('direction') in ('BUY','SELL') else td_dir or 'WAIT'
        conflict=bool(sweep and td_dir and direction!=td_dir and seq.get('model')=='LIQUIDITY_CONFLICT')
        disp_start=sweep['i'] if sweep else max(0,len(m30)-30); disp=_displacement(m30,direction,disp_start) if direction in ('BUY','SELL') else None; fvg=_fvg(m30,direction,disp_start) if direction in ('BUY','SELL') else None

        # Use the repository's structure-confirmed OB detector for real zones, not a single candle heuristic.
        all_obs=detect_order_blocks(m30,lookback=min(180,len(m30)),max_per_side=3,min_confidence=45) if direction in ('BUY','SELL') else []
        side='bullish' if direction=='BUY' else 'bearish'
        side_obs=[o for o in all_obs if o.get('type')==side]
        primary_obs=[o for o in side_obs if o.get('role')!='inducement'] or side_obs
        chosen=primary_obs[-1] if primary_obs else None
        ob={'type':('Bullish OB' if direction=='BUY' else 'Bearish OB'),'low':chosen['bottom'],'high':chosen['top'],'i':chosen['formed_index'],'fresh':chosen.get('freshness'),'direction':direction,'role':chosen.get('role','primary'),'confidence':chosen.get('confidence')} if chosen else None

        # If the structural pivot inducement is absent, retain the OB engine's explicit inducement role.
        inducement=seq.get('inducement')
        if inducement is None:
            ind_obs=[o for o in side_obs if o.get('is_inducement')]
            if ind_obs:
                io=ind_obs[-1]; inducement={'price':(float(io['top'])+float(io['bottom']))/2,'i':io['formed_index'],'type':'OB_INDUCEMENT'}

        dr=_range(h4,p4); entry=float(m30.Close.iloc[-1])
        sl=ob['low']-atr*.15 if direction=='BUY' and ob else ob['high']+atr*.15 if direction=='SELL' and ob else entry-atr*1.5 if direction=='BUY' else entry+atr*1.5 if direction=='SELL' else entry
        risk=abs(entry-sl); targets=_targets(pools,entry,direction,risk) if direction in ('BUY','SELL') else []
        trigger_i=max([x['i'] for x in (sweep,seq.get('choch'),seq.get('bos'),disp) if x] or [disp_start])
        confirmation_i,candle=_confirmation_history(m30,direction,trigger_i) if direction in ('BUY','SELL') else (None,None)

        sweep_ok=bool(sweep and sweep['event']==('SSL_SWEEP' if direction=='BUY' else 'BSL_SWEEP')); inducement_ok=bool(inducement); choch_ok=bool(seq.get('choch')); bos_ok=bool(seq.get('bos')); structure_ok=bos_ok and (seq.get('model')=='CONTINUATION' or choch_ok)
        displacement_ok=bool(disp and disp['direction']==direction and disp['i']>=trigger_i); ob_ok=bool(ob and ob['direction']==direction); fv_ok=bool(fvg and fvg['direction']==direction)
        a4=(direction=='BUY' and b4=='BULLISH') or (direction=='SELL' and b4=='BEARISH'); a1=(direction=='BUY' and b1=='BULLISH') or (direction=='SELL' and b1=='BEARISH')
        zone_touch=_zone_touch(m30,ob,direction); invalidated=False
        if ob:
            invalidated=bool((direction=='BUY' and float(m30.Close.iloc[-1])<ob['low']) or (direction=='SELL' and float(m30.Close.iloc[-1])>ob['high']))
        confirmation_ok=confirmation_i is not None and not invalidated
        score=sum([15 if a4 else 0,10 if a1 else 0,20 if sweep_ok else 0,10 if inducement_ok else 0,15 if displacement_ok else 0,15 if structure_ok else 0,8 if ob_ok else 0,4 if fv_ok else 0,3 if confirmation_ok else 0])
        confirmed=direction in ('BUY','SELL') and not conflict and sweep_ok and inducement_ok and structure_ok and displacement_ok and ob_ok and confirmation_ok and score>=70
        if confirmed: status='CONFIRMED'; waiting_for='ENTRY WINDOW / RETEST'
        elif invalidated: status='INVALIDATED'; waiting_for='NEW SWEEP / NEW STRUCTURE'
        elif not sweep_ok: status='WAIT'; waiting_for='LIQUIDITY SWEEP'
        elif not structure_ok: status='WAIT'; waiting_for='CHoCH/MSS → BOS'
        elif not displacement_ok: status='WAIT'; waiting_for='DISPLACEMENT'
        elif not ob_ok: status='WAIT'; waiting_for='VALID ORDER BLOCK'
        elif not confirmation_ok: status='WAIT'; waiting_for='CONFIRMATION CANDLE'
        else: status='WAIT'; waiting_for='FINAL ENTRY'
        return {'symbol':symbol,'df':m30,'direction':direction,'status':status,'waiting_for':waiting_for,'score':min(score,100),'bias_4h':b4,'bias_1h':b1,'bias_30m':b30,'topdown':td,'dealing_range':dr,'liquidity':pools,'sweep':sweep,'inducement':inducement,'displacement':disp,'structure_event':seq.get('bos') or seq.get('choch'),'choch':seq.get('choch'),'bos':seq.get('bos'),'setup_type':seq.get('model','NONE'),'pre_sweep_bias':seq.get('pre_bias'),'order_block':ob,'order_blocks':all_obs,'fvg':fvg,'candle_confirmation':candle,'confirmation_i':confirmation_i,'entry':entry,'sl':sl,'risk':risk,'targets':targets,'key_levels_4h':td.get('key_levels_4h',[]),'zone_touch':zone_touch,'invalidated':invalidated,'requirements':{'sweep':sweep_ok,'inducement':inducement_ok,'choch':choch_ok,'bos':bos_ok,'structure':structure_ok,'displacement':displacement_ok,'order_block':ob_ok,'fvg':fv_ok,'confirmation':confirmation_ok}}
    except Exception as e:
        return {'error':f'SMC analysis failed for {symbol}: {e}'}


def _p(x): return f'{float(x):.5f}' if isinstance(x,(int,float,np.floating)) else '—'


def format_smc_report(a):
    if a.get('error'): return '🏦 SMC ANALYSIS\n\n❌ '+a['error']
    dr,s,o,f=a['dealing_range'],a.get('sweep'),a.get('order_block'),a.get('fvg'); ch,bos,ind,d=a.get('choch'),a.get('bos'),a.get('inducement'),a.get('displacement'); r=a.get('requirements',{})
    lines=[f"🏦 SMC ANALYSIS — {a['symbol']} M30",'',f"BIAS: {a['direction']} | SETUP: {a.get('setup_type','—')} | STATUS: {a['status']}",f"WAITING FOR: {a.get('waiting_for','—')}",'','TOP-DOWN CONTEXT',f"4H: {a['bias_4h']} | 1H: {a['bias_1h']} | 30M: {a['bias_30m']}",f"4H range: {_p(dr['low'])} — {_p(dr['high'])} | EQ: {_p(dr['equilibrium'])} | {dr['location']}",'','LIQUIDITY / SEQUENCE',f"Sweep: {s['event'] if s else '—'}",f"Inducement: {_p(ind['price']) if ind else '—'}",f"CHoCH/MSS: {_p(ch['level']) if ch else '—'}",f"BOS: {_p(bos['level']) if bos else '—'}",f"Displacement: {d['direction']} ({d['atr_multiple']:.2f} ATR)" if d else 'Displacement: —','',f"ORDER BLOCK: {o['type']} {_p(o['low'])} — {_p(o['high'])}" if o else 'ORDER BLOCK: —',f"OB role: {o.get('role')} | confidence {o.get('confidence')}" if o else '',f"FVG: {_p(f['low'])} — {_p(f['high'])}" if f else 'FVG: —',f"Confirmation: {a.get('candle_confirmation') or '—'}",f"Confirmation candle index: {a.get('confirmation_i') if a.get('confirmation_i') is not None else '—'}",'', 'SEQUENCE CHECK',f"Sweep {'✅' if r.get('sweep') else '❌'} | Inducement {'✅' if r.get('inducement') else '❌'}",f"CHoCH {'✅' if r.get('choch') else '—'} | BOS {'✅' if r.get('bos') else '❌'}",f"Displacement {'✅' if r.get('displacement') else '❌'} | OB {'✅' if r.get('order_block') else '❌'} | FVG {'✅' if r.get('fvg') else '—'}",f"Confirmation {'✅' if r.get('confirmation') else '❌'}",'', '━━━━━━━━━━━━━━━━','🎯 DECISION',f"ENTRY: {_p(a['entry'])}",f"SL: {_p(a['sl'])}",f"ZONE TOUCH: {'YES' if a.get('zone_touch') else 'NO'}",f"SMC SCORE: {a['score']}/100"]
    for i,t in enumerate(a.get('targets',[])[:3],1): lines.append(f"TP{i}: {_p(t['price'])} ({t['kind']} liquidity, {t['rr']:.2f}R)")
    if not a.get('targets'): lines.append('TP: No qualifying liquidity target ≥1.2R')
    return '\n'.join(x for x in lines if x!='')


def _draw_label(ax,x,y,text,va='center'):
    ax.annotate(text,xy=(x,y),xytext=(x,y),fontsize=8,fontweight='bold',ha='left',va=va,bbox=dict(boxstyle='round,pad=.22',alpha=.78),zorder=20)


def generate_smc_chart(a,title_prefix='SMC'):
    df=a['df'].tail(140).copy(); df['ATR']=_atr(df); fig,axes=mpf.plot(df[['Open','High','Low','Close']],type='candle',style='charles',volume=False,returnfig=True,figsize=(14,8),warn_too_much_data=10000); ax=axes[0]; atr=float(df.ATR.iloc[-1]); n=len(df); offset=2
    lo=float(df.Low.min())-4*atr; hi=float(df.High.max())+4*atr
    for q in a.get('key_levels_4h',[]):
        px=float(q['price'])
        if lo<=px<=hi:
            ax.axhline(px,linestyle='--',linewidth=.9,alpha=.45); ax.text(n-1,px,f"4H {q.get('side','LEVEL')}",fontsize=7,ha='right',va='bottom')
    for q in a.get('liquidity',[])[:10]:
        px=float(q['price'])
        if lo<=px<=hi: ax.axhline(px,linestyle=':',linewidth=.7,alpha=.25)
    if a.get('dealing_range'): ax.axhline(a['dealing_range']['equilibrium'],linestyle='-.',linewidth=.8,alpha=.3); _draw_label(ax,2,a['dealing_range']['equilibrium'],'EQUILIBRIUM')

    # Real rectangular OB zones, including inducement and primary zones.
    for z in a.get('order_blocks',[]):
        left=int(z.get('formed_index',0)); right=n-1; width=max(1,right-left); bottom=float(z['bottom']); height=float(z['top']-z['bottom'])
        if right < 0 or bottom>hi or bottom+height<lo: continue
        face='#78c679' if z.get('type')=='bullish' else '#e34a33'; alpha=.18 if z.get('role')!='inducement' else .10
        ax.add_patch(Rectangle((left,bottom),width,height,facecolor=face,edgecolor=face,alpha=alpha,zorder=2))
        label=('INDUCEMENT OB' if z.get('is_inducement') else ('BULLISH OB' if z.get('type')=='bullish' else 'BEARISH OB'))
        ax.text(max(0,left+1),float(z['top']),label,fontsize=7,fontweight='bold',va='bottom',zorder=10)

    if a.get('fvg'):
        f=a['fvg']; left=max(0,int(f['i'])-2); width=max(1,n-left); ax.add_patch(Rectangle((left,f['low']),width,f['high']-f['low'],facecolor='#5dade2',edgecolor='#2980b9',alpha=.16,zorder=3)); ax.text(left+1,f['high'],'FVG',fontsize=8,fontweight='bold',va='bottom',zorder=12)
    if a.get('sweep'):
        s=a['sweep']; i=int(s['i']); px=float(s['price']); ax.axvline(i,linestyle='--',linewidth=.8,alpha=.35); _draw_label(ax,i,px,'SWEEP')
    if a.get('inducement'):
        ind=a['inducement']; i=int(ind.get('i',n-5)); px=float(ind['price']); _draw_label(ax,i,px,'INDUCEMENT')
    if a.get('choch'):
        ch=a['choch']; i=int(ch['i']); px=float(ch['level']); ax.axhline(px,linestyle='--',linewidth=.9,alpha=.45); _draw_label(ax,i,px,'CHoCH / MSS')
    if a.get('bos'):
        b=a['bos']; i=int(b['i']); px=float(b['level']); ax.axhline(px,linestyle='--',linewidth=.9,alpha=.55); _draw_label(ax,i,px,'BOS')
    if a.get('displacement'):
        i=int(a['displacement']['i']); y=float(df.High.iloc[min(i,len(df)-1)]); _draw_label(ax,i,y,'DISPLACEMENT',va='bottom')
    if a.get('confirmation_i') is not None:
        i=int(a['confirmation_i']); y=float(df.High.iloc[min(i,len(df)-1)]); _draw_label(ax,i,y,'CONFIRMATION',va='bottom')
    if a.get('entry') is not None:
        ax.axhline(float(a['entry']),linestyle='-',linewidth=.9,alpha=.35); ax.text(n-1,float(a['entry']),'ENTRY',fontsize=7,ha='right',va='bottom')
    if a.get('sl') is not None:
        ax.axhline(float(a['sl']),linestyle=':',linewidth=.8,alpha=.4); ax.text(n-1,float(a['sl']),'SL',fontsize=7,ha='right',va='bottom')
    for j,t in enumerate(a.get('targets',[])[:3],1):
        px=float(t['price']); ax.axhline(px,linestyle=':',linewidth=.8,alpha=.35); ax.text(n-1,px,f'TP{j}',fontsize=7,ha='right',va='bottom')
    ax.set_title(f"{a['symbol']} | M30 | {title_prefix} {a['direction']} | {a.get('setup_type','—')} | {a['status']}")
    fig.tight_layout(); out=BytesIO(); fig.savefig(out,format='png',dpi=160,bbox_inches='tight'); plt.close(fig); out.seek(0); return out


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
    return '\n'.join([f"🔀 HYBRID ANALYSIS — {h['symbol']} M30",'',f"SMC: {s.get('direction')} | {s.get('status')} | {s.get('score')}/100",f"Setup: {s.get('setup_type','—')}",f"Trendline: {t.get('direction','—')} | {t.get('strength',0)}/100",f"Liquidity sweep: {sw['event'] if sw else '—'}",f"Inducement: {_p(s['inducement']['price']) if s.get('inducement') else '—'}",f"Order Block: {o['type'] if o else '—'}",f"SMC waiting for: {s.get('waiting_for','—')}",'','━━━━━━━━━━━━━━━━','🎯 DECISION',f"SMC: {s.get('direction')}",f"Trendline: {t.get('direction')}",f"CONFLUENCE: {'AGREE' if h['agreement'] else 'CONFLICT'}",f"BIAS: {h['direction']}",f"STATUS: {h['status']}",f"HYBRID SCORE: {h['score']}/100",'', 'SMC and Trendline remain independent; conflict = WAIT.'])


def generate_hybrid_chart(h):
    if h.get('error') or not h.get('smc') or h['smc'].get('error'): return None
    return generate_smc_chart(h['smc'],title_prefix='HYBRID')
