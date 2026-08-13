from __future__ import annotations
from io import BytesIO
from typing import Any
import mplfinance as mpf
import numpy as np
import pandas as pd
import market_data
from market_analysis import detect_confirmation_candle
from topdown_engine import get_topdown_bias

def _atr(df,n=14):
    p=df.Close.shift(1); tr=pd.concat([df.High-df.Low,(df.High-p).abs(),(df.Low-p).abs()],axis=1).max(axis=1); return tr.rolling(n,min_periods=1).mean()

def _pivots(df,left=3,right=3,min_leg_atr=.55):
    if len(df)<left+right+10:return []
    a=_atr(df);c=df.Close.to_numpy(float);h=df.High.to_numpy(float);l=df.Low.to_numpy(float);out=[]
    for i in range(left,len(df)-right):
        w=c[i-left:i+right+1];typ='high' if c[i]>=w.max() and c[i]>w.min() else 'low' if c[i]<=w.min() and c[i]<w.max() else None
        if not typ:continue
        if out and out[-1]['type']==typ:
            better=c[i]>out[-1]['close'] if typ=='high' else c[i]<out[-1]['close']
            if better:out[-1]={'i':i,'price':h[i] if typ=='high' else l[i],'close':c[i],'type':typ}
            continue
        if out and abs(c[i]-out[-1]['close'])/max(float(a.iloc[i]),1e-9)<min_leg_atr:continue
        out.append({'i':i,'price':h[i] if typ=='high' else l[i],'close':c[i],'type':typ})
    return out[-24:]

def _bias(p):
    hs=[x for x in p if x['type']=='high'];ls=[x for x in p if x['type']=='low']
    if len(hs)<2 or len(ls)<2:return 'NEUTRAL'
    if hs[-1]['close']>hs[-2]['close'] and ls[-1]['close']>ls[-2]['close']:return 'BULLISH'
    if hs[-1]['close']<hs[-2]['close'] and ls[-1]['close']<ls[-2]['close']:return 'BEARISH'
    return 'TRANSITION'

def _liquidity(p,atr):
    tol=max(atr*.2,1e-9);out=[]
    for typ,side in [('high','BSL'),('low','SSL')]:
        q=[x for x in p if x['type']==typ]
        for x in q[-12:]:out.append({'side':side,'price':x['price'],'kind':'SWING','i':x['i']})
        for a,b in zip(q[-12:-1],q[-11:]):
            if abs(a['price']-b['price'])<=tol:out.append({'side':side,'price':(a['price']+b['price'])/2,'kind':'EQUAL','i':b['i']})
    return sorted(out,key=lambda x:x['i'],reverse=True)

def _sweep(df,pools,atr):
    x=df.iloc[-1]
    for p in pools:
        if p['side']=='SSL' and x.Low<p['price']-atr*.03 and x.Close>p['price']:return {**p,'event':'SSL_SWEEP'}
        if p['side']=='BSL' and x.High>p['price']+atr*.03 and x.Close<p['price']:return {**p,'event':'BSL_SWEEP'}
    return None

def _displacement(df,atr):
    for i in range(max(1,len(df)-8),len(df)):
        o,h,l,c=map(float,[df.Open.iloc[i],df.High.iloc[i],df.Low.iloc[i],df.Close.iloc[i]]);rng=max(h-l,1e-9);body=abs(c-o);a=max(float(atr.iloc[i]),1e-9)
        if rng>=1.25*a and body/rng>=.65:return {'i':i,'direction':'BUY' if c>o else 'SELL','atr_multiple':rng/a}
    return None

def _event(df,p,bias):
    c=float(df.Close.iloc[-1]);hs=[x for x in p if x['type']=='high'];ls=[x for x in p if x['type']=='low']
    if bias in ('BUY','BULLISH') and hs and c>hs[-1]['price']:return {'type':'BOS','direction':'BUY','level':hs[-1]['price']}
    if bias in ('SELL','BEARISH') and ls and c<ls[-1]['price']:return {'type':'BOS','direction':'SELL','level':ls[-1]['price']}
    if hs and c>hs[-1]['price']:return {'type':'CHoCH','direction':'BUY','level':hs[-1]['price']}
    if ls and c<ls[-1]['price']:return {'type':'CHoCH','direction':'SELL','level':ls[-1]['price']}
    return None

def _ob(df,disp,atr):
    if not disp:return None
    for i in range(disp['i']-1,max(-1,disp['i']-8),-1):
        o,h,l,c=map(float,[df.Open.iloc[i],df.High.iloc[i],df.Low.iloc[i],df.Close.iloc[i]])
        if h-l<.25*atr:continue
        if (disp['direction']=='BUY' and c<o) or (disp['direction']=='SELL' and c>o):return {'type':'Bullish OB' if disp['direction']=='BUY' else 'Bearish OB','low':l,'high':h,'i':i,'fresh':True,'direction':disp['direction']}
    return None

def _fvg(df,atr,direction):
    if direction not in ('BUY','SELL'):return None
    for i in range(len(df)-1,1,-1):
        h2,l2=float(df.High.iloc[i-2]),float(df.Low.iloc[i-2]);h,l=float(df.High.iloc[i]),float(df.Low.iloc[i])
        if direction=='BUY' and l>h2 and l-h2>=.1*float(atr.iloc[i]):return {'low':h2,'high':l,'direction':'BUY','i':i}
        if direction=='SELL' and h<l2 and l2-h>=.1*float(atr.iloc[i]):return {'low':h,'high':l2,'direction':'SELL','i':i}
    return None

def _range(df,p):
    hs=[x['price'] for x in p if x['type']=='high'];ls=[x['price'] for x in p if x['type']=='low'];hi=max(hs[-3:]) if hs else float(df.High.tail(80).max());lo=min(ls[-3:]) if ls else float(df.Low.tail(80).min());eq=(hi+lo)/2;px=float(df.Close.iloc[-1]);return {'high':hi,'low':lo,'equilibrium':eq,'location':'PREMIUM' if px>eq else 'DISCOUNT' if px<eq else 'EQUILIBRIUM'}

def _targets(pools,price,direction,risk):
    side='BSL' if direction=='BUY' else 'SSL';q=[p for p in pools if p['side']==side and (p['price']>price if direction=='BUY' else p['price']<price)];q.sort(key=lambda x:abs(x['price']-price));return [{**p,'rr':abs(p['price']-price)/max(risk,1e-9)} for p in q if abs(p['price']-price)/max(risk,1e-9)>=1.2][:3]

def run_smc_analysis(symbol,count=260):
    symbol=str(symbol).strip().upper()
    try:
        h4=market_data.fetch_candles(symbol,'4h',count);h1=market_data.fetch_candles(symbol,'1h',count);m30=market_data.fetch_candles(symbol,'30min',count)
        if any(x is None or len(x)<50 for x in (h4,h1,m30)):return {'error':f'Insufficient data for {symbol}.'}
        for x in (h4,h1,m30):x['ATR']=_atr(x)
        td=get_topdown_bias(symbol,count_4h=count,count_1h=count);p4,p1,p30=_pivots(h4),_pivots(h1),_pivots(m30);b4,b1,b30=map(_bias,(p4,p1,p30));direction=td.get('direction') if td.get('direction') in ('BUY','SELL') else ('BUY' if td.get('bias_4h')=='BUY' and b30=='BULLISH' else 'SELL' if td.get('bias_4h')=='SELL' and b30=='BEARISH' else 'WAIT')
        atr=float(m30.ATR.iloc[-1]);pools=_liquidity(p4+p1+p30,atr);sweep=_sweep(m30,pools,atr);disp=_displacement(m30,m30.ATR);ev=_event(m30,p30,direction if direction!='WAIT' else b30)
        if ev and ev['type']=='CHoCH' and disp and disp['direction']==ev['direction']:ev={**ev,'type':'MSS'}
        ob=_ob(m30,disp,atr);fvg=_fvg(m30,m30.ATR,direction);dr=_range(h4,p4);entry=float(m30.Close.iloc[-1]);sl=ob['low']-atr*.15 if direction=='BUY' and ob else ob['high']+atr*.15 if direction=='SELL' and ob else entry-atr*1.5 if direction=='BUY' else entry+atr*1.5 if direction=='SELL' else entry;risk=abs(entry-sl);targets=_targets(pools,entry,direction,risk) if direction in ('BUY','SELL') else [];ok,candle=detect_confirmation_candle(m30,direction) if direction in ('BUY','SELL') else (False,None)
        a4=(direction=='BUY' and b4=='BULLISH') or (direction=='SELL' and b4=='BEARISH');a1=(direction=='BUY' and b1=='BULLISH') or (direction=='SELL' and b1=='BEARISH');sw=bool(sweep and sweep['event']==('SSL_SWEEP' if direction=='BUY' else 'BSL_SWEEP'));dp=bool(disp and disp['direction']==direction);es=bool(ev and ev['direction']==direction and ev['type'] in ('BOS','MSS'));bo=bool(ob and ob['direction']==direction);fv=bool(fvg);score=sum([20 if a4 else 0,15 if a1 else 0,15 if sw else 0,15 if dp else 0,15 if es else 0,10 if bo else 0,5 if fv else 0,5 if ok else 0])
        return {'symbol':symbol,'df':m30,'direction':direction,'status':'CONFIRMED' if direction in ('BUY','SELL') and score>=70 and es and dp and bo else 'WAIT','score':min(score,100),'bias_4h':b4,'bias_1h':b1,'bias_30m':b30,'topdown':td,'dealing_range':dr,'liquidity':pools,'sweep':sweep,'displacement':disp,'structure_event':ev,'order_block':ob,'fvg':fvg,'candle_confirmation':candle if ok else None,'entry':entry,'sl':sl,'risk':risk,'targets':targets,'key_levels_4h':td.get('key_levels_4h',[])}
    except Exception as e:return {'error':f'SMC analysis failed for {symbol}: {e}'}

def _p(x):return f'{float(x):.5f}' if isinstance(x,(int,float,np.floating)) else '—'

def format_smc_report(a):
    if a.get('error'):return '🏦 SMC ANALYSIS\n\n❌ '+a['error']
    dr=a['dealing_range'];s=a.get('sweep');o=a.get('order_block');f=a.get('fvg');e=a.get('structure_event');d=a.get('displacement');L=[f"🏦 SMC ANALYSIS — {a['symbol']} M30",'', 'TOP-DOWN CONTEXT',f"4H structure: {a['bias_4h']}",f"1H structure: {a['bias_1h']}",f"30M structure: {a['bias_30m']}",f"4H dealing range: {_p(dr['low'])} — {_p(dr['high'])}",f"Equilibrium: {_p(dr['equilibrium'])}",f"Price location: {dr['location']}",'','LIQUIDITY',f"Sweep: {s['event'] if s else '—'}",f"Nearest BSL: {_p(next((q['price'] for q in a['liquidity'] if q['side']=='BSL'),None))}",f"Nearest SSL: {_p(next((q['price'] for q in a['liquidity'] if q['side']=='SSL'),None))}",'','STRUCTURE / ORDER FLOW',f"Event: {e['type']} {e['direction']} @ {_p(e['level'])}" if e else 'Event: —',f"Displacement: {d['direction']} ({d['atr_multiple']:.2f} ATR)" if d else 'Displacement: —',f"Order Block: {o['type']} {_p(o['low'])} — {_p(o['high'])}" if o else 'Order Block: —',f"FVG: {_p(f['low'])} — {_p(f['high'])}" if f else 'FVG: —',f"Candle confirmation: {a.get('candle_confirmation') or '—'}",'','━━━━━━━━━━━━━━━━','🎯 DECISION',f"BIAS: {a['direction']}",f"STATUS: {a['status']}",f"SMC SCORE: {a['score']}/100",f"ENTRY: {_p(a['entry'])}",f"SL: {_p(a['sl'])}"]
    for i,t in enumerate(a.get('targets',[])[:3],1):L.append(f"TP{i}: {_p(t['price'])} ({t['kind']} liquidity, {t['rr']:.2f}R)")
    if not a.get('targets'):L.append('TP: No qualifying liquidity target ≥1.2R')
    return '\n'.join(L)

def generate_smc_chart(a):
    df=a['df'].tail(120).copy();df['ATR']=_atr(df);fig,axes=mpf.plot(df[['Open','High','Low','Close']],type='candle',style='charles',volume=False,returnfig=True,figsize=(12,7),warn_too_much_data=10000);ax=axes[0];atr=float(df.ATR.iloc[-1]);n=len(df)
    for q in a.get('key_levels_4h',[]):
        px=float(q['price'])
        if df.Low.min()-5*atr<=px<=df.High.max()+5*atr:ax.axhline(px,linestyle='--',linewidth=.9,alpha=.55);ax.text(n-1,px,f"  4H {q['side']} ({q['touches']})",fontsize=6.5,ha='right')
    for q in a.get('liquidity',[])[:6]:
        px=float(q['price'])
        if df.Low.min()-4*atr<=px<=df.High.max()+4*atr:ax.axhline(px,linestyle=':',linewidth=.7,alpha=.28)
    if a.get('dealing_range'):ax.axhline(a['dealing_range']['equilibrium'],linestyle='-.',linewidth=.7,alpha=.35)
    if a.get('order_block'):o=a['order_block'];ax.axhspan(o['low'],o['high'],alpha=.18);ax.text(2,o['high'],o['type'],fontsize=7,fontweight='bold')
    if a.get('fvg'):f=a['fvg'];ax.axhspan(f['low'],f['high'],alpha=.12);ax.text(2,f['high'],'FVG',fontsize=7,fontweight='bold')
    ax.set_title(f"{a['symbol']} | M30 | SMC {a['direction']} | {a['status']}");fig.tight_layout();out=BytesIO();fig.savefig(out,format='png',dpi=150,bbox_inches='tight');out.seek(0);return out

def run_hybrid_analysis(symbol):
    import strategies
    s=run_smc_analysis(symbol);t=strategies.run_trendline_analysis(symbol)
    if s.get('error') or t.get('error'):return {'error':s.get('error') or t.get('error'),'smc':s,'trendline':t}
    agree=s.get('direction') in ('BUY','SELL') and s.get('direction')==t.get('direction');score=min(100,int((s.get('score',0)+t.get('strength',0))/2+(15 if agree else 0)));return {'symbol':symbol,'direction':s['direction'] if agree else 'WAIT','status':'CONFIRMED' if agree and s['status']=='CONFIRMED' else 'WAIT','score':score,'agreement':agree,'smc':s,'trendline':t}

def format_hybrid_report(h):
    if h.get('error'):return '🔀 HYBRID ANALYSIS\n\n❌ '+h['error']
    s,t=h['smc'],h['trendline'];sw=s.get('sweep');o=s.get('order_block');return '\n'.join([f"🔀 HYBRID ANALYSIS — {h['symbol']} M30",'',f"SMC: {s.get('direction')} | {s.get('status')} | {s.get('score')}/100",f"Trendline: {t.get('direction','—')} | {t.get('strength',0)}/100",f"Liquidity sweep: {sw['event'] if sw else '—'}",f"Order Block: {o['type'] if o else '—'}",'','━━━━━━━━━━━━━━━━','🎯 DECISION',f"SMC: {s.get('direction')}",f"Trendline: {t.get('direction')}",f"CONFLUENCE: {'AGREE' if h['agreement'] else 'CONFLICT'}",f"BIAS: {h['direction']}",f"STATUS: {h['status']}",f"HYBRID SCORE: {h['score']}/100",'','SMC and Trendline remain independent; conflict = WAIT.'])
