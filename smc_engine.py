"""Independent Smart Money Concepts analysis engine.

Uses OHLC-derived evidence only: structure, liquidity pools/sweeps,
displacement, validated OB/FVG candidates, dealing range and liquidity
targets. It does not claim direct visibility into institutional orders.
"""
from __future__ import annotations
from io import BytesIO
import numpy as np
import pandas as pd
import mplfinance as mpf
import market_data
from market_analysis import detect_confirmation_candle


def _atr(df, n=14):
    p=df.Close.shift(1)
    tr=pd.concat([df.High-df.Low,(df.High-p).abs(),(df.Low-p).abs()],axis=1).max(axis=1)
    return tr.rolling(n,min_periods=1).mean()


def _pivots(df, left=3, right=3):
    c=df.Close.to_numpy(float); out=[]
    for i in range(left,len(df)-right):
        w=c[i-left:i+right+1]
        if c[i]>=w.max() and c[i]>w.min(): out.append({'i':i,'price':float(df.High.iloc[i]),'type':'high'})
        elif c[i]<=w.min() and c[i]<w.max(): out.append({'i':i,'price':float(df.Low.iloc[i]),'type':'low'})
    return out


def _bias(p):
    h=[x for x in p if x['type']=='high']; l=[x for x in p if x['type']=='low']
    if len(h)<2 or len(l)<2:return 'NEUTRAL'
    if h[-1]['price']>h[-2]['price'] and l[-1]['price']>l[-2]['price']:return 'BULLISH'
    if h[-1]['price']<h[-2]['price'] and l[-1]['price']<l[-2]['price']:return 'BEARISH'
    return 'MIXED'


def _liquidity(p, tol):
    out=[]
    for typ,side in [('high','BSL'),('low','SSL')]:
        q=[x for x in p if x['type']==typ]
        for i,x in enumerate(q[-12:]):
            out.append({'side':side,'price':x['price'],'kind':'SWING','i':x['i']})
            if i and abs(x['price']-q[-12+i-1]['price'])<=tol:
                out.append({'side':side,'price':(x['price']+q[-12+i-1]['price'])/2,'kind':'EQUAL','i':x['i']})
    return sorted(out,key=lambda x:(x['kind']!='EQUAL', -x['i']))


def _sweep(df, pools, atr):
    x=df.iloc[-1]
    for p in pools:
        if p['side']=='SSL' and x.Low<p['price']-atr*.05 and x.Close>p['price']:return {**p,'event':'SSL_SWEEP'}
        if p['side']=='BSL' and x.High>p['price']+atr*.05 and x.Close<p['price']:return {**p,'event':'BSL_SWEEP'}
    return None


def _displacement(df, atr):
    for i in range(max(2,len(df)-8),len(df)):
        o,h,l,c=map(float,[df.Open.iloc[i],df.High.iloc[i],df.Low.iloc[i],df.Close.iloc[i]])
        if h-l>=1.25*float(atr.iloc[i]) and abs(c-o)/max(h-l,1e-9)>=.65:
            return {'i':i,'direction':'BUY' if c>o else 'SELL','atr':(h-l)/max(float(atr.iloc[i]),1e-9)}
    return None


def _event(df,p,bias):
    c=float(df.Close.iloc[-1]); h=[x for x in p if x['type']=='high']; l=[x for x in p if x['type']=='low']
    if bias=='BUY' and h and c>h[-1]['price']:return {'type':'BOS','direction':'BUY','level':h[-1]['price']}
    if bias=='SELL' and l and c<l[-1]['price']:return {'type':'BOS','direction':'SELL','level':l[-1]['price']}
    if h and c>h[-1]['price']:return {'type':'MSS','direction':'BUY','level':h[-1]['price']}
    if l and c<l[-1]['price']:return {'type':'MSS','direction':'SELL','level':l[-1]['price']}
    return None


def _ob(df,disp):
    if not disp:return None
    for i in range(disp['i']-1,max(-1,disp['i']-7),-1):
        o,c=float(df.Open.iloc[i]),float(df.Close.iloc[i])
        if disp['direction']=='BUY' and c<o:return {'type':'Bullish OB','low':float(df.Low.iloc[i]),'high':float(df.High.iloc[i]),'i':i,'fresh':True}
        if disp['direction']=='SELL' and c>o:return {'type':'Bearish OB','low':float(df.Low.iloc[i]),'high':float(df.High.iloc[i]),'i':i,'fresh':True}
    return None


def _fvg(df,atr,direction):
    for i in range(len(df)-1,1,-1):
        h2,l2=float(df.High.iloc[i-2]),float(df.Low.iloc[i-2]); h,l=float(df.High.iloc[i]),float(df.Low.iloc[i]); a=float(atr.iloc[i])
        if direction=='BUY' and l>h2 and l-h2>=.10*a:return {'low':h2,'high':l,'direction':'BUY','i':i}
        if direction=='SELL' and h<l2 and l2-h>=.10*a:return {'low':h,'high':l2,'direction':'SELL','i':i}
    return None


def _range(df,p):
    h=[x['price'] for x in p if x['type']=='high']; l=[x['price'] for x in p if x['type']=='low']; hi=max(h[-3:]) if h else float(df.High.max()); lo=min(l[-3:]) if l else float(df.Low.min()); eq=(hi+lo)/2; px=float(df.Close.iloc[-1])
    return {'high':hi,'low':lo,'equilibrium':eq,'location':'PREMIUM' if px>eq else 'DISCOUNT' if px<eq else 'EQUILIBRIUM'}


def _targets(pools, price, direction, risk):
    side='BSL' if direction=='BUY' else 'SSL'; q=[p for p in pools if p['side']==side and (p['price']>price if direction=='BUY' else p['price']<price)]
    q.sort(key=lambda x:abs(x['price']-price)); return [{**p,'rr':abs(p['price']-price)/max(risk,1e-9)} for p in q if abs(p['price']-price)/max(risk,1e-9)>=1.2][:3]


def run_smc_analysis(symbol, count=260):
    symbol=str(symbol).strip().upper()
    try:
        h4=market_data.fetch_candles(symbol,'4h',count); h1=market_data.fetch_candles(symbol,'1h',count); m30=market_data.fetch_candles(symbol,'30min',count)
        if any(x is None or len(x)<40 for x in (h4,h1,m30)):return {'error':f'Insufficient data for {symbol}.'}
        for x in (h4,h1,m30):x['ATR']=_atr(x)
        p4,p1,p30=_pivots(h4),_pivots(h1),_pivots(m30); b4,b1,b30=map(_bias,(p4,p1,p30))
        direction='BUY' if b4=='BULLISH' and b1!='BEARISH' else 'SELL' if b4=='BEARISH' and b1!='BULLISH' else ('BUY' if b30=='BULLISH' else 'SELL' if b30=='BEARISH' else 'WAIT')
        atr=float(m30.ATR.iloc[-1]); pools=_liquidity(p4+p1+p30,max(atr*.30,1e-9)); sweep=_sweep(m30,pools,atr); disp=_displacement(m30,m30.ATR); event=_event(m30,p30,direction); ob=_ob(m30,disp); fvg=_fvg(m30,m30.ATR,direction); dr=_range(h4,p4); entry=float(m30.Close.iloc[-1])
        if ob: sl=ob['low']-atr*.15 if direction=='BUY' else ob['high']+atr*.15
        else: sl=entry-atr*1.5 if direction=='BUY' else entry+atr*1.5
        risk=abs(entry-sl); targets=_targets(pools,entry,direction,risk); ok,name=detect_confirmation_candle(m30,direction) if direction in ('BUY','SELL') else (False,None)
        score=(20 if b4==('BULLISH' if direction=='BUY' else 'BEARISH') else 0)+(15 if b1==('BULLISH' if direction=='BUY' else 'BEARISH') else 0)+(20 if sweep and sweep['event']==('SSL_SWEEP' if direction=='BUY' else 'BSL_SWEEP') else 0)+(15 if disp and disp['direction']==direction else 0)+(10 if event and event['direction']==direction else 0)+(10 if ob else 0)+(5 if fvg else 0)+(5 if ok else 0)
        return {'symbol':symbol,'df':m30,'direction':direction,'status':'CONFIRMED' if score>=70 and (event or sweep) and ob else 'WAIT','score':min(score,100),'bias_4h':b4,'bias_1h':b1,'bias_30m':b30,'dealing_range':dr,'liquidity':pools,'sweep':sweep,'displacement':disp,'structure_event':event,'order_block':ob,'fvg':fvg,'candle_confirmation':name if ok else None,'entry':entry,'sl':sl,'risk':risk,'targets':targets}
    except Exception as e:return {'error':f'SMC analysis failed for {symbol}: {e}'}


def format_smc_report(a):
    if a.get('error'):return '🏦 SMC ANALYSIS\n\n❌ '+a['error']
    def p(x):return f'{x:.5f}' if isinstance(x,(int,float)) else '—'
    d=a['dealing_range']; s=a.get('sweep'); o=a.get('order_block'); f=a.get('fvg'); e=a.get('structure_event'); x=a.get('displacement')
    L=[f"🏦 SMC ANALYSIS — {a['symbol']} M30",'', 'HTF CONTEXT',f"4H: {a['bias_4h']}",f"1H: {a['bias_1h']}",f"30M: {a['bias_30m']}",f"Dealing Range: {p(d['low'])} — {p(d['high'])}",f"Equilibrium: {p(d['equilibrium'])}",f"Location: {d['location']}",'','LIQUIDITY',f"Sweep: {s['event'] if s else '—'}",f"BSL: {p(next((q['price'] for q in a['liquidity'] if q['side']=='BSL'),None))}",f"SSL: {p(next((q['price'] for q in a['liquidity'] if q['side']=='SSL'),None))}",'','STRUCTURE',f"Event: {e['type']+' '+e['direction'] if e else 'NOT CONFIRMED'}",f"Displacement: {'CONFIRMED' if x and x['direction']==a['direction'] else '—'}",'','ORDER FLOW',f"OB: {o['type']} {p(o['low'])} — {p(o['high'])}" if o else 'OB: —',f"FVG: {p(f['low'])} — {p(f['high'])}" if f else 'FVG: —',f"Candle: {a.get('candle_confirmation') or '—'}",'','━━━━━━━━━━━━━━━━','🎯 DECISION',f"BIAS: {a['direction']}",f"STATUS: {a['status']}",f"SMC SCORE: {a['score']}/100",f"ENTRY: {p(a['entry'])}",f"SL: {p(a['sl'])}"]
    for i,t in enumerate(a.get('targets',[])[:3],1):L.append(f"TP{i}: {p(t['price'])} ({t['kind']} liquidity, {t['rr']:.2f}R)")
    if not a.get('targets'):L.append('TP: No qualifying liquidity target ≥1.2R')
    return '\n'.join(L)


def generate_smc_chart(a):
    df=a['df'].tail(110); fig,axes=mpf.plot(df[['Open','High','Low','Close']],type='candle',style='charles',volume=False,returnfig=True,figsize=(12,7),warn_too_much_data=10000); ax=axes[0]; atr=float(df.ATR.iloc[-1]); n=len(df)
    for q in a.get('liquidity',[])[:8]:
        if df.Low.min()-5*atr<=q['price']<=df.High.max()+5*atr:ax.axhline(q['price'],linestyle=':',linewidth=.8,alpha=.4);ax.text(n-2,q['price'],f" {q['side']} {q['kind']}",fontsize=6.5,ha='right')
    o=a.get('order_block'); f=a.get('fvg')
    if o:ax.axhspan(o['low'],o['high'],alpha=.18);ax.text(2,o['high'],o['type'],fontsize=7,fontweight='bold')
    if f:ax.axhspan(f['low'],f['high'],alpha=.12);ax.text(2,f['high'],'FVG',fontsize=7,fontweight='bold')
    ax.set_title(f"{a['symbol']} | M30 | SMC {a['direction']} | {a['status']}");fig.tight_layout();out=BytesIO();fig.savefig(out,format='png',dpi=150,bbox_inches='tight');out.seek(0);return out


def run_hybrid_analysis(symbol):
    import strategies
    s=run_smc_analysis(symbol); t=strategies.run_trendline_analysis(symbol)
    if s.get('error') or t.get('error'):return {'error':s.get('error') or t.get('error'),'smc':s,'trendline':t}
    agree=s.get('direction') in ('BUY','SELL') and s.get('direction')==t.get('direction'); score=min(100,int((s.get('score',0)+t.get('strength',0))/2+(15 if agree else 0)))
    return {'symbol':symbol,'direction':s['direction'] if agree else 'WAIT','status':'CONFIRMED' if agree and s['status']=='CONFIRMED' else 'WAIT','score':score,'agreement':agree,'smc':s,'trendline':t}


def format_hybrid_report(h):
    if h.get('error'):return '🔀 HYBRID ANALYSIS\n\n❌ '+h['error']
    s,t=h['smc'],h['trendline']; sw=s.get('sweep')
    return '\n'.join([f"🔀 HYBRID ANALYSIS — {h['symbol']} M30",'',f"SMC: {s.get('direction')} | {s.get('status')} | {s.get('score')}/100",f"Sweep: {sw['event'] if sw else '—'}",f"OB: {s.get('order_block',{}).get('type','—') if s.get('order_block') else '—'}",f"Trendline: {t.get('direction','—')} | {t.get('strength',0)}/100",'', '━━━━━━━━━━━━━━━━','🎯 DECISION',f"SMC: {s.get('direction')}",f"Trendline: {t.get('direction')}",f"CONFLUENCE: {'AGREE' if h['agreement'] else 'CONFLICT'}",f"BIAS: {h['direction']}",f"STATUS: {h['status']}",f"HYBRID SCORE: {h['score']}/100",'', 'SMC and Trendline remain independent; conflict = WAIT.'])
