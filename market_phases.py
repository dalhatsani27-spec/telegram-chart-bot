"""Market phase engine: accumulation, manipulation, distribution/markdown and transitions.

Uses observable OHLC behaviour rather than predicting intent. A phase is a
classification with evidence: compression, range acceptance, liquidity sweep,
displacement, directional expansion and structure progression.
"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df['High'].astype(float), df['Low'].astype(float), df['Close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _efficiency(df: pd.DataFrame, n: int = 20) -> float:
    c = df['Close'].astype(float).tail(n)
    if len(c) < 3: return 0.0
    path = c.diff().abs().sum()
    return float(abs(c.iloc[-1]-c.iloc[0]) / path) if path else 0.0


def _range_stats(df: pd.DataFrame, n: int = 30) -> Dict[str, float]:
    x = df.tail(n); hi=float(x['High'].max()); lo=float(x['Low'].min()); span=max(hi-lo,1e-9)
    close=float(df['Close'].iloc[-1])
    return {'high':hi,'low':lo,'span':span,'position':(close-lo)/span*100}


def analyze_market_phase(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or len(df) < 60:
        return {'phase':'UNKNOWN','sentiment':'NEUTRAL','confidence':0,'evidence':['Insufficient candles']}
    a=_atr(df); atr=float(a.iloc[-1]); rs=_range_stats(df); eff=_efficiency(df)
    recent=df.tail(20); body=(recent['Close']-recent['Open']).abs(); rng=(recent['High']-recent['Low']).replace(0,np.nan)
    body_ratio=float((body/rng).mean())
    compression=float(recent['High'].max()-recent['Low'].min())/max(float(df['High'].tail(60).max()-df['Low'].tail(60).min()),1e-9)
    slope=float(df['Close'].tail(20).iloc[-1]-df['Close'].tail(20).iloc[0])/max(atr*20,1e-9)
    prev_hi=float(df['High'].iloc[-25:-5].max()); prev_lo=float(df['Low'].iloc[-25:-5].min())
    last_hi=float(df['High'].iloc[-5:].max()); last_lo=float(df['Low'].iloc[-5:].min()); last_c=float(df['Close'].iloc[-1])
    sweep_high=last_hi>prev_hi and last_c<prev_hi
    sweep_low=last_lo<prev_lo and last_c>prev_lo
    expansion=abs(slope)>0.35 and eff>0.45 and body_ratio>0.50
    compression_state=compression<0.55 and eff<0.35

    if compression_state and rs['position']<65:
        phase='ACCUMULATION'; sentiment='BULLISH_BIAS'; evidence=['Volatility compression','Low directional efficiency','Price accepting lower/mid range']
    elif compression_state and rs['position']>35:
        phase='DISTRIBUTION'; sentiment='BEARISH_BIAS'; evidence=['Volatility compression','Low directional efficiency','Price accepting upper/mid range']
    elif sweep_low and expansion and slope>0:
        phase='MARKUP'; sentiment='BULLISH'; evidence=['Sell-side liquidity sweep','Bullish displacement/expansion','Directional efficiency increased']
    elif sweep_high and expansion and slope<0:
        phase='MARKDOWN'; sentiment='BEARISH'; evidence=['Buy-side liquidity sweep','Bearish displacement/expansion','Directional efficiency increased']
    elif expansion and slope>0:
        phase='MARKUP'; sentiment='BULLISH'; evidence=['Directional expansion','Bullish structure pressure']
    elif expansion and slope<0:
        phase='MARKDOWN'; sentiment='BEARISH'; evidence=['Directional expansion','Bearish structure pressure']
    elif sweep_low:
        phase='ACCUMULATION_TO_MARKUP'; sentiment='BULLISH_WATCH'; evidence=['Sell-side sweep','Reclaim of prior range low']
    elif sweep_high:
        phase='DISTRIBUTION_TO_MARKDOWN'; sentiment='BEARISH_WATCH'; evidence=['Buy-side sweep','Rejection of prior range high']
    else:
        phase='RE-ACCUMULATION/RE-DISTRIBUTION'; sentiment='NEUTRAL_WATCH'; evidence=['No decisive expansion','Market remains rotational']

    score=min(100,int(45 + abs(slope)*35 + eff*20))
    return {'phase':phase,'sentiment':sentiment,'confidence':score,'evidence':evidence,'range':rs,'metrics':{'atr':atr,'efficiency':eff,'body_ratio':body_ratio,'compression':compression,'slope':slope},'liquidity':{'sweep_high':bool(sweep_high),'sweep_low':bool(sweep_low)}}


def format_phase_report(result: Dict[str, Any]) -> str:
    return '\n'.join(['📊 MARKET PHASE ENGINE',f"Phase: {result['phase']}",f"Sentiment: {result['sentiment']}",f"Confidence: {result['confidence']}/100",'', 'Evidence:']+[f"• {x}" for x in result.get('evidence',[])])
