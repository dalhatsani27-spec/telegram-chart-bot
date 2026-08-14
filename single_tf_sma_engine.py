"""Single-timeframe 20-SMA directional geometry analyzer.

The 20 SMA tells direction. It does NOT create the trendline.
After direction is established, a clean trendline is fitted on that direction:
  bullish/rising 20 SMA -> support rail from swing lows (green)
  bearish/falling 20 SMA -> resistance rail from swing highs (red)
Entry requires a trendline touch/rejection OR a confirming candle at the rail.
If no clean rail exists, the analyzer falls back to pattern, then S/R.
"""
from __future__ import annotations
import io
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

TF_LABELS = {"4h":"H4", "1h":"H1", "30min":"M30", "15min":"M15", "5min":"M5", "1min":"M1"}
SMA_PERIOD = 20


def _atr(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-prev).abs(), (df["Low"]-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=1).mean()


def _pivots(df: pd.DataFrame, span: int = 3) -> Tuple[List[Dict], List[Dict]]:
    highs, lows = [], []
    h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    for i in range(span, len(df)-span):
        if h[i] >= np.max(h[i-span:i+span+1]): highs.append({"index": i, "price": float(h[i])})
        if l[i] <= np.min(l[i-span:i+span+1]): lows.append({"index": i, "price": float(l[i])})
    return highs, lows


def _fit(points: List[Dict], df: pd.DataFrame) -> Optional[Dict]:
    if len(points) < 3: return None
    pts = points[-6:]
    xs = np.array([p["index"] for p in pts], float)
    ys = np.array([p["price"] for p in pts], float)
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    atr = float(df["ATR"].iloc[-1])
    resid = np.abs(ys - pred)
    touches = int(np.sum(resid <= max(atr * 0.45, 1e-9)))
    move = abs(slope * (xs[-1] - xs[0]))
    if move < atr * 0.30: return None
    return {
        "x0": int(xs[0]), "x1": int(xs[-1]),
        "y0": float(slope*xs[0] + intercept),
        "y1": float(slope*xs[-1] + intercept),
        "slope": float(slope), "touches": touches,
        "y_end": float(slope*(len(df)-1) + intercept),
        "points": [dict(p) for p in pts],
    }


def _levels(df, highs, lows):
    atr = float(df["ATR"].iloc[-1]); tol = max(atr*0.45, 1e-9)
    def cluster(vals):
        out=[]
        for v in sorted(vals):
            if not out or abs(v-out[-1]) > tol: out.append(v)
            else: out[-1] = (out[-1] + v) / 2
        return out
    return cluster([p["price"] for p in lows[-12:]]), cluster([p["price"] for p in highs[-12:]])


def _pattern(df, highs, lows):
    if len(highs)<2 or len(lows)<2: return None
    rh, rl = highs[-4:], lows[-4:]
    hs = np.polyfit([p["index"] for p in rh], [p["price"] for p in rh], 1)[0]
    ls = np.polyfit([p["index"] for p in rl], [p["price"] for p in rl], 1)[0]
    atr = float(df["ATR"].iloc[-1])
    if hs > 0 and ls > 0 and abs(hs) > atr/len(df)*0.2 and abs(ls) > atr/len(df)*0.2:
        return {"name":"ASCENDING CHANNEL", "confidence":82, "type":"channel"}
    if hs < 0 and ls < 0:
        return {"name":"DESCENDING CHANNEL", "confidence":82, "type":"channel"}
    if hs < 0 and ls > 0:
        return {"name":"SYMMETRICAL TRIANGLE", "confidence":84, "type":"triangle"}
    return None


def _entry_confirmation(df: pd.DataFrame, rail: Dict, direction: str, atr: float) -> Dict[str, Any]:
    """Evaluate the latest CLOSED candle against the directional rail.

    Entry is deliberately event-based: the market must interact with the rail.
    A mere SMA/rail proximity is never an entry by itself.
    """
    c = df.iloc[-1]
    rail_now = float(rail["y_end"])
    tolerance = max(atr * 0.20, 1e-9)
    body = abs(float(c["Close"] - c["Open"]))
    rng = max(float(c["High"] - c["Low"]), 1e-9)
    body_ratio = body / rng
    bullish_candle = c["Close"] > c["Open"] and body_ratio >= 0.40
    bearish_candle = c["Close"] < c["Open"] and body_ratio >= 0.40

    if direction == "BUY":
        touched = float(c["Low"]) <= rail_now + tolerance
        rejected = touched and float(c["Close"]) > rail_now and float(c["Close"]) >= float(c["Open"])
        candle_confirmation = touched and bullish_candle and float(c["Close"]) > rail_now
    elif direction == "SELL":
        touched = float(c["High"]) >= rail_now - tolerance
        rejected = touched and float(c["Close"]) < rail_now and float(c["Close"]) <= float(c["Open"])
        candle_confirmation = touched and bearish_candle and float(c["Close"]) < rail_now
    else:
        touched = rejected = candle_confirmation = False

    return {
        "touched": bool(touched),
        "rejected": bool(rejected),
        "candle_confirmation": bool(candle_confirmation),
        "confirmed": bool(rejected or candle_confirmation),
        "body_ratio": float(body_ratio),
        "rail_price": rail_now,
    }


def analyze(df: pd.DataFrame, symbol: str, tf: str) -> Dict[str, Any]:
    df = df.copy()
    # 20 SMA on Median Price is the directional compass.
    df["SMA20"] = ((df["High"] + df["Low"]) / 2.0).rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    df["ATR"] = _atr(df)
    df = df.dropna().copy()
    if len(df) < 45: return {"error": f"Insufficient {TF_LABELS.get(tf,tf)} candles"}

    sma_now = float(df["SMA20"].iloc[-1])
    sma_prev = float(df["SMA20"].iloc[-4])
    sma_slope = sma_now - sma_prev
    direction = "BUY" if sma_slope > 0 else "SELL" if sma_slope < 0 else "NEUTRAL"

    highs, lows = _pivots(df)
    # Direction chooses which side is allowed to become the primary rail.
    support = _fit(lows, df)
    resistance = _fit(highs, df)
    primary = support if direction == "BUY" else resistance if direction == "SELL" else None
    rail_kind = "support" if direction == "BUY" else "resistance" if direction == "SELL" else None

    atr = float(df["ATR"].iloc[-1])
    distance = None
    near = False
    entry = {"touched":False,"rejected":False,"candle_confirmation":False,"confirmed":False,"body_ratio":0.0,"rail_price":None}
    if primary:
        distance = abs(sma_now - primary["y_end"])
        near = distance <= atr * 0.35
        entry = _entry_confirmation(df, primary, direction, atr)

    # Pattern/SR are fallbacks only when there is no clean directional rail.
    pattern = _pattern(df, highs, lows) if primary is None else None
    supports, resistances = _levels(df, highs, lows)
    price = float(df["Close"].iloc[-1])
    bias = direction
    strong_direction = bool(primary and near and ((direction=="BUY" and primary["slope"]>0) or (direction=="SELL" and primary["slope"]<0)))
    entry_confirmed = bool(primary and entry["confirmed"])
    status = "ENTRY CONFIRMED" if entry_confirmed else "STRONG DIRECTION / WAIT ENTRY" if strong_direction else "DIRECTIONAL / RAIL SEPARATED" if primary else "PATTERN" if pattern else "S/R FALLBACK"

    return {
        "df":df,"symbol":symbol,"timeframe":tf,"timeframe_label":TF_LABELS.get(tf,tf),
        "sma":sma_now,"sma_slope":sma_slope,"direction":direction,"bias":bias,
        "trendline":primary,"trendline_kind":rail_kind,"distance":distance,"near":near,
        "strong":strong_direction,"entry":entry,"entry_confirmed":entry_confirmed,
        "pattern":pattern,"supports":supports,"resistances":resistances,"atr":atr,"price":price,
        "status":status,"high_pivots":highs,"low_pivots":lows,
    }


def report(r: Dict[str, Any]) -> str:
    if r.get("error"): return "❌ SINGLE-TF ANALYSIS\n\n" + r["error"]
    tl = r.get("trendline")
    lines = [f"📐 SINGLE-TF MARKET MAP — {r['symbol']}", "━━━━━━━━━━━━━━━━━━━━",
             f"ANALYSIS TIMEFRAME: {r['timeframe_label']}",
             "20 SMA: MEDIAN PRICE · 20 PERIOD",
             f"SMA DIRECTION: {'RISING 🟢' if r['direction']=='BUY' else 'FALLING 🔴' if r['direction']=='SELL' else 'FLAT ⚪'}"]
    if tl:
        colour = "🟢" if r["trendline_kind"] == "support" else "🔴"
        name = "BULLISH SUPPORT TRENDLINE" if r["trendline_kind"] == "support" else "BEARISH RESISTANCE TRENDLINE"
        e = r["entry"]
        lines += [f"TRENDLINE: {colour} {name}", f"TOUCHES: {tl['touches']}",
                  f"20 SMA ↔ TRENDLINE: {r['distance']:.4g} · {'NEAR/TOUCHING ✅' if r['near'] else 'SEPARATED ⚠️'}",
                  f"TRENDLINE TOUCH: {'YES ✅' if e['touched'] else 'NO'}",
                  f"REJECTION: {'CONFIRMED ✅' if e['rejected'] else 'WAIT'}",
                  f"CANDLE CONFIRMATION: {'CONFIRMED ✅' if e['candle_confirmation'] else 'WAIT'}"]
    elif r.get("pattern"):
        p=r["pattern"]; lines += [f"PATTERN: {p['name']}", f"PATTERN CONFIDENCE: {p['confidence']}%"]
    else: lines += ["TRENDLINE: NONE CLEAN", "STRUCTURE: S/R FALLBACK"]
    if r["supports"]: lines.append("SUPPORT: " + ", ".join(f"{x:.5f}" for x in r["supports"][-3:]))
    if r["resistances"]: lines.append("RESISTANCE: " + ", ".join(f"{x:.5f}" for x in r["resistances"][-3:]))
    lines += ["━━━━━━━━━━━━━━━━━━━━", f"BIAS: {r['bias']}", f"MARKET STATE: {r['status']}",
              f"ENTRY: {'🔥 CONFIRMED' if r['entry_confirmed'] else '⏳ WAIT'}"]
    return "\n".join(lines)


def render(r: Dict[str, Any]):
    import matplotlib.pyplot as plt
    df=r["df"].tail(140); offset=len(r["df"])-len(df)
    fig, ax=plt.subplots(figsize=(12,6.5)); x=np.arange(len(df))
    # Candles: use OHLC so the geometry is visible; do not replace candles with a close-only line.
    for i, (_, row) in enumerate(df.iterrows()):
        o,h,l,c=map(float,(row["Open"],row["High"],row["Low"],row["Close"]))
        ax.vlines(i,l,h,linewidth=0.8)
        lower, upper=min(o,c),max(o,c)
        ax.plot([i,i],[lower,upper],linewidth=3.0)
    ax.plot(x,df["SMA20"].values,label="20 SMA (Median)",linewidth=1.7)
    tl=r.get("trendline")
    if tl:
        color="green" if r["trendline_kind"]=="support" else "red"
        xs=np.array([tl["x0"]-offset,len(df)-1]); ys=np.array([tl["y0"],tl["y_end"]])
        ax.plot(xs,ys,color=color,linewidth=2.4,label=("Bullish Support" if color=="green" else "Bearish Resistance"))
        for p in tl["points"]:
            px=p["index"]-offset
            if 0<=px<len(df): ax.scatter(px,p["price"],s=22,color=color,zorder=5)
    for lv in r.get("supports",[])[:-1]: ax.axhspan(lv-r["atr"]*.12,lv+r["atr"]*.12,alpha=.10)
    for lv in r.get("resistances",[])[:-1]: ax.axhspan(lv-r["atr"]*.12,lv+r["atr"]*.12,alpha=.10)
    ax.set_title(f"{r['symbol']} {r['timeframe_label']} — {r['status']}"); ax.grid(alpha=.15); ax.legend(loc="upper left"); fig.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=140,bbox_inches="tight"); plt.close(fig); buf.seek(0); return buf
