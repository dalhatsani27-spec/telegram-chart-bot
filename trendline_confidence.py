"""
Trendline V4 confidence + final-decision engine.

This layer is deterministic and sits on top of the existing Trendline V3
geometry/pattern/structure detectors. It is intentionally NOT a prediction
model: confidence is an evidence-weighted quality score, while entry is
controlled by mandatory structural gates.
"""
from __future__ import annotations

from typing import Any, Dict


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def _pattern_score(family: Dict[str, Any], direction: str):
    sp = family.get("scanned_pattern") or {}
    name = str(sp.get("name") or "")
    bias = str(sp.get("bias") or "NEUTRAL").upper()
    raw = float(sp.get("confidence") or 0.0)
    stage = str(sp.get("stage") or family.get("pattern_stage") or "").upper()
    if raw > 0:
        score, detail = raw, name or "Named pattern"
    elif family.get("wedge"):
        w = family["wedge"]
        score, detail = 70.0, w.get("pattern", "Converging structure")
        bias = str(w.get("bias") or "NEUTRAL").upper()
    elif family.get("mw_pattern"):
        m = family["mw_pattern"]
        score, detail = 68.0, m.get("name", "M/W structure")
        bias = str(m.get("bias") or "NEUTRAL").upper()
    elif family.get("channel"):
        score, detail, bias = 58.0, "Trend channel", direction
    else:
        score, detail, bias = 35.0, "No named pattern", "NEUTRAL"
    if direction in ("BUY", "SELL") and bias in ("BUY", "SELL"):
        score += 10 if bias == direction else -15
    if stage == "CONFIRMED": score += 12
    elif stage == "TRIGGERED": score += 5
    elif stage == "FORMING": score -= 18
    elif stage == "FAKEOUT": score -= 35
    return _clamp(score), detail


def _geometry_score(family: Dict[str, Any]):
    quality = family.get("primary_quality")
    touches = int(family.get("primary_touches") or 0)
    line = (family.get("uptrends") or family.get("downtrends") or family.get("family_lines") or [None])[0]
    violations = int((line or {}).get("violations") or 0)
    base = {"confirmed": 88.0, "crowded": 70.0, "unconfirmed": 55.0}.get(quality, 48.0 if line else 32.0)
    base += min(10.0, max(0, touches - 2) * 4.0) - min(22.0, violations * 4.0)
    if family.get("channel"): base += 5
    if family.get("wedge"): base += 4
    return _clamp(base), f"{touches} touches" + (f", {violations} close violation(s)" if violations else ", no close violations")


def _breakout_score(family: Dict[str, Any]):
    brk = family.get("breakout_grade") or {}
    if not brk: return 55.0, "No fresh trendline breakout; continuation state"
    strength = str(brk.get("strength") or "weak").lower()
    score = {"confirmed": 95.0, "developing": 65.0, "weak": 32.0}.get(strength, 45.0)
    pen = float(brk.get("penetration_atr") or 0.0)
    consecutive = int(brk.get("consecutive_closes") or 0)
    body = float(brk.get("body_ratio") or 0.0)
    score += min(5.0, pen * 4.0) + min(5.0, max(0, consecutive - 1) * 3.0) + min(5.0, body * 4.0)
    return _clamp(score), f"{strength}, {pen:.2f} ATR penetration, {consecutive} close(s), body {body:.2f}"


def _confirmation_score(family: Dict[str, Any]):
    rules = family.get("entry_rules") or {}
    checks = rules.get("checks")
    if not checks: return 45.0, "Entry confirmation unavailable"
    passed = int(rules.get("passed") or 0); required = int(rules.get("required") or 3)
    score = (passed / max(4, len(checks))) * 100.0 + (8.0 if passed >= required else 0.0)
    return _clamp(score), f"{passed}/{len(checks)} checks passed (need {required}+)"


def _momentum_score(family: Dict[str, Any], direction: str):
    checks = (family.get("entry_rules") or {}).get("checks") or {}
    momentum = checks.get("momentum", (False, "n/a")); rsi = checks.get("rsi", (False, "n/a")); candle = checks.get("candle", (False, "n/a"))
    score = 40 + (25 if momentum[0] else 0) + (20 if rsi[0] else 0) + (15 if candle[0] else 0)
    return _clamp(score), f"Momentum={'PASS' if momentum[0] else 'WAIT'}; RSI={'PASS' if rsi[0] else 'WAIT'}; Candle={'PASS' if candle[0] else 'WAIT'}"


def _fib_score(family: Dict[str, Any], direction: str):
    df = family.get("df"); pivots = family.get("pivots") or []
    if df is None or len(pivots) < 2: return 35.0, "Insufficient impulse anchors"
    a, b = pivots[-2], pivots[-1]
    if a.get("type") == b.get("type"): return 35.0, "No clean alternating impulse"
    high, low = max(float(a["price"]), float(b["price"])), min(float(a["price"]), float(b["price"]))
    leg = high - low
    if leg <= 0: return 35.0, "Flat impulse"
    close = float(df["Close"].iloc[-1])
    retr = (high - close) / leg if b.get("type") == "high" else (close - low) / leg
    if direction == "BUY" and b.get("type") != "high": return 38.0, "Latest leg is not a bullish impulse"
    if direction == "SELL" and b.get("type") != "low": return 38.0, "Latest leg is not a bearish impulse"
    levels = (0.382, 0.50, 0.618, 0.786); nearest = min(levels, key=lambda r: abs(retr-r)); distance = abs(retr-nearest)
    score = 85 if distance <= .035 else 72 if distance <= .075 else 52 if distance <= .13 else 35
    return score, f"{retr*100:.1f}% retracement, nearest {nearest*100:.1f}%"


def _topdown_score(family: Dict[str, Any], direction: str):
    td = family.get("topdown") or {}
    if direction not in ("BUY", "SELL"): return 35.0, "No directional setup"
    bias4 = str(td.get("bias_4h") or "NEUTRAL").upper(); bias1 = str(td.get("direction") or "NEUTRAL").upper(); allowed = bool(td.get("allowed"))
    score = 50 + (20 if bias4 == direction else -10 if bias4 in ("BUY","SELL") else 0) + (20 if bias1 == direction else -10 if bias1 in ("BUY","SELL") else 0) + (10 if allowed and bias1 == direction else 0)
    return _clamp(score), f"4H={bias4}, 1H={bias1}, permission={'YES' if allowed else 'NO'}"


def _ob_score(family: Dict[str, Any], direction: str):
    active = family.get("active_order_block")
    if not active:
        wanted = "bullish" if direction == "BUY" else "bearish" if direction == "SELL" else None
        candidates = [o for o in (family.get("order_blocks") or []) if wanted and o.get("type") == wanted and o.get("freshness") == "untested"]
        active = candidates[0] if candidates else None
    if not active: return 50.0, "No active aligned order block"
    side = str(active.get("type") or ""); quality = float(active.get("confidence") or 50)
    aligned = (direction == "BUY" and side == "bullish") or (direction == "SELL" and side == "bearish")
    return _clamp(quality if aligned else 100-quality), f"{side} OB {'aligned' if aligned else 'opposite'}, {quality:.0f}% quality"


WEIGHTS = {"geometry":20,"pattern":20,"breakout":15,"momentum":10,"confirmation":10,"fibonacci":10,"topdown":10,"order_block":5}


def calculate_confidence(family: Dict[str, Any]) -> Dict[str, Any]:
    direction = str(family.get("direction") or "NEUTRAL").upper()
    specs = {"geometry":_geometry_score(family),"pattern":_pattern_score(family,direction),"breakout":_breakout_score(family),"momentum":_momentum_score(family,direction),"confirmation":_confirmation_score(family),"fibonacci":_fib_score(family,direction),"topdown":_topdown_score(family,direction),"order_block":_ob_score(family,direction)}
    weighted = {}
    for name,(score,detail) in specs.items():
        weighted[name]={"score":round(_clamp(score),1),"weight":WEIGHTS[name],"contribution":round(_clamp(score)*WEIGHTS[name]/100,1),"detail":detail}
    final=sum(v["contribution"] for v in weighted.values())
    if direction == "NEUTRAL": final=min(final,49)
    stage=str((family.get("scanned_pattern") or {}).get("stage") or family.get("pattern_stage") or "").upper()
    if stage == "FAKEOUT": final=min(final,34)
    elif stage == "FORMING": final=min(final,54)
    final=int(round(_clamp(final)))
    grade="HIGH" if final>=80 else "GOOD" if final>=70 else "MODERATE" if final>=60 else "LOW" if final>=50 else "WAIT"
    return {"score":final,"grade":grade,"direction":direction,"pattern":weighted["pattern"]["detail"],"components":weighted,"formula":"20% geometry + 20% pattern + 15% breakout + 10% momentum + 10% confirmation + 10% Fibonacci + 10% top-down + 5% OB","raw_scores":specs}


def _structure_gate(family: Dict[str, Any], direction: str):
    anns=[a for a in (family.get("trendline_annotations") or []) if str(a.get("label")) in {"HH","HL","LH","LL"}]
    labels=[str(a.get("label")) for a in anns[-6:]]
    kind=str(family.get("family_kind") or "none").lower()
    state=str(family.get("continuation_state") or "CONTINUATION").upper()
    # A broken descending line is a bullish transition only after HL/HH.
    # A broken ascending line is a bearish transition only after LH/LL.
    if state == "TRANSITION":
        if kind == "descending" and direction == "BUY":
            ok = "HL" in labels[-3:] and "HH" in labels[-2:]
            return ok, "bullish HL → HH structure" if ok else "bullish HL → HH structure not confirmed"
        if kind == "ascending" and direction == "SELL":
            ok = "LH" in labels[-3:] and "LL" in labels[-2:]
            return ok, "bearish LH → LL structure" if ok else "bearish LH → LL structure not confirmed"
        return False, "reversal structure not confirmed"
    if state == "REVERSAL_CONFIRMED": return True, "reversal structure confirmed"
    return True, "continuation structure intact"


def evaluate_final_decision(family: Dict[str, Any]) -> Dict[str, Any]:
    """Single final gate. Confidence ranks quality; mandatory gates decide entry."""
    direction=str(family.get("direction") or "NEUTRAL").upper()
    conf=family.get("confidence_breakdown") or calculate_confidence(family)
    lifecycle=family.get("trendline_retest") or {}
    life=str(lifecycle.get("status") or "INTACT").upper()
    state=str(family.get("continuation_state") or "CONTINUATION").upper()
    rules=family.get("entry_rules") or {}
    passed=int(rules.get("passed") or 0); required=int(rules.get("required") or 3)
    sp=family.get("scanned_pattern") or {}; stage=str(sp.get("stage") or family.get("pattern_stage") or "").upper()
    td=family.get("topdown") or {}; bias4=str(td.get("bias_4h") or "NEUTRAL").upper(); bias1=str(td.get("direction") or "NEUTRAL").upper()
    active_ob=family.get("active_order_block") or {}
    ob_side=str(active_ob.get("type") or "")
    gates=[]
    ok_direction=direction in ("BUY","SELL"); gates.append(("Direction",ok_direction,"direction is BUY/SELL" if ok_direction else "no trade direction"))
    structure_ok,structure_detail=_structure_gate(family,direction); gates.append(("Structure",structure_ok,structure_detail))
    life_ok = life not in ("FAKEOUT",) and (life == "BREAK_RETEST_CONFIRMED" or life == "INTACT")
    gates.append(("Break/retest",life_ok,f"trendline lifecycle = {life}"))
    confirm_ok=passed>=required; gates.append(("Entry confirmation",confirm_ok,f"{passed}/{len(rules.get('checks') or {}) or 4} checks; need {required}"))
    pattern_ok=stage in ("", "CONFIRMED", "") or stage not in ("FORMING","TRIGGERED","FAKEOUT")
    if sp: pattern_ok=stage=="CONFIRMED"
    gates.append(("Pattern",pattern_ok, f"{stage or 'no named pattern'}"))
    td_ok=(bias4 in ("NEUTRAL",direction) and bias1 in ("NEUTRAL",direction) and bool(td.get("allowed", True)))
    gates.append(("Top-down",td_ok,f"4H={bias4}, 1H={bias1}, permission={'YES' if td.get('allowed',True) else 'NO'}"))
    ob_ok=not active_ob or active_ob.get("is_inducement") or ((direction=="BUY" and ob_side!="bearish") or (direction=="SELL" and ob_side!="bullish"))
    gates.append(("Order block",ob_ok,"aligned/neutral" if ob_ok else f"opposite {ob_side} OB is active"))
    # Confidence is a quality threshold, not a probability. A very high score
    # cannot override a failed structural gate.
    conf_ok=int(conf.get("score",0))>=75; gates.append(("Confidence",conf_ok,f"{conf.get('score',0)}% {conf.get('grade','WAIT')} (minimum 75%)"))
    confirmed=all(x[1] for x in gates)
    failed=[f"{name}: {detail}" for name,ok,detail in gates if not ok]
    return {"confirmed":confirmed,"status":"CONFIRMED" if confirmed else "WAIT","direction":direction,"confidence":conf.get("score",0),"grade":conf.get("grade","WAIT"),"gates":gates,"failed":failed,"reason":"All mandatory confluence gates passed." if confirmed else (failed[0] if failed else "Mandatory confirmation pending."),"confidence_breakdown":conf}


def format_confidence_block(result: Dict[str, Any]) -> str:
    if not result: return ""
    labels={"geometry":"Geometry","pattern":"Pattern","breakout":"Breakout","momentum":"Momentum","confirmation":"Confirmation","fibonacci":"Fibonacci","topdown":"4H/1H Top-down","order_block":"Order Block"}
    lines=["════════════════════════════",f"🎯 FINAL CONFIDENCE: {result['score']}% · {result['grade']}",f"Direction: {result.get('direction','NEUTRAL')}",f"Pattern: {result.get('pattern','None')}","Confidence details:"]
    for key,item in result.get("components",{}).items():
        lines.append(f"  • {labels.get(key,key)}: {item['score']:.0f}/100 × {item['weight']}% = {item['contribution']:.1f}")
        lines.append(f"    {item['detail']}")
    lines.append(f"Method: {result.get('formula','')}")
    lines.append("Confidence is evidence-weighted, not a probability of profit.")
    lines.append("════════════════════════════")
    return "\n".join(lines)


def format_v4_report(family: Dict[str, Any], symbol: str) -> str:
    """Trader-facing report: evidence first, one final decision at the end."""
    if family.get("error"): return family["error"]
    conf=family.get("confidence_breakdown") or calculate_confidence(family)
    decision=evaluate_final_decision({**family,"confidence_breakdown":conf})
    td=family.get("topdown") or {}; direction=decision["direction"]
    bias4=str(td.get("bias_4h") or "NEUTRAL").upper(); bias1=str(td.get("direction") or "NEUTRAL").upper()
    structure_labels=[str(a.get("label")) for a in (family.get("trendline_annotations") or []) if a.get("label") in {"HH","HL","LH","LL"}][-6:]
    structure=" → ".join(structure_labels) if structure_labels else "—"
    life=family.get("trendline_retest") or {}; life_status=str(life.get("status") or "INTACT")
    wedge=family.get("wedge") or {}; sp=family.get("scanned_pattern") or {}
    pattern=str(sp.get("name") or wedge.get("pattern") or family.get("family_kind") or "NONE")
    pattern_stage=str(sp.get("stage") or family.get("pattern_stage") or "").upper()
    rules=family.get("entry_rules") or {}; checks=rules.get("checks") or {}
    lines=[f"📐 TRENDLINE V4 ANALYSIS — {symbol} M30","","BIAS",f"4H: {bias4}",f"1H: {bias1}",f"30M: {direction}","","STRUCTURE",f"Swings: {structure}",f"Market state: {str(family.get('continuation_state') or 'CONTINUATION').upper()}","","TRENDLINE",f"Type: {str(family.get('family_kind') or 'NONE').upper()}",f"Touches: {int(family.get('primary_touches') or 0)}",f"Lifecycle: {life_status}","","PATTERN",f"{pattern}: {pattern_stage or 'STRUCTURAL'}"]
    if wedge: lines.append(f"Falling/Rising wedge geometry: {str(wedge.get('bias') or 'NEUTRAL').upper()} · confirmed by geometry")
    lines += ["","CONFLUENCE",f"Breakout: {'✅' if life_status in ('BREAK_CONFIRMED','BREAK_RETEST_CONFIRMED') else '⏳'}",f"Retest: {'✅' if life_status=='BREAK_RETEST_CONFIRMED' else '⏳'}",f"Entry checks: {int(rules.get('passed') or 0)}/{len(checks) or 4} (need {int(rules.get('required') or 3)})",f"4H/1H permission: {'✅' if (bias4 in ('NEUTRAL',direction) and bias1 in ('NEUTRAL',direction)) else '⚠️'}"]
    lines += ["","━━━━━━━━━━━━━━━━","","🎯 FINAL DECISION",f"STATUS: {'🔥 ENTRY CONFIRMED' if decision['confirmed'] else '⏳ WAIT'}",f"BIAS: {direction}",f"CONFIDENCE: {decision['confidence']}% · {decision['grade']}"]
    if decision["confirmed"]:
        lines += ["","ALL REQUIRED CONFLUENCE: ✅","ENTRY GATE: PASSED"]
    else:
        lines += ["","FAILED/PENDING GATES:"] + [f"• {x}" for x in decision["failed"][:8]]
        lines += ["","No trade yet — confidence cannot override a failed mandatory gate."]
    lines += ["",format_confidence_block(conf)]
    return "\n".join(lines)
