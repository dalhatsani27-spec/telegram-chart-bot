"""Runtime compatibility patch for the chart-analysis bot.

Python imports sitecustomize automatically when the repository is on sys.path.
The patch keeps the main strategy files stable while fixing stateful execution
logic and the Trendline chart rendering path.
"""
from __future__ import annotations

import ast
import inspect
import textwrap


def _install_analysis_patch() -> None:
    try:
        import strategies
    except Exception as exc:
        print(f"[sitecustomize] strategies patch deferred: {exc!r}")
        return

    def stateful_confirmation(df, direction, trigger_index=None):
        checks = {"candle": (False, "no candle data"), "structure": (False, "no structure data"),
                  "momentum": (False, "no momentum data"), "rsi": (False, "no RSI data")}
        result = {"checks": checks, "passed": 0, "required": 3, "confirmed": False,
                  "event_index": None, "age_bars": None,
                  "note": "No valid confirmation event in the execution window."}
        if df is None or len(df) < 25 or direction not in ("BUY", "SELL"):
            return result
        start = max(20, int(trigger_index)) if trigger_index is not None else max(20, len(df) - 10)
        best = None
        for i in range(start, len(df)):
            sub = df.iloc[: i + 1].copy()
            local = {k: (False, v[1]) for k, v in checks.items()}
            try:
                found, name = strategies.detect_confirmation_candle(sub, direction)
                local["candle"] = (bool(found), name or "no matching confirmation candle")
            except Exception:
                pass
            try:
                minor = strategies.analyse_structure(sub, left=2, right=2, lookback=min(30, len(sub) - 1))
                want = "BULLISH" if direction == "BUY" else "BEARISH"
                local["structure"] = (minor.get("last_event") in ("BOS", "MSS") and minor.get("event_bias") == want,
                                       minor.get("note") or "no confirming minor BOS/MSS")
            except Exception:
                pass
            try:
                if "Volume" in sub.columns and sub["Volume"].tail(20).sum() > 0:
                    rv = float(sub["Volume"].iloc[-1]); av = float(sub["Volume"].iloc[-11:-1].mean()) if len(sub) > 11 else rv
                    local["momentum"] = (rv > av * 1.05, f"vol {rv:.0f} vs 10-bar avg {av:.0f}")
                elif len(sub) > 6:
                    roc = float((sub["Close"].iloc[-1] - sub["Close"].iloc[-6]) / sub["Close"].iloc[-6] * 100)
                    local["momentum"] = (roc > 0.05 if direction == "BUY" else roc < -0.05,
                                          f"5-bar RoC {roc:+.2f}% (no volume feed)")
            except Exception:
                pass
            try:
                if "RSI" in sub.columns:
                    rsi = float(sub["RSI"].iloc[-1])
                    local["rsi"] = (rsi > 50 if direction == "BUY" else rsi < 50, f"RSI {rsi:.1f}")
            except Exception:
                pass
            passed = sum(1 for ok, _ in local.values() if ok)
            if passed >= 3:
                best = (i, local, passed)
        if best is None:
            return result
        event_i, local, passed = best
        age = (len(df) - 1) - event_i
        if age > 18:
            return result
        result.update(checks=local, passed=passed, confirmed=True, event_index=int(event_i), age_bars=int(age),
                      note=f"Confirmation retained from bar {event_i} ({age} bars ago).")
        return result

    strategies._entry_confirmation = stateful_confirmation

    original_fit_primary = strategies._fit_primary
    def fit_primary_with_fallback(pivots, kind, n, df):
        line = original_fit_primary(pivots, kind, n, df)
        if line is not None:
            return line
        try:
            candidate = strategies._fit_line_any_slope(pivots, kind, n, df)
            if candidate and candidate.get("touches", 0) >= 2:
                slope = float(candidate.get("slope", 0))
                if (kind == "support" and slope > 0) or (kind == "resistance" and slope < 0):
                    candidate["kind"] = kind; candidate["method"] = "two_point_fallback"
                    candidate["quality"] = "confirmed" if candidate.get("touches", 0) >= 3 else "unconfirmed"
                    candidate["bars_since_last_touch"] = n - 1 - int(candidate.get("x1", 0))
                    return candidate
        except Exception:
            pass
        return None
    strategies._fit_primary = fit_primary_with_fallback

    def trendline_retest_state(df, line, breakout=None, break_kind=None):
        out = {"status": "INTACT", "break_index": None, "retest_index": None, "retest_level": None,
               "fakeout": False, "note": "No confirmed trendline retest."}
        if df is None or df.empty or not line:
            return out
        n = len(df); role = str(line.get("kind") or "support")
        def lv(i):
            return strategies._line_value(line["x0"], line["y0"], line["x1"], line["y1"], i)
        close = df["Close"].to_numpy(float); high = df["High"].to_numpy(float)
        low = df["Low"].to_numpy(float); open_ = df["Open"].to_numpy(float)
        atr = df["ATR"].to_numpy(float) if "ATR" in df.columns else (df["High"] - df["Low"]).to_numpy(float)
        kind = break_kind or ("support_break_down" if role == "support" else "resistance_break_up")
        is_down = kind == "support_break_down"
        start = max(int(line.get("x1", 0)) + 1, n - 35); break_i = None
        for i in range(start, n):
            lv_i = lv(i); a = max(float(atr[i]), 1e-9); body = abs(close[i] - open_[i]); rng = max(high[i] - low[i], 1e-9)
            beyond = close[i] < lv_i if is_down else close[i] > lv_i
            if beyond and abs(close[i] - lv_i) / a >= 0.10 and body / rng >= 0.35:
                break_i = i; break
        if break_i is not None:
            out["break_index"] = break_i
            for i in range(break_i + 1, n):
                lv_i = lv(i); a = max(float(atr[i]), 1e-9)
                touched = high[i] >= lv_i - a * 0.35 if is_down else low[i] <= lv_i + a * 0.35
                held = close[i] < lv_i if is_down else close[i] > lv_i
                if touched and held:
                    out.update(status="BREAK_RETEST_CONFIRMED", retest_index=i, retest_level=float(lv_i),
                               note="Break confirmed, then broken rail was retested and held.")
                    return out
                reclaimed = close[i] > lv_i if is_down else close[i] < lv_i
                if reclaimed:
                    out.update(status="FAKEOUT", retest_index=i, retest_level=float(lv_i), fakeout=True,
                               note="Break was reclaimed before a valid retest held.")
                    return out
            out.update(status="BREAK_CONFIRMED", retest_level=float(lv(n - 1)),
                       note="Trendline break detected; waiting for a clean retest.")
            return out
        scan_start = max(int(line.get("x1", 0)) + 1, n - 18)
        for i in range(scan_start, n):
            lv_i = lv(i); a = max(float(atr[i]), 1e-9); body = abs(close[i] - open_[i]); rng = max(high[i] - low[i], 1e-9)
            if role == "support":
                touched = low[i] <= lv_i + a * 0.35; rejected = close[i] > lv_i + a * 0.10 and close[i] > open_[i]; ret_dir = "BUY"
            else:
                touched = high[i] >= lv_i - a * 0.35; rejected = close[i] < lv_i - a * 0.10 and close[i] < open_[i]; ret_dir = "SELL"
            if touched and rejected and body / rng >= 0.35:
                out.update(status="RETEST_CONFIRMED", retest_index=i, retest_level=float(lv_i), direction=ret_dir,
                           note=f"{role.capitalize()} trendline retest rejected with a directional candle.")
                return out
        return out
    strategies._trendline_retest_state = trendline_retest_state

    original_state = strategies._classify_trendline_state
    def classify_state(family):
        tr = family.get("trendline_retest") or {}
        if tr.get("status") == "RETEST_CONFIRMED":
            return {"state": "CONTINUATION", "reason": "Trendline retest was confirmed by a directional rejection candle."}
        return original_state(family)
    strategies._classify_trendline_state = classify_state

    original_status_text = strategies._trendline_status_text
    def status_text(family):
        tr = family.get("trendline_retest") or {}
        if tr.get("status") == "RETEST_CONFIRMED":
            brk = family.get("breakout_grade") or {}
            return {"breakout": "NOT BROKEN", "close": "—", "retest": "CONFIRMED",
                    "displacement": "CONFIRMED" if brk.get("strength") == "confirmed" else "DEVELOPING",
                    "status": "RETEST_CONFIRMED"}
        return original_status_text(family)
    strategies._trendline_status_text = status_text

    try:
        import chart_engine
        inject = '''
full_df = family.get("df")
chart_offset = max(0, len(full_df) - chart_len) if full_df is not None else 0
for z in (family.get("order_blocks") or []):
    try:
        left = int(z.get("formed_index", 0)) - chart_offset
        right = chart_len - 1
        bottom = float(z.get("bottom")); top = float(z.get("top"))
        if right < 0 or left > right or top <= bottom: continue
        left = max(0, left)
        ob_type = str(z.get("type", "")).lower()
        edge = "#00e676" if ob_type == "bullish" else "#ff1744"
        if z.get("is_inducement"): edge = "#ffab00"
        alpha = 0.14 if z.get("freshness") == "untested" else 0.07
        if z.get("is_inducement"): alpha = 0.10
        ax.add_patch(mpatches.Rectangle((left, bottom), max(1, right-left), top-bottom,
                                         facecolor=edge, edgecolor=edge, linewidth=1.2, alpha=alpha, zorder=2.5))
        role = "INDUCEMENT OB" if z.get("is_inducement") else ("BULLISH OB" if ob_type == "bullish" else "BEARISH OB")
        fresh = str(z.get("freshness") or "")
        label = f"{role} · {fresh}" if fresh else role
        ax.text(left + 1, top, label, fontsize=6.8, fontweight="bold", color=edge, va="bottom", zorder=12,
                bbox=dict(boxstyle="round,pad=.15", facecolor="#0d1117", alpha=.72, edgecolor="none"))
    except Exception: continue
tr_runtime = family.get("trendline_retest") or {}
if tr_runtime.get("status") == "RETEST_CONFIRMED" and tr_runtime.get("retest_index") is not None:
    rx = int(tr_runtime["retest_index"]) - offset; ry = float(tr_runtime.get("retest_level"))
    if 0 <= rx < chart_len:
        ax.scatter([rx], [ry], s=82, c="#00e676", edgecolors="#ffffff", linewidths=1.5, zorder=16, marker="o")
        ax.annotate("RETEST", (rx, ry), fontsize=7.5, color="#00e676", fontweight="bold", xytext=(6, -14), textcoords="offset points", zorder=16)
'''
        def transform_renderer(fn):
            try:
                src = textwrap.dedent(inspect.getsource(fn)); tree = ast.parse(src)
                fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
                inject_nodes = ast.parse(textwrap.dedent(inject)).body
                for idx, node in enumerate(fn_node.body):
                    if isinstance(node, ast.Assign):
                        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                        if "tr" in targets:
                            fn_node.body[idx:idx] = inject_nodes; break
                else: return fn
                ast.fix_missing_locations(tree); ns = dict(fn.__globals__)
                exec(compile(tree, filename=inspect.getsourcefile(fn) or "<runtime-patch>", mode="exec"), ns)
                return ns[fn.__name__]
            except Exception as exc:
                print(f"[sitecustomize] renderer patch failed for {fn.__name__}: {exc!r}"); return fn
        chart_engine.generate_trendline_map = transform_renderer(chart_engine.generate_trendline_map)
        chart_engine.generate_trendline_educational_map = transform_renderer(chart_engine.generate_trendline_educational_map)
    except Exception as exc:
        print(f"[sitecustomize] chart patch unavailable: {exc!r}")


_install_analysis_patch()
