"""Tiny, Render-friendly online performance memory.

No candle history or ML model is held in RAM. The module stores only compact
aggregate statistics. Persistence is optional and uses LEARNING_STATE_PATH.
For durable Render Free persistence, point that path at an externally mounted
persistent location/provider; otherwise the bot still works statelessly.
"""
from __future__ import annotations
import json, os, tempfile
from typing import Any, Dict

DEFAULT_PATH = os.getenv("LEARNING_STATE_PATH", "learning_state.json")
MAX_KEYS = 300


def _load(path: str = DEFAULT_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: Dict[str, Any], path: str = DEFAULT_PATH) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".learning_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _key(strategy: str, regime: str, direction: str) -> str:
    return f"{strategy}|{regime}|{direction}"


def record_outcome(strategy: str, regime: str, direction: str, r_multiple: float, path: str = DEFAULT_PATH) -> Dict[str, Any]:
    data = _load(path)
    stats = data.setdefault("stats", {})
    key = _key(strategy, regime, direction)
    s = stats.setdefault(key, {"n":0,"wins":0,"losses":0,"sum_r":0.0,"sum_r2":0.0,"avg_r":0.0})
    r = float(r_multiple)
    s["n"] += 1
    s["wins"] += int(r > 0)
    s["losses"] += int(r <= 0)
    s["sum_r"] += r
    s["sum_r2"] += r*r
    s["avg_r"] = round(s["sum_r"] / s["n"], 5)
    if len(stats) > MAX_KEYS:
        # Keep the most useful recent-ish keys without retaining raw history.
        for k in sorted(stats)[:-MAX_KEYS]: del stats[k]
    data["version"] = 1
    _save(data, path)
    return s


def get_stats(strategy: str, regime: str, direction: str, path: str = DEFAULT_PATH) -> Dict[str, Any]:
    return _load(path).get("stats", {}).get(_key(strategy, regime, direction), {"n":0,"wins":0,"losses":0,"sum_r":0.0,"avg_r":0.0})


def calibration(strategy: str, regime: str, direction: str, path: str = DEFAULT_PATH) -> Dict[str, Any]:
    s = get_stats(strategy, regime, direction, path)
    n = int(s.get("n",0)); avg = float(s.get("avg_r",0.0))
    if n < 30:
        return {"usable":False,"sample":n,"expectancy_r":avg,"adjustment":0}
    win = float(s.get("wins",0))/max(n,1)
    # Small bounded adjustment; learning never creates a trade by itself.
    adjustment = max(-8, min(8, round((avg * 12) + ((win-.5)*10))))
    return {"usable":True,"sample":n,"win_rate":round(win,4),"expectancy_r":avg,"adjustment":adjustment}
