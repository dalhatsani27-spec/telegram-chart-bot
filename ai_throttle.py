"""
ai_throttle.py
================
Wraps the existing OpenRouter commentary/translation calls with:
  - A simple per-hour call budget (falls back to the plain template text once
    exhausted, never errors out).
  - A short-lived cache so repeated requests for the same text+language don't
    re-hit the API.
The caller decides WHEN to call this at all -- the real fix for "don't burn
the free tier" is calling this only on confirmed trade events, not on every
routine scan. This module is the safety net underneath that discipline.
"""

import time
import hashlib

MAX_CALLS_PER_HOUR = 20
CACHE_TTL_SECONDS = 600

_call_timestamps = []
_cache = {}  # key -> (timestamp, result)


def _budget_available():
    now = time.time()
    global _call_timestamps
    _call_timestamps = [t for t in _call_timestamps if now - t < 3600]
    return len(_call_timestamps) < MAX_CALLS_PER_HOUR


def _record_call():
    _call_timestamps.append(time.time())


def _cache_key(text, language):
    return hashlib.sha256(f"{language}:{text}".encode()).hexdigest()


def throttled_call(text, language, call_fn, fallback_text):
    """
    call_fn(text, language) -> str   (the actual OpenRouter call, e.g. translate_text or fetch_ai_commentary)
    fallback_text: what to return if we're out of budget or the call fails.
    """
    key = _cache_key(text, language)
    now = time.time()
    if key in _cache:
        ts, result = _cache[key]
        if now - ts < CACHE_TTL_SECONDS:
            return result

    if not _budget_available():
        return fallback_text

    try:
        result = call_fn(text, language)
        _record_call()
        _cache[key] = (now, result)
        return result
    except Exception:
        return fallback_text
