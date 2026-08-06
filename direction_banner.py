"""
direction_banner.py
====================
One shared, unmistakable LONG/SHORT/NEUTRAL banner used by every strategy
report and every live trade message, so direction is never just one word
buried mid-sentence -- it's the first thing your eye hits.

Import and use from any module that builds Telegram text:

    from direction_banner import direction_banner, direction_tag

    lines.append(direction_banner(result["direction"], extra=symbol))
    ...
    lines.append(f"Stage 8 — Entry: {direction_tag(result['direction'])}")
"""

_LONG_WORDS = {"BUY", "LONG", "BULLISH"}
_SHORT_WORDS = {"SELL", "SHORT", "BEARISH"}


def normalize_direction(direction) -> str:
    """Collapse BUY/LONG/BULLISH -> LONG, SELL/SHORT/BEARISH -> SHORT,
    everything else -> NEUTRAL. Case-insensitive, None-safe."""
    d = str(direction or "").strip().upper()
    if d in _LONG_WORDS:
        return "LONG"
    if d in _SHORT_WORDS:
        return "SHORT"
    return "NEUTRAL"


def direction_banner(direction, extra: str = "") -> str:
    """
    Big three-line block, unmissable even skimming on a phone.
    `extra` appends context on the label line (symbol, strategy name, etc).
    """
    norm = normalize_direction(direction)
    if norm == "LONG":
        emoji, label, bar = "🟢", "LONG", "🟩"
    elif norm == "SHORT":
        emoji, label, bar = "🔴", "SHORT", "🟥"
    else:
        emoji, label, bar = "⚪", "NEUTRAL", "⬜"
    row = bar * 12
    tail = f"  ·  {extra}" if extra else ""
    return f"{row}\n{emoji}  {label}{tail}  {emoji}\n{row}"


def direction_tag(direction) -> str:
    """Compact inline tag for use inside an existing line, e.g.
    f"Stage 8 — Entry: {direction_tag(direction)}" """
    norm = normalize_direction(direction)
    if norm == "LONG":
        return "🟢 LONG"
    if norm == "SHORT":
        return "🔴 SHORT"
    return "⚪ NEUTRAL"
