# Enable the SMC Telegram interface

The SMC integration is committed in `bot_smc.py`.

For the deployed Telegram bot, change the Render service **Start Command** from:

```text
python bot.py
```

to:

```text
python bot_smc.py
```

`bot_smc.py` imports the existing `bot.py`, preserves the existing Trendline/OTE/Control Panel behaviour, and layers in:

- 🧠 SMC on the main menu
- SMC asset selector
- Liquidity → Sweep → Displacement → BOS/CHOCH → OB/FVG → Premium/Discount analysis
- 📊 Market Phase & Sentiment menu
- Accumulation / Distribution / Markup / Markdown classification
- Typed-symbol SMC analysis

No broker order is placed by the SMC engine itself. The analysis returns WAIT when the sequence is not sufficiently confirmed.
