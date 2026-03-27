# Crypto Trading Bot
# EMA Crossover Trend Strategy — Production Architecture

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys (NEVER commit these)
cp .env.example .env
# Edit .env with your Binance testnet keys

# 3. Run tests
python -m pytest tests/test_bot.py -v

# 4. Paper trade (real data, simulated fills)
python -m core.engine

# 5. Backtest
# Set config.mode = "backtest" in config/config.py
```

## UI Dashboard (Streamlit)

The bot can publish a lightweight JSON snapshot to `logs/state.json` and a Streamlit UI can read it.

### 1) Start the bot (writes logs/state.json)

```bash
python engine.py
```

### 2) Start the UI

```bash
streamlit run ui/app.py
```

### UI Controls

The UI can send commands by writing `logs/control.json`:
- **Pause**: stops processing new candles (does not close positions)
- **Resume**: continues processing
- **Stop**: triggers graceful shutdown


## Architecture

```
crypto_bot/
├── config/
│   └── config.py          # All parameters — single source of truth
├── core/
│   ├── models.py           # Data types: Signal, Trade, Portfolio
│   └── engine.py           # Main orchestrator loop
├── indicators/
│   └── indicators.py       # EMA, RSI, ATR, MACD, BB, ADX (pure functions)
├── strategies/
│   └── ema_crossover.py    # Signal scoring & generation
├── risk/
│   └── risk_engine.py      # Position sizing, circuit breakers
├── execution/
│   └── exchange.py         # ccxt exchange connector
└── tests/
    └── test_bot.py         # Full test suite
```

## Safety Checklist Before Going Live

- [ ] All tests passing: `pytest tests/ -v`
- [ ] Paper traded for 2+ weeks with positive results
- [ ] API keys are READ-ONLY unless orders are enabled
- [ ] `testnet: True` in ExchangeConfig
- [ ] `dry_run: True` in BotConfig
- [ ] Max drawdown set to personal risk tolerance
- [ ] Circuit breakers tested manually
- [ ] Start with minimum position sizes
