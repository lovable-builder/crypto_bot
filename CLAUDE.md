# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Start the web server (primary entry point)
python server.py
# Serves at http://localhost:8000 — FastAPI REST + WebSocket + static SPA

# Windows shortcut (creates venv, installs deps, kills port 8000, opens browser)
start.bat

# Run the standalone bot engine (paper/live trading)
python engine.py

# Legacy Streamlit UI
streamlit run ui/app.py
```

## Tests

```bash
# Run all tests
pytest test_bot.py -v

# Run a single test
pytest test_bot.py -v -k "test_name"

# Run with coverage
pytest test_bot.py --cov
```

## Dependencies

```bash
pip install -r requirements.txt
```

Python virtual environment is at `.venv/`. The app uses `python-dotenv` — place API keys in `.env` (e.g. `OPENAI_API_KEY`).

## Architecture

### Two independent subsystems

**1. Market Scanner (web-facing)**
`server.py` → `scanner.py` → `indicators.py`, `ema_crossover.py`, `probability.py`, `news.py`

FastAPI serves the SPA from `static/index.html`. A WebSocket at `/ws` streams real-time scan results to the browser. The scanner loops over 80+ Gate.io pairs, computes indicators, scores signals, and enriches them with news sentiment and probability grades before broadcasting.

**2. Bot Engine (trading)**
`engine.py` → `exchange.py`, `indicators.py`, `ema_crossover.py`, `risk_engine.py`, `journal.py`, `capital.py`, `evaluation.py`, `ai_analyst.py`

The engine runs a live loop: fetch OHLCV → compute indicators → evaluate strategy → validate with risk engine → place order → update journal/capital. Modes: `paper` (real data, simulated fills), `backtest` (historical replay), `live` (real orders).

### Signal pipeline

```
OHLCV (CCXT) → indicators.py → ema_crossover.py (weighted score)
  → probability.py (grade A/B/C + win prob) → news.py (sentiment confluence)
  → risk_engine.py (position size, circuit breakers) → exchange.py (order)
  → journal.py + capital.py (persistence)
```

### Key files

| File | Role |
|------|------|
| `config.py` | All tunable parameters (strategy weights, risk limits, universe) |
| `models.py` | Shared data types: `Signal`, `Trade`, `Order`, `PortfolioState` |
| `server.py` | FastAPI app — REST endpoints + WebSocket + scan orchestration |
| `scanner.py` | `MarketScanner` — pairs loop, signal scoring, WebSocket broadcast |
| `engine.py` | `BotEngine` — live/paper trading loop |
| `exchange.py` | CCXT wrapper — fetch OHLCV, place/monitor orders, paper fill sim |
| `indicators.py` | Pure numpy functions: EMA, RSI, ATR, MACD, Bollinger, ADX |
| `ema_crossover.py` | Weighted scoring strategy — produces `Signal` with entry/stop/target |
| `risk_engine.py` | Circuit breakers (daily/weekly/max DD) + Kelly-inspired position sizing |
| `probability.py` | Win probability estimation → grades A (1.5%), B (1.0%), C (0.5% risk) |
| `news.py` | Gate MCP news client at `api.gatemcp.ai` — maps sentiment to confluence labels |
| `journal.py` | Reads/writes `logs/journal.json` — trade log and performance summary |
| `capital.py` | Compounding capital state in `logs/capital.json` — dynamic risk per grade |
| `evaluation.py` | Analytics: Sharpe, Sortino, profit factor, per-grade breakdown |
| `ai_analyst.py` | GPT-4o performance analysis — 6-hour cache in `logs/ai_analysis_cache.json` |
| `ui_state.py` | Reads `logs/control.json` (UI commands) / writes `logs/state.json` (portfolio) |

### State persistence

All runtime state is in `logs/*.json` files — human-readable, no database:
- `journal.json` — every signal/trade with outcome
- `capital.json` — compounding capital balance
- `control.json` — UI commands (pause/resume/stop)
- `state.json` — latest portfolio snapshot for dashboards

### Windows-specific notes

CCXT async calls use a `ThreadPoolExecutor` workaround to avoid the `aiodns` bug on Windows. Do not switch these to `asyncio.run_in_executor` with the default event loop policy without testing first.

### Frontend

`static/index.html` is a self-contained SPA (~208 KB) using Lightweight-Charts and Chart.js with no build step. The `react-dashboard/` directory is an in-development alternative; the SPA in `static/` is the production UI.
