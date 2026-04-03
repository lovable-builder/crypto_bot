# Session Summary — Crypto Bot Development
**Date:** 2026-04-03

---

## What Was Built / Changed This Session

### 1. Trade Exit Improvements (engine.py, ema_crossover.py, config.py)

**Problem:** EMA crossover is a momentum strategy, but trades had no time-aware exits — only SL and TP. Stale trades locked capital indefinitely.

**Two exit mechanisms added:**

- **Counter-signal exit** (`ema_crossover.py` → `peek_signal()`) — When the opposite EMA crossover fires for an open trade's symbol (e.g., holding LONG, BEARISH signal fires), exit immediately at the candle's close. This is the natural end of a momentum leg.
- **Max-hold backstop** (`config.py` → `max_hold_candles: int = 72`) — If a trade has been open for more than 72 candles (12 days on 4H) with no SL/TP/counter-signal hit, close it at market with reason `"max_hold_time"`. Prevents zombie positions.

**Key code:**
- `ema_crossover.py`: new `peek_signal()` method — evaluates the current bar's signal without updating `_last_signal_bar` (no cooldown side-effect). Used by the engine for counter-signal detection on open trades.
- `engine.py`: `_process_symbol()` now calls `peek_signal()` before monitoring open trades, passes result to `_monitor_open_trades()`. `_monitor_open_trades()` has two new exit blocks (after SL/TP). `_tf_seconds()` helper extracted for shared timeframe→seconds math.
- Exit priority: **SL → TP → counter-signal → max-hold**

---

### 2. Live PnL Display Fix (server.py, static/index.html)

**Problem:** PnL cells showed `—` when opening the Journal tab (up to 60s delay). "Check Now" button relied on WebSocket broadcast, not an immediate response.

**Fixes:**
- `server.py`: `_run_price_check()` now returns `positions` and `checked_at` in its result dict (the REST response at `/api/monitor/check` now includes live positions).
- `static/index.html`: `checkPricesNow()` calls `updateLivePrices()` directly from the REST response — instant, no waiting for WS. `_enterJournalPage()` calls `checkPricesNow(true)` (silent) immediately after rendering entries.

---

### 3. Full App Review — Issues Identified and Fixed

#### Critical Fixes Applied

| Issue | File | Fix |
|---|---|---|
| Concurrent trade limit race condition | `journal.py` | Limit check moved inside `_lock` — atomically read+check+write |
| Daily budget bypass via direct calls | `journal.py` | Budget check added inside `log_signal()` — always enforced |
| Silent capital sync failure | `journal.py` | Silent `except: pass` replaced with ERROR log + message to run `/api/capital/reconcile` |
| Scan loop race (global list mutation) | `server.py` | Both spot/futures loops now iterate local `enriched` ref; `log_signal()` enforces limits internally |
| Sentiment failure invisible to user | `server.py` | Try/except around `get_batch_sentiment()` — broadcasts `warning` to UI when unavailable |
| Capital/journal divergence (no auto-sync) | `server.py` | New `_reconcile_loop()` task: rebuilds `capital.json` from journal every hour |
| Recovery mode infinite (no timeout) | `config.py`, `models.py`, `risk_engine.py` | `max_recovery_days=30`; `recovery_entered_at` on `PortfolioState`; force-exits recovery after 30 days |
| Partial fill tracked as full fill | `engine.py` | `Trade.quantity` now uses `order.filled_qty` with fallback; warns on partial fills |

#### Issues Identified But NOT Yet Fixed (lower priority, do next session)

- **ADX off-by-one seed** (`indicators.py` line 212): `arr[1:p+1]` should be `arr[0:p]` — biases ADX low by ~2-3%
- **Limit order offset too wide** (`engine.py` line 322): offset is `entry_price × 0.02%` ≈ $13 on BTC; real spread ≈ $1; orders unlikely to fill in normal market conditions
- **Exit slippage not modelled** (`engine.py`): SL/TP exits computed at exact level, not actual fill — paper P&L is optimistic
- **Silent symbol scan failure** (`engine.py` line 183): symbols with <210 candles silently skipped, no UI alert
- **Equity drawdown includes unrealized losses** (`models.py`): can trigger recovery mode prematurely during volatile bars

---

## Architecture Overview

```
Two independent subsystems:

[Market Scanner]
server.py → scanner.py → indicators.py, ema_crossover.py, probability.py, news.py
FastAPI SPA at :8000. WebSocket /ws streams real-time scan results.
Scans 80+ Gate.io spot + 50 futures pairs every 5 minutes.
Auto-logs high-grade signals to journal.json.

[Bot Engine] (separate process)
engine.py → exchange.py, indicators.py, ema_crossover.py, risk_engine.py, journal.py, capital.py
Live trading loop: fetch OHLCV → compute indicators → evaluate strategy → risk check → place order → journal.
Modes: paper (real data, simulated fills), backtest, live.
```

**Key files:**
| File | Role |
|---|---|
| `config.py` | All tunable params — strategy weights, risk limits, universe |
| `models.py` | Shared types: Signal, Trade, Order, PortfolioState |
| `server.py` | FastAPI app — REST + WebSocket + scan orchestration |
| `scanner.py` | MarketScanner — pairs loop, signal scoring, WS broadcast |
| `engine.py` | BotEngine — live/paper trading loop |
| `ema_crossover.py` | Weighted scoring strategy → Signal with entry/stop/target |
| `risk_engine.py` | Circuit breakers (daily/weekly/max DD) + Kelly-inspired sizing |
| `capital.py` | Compounding capital state in `logs/capital.json` |
| `journal.py` | Trade log in `logs/journal.json` |
| `probability.py` | Win probability → grades A (1.5% risk), B (1.0%), C (0.5%) |
| `news.py` | Gate MCP news client at `api.gatemcp.ai` — sentiment confluence |
| `indicators.py` | Pure numpy: EMA, RSI, ATR, MACD, Bollinger, ADX |
| `exchange.py` | CCXT wrapper — fetch OHLCV, place/monitor orders, paper fill sim |
| `static/index.html` | Self-contained SPA (~208 KB) — Lightweight-Charts + Chart.js |

**State persistence:** All runtime state in `logs/*.json` — no database.

---

## Signal Pipeline

```
OHLCV (CCXT/Gate.io)
  → indicators.py (EMA, RSI, ATR, MACD, ADX, Volume)
  → ema_crossover.py (weighted 6-component confluence score ≥ 0.65)
  → probability.py (grade A/B/C + win probability)
  → news.py (sentiment confluence — STRONG/MIXED/WEAK)
  → risk_engine.py (position size, circuit breakers)
  → exchange.py (order placement)
  → journal.py + capital.py (persistence)
```

**Exit priority (as of this session):**
SL (stop loss) → TP (take profit) → Counter-signal → Max-hold (72 candles)

---

## Current State

- All tests pass: `pytest test_bot.py -v` → 38/38
- Server: `python server.py` (port 8000)
- Engine: `python engine.py` (separate process, paper mode)
- Windows workaround: CCXT async uses ThreadPoolExecutor (avoids aiodns bug on Windows)
