"""
=============================================================
  BOT ENGINE — MAIN ORCHESTRATOR
  The central event loop that ties every component together.

  Architecture:
  ┌──────────────┐     ┌────────────┐     ┌──────────────┐
  │  Data Feed   │────▶│  Strategy  │────▶│ Risk Engine  │
  │  (Exchange)  │     │  (Signals) │     │  (Sizing)    │
  └──────────────┘     └────────────┘     └──────┬───────┘
                                                  │
  ┌──────────────┐     ┌────────────┐             ▼
  │   Logger &   │◀────│ Portfolio  │◀────┌──────────────┐
  │   Notifier   │     │  Manager   │     │  Execution   │
  └──────────────┘     └────────────┘     │  Engine      │
                                          └──────────────┘
  Main loop (every candle close):
  1. Fetch latest OHLCV for all symbols
  2. Compute indicators
  3. Run strategy → generate signals
  4. Risk engine validates + sizes
  5. Execution engine places orders
  6. Monitor open trades (stops, targets)
  7. Update portfolio state
  8. Log everything
=============================================================
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from config import BotConfig, get_config
from models import PortfolioState, Trade, Signal, Side
from indicators import compute_snapshot, IndicatorSnapshot
from ema_crossover import EMACrossoverStrategy
from risk_engine import RiskEngine
from exchange import ExchangeConnector, ExchangeError

from ui_state import read_control_command, clear_control_command, write_state

logger = logging.getLogger(__name__)


class BotEngine:
    """
    Main trading bot orchestrator.

    Modes:
    - "paper":    Real market data, simulated fills, no real orders
    - "backtest": Historical data replay, vectorized simulation
    - "live":     Real data, real orders (requires testnet=False)
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.cfg       = config or get_config()
        self.portfolio = PortfolioState(
            initial_capital=self.cfg.backtest.initial_capital,
            cash=self.cfg.backtest.initial_capital,
            equity=self.cfg.backtest.initial_capital,
            peak_equity=self.cfg.backtest.initial_capital,
        )
        self.strategy   = EMACrossoverStrategy(self.cfg.strategy)
        self.risk       = RiskEngine(self.cfg.risk)
        self.exchange   = ExchangeConnector(self.cfg.exchange)
        self._bar_index: Dict[str, int] = {}
        self._running = False
        self._paused = False
        self._last_error: Optional[str] = None
        self._setup_logging()

    # ─── Lifecycle ────────────────────────────────────────────

    async def start(self):
        """Connect to exchange and begin the main loop."""
        logger.info("="*60)
        logger.info(f"  BOT STARTING — Mode: {self.cfg.mode.upper()}")
        logger.info(f"  Symbols: {self.cfg.universe.symbols}")
        logger.info(f"  Timeframe: {self.cfg.universe.timeframe}")
        logger.info(f"  Testnet: {self.cfg.exchange.testnet}")
        logger.info(f"  Dry run: {self.cfg.dry_run}")
        logger.info("="*60)

        if self.cfg.mode != "backtest":
            await self.exchange.connect()

        self._running = True

        if self.cfg.mode == "paper" or self.cfg.mode == "live":
            await self._run_live_loop()
        elif self.cfg.mode == "backtest":
            await self._run_backtest()

    async def stop(self):
        """Graceful shutdown: cancel pending orders, log final state."""
        self._running = False
        logger.info("Shutting down bot...")
        logger.info(str(self.portfolio))
        if self.exchange.is_connected:
            await self.exchange.disconnect()

    # ─── Live Loop ────────────────────────────────────────────

    async def _run_live_loop(self):
        """
        Real-time loop. Wakes up when a new candle closes.
        Uses candle alignment: waits until the top of each period.
        """
        logger.info("Live loop started")

        while self._running:
            try:
                cycle_start = datetime.utcnow()

                # UI control channel
                cmd = read_control_command(self.cfg)
                if cmd:
                    if cmd == "pause":
                        self._paused = True
                        logger.warning("Paused by UI control")
                    elif cmd == "resume":
                        self._paused = False
                        logger.warning("Resumed by UI control")
                    elif cmd == "stop":
                        logger.warning("Stop requested by UI control")
                        await self.stop()
                        clear_control_command(self.cfg)
                        return
                    clear_control_command(self.cfg)

                # Process each symbol (skip if paused)
                if not self._paused:
                    for symbol in self.cfg.universe.symbols:
                        await self._process_symbol(symbol)

                # Portfolio-level checks
                self.risk.update_circuit_breaker(self.portfolio)

                # Write state snapshot for UI
                elapsed = (datetime.utcnow() - cycle_start).total_seconds()
                write_state(
                    cfg=self.cfg,
                    portfolio=self.portfolio,
                    last_cycle_seconds=elapsed,
                    last_error=self._last_error,
                    paused=self._paused,
                )

                self._last_error = None

                logger.debug(f"Cycle complete in {elapsed:.2f}s")

                # Sleep until next candle
                await self._sleep_until_next_candle()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                self._last_error = str(e)
                # write snapshot including error
                try:
                    write_state(
                        cfg=self.cfg,
                        portfolio=self.portfolio,
                        last_cycle_seconds=None,
                        last_error=self._last_error,
                        paused=self._paused,
                    )
                except Exception:
                    pass
                await asyncio.sleep(60)  # Brief pause before retry

    async def _process_symbol(self, symbol: str):
        """Full per-symbol pipeline: data → signals → risk → execution."""

        # 1. Fetch data (need 300+ bars for EMA200)
        candles = await self.exchange.fetch_ohlcv(
            symbol, self.cfg.universe.timeframe, limit=300
        )
        if len(candles) < 210:
            logger.warning(f"{symbol}: insufficient candles ({len(candles)})")
            return

        # 2. Build numpy arrays
        closes  = np.array([c.close  for c in candles])
        highs   = np.array([c.high   for c in candles])
        lows    = np.array([c.low    for c in candles])
        volumes = np.array([c.volume for c in candles])

        # 3. Compute indicators
        snap = compute_snapshot(
            closes, highs, lows, volumes,
            cfg_fast=self.cfg.strategy.ema_fast,
            cfg_slow=self.cfg.strategy.ema_slow,
            cfg_trend=self.cfg.strategy.ema_trend_filter,
            cfg_rsi=self.cfg.strategy.rsi_period,
            cfg_atr=self.cfg.strategy.atr_period,
        )

        # 4. Monitor open trades first (stops / targets)
        await self._monitor_open_trades(symbol, candles[-1], snap)

        # 5. Generate new signal (only if no open trade in this symbol)
        open_symbols = {t.symbol for t in self.portfolio.open_trades.values()}
        if symbol in open_symbols:
            return

        bar_idx = self._bar_index.get(symbol, 0) + 1
        self._bar_index[symbol] = bar_idx

        signal = self.strategy.evaluate(symbol, snap, bar_idx)

        if signal:
            logger.info(self.strategy.describe_signal(signal))
            await self._execute_signal(signal, snap)

    # ─── Trade Monitoring ─────────────────────────────────────

    async def _monitor_open_trades(
        self,
        symbol: str,
        latest_candle,
        snap: IndicatorSnapshot,
    ):
        """Check stops and targets for all open trades in this symbol."""
        trades_to_close = []

        for trade_id, trade in list(self.portfolio.open_trades.items()):
            if trade.symbol != symbol:
                continue

            # Update trailing stop
            if self.cfg.strategy.use_trailing_stop and not np.isnan(snap.atr):
                new_trail = self.risk.compute_trailing_stop(
                    trade, latest_candle.close, snap.atr,
                    self.cfg.strategy.trailing_stop_atr_multiplier,
                )
                trade.trailing_stop = new_trail

            # Check stop loss
            stop_hit, reason, exit_price = self.risk.check_stop_hit(
                trade, latest_candle.high, latest_candle.low
            )
            if stop_hit:
                trades_to_close.append((trade, reason, exit_price))
                continue

            # Check take profit
            tp_hit, reason, exit_price = self.risk.check_target_hit(
                trade, latest_candle.high, latest_candle.low
            )
            if tp_hit:
                trades_to_close.append((trade, reason, exit_price))

        for trade, reason, exit_price in trades_to_close:
            await self._close_trade(trade, exit_price, reason)

    # ─── Signal Execution ─────────────────────────────────────

    async def _execute_signal(
        self,
        signal: Signal,
        snap: IndicatorSnapshot,
    ):
        """Risk check → size → place entry order."""

        # Risk engine evaluation
        size_result = self.risk.evaluate_signal(signal, self.portfolio)

        if not size_result.approved:
            logger.info(f"Signal rejected by risk engine: {size_result.rejection_reason}")
            return

        logger.info(
            f"Signal approved: {signal.symbol} {signal.side.value} "
            f"qty={size_result.quantity:.6f} "
            f"notional=${size_result.notional_usdt:,.2f} "
            f"risk=${size_result.risk_usdt:.2f}"
        )

        if self.cfg.dry_run:
            logger.info("[DRY RUN] Would place entry order — skipping")
            return

        # Place entry order
        try:
            if self.cfg.execution.order_type == "market":
                order = await self.exchange.place_market_order(
                    signal.symbol, signal.side, size_result.quantity
                )
            else:
                # Limit order: slightly inside spread for better fill
                offset = signal.entry_price * self.cfg.execution.limit_order_offset_pct / 100
                limit_price = (
                    signal.entry_price - offset if signal.side == Side.LONG
                    else signal.entry_price + offset
                )
                order = await self.exchange.place_limit_order(
                    signal.symbol, signal.side, size_result.quantity, limit_price
                )

            if order.status.value in ("FILLED", "OPEN"):
                fill_price = order.filled_price or signal.entry_price
                trade = Trade(
                    symbol=signal.symbol,
                    side=signal.side,
                    strategy=signal.strategy,
                    signal_id=signal.id,
                    entry_order=order,
                    entry_price=fill_price,
                    quantity=size_result.quantity,
                    entry_time=datetime.utcnow(),
                    stop_price=signal.stop_price,
                    target_price=signal.target_price,
                )
                self.portfolio.open_trades[trade.id] = trade
                self.portfolio.cash -= fill_price * size_result.quantity
                logger.info(f"Trade opened: {trade}")

        except ExchangeError as e:
            logger.error(f"Order placement failed: {e}")

    async def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        reason: str,
    ):
        """Close an open trade, record results, update portfolio."""
        if self.cfg.dry_run:
            logger.info(f"[DRY RUN] Would close {trade.symbol} @ ${exit_price:,.2f} ({reason})")
        else:
            try:
                # Place closing order
                close_side = Side.SHORT if trade.side == Side.LONG else Side.LONG
                await self.exchange.place_market_order(
                    trade.symbol, close_side, trade.quantity
                )
            except ExchangeError as e:
                logger.error(f"Failed to close trade: {e}")
                return

        trade.close(exit_price, reason, self.cfg.risk.estimated_fee_pct / 100)
        self.portfolio.cash += exit_price * trade.quantity
        del self.portfolio.open_trades[trade.id]
        self.portfolio.closed_trades.append(trade)

        self.risk.record_trade_result(trade, self.portfolio)

        pnl_color = "+" if trade.net_pnl >= 0 else ""
        logger.info(
            f"Trade closed: {trade.symbol} {reason.upper()} "
            f"PnL={pnl_color}${trade.net_pnl:.2f} ({pnl_color}{trade.pnl_pct:.2f}%) "
            f"R={trade.r_multiple:+.2f}R"
        )
        self._log_portfolio_summary()

    # ─── Backtesting ──────────────────────────────────────────

    async def _run_backtest(self):
        """
        Historical simulation. Processes candle by candle.
        No async needed here — pure CPU work.
        """
        logger.info("Backtest starting...")

        # For each symbol, simulate the full history
        for symbol in self.cfg.universe.symbols:
            logger.info(f"Backtesting {symbol}...")
            # In production: load from CSV or fetch full history
            # Here: placeholder showing the pattern
            logger.info(
                f"  To run backtest: load OHLCV data for {symbol} "
                f"from {self.cfg.backtest.start_date} to {self.cfg.backtest.end_date}, "
                f"then call _backtest_symbol(symbol, candles)"
            )

        logger.info(str(self.portfolio))

    async def _backtest_symbol(self, symbol: str, all_candles: list):
        """
        Walk-forward simulation on historical candle data.
        Processes bar by bar to prevent look-ahead bias.
        """
        warmup = max(
            self.cfg.strategy.ema_trend_filter + 10,  # Need 210+ bars for EMA200
            50
        )

        for i in range(warmup, len(all_candles)):
            window = all_candles[:i + 1]   # Only data up to current bar

            closes  = np.array([c.close  for c in window])
            highs   = np.array([c.high   for c in window])
            lows    = np.array([c.low    for c in window])
            volumes = np.array([c.volume for c in window])

            snap = compute_snapshot(closes, highs, lows, volumes)
            current_candle = window[-1]

            # Monitor → Signal → Risk → Simulate fill
            await self._monitor_open_trades(symbol, current_candle, snap)

            open_symbols = {t.symbol for t in self.portfolio.open_trades.values()}
            if symbol not in open_symbols:
                signal = self.strategy.evaluate(symbol, snap, i)
                if signal:
                    size_result = self.risk.evaluate_signal(signal, self.portfolio)
                    if size_result.approved:
                        # Simulate fill at next open (avoids look-ahead)
                        fill_price = all_candles[i + 1].open if i + 1 < len(all_candles) else signal.entry_price
                        order = await self.exchange.simulate_fill(
                            type('Order', (), {
                                'symbol': symbol, 'side': signal.side,
                                'quantity': size_result.quantity, 'price': fill_price,
                            })(),
                            fill_price,
                        )

            # Update portfolio equity mark-to-market
            self.portfolio.update_equity({symbol: current_candle.close})

        logger.info(
            f"Backtest complete for {symbol}: "
            f"{len(self.portfolio.closed_trades)} trades"
        )

    # ─── Utilities ────────────────────────────────────────────

    def _log_portfolio_summary(self):
        p = self.portfolio
        logger.info(
            f"Portfolio | Equity: ${p.equity:,.2f} | "
            f"Return: {p.total_return_pct:+.1f}% | "
            f"DD: {p.current_drawdown_pct:.1f}% | "
            f"WR: {p.win_rate*100:.0f}% | "
            f"PF: {p.profit_factor:.2f} | "
            f"Trades: {len(p.closed_trades)}"
        )

    async def _sleep_until_next_candle(self):
        """Calculate time until the next candle close and sleep."""
        tf_seconds = {
            "1m": 60, "5m": 300, "15m": 900, "1h": 3600,
            "4h": 14400, "1d": 86400,
        }.get(self.cfg.universe.timeframe, 3600)

        now     = datetime.utcnow().timestamp()
        elapsed = now % tf_seconds
        sleep_s = tf_seconds - elapsed + 5  # +5s buffer for candle to finalize
        logger.debug(f"Sleeping {sleep_s:.0f}s until next {self.cfg.universe.timeframe} candle")
        await asyncio.sleep(sleep_s)

    def _setup_logging(self):
        log_cfg = self.cfg.logging
        level   = getattr(logging, log_cfg.level, logging.INFO)
        handlers = [logging.StreamHandler()]

        if log_cfg.log_to_file:
            import os
            os.makedirs(log_cfg.log_dir, exist_ok=True)
            handlers.append(
                logging.FileHandler(
                    f"{log_cfg.log_dir}bot_{datetime.utcnow().strftime('%Y%m%d')}.log"
                )
            )

        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
            force=True,
        )


# ─── Entry Point ─────────────────────────────────────────────

async def main():
    """Launch the bot."""
    config = get_config()
    bot = BotEngine(config)

    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
