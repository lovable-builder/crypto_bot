"""
=============================================================
  CRYPTO TRADING BOT — CONFIGURATION
  All parameters in one place. Never hardcode values elsewhere.
=============================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict
import os


# ─── Exchange ────────────────────────────────────────────────
@dataclass
class ExchangeConfig:
    name: str = "binance"                    # binance | bybit | okx
    api_key: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))
    testnet: bool = True                     # ALWAYS start on testnet
    rate_limit_ms: int = 100                 # ms between API calls
    max_retries: int = 3


# ─── Trading Universe ────────────────────────────────────────
@dataclass
class UniverseConfig:
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
    ])
    timeframe: str = "4h"                    # 1m | 5m | 15m | 1h | 4h | 1d
    quote_currency: str = "USDT"


# ─── EMA Crossover Strategy ──────────────────────────────────
@dataclass
class EMAStrategyConfig:
    # Signal parameters
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend_filter: int = 200              # Only trade in direction of 200 EMA

    # Momentum confirmation
    rsi_period: int = 14
    rsi_overbought: float = 65.0             # Tighter than classic 70 for crypto
    rsi_oversold: float = 35.0              # Tighter than classic 30 for crypto

    # Volatility
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5        # Stop = ATR × 1.5
    atr_tp_multiplier: float = 3.0          # TP   = ATR × 3.0 (2:1 R/R)

    # Volume filter
    volume_lookback: int = 20
    volume_spike_threshold: float = 1.5     # Volume must be 1.5× average

    # Signal confluence threshold (0–1)
    min_confluence_score: float = 0.65

    # Trade direction
    allow_long: bool = True
    allow_short: bool = True                 # Set False for spot-only accounts

    # Trailing stop
    use_trailing_stop: bool = True
    trailing_stop_atr_multiplier: float = 2.0

    # Minimum bars since last signal (avoid overtrading)
    signal_cooldown_bars: int = 3

    # Maximum candles a trade may stay open without hitting SL, TP, or counter-signal
    # Default: 72 candles × 4h = 12 days
    max_hold_candles: int = 72


# ─── Risk Management ─────────────────────────────────────────
@dataclass
class RiskConfig:
    # Per-trade sizing
    risk_per_trade_pct: float = 1.5         # Risk 1.5% of account per trade
    max_position_pct: float = 20.0          # Max 20% of account in any one position

    # Portfolio-level limits
    max_open_positions: int = 5
    max_correlated_positions: int = 2       # Max 2 positions in correlated assets
    max_capital_deployed_pct: float = 60.0  # Never deploy more than 60% at once

    # Drawdown circuit breakers
    daily_loss_limit_pct: float = 3.0       # Pause 24h if daily loss > 3%
    weekly_loss_limit_pct: float = 7.0      # Pause 48h if weekly loss > 7%
    max_drawdown_pct: float = 20.0          # FULL HALT if drawdown > 20%

    # Recovery mode (post-drawdown)
    recovery_position_size_pct: float = 50.0  # Trade at 50% size in recovery
    recovery_win_streak_required: int = 5     # 5 wins needed to exit recovery
    max_recovery_days: int = 30               # Force-exit recovery after this many calendar days

    # Slippage & fees (Binance spot)
    estimated_slippage_pct: float = 0.05
    estimated_fee_pct: float = 0.10          # 0.1% maker/taker

    # Minimum trade size
    min_trade_usdt: float = 10.0


# ─── Execution ───────────────────────────────────────────────
@dataclass
class ExecutionConfig:
    order_type: str = "limit"               # limit | market
    limit_order_offset_pct: float = 0.02    # Place limit 0.02% inside spread
    order_timeout_seconds: int = 30         # Cancel & retry if not filled
    use_reduce_only: bool = True            # For shorts/futures
    enable_notifications: bool = True


# ─── Backtesting ─────────────────────────────────────────────
@dataclass
class BacktestConfig:
    start_date: str = "2021-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 10_000.0
    commission_pct: float = 0.10
    slippage_pct: float = 0.05
    data_source: str = "csv"               # csv | ccxt_live | binance_vision


# ─── Logging ─────────────────────────────────────────────────
@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs/"
    max_log_size_mb: int = 50
    trade_log_csv: str = "logs/trades.csv"
    performance_log_csv: str = "logs/performance.csv"


# ─── Master Config (single entry point) ─────────────────────
@dataclass
class BotConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    strategy: EMAStrategyConfig = field(default_factory=EMAStrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Global mode
    mode: str = "paper"                     # paper | backtest | live
    dry_run: bool = True                    # Extra safety: log orders but don't send


# ─── Singleton accessor ──────────────────────────────────────
_config: BotConfig = None

def get_config() -> BotConfig:
    global _config
    if _config is None:
        _config = BotConfig()
    return _config
