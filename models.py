from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
from datetime import datetime
import uuid


class Side(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"

class SignalStrength(Enum):
    STRONG   = "STRONG"
    MODERATE = "MODERATE"
    WEAK     = "WEAK"
    NONE     = "NONE"

class OrderStatus(Enum):
    PENDING   = "PENDING"
    OPEN      = "OPEN"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"

class TradeStatus(Enum):
    OPEN   = "OPEN"
    CLOSED = "CLOSED"

class CircuitBreakerState(Enum):
    NORMAL   = "NORMAL"
    PAUSED   = "PAUSED"
    HALTED   = "HALTED"
    RECOVERY = "RECOVERY"


@dataclass
class OHLCV:
    timestamp: datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float
    symbol: str = ""

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class SignalComponent:
    name:    str
    value:   float
    score:   float
    weight:  float
    details: str = ""

@dataclass
class Signal:
    id:           str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol:       str = ""
    side:         Optional[Side] = None
    timestamp:    datetime = field(default_factory=datetime.utcnow)
    strategy:     str = ""
    components:   List[SignalComponent] = field(default_factory=list)
    score:        float = 0.0
    strength:     SignalStrength = SignalStrength.NONE
    entry_price:  float = 0.0
    stop_price:   float = 0.0
    target_price: float = 0.0
    atr_value:    float = 0.0

    @property
    def risk_reward(self) -> float:
        if self.stop_price == 0 or self.entry_price == 0:
            return 0.0
        risk   = abs(self.entry_price - self.stop_price)
        reward = abs(self.target_price - self.entry_price)
        return round(reward / risk, 2) if risk > 0 else 0.0

    @property
    def is_valid(self) -> bool:
        return (
            self.side is not None
            and self.score >= 0.45
            and self.entry_price > 0
            and self.stop_price > 0
            and self.target_price > 0
            and self.risk_reward >= 1.5
        )

    def __repr__(self):
        return (f"Signal({self.symbol} {self.side.value if self.side else 'NONE'} "
                f"score={self.score:.2f} rr={self.risk_reward:.1f}:1)")


@dataclass
class Order:
    id:           str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    exchange_id:  Optional[str] = None
    symbol:       str = ""
    side:         Optional[Side] = None
    price:        float = 0.0
    quantity:     float = 0.0
    order_type:   str = "limit"
    status:       OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_qty:   float = 0.0
    timestamp:    datetime = field(default_factory=datetime.utcnow)
    filled_at:    Optional[datetime] = None
    fee:          float = 0.0

    @property
    def notional(self) -> float:
        p = self.filled_price or self.price
        return p * self.quantity


@dataclass
class Trade:
    id:              str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol:          str = ""
    side:            Optional[Side] = None
    strategy:        str = ""
    signal_id:       str = ""
    entry_order:     Optional[Order] = None
    entry_price:     float = 0.0
    quantity:        float = 0.0
    entry_time:      Optional[datetime] = None
    stop_price:      float = 0.0
    target_price:    float = 0.0
    trailing_stop:   Optional[float] = None
    exit_order:      Optional[Order] = None
    exit_price:      float = 0.0
    exit_time:       Optional[datetime] = None
    exit_reason:     str = ""
    status:          TradeStatus = TradeStatus.OPEN
    gross_pnl:       float = 0.0
    fees:            float = 0.0
    net_pnl:         float = 0.0
    pnl_pct:         float = 0.0
    r_multiple:      float = 0.0

    def close(self, exit_price: float, reason: str, fee_pct: float = 0.001):
        self.exit_price  = exit_price
        self.exit_time   = datetime.utcnow()
        self.exit_reason = reason
        self.status      = TradeStatus.CLOSED
        direction = 1 if self.side == Side.LONG else -1
        self.gross_pnl = direction * (exit_price - self.entry_price) * self.quantity
        self.fees      = (self.entry_price + exit_price) * self.quantity * fee_pct
        self.net_pnl   = self.gross_pnl - self.fees
        risk = abs(self.entry_price - self.stop_price) * self.quantity
        self.r_multiple = round(self.net_pnl / risk, 2) if risk > 0 else 0.0
        self.pnl_pct    = round(self.net_pnl / (self.entry_price * self.quantity) * 100, 3)

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def max_risk_usdt(self) -> float:
        return abs(self.entry_price - self.stop_price) * self.quantity

    def __repr__(self):
        status = f"PnL={self.net_pnl:+.2f}" if self.status == TradeStatus.CLOSED else "OPEN"
        return f"Trade({self.symbol} {self.side.value if self.side else ''} {status})"


@dataclass
class PortfolioState:
    timestamp:            datetime = field(default_factory=datetime.utcnow)
    initial_capital:      float = 10_000.0
    cash:                 float = 10_000.0
    equity:               float = 10_000.0
    open_trades:          Dict[str, Trade] = field(default_factory=dict)
    closed_trades:        List[Trade] = field(default_factory=list)
    peak_equity:          float = 10_000.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct:     float = 0.0
    daily_pnl:            float = 0.0
    weekly_pnl:           float = 0.0
    total_pnl:            float = 0.0
    circuit_breaker:      CircuitBreakerState = CircuitBreakerState.NORMAL
    recovery_wins:        int = 0
    recovery_entered_at:  Optional[datetime] = None

    @property
    def daily_pnl_pct(self) -> float:
        return self.daily_pnl / self.initial_capital * 100

    @property
    def weekly_pnl_pct(self) -> float:
        return self.weekly_pnl / self.initial_capital * 100

    def update_equity(self, market_prices: Dict[str, float]):
        unrealized = 0.0
        for trade in self.open_trades.values():
            price = market_prices.get(trade.symbol, trade.entry_price)
            direction = 1 if trade.side == Side.LONG else -1
            unrealized += direction * (price - trade.entry_price) * trade.quantity
        self.equity = self.cash + unrealized
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.current_drawdown_pct = (
            (self.peak_equity - self.equity) / self.peak_equity * 100
            if self.peak_equity > 0 else 0.0
        )
        self.max_drawdown_pct = max(self.max_drawdown_pct, self.current_drawdown_pct)

    @property
    def total_return_pct(self) -> float:
        return (self.equity - self.initial_capital) / self.initial_capital * 100

    @property
    def num_open_positions(self) -> int:
        return len(self.open_trades)

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.is_winner)
        return wins / len(self.closed_trades)

    @property
    def profit_factor(self) -> float:
        gross_wins   = sum(t.gross_pnl for t in self.closed_trades if t.gross_pnl > 0)
        gross_losses = abs(sum(t.gross_pnl for t in self.closed_trades if t.gross_pnl < 0))
        return round(gross_wins / gross_losses, 2) if gross_losses > 0 else float('inf')

    def __repr__(self):
        return (f"Portfolio(equity=${self.equity:,.2f} "
                f"return={self.total_return_pct:+.1f}% "
                f"dd={self.current_drawdown_pct:.1f}% "
                f"trades={len(self.closed_trades)})")
