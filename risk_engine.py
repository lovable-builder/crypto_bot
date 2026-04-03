import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta, timezone

from models import (
    Signal, Trade, Side, PortfolioState,
    CircuitBreakerState
)
from config import RiskConfig

logger = logging.getLogger(__name__)


class PositionSizeResult:
    def __init__(self, approved: bool, quantity: float = 0.0,
                 risk_usdt: float = 0.0, notional_usdt: float = 0.0,
                 pct_of_equity: float = 0.0, rejection_reason: str = ""):
        self.approved = approved
        self.quantity = quantity
        self.risk_usdt = risk_usdt
        self.notional_usdt = notional_usdt
        self.pct_of_equity = pct_of_equity
        self.rejection_reason = rejection_reason

    def __repr__(self):
        if not self.approved:
            return f"PositionSize(REJECTED: {self.rejection_reason})"
        return (f"PositionSize(qty={self.quantity:.6f} "
                f"notional=${self.notional_usdt:,.2f} "
                f"risk=${self.risk_usdt:.2f} "
                f"={self.pct_of_equity:.1f}% of equity)")


class RiskEngine:
    def __init__(self, config: RiskConfig):
        self.cfg = config

    def evaluate_signal(self, signal: Signal, portfolio: PortfolioState) -> PositionSizeResult:
        cb = self._check_circuit_breaker(portfolio)
        if cb:
            return PositionSizeResult(False, rejection_reason=cb)
        if not signal.is_valid:
            return PositionSizeResult(False, rejection_reason=f"Signal invalid: RR={signal.risk_reward:.1f}")
        pr = self._check_portfolio_rules(signal, portfolio)
        if pr:
            return PositionSizeResult(False, rejection_reason=pr)
        return self._size_position(signal, portfolio)

    def _check_circuit_breaker(self, portfolio: PortfolioState) -> Optional[str]:
        if portfolio.circuit_breaker == CircuitBreakerState.HALTED:
            return "SYSTEM HALTED: max drawdown exceeded."
        if portfolio.circuit_breaker == CircuitBreakerState.PAUSED:
            return "SYSTEM PAUSED: daily/weekly loss limit hit."
        if portfolio.current_drawdown_pct >= self.cfg.max_drawdown_pct:
            portfolio.circuit_breaker = CircuitBreakerState.HALTED
            logger.critical(f"MAX DRAWDOWN BREACHED: {portfolio.current_drawdown_pct:.1f}%")
            return f"MAX DRAWDOWN {portfolio.current_drawdown_pct:.1f}% — HALTED"
        if portfolio.daily_pnl_pct <= -self.cfg.daily_loss_limit_pct:
            portfolio.circuit_breaker = CircuitBreakerState.PAUSED
            return f"Daily loss limit {portfolio.daily_pnl_pct:.1f}% — paused"
        if portfolio.weekly_pnl_pct <= -self.cfg.weekly_loss_limit_pct:
            portfolio.circuit_breaker = CircuitBreakerState.PAUSED
            return f"Weekly loss limit {portfolio.weekly_pnl_pct:.1f}% — paused"
        return None

    def _check_portfolio_rules(self, signal: Signal, portfolio: PortfolioState) -> Optional[str]:
        if portfolio.num_open_positions >= self.cfg.max_open_positions:
            return f"Max positions ({self.cfg.max_open_positions}) already open"
        open_symbols = {t.symbol for t in portfolio.open_trades.values()}
        if signal.symbol in open_symbols:
            return f"Already have open trade in {signal.symbol}"
        deployed = sum(t.entry_price * t.quantity for t in portfolio.open_trades.values())
        deployed_pct = deployed / portfolio.equity * 100
        if deployed_pct >= self.cfg.max_capital_deployed_pct:
            return f"Capital deployment limit: {deployed_pct:.0f}% deployed"
        risk_amt = portfolio.equity * self.cfg.risk_per_trade_pct / 100
        if risk_amt < self.cfg.min_trade_usdt:
            return f"Risk amount ${risk_amt:.2f} below minimum"
        return None

    def _size_position(self, signal: Signal, portfolio: PortfolioState) -> PositionSizeResult:
        equity = portfolio.equity
        risk_pct = self.cfg.risk_per_trade_pct
        if portfolio.circuit_breaker == CircuitBreakerState.RECOVERY:
            risk_pct *= (self.cfg.recovery_position_size_pct / 100)

        risk_amount = equity * risk_pct / 100
        stop_dist   = abs(signal.entry_price - signal.stop_price)

        if stop_dist == 0:
            return PositionSizeResult(False, rejection_reason="Stop distance is zero")

        raw_quantity = risk_amount / stop_dist
        max_notional = equity * self.cfg.max_position_pct / 100
        max_quantity = max_notional / signal.entry_price
        quantity     = min(raw_quantity, max_quantity)
        notional     = quantity * signal.entry_price

        if notional < self.cfg.min_trade_usdt:
            return PositionSizeResult(False, rejection_reason=f"Notional ${notional:.2f} below minimum")

        effective_risk = stop_dist * quantity
        fees_estimate  = notional * self.cfg.estimated_fee_pct / 100 * 2
        total_risk     = effective_risk + fees_estimate

        return PositionSizeResult(
            approved=True, quantity=round(quantity, 6),
            risk_usdt=round(total_risk, 2), notional_usdt=round(notional, 2),
            pct_of_equity=round(notional / equity * 100, 2),
        )

    def compute_trailing_stop(self, trade: Trade, current_price: float,
                               atr: float, atr_multiplier: float = 2.0) -> float:
        trail_dist = atr * atr_multiplier
        if trade.side == Side.LONG:
            new_stop = current_price - trail_dist
            current  = trade.trailing_stop or trade.stop_price
            return max(new_stop, current)
        else:
            new_stop = current_price + trail_dist
            current  = trade.trailing_stop or trade.stop_price
            return min(new_stop, current)

    def check_stop_hit(self, trade: Trade, candle_high: float,
                        candle_low: float) -> Tuple[bool, str, float]:
        stop = trade.trailing_stop or trade.stop_price
        if trade.side == Side.LONG:
            if candle_low <= stop:
                return True, "stop_loss", stop
        else:
            if candle_high >= stop:
                return True, "stop_loss", stop
        return False, "", 0.0

    def check_target_hit(self, trade: Trade, candle_high: float,
                          candle_low: float) -> Tuple[bool, str, float]:
        if trade.side == Side.LONG:
            if candle_high >= trade.target_price:
                return True, "take_profit", trade.target_price
        else:
            if candle_low <= trade.target_price:
                return True, "take_profit", trade.target_price
        return False, "", 0.0

    def update_circuit_breaker(self, portfolio: PortfolioState) -> CircuitBreakerState:
        dd = portfolio.current_drawdown_pct
        if dd >= self.cfg.max_drawdown_pct:
            portfolio.circuit_breaker = CircuitBreakerState.HALTED
        elif portfolio.circuit_breaker == CircuitBreakerState.RECOVERY:
            # Exit recovery if win streak requirement is met
            if portfolio.recovery_wins >= self.cfg.recovery_win_streak_required:
                portfolio.circuit_breaker = CircuitBreakerState.NORMAL
                portfolio.recovery_wins = 0
                portfolio.recovery_entered_at = None
                logger.info("Recovery mode exited: win streak requirement met")
            # Force-exit recovery after max_recovery_days to avoid permanent half-size trading
            elif portfolio.recovery_entered_at is not None:
                days_in_recovery = (datetime.now(timezone.utc) - portfolio.recovery_entered_at) / timedelta(days=1)
                if days_in_recovery >= self.cfg.max_recovery_days:
                    portfolio.circuit_breaker = CircuitBreakerState.NORMAL
                    portfolio.recovery_wins = 0
                    portfolio.recovery_entered_at = None
                    logger.warning(
                        "Recovery mode force-exited after %.0f days (max=%d)",
                        days_in_recovery, self.cfg.max_recovery_days,
                    )
        return portfolio.circuit_breaker

    def record_trade_result(self, trade: Trade, portfolio: PortfolioState):
        portfolio.daily_pnl  += trade.net_pnl
        portfolio.weekly_pnl += trade.net_pnl
        portfolio.total_pnl  += trade.net_pnl
        if portfolio.circuit_breaker == CircuitBreakerState.RECOVERY:
            if trade.is_winner:
                portfolio.recovery_wins += 1
            else:
                portfolio.recovery_wins = 0
        if (portfolio.circuit_breaker == CircuitBreakerState.NORMAL and
                portfolio.current_drawdown_pct >= self.cfg.max_drawdown_pct * 0.6):
            portfolio.circuit_breaker = CircuitBreakerState.RECOVERY
            portfolio.recovery_entered_at = datetime.now(timezone.utc)
            logger.warning("Recovery mode entered: drawdown=%.1f%%", portfolio.current_drawdown_pct)

    def reset_daily_pnl(self, portfolio: PortfolioState):
        portfolio.daily_pnl = 0.0
        if portfolio.circuit_breaker == CircuitBreakerState.PAUSED:
            portfolio.circuit_breaker = CircuitBreakerState.NORMAL

    def reset_weekly_pnl(self, portfolio: PortfolioState):
        portfolio.weekly_pnl = 0.0
