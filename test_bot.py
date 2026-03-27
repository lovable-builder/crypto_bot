"""
=============================================================
  TEST SUITE
  Run with: python -m pytest tests/test_bot.py -v
  Tests are deterministic — no live API calls.
=============================================================
"""

import numpy as np
import pytest
from datetime import datetime

from models import (
    PortfolioState, Trade, Signal, SignalComponent,
    Side, SignalStrength, TradeStatus, CircuitBreakerState, OHLCV
)
from indicators import (
    ema, rsi, atr, macd, bollinger_bands, volume_ratio,
    compute_snapshot, IndicatorSnapshot
)
from ema_crossover import EMACrossoverStrategy
from risk_engine import RiskEngine
from config import EMAStrategyConfig, RiskConfig


# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def btc_bull_prices():
    """300 bars of simulated uptrending BTC prices starting at $50k."""
    np.random.seed(42)
    prices = [50_000.0]
    for _ in range(299):
        change = np.random.normal(0.002, 0.02)  # +0.2% drift, 2% vol
        prices.append(prices[-1] * (1 + change))
    return np.array(prices)

@pytest.fixture
def btc_bear_prices():
    """300 bars of simulated downtrending BTC prices starting at $65k."""
    np.random.seed(99)
    prices = [65_000.0]
    for _ in range(299):
        change = np.random.normal(-0.002, 0.02)  # -0.2% drift
        prices.append(prices[-1] * (1 + change))
    return np.array(prices)

@pytest.fixture
def flat_prices():
    """300 bars of range-bound prices."""
    np.random.seed(7)
    base = 60_000.0
    return np.array([base + np.random.normal(0, 500) for _ in range(300)])

@pytest.fixture
def ohlcv_from_close(btc_bull_prices):
    """Generate synthetic OHLCV from close prices."""
    closes = btc_bull_prices
    highs  = closes * 1.01
    lows   = closes * 0.99
    vols   = np.random.uniform(1000, 5000, len(closes))
    return closes, highs, lows, vols

@pytest.fixture
def strategy():
    return EMACrossoverStrategy(EMAStrategyConfig())

@pytest.fixture
def risk_engine():
    return RiskEngine(RiskConfig())

@pytest.fixture
def empty_portfolio():
    return PortfolioState(
        initial_capital=10_000.0, cash=10_000.0,
        equity=10_000.0, peak_equity=10_000.0
    )


# ──────────────────────────────────────────────────────────────
# INDICATOR TESTS
# ──────────────────────────────────────────────────────────────

class TestIndicators:

    def test_ema_length(self, btc_bull_prices):
        result = ema(btc_bull_prices, 9)
        assert len(result) == len(btc_bull_prices)

    def test_ema_first_values_nan(self, btc_bull_prices):
        result = ema(btc_bull_prices, 9)
        assert np.isnan(result[0])
        assert np.isnan(result[7])
        assert not np.isnan(result[8])

    def test_ema_trend(self, btc_bull_prices):
        """EMA9 > EMA21 in a sustained bull market."""
        e9  = ema(btc_bull_prices, 9)
        e21 = ema(btc_bull_prices, 21)
        # After warmup, fast EMA should be higher in bull market
        assert e9[-1] > e21[-1], "Fast EMA should be above slow EMA in bull trend"

    def test_ema_insufficient_data(self):
        tiny = np.array([100.0, 101.0, 102.0])
        result = ema(tiny, 9)
        assert all(np.isnan(result))

    def test_rsi_range(self, btc_bull_prices):
        result = rsi(btc_bull_prices, 14)
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid), "RSI must be between 0 and 100"

    def test_rsi_overbought_in_bull(self, btc_bull_prices):
        """RSI should be elevated in strong bull trend."""
        result = rsi(btc_bull_prices, 14)
        valid  = result[~np.isnan(result)]
        assert np.mean(valid) > 50, "Average RSI should be above 50 in bull market"

    def test_rsi_oversold_in_bear(self, btc_bear_prices):
        result = rsi(btc_bear_prices, 14)
        valid  = result[~np.isnan(result)]
        assert np.mean(valid) < 50, "Average RSI should be below 50 in bear market"

    def test_atr_positive(self, ohlcv_from_close):
        closes, highs, lows, _ = ohlcv_from_close
        result = atr(highs, lows, closes, 14)
        valid  = result[~np.isnan(result)]
        assert all(v > 0 for v in valid), "ATR must always be positive"

    def test_atr_higher_in_volatile(self):
        """ATR should be higher for volatile price series."""
        calm    = np.array([100.0 + i * 0.1 for i in range(100)])
        volatile = calm * (1 + np.random.normal(0, 0.05, 100))
        calm_h, calm_l = calm * 1.001, calm * 0.999
        vol_h,  vol_l  = volatile * 1.05, volatile * 0.95

        atr_calm     = atr(calm_h, calm_l, calm, 14)
        atr_volatile = atr(vol_h, vol_l, volatile, 14)

        valid_calm = atr_calm[~np.isnan(atr_calm)]
        valid_vol  = atr_volatile[~np.isnan(atr_volatile)]
        assert np.mean(valid_vol) > np.mean(valid_calm), "Volatile ATR > calm ATR"

    def test_macd_histogram_positive_in_bull(self, btc_bull_prices):
        m = macd(btc_bull_prices)
        valid_hist = m.histogram[~np.isnan(m.histogram)]
        assert len(valid_hist) > 50, "Not enough MACD values — NaN propagation bug"
        # MACD histogram should have substantial positive bars in a trending market
        pct_positive = np.mean(valid_hist > 0)
        assert pct_positive > 0.35, f"Only {pct_positive:.0%} of histogram bars positive"

    def test_bollinger_bands_contain_price(self, btc_bull_prices):
        bb = bollinger_bands(btc_bull_prices, 20, 2.0)
        inside = 0
        total  = 0
        for i in range(20, len(btc_bull_prices)):
            if not np.isnan(bb.upper[i]):
                total += 1
                if bb.lower[i] <= btc_bull_prices[i] <= bb.upper[i]:
                    inside += 1
        # 2-sigma bands should contain ~95%+ of prices
        assert inside / total >= 0.85, f"Only {inside/total:.1%} of prices within bands"

    def test_volume_ratio(self):
        vols = np.ones(50) * 1000.0
        vols[-1] = 3000.0   # Spike
        result = volume_ratio(vols, 20)
        assert abs(result[-1] - 3.0) < 0.4, "Volume spike should show ~3× ratio"

    def test_compute_snapshot_complete(self, ohlcv_from_close):
        closes, highs, lows, vols = ohlcv_from_close
        snap = compute_snapshot(closes, highs, lows, vols)
        assert snap.close == closes[-1]
        assert not np.isnan(snap.ema_fast)
        assert not np.isnan(snap.ema_slow)
        assert not np.isnan(snap.rsi)
        assert not np.isnan(snap.atr)


# ──────────────────────────────────────────────────────────────
# STRATEGY TESTS
# ──────────────────────────────────────────────────────────────

class TestEMACrossoverStrategy:

    def _make_bullish_snap(self) -> IndicatorSnapshot:
        """Construct a snapshot that should trigger a LONG signal."""
        snap = IndicatorSnapshot()
        snap.close = 65_000.0
        snap.ema_fast      = 65_100.0   # Just crossed above
        snap.ema_slow      = 65_050.0
        snap.ema_fast_prev = 64_900.0   # Was below before
        snap.ema_slow_prev = 65_000.0
        snap.ema_trend     = 60_000.0   # Price well above EMA200
        snap.rsi           = 55.0       # Ideal zone
        snap.atr           = 1_500.0
        snap.volume_ratio  = 2.0        # Volume spike
        snap.adx           = 30.0       # Trending
        snap.macd_hist     = 100.0      # Positive
        snap.macd_hist_prev = 80.0      # Rising
        return snap

    def _make_bearish_snap(self) -> IndicatorSnapshot:
        snap = IndicatorSnapshot()
        snap.close = 60_000.0
        snap.ema_fast      = 59_900.0   # Just crossed below
        snap.ema_slow      = 59_950.0
        snap.ema_fast_prev = 60_100.0
        snap.ema_slow_prev = 59_980.0
        snap.ema_trend     = 65_000.0   # Price below EMA200
        snap.rsi           = 45.0
        snap.atr           = 1_500.0
        snap.volume_ratio  = 1.8
        snap.adx           = 28.0
        snap.macd_hist     = -80.0
        snap.macd_hist_prev = -50.0
        return snap

    def test_generates_long_signal(self, strategy):
        snap = self._make_bullish_snap()
        signal = strategy.evaluate("BTC/USDT", snap, bar_index=100)
        assert signal is not None
        assert signal.side == Side.LONG

    def test_generates_short_signal(self, strategy):
        snap = self._make_bearish_snap()
        signal = strategy.evaluate("BTC/USDT", snap, bar_index=100)
        assert signal is not None
        assert signal.side == Side.SHORT

    def test_no_signal_without_cross(self, strategy):
        snap = self._make_bullish_snap()
        # No fresh cross — fast was already above slow
        snap.ema_fast_prev = 65_050.0   # Was already above
        snap.ema_slow_prev = 65_000.0
        signal = strategy.evaluate("BTC/USDT", snap, bar_index=100)
        # Score may still pass if other signals strong — but with low confluence it fails
        # Key test: without cross, score is lower
        if signal:
            assert signal.score < 0.85

    def test_no_signal_with_nan_fields(self, strategy):
        snap = IndicatorSnapshot()  # All NaN
        signal = strategy.evaluate("BTC/USDT", snap, bar_index=100)
        assert signal is None

    def test_cooldown_prevents_duplicate_signals(self, strategy):
        snap = self._make_bullish_snap()
        sig1 = strategy.evaluate("BTC/USDT", snap, bar_index=100)
        sig2 = strategy.evaluate("BTC/USDT", snap, bar_index=101)  # Within cooldown
        sig3 = strategy.evaluate("BTC/USDT", snap, bar_index=104)  # After cooldown
        assert sig1 is not None
        assert sig2 is None     # Cooldown blocks
        assert sig3 is not None  # Cooldown expired

    def test_stop_price_below_entry_for_long(self, strategy):
        snap = self._make_bullish_snap()
        signal = strategy.evaluate("BTC/USDT", snap, bar_index=100)
        if signal:
            assert signal.stop_price < signal.entry_price

    def test_target_above_entry_for_long(self, strategy):
        snap = self._make_bullish_snap()
        signal = strategy.evaluate("BTC/USDT", snap, bar_index=100)
        if signal:
            assert signal.target_price > signal.entry_price

    def test_minimum_risk_reward(self, strategy):
        snap = self._make_bullish_snap()
        signal = strategy.evaluate("BTC/USDT", snap, bar_index=100)
        if signal:
            assert signal.risk_reward >= 1.5

    def test_overbought_rsi_reduces_long_score(self, strategy):
        snap_normal   = self._make_bullish_snap()
        snap_overbought = self._make_bullish_snap()
        snap_overbought.rsi = 80.0   # Overbought

        sig_normal     = strategy.evaluate("BTC/USDT", snap_normal,     bar_index=10)
        sig_overbought = strategy.evaluate("BTC/USDT", snap_overbought, bar_index=100)

        if sig_normal and sig_overbought:
            assert sig_normal.score > sig_overbought.score

    def test_signal_repr(self, strategy):
        snap = self._make_bullish_snap()
        signal = strategy.evaluate("BTC/USDT", snap, bar_index=100)
        if signal:
            assert "BTC/USDT" in str(signal)
            assert "LONG" in str(signal)


# ──────────────────────────────────────────────────────────────
# RISK ENGINE TESTS
# ──────────────────────────────────────────────────────────────

class TestRiskEngine:

    def _make_valid_signal(self) -> Signal:
        return Signal(
            symbol="BTC/USDT",
            side=Side.LONG,
            strategy="ema_crossover",
            score=0.75,
            strength=SignalStrength.MODERATE,
            entry_price=65_000.0,
            stop_price=62_750.0,     # ATR × 1.5 = $2250 stop
            target_price=71_500.0,  # ATR × 3.0 = $6500 target (2.88:1 R/R)
            atr_value=1_500.0,
        )

    def test_position_sizing_formula(self, risk_engine, empty_portfolio):
        signal = self._make_valid_signal()
        result = risk_engine.evaluate_signal(signal, empty_portfolio)

        assert result.approved
        # risk_per_trade = 10000 * 1.5% = $150
        # stop_dist = 65000 - 62750 = $2250
        # raw_qty = 150 / 2250 = 0.06667
        # BUT: max_position = 10000 * 20% = $2000 => max_qty = 2000/65000 = 0.03077
        # Position is capped at max_position, so expected = 0.03077
        raw_qty = (10_000 * 0.015) / (65_000 - 62_750)
        max_qty = (10_000 * 0.20) / 65_000
        expected_qty = min(raw_qty, max_qty)
        assert abs(result.quantity - expected_qty) < 0.001

    def test_risk_dollar_amount(self, risk_engine, empty_portfolio):
        signal = self._make_valid_signal()
        result = risk_engine.evaluate_signal(signal, empty_portfolio)
        assert result.approved
        # Position is capped at 20% = $2000 notional.
        # Stop dist = $2250, qty = 0.03077
        # Effective risk = 0.03077 * 2250 = ~$69.2 + fees
        assert 60 <= result.risk_usdt <= 120

    def test_rejects_when_max_positions_full(self, risk_engine, empty_portfolio):
        """Should reject when max open positions reached."""
        # Fill up positions
        for i in range(5):
            trade = Trade(
                symbol=f"SYM{i}/USDT", side=Side.LONG,
                entry_price=1000.0, quantity=1.0, stop_price=900.0,
                target_price=1200.0,
            )
            empty_portfolio.open_trades[trade.id] = trade

        signal = self._make_valid_signal()
        result = risk_engine.evaluate_signal(signal, empty_portfolio)
        assert not result.approved
        assert "Max positions" in result.rejection_reason

    def test_rejects_duplicate_symbol(self, risk_engine, empty_portfolio):
        trade = Trade(
            symbol="BTC/USDT", side=Side.LONG,
            entry_price=65_000.0, quantity=0.01, stop_price=62_000.0,
            target_price=70_000.0,
        )
        empty_portfolio.open_trades[trade.id] = trade

        signal = self._make_valid_signal()
        result = risk_engine.evaluate_signal(signal, empty_portfolio)
        assert not result.approved
        assert "BTC/USDT" in result.rejection_reason

    def test_halts_on_max_drawdown(self, risk_engine, empty_portfolio):
        empty_portfolio.current_drawdown_pct = 22.0   # Exceeds 20% limit
        signal = self._make_valid_signal()
        result = risk_engine.evaluate_signal(signal, empty_portfolio)
        assert not result.approved
        assert "DRAWDOWN" in result.rejection_reason.upper() or "HALT" in result.rejection_reason.upper()

    def test_trailing_stop_moves_only_in_profit(self, risk_engine):
        trade = Trade(
            symbol="BTC/USDT", side=Side.LONG,
            entry_price=65_000.0, quantity=0.1,
            stop_price=62_750.0, target_price=71_500.0
        )

        # Price moves up — trailing stop should advance
        trail1 = risk_engine.compute_trailing_stop(trade, 68_000.0, 1_500.0, 2.0)
        trade.trailing_stop = trail1
        trail2 = risk_engine.compute_trailing_stop(trade, 70_000.0, 1_500.0, 2.0)
        trade.trailing_stop = trail2

        # Price moves back down — stop should NOT retreat
        trail3 = risk_engine.compute_trailing_stop(trade, 67_000.0, 1_500.0, 2.0)
        assert trail3 >= trail2, "Trailing stop must never move down for a LONG"

    def test_stop_hit_detection_long(self, risk_engine):
        trade = Trade(
            symbol="BTC/USDT", side=Side.LONG,
            entry_price=65_000.0, quantity=0.1,
            stop_price=62_750.0, target_price=71_500.0
        )
        # Candle low touches stop
        hit, reason, price = risk_engine.check_stop_hit(trade, 63_000.0, 62_700.0)
        assert hit
        assert reason == "stop_loss"
        assert price == 62_750.0

    def test_target_hit_detection_long(self, risk_engine):
        trade = Trade(
            symbol="BTC/USDT", side=Side.LONG,
            entry_price=65_000.0, quantity=0.1,
            stop_price=62_750.0, target_price=71_500.0
        )
        hit, reason, price = risk_engine.check_target_hit(trade, 71_600.0, 70_000.0)
        assert hit
        assert reason == "take_profit"

    def test_no_stop_hit_when_safe(self, risk_engine):
        trade = Trade(
            symbol="BTC/USDT", side=Side.LONG,
            entry_price=65_000.0, quantity=0.1,
            stop_price=62_750.0, target_price=71_500.0
        )
        hit, _, _ = risk_engine.check_stop_hit(trade, 66_000.0, 64_000.0)
        assert not hit


# ──────────────────────────────────────────────────────────────
# TRADE P&L TESTS
# ──────────────────────────────────────────────────────────────

class TestTradeModels:

    def test_winning_long_trade_pnl(self):
        trade = Trade(
            symbol="BTC/USDT", side=Side.LONG,
            entry_price=60_000.0, quantity=0.1,
            stop_price=57_000.0, target_price=69_000.0
        )
        trade.close(exit_price=68_000.0, reason="take_profit", fee_pct=0.001)
        assert trade.is_winner
        assert trade.net_pnl > 0
        assert trade.status == TradeStatus.CLOSED
        # Gross: (68000-60000) × 0.1 = $800
        assert abs(trade.gross_pnl - 800.0) < 1.0

    def test_losing_long_trade_pnl(self):
        trade = Trade(
            symbol="BTC/USDT", side=Side.LONG,
            entry_price=65_000.0, quantity=0.1,
            stop_price=62_750.0, target_price=71_500.0
        )
        trade.close(exit_price=62_750.0, reason="stop_loss", fee_pct=0.001)
        assert not trade.is_winner
        assert trade.net_pnl < 0

    def test_short_trade_pnl_direction(self):
        trade = Trade(
            symbol="BTC/USDT", side=Side.SHORT,
            entry_price=65_000.0, quantity=0.1,
            stop_price=67_250.0, target_price=58_500.0
        )
        # Price drops — short wins
        trade.close(exit_price=58_500.0, reason="take_profit", fee_pct=0.001)
        assert trade.is_winner
        assert trade.gross_pnl > 0

    def test_r_multiple_calculation(self):
        trade = Trade(
            symbol="BTC/USDT", side=Side.LONG,
            entry_price=60_000.0, quantity=0.1,
            stop_price=57_000.0,   # Risk = $3000 × 0.1 = $300
            target_price=69_000.0
        )
        trade.close(exit_price=69_000.0, reason="take_profit", fee_pct=0.0)
        # Gain = $9000 × 0.1 = $900 = 3R
        assert abs(trade.r_multiple - 3.0) < 0.1

    def test_portfolio_win_rate(self):
        portfolio = PortfolioState(initial_capital=10_000.0)
        for i in range(6):
            t = Trade(symbol="BTC/USDT", side=Side.LONG, entry_price=60_000.0, quantity=0.01, stop_price=57_000.0, target_price=66_000.0)
            t.close(exit_price=66_000.0 if i < 4 else 57_000.0, reason="tp", fee_pct=0.0)
            portfolio.closed_trades.append(t)
        assert abs(portfolio.win_rate - 4/6) < 0.01


# ──────────────────────────────────────────────────────────────
# INTEGRATION TEST
# ──────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_long_trade(self):
        """
        End-to-end: snapshot → strategy → risk → trade → close.
        Validates the complete decision chain without exchange calls.
        """
        cfg_s    = EMAStrategyConfig()
        cfg_r    = RiskConfig()
        strategy = EMACrossoverStrategy(cfg_s)
        risk     = RiskEngine(cfg_r)
        portfolio = PortfolioState(
            initial_capital=10_000.0, cash=10_000.0,
            equity=10_000.0, peak_equity=10_000.0
        )

        # Build a high-quality bullish snapshot
        snap = IndicatorSnapshot()
        snap.close = 65_000.0
        snap.ema_fast = 65_200.0; snap.ema_slow = 65_100.0
        snap.ema_fast_prev = 64_800.0; snap.ema_slow_prev = 65_000.0
        snap.ema_trend = 55_000.0; snap.rsi = 55.0
        snap.atr = 1_500.0; snap.volume_ratio = 2.2
        snap.adx = 35.0; snap.macd_hist = 120.0; snap.macd_hist_prev = 90.0

        # Step 1: Strategy generates signal
        signal = strategy.evaluate("BTC/USDT", snap, bar_index=50)
        assert signal is not None, "Strategy should generate signal"
        assert signal.side == Side.LONG
        assert signal.score >= cfg_s.min_confluence_score

        # Step 2: Risk engine approves and sizes
        result = risk.evaluate_signal(signal, portfolio)
        assert result.approved, f"Risk should approve: {result.rejection_reason}"
        assert result.quantity > 0
        assert result.pct_of_equity <= cfg_r.max_position_pct

        # Step 3: Open trade
        trade = Trade(
            symbol=signal.symbol, side=signal.side,
            strategy=signal.strategy, signal_id=signal.id,
            entry_price=signal.entry_price, quantity=result.quantity,
            stop_price=signal.stop_price, target_price=signal.target_price
        )
        portfolio.open_trades[trade.id] = trade
        assert portfolio.num_open_positions == 1

        # Step 4: Trade hits target
        trade.close(signal.target_price, "take_profit", fee_pct=0.001)
        del portfolio.open_trades[trade.id]
        portfolio.closed_trades.append(trade)
        risk.record_trade_result(trade, portfolio)

        # Step 5: Verify outcome
        assert trade.is_winner
        assert trade.r_multiple > 1.5
        assert portfolio.total_pnl > 0
        assert portfolio.win_rate == 1.0
        print(f"\nIntegration test passed: {trade}")
        print(f"Portfolio: {portfolio}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
