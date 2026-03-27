"""
EMA Crossover Trend Following Strategy.
Pure signal generator: snapshot → Signal or None.
"""

import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

from models import Signal, SignalComponent, SignalStrength, Side
from indicators import IndicatorSnapshot
from config import EMAStrategyConfig

SIGNAL_WEIGHTS = {
    "ema_crossover":  0.30,
    "trend_filter":   0.20,
    "adx_strength":   0.15,
    "rsi_condition":  0.15,
    "volume_confirm": 0.12,
    "macd_momentum":  0.08,
}
assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9


class EMACrossoverStrategy:
    def __init__(self, config: EMAStrategyConfig):
        self.cfg = config
        self.name = "ema_crossover"
        self._last_signal_bar: Dict[str, int] = {}

    def evaluate(self, symbol: str, snapshot: IndicatorSnapshot,
                 bar_index: int = 0) -> Optional[Signal]:
        if self._has_nan_required(snapshot):
            return None
        if self._in_cooldown(symbol, bar_index):
            return None

        long_sig  = self._eval_long(symbol, snapshot)
        short_sig = self._eval_short(symbol, snapshot)
        candidates = [s for s in [long_sig, short_sig] if s is not None]
        if not candidates:
            return None

        signal = max(candidates, key=lambda s: s.score)
        if signal.score >= self.cfg.min_confluence_score:
            self._last_signal_bar[symbol] = bar_index
            return signal
        return None

    def _eval_long(self, symbol: str, s: IndicatorSnapshot) -> Optional[Signal]:
        if not self.cfg.allow_long or not s.ema_cross_bullish:
            return None
        components = [
            self._score_ema_cross_long(s),
            self._score_trend_long(s),
            self._score_adx(s),
            self._score_rsi_long(s),
            self._score_volume(s),
            self._score_macd_long(s),
        ]
        score = self._weighted_score(components)
        entry = s.close
        stop  = entry - s.atr * self.cfg.atr_stop_multiplier
        tp    = entry + s.atr * self.cfg.atr_tp_multiplier
        return Signal(
            symbol=symbol, side=Side.LONG, strategy=self.name,
            components=components, score=round(score, 4),
            strength=self._classify(score),
            entry_price=entry, stop_price=stop, target_price=tp,
            atr_value=s.atr, timestamp=datetime.utcnow(),
        )

    def _eval_short(self, symbol: str, s: IndicatorSnapshot) -> Optional[Signal]:
        if not self.cfg.allow_short or not s.ema_cross_bearish:
            return None
        components = [
            self._score_ema_cross_short(s),
            self._score_trend_short(s),
            self._score_adx(s),
            self._score_rsi_short(s),
            self._score_volume(s),
            self._score_macd_short(s),
        ]
        score = self._weighted_score(components)
        entry = s.close
        stop  = entry + s.atr * self.cfg.atr_stop_multiplier
        tp    = entry - s.atr * self.cfg.atr_tp_multiplier
        return Signal(
            symbol=symbol, side=Side.SHORT, strategy=self.name,
            components=components, score=round(score, 4),
            strength=self._classify(score),
            entry_price=entry, stop_price=stop, target_price=tp,
            atr_value=s.atr, timestamp=datetime.utcnow(),
        )

    # ── Component scorers ──────────────────────────────────────

    def _score_ema_cross_long(self, s: IndicatorSnapshot) -> SignalComponent:
        if s.ema_cross_bullish:
            sc, det = 1.0, "Fresh bullish cross"
        elif s.ema_fast > s.ema_slow:
            gap = (s.ema_fast - s.ema_slow) / s.ema_slow * 100
            sc, det = min(0.6, gap * 10), f"EMA fast > slow by {gap:.2f}%"
        else:
            sc, det = 0.0, "No bullish cross"
        return SignalComponent("ema_crossover", s.ema_fast - s.ema_slow,
                               sc, SIGNAL_WEIGHTS["ema_crossover"], det)

    def _score_ema_cross_short(self, s: IndicatorSnapshot) -> SignalComponent:
        if s.ema_cross_bearish:
            sc, det = 1.0, "Fresh bearish cross"
        elif s.ema_fast < s.ema_slow:
            gap = (s.ema_slow - s.ema_fast) / s.ema_slow * 100
            sc, det = min(0.6, gap * 10), f"EMA fast < slow by {gap:.2f}%"
        else:
            sc, det = 0.0, "No bearish cross"
        return SignalComponent("ema_crossover", s.ema_fast - s.ema_slow,
                               sc, SIGNAL_WEIGHTS["ema_crossover"], det)

    def _score_trend_long(self, s: IndicatorSnapshot) -> SignalComponent:
        if np.isnan(s.ema_trend):
            sc, det = 0.5, "EMA200 unavailable"
        elif s.above_trend_ema:
            dist = (s.close - s.ema_trend) / s.ema_trend * 100
            sc, det = min(1.0, 0.5 + dist * 0.1), f"Price {dist:.1f}% above EMA200"
        else:
            sc, det = 0.0, "Price below EMA200"
        return SignalComponent("trend_filter", s.close - s.ema_trend,
                               sc, SIGNAL_WEIGHTS["trend_filter"], det)

    def _score_trend_short(self, s: IndicatorSnapshot) -> SignalComponent:
        if np.isnan(s.ema_trend):
            sc, det = 0.5, "EMA200 unavailable"
        elif s.below_trend_ema:
            dist = (s.ema_trend - s.close) / s.ema_trend * 100
            sc, det = min(1.0, 0.5 + dist * 0.1), f"Price {dist:.1f}% below EMA200"
        else:
            sc, det = 0.0, "Price above EMA200"
        return SignalComponent("trend_filter", s.close - s.ema_trend,
                               sc, SIGNAL_WEIGHTS["trend_filter"], det)

    def _score_adx(self, s: IndicatorSnapshot) -> SignalComponent:
        if np.isnan(s.adx):
            sc, det = 0.5, "ADX unavailable"
        elif s.adx >= 40:
            sc, det = 1.0, f"ADX={s.adx:.1f}: strong trend"
        elif s.adx >= 25:
            sc = 0.5 + (s.adx - 25) / 30
            det = f"ADX={s.adx:.1f}: trending"
        else:
            sc = max(0.0, s.adx / 50)
            det = f"ADX={s.adx:.1f}: weak/ranging"
        return SignalComponent("adx_strength", s.adx,
                               sc, SIGNAL_WEIGHTS["adx_strength"], det)

    def _score_rsi_long(self, s: IndicatorSnapshot) -> SignalComponent:
        if np.isnan(s.rsi):
            sc, det = 0.5, "RSI unavailable"
        elif s.rsi > self.cfg.rsi_overbought:
            sc, det = 0.0, f"RSI={s.rsi:.1f}: overbought"
        elif 40 <= s.rsi <= 60:
            sc, det = 1.0, f"RSI={s.rsi:.1f}: ideal zone"
        elif s.rsi < 35:
            sc, det = 0.9, f"RSI={s.rsi:.1f}: oversold bounce"
        else:
            sc, det = 0.6, f"RSI={s.rsi:.1f}: neutral"
        return SignalComponent("rsi_condition", s.rsi,
                               sc, SIGNAL_WEIGHTS["rsi_condition"], det)

    def _score_rsi_short(self, s: IndicatorSnapshot) -> SignalComponent:
        if np.isnan(s.rsi):
            sc, det = 0.5, "RSI unavailable"
        elif s.rsi < self.cfg.rsi_oversold:
            sc, det = 0.0, f"RSI={s.rsi:.1f}: oversold"
        elif 40 <= s.rsi <= 60:
            sc, det = 1.0, f"RSI={s.rsi:.1f}: ideal zone"
        elif s.rsi > 65:
            sc, det = 0.9, f"RSI={s.rsi:.1f}: overbought fade"
        else:
            sc, det = 0.6, f"RSI={s.rsi:.1f}: neutral"
        return SignalComponent("rsi_condition", s.rsi,
                               sc, SIGNAL_WEIGHTS["rsi_condition"], det)

    def _score_volume(self, s: IndicatorSnapshot) -> SignalComponent:
        if np.isnan(s.volume_ratio):
            sc, det = 0.5, "Volume unavailable"
        elif s.volume_ratio >= 2.0:
            sc, det = 1.0, f"Vol ratio={s.volume_ratio:.1f}×: spike"
        elif s.volume_ratio >= self.cfg.volume_spike_threshold:
            sc = 0.5 + (s.volume_ratio - 1.5) / 1.0 * 0.5
            det = f"Vol ratio={s.volume_ratio:.1f}×: confirmed"
        else:
            sc = max(0.0, s.volume_ratio / 3.0)
            det = f"Vol ratio={s.volume_ratio:.1f}×: low"
        return SignalComponent("volume_confirm", s.volume_ratio,
                               sc, SIGNAL_WEIGHTS["volume_confirm"], det)

    def _score_macd_long(self, s: IndicatorSnapshot) -> SignalComponent:
        if np.isnan(s.macd_hist):
            sc, det = 0.5, "MACD unavailable"
        elif s.macd_hist > 0 and (np.isnan(s.macd_hist_prev) or s.macd_hist > s.macd_hist_prev):
            sc, det = 1.0, "MACD histogram positive+rising"
        elif s.macd_hist > 0:
            sc, det = 0.7, "MACD histogram positive"
        elif not np.isnan(s.macd_hist_prev) and s.macd_hist > s.macd_hist_prev:
            sc, det = 0.4, "MACD turning up"
        else:
            sc, det = 0.0, "MACD histogram negative"
        return SignalComponent("macd_momentum", s.macd_hist,
                               sc, SIGNAL_WEIGHTS["macd_momentum"], det)

    def _score_macd_short(self, s: IndicatorSnapshot) -> SignalComponent:
        if np.isnan(s.macd_hist):
            sc, det = 0.5, "MACD unavailable"
        elif s.macd_hist < 0 and (np.isnan(s.macd_hist_prev) or s.macd_hist < s.macd_hist_prev):
            sc, det = 1.0, "MACD histogram negative+falling"
        elif s.macd_hist < 0:
            sc, det = 0.7, "MACD histogram negative"
        elif not np.isnan(s.macd_hist_prev) and s.macd_hist < s.macd_hist_prev:
            sc, det = 0.4, "MACD turning down"
        else:
            sc, det = 0.0, "MACD histogram positive"
        return SignalComponent("macd_momentum", s.macd_hist,
                               sc, SIGNAL_WEIGHTS["macd_momentum"], det)

    # ── Helpers ───────────────────────────────────────────────

    def _weighted_score(self, components: List[SignalComponent]) -> float:
        total_w = sum(c.weight for c in components)
        return sum(c.score * c.weight for c in components) / total_w if total_w else 0.0

    def _classify(self, score: float) -> SignalStrength:
        if score >= 0.80: return SignalStrength.STRONG
        if score >= 0.65: return SignalStrength.MODERATE
        if score >= 0.45: return SignalStrength.WEAK
        return SignalStrength.NONE

    def _has_nan_required(self, s: IndicatorSnapshot) -> bool:
        return any(np.isnan(v) for v in [s.close, s.ema_fast, s.ema_slow, s.atr])

    def _in_cooldown(self, symbol: str, bar_index: int) -> bool:
        return (bar_index - self._last_signal_bar.get(symbol, -999)) < self.cfg.signal_cooldown_bars

    def describe_signal(self, signal: Signal) -> str:
        lines = [
            f"\n{'='*55}",
            f"  SIGNAL: {signal.symbol} {signal.side.value}",
            f"  Score:  {signal.score:.2f}  ({signal.strength.value})",
            f"  Entry:  ${signal.entry_price:,.2f}",
            f"  Stop:   ${signal.stop_price:,.2f}",
            f"  Target: ${signal.target_price:,.2f}",
            f"  R/R:    {signal.risk_reward:.1f}:1",
            f"{'─'*55}",
        ]
        for c in signal.components:
            bar = "█" * int(c.score * 10) + "░" * (10 - int(c.score * 10))
            lines.append(f"  {c.name:<18} [{bar}] {c.score:.2f}  {c.details}")
        lines.append(f"{'='*55}")
        return "\n".join(lines)
