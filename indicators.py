"""
=============================================================
  INDICATOR ENGINE
  Pure functions only. No side effects. No state.
  All inputs/outputs are numpy arrays or scalars.
  Vectorized for speed. Testable in isolation.
=============================================================
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


# ─── Moving Averages ─────────────────────────────────────────

def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average. NaN-safe: seeds from first full non-NaN window.
    Uses Wilder's smoothing: k = 2 / (period + 1)
    Returns array of same length; values before first valid window are NaN.
    """
    if len(prices) < period:
        return np.full(len(prices), np.nan)

    k = 2.0 / (period + 1)
    result = np.full(len(prices), np.nan)

    # Find first index with 'period' consecutive non-NaN values
    for start in range(len(prices) - period + 1):
        window = prices[start:start + period]
        if not np.any(np.isnan(window)):
            seed_idx = start + period - 1
            result[seed_idx] = np.mean(window)
            for i in range(seed_idx + 1, len(prices)):
                if not np.isnan(prices[i]) and not np.isnan(result[i - 1]):
                    result[i] = prices[i] * k + result[i - 1] * (1 - k)
                elif not np.isnan(result[i - 1]):
                    result[i] = result[i - 1]
            break

    return result


def sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average using convolution."""
    if len(prices) < period:
        return np.full(len(prices), np.nan)
    result = np.full(len(prices), np.nan)
    kernel = np.ones(period) / period
    result[period - 1:] = np.convolve(prices, kernel, mode='valid')
    return result


# ─── RSI ─────────────────────────────────────────────────────

def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (Wilder's RSI).
    Returns 0–100. Standard overbought: 70, oversold: 30.
    For crypto we use tighter bounds: 65 / 35.
    """
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan)

    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(len(prices), np.nan)

    # Initial averages (SMA seed)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


# ─── ATR ─────────────────────────────────────────────────────

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range = max(H-L, |H-prev_C|, |L-prev_C|)"""
    hl  = high[1:] - low[1:]
    hpc = np.abs(high[1:] - close[:-1])
    lpc = np.abs(low[1:]  - close[:-1])
    tr  = np.maximum(hl, np.maximum(hpc, lpc))
    return np.concatenate([[np.nan], tr])


def atr(high: np.ndarray, low: np.ndarray,
        close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range — the gold standard for volatility measurement.
    Used for: stop placement, position sizing, breakout detection.
    """
    tr = true_range(high, low, close)
    result = np.full(len(close), np.nan)

    valid = tr[~np.isnan(tr)]
    if len(valid) < period:
        return result

    # Wilder's smoothing
    first_valid_idx = np.where(~np.isnan(tr))[0][0]
    result[first_valid_idx + period - 1] = np.mean(valid[:period])

    for i in range(first_valid_idx + period, len(tr)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


# ─── MACD ────────────────────────────────────────────────────

@dataclass
class MACDResult:
    macd:      np.ndarray
    signal:    np.ndarray
    histogram: np.ndarray

def macd(prices: np.ndarray,
         fast: int = 12, slow: int = 26, signal_period: int = 9) -> MACDResult:
    """
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal

    Histogram crossing zero = momentum shift.
    Rising histogram = increasing momentum.
    """
    ema_fast   = ema(prices, fast)
    ema_slow   = ema(prices, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    hist        = macd_line - signal_line
    return MACDResult(macd=macd_line, signal=signal_line, histogram=hist)


# ─── Bollinger Bands ─────────────────────────────────────────

@dataclass
class BBResult:
    upper:  np.ndarray
    middle: np.ndarray
    lower:  np.ndarray
    width:  np.ndarray   # (upper - lower) / middle — normalized band width

def bollinger_bands(prices: np.ndarray,
                    period: int = 20, std_dev: float = 2.0) -> BBResult:
    """
    Bollinger Bands.
    Width compression = volatility contraction = breakout approaching.
    Price at bands + RSI extreme = mean reversion signal.
    """
    mid   = sma(prices, period)
    std   = np.array([
        np.std(prices[max(0, i - period + 1):i + 1]) if i >= period - 1 else np.nan
        for i in range(len(prices))
    ])
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = np.where(mid > 0, (upper - lower) / mid, np.nan)
    return BBResult(upper=upper, middle=mid, lower=lower, width=width)


# ─── ADX ─────────────────────────────────────────────────────

@dataclass
class ADXResult:
    adx:   np.ndarray   # Trend strength: >25 = trending, >40 = strong
    di_plus:  np.ndarray
    di_minus: np.ndarray

def adx(high: np.ndarray, low: np.ndarray,
        close: np.ndarray, period: int = 14) -> ADXResult:
    """
    Average Directional Index.
    ADX > 25: trend is strong enough to trade.
    ADX < 20: market is ranging — avoid trend strategies.
    +DI > -DI: bullish. -DI > +DI: bearish.
    """
    n = len(close)
    result_adx = np.full(n, np.nan)
    di_p = np.full(n, np.nan)
    di_m = np.full(n, np.nan)

    tr_arr = true_range(high, low, close)
    dm_plus  = np.zeros(n)
    dm_minus = np.zeros(n)

    for i in range(1, n):
        up   = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        dm_plus[i]  = up   if (up > down and up > 0)   else 0.0
        dm_minus[i] = down if (down > up and down > 0) else 0.0

    # Smooth with Wilder's
    def _wilder(arr, p):
        out = np.full(n, np.nan)
        if n < p:
            return out
        out[p] = np.nansum(arr[1:p + 1])
        for i in range(p + 1, n):
            out[i] = out[i - 1] - out[i - 1] / p + arr[i]
        return out

    atr14   = _wilder(tr_arr, period)
    dmp14   = _wilder(dm_plus, period)
    dmm14   = _wilder(dm_minus, period)

    for i in range(period, n):
        if atr14[i] > 0:
            di_p[i] = 100 * dmp14[i] / atr14[i]
            di_m[i] = 100 * dmm14[i] / atr14[i]

    dx = np.full(n, np.nan)
    for i in range(period, n):
        denom = di_p[i] + di_m[i]
        if denom > 0:
            dx[i] = 100 * abs(di_p[i] - di_m[i]) / denom

    # ADX = smoothed DX
    result_adx[2 * period] = np.nanmean(dx[period:2 * period + 1])
    for i in range(2 * period + 1, n):
        if not np.isnan(dx[i]) and not np.isnan(result_adx[i - 1]):
            result_adx[i] = (result_adx[i - 1] * (period - 1) + dx[i]) / period

    return ADXResult(adx=result_adx, di_plus=di_p, di_minus=di_m)


# ─── Volume ──────────────────────────────────────────────────

def volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Current volume / average volume over lookback.
    > 1.5 = above-average participation (confirmation signal).
    > 3.0 = volume spike (potential institutional move or news).
    """
    avg = sma(volume, period)
    return np.where(avg > 0, volume / avg, np.nan)


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On-Balance Volume.
    Rising OBV + flat price = accumulation (bullish divergence).
    Falling OBV + flat price = distribution (bearish divergence).
    """
    result = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            result[i] = result[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            result[i] = result[i - 1] - volume[i]
        else:
            result[i] = result[i - 1]
    return result


# ─── Composite Indicator Snapshot ────────────────────────────

@dataclass
class IndicatorSnapshot:
    """All indicator values for the latest bar. Used by strategy."""
    # Price
    close:       float = np.nan
    high:        float = np.nan
    low:         float = np.nan
    volume:      float = np.nan

    # EMAs
    ema_fast:    float = np.nan   # 9
    ema_slow:    float = np.nan   # 21
    ema_trend:   float = np.nan   # 200

    # Prev EMAs (for crossover detection)
    ema_fast_prev:  float = np.nan
    ema_slow_prev:  float = np.nan

    # Momentum
    rsi:         float = np.nan
    macd_hist:   float = np.nan
    macd_hist_prev: float = np.nan

    # Volatility
    atr:         float = np.nan
    bb_upper:    float = np.nan
    bb_lower:    float = np.nan
    bb_width:    float = np.nan
    adx:         float = np.nan
    di_plus:     float = np.nan
    di_minus:    float = np.nan

    # Volume
    volume_ratio: float = np.nan
    obv:          float = np.nan

    @property
    def ema_cross_bullish(self) -> bool:
        """Fast EMA just crossed above slow EMA."""
        return (
            self.ema_fast > self.ema_slow and
            self.ema_fast_prev <= self.ema_slow_prev
        )

    @property
    def ema_cross_bearish(self) -> bool:
        """Fast EMA just crossed below slow EMA."""
        return (
            self.ema_fast < self.ema_slow and
            self.ema_fast_prev >= self.ema_slow_prev
        )

    @property
    def above_trend_ema(self) -> bool:
        return self.close > self.ema_trend

    @property
    def below_trend_ema(self) -> bool:
        return self.close < self.ema_trend

    @property
    def trend_is_strong(self) -> bool:
        return not np.isnan(self.adx) and self.adx > 25


def compute_snapshot(
    closes:  np.ndarray,
    highs:   np.ndarray,
    lows:    np.ndarray,
    volumes: np.ndarray,
    cfg_fast:   int = 9,
    cfg_slow:   int = 21,
    cfg_trend:  int = 200,
    cfg_rsi:    int = 14,
    cfg_atr:    int = 14,
    cfg_vol_lb: int = 20,
) -> IndicatorSnapshot:
    """
    Compute all indicators for the latest bar.
    Returns a single IndicatorSnapshot with the most recent values.
    """
    snap = IndicatorSnapshot()
    n = len(closes)
    if n < 2:
        return snap

    snap.close  = closes[-1]
    snap.high   = highs[-1]
    snap.low    = lows[-1]
    snap.volume = volumes[-1]

    # EMAs
    ema_f = ema(closes, cfg_fast)
    ema_s = ema(closes, cfg_slow)
    ema_t = ema(closes, cfg_trend)

    snap.ema_fast  = ema_f[-1]
    snap.ema_slow  = ema_s[-1]
    snap.ema_trend = ema_t[-1]
    snap.ema_fast_prev = ema_f[-2]
    snap.ema_slow_prev = ema_s[-2]

    # RSI
    rsi_arr = rsi(closes, cfg_rsi)
    snap.rsi = rsi_arr[-1]

    # MACD
    m = macd(closes)
    snap.macd_hist      = m.histogram[-1]
    snap.macd_hist_prev = m.histogram[-2] if n > 2 else np.nan

    # ATR
    atr_arr  = atr(highs, lows, closes, cfg_atr)
    snap.atr = atr_arr[-1]

    # Bollinger Bands
    bb = bollinger_bands(closes)
    snap.bb_upper = bb.upper[-1]
    snap.bb_lower = bb.lower[-1]
    snap.bb_width = bb.width[-1]

    # ADX
    adx_result  = adx(highs, lows, closes)
    snap.adx    = adx_result.adx[-1]
    snap.di_plus  = adx_result.di_plus[-1]
    snap.di_minus = adx_result.di_minus[-1]

    # Volume
    snap.volume_ratio = volume_ratio(volumes, cfg_vol_lb)[-1]
    snap.obv          = obv(closes, volumes)[-1]

    return snap
