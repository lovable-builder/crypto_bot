"""
=============================================================
  GATE.IO MARKET SCANNER
  Uses synchronous CCXT in a thread-pool executor to avoid
  the aiodns / aiohttp DNS issue on Windows, while keeping
  the outer API fully async for the FastAPI server.
=============================================================
"""

import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Optional

import numpy as np

from config import EMAStrategyConfig
from ema_crossover import EMACrossoverStrategy
from indicators import compute_snapshot
from models import Signal
import probability as prob_engine

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], Awaitable[None]]


def _isnan(v) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return v is None


@dataclass
class Opportunity:
    symbol: str
    price: float
    change_24h: float
    volume_usdt: float
    signal: str        # "BULLISH" | "BEARISH" | "NEUTRAL"
    score: float       # 0.0 – 1.0
    ema_cross: str     # "BULLISH" | "BEARISH" | "NONE"
    rsi: Optional[float]
    adx: Optional[float]
    macd_hist: Optional[float]
    trend: str         # "UPTREND" | "DOWNTREND" | "SIDEWAYS"
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    scanned_at: str
    # Probability / grading fields
    grade: str         = "C"     # "A" | "B" | "C"
    probability: float = 0.50
    expected_value: float = 0.0
    mtf_aligned: bool  = False
    daily_trend: str   = "SIDEWAYS"
    market_type: str   = "spot"  # "spot" | "futures"
    leverage: int      = 1
    funding_rate: Optional[float] = None  # % per 8h, futures only
    # Breakout pre-filter fields
    breakout_score:     float = 0.0
    breakout_potential: str   = "LOW"   # "HIGH" | "MEDIUM" | "LOW"
    bb_squeeze:         bool  = False
    volume_buildup:     bool  = False
    near_resistance_pct: Optional[float] = None
    swing_high_near:    Optional[float] = None

    def to_dict(self) -> dict:
        def safe(v, d=2):
            return round(v, d) if (v is not None and not _isnan(v)) else None

        return {
            "symbol":      self.symbol,
            "price":       safe(self.price, 6),
            "change_24h":  safe(self.change_24h, 2),
            "volume_usdt": safe(self.volume_usdt, 0),
            "signal":      self.signal,
            "score":       safe(self.score, 4),
            "ema_cross":   self.ema_cross,
            "rsi":         safe(self.rsi, 1),
            "adx":         safe(self.adx, 1),
            "macd_hist":   safe(self.macd_hist, 8),
            "trend":       self.trend,
            "entry":          safe(self.entry, 6),
            "stop":           safe(self.stop, 6),
            "target":         safe(self.target, 6),
            "scanned_at":     self.scanned_at,
            "grade":          self.grade,
            "probability":    round(self.probability, 4),
            "expected_value": round(self.expected_value, 3),
            "mtf_aligned":    self.mtf_aligned,
            "daily_trend":    self.daily_trend,
            "market_type":    self.market_type,
            "leverage":       self.leverage,
            "funding_rate":   round(self.funding_rate * 100, 4) if self.funding_rate is not None else None,
            "breakout_score":      round(float(self.breakout_score), 3),
            "breakout_potential":  self.breakout_potential,
            "bb_squeeze":          bool(self.bb_squeeze),
            "volume_buildup":      bool(self.volume_buildup),
            "near_resistance_pct": safe(self.near_resistance_pct, 2),
            "swing_high_near":     safe(self.swing_high_near, 6),
        }


class MarketScanner:
    """
    Scans Gate.io spot markets for EMA crossover opportunities.

    Uses *synchronous* CCXT executed in a ThreadPoolExecutor.
    This sidesteps the aiodns / aiohttp DNS resolution bug on Windows
    while presenting a clean async interface to the FastAPI server.
    """

    def __init__(
        self,
        timeframe: str = "4h",
        min_volume_usdt: float = 500_000,
        workers: int = 8,
    ):
        self.timeframe = timeframe
        self.min_volume_usdt = min_volume_usdt
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._exchange = None
        self._futures_exchange = None
        self._strategy = EMACrossoverStrategy(EMAStrategyConfig())
        self._connected = False
        self._daily_cache: Dict[str, tuple] = {}   # symbol → (ts, raw_daily)

    # ── helpers ────────────────────────────────────────────────

    async def _run(self, fn, *args):
        """Run a blocking ccxt call in the thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, fn, *args)

    # ── lifecycle ──────────────────────────────────────────────

    async def connect(self):
        import ccxt  # sync ccxt — no aiohttp dependency
        self._exchange = ccxt.gateio({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        await self._run(self._exchange.load_markets)
        # Futures (perpetual swaps) — second exchange instance
        try:
            self._futures_exchange = ccxt.gateio({
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            })
            await self._run(self._futures_exchange.load_markets)
            logger.info("Scanner connected to Gate.io futures (swap)")
        except Exception as e:
            logger.warning(f"Futures exchange init failed: {e} — futures scanning disabled")
            self._futures_exchange = None
        self._connected = True
        logger.info("Scanner connected to Gate.io spot")

    def disconnect(self):
        self._pool.shutdown(wait=False)
        self._connected = False

    async def fetch_current_prices(self, symbols: List[str],
                                    market_type: str = "spot") -> Dict[str, float]:
        """Fetch last price for each symbol. Returns {symbol: price}."""
        exchange = self._futures_exchange if market_type == "futures" else self._exchange
        if exchange is None:
            return {}
        def _fetch_sync():
            prices = {}
            try:
                # Batch fetch — more reliable than individual calls, works for futures symbols
                tickers = exchange.fetch_tickers(symbols)
                for sym, t in tickers.items():
                    p = t.get("last") or t.get("close")
                    if p:
                        prices[sym] = float(p)
            except Exception:
                # Fallback: individual fetches
                for sym in symbols:
                    try:
                        t = exchange.fetch_ticker(sym)
                        p = t.get("last") or t.get("close")
                        if p:
                            prices[sym] = float(p)
                    except Exception:
                        pass
            return prices
        return await self._run(_fetch_sync)

    async def scan_futures(
        self,
        max_pairs: int = 50,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> List[Opportunity]:
        """Scan top Gate.io perpetual futures markets."""
        if not self._connected:
            await self.connect()
        if self._futures_exchange is None:
            logger.warning("Futures exchange not available")
            return []

        logger.info(f"Futures scan start  tf={self.timeframe}  max_pairs={max_pairs}")
        markets = await self.fetch_top_markets(limit=max_pairs, market_type="futures")
        total = len(markets)
        logger.info(f"  {total} eligible futures pairs")

        results: List[Opportunity] = []
        batch_size = 8

        for i in range(0, total, batch_size):
            batch = markets[i: i + batch_size]
            batch_out = await asyncio.gather(
                *[self._analyze(m) for m in batch],
                return_exceptions=True,
            )
            for r in batch_out:
                if isinstance(r, Opportunity):
                    results.append(r)
            if progress_cb:
                await progress_cb(min(i + batch_size, total), total)

        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"  Futures scan done: {len(results)}/{total} pairs")
        return results

    # ── market selection ───────────────────────────────────────

    async def fetch_top_markets(self, limit: int = 80,
                                market_type: str = "spot") -> List[Dict]:
        """
        One API call → all tickers.
        For spot: filters USDT pairs. For futures: filters USDT-margined perps.
        Returns top `limit` by volume.
        """
        exchange = self._futures_exchange if market_type == "futures" else self._exchange
        if exchange is None:
            return []
        try:
            tickers = await self._run(exchange.fetch_tickers)
        except Exception as e:
            logger.error(f"fetch_tickers ({market_type}) failed: {e}")
            return []

        # Stablecoins and pegged assets — price is near 1.0 so stop distances are
        # micro-fractions, causing position sizing to blow up to 100x+ capital.
        _STABLECOIN_BASES = {
            "USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "GUSD", "LUSD",
            "FRAX", "USDD", "CRVUSD", "PYUSD", "USDE", "USDB", "EURC", "EURT",
        }
        markets = []
        for symbol, t in tickers.items():
            if market_type == "futures":
                # Gate.io perps: BTC/USDT:USDT
                if not symbol.endswith("/USDT:USDT"):
                    continue
            else:
                if not symbol.endswith("/USDT"):
                    continue
            # Skip stablecoin/pegged pairs (base token is itself a stable)
            base = symbol.split("/")[0]
            if base in _STABLECOIN_BASES:
                continue
            vol = t.get("quoteVolume") or 0
            if vol < self.min_volume_usdt:
                continue
            funding = t.get("fundingRate")
            markets.append({
                "symbol":       symbol,
                "price":        t.get("last") or 0,
                "change_24h":   t.get("percentage") or 0,
                "volume_usdt":  vol,
                "market_type":  market_type,
                "funding_rate": funding,
            })

        markets.sort(key=lambda x: x["volume_usdt"], reverse=True)
        return markets[:limit]

    # ── per-symbol analysis ────────────────────────────────────

    def _fetch_ohlcv(self, symbol: str, limit: int = 300) -> list:
        return self._exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)

    def _analyze_sync(self, market: Dict) -> Optional[Opportunity]:
        """CPU + blocking IO work — runs inside the thread pool."""
        symbol = market["symbol"]
        market_type = market.get("market_type", "spot")
        funding_rate = market.get("funding_rate")
        exchange = self._futures_exchange if market_type == "futures" else self._exchange
        if exchange is None:
            return None
        try:
            raw = exchange.fetch_ohlcv(symbol, self.timeframe, limit=300)
            if len(raw) < 220:
                return None

            closes  = np.array([b[4] for b in raw], dtype=float)
            highs   = np.array([b[2] for b in raw], dtype=float)
            lows    = np.array([b[3] for b in raw], dtype=float)
            volumes = np.array([b[5] for b in raw], dtype=float)

            snap = compute_snapshot(closes, highs, lows, volumes)

            # ── Daily timeframe (MTF alignment) ──────────────────
            import time as _time
            now_ts = _time.time()
            cached = self._daily_cache.get(symbol)
            if cached and (now_ts - cached[0]) < 1800:   # 30-min cache
                raw_daily = cached[1]
            else:
                try:
                    raw_daily = exchange.fetch_ohlcv(symbol, "1d", limit=300)
                    self._daily_cache[symbol] = (now_ts, raw_daily)
                except Exception:
                    raw_daily = []

            if len(raw_daily) >= 50:
                try:
                    d_closes  = np.array([b[4] for b in raw_daily], dtype=float)
                    d_highs   = np.array([b[2] for b in raw_daily], dtype=float)
                    d_lows    = np.array([b[3] for b in raw_daily], dtype=float)
                    d_volumes = np.array([b[5] for b in raw_daily], dtype=float)
                    d_snap = compute_snapshot(d_closes, d_highs, d_lows, d_volumes)
                    # Use EMA200 if available, fall back to EMA21 for shorter-history coins
                    ref_ema = d_snap.ema_trend if not _isnan(d_snap.ema_trend) and d_snap.ema_trend > 0 else (
                        d_snap.ema_slow if not _isnan(d_snap.ema_slow) and d_snap.ema_slow > 0 else None
                    )
                    if ref_ema:
                        d_pct = (d_snap.close - ref_ema) / ref_ema * 100
                        daily_trend = "UPTREND" if d_pct > 0.5 else (
                            "DOWNTREND" if d_pct < -0.5 else "SIDEWAYS"
                        )
                    else:
                        daily_trend = "SIDEWAYS"
                except Exception:
                    daily_trend = "SIDEWAYS"
            else:
                daily_trend = "SIDEWAYS"

            # ── EMA cross label ───────────────────────────────────
            if snap.ema_cross_bullish:
                ema_cross = "BULLISH"
            elif snap.ema_cross_bearish:
                ema_cross = "BEARISH"
            else:
                ema_cross = "NONE"

            # ── Trend via EMA200 ──────────────────────────────────
            if not _isnan(snap.ema_trend) and snap.ema_trend > 0:
                pct = (snap.close - snap.ema_trend) / snap.ema_trend * 100
                trend = "UPTREND" if pct > 1 else ("DOWNTREND" if pct < -1 else "SIDEWAYS")
            else:
                trend = "SIDEWAYS"

            # ── Formal strategy signal ────────────────────────────
            signal_obj: Optional[Signal] = self._strategy.evaluate(symbol, snap)

            if signal_obj:
                sig_label = "BULLISH" if signal_obj.side.value == "long" else "BEARISH"
                score  = signal_obj.score
                entry  = round(signal_obj.entry_price, 6)
                stop   = round(signal_obj.stop_price, 6)
                target = round(signal_obj.target_price, 6)
            else:
                # Directional bias scoring even without a fresh crossover
                bull = bear = 0.0

                if snap.ema_cross_bullish:
                    bull += 0.30
                elif snap.ema_fast > snap.ema_slow:
                    bull += 0.15
                if snap.ema_cross_bearish:
                    bear += 0.30
                elif snap.ema_fast < snap.ema_slow:
                    bear += 0.15

                if trend == "UPTREND":
                    bull += 0.20
                elif trend == "DOWNTREND":
                    bear += 0.20

                if not _isnan(snap.adx) and snap.adx > 25:
                    bull += 0.08
                    bear += 0.08

                if not _isnan(snap.rsi):
                    if 45 <= snap.rsi <= 65:
                        bull += 0.10
                    elif 35 <= snap.rsi < 45:
                        bear += 0.10

                if not _isnan(snap.macd_hist):
                    if snap.macd_hist > 0:
                        bull += 0.10
                    else:
                        bear += 0.10

                if not _isnan(snap.volume_ratio) and snap.volume_ratio > 1.5:
                    bull += 0.07
                    bear += 0.07

                if bull >= bear and bull > 0.20:
                    sig_label, score = "BULLISH", bull
                elif bear > bull and bear > 0.20:
                    sig_label, score = "BEARISH", bear
                else:
                    sig_label, score = "NEUTRAL", max(bull, bear)

                entry = stop = target = None

            # ── MTF alignment ─────────────────────────────────────
            mtf_aligned = (
                (sig_label == "BULLISH" and daily_trend == "UPTREND") or
                (sig_label == "BEARISH" and daily_trend == "DOWNTREND")
            )

            # ── Probability + grade ───────────────────────────────
            adx_val = snap.adx if not _isnan(snap.adx) else None
            probability = prob_engine.estimate_win_probability(
                score=score,
                mtf_aligned=mtf_aligned,
                sentiment_confluence=False,   # refined post-sentiment in server.py
                adx=adx_val,
                rsi=snap.rsi if not _isnan(snap.rsi) else None,
            )
            grade = prob_engine.signal_grade(probability, score, mtf_aligned)

            # ── Grade-adjusted target (formal signal only) ────────
            if signal_obj and not _isnan(snap.atr) and snap.atr > 0:
                atr_tp_mult = {"A": 4.0, "B": 3.0, "C": 2.5}.get(grade, 3.0)
                if sig_label == "BULLISH":
                    target = round(entry + snap.atr * atr_tp_mult, 6)
                else:
                    target = round(entry - snap.atr * atr_tp_mult, 6)

            # ── Expected value ────────────────────────────────────
            grade_risk = prob_engine.GRADE_RISK.get(grade, 1.5)
            if entry and stop and target and abs(entry - stop) > 0:
                rr = abs(target - entry) / abs(entry - stop)
            else:
                rr = 2.0
            ev = prob_engine.expected_value_pct(probability, rr, grade_risk)

            import capital as _cap
            lev = _cap.get_leverage(grade, market_type)
            return Opportunity(
                symbol=symbol,
                price=market["price"],
                change_24h=market["change_24h"],
                volume_usdt=market["volume_usdt"],
                signal=sig_label,
                score=score,
                ema_cross=ema_cross,
                rsi=snap.rsi if not _isnan(snap.rsi) else None,
                adx=snap.adx if not _isnan(snap.adx) else None,
                macd_hist=snap.macd_hist if not _isnan(snap.macd_hist) else None,
                trend=trend,
                entry=entry,
                stop=stop,
                target=target,
                scanned_at=datetime.utcnow().isoformat() + "Z",
                grade=grade,
                probability=probability,
                expected_value=ev,
                mtf_aligned=mtf_aligned,
                daily_trend=daily_trend,
                market_type=market_type,
                leverage=lev,
                funding_rate=funding_rate,
                breakout_score=snap.breakout_score,
                breakout_potential=(
                    "HIGH" if snap.breakout_score > 0.60 else
                    "MEDIUM" if snap.breakout_score > 0.35 else "LOW"
                ),
                bb_squeeze=snap.bb_squeeze,
                volume_buildup=snap.volume_buildup,
                near_resistance_pct=(
                    snap.near_resistance_pct
                    if not _isnan(snap.near_resistance_pct) else None
                ),
                swing_high_near=(
                    snap.swing_high_near
                    if not _isnan(snap.swing_high_near) else None
                ),
            )

        except Exception as e:
            logger.warning(f"  {symbol}: {e}")
            return None

    async def _analyze(self, market: Dict) -> Optional[Opportunity]:
        return await self._run(self._analyze_sync, market)

    # ── full scan ──────────────────────────────────────────────

    async def scan(
        self,
        max_pairs: int = 80,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> List[Opportunity]:
        """
        Scan top markets. Returns Opportunity list sorted by score desc.
        Fires `progress_cb(done, total)` after each batch.
        """
        if not self._connected:
            await self.connect()

        logger.info(f"Scan start  tf={self.timeframe}  max_pairs={max_pairs}")
        markets = await self.fetch_top_markets(limit=max_pairs, market_type="spot")
        total = len(markets)
        logger.info(f"  {total} eligible USDT pairs")

        results: List[Opportunity] = []
        batch_size = 8  # more concurrency since we're thread-backed

        for i in range(0, total, batch_size):
            batch = markets[i : i + batch_size]
            batch_out = await asyncio.gather(
                *[self._analyze(m) for m in batch],
                return_exceptions=True,
            )
            for r in batch_out:
                if isinstance(r, Opportunity):
                    results.append(r)

            if progress_cb:
                await progress_cb(min(i + batch_size, total), total)

        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"  Scan done: {len(results)}/{total} pairs")
        return results

    # ── OHLCV for chart endpoint ───────────────────────────────

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> list:
        """Async wrapper for chart data fetching."""
        return await self._run(
            lambda: self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        )
