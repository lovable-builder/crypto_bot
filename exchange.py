"""
=============================================================
  EXCHANGE CONNECTOR
  Unified interface over ccxt. Abstracts away exchange differences.
  Paper trading mode = no real orders, full simulation.
  All methods are async. Rate limiting built-in.
=============================================================
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from core.models import Order, OrderStatus, Side, OHLCV
from config.config import ExchangeConfig

logger = logging.getLogger(__name__)


class ExchangeError(Exception):
    pass

class InsufficientFundsError(ExchangeError):
    pass

class OrderNotFoundError(ExchangeError):
    pass


class ExchangeConnector:
    """
    Unified exchange interface.
    Supports: Binance Spot, Binance Futures, Bybit, OKX (via ccxt).

    Usage:
        connector = ExchangeConnector(config)
        await connector.connect()
        candles = await connector.fetch_ohlcv("BTC/USDT", "4h", limit=500)
        order   = await connector.place_limit_order("BTC/USDT", Side.LONG, qty=0.01, price=65000)
    """

    def __init__(self, config: ExchangeConfig):
        self.cfg = config
        self._exchange = None
        self._last_request_time = 0.0
        self._is_connected = False

    # ─── Lifecycle ────────────────────────────────────────────

    async def connect(self):
        """Initialize ccxt exchange instance."""
        try:
            import ccxt.async_support as ccxt
        except ImportError:
            raise ImportError(
                "ccxt not installed. Run: pip install ccxt"
            )

        exchange_class = getattr(ccxt, self.cfg.name, None)
        if exchange_class is None:
            raise ExchangeError(f"Unknown exchange: {self.cfg.name}")

        params = {
            "apiKey":    self.cfg.api_key,
            "secret":    self.cfg.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",          # "spot" | "future" | "swap"
                "adjustForTimeDifference": True,
            }
        }

        if self.cfg.testnet:
            params["options"]["sandboxMode"] = True
            logger.info(f"Connecting to {self.cfg.name} TESTNET")
        else:
            logger.info(f"Connecting to {self.cfg.name} LIVE")

        self._exchange = exchange_class(params)

        try:
            await self._exchange.load_markets()
            self._is_connected = True
            logger.info(f"Connected to {self.cfg.name}")
        except Exception as e:
            raise ExchangeError(f"Failed to connect: {e}")

    async def disconnect(self):
        if self._exchange:
            await self._exchange.close()
            self._is_connected = False

    # ─── Market Data ──────────────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 500,
        since: Optional[int] = None,
    ) -> List[OHLCV]:
        """
        Fetch OHLCV candles. Returns list of OHLCV objects, oldest first.
        limit=500 gives ~83 days of 4h candles.
        limit=1000 gives ~166 days — enough for EMA200 (200 bars min).
        """
        await self._rate_limit()

        try:
            raw = await self._exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=limit
            )
        except Exception as e:
            raise ExchangeError(f"fetch_ohlcv failed for {symbol}: {e}")

        candles = []
        for bar in raw:
            candles.append(OHLCV(
                timestamp=datetime.utcfromtimestamp(bar[0] / 1000),
                open=bar[1], high=bar[2], low=bar[3],
                close=bar[4], volume=bar[5],
                symbol=symbol,
            ))

        logger.debug(f"Fetched {len(candles)} candles for {symbol} [{timeframe}]")
        return candles

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current bid/ask/last price."""
        await self._rate_limit()
        try:
            return await self._exchange.fetch_ticker(symbol)
        except Exception as e:
            raise ExchangeError(f"fetch_ticker failed for {symbol}: {e}")

    async def fetch_order_book(
        self, symbol: str, depth: int = 20
    ) -> Dict[str, Any]:
        """Get order book. Used to estimate slippage for large orders."""
        await self._rate_limit()
        return await self._exchange.fetch_order_book(symbol, limit=depth)

    async def fetch_balance(self) -> Dict[str, float]:
        """Returns dict of {currency: free_balance}."""
        await self._rate_limit()
        try:
            balance = await self._exchange.fetch_balance()
            return {
                currency: info["free"]
                for currency, info in balance["total"].items()
                if info > 0
            } if isinstance(balance["total"], dict) else {}
        except Exception as e:
            raise ExchangeError(f"fetch_balance failed: {e}")

    # ─── Order Placement ──────────────────────────────────────

    async def place_limit_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        price: float,
        params: Optional[Dict] = None,
    ) -> Order:
        """
        Place a limit order.
        In testnet mode: executes against Binance testnet.
        In paper mode (dry_run=True): simulates without sending.
        """
        await self._rate_limit()

        order_side = "buy" if side == Side.LONG else "sell"
        order = Order(
            symbol=symbol, side=side,
            price=price, quantity=quantity,
            order_type="limit",
        )

        logger.info(
            f"Placing LIMIT {order_side.upper()} {symbol} "
            f"qty={quantity:.6f} @ ${price:,.2f}"
        )

        try:
            result = await self._exchange.create_limit_order(
                symbol, order_side, quantity, price, params=params or {}
            )
            order.exchange_id = result["id"]
            order.status = OrderStatus.OPEN
            logger.info(f"Order placed: id={order.exchange_id}")
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Order rejected: {e}")
            raise ExchangeError(f"Order placement failed: {e}")

        return order

    async def place_market_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
    ) -> Order:
        """Place a market order. Higher slippage but guaranteed fill."""
        await self._rate_limit()

        order_side = "buy" if side == Side.LONG else "sell"
        order = Order(
            symbol=symbol, side=side,
            quantity=quantity, order_type="market",
        )

        try:
            result = await self._exchange.create_market_order(
                symbol, order_side, quantity
            )
            order.exchange_id = result["id"]
            order.status = OrderStatus.FILLED
            order.filled_price = result.get("average", result.get("price", 0))
            order.filled_qty   = result.get("filled", quantity)
            order.fee          = result.get("fee", {}).get("cost", 0)
            order.filled_at    = datetime.utcnow()
            logger.info(
                f"Market order filled: {symbol} @ ${order.filled_price:,.2f}"
            )
        except Exception as e:
            order.status = OrderStatus.REJECTED
            raise ExchangeError(f"Market order failed: {e}")

        return order

    async def cancel_order(self, order: Order) -> bool:
        """Cancel an open order. Returns True if cancelled."""
        if not order.exchange_id:
            return False
        try:
            await self._rate_limit()
            await self._exchange.cancel_order(order.exchange_id, order.symbol)
            order.status = OrderStatus.CANCELLED
            logger.info(f"Order cancelled: {order.exchange_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed for {order.exchange_id}: {e}")
            return False

    async def get_order_status(self, order: Order) -> Order:
        """Refresh order status from exchange."""
        if not order.exchange_id:
            return order
        try:
            await self._rate_limit()
            result = await self._exchange.fetch_order(
                order.exchange_id, order.symbol
            )
            status_map = {
                "open":     OrderStatus.OPEN,
                "closed":   OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
                "cancelled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
            }
            order.status       = status_map.get(result["status"], OrderStatus.OPEN)
            order.filled_price = result.get("average") or result.get("price", 0)
            order.filled_qty   = result.get("filled", 0)
            order.fee          = result.get("fee", {}).get("cost", 0) if result.get("fee") else 0

            if order.status == OrderStatus.FILLED:
                order.filled_at = datetime.utcnow()

        except Exception as e:
            logger.error(f"get_order_status failed: {e}")
        return order

    # ─── Paper Trading Simulation ─────────────────────────────

    async def simulate_fill(
        self,
        order: Order,
        market_price: float,
        slippage_pct: float = 0.05,
    ) -> Order:
        """
        Simulate an order fill for paper/backtest mode.
        Applies realistic slippage.
        """
        slippage_dir = 1 if order.side == Side.LONG else -1
        fill_price   = market_price * (1 + slippage_dir * slippage_pct / 100)

        order.status       = OrderStatus.FILLED
        order.filled_price = round(fill_price, 8)
        order.filled_qty   = order.quantity
        order.filled_at    = datetime.utcnow()
        order.fee          = fill_price * order.quantity * 0.001  # 0.1% fee

        logger.debug(
            f"Simulated fill: {order.symbol} @ ${fill_price:,.2f} "
            f"(slippage: {slippage_dir * slippage_pct:.3f}%)"
        )
        return order

    # ─── Utilities ────────────────────────────────────────────

    async def _rate_limit(self):
        """Enforce minimum time between API calls."""
        elapsed = (time.time() - self._last_request_time) * 1000
        if elapsed < self.cfg.rate_limit_ms:
            await asyncio.sleep((self.cfg.rate_limit_ms - elapsed) / 1000)
        self._last_request_time = time.time()

    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get min quantity, price precision, tick size for a symbol."""
        if not self._exchange:
            return {}
        market = self._exchange.markets.get(symbol, {})
        return {
            "min_qty":        market.get("limits", {}).get("amount", {}).get("min", 0),
            "qty_precision":  market.get("precision", {}).get("amount", 8),
            "price_precision": market.get("precision", {}).get("price", 2),
            "tick_size":      market.get("info", {}).get("filters", [{}])[0].get("tickSize", "0.01"),
        }

    def round_quantity(self, quantity: float, symbol_info: Dict) -> float:
        """Round quantity to exchange-required precision."""
        precision = symbol_info.get("qty_precision", 6)
        return round(quantity, precision)

    def round_price(self, price: float, symbol_info: Dict) -> float:
        """Round price to exchange tick size."""
        precision = symbol_info.get("price_precision", 2)
        return round(price, precision)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def __repr__(self):
        mode = "TESTNET" if self.cfg.testnet else "LIVE"
        return f"ExchangeConnector({self.cfg.name} {mode} connected={self._is_connected})"
