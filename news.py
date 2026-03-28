"""
=============================================================
  GATE.IO NEWS & SENTIMENT CLIENT
  Calls api.gatemcp.ai/mcp/news via JSON-RPC 2.0 over HTTPS.
  Uses urllib.request (no aiohttp / aiodns) in a thread pool.
=============================================================
"""

import asyncio
import json
import logging
import math
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

NEWS_MCP_URL = "https://api.gatemcp.ai/mcp/news"
TIMEOUT = 15


# ── data classes ──────────────────────────────────────────────

@dataclass
class Tweet:
    content: str
    url: str
    likes: int
    comments: int
    created_time: str
    sentiment: float


@dataclass
class Sentiment:
    coin: str
    overall: float          # −1 … +1
    label: str              # "Bearish" | "Neutral" | "Bullish"
    mention_count: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    tweets: List[Tweet] = field(default_factory=list)
    fetched_at: str = ""

    @property
    def label_en(self) -> str:
        if self.overall > 0.1:
            return "Bullish"
        if self.overall < -0.1:
            return "Bearish"
        return "Neutral"

    @property
    def emoji(self) -> str:
        return {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}[self.label_en]

    def to_dict(self) -> dict:
        return {
            "coin":            self.coin,
            "overall":         round(self.overall, 4),
            "label":           self.label_en,
            "emoji":           self.emoji,
            "mention_count":   self.mention_count,
            "positive_ratio":  round(self.positive_ratio, 3),
            "negative_ratio":  round(self.negative_ratio, 3),
            "neutral_ratio":   round(self.neutral_ratio, 3),
            "tweets":          [
                {
                    "content":      t.content,
                    "url":          t.url,
                    "likes":        t.likes,
                    "comments":     t.comments,
                    "created_time": t.created_time,
                }
                for t in self.tweets[:5]
            ],
            "fetched_at": self.fetched_at,
        }


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published_at: str
    coins: List[str]
    sentiment: float

    def to_dict(self) -> dict:
        return {
            "title":        self.title,
            "url":          self.url,
            "source":       self.source,
            "published_at": self.published_at,
            "coins":        self.coins,
            "sentiment":    self.sentiment,
        }


# ── client ────────────────────────────────────────────────────

class NewsClient:
    """
    Thin async wrapper around the Gate MCP /news endpoint.
    Manages session lifecycle; all HTTP done in a thread pool.
    """

    def __init__(self, workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._session_id: Optional[str] = None
        self._ssl_ctx = ssl.create_default_context()

    # ── helpers ───────────────────────────────────────────────

    async def _run(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, fn, *args)

    def _post(self, body: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept":       "application/json",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            NEWS_MCP_URL, data=data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=self._ssl_ctx) as r:
            # capture session id on first call
            sid = r.headers.get("mcp-session-id")
            if sid and not self._session_id:
                self._session_id = sid
            return json.loads(r.read().decode())

    def _call_tool(self, tool: str, arguments: dict) -> dict:
        body = {
            "jsonrpc": "2.0",
            "method":  "tools/call",
            "params":  {"name": tool, "arguments": arguments},
            "id":      1,
        }
        return self._post(body)

    def _initialize(self):
        body = {
            "jsonrpc": "2.0",
            "method":  "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities":    {},
                "clientInfo":      {"name": "gate-scanner", "version": "1.0"},
            },
            "id": 0,
        }
        self._post(body)

    # ── session management ────────────────────────────────────

    async def ensure_session(self):
        if not self._session_id:
            await self._run(self._initialize)

    # ── public API ────────────────────────────────────────────

    async def get_sentiment(
        self, coin: str, time_range: str = "24h"
    ) -> Optional[Sentiment]:
        """Fetch social sentiment for a single coin."""
        await self.ensure_session()
        try:
            resp = await self._run(
                self._call_tool,
                "news_feed_get_social_sentiment",
                {"coin": coin.upper(), "time_range": time_range},
            )
            sc = resp.get("result", {}).get("structuredContent") or {}
            if not sc:
                # fall back to parsing text content
                text = resp.get("result", {}).get("content", [{}])[0].get("text", "{}")
                sc = json.loads(text)

            dist = sc.get("sentiment_distribution", {})
            tweets = [
                Tweet(
                    content=t.get("content", ""),
                    url=t.get("url", ""),
                    likes=t.get("likes_num", 0),
                    comments=t.get("comments_num", 0),
                    created_time=t.get("created_time", ""),
                    sentiment=t.get("sentiment", 0),
                )
                for t in sc.get("top_tweets", [])
            ]
            return Sentiment(
                coin=coin.upper(),
                overall=sc.get("overall_sentiment", 0),
                label=sc.get("sentiment_label", ""),
                mention_count=sc.get("mention_count", 0),
                positive_ratio=dist.get("positive_ratio", 0),
                negative_ratio=dist.get("negative_ratio", 0),
                neutral_ratio=dist.get("neutral_ratio", 0),
                tweets=tweets,
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            logger.warning(f"Sentiment fetch failed for {coin}: {e}")
            return None

    async def get_batch_sentiment(
        self, coins: List[str], time_range: str = "24h"
    ) -> Dict[str, Sentiment]:
        """Fetch sentiment for multiple coins concurrently."""
        await self.ensure_session()
        tasks = [self.get_sentiment(c, time_range) for c in coins]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = {}
        for coin, r in zip(coins, results):
            if isinstance(r, Sentiment):
                out[coin.upper()] = r
        return out

    MARKET_COINS = ["BTC", "ETH", "SOL", "XRP", "BNB"]

    async def get_news(
        self, coin: Optional[str] = None, limit: int = 20
    ) -> List[NewsItem]:
        """
        Build a news feed from social sentiment tweets — the most reliable
        Gate MCP data source (always returns fresh content with engagement).

        coin=None → aggregate top tweets from BTC+ETH+SOL
        coin set  → tweets for that specific coin
        """
        await self.ensure_session()
        coins = [coin.upper()] if coin else self.MARKET_COINS[:3]

        all_tweets: List[NewsItem] = []
        for c in coins:
            sent = await self.get_sentiment(c, "24h")
            if not sent:
                continue
            for t in sent.tweets:
                all_tweets.append(NewsItem(
                    title=t.content,
                    url=t.url,
                    source="X / Twitter",
                    published_at=t.created_time,
                    coins=[c],
                    sentiment=t.sentiment,
                ))

        # Sort by engagement (likes), deduplicate by url
        seen: set = set()
        unique: List[NewsItem] = []
        for item in sorted(all_tweets, key=lambda x: 0, reverse=True):
            if item.url not in seen:
                seen.add(item.url)
                unique.append(item)

        return unique[:limit]

    async def get_exchange_announcements(self, limit: int = 10) -> List[dict]:
        """Gate.io listing/delisting/maintenance announcements."""
        await self.ensure_session()
        try:
            resp = await self._run(
                self._call_tool,
                "news_feed_get_exchange_announcements",
                {"exchange": "gate", "limit": limit},
            )
            sc = resp.get("result", {}).get("structuredContent") or {}
            if not sc:
                text = resp.get("result", {}).get("content", [{}])[0].get("text", "{}")
                sc = json.loads(text)
            return sc.get("items", [])
        except Exception as e:
            logger.warning(f"Announcements fetch failed: {e}")
            return []

    def shutdown(self):
        self._pool.shutdown(wait=False)


# ── confluence scoring ────────────────────────────────────────

def confluence_label(tech_signal: str, sentiment: Optional[Sentiment]) -> str:
    """
    Compare technical signal direction with news sentiment.

    Returns one of:
      "STRONG"   — signal + sentiment agree strongly
      "ALIGNED"  — signal + sentiment generally agree
      "MIXED"    — sentiment is neutral / insufficient data
      "CONFLICT" — sentiment contradicts the technical signal
    """
    if sentiment is None:
        return "MIXED"

    s = sentiment.overall
    if math.isnan(s):
        return "MIXED"

    if tech_signal == "BULLISH":
        if s > 0.15:   return "STRONG"
        if s > 0.0:    return "ALIGNED"
        if s < -0.15:  return "CONFLICT"
        return "MIXED"

    if tech_signal == "BEARISH":
        if s < -0.15:  return "STRONG"
        if s < 0.0:    return "ALIGNED"
        if s > 0.15:   return "CONFLICT"
        return "MIXED"

    return "MIXED"
