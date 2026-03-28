"""
=============================================================
  GATE.IO MARKET SCANNER  —  FastAPI Backend
  • Serves the SPA frontend (static/)
  • REST:  /api/results  /api/scan  /api/ohlcv  /api/news  /api/sentiment
  • WS:    /ws — real-time scan progress + results push
=============================================================
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from scanner import MarketScanner
from news import NewsClient, confluence_label
import journal as jnl
import capital as cap_mgr
import probability as prob_engine

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────
scanner:      MarketScanner = None
news_client:  NewsClient    = None

latest_results: List[dict]          = []
sentiment_cache: Dict[str, dict]    = {}   # coin → sentiment dict
market_news_cache: List[dict]       = []   # latest market-wide news
announcements_cache: List[dict]     = []   # Gate exchange announcements
connected_ws: Set[WebSocket]        = set()

scan_status = {
    "scanning":      False,
    "last_scan":     None,
    "total_scanned": 0,
    "timeframe":     "4h",
    "progress":      {"done": 0, "total": 0},
}

NEWS_REFRESH_INTERVAL = 180   # refresh market news every 3 min
AUTO_SCAN_INTERVAL    = 300   # re-scan every 5 min


# ── helpers ───────────────────────────────────────────────────

def _base_coin(symbol: str) -> str:
    """BTC/USDT → BTC"""
    return symbol.split("/")[0].upper()


async def broadcast(data: dict):
    dead = set()
    text = json.dumps(data)
    for ws in list(connected_ws):
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    connected_ws.difference_update(dead)


# ── sentiment enrichment ──────────────────────────────────────

async def enrich_with_sentiment(results: List[dict]) -> List[dict]:
    """
    Fetch social sentiment for the top 25 coins (by score) and
    inject sentiment + confluence fields into each result dict.
    """
    top_coins = list({_base_coin(r["symbol"]) for r in results[:25]})
    logger.info(f"Fetching sentiment for {len(top_coins)} coins…")

    sentiments = await news_client.get_batch_sentiment(top_coins)

    for r in results:
        coin = _base_coin(r["symbol"])
        sent = sentiments.get(coin)
        if sent:
            r["sentiment"] = sent.to_dict()
            r["confluence"] = confluence_label(r["signal"], sent)
            sentiment_cache[coin] = sent.to_dict()
        else:
            r["sentiment"] = None
            r["confluence"] = "MIXED"

        # Re-compute probability now that we have sentiment data
        if r.get("score") is not None:
            sentiment_strong = (r.get("confluence") == "STRONG")
            new_prob = prob_engine.estimate_win_probability(
                score=r.get("score", 0),
                mtf_aligned=r.get("mtf_aligned", False),
                sentiment_confluence=sentiment_strong,
                adx=r.get("adx"),
                rsi=r.get("rsi"),
            )
            new_grade = prob_engine.signal_grade(
                new_prob, r.get("score", 0), r.get("mtf_aligned", False)
            )
            grade_risk = prob_engine.GRADE_RISK.get(new_grade, 1.5)
            entry, stop, target = r.get("entry"), r.get("stop"), r.get("target")
            if entry and stop and target and abs(entry - stop) > 0:
                rr = abs(target - entry) / abs(entry - stop)
            else:
                rr = 2.0
            r["probability"]    = round(new_prob, 4)
            r["grade"]          = new_grade
            r["expected_value"] = prob_engine.expected_value_pct(new_prob, rr, grade_risk)

    logger.info("Sentiment enrichment done")
    return results


# ── scan flow ─────────────────────────────────────────────────

async def run_scan(timeframe: str = "4h", max_pairs: int = 80):
    global latest_results, scan_status

    if scan_status["scanning"]:
        return

    scan_status["scanning"] = True
    scan_status["timeframe"] = timeframe
    scan_status["progress"]  = {"done": 0, "total": 0}
    scanner.timeframe = timeframe

    await broadcast({"type": "scan_start", "timeframe": timeframe})

    async def progress_cb(done: int, total: int):
        scan_status["progress"] = {"done": done, "total": total}
        await broadcast({"type": "scan_progress", "done": done, "total": total})

    try:
        results = await scanner.scan(max_pairs=max_pairs, progress_cb=progress_cb)
        raw = [r.to_dict() for r in results]

        # Enrich top results with news sentiment
        await broadcast({"type": "scan_progress", "done": max_pairs, "total": max_pairs,
                         "message": "Fetching sentiment…"})
        enriched = await enrich_with_sentiment(raw)

        latest_results = enriched
        scan_status.update({
            "scanning":      False,
            "last_scan":     datetime.now(timezone.utc).isoformat(),
            "total_scanned": len(latest_results),
            "progress":      {"done": max_pairs, "total": max_pairs},
        })
        # Auto-log actionable signals to journal
        logged = 0
        for r in latest_results:
            r["timeframe"] = timeframe
            entry = jnl.log_signal(r)
            if entry:
                logged += 1
        if logged:
            logger.info(f"Journal: auto-logged {logged} new signals")
            await broadcast({"type": "journal_update", "summary": jnl.get_summary(),
                             "capital": cap_mgr.get_state()})

        await broadcast({
            "type":   "scan_results",
            "data":   latest_results,
            "status": scan_status,
        })
        logger.info(f"Broadcast {len(latest_results)} enriched results")

    except Exception as e:
        scan_status["scanning"] = False
        logger.error(f"Scan error: {e}", exc_info=True)
        await broadcast({"type": "error", "message": str(e)})


# ── background news refresh ───────────────────────────────────

async def _news_refresh_loop():
    """Periodically refresh market news and push to clients."""
    global market_news_cache, announcements_cache
    while True:
        await asyncio.sleep(NEWS_REFRESH_INTERVAL)
        try:
            news  = await news_client.get_news(limit=20)
            anns  = await news_client.get_exchange_announcements(limit=10)
            market_news_cache   = [n.to_dict() for n in news]
            announcements_cache = anns
            await broadcast({
                "type":          "news_update",
                "news":          market_news_cache,
                "announcements": announcements_cache,
            })
            logger.info(f"News refreshed: {len(market_news_cache)} items")
        except Exception as e:
            logger.warning(f"News refresh failed: {e}")


async def _auto_scan_loop():
    while True:
        await asyncio.sleep(AUTO_SCAN_INTERVAL)
        logger.info("Auto-scan triggered")
        await run_scan(timeframe=scan_status.get("timeframe", "4h"))


# ── app lifecycle ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scanner, news_client, market_news_cache, announcements_cache

    # Start exchange scanner
    scanner = MarketScanner(timeframe="4h", min_volume_usdt=500_000)
    await scanner.connect()

    # Start news client + warm up session
    news_client = NewsClient()
    await news_client.ensure_session()

    # Fetch initial news before first scan
    try:
        news  = await news_client.get_news(limit=20)
        anns  = await news_client.get_exchange_announcements(limit=10)
        market_news_cache   = [n.to_dict() for n in news]
        announcements_cache = anns
        logger.info(f"Initial news: {len(market_news_cache)} items")
    except Exception as e:
        logger.warning(f"Initial news fetch failed: {e}")

    # Background tasks
    scan_task = asyncio.create_task(_auto_scan_loop())
    news_task = asyncio.create_task(_news_refresh_loop())
    asyncio.create_task(run_scan())   # initial scan on startup

    yield

    scan_task.cancel()
    news_task.cancel()
    scanner.disconnect()
    news_client.shutdown()


app = FastAPI(title="Gate.io Market Scanner", lifespan=lifespan)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── REST endpoints ────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")


@app.get("/api/status")
async def api_status():
    return scan_status


@app.get("/api/results")
async def api_results():
    return {"data": latest_results, "status": scan_status}


@app.post("/api/scan")
async def api_scan(timeframe: str = "4h", max_pairs: int = 80):
    if scan_status["scanning"]:
        return {"ok": False, "reason": "scan already running"}
    asyncio.create_task(run_scan(timeframe=timeframe, max_pairs=max_pairs))
    return {"ok": True}


@app.get("/api/ohlcv/{symbol:path}")
async def api_ohlcv(symbol: str, timeframe: str = "4h", limit: int = 200):
    try:
        raw = await scanner.fetch_ohlcv(
            symbol.replace("-", "/"), timeframe, limit=limit
        )
        return {"data": [
            {"t": b[0], "o": b[1], "h": b[2], "l": b[3], "c": b[4], "v": b[5]}
            for b in raw
        ]}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/news")
async def api_news():
    """Latest market-wide news + Gate announcements."""
    return {
        "news":          market_news_cache,
        "announcements": announcements_cache,
    }


@app.get("/api/news/{coin}")
async def api_coin_news(coin: str, limit: int = 10):
    """Live news + sentiment for a specific coin (e.g. BTC, ETH)."""
    coin = coin.upper()
    try:
        news_items = await news_client.get_news(coin=coin, limit=limit)
        sent = await news_client.get_sentiment(coin)
        return {
            "coin":      coin,
            "sentiment": sent.to_dict() if sent else None,
            "news":      [n.to_dict() for n in news_items],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/journal")
async def api_journal(limit: int = 200):
    return {"entries": jnl.get_entries(limit), "summary": jnl.get_summary()}


@app.post("/api/journal/log")
async def api_journal_log(symbol: str):
    """Manually log the latest scan result for a symbol."""
    match = next((r for r in latest_results if r["symbol"] == symbol), None)
    if not match:
        return {"ok": False, "reason": "symbol not in latest scan"}
    entry = jnl.log_signal({**match, "timeframe": scan_status.get("timeframe", "4h")})
    if entry:
        await broadcast({"type": "journal_update", "summary": jnl.get_summary()})
        return {"ok": True, "entry": entry}
    return {"ok": False, "reason": "already logged or no entry/stop/target"}


@app.put("/api/journal/{entry_id}")
async def api_journal_update(entry_id: str, status: str,
                              exit_price: Optional[float] = None,
                              notes: str = ""):
    updated = jnl.update_entry(entry_id, status, exit_price, notes)
    if updated:
        await broadcast({"type": "journal_update", "summary": jnl.get_summary()})
        return {"ok": True, "entry": updated}
    return {"ok": False, "reason": "entry not found"}


@app.get("/api/capital")
async def api_capital():
    """Current capital state and position sizing."""
    return cap_mgr.get_state()


@app.put("/api/capital")
async def api_capital_update(risk_pct: Optional[float] = None, reset: bool = False):
    """Update risk % or reset capital to initial."""
    state = cap_mgr.update_settings(risk_pct=risk_pct, reset=reset)
    await broadcast({"type": "capital_update", "capital": cap_mgr.get_state()})
    return state


@app.get("/api/sentiment/{coin}")
async def api_sentiment(coin: str, time_range: str = "24h"):
    """Social sentiment for a coin."""
    coin = coin.upper()
    # Return cache if fresh (< 5 min old)
    if coin in sentiment_cache:
        return sentiment_cache[coin]
    sent = await news_client.get_sentiment(coin, time_range)
    if sent:
        sentiment_cache[coin] = sent.to_dict()
        return sentiment_cache[coin]
    return {"error": "sentiment unavailable"}


# ── WebSocket endpoint ────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected_ws.add(ws)
    logger.info(f"WS connected ({len(connected_ws)} clients)")

    await ws.send_text(json.dumps({
        "type":          "init",
        "data":          latest_results,
        "status":        scan_status,
        "news":          market_news_cache,
        "announcements": announcements_cache,
        "capital":       cap_mgr.get_state(),
    }))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("action") == "scan":
                asyncio.create_task(run_scan(
                    timeframe=msg.get("timeframe", "4h"),
                    max_pairs=int(msg.get("max_pairs", 80)),
                ))
    except WebSocketDisconnect:
        connected_ws.discard(ws)
        logger.info(f"WS disconnected ({len(connected_ws)} clients)")


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000,
                reload=False, log_level="info")
