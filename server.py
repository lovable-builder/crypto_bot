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

# Load .env file if present (safe no-op if file doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass
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
import evaluation as eval_engine
import ai_analyst

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────
scanner:      MarketScanner = None
news_client:  NewsClient    = None

latest_spot_results:    List[dict]       = []
latest_futures_results: List[dict]       = []
sentiment_cache: Dict[str, dict]         = {}   # coin → sentiment dict
market_news_cache: List[dict]            = []   # latest market-wide news
announcements_cache: List[dict]          = []   # Gate exchange announcements
connected_ws: Set[WebSocket]             = set()

scan_status = {
    "scanning":          False,
    "last_scan":         None,
    "total_scanned":     0,
    "timeframe":         "4h",
    "progress":          {"done": 0, "total": 0},
    "futures_scanning":  False,
    "futures_last_scan": None,
    "futures_total":     0,
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

async def run_spot_scan(timeframe: str = "4h", max_pairs: int = 80):
    global latest_spot_results, scan_status

    if scan_status["scanning"]:
        return

    scan_status["scanning"] = True
    scan_status["timeframe"] = timeframe
    scan_status["progress"]  = {"done": 0, "total": 0}
    scanner.timeframe = timeframe

    await broadcast({"type": "scan_start", "timeframe": timeframe, "market": "spot"})

    async def progress_cb(done: int, total: int):
        scan_status["progress"] = {"done": done, "total": total}
        await broadcast({"type": "scan_progress", "done": done, "total": total, "market": "spot"})

    try:
        results = await scanner.scan(max_pairs=max_pairs, progress_cb=progress_cb)
        raw = [r.to_dict() for r in results]

        await broadcast({"type": "scan_progress", "done": max_pairs, "total": max_pairs,
                         "message": "Fetching sentiment…", "market": "spot"})
        enriched = await enrich_with_sentiment(raw)

        latest_spot_results = enriched
        scan_status.update({
            "scanning":      False,
            "last_scan":     datetime.now(timezone.utc).isoformat(),
            "total_scanned": len(latest_spot_results),
            "progress":      {"done": max_pairs, "total": max_pairs},
        })
        logged = 0
        budget = cap_mgr.get_daily_budget()
        if budget["daily_blocked"]:
            logger.warning(
                f"Daily loss cap hit ({budget['daily_loss_usdt']:.2f} / "
                f"{budget['daily_cap_usdt']:.2f} USDT) — no new trades logged"
            )
            await broadcast({
                "type":             "daily_limit_hit",
                "daily_loss_usdt":  budget["daily_loss_usdt"],
                "daily_cap_usdt":   budget["daily_cap_usdt"],
                "daily_used_pct":   budget["daily_used_pct"],
            })
        else:
            open_count = len([e for e in jnl.get_entries(500) if e.get("status") == "OPEN"])
            for r in latest_spot_results:
                if open_count >= cap_mgr.MAX_CONCURRENT_TRADES:
                    logger.info(f"Concurrent trade limit reached ({open_count}/{cap_mgr.MAX_CONCURRENT_TRADES}) — skipping spot logging")
                    break
                r["timeframe"] = timeframe
                r["dynamic_risk_pct"] = budget["per_trade_risk_pct"]
                entry = jnl.log_signal(r)
                if entry:
                    logged += 1
                    open_count += 1
        if logged:
            logger.info(f"Journal: auto-logged {logged} spot signals (risk={budget['per_trade_risk_pct']}%/trade)")
            await broadcast({"type": "journal_update", "summary": jnl.get_summary(),
                             "capital": cap_mgr.get_state()})

        await broadcast({
            "type":   "scan_results",
            "market": "spot",
            "data":   latest_spot_results,
            "status": scan_status,
        })
        logger.info(f"Spot scan: broadcast {len(latest_spot_results)} results")

    except Exception as e:
        scan_status["scanning"] = False
        logger.error(f"Spot scan error: {e}", exc_info=True)
        await broadcast({"type": "error", "message": str(e)})


# Keep backward-compat alias
async def run_scan(timeframe: str = "4h", max_pairs: int = 80):
    await run_spot_scan(timeframe=timeframe, max_pairs=max_pairs)


async def run_futures_scan(timeframe: str = "4h", max_pairs: int = 50):
    global latest_futures_results, scan_status

    if scan_status["futures_scanning"]:
        return

    scan_status["futures_scanning"] = True
    await broadcast({"type": "scan_start", "timeframe": timeframe, "market": "futures"})

    async def progress_cb(done: int, total: int):
        await broadcast({"type": "scan_progress", "done": done, "total": total, "market": "futures"})

    try:
        results = await scanner.scan_futures(max_pairs=max_pairs, progress_cb=progress_cb)
        raw = [r.to_dict() for r in results]

        await broadcast({"type": "scan_progress", "done": max_pairs, "total": max_pairs,
                         "message": "Fetching sentiment…", "market": "futures"})
        enriched = await enrich_with_sentiment(raw)

        latest_futures_results = enriched
        scan_status.update({
            "futures_scanning":  False,
            "futures_last_scan": datetime.now(timezone.utc).isoformat(),
            "futures_total":     len(latest_futures_results),
        })
        logged = 0
        budget = cap_mgr.get_daily_budget()
        if budget["daily_blocked"]:
            logger.warning(
                f"Daily loss cap hit — skipping futures journal logging "
                f"({budget['daily_loss_usdt']:.2f}/{budget['daily_cap_usdt']:.2f} USDT)"
            )
        else:
            open_count = len([e for e in jnl.get_entries(500) if e.get("status") == "OPEN"])
            for r in latest_futures_results:
                if open_count >= cap_mgr.MAX_CONCURRENT_TRADES:
                    logger.info(f"Concurrent trade limit reached ({open_count}/{cap_mgr.MAX_CONCURRENT_TRADES}) — skipping futures logging")
                    break
                r["timeframe"] = timeframe
                r["dynamic_risk_pct"] = budget["per_trade_risk_pct"]
                entry = jnl.log_signal(r)
                if entry:
                    logged += 1
                    open_count += 1
        if logged:
            logger.info(f"Journal: auto-logged {logged} futures signals (risk={budget['per_trade_risk_pct']}%/trade)")
            await broadcast({"type": "journal_update", "summary": jnl.get_summary(),
                             "capital": cap_mgr.get_state()})

        await broadcast({
            "type":   "scan_results",
            "market": "futures",
            "data":   latest_futures_results,
            "status": scan_status,
        })
        logger.info(f"Futures scan: broadcast {len(latest_futures_results)} results")

    except Exception as e:
        scan_status["futures_scanning"] = False
        logger.error(f"Futures scan error: {e}", exc_info=True)
        await broadcast({"type": "error", "message": str(e), "market": "futures"})


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
        logger.info("Auto-scan triggered (spot)")
        await run_spot_scan(timeframe=scan_status.get("timeframe", "4h"))
        await asyncio.sleep(10)   # stagger to avoid rate limiting
        logger.info("Auto-scan triggered (futures)")
        await run_futures_scan(timeframe=scan_status.get("timeframe", "4h"))


PRICE_MONITOR_INTERVAL = 60   # check SL/TP every 60 seconds


async def _run_price_check() -> dict:
    """
    Fetch live prices for all OPEN journal entries, check SL/TP, auto-close hits,
    and broadcast live_prices with unrealized PnL for every open position.
    Returns a summary dict.
    """
    open_entries = [e for e in jnl.get_entries(500) if e.get("status") == "OPEN"]
    if not open_entries:
        return {"open": 0, "closed": 0, "prices": {}}

    spot_entries    = [e for e in open_entries if e.get("market_type", "spot") == "spot"]
    futures_entries = [e for e in open_entries if e.get("market_type") == "futures"]
    prices: Dict[str, float] = {}
    if spot_entries:
        sp = await scanner.fetch_current_prices(
            list({e["symbol"] for e in spot_entries}), "spot")
        prices.update(sp)
    if futures_entries:
        fp = await scanner.fetch_current_prices(
            list({e["symbol"] for e in futures_entries}), "futures")
        prices.update(fp)

    closed_count = 0
    live_positions = []

    for entry in open_entries:
        price = prices.get(entry["symbol"])
        if not price:
            continue
        tp      = entry.get("target")
        sl      = entry.get("stop")
        sig     = entry.get("signal", "")
        entry_p = entry.get("entry") or 0
        pos_val = entry.get("position_value") or 0

        # Unrealized PnL calculation
        if entry_p and price:
            if sig == "BULLISH":
                unreal_pct = (price - entry_p) / entry_p * 100
            else:  # BEARISH
                unreal_pct = (entry_p - price) / entry_p * 100
            unreal_usdt = pos_val * unreal_pct / 100
        else:
            unreal_pct = unreal_usdt = 0.0

        # Distance to TP and SL in %
        if entry_p:
            if sig == "BULLISH":
                pct_to_tp = (tp - price) / price * 100 if tp else None
                pct_to_sl = (price - sl) / price * 100 if sl else None
            else:
                pct_to_tp = (price - tp) / price * 100 if tp else None
                pct_to_sl = (sl - price) / price * 100 if sl else None
        else:
            pct_to_tp = pct_to_sl = None

        live_positions.append({
            "id":          entry["id"],
            "symbol":      entry["symbol"],
            "price":       price,
            "unreal_pct":  round(unreal_pct, 3),
            "unreal_usdt": round(unreal_usdt, 2),
            "pct_to_tp":   round(pct_to_tp, 2) if pct_to_tp is not None else None,
            "pct_to_sl":   round(pct_to_sl, 2) if pct_to_sl is not None else None,
        })

        # SL/TP hit check
        if not tp or not sl:
            continue
        hit = None
        exit_price = None
        if sig == "BULLISH":
            if entry_p and sl >= entry_p:
                logger.warning("Price monitor skipping %s BULLISH: sl=%s >= entry=%s (malformed)",
                               entry["symbol"], sl, entry_p)
                continue
            if price >= tp:   hit, exit_price = "WIN",  tp
            elif price <= sl: hit, exit_price = "LOSS", sl
        elif sig == "BEARISH":
            if entry_p and sl <= entry_p:
                logger.warning("Price monitor skipping %s BEARISH: sl=%s <= entry=%s (malformed)",
                               entry["symbol"], sl, entry_p)
                continue
            if price <= tp:   hit, exit_price = "WIN",  tp
            elif price >= sl: hit, exit_price = "LOSS", sl

        if hit:
            updated = jnl.update_entry(entry["id"], hit, exit_price,
                                       notes="auto-closed by price monitor")
            if updated:
                closed_count += 1
                logger.info(f"Price monitor: {entry['symbol']} → {hit} "
                            f"(price={price}, exit={exit_price})")

    # Always broadcast live prices / unrealized PnL to UI
    checked_at = datetime.now(timezone.utc).isoformat()
    await broadcast({
        "type":        "live_prices",
        "positions":   live_positions,
        "checked_at":  checked_at,
    })

    if closed_count:
        await broadcast({
            "type":    "journal_update",
            "summary": jnl.get_summary(),
            "capital": cap_mgr.get_state(),
            "reload":  True,
        })

    return {"open": len(open_entries), "closed": closed_count, "prices": prices}


async def _price_monitor_loop():
    """Background task: auto-close OPEN journal entries when SL or TP is hit."""
    await asyncio.sleep(30)   # let the server fully start first
    _last_daily_reset_date = ""
    while True:
        try:
            # Midnight UTC reset
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != _last_daily_reset_date and _last_daily_reset_date:
                budget = cap_mgr.get_daily_budget()
                await broadcast({
                    "type":    "daily_budget_reset",
                    "date":    today,
                    "capital": cap_mgr.get_state(),
                    "budget":  budget,
                })
                logger.info(f"Daily budget reset for {today}")
            _last_daily_reset_date = today

            await _run_price_check()

        except Exception as e:
            logger.warning(f"Price monitor error: {e}")

        await asyncio.sleep(PRICE_MONITOR_INTERVAL)


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

    # Rebuild capital from journal on startup to ensure sync
    try:
        entries = jnl.get_entries()
        closed = [e for e in entries if e.get("status") in ("WIN", "LOSS")]
        cap_mgr.rebuild_from_journal(closed)
    except Exception as e:
        logger.warning(f"Capital rebuild on startup failed: {e}")

    # Background tasks
    scan_task    = asyncio.create_task(_auto_scan_loop())
    news_task    = asyncio.create_task(_news_refresh_loop())
    monitor_task = asyncio.create_task(_price_monitor_loop())
    asyncio.create_task(run_scan())   # initial scan on startup

    yield

    scan_task.cancel()
    news_task.cancel()
    monitor_task.cancel()
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
async def api_results(market: str = "spot"):
    """Returns spot or futures scan results. market=spot|futures"""
    data = latest_futures_results if market == "futures" else latest_spot_results
    return {"data": data, "status": scan_status, "market": market}


@app.get("/api/results/breakout")
async def api_results_breakout(market: str = "spot"):
    """Returns only HIGH/MEDIUM breakout_potential coins from latest scan."""
    data = latest_futures_results if market == "futures" else latest_spot_results
    filtered = [
        r for r in data
        if r.get("breakout_potential") in ("HIGH", "MEDIUM")
    ]
    return {
        "data":   filtered,
        "total":  len(filtered),
        "market": market,
        "status": scan_status,
    }


@app.post("/api/scan")
async def api_scan(timeframe: str = "4h", max_pairs: int = 80, market: str = "spot"):
    if market == "futures":
        if scan_status["futures_scanning"]:
            return {"ok": False, "reason": "futures scan already running"}
        asyncio.create_task(run_futures_scan(timeframe=timeframe, max_pairs=max_pairs))
    else:
        if scan_status["scanning"]:
            return {"ok": False, "reason": "spot scan already running"}
        asyncio.create_task(run_spot_scan(timeframe=timeframe, max_pairs=max_pairs))
    return {"ok": True, "market": market}


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
    """Manually log the latest scan result for a symbol (checks spot + futures)."""
    match = (next((r for r in latest_spot_results    if r["symbol"] == symbol), None) or
             next((r for r in latest_futures_results if r["symbol"] == symbol), None))
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
        await broadcast({
            "type":    "journal_update",
            "summary": jnl.get_summary(),
            "capital": cap_mgr.get_state(),
            "reload":  True,
        })
        return {"ok": True, "entry": updated}
    return {"ok": False, "reason": "entry not found"}


@app.get("/api/capital")
async def api_capital():
    """Current capital state and position sizing."""
    return cap_mgr.get_state()


@app.post("/api/capital/reconcile")
async def api_capital_reconcile():
    """Rebuild capital.json from journal entries to fix any sync drift."""
    try:
        entries = jnl.get_entries()
        closed = [e for e in entries if e.get("status") in ("WIN", "LOSS")]
        state = cap_mgr.rebuild_from_journal(closed)
        # Broadcast capital update to connected clients
        await broadcast({
            "type": "capital_reconciled",
            "capital": state.get("capital"),
            "peak": state.get("peak"),
        })
        return {"success": True, "capital": state.get("capital"), "peak": state.get("peak")}
    except Exception as e:
        logger.error(f"Capital reconcile failed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/monitor/check")
async def api_monitor_check():
    """Manually trigger an immediate price check for all OPEN positions."""
    try:
        result = await _run_price_check()
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Manual price check failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/evaluation")
async def api_evaluation():
    """Comprehensive trade performance analytics by grade, calibration, and capital growth."""
    return eval_engine.get_evaluation()


@app.get("/api/evaluation/ai")
async def api_evaluation_ai(force: bool = False):
    """
    Return cached AI analysis of trading performance.
    If cache is stale (>6h) or force=True, triggers a background regeneration
    and returns the (potentially old) cached result immediately.
    Returns {analysis, generated_at, fresh, pending} structure.
    """
    import os
    if not os.getenv("OPENAI_API_KEY", ""):
        return {"error": "OPENAI_API_KEY environment variable not set", "analysis": None}

    # Guard: don't trigger AI analysis until there are enough closed trades
    closed_count = jnl.get_summary().get("closed", 0)
    if closed_count < ai_analyst.MIN_TRADES:
        return {
            "error": f"Need at least {ai_analyst.MIN_TRADES} closed trades (have {closed_count})",
            "analysis": None, "pending": False, "insufficient_trades": True,
        }

    cached = ai_analyst.get_cached_analysis()

    # Kick off background refresh if stale or forced
    if force or ai_analyst.cache_is_stale():
        asyncio.create_task(_run_ai_analysis())

    if cached:
        return {**cached, "pending": force or not cached["fresh"]}

    # No cache at all — wait for the first analysis
    return {"analysis": None, "generated_at": None, "fresh": False, "pending": True}


@app.post("/api/evaluation/ai/refresh")
async def api_evaluation_ai_refresh():
    """Force a fresh AI analysis regardless of cache age."""
    import os
    if not os.getenv("OPENAI_API_KEY", ""):
        return {"error": "OPENAI_API_KEY environment variable not set"}
    closed_count = jnl.get_summary().get("closed", 0)
    if closed_count < ai_analyst.MIN_TRADES:
        return {"error": f"Need at least {ai_analyst.MIN_TRADES} closed trades (have {closed_count})"}
    asyncio.create_task(_run_ai_analysis())
    return {"ok": True, "message": "AI analysis refresh triggered"}


async def _run_ai_analysis():
    """Background task: generate fresh AI analysis and broadcast result."""
    try:
        evaluation = eval_engine.get_evaluation()
        recent_trades = jnl.get_entries(limit=20)
        analysis = await ai_analyst.analyze_performance(evaluation, recent_trades)
        cached = ai_analyst.get_cached_analysis()
        await broadcast({"type": "ai_analysis_update", "analysis": cached})
        logger.info("AI analysis completed and broadcast")
    except Exception as e:
        logger.warning("AI analysis failed: %s", e)
        await broadcast({"type": "ai_analysis_update", "error": str(e)})


@app.put("/api/capital")
async def api_capital_update(risk_pct: Optional[float] = None,
                              reset: bool = False,
                              capital: Optional[float] = None):
    """Update risk %, set a new capital amount, or reset capital to initial."""
    state = cap_mgr.update_settings(risk_pct=risk_pct, reset=reset, new_capital=capital)
    await broadcast({"type": "capital_update", "capital": cap_mgr.get_state()})
    return state


@app.delete("/api/journal")
async def api_journal_reset():
    """Clear all journal entries (backs up first)."""
    result = jnl.reset_journal()
    await broadcast({"type": "journal_update", "summary": jnl.get_summary()})
    return result


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
        "type":            "init",
        "data":            latest_spot_results,
        "futures_data":    latest_futures_results,
        "status":          scan_status,
        "news":            market_news_cache,
        "announcements":   announcements_cache,
        "capital":         cap_mgr.get_state(),
    }))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("action") == "scan":
                tf = msg.get("timeframe", "4h")
                if msg.get("market") == "futures":
                    asyncio.create_task(run_futures_scan(
                        timeframe=tf, max_pairs=int(msg.get("max_pairs", 50))))
                else:
                    asyncio.create_task(run_spot_scan(
                        timeframe=tf, max_pairs=int(msg.get("max_pairs", 80))))
    except WebSocketDisconnect:
        connected_ws.discard(ws)
        logger.info(f"WS disconnected ({len(connected_ws)} clients)")


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000,
                reload=False, log_level="info")
