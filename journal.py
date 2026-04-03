"""
=============================================================
  TRADE JOURNAL
  Logs every actionable signal (entry/SL/TP present) to
  logs/journal.json.  Supports reading, updating (win/loss),
  and performance summary.
=============================================================
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import capital as cap_mgr

logger = logging.getLogger(__name__)

JOURNAL_FILE = os.path.join(os.path.dirname(__file__), "logs", "journal.json")
_lock = threading.Lock()


# ── Entry model ───────────────────────────────────────────────

def _make_entry(opp: dict) -> dict:
    entry_p = opp.get("entry")
    stop_p  = opp.get("stop")
    target_p = opp.get("target")
    rr = None
    if entry_p and stop_p and target_p:
        risk   = abs(entry_p - stop_p)
        reward = abs(target_p - entry_p)
        rr = round(reward / risk, 2) if risk > 0 else None

    # Grade-based dynamic position sizing
    # Respect dynamic_risk_pct if already set by caller (e.g. daily budget cap)
    grade = opp.get("grade", "B")
    market_type = opp.get("market_type", "spot")
    try:
        dynamic_risk_pct = float(opp["dynamic_risk_pct"]) if opp.get("dynamic_risk_pct") is not None \
                           else cap_mgr.get_dynamic_risk(grade)
    except Exception:
        dynamic_risk_pct = 1.0
    try:
        leverage = cap_mgr.get_leverage(grade, market_type)
    except Exception:
        leverage = 1

    pos = None
    if entry_p and stop_p:
        try:
            pos = cap_mgr.compute_position(
                entry_p, stop_p,
                risk_pct=dynamic_risk_pct,
                leverage=leverage,
                signal=opp.get("signal", "BULLISH"),
            )
        except Exception:
            pos = None

    sent = opp.get("sentiment") or {}
    now  = datetime.now(timezone.utc)
    return {
        "id":               f"{opp['symbol'].replace('/', '-')}-{now.strftime('%Y%m%dT%H%M%S')}",
        "logged_at":        now.isoformat(),
        "symbol":           opp["symbol"],
        "signal":           opp.get("signal", ""),
        "score":            opp.get("score"),
        "grade":            grade,
        "probability":      opp.get("probability"),
        "expected_value":   opp.get("expected_value"),
        "mtf_aligned":      opp.get("mtf_aligned", False),
        "daily_trend":      opp.get("daily_trend", "SIDEWAYS"),
        "confluence":       opp.get("confluence", "MIXED"),
        "ema_cross":        opp.get("ema_cross", "NONE"),
        "trend":            opp.get("trend", ""),
        "timeframe":        opp.get("timeframe", "4h"),
        "entry":            entry_p,
        "stop":             stop_p,
        "target":           target_p,
        "risk_reward":      rr,
        "market_type":       market_type,
        "leverage":          leverage,
        "dynamic_risk_pct":  dynamic_risk_pct,
        "position_value":    pos["position_value"]    if pos else None,
        "margin_required":   pos.get("margin_required") if pos else None,
        "liquidation_price": pos.get("liquidation_price") if pos else None,
        "funding_rate":      opp.get("funding_rate"),
        "qty":               pos["qty"]              if pos else None,
        "risk_amount":       pos["risk_amount"]      if pos else None,
        "capital_at_log":    pos["capital"]          if pos else None,
        "rsi":              opp.get("rsi"),
        "adx":              opp.get("adx"),
        "sentiment_score":  sent.get("overall"),
        "sentiment_label":  sent.get("label"),
        # outcome (filled in later)
        "status":           "OPEN",   # OPEN | WIN | LOSS | CANCELLED
        "exit_price":       None,
        "exit_time":        None,
        "pnl_pct":          None,     # % of price movement
        "pnl":              None,     # USDT profit/loss (capped at 2R for losses)
        "pnl_raw":          None,     # actual USDT P&L before any cap (audit trail)
        "risk_exceeded":    False,    # True if raw loss > 2× risk_amount
        "notes":            "",
        # trailing stop fields (off by default)
        "trailing_stop_enabled":  False,
        "trailing_stop_mode":     "breakeven",   # "breakeven" | "trailing"
        "trailing_stop_pct":      1.5,           # % distance for trailing mode
        "original_stop":          stop_p,        # original SL — never mutated
        "stop_updated_at":        None,
    }


# ── Storage ───────────────────────────────────────────────────

def _load() -> List[dict]:
    if not os.path.exists(JOURNAL_FILE):
        return []
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save(entries: List[dict]):
    os.makedirs(os.path.dirname(JOURNAL_FILE), exist_ok=True)
    with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


# ── Public API ────────────────────────────────────────────────

def log_signal(opp: dict) -> Optional[dict]:
    """
    Log a signal if it has entry/stop/target and hasn't been logged
    for this symbol in the last 4 hours (prevents duplicate logging
    across auto-scans).

    Returns the new entry dict, or None if skipped.
    """
    if not (opp.get("entry") and opp.get("stop") and opp.get("target")):
        return None

    # Spot markets cannot short — silently reject BEARISH signals
    if opp.get("market_type", "spot") == "spot" and opp.get("signal") == "BEARISH":
        return None

    with _lock:
        entries = _load()

        # ── Enforce concurrent trade limit atomically (inside lock so no race) ──
        open_count = sum(1 for e in entries if e.get("status") == "OPEN")
        if open_count >= cap_mgr.MAX_CONCURRENT_TRADES:
            return None

        # ── Enforce daily budget (prevents bypass via direct calls) ────────────
        try:
            budget = cap_mgr.get_daily_budget()
            if budget["daily_blocked"]:
                logger.warning(
                    "Journal: daily budget cap hit — skipping %s", opp.get("symbol")
                )
                return None
        except Exception as budget_err:
            logger.warning("Journal: could not check daily budget: %s", budget_err)

        # Deduplicate: same symbol with status OPEN (don't block re-logging after a trade closes)
        for e in entries:
            if e["symbol"] == opp["symbol"] and e.get("status") == "OPEN":
                return None   # already has an open trade for this symbol

        entry = _make_entry(opp)
        entries.insert(0, entry)            # newest first
        _save(entries)
        logger.info(f"Journal: logged {entry['symbol']} {entry['signal']} entry={entry['entry']}")
        return entry


def get_entries(limit: int = 200) -> List[dict]:
    with _lock:
        return _load()[:limit]


def update_entry(entry_id: str, status: str,
                 exit_price: Optional[float] = None,
                 notes: str = "") -> Optional[dict]:
    """
    Mark a trade as WIN / LOSS / CANCELLED and record exit price.
    Returns updated entry or None if not found.
    """
    status = status.upper()
    if status not in ("WIN", "LOSS", "CANCELLED", "OPEN"):
        raise ValueError(f"Invalid status: {status}")

    with _lock:
        entries = _load()
        for e in entries:
            if e["id"] == entry_id:
                e["status"]     = status
                e["exit_price"] = exit_price
                e["exit_time"]  = datetime.now(timezone.utc).isoformat() if status != "OPEN" else None
                e["notes"]      = notes or e.get("notes", "")

                # Compute P&L
                if exit_price and e.get("entry"):
                    pnl_pct = (exit_price - e["entry"]) / e["entry"] * 100
                    if e["signal"] == "BEARISH":
                        pnl_pct = -pnl_pct
                    e["pnl_pct"] = round(pnl_pct, 3)
                    pos_val     = e.get("position_value") or 0
                    risk_amount = e.get("risk_amount") or 0
                    raw_pnl     = round(pos_val * pnl_pct / 100, 2)
                    e["pnl_raw"] = raw_pnl

                    # Cap capital impact at 2R (2× intended risk) for losses
                    if risk_amount and raw_pnl < -(risk_amount * 2):
                        capped_pnl = round(-(risk_amount * 2), 2)
                        e["pnl"]           = capped_pnl
                        e["risk_exceeded"] = True
                        logger.warning(
                            "Journal %s: raw loss %s capped to %s (2R limit, risk_amount=%s)",
                            entry_id, raw_pnl, capped_pnl, risk_amount,
                        )
                    else:
                        e["pnl"]           = raw_pnl
                        e["risk_exceeded"] = False

                    # Compound capital using effective (potentially capped) pnl
                    if pos_val:
                        try:
                            effective_pct = e["pnl"] / pos_val * 100
                            cap_mgr.apply_trade_result(effective_pct, pos_val)
                        except Exception as cap_err:
                            logger.error(
                                "Capital update failed for %s — journal and capital.json "
                                "may be out of sync. Run /api/capital/reconcile to fix. "
                                "Error: %s",
                                entry_id, cap_err,
                            )

                _save(entries)
                logger.info(f"Journal: updated {entry_id} → {status}")
                return e
    return None


def update_stop(entry_id: str, new_stop: float) -> Optional[dict]:
    """Partially update the stop price of an OPEN entry (used by trailing stop logic)."""
    with _lock:
        entries = _load()
        for e in entries:
            if e["id"] == entry_id and e.get("status") == "OPEN":
                e["stop"] = round(new_stop, 8)
                e["stop_updated_at"] = datetime.now(timezone.utc).isoformat()
                _save(entries)
                logger.info(f"Journal: trailing stop updated {entry_id} stop→{new_stop:.6g}")
                return e
    return None


def set_trailing(entry_id: str, enabled: bool,
                 mode: str = "breakeven", pct: float = 1.5) -> Optional[dict]:
    """Enable / configure trailing stop for an OPEN entry."""
    with _lock:
        entries = _load()
        for e in entries:
            if e["id"] == entry_id and e.get("status") == "OPEN":
                e["trailing_stop_enabled"] = enabled
                e["trailing_stop_mode"]    = mode
                e["trailing_stop_pct"]     = pct
                # record the original stop the first time trailing is activated
                if enabled and not e.get("original_stop"):
                    e["original_stop"] = e.get("stop")
                _save(entries)
                return e
    return None


def delete_entry(entry_id: str) -> bool:
    """Permanently delete a single journal entry by ID. Returns True if found and deleted."""
    with _lock:
        entries = _load()
        new_entries = [e for e in entries if e["id"] != entry_id]
        if len(new_entries) == len(entries):
            return False
        _save(new_entries)
        logger.info(f"Journal: deleted entry {entry_id}")
        return True


def reset_journal() -> dict:
    """Clear all journal entries (backs up to journal_backup.json first)."""
    import shutil
    backup = JOURNAL_FILE.replace("journal.json", "journal_backup.json")
    with _lock:
        if os.path.exists(JOURNAL_FILE):
            shutil.copy2(JOURNAL_FILE, backup)
        _save([])
    logger.warning("Journal reset — backup at %s", backup)
    return {"ok": True, "backed_up_to": backup}


def get_summary() -> dict:
    """Performance summary over all closed trades."""
    entries = _load()
    closed  = [e for e in entries if e["status"] in ("WIN", "LOSS")]
    wins    = [e for e in closed if e["status"] == "WIN"]
    losses  = [e for e in closed if e["status"] == "LOSS"]

    pnls_pct  = [e["pnl_pct"] for e in closed if e.get("pnl_pct") is not None]
    pnls_usdt = [e["pnl"]     for e in closed if e.get("pnl")     is not None]
    rrs       = [e["risk_reward"] for e in entries if e.get("risk_reward")]
    scores    = [e["score"]       for e in entries if e.get("score")]

    win_rate   = round(len(wins) / len(closed) * 100, 1) if closed else 0
    avg_pnl    = round(sum(pnls_usdt) / len(pnls_usdt), 2) if pnls_usdt else 0
    total_pnl  = round(sum(pnls_usdt), 2) if pnls_usdt else 0
    avg_rr     = round(sum(rrs) / len(rrs), 2) if rrs else 0
    avg_score  = round(sum(scores) / len(scores) * 100, 1) if scores else 0

    evs = [e["expected_value"] for e in entries if e.get("expected_value") is not None]
    avg_ev = round(sum(evs) / len(evs), 3) if evs else 0

    return {
        "total":      len(entries),
        "open":       len([e for e in entries if e["status"] == "OPEN"]),
        "closed":     len(closed),
        "wins":       len(wins),
        "losses":     len(losses),
        "win_rate":   win_rate,
        "avg_pnl":    avg_pnl,
        "total_pnl":  total_pnl,
        "avg_rr":     avg_rr,
        "avg_score":  avg_score,
        "cancelled":  len([e for e in entries if e["status"] == "CANCELLED"]),
        "grade_a":    len([e for e in entries if e.get("grade") == "A"]),
        "grade_b":    len([e for e in entries if e.get("grade") == "B"]),
        "grade_c":    len([e for e in entries if e.get("grade") == "C"]),
        "avg_ev":     avg_ev,
    }
