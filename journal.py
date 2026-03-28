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
    grade = opp.get("grade", "B")
    try:
        dynamic_risk_pct = cap_mgr.get_dynamic_risk(grade)
    except Exception:
        dynamic_risk_pct = 1.5

    pos = None
    if entry_p and stop_p:
        try:
            pos = cap_mgr.compute_position(entry_p, stop_p, risk_pct=dynamic_risk_pct)
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
        "dynamic_risk_pct": dynamic_risk_pct,
        "position_value":   pos["position_value"] if pos else None,
        "qty":              pos["qty"]             if pos else None,
        "risk_amount":      pos["risk_amount"]     if pos else None,
        "capital_at_log":   pos["capital"]         if pos else None,
        "rsi":              opp.get("rsi"),
        "adx":              opp.get("adx"),
        "sentiment_score":  sent.get("overall"),
        "sentiment_label":  sent.get("label"),
        # outcome (filled in later)
        "status":           "OPEN",   # OPEN | WIN | LOSS | CANCELLED
        "exit_price":       None,
        "exit_time":        None,
        "pnl_pct":          None,
        "notes":            "",
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

    with _lock:
        entries = _load()

        # Deduplicate: same symbol within 4h
        cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
        for e in entries:
            if e["symbol"] == opp["symbol"]:
                try:
                    logged = datetime.fromisoformat(e["logged_at"])
                    if logged > cutoff:
                        return None          # already logged recently
                except Exception:
                    pass

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

                # Compute P&L %
                if exit_price and e.get("entry"):
                    pnl = (exit_price - e["entry"]) / e["entry"] * 100
                    if e["signal"] == "BEARISH":
                        pnl = -pnl
                    e["pnl_pct"] = round(pnl, 3)
                    # Compound capital
                    if e.get("position_value"):
                        try:
                            cap_mgr.apply_trade_result(pnl, e["position_value"])
                        except Exception:
                            pass

                _save(entries)
                logger.info(f"Journal: updated {entry_id} → {status}")
                return e
    return None


def get_summary() -> dict:
    """Performance summary over all closed trades."""
    entries = _load()
    closed  = [e for e in entries if e["status"] in ("WIN", "LOSS")]
    wins    = [e for e in closed if e["status"] == "WIN"]
    losses  = [e for e in closed if e["status"] == "LOSS"]

    pnls    = [e["pnl_pct"] for e in closed if e.get("pnl_pct") is not None]
    rrs     = [e["risk_reward"] for e in entries if e.get("risk_reward")]
    scores  = [e["score"] for e in entries if e.get("score")]

    win_rate = round(len(wins) / len(closed) * 100, 1) if closed else 0
    avg_pnl  = round(sum(pnls) / len(pnls), 2) if pnls else 0
    total_pnl = round(sum(pnls), 2) if pnls else 0
    avg_rr   = round(sum(rrs) / len(rrs), 2) if rrs else 0
    avg_score = round(sum(scores) / len(scores) * 100, 1) if scores else 0

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
