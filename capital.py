"""
=============================================================
  CAPITAL MANAGER
  Tracks 10,000 USDT starting capital with compounding.
  Computes risk-based position sizing per trade.
=============================================================
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

CAPITAL_FILE    = os.path.join(os.path.dirname(__file__), "logs", "capital.json")
_lock           = threading.Lock()
INITIAL_CAPITAL  = 10_000.0
DEFAULT_RISK_PCT = 1.5       # % of capital risked per trade

GRADE_BASE_RISK  = {"A": 2.5, "B": 1.5, "C": 0.75}
MAX_RISK_PCT     = 3.0
MIN_RISK_PCT     = 0.5
STREAK_LOOKBACK  = 10   # closed trades to check for streak


def _default_state() -> dict:
    return {
        "capital":      INITIAL_CAPITAL,
        "initial":      INITIAL_CAPITAL,
        "risk_pct":     DEFAULT_RISK_PCT,
        "peak":         INITIAL_CAPITAL,
        "total_trades": 0,
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }


def _load() -> dict:
    if not os.path.exists(CAPITAL_FILE):
        return _default_state()
    try:
        with open(CAPITAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _default_state()


def _save(state: dict):
    os.makedirs(os.path.dirname(CAPITAL_FILE), exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(CAPITAL_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def get_streak() -> dict:
    """
    Reads the last STREAK_LOOKBACK closed journal entries and
    returns consecutive win/loss counts from the most recent trade.
    Reads journal.json directly to avoid circular imports.
    """
    import json as _json
    journal_file = os.path.join(os.path.dirname(__file__), "logs", "journal.json")
    try:
        with open(journal_file, "r", encoding="utf-8") as f:
            entries = _json.load(f)
    except Exception:
        return {"win_streak": 0, "loss_streak": 0}

    closed = [e for e in entries if e.get("status") in ("WIN", "LOSS")][:STREAK_LOOKBACK]

    win_streak = loss_streak = 0
    for e in closed:
        if e["status"] == "WIN":
            if loss_streak == 0:
                win_streak += 1
            else:
                break
        else:
            if win_streak == 0:
                loss_streak += 1
            else:
                break

    return {"win_streak": win_streak, "loss_streak": loss_streak}


def get_dynamic_risk(grade: str) -> float:
    """
    Returns the actual risk % for a given signal grade, adjusted
    for recent win/loss streak and current drawdown.

    Grade base: A=2.5%, B=1.5%, C=0.75%
    Win streak ≥ 3  → × 1.20  (max 3.0%)
    Loss streak ≥ 2 → × 0.75  (min 0.5%)
    Drawdown > 10%  → × 0.50  (emergency preservation mode)
    """
    state  = _load()
    streak = get_streak()
    base   = GRADE_BASE_RISK.get(grade.upper(), DEFAULT_RISK_PCT)

    multiplier = 1.0
    if streak["win_streak"] >= 3:
        multiplier = 1.20
    elif streak["loss_streak"] >= 2:
        multiplier = 0.75

    # Drawdown override — most conservative rule wins
    dd = abs(state.get("drawdown_pct", 0) or 0)
    if dd > 10:
        multiplier = min(multiplier, 0.50)

    risk = round(base * multiplier, 2)
    return max(MIN_RISK_PCT, min(MAX_RISK_PCT, risk))


def get_state() -> dict:
    with _lock:
        s = _load()
        s["growth_pct"]   = round((s["capital"] / s["initial"] - 1) * 100, 2)
        s["drawdown_pct"] = round(
            (s["capital"] / s["peak"] - 1) * 100, 2
        ) if s["peak"] > 0 else 0.0
    # Streak and dynamic risk (reads journal, outside _lock to avoid deadlock)
    streak = get_streak()
    s["win_streak"]   = streak["win_streak"]
    s["loss_streak"]  = streak["loss_streak"]
    s["dynamic_risk"] = {
        "A": get_dynamic_risk("A"),
        "B": get_dynamic_risk("B"),
        "C": get_dynamic_risk("C"),
    }
    return s


def compute_position(entry: float, stop: float,
                     capital: Optional[float] = None,
                     risk_pct: Optional[float] = None) -> dict:
    """Compute position size from risk parameters."""
    state         = _load()
    cap           = capital   if capital   is not None else state["capital"]
    rp            = risk_pct  if risk_pct  is not None else state["risk_pct"]
    risk_amount   = cap * rp / 100
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return {"error": "entry equals stop"}
    qty            = risk_amount / risk_per_unit
    position_value = qty * entry
    position_pct   = position_value / cap * 100
    return {
        "capital":        round(cap, 2),
        "risk_pct":       round(rp, 2),
        "risk_amount":    round(risk_amount, 2),
        "qty":            round(qty, 8),
        "position_value": round(position_value, 2),
        "position_pct":   round(position_pct, 2),
        "entry":          entry,
        "stop":           stop,
    }


def apply_trade_result(pnl_pct: float, position_value: float) -> dict:
    """
    Compound result into capital.
    pnl_pct: percentage gain/loss on the position value.
    """
    with _lock:
        state    = _load()
        pnl_usdt = position_value * pnl_pct / 100
        state["capital"] = max(0.01, round(state["capital"] + pnl_usdt, 4))
        if state["capital"] > state["peak"]:
            state["peak"] = state["capital"]
        state["total_trades"] = state.get("total_trades", 0) + 1
        _save(state)
        logger.info(
            f"Capital: {state['capital']:.2f} USDT  pnl={pnl_usdt:+.2f}"
        )
        return state


def update_settings(risk_pct: Optional[float] = None,
                    reset: bool = False) -> dict:
    with _lock:
        state = _default_state() if reset else _load()
        if risk_pct is not None:
            state["risk_pct"] = round(max(0.1, min(10.0, risk_pct)), 2)
        _save(state)
        return state
