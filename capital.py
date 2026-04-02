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
DEFAULT_RISK_PCT = 1.0       # % of capital risked per trade (Grade B default)

GRADE_BASE_RISK  = {"A": 1.5, "B": 1.0, "C": 0.5}
MAX_RISK_PCT     = 1.5   # hard cap — never lose more than 1.5% of capital on one trade
MIN_RISK_PCT     = 0.25
STREAK_LOOKBACK  = 10   # closed trades to check for streak
GRADE_LEVERAGE   = {"A": 3, "B": 2, "C": 1}   # futures only; spot and grade C always 1x

DAILY_LOSS_CAP_PCT    = 1.0   # max total loss per UTC day (% of capital)
MAX_CONCURRENT_TRADES = 2     # daily budget divided across this many simultaneous trades


def _default_state() -> dict:
    return {
        "capital":        INITIAL_CAPITAL,
        "initial":        INITIAL_CAPITAL,
        "risk_pct":       DEFAULT_RISK_PCT,
        "peak":           INITIAL_CAPITAL,
        "total_trades":   0,
        "daily_loss_usdt": 0.0,
        "daily_date":     "",
        "updated_at":     datetime.now(timezone.utc).isoformat(),
    }


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _reset_daily_if_new_day(state: dict) -> bool:
    """Reset daily loss tracking if UTC date has changed. Returns True if reset."""
    today = _today_utc()
    if state.get("daily_date") != today:
        state["daily_loss_usdt"] = 0.0
        state["daily_date"] = today
        return True
    return False


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


def get_leverage(grade: str, market_type: str) -> int:
    """Return leverage for a trade. Futures: A=3x, B=2x. Spot or C: always 1x."""
    if market_type != "futures" or grade.upper() == "C":
        return 1
    return GRADE_LEVERAGE.get(grade.upper(), 1)


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
    s["grade_leverage"]  = GRADE_LEVERAGE
    s["daily_budget"]    = get_daily_budget()
    return s


def compute_position(entry: float, stop: float,
                     capital: Optional[float] = None,
                     risk_pct: Optional[float] = None,
                     leverage: int = 1,
                     signal: str = "BULLISH") -> dict:
    """Compute position size from risk parameters. Leverage reduces margin required."""
    state         = _load()
    cap           = capital   if capital   is not None else state["capital"]
    rp            = risk_pct  if risk_pct  is not None else state["risk_pct"]
    risk_amount   = cap * rp / 100
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return {"error": "entry equals stop"}
    # With leverage, qty is amplified so same risk_amount covers same dollar loss
    lev           = max(1, int(leverage))
    qty            = (risk_amount / risk_per_unit) * lev
    position_value = qty * entry          # full notional exposure
    margin_required = position_value / lev

    # Hard cap: margin must not exceed 20% of capital (prevents stablecoin blowup
    # and oversized positions when stop distance is tiny)
    MAX_MARGIN_PCT = 20.0
    max_margin = cap * MAX_MARGIN_PCT / 100
    if margin_required > max_margin:
        margin_required = max_margin
        position_value  = margin_required * lev
        qty             = position_value / entry

    margin_required = round(margin_required, 2)
    position_pct   = margin_required / cap * 100   # % of capital locked as margin
    # Approximate liquidation price (90% of margin consumed = liquidated)
    if signal.upper() == "BEARISH":
        liq_price = round(entry * (1 + 0.9 / lev), 6)
    else:
        liq_price = round(entry * (1 - 0.9 / lev), 6)
    return {
        "capital":          round(cap, 2),
        "risk_pct":         round(rp, 2),
        "risk_amount":      round(risk_amount, 2),
        "qty":              round(qty, 8),
        "position_value":   round(position_value, 2),
        "margin_required":  margin_required,
        "position_pct":     round(position_pct, 2),
        "leverage":         lev,
        "liquidation_price": liq_price,
        "entry":            entry,
        "stop":             stop,
    }


def apply_trade_result(pnl_pct: float, position_value: float) -> dict:
    """
    Compound result into capital and track daily loss.
    pnl_pct: percentage gain/loss on the position value.
    """
    with _lock:
        state    = _load()
        _reset_daily_if_new_day(state)
        pnl_usdt = position_value * pnl_pct / 100
        state["capital"] = max(0.01, round(state["capital"] + pnl_usdt, 4))
        if state["capital"] > state["peak"]:
            state["peak"] = state["capital"]
        state["total_trades"] = state.get("total_trades", 0) + 1
        # Track daily losses (only accumulate losses)
        if pnl_usdt < 0:
            state["daily_loss_usdt"] = round(
                state.get("daily_loss_usdt", 0.0) + abs(pnl_usdt), 4
            )
        _save(state)
        logger.info(
            f"Capital: {state['capital']:.2f} USDT  pnl={pnl_usdt:+.2f}  "
            f"daily_loss={state['daily_loss_usdt']:.2f}"
        )
        return state


def get_daily_budget() -> dict:
    """
    Returns the current day's loss budget status.

    per_trade_risk_pct = DAILY_LOSS_CAP_PCT / MAX_CONCURRENT_TRADES
    e.g. 1% cap / 2 trades = 0.5% per trade max
    """
    with _lock:
        state = _load()
        _reset_daily_if_new_day(state)
        cap        = state["capital"]
        daily_cap  = round(cap * DAILY_LOSS_CAP_PCT / 100, 2)
        daily_loss = state.get("daily_loss_usdt", 0.0)
        per_trade_pct = round(DAILY_LOSS_CAP_PCT / MAX_CONCURRENT_TRADES, 2)
        blocked    = daily_loss >= daily_cap
        _save(state)
    return {
        "daily_cap_usdt":       daily_cap,
        "daily_loss_usdt":      round(daily_loss, 2),
        "daily_remaining_usdt": round(max(0.0, daily_cap - daily_loss), 2),
        "daily_used_pct":       round(daily_loss / daily_cap * 100, 1) if daily_cap else 0.0,
        "daily_blocked":        blocked,
        "per_trade_risk_pct":   per_trade_pct,
        "daily_cap_pct":        DAILY_LOSS_CAP_PCT,
        "max_concurrent":       MAX_CONCURRENT_TRADES,
    }


def rebuild_from_journal(closed_entries: list) -> dict:
    """Rebuild capital.json from journal entries to fix sync drift.

    Idempotent: calling twice gives the same result.
    Useful after manual journal edits or to detect platform bugs.
    """
    with _lock:
        state = _load()
        initial = state.get("initial", INITIAL_CAPITAL)

        # Re-apply all closed trades in chronological order
        running_cap = initial
        peak_cap    = initial

        for e in sorted(closed_entries, key=lambda x: x.get("exit_time", "") or ""):
            pnl = float(e.get("pnl") or 0)
            running_cap = round(running_cap + pnl, 2)
            if running_cap > peak_cap:
                peak_cap = round(running_cap, 2)

        state["capital"]      = max(0.01, running_cap)
        state["peak"]         = peak_cap
        state["initial"]      = initial
        state["updated_at"]   = datetime.now(timezone.utc).isoformat()
        _save(state)

        logger.info(f"Capital rebuilt: initial={initial}, current={state['capital']}, peak={peak_cap} from {len(closed_entries)} trades")
        return state


def update_settings(risk_pct: Optional[float] = None,
                    reset: bool = False,
                    new_capital: Optional[float] = None) -> dict:
    with _lock:
        state = _default_state() if reset else _load()
        if risk_pct is not None:
            state["risk_pct"] = round(max(0.1, min(10.0, risk_pct)), 2)
        if new_capital is not None:
            amount = round(max(1.0, new_capital), 2)
            state["capital"] = amount
            state["initial"] = amount   # reset baseline to new deposit
            state["peak"]    = amount
            logger.info("Capital manually set to %.2f USDT", amount)
        _save(state)
        return state
