"""Lightweight state snapshot + control channel for the Streamlit UI.

This avoids running an API server:
  - Engine writes logs/state.json
  - UI reads logs/state.json
  - UI writes logs/control.json with {"command": "pause"|"resume"|"stop"}
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{int(time.time() * 1000)}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)


def _trade_to_dict(trade) -> Dict[str, Any]:
    return {
        "id": getattr(trade, "id", None),
        "symbol": getattr(trade, "symbol", None),
        "side": getattr(getattr(trade, "side", None), "value", None),
        "strategy": getattr(trade, "strategy", ""),
        "entry_price": getattr(trade, "entry_price", None),
        "quantity": getattr(trade, "quantity", None),
        "stop_price": getattr(trade, "stop_price", None),
        "target_price": getattr(trade, "target_price", None),
        "trailing_stop": getattr(trade, "trailing_stop", None),
        "entry_time": trade.entry_time.isoformat() if getattr(trade, "entry_time", None) else None,
    }


def write_state(
    *,
    cfg,
    portfolio,
    last_cycle_seconds: Optional[float] = None,
    last_error: Optional[str] = None,
    paused: bool = False,
) -> None:
    """Write logs/state.json snapshot."""
    log_dir = getattr(getattr(cfg, "logging", None), "log_dir", "logs/")
    state_path = os.path.join(log_dir, "state.json")

    open_trades = [_trade_to_dict(t) for t in portfolio.open_trades.values()]
    snap = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": getattr(cfg, "mode", ""),
        "dry_run": getattr(cfg, "dry_run", None),
        "timeframe": getattr(getattr(cfg, "universe", None), "timeframe", ""),
        "symbols": list(getattr(getattr(cfg, "universe", None), "symbols", []) or []),
        "paused": paused,
        "portfolio": {
            "equity": portfolio.equity,
            "cash": portfolio.cash,
            "peak_equity": portfolio.peak_equity,
            "drawdown_pct": portfolio.current_drawdown_pct,
            "max_drawdown_pct": portfolio.max_drawdown_pct,
            "daily_pnl": portfolio.daily_pnl,
            "weekly_pnl": portfolio.weekly_pnl,
            "total_pnl": portfolio.total_pnl,
            "win_rate": portfolio.win_rate,
            "profit_factor": portfolio.profit_factor,
            "circuit_breaker": getattr(getattr(portfolio, "circuit_breaker", None), "value", str(portfolio.circuit_breaker)),
        },
        "open_trades": open_trades,
        "stats": {
            "num_open_positions": portfolio.num_open_positions,
            "num_closed_trades": len(portfolio.closed_trades),
            "last_cycle_seconds": last_cycle_seconds,
        },
        "last_error": last_error,
    }
    _atomic_write_json(state_path, snap)


def read_control_command(cfg) -> Optional[str]:
    """Read logs/control.json and return pause|resume|stop or None."""
    log_dir = getattr(getattr(cfg, "logging", None), "log_dir", "logs/")
    control_path = os.path.join(log_dir, "control.json")
    if not os.path.exists(control_path):
        return None
    try:
        with open(control_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cmd = (data.get("command") or "").strip().lower()
        if cmd in {"pause", "resume", "stop"}:
            return cmd
    except Exception:
        return None
    return None


def clear_control_command(cfg) -> None:
    log_dir = getattr(getattr(cfg, "logging", None), "log_dir", "logs/")
    control_path = os.path.join(log_dir, "control.json")
    try:
        if os.path.exists(control_path):
            os.remove(control_path)
    except Exception:
        pass
