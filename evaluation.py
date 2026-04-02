"""
=============================================================
  TRADE EVALUATION ENGINE
  Reads logs/journal.json and computes comprehensive
  performance analytics:
    - Overall win rate + PnL
    - Per-grade breakdown (A / B / C)
    - Probability calibration accuracy
    - Recent-10 rolling performance
    - Capital growth stats
    - Risk metrics: Profit Factor, Expectancy, Payoff Ratio,
      consecutive win/loss streaks, risk-exceeded count
    - Return metrics: Sharpe, Sortino, Calmar, Recovery Factor
    - Monthly breakdown table

  No imports from capital/journal/scanner — reads JSON
  directly to avoid circular dependencies.
=============================================================
"""
import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

JOURNAL_FILE = os.path.join(os.path.dirname(__file__), "logs", "journal.json")
CAPITAL_FILE = os.path.join(os.path.dirname(__file__), "logs", "capital.json")

# Assume ~3 trades/week on average for annualisation
TRADES_PER_YEAR = 156   # 3 × 52


def _load_entries() -> list:
    if not os.path.exists(JOURNAL_FILE):
        return []
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("evaluation: could not read journal: %s", e)
        return []


def _load_capital() -> dict:
    if not os.path.exists(CAPITAL_FILE):
        return {}
    try:
        with open(CAPITAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _std(values: list) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _downside_std(values: list) -> float:
    """Standard deviation of negative values only (for Sortino)."""
    neg = [v for v in values if v < 0]
    if len(neg) < 2:
        return 0.0
    mean = 0.0  # Sortino uses 0 as the target return
    variance = sum(v ** 2 for v in neg) / len(neg)
    return math.sqrt(variance)


def get_evaluation() -> dict:
    """Return comprehensive trade performance evaluation."""
    entries = _load_entries()
    cap = _load_capital()

    # ── Partition entries ──────────────────────────────────────
    closed  = [e for e in entries if e.get("status") in ("WIN", "LOSS")]
    open_   = [e for e in entries if e.get("status") == "OPEN"]
    wins    = [e for e in closed if e.get("status") == "WIN"]
    losses  = [e for e in closed if e.get("status") == "LOSS"]
    total   = len(entries)

    def _pnl(e: dict) -> float:
        return float(e.get("pnl", 0) or 0)

    # ── Summary ────────────────────────────────────────────────
    total_pnl = sum(_pnl(e) for e in closed)
    avg_pnl   = total_pnl / len(closed) if closed else 0.0
    win_rate  = len(wins) / len(closed) if closed else 0.0

    best_trade  = None
    worst_trade = None
    if closed:
        best_e  = max(closed, key=_pnl)
        worst_e = min(closed, key=_pnl)
        best_trade  = {"symbol": best_e.get("symbol"),  "pnl": round(_pnl(best_e), 2),  "grade": best_e.get("grade", "?")}
        worst_trade = {"symbol": worst_e.get("symbol"), "pnl": round(_pnl(worst_e), 2), "grade": worst_e.get("grade", "?")}

    summary = {
        "total":       total,
        "open":        len(open_),
        "closed":      len(closed),
        "wins":        len(wins),
        "losses":      len(losses),
        "cancelled":   len([e for e in entries if e.get("status") == "CANCELLED"]),
        "win_rate":    round(win_rate * 100, 1),
        "total_pnl":   round(total_pnl, 2),
        "avg_pnl":     round(avg_pnl, 2),
        "best_trade":  best_trade,
        "worst_trade": worst_trade,
    }

    # ── Per-grade breakdown ────────────────────────────────────
    by_grade = {}
    for grade in ("A", "B", "C"):
        g_closed = [e for e in closed  if e.get("grade") == grade]
        g_wins   = [e for e in g_closed if e.get("status") == "WIN"]
        g_open   = [e for e in open_   if e.get("grade") == grade]
        g_pnl    = sum(_pnl(e) for e in g_closed)
        g_probs  = [float(e["probability"]) for e in entries
                    if e.get("grade") == grade and e.get("probability") is not None]
        by_grade[grade] = {
            "trades":    len(g_closed) + len(g_open),
            "open":      len(g_open),
            "closed":    len(g_closed),
            "wins":      len(g_wins),
            "losses":    len(g_closed) - len(g_wins),
            "win_rate":  round(len(g_wins) / len(g_closed) * 100, 1) if g_closed else 0.0,
            "total_pnl": round(g_pnl, 2),
            "avg_pnl":   round(g_pnl / len(g_closed), 2) if g_closed else 0.0,
            "avg_prob":  round(sum(g_probs) / len(g_probs) * 100, 1) if g_probs else 0.0,
        }

    # ── Probability calibration ────────────────────────────────
    closed_with_prob = [e for e in closed if e.get("probability") is not None]
    if closed_with_prob:
        expected_wr = sum(float(e["probability"]) for e in closed_with_prob) / len(closed_with_prob)
        actual_wr   = len([e for e in closed_with_prob if e.get("status") == "WIN"]) / len(closed_with_prob)
        cal_error   = abs(expected_wr - actual_wr)
    else:
        expected_wr = actual_wr = cal_error = 0.0

    calibration = {
        "expected_win_rate": round(expected_wr * 100, 1),
        "actual_win_rate":   round(actual_wr * 100, 1),
        "calibration_error": round(cal_error * 100, 1),
        "sample_size":       len(closed_with_prob),
        "note": (
            "Well calibrated" if cal_error < 0.05
            else "Slight overestimate" if expected_wr > actual_wr
            else "Slight underestimate"
        ) if closed_with_prob else "No closed trades yet",
    }

    # ── Recent 10 trades ──────────────────────────────────────
    recent = sorted(closed, key=lambda e: e.get("exit_time", "") or "", reverse=True)[:10]
    r_wins = [e for e in recent if e.get("status") == "WIN"]
    recent_10 = {
        "trades":    len(recent),
        "wins":      len(r_wins),
        "losses":    len(recent) - len(r_wins),
        "win_rate":  round(len(r_wins) / len(recent) * 100, 1) if recent else 0.0,
        "total_pnl": round(sum(_pnl(e) for e in recent), 2),
    }

    # ── Capital stats ─────────────────────────────────────────
    initial  = float(cap.get("initial", 10000))
    current  = float(cap.get("capital", initial))
    peak     = float(cap.get("peak", current))
    drawdown = round((peak - current) / peak * 100, 2) if peak > 0 else 0.0
    capital_stats = {
        "start":            initial,
        "current":          round(current, 2),
        "growth_pct":       round((current - initial) / initial * 100, 2),
        "peak":             round(peak, 2),
        "max_drawdown_pct": drawdown,
    }

    # ── Risk metrics ──────────────────────────────────────────
    win_pnls  = [_pnl(e) for e in wins  if _pnl(e) > 0]
    loss_pnls = [abs(_pnl(e)) for e in losses if _pnl(e) < 0]

    gross_wins   = sum(win_pnls)
    gross_losses = sum(loss_pnls)
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else None

    avg_win_usdt  = round(gross_wins  / len(win_pnls),  2) if win_pnls  else 0.0
    avg_loss_usdt = round(gross_losses / len(loss_pnls), 2) if loss_pnls else 0.0
    payoff_ratio  = round(avg_win_usdt / avg_loss_usdt, 2) if avg_loss_usdt > 0 else None

    expectancy = round(total_pnl / len(closed), 2) if closed else 0.0

    # Max consecutive wins / losses
    max_cons_wins = max_cons_losses = 0
    cur_wins = cur_losses = 0
    for e in sorted(closed, key=lambda x: x.get("exit_time", "") or ""):
        if e.get("status") == "WIN":
            cur_wins += 1; cur_losses = 0
            max_cons_wins = max(max_cons_wins, cur_wins)
        else:
            cur_losses += 1; cur_wins = 0
            max_cons_losses = max(max_cons_losses, cur_losses)

    largest_win_usdt  = round(max(win_pnls),   2) if win_pnls  else 0.0
    largest_loss_usdt = round(-max(loss_pnls), 2) if loss_pnls else 0.0

    risk_exceeded_count = len([e for e in entries if e.get("risk_exceeded")])

    risk_metrics = {
        "profit_factor":          profit_factor,
        "expectancy":             expectancy,
        "payoff_ratio":           payoff_ratio,
        "avg_win_usdt":           avg_win_usdt,
        "avg_loss_usdt":          avg_loss_usdt,
        "max_consecutive_wins":   max_cons_wins,
        "max_consecutive_losses": max_cons_losses,
        "largest_win_usdt":       largest_win_usdt,
        "largest_loss_usdt":      largest_loss_usdt,
        "risk_exceeded_count":    risk_exceeded_count,
    }

    # ── Return metrics (Sharpe, Sortino, Calmar) ──────────────
    # Use capital_at_log as baseline per trade; fall back to initial capital
    returns = []
    for e in sorted(closed, key=lambda x: x.get("exit_time", "") or ""):
        base = float(e.get("capital_at_log") or initial)
        if base > 0:
            returns.append(_pnl(e) / base)

    sharpe = sortino = calmar = recovery_factor = None

    if len(returns) >= 5:
        mean_r = sum(returns) / len(returns)
        std_r  = _std(returns)
        dstd_r = _downside_std(returns)

        if std_r > 0:
            sharpe = round(mean_r / std_r * math.sqrt(TRADES_PER_YEAR), 2)
        if dstd_r > 0:
            sortino = round(mean_r / dstd_r * math.sqrt(TRADES_PER_YEAR), 2)

    if drawdown > 0:
        total_return_pct = (current - initial) / initial * 100
        calmar          = round(total_return_pct / drawdown, 2)
        recovery_factor = round(total_pnl / (peak * drawdown / 100), 2) if (peak * drawdown) > 0 else None

    return_metrics = {
        "sharpe_ratio":   sharpe,
        "sortino_ratio":  sortino,
        "calmar_ratio":   calmar,
        "recovery_factor": recovery_factor,
    }

    # ── Monthly breakdown ─────────────────────────────────────
    monthly: dict = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for e in closed:
        ts = e.get("exit_time") or e.get("logged_at", "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            key = dt.strftime("%Y-%m")
        except Exception:
            continue
        monthly[key]["trades"] += 1
        if e.get("status") == "WIN":
            monthly[key]["wins"] += 1
        monthly[key]["pnl"] += _pnl(e)

    monthly_breakdown = []
    for month in sorted(monthly.keys()):
        m = monthly[month]
        n = m["trades"]
        monthly_breakdown.append({
            "month":    month,
            "trades":   n,
            "wins":     m["wins"],
            "losses":   n - m["wins"],
            "win_rate": round(m["wins"] / n * 100, 1) if n else 0.0,
            "pnl":      round(m["pnl"], 2),
        })

    # ── Equity curve ──────────────────────────────────────────
    # Cumulative capital after each closed trade, sorted by exit_time
    equity_curve = []
    running_cap = initial
    for e in sorted(closed, key=lambda x: x.get("exit_time", "") or ""):
        running_cap = round(running_cap + _pnl(e), 2)
        equity_curve.append({
            "date":    (e.get("exit_time") or "")[:10],
            "capital": running_cap,
            "symbol":  e.get("symbol", ""),
            "status":  e.get("status", ""),
        })

    # ── Action plan ───────────────────────────────────────────
    action_plan = _generate_action_plan(
        summary, by_grade, risk_metrics, return_metrics,
        monthly_breakdown, capital_stats
    )

    return {
        "summary":          summary,
        "by_grade":         by_grade,
        "calibration":      calibration,
        "recent_10":        recent_10,
        "capital":          capital_stats,
        "risk_metrics":     risk_metrics,
        "return_metrics":   return_metrics,
        "monthly_breakdown": monthly_breakdown,
        "equity_curve":     equity_curve,
        "action_plan":      action_plan,
    }


def _generate_action_plan(
    summary: dict,
    by_grade: dict,
    risk_metrics: dict,
    return_metrics: dict,
    monthly_breakdown: list,
    capital_stats: dict,
) -> list:
    """
    Rule-based trading improvement recommendations.
    Returns list of {priority, area, finding, action}.
    """
    plan = []

    def add(priority: str, area: str, finding: str, action: str):
        plan.append({"priority": priority, "area": area,
                     "finding": finding, "action": action})

    wr  = summary.get("win_rate", 0)
    n   = summary.get("closed", 0)
    pf  = risk_metrics.get("profit_factor") or 0
    mcl = risk_metrics.get("max_consecutive_losses") or 0
    dd  = abs(capital_stats.get("drawdown_pct") or 0)
    sharpe = return_metrics.get("sharpe_ratio")
    expectancy = risk_metrics.get("expectancy") or 0

    # Grades
    grade_a = by_grade.get("A", {})
    grade_b = by_grade.get("B", {})
    grade_c = by_grade.get("C", {})
    a_wr    = grade_a.get("win_rate", 0)
    b_wr    = grade_b.get("win_rate", 0)
    c_wr    = grade_c.get("win_rate", 0)
    c_pnl   = grade_c.get("total_pnl", 0)
    a_trades = grade_a.get("total", 0)
    c_trades = grade_c.get("total", 0)

    # Monthly
    neg_months = sum(1 for m in monthly_breakdown if m.get("pnl", 0) < 0)
    pos_months = sum(1 for m in monthly_breakdown if m.get("pnl", 0) > 0)

    # ── HIGH priority rules ──────────────────────────────────
    if n >= 5 and wr < 45:
        add("HIGH", "Win Rate",
            f"Win rate {wr}% is below breakeven — losing more trades than winning",
            "Only take Grade A setups for next 10 trades. Require MTF alignment + volume confirmation.")

    if dd > 8:
        add("HIGH", "Drawdown",
            f"Current drawdown {dd:.1f}% is elevated",
            "Cut per-trade risk to 0.25% until drawdown recovers below 3%. Avoid new C-grade trades.")

    if mcl >= 4:
        add("HIGH", "Consecutive Losses",
            f"{mcl} consecutive losses detected — likely in a losing streak",
            "Pause trading for 24h. Review setups: ensure EMA cross + volume spike before entry.")

    if n >= 10 and c_trades > 0 and c_pnl < -50:
        add("HIGH", "Grade C Losses",
            f"Grade C trades have lost ${abs(c_pnl):.0f} USDT total ({c_trades} trades)",
            "Stop taking Grade C trades entirely. Grade C expected value is negative.")

    # ── MEDIUM priority rules ────────────────────────────────
    if pf and 0 < pf < 1.3 and n >= 5:
        add("MEDIUM", "Profit Factor",
            f"Profit factor {pf:.2f} is near breakeven (>1.5 is healthy)",
            "Tighten entry criteria: require score ≥ 0.70 and ADX > 25 before logging a trade.")

    if n >= 10 and a_trades > 0 and a_wr > b_wr + 10:
        add("MEDIUM", "Grade Allocation",
            f"Grade A win rate {a_wr}% outperforms Grade B {b_wr}% — allocate more to A",
            "Increase Grade A position size while staying within 0.5% daily cap.")

    if neg_months > pos_months and len(monthly_breakdown) >= 3:
        add("MEDIUM", "Monthly Consistency",
            f"More losing months ({neg_months}) than winning ({pos_months})",
            "Review what happened in losing months — check if MTF alignment was off or news events were ignored.")

    if sharpe is not None and sharpe < 0.5 and n >= 10:
        add("MEDIUM", "Risk-Adjusted Return",
            f"Sharpe ratio {sharpe:.2f} is low — returns don't justify risk taken",
            "Improve R:R target — aim for minimum 2.5:1. Review where trades exited early or late.")

    # ── LOW priority rules ───────────────────────────────────
    if n >= 5 and expectancy < 5:
        add("LOW", "Expectancy",
            f"Average expectancy ${expectancy:.1f} USDT/trade is low",
            "Focus on higher-probability setups. Let winners run to full target before exiting.")

    if n >= 10 and c_wr > a_wr and c_trades >= 3:
        add("LOW", "Grade Calibration",
            "Grade C is outperforming Grade A — grading model may need recalibration",
            "Review Grade A criteria: check if MTF alignment filter is too strict.")

    if not plan:
        add("LOW", "No Issues",
            "Strategy metrics look healthy — no critical issues detected",
            "Continue current approach. Focus on maintaining >55% win rate and >1.5 profit factor.")

    return plan
