"""
=============================================================
  AI ANALYST
  Uses OpenAI GPT-4o to analyse trading performance data
  and generate structured, actionable insights.

  Features:
    - Sends compact JSON summary to GPT-4o (not raw journal)
    - Returns structured sections: overall_assessment,
      strengths, weaknesses, risk_warnings, recommendations,
      grade_insight, market_insight
    - 6-hour cache in logs/ai_analysis_cache.json
    - Reads OPENAI_API_KEY from environment variable
=============================================================
"""
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_FILE    = os.path.join(os.path.dirname(__file__), "logs", "ai_analysis_cache.json")
CACHE_TTL_H   = 6         # hours before cache is considered stale
GPT_MODEL     = "gpt-4o"
MIN_TRADES    = 3         # minimum closed trades required before calling AI


# ── Cache helpers ─────────────────────────────────────────────

def _load_cache() -> Optional[dict]:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(analysis: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    payload = {
        "analysis":     analysis,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _cache_is_fresh(cached: dict) -> bool:
    ts = cached.get("generated_at")
    if not ts:
        return False
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(ts)
        return age < timedelta(hours=CACHE_TTL_H)
    except Exception:
        return False


def get_cached_analysis() -> Optional[dict]:
    """Return cached analysis if it exists (regardless of freshness)."""
    cached = _load_cache()
    if not cached:
        return None
    return {
        "analysis":     cached.get("analysis"),
        "generated_at": cached.get("generated_at"),
        "fresh":        _cache_is_fresh(cached),
    }


def cache_is_stale() -> bool:
    cached = _load_cache()
    if not cached:
        return True
    return not _cache_is_fresh(cached)


# ── Prompt builder ────────────────────────────────────────────

def _build_prompt(evaluation: dict, recent_trades: list) -> str:
    s  = evaluation.get("summary", {})
    rm = evaluation.get("risk_metrics", {})
    rt = evaluation.get("return_metrics", {})
    bg = evaluation.get("by_grade", {})
    mb = evaluation.get("monthly_breakdown", [])
    cp = evaluation.get("capital", {})

    def _fmt(v, suffix="", decimals=2):
        if v is None:
            return "N/A"
        return f"{round(float(v), decimals)}{suffix}"

    # Compact payload — avoids token bloat while giving all key numbers
    payload = {
        "performance": {
            "closed_trades":        s.get("closed", 0),
            "win_rate_pct":         s.get("win_rate", 0),
            "total_pnl_usdt":       s.get("total_pnl", 0),
            "avg_pnl_usdt":         s.get("avg_pnl", 0),
            "best_trade":           s.get("best_trade"),
            "worst_trade":          s.get("worst_trade"),
        },
        "risk": {
            "profit_factor":          rm.get("profit_factor"),
            "expectancy_usdt":        rm.get("expectancy"),
            "payoff_ratio":           rm.get("payoff_ratio"),
            "avg_win_usdt":           rm.get("avg_win_usdt"),
            "avg_loss_usdt":          rm.get("avg_loss_usdt"),
            "max_consecutive_losses": rm.get("max_consecutive_losses"),
            "max_consecutive_wins":   rm.get("max_consecutive_wins"),
            "largest_loss_usdt":      rm.get("largest_loss_usdt"),
            "risk_exceeded_count":    rm.get("risk_exceeded_count", 0),
        },
        "ratios": {
            "sharpe_ratio":   rt.get("sharpe_ratio"),
            "sortino_ratio":  rt.get("sortino_ratio"),
            "calmar_ratio":   rt.get("calmar_ratio"),
        },
        "capital": {
            "start_usdt":       cp.get("start"),
            "current_usdt":     cp.get("current"),
            "growth_pct":       cp.get("growth_pct"),
            "max_drawdown_pct": cp.get("max_drawdown_pct"),
        },
        "by_grade": {
            g: {
                "trades":    d.get("trades"),
                "win_rate":  d.get("win_rate"),
                "total_pnl": d.get("total_pnl"),
                "avg_prob":  d.get("avg_prob"),
            }
            for g, d in bg.items()
        },
        "last_3_months": mb[-3:] if mb else [],
        "last_20_trades": [
            {
                "symbol":   t.get("symbol"),
                "signal":   t.get("signal"),
                "grade":    t.get("grade"),
                "status":   t.get("status"),
                "pnl":      t.get("pnl"),
                "pnl_pct":  t.get("pnl_pct"),
                "risk_rwd": t.get("risk_reward"),
                "prob":     round(float(t["probability"]) * 100) if t.get("probability") else None,
            }
            for t in recent_trades[-20:]
        ],
    }

    system_prompt = (
        "You are a professional quantitative trading analyst reviewing a crypto trader's "
        "performance journal. Be concise, direct, and specific — always cite the numbers. "
        "Focus on what the trader should DO differently, not generic platitudes. "
        "If something is good, say so clearly. If something is dangerous, say so firmly. "
        "The trader uses an EMA crossover strategy on Gate.io with grades A/B/C for signal quality."
    )

    user_prompt = f"""Analyse this trading performance data and return a JSON object with exactly these keys:

- overall_assessment: string (2–3 sentences summarising the trader's current state)
- strengths: array of 3 strings (specific positives with numbers)
- weaknesses: array of 3 strings (specific problems with numbers)
- risk_warnings: array of strings (0–4 items, only if genuinely concerning — e.g. low profit_factor, high drawdown, risk_exceeded_count > 0, many consecutive losses)
- recommendations: array of 4–5 strings (specific, actionable changes with clear rationale)
- grade_insight: string (1–2 sentences on A vs B vs C performance differences)
- market_insight: string (1 sentence on spot vs futures if relevant, or "Insufficient futures data" if not)

Performance data:
{json.dumps(payload, indent=2)}

Respond with ONLY the JSON object, no markdown wrapping."""

    return system_prompt, user_prompt


# ── Main analysis function ────────────────────────────────────

async def analyze_performance(evaluation: dict, recent_trades: list) -> dict:
    """
    Call GPT-4o with the evaluation summary.
    Returns the structured analysis dict.
    Raises ValueError if API key is missing or data is insufficient.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    closed = evaluation.get("summary", {}).get("closed", 0)
    if closed < MIN_TRADES:
        raise ValueError(f"Need at least {MIN_TRADES} closed trades for AI analysis (have {closed})")

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ValueError("openai package not installed — run: pip install openai")

    client = AsyncOpenAI(api_key=api_key)
    system_prompt, user_prompt = _build_prompt(evaluation, recent_trades)

    logger.info("AI analyst: calling %s…", GPT_MODEL)
    response = await client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    analysis = json.loads(content)

    # Ensure all expected keys exist (graceful fallback)
    for key in ("overall_assessment", "strengths", "weaknesses",
                "risk_warnings", "recommendations", "grade_insight", "market_insight"):
        if key not in analysis:
            analysis[key] = [] if key in ("strengths", "weaknesses", "risk_warnings", "recommendations") else "—"

    _save_cache(analysis)
    logger.info("AI analyst: analysis generated and cached")
    return analysis
