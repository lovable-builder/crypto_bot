"""
=============================================================
  PROBABILITY ENGINE
  Maps composite score + signal context to estimated win
  probability, signal grade, and expected value per trade.

  Design principles:
  - Conservative calibration: cap at 0.72 (never overfit)
  - Linear base mapping: score 0.65 → 0.50, score 1.0 → 0.68
  - Additive bonuses for independent confirming factors
  - Grade A only when ALL three gates pass (prob, score, MTF)
=============================================================
"""

import math
from typing import Optional

# ── Calibration constants ─────────────────────────────────────

_SCORE_LOW  = 0.65   # score at which base prob = 0.50
_SCORE_HIGH = 1.00   # score at which base prob = 0.68
_PROB_LOW   = 0.50
_PROB_HIGH  = 0.68
_PROB_CAP   = 0.72   # never claim better than 72%

_BONUS_MTF        = 0.05   # daily trend aligned with 4h signal
_BONUS_SENTIMENT  = 0.03   # strong sentiment confluence (STRONG label)
_BONUS_ADX        = 0.03   # ADX > 30 (confirmed trending market)

# ── Grade thresholds ──────────────────────────────────────────

_GRADE_A_PROB  = 0.60
_GRADE_A_SCORE = 0.75
_GRADE_B_PROB  = 0.54
_GRADE_B_SCORE = 0.65

# ── Grade-default risk % ──────────────────────────────────────

GRADE_RISK = {"A": 2.5, "B": 1.5, "C": 0.75}


def _isnan(v) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return v is None


def estimate_win_probability(
    score: float,
    mtf_aligned: bool,
    sentiment_confluence: bool,
    adx: Optional[float],
    rsi: Optional[float] = None,   # reserved for future calibration
) -> float:
    """
    Returns estimated win probability in [0.40, 0.72].

    Linear interpolation from the composite score, then additive
    bonuses for independent confirming factors.

    The hard cap at 0.72 is intentional — models are always
    imperfect and overconfidence is the fastest way to blow
    an account.

    Args:
        score:               Weighted composite score (0–1) from strategy
        mtf_aligned:         True if daily trend agrees with 4h signal
        sentiment_confluence:True if sentiment label is STRONG match
        adx:                 14-period ADX (None if unavailable)
        rsi:                 14-period RSI (reserved, not yet penalising)
    """
    score = max(_SCORE_LOW, min(1.0, float(score)))

    # Linear base probability
    slope = (_PROB_HIGH - _PROB_LOW) / (_SCORE_HIGH - _SCORE_LOW)
    base_prob = _PROB_LOW + slope * (score - _SCORE_LOW)

    # Additive bonuses
    bonus = 0.0
    if mtf_aligned:
        bonus += _BONUS_MTF
    if sentiment_confluence:
        bonus += _BONUS_SENTIMENT
    if adx is not None and not _isnan(adx) and adx > 30:
        bonus += _BONUS_ADX

    prob = base_prob + bonus
    return round(min(_PROB_CAP, max(0.40, prob)), 4)


def signal_grade(
    probability: float,
    score: float,
    mtf_aligned: bool,
) -> str:
    """
    Returns 'A', 'B', or 'C'.

    Grade A — highest confidence, warrants 2.5% risk per trade.
      Requires prob ≥ 0.60 AND score ≥ 0.75 AND daily MTF aligned.
      The MTF gate is strict: a high-scoring counter-trend trade
      is still only Grade B at best.

    Grade B — solid signal, standard 1.5% risk.
      Requires prob ≥ 0.54 AND score ≥ 0.65.

    Grade C — marginal setup, 0.75% risk (informational).
    """
    if probability >= _GRADE_A_PROB and score >= _GRADE_A_SCORE and mtf_aligned:
        return "A"
    if probability >= _GRADE_B_PROB and score >= _GRADE_B_SCORE:
        return "B"
    return "C"


def expected_value_pct(
    probability: float,
    rr_ratio: float,
    risk_pct: float,
) -> float:
    """
    Expected value as a percentage of capital per trade.

    EV = P(win) × (R:R × risk%) − P(loss) × risk%

    Example: P=0.62, R:R=3.0, risk=2.5%
      EV = 0.62 × 7.5% − 0.38 × 2.5% = 4.65% − 0.95% = +3.70%

    A positive EV means the strategy has mathematical edge.
    """
    win_return = rr_ratio * risk_pct
    ev = probability * win_return - (1.0 - probability) * risk_pct
    return round(ev, 3)
