from __future__ import annotations

import math


def dynamic_lambda(
    pre_match_xg_per90: float,
    in_match_xg_recent: float,
    recent_window_minutes: int,
    horizon_minutes: int,
    blend: float = 0.5,
) -> float:
    """Estimate expected goals for a team over the next `horizon_minutes`,
    blending a pre-match prior rate (from full-match expected xG per 90)
    with the observed in-match rate over the last `recent_window_minutes`.

    `blend` is the weight given to the in-match observed rate: 0 ignores
    the live signal entirely, 1 ignores the pre-match prior entirely.
    """
    if recent_window_minutes <= 0:
        raise ValueError("recent_window_minutes must be positive")
    prior_rate_per_minute = pre_match_xg_per90 / 90.0
    observed_rate_per_minute = in_match_xg_recent / recent_window_minutes
    blended_rate_per_minute = (
        blend * observed_rate_per_minute + (1 - blend) * prior_rate_per_minute
    )
    return blended_rate_per_minute * horizon_minutes


def prob_goal_in_window(expected_goals: float) -> float:
    """Convert an expected-goals value (a Poisson lambda) into the
    probability of at least one goal occurring: 1 - P(zero goals)."""
    if expected_goals < 0:
        raise ValueError("expected_goals cannot be negative")
    return 1.0 - math.exp(-expected_goals)
