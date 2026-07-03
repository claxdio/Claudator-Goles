from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchState:
    minute: int
    home_score: int
    away_score: int
    home_xg_last15: float
    away_xg_last15: float
    home_shots_last15: int
    away_shots_last15: int


def compute_state_at_minute(
    shots: list[dict], cutoff_minute: int, window: int = 15
) -> MatchState:
    """Reconstruct match state as of `cutoff_minute`, using only shots with
    minute <= cutoff_minute (so a backtest never sees the future)."""
    home_score = sum(
        1 for s in shots if s["minute"] <= cutoff_minute and s["team"] == "home" and s["is_goal"]
    )
    away_score = sum(
        1 for s in shots if s["minute"] <= cutoff_minute and s["team"] == "away" and s["is_goal"]
    )

    window_start = cutoff_minute - window
    home_xg = sum(
        s["xg"] for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "home"
    )
    away_xg = sum(
        s["xg"] for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "away"
    )
    home_shots = sum(
        1 for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "home"
    )
    away_shots = sum(
        1 for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "away"
    )

    return MatchState(
        minute=cutoff_minute,
        home_score=home_score,
        away_score=away_score,
        home_xg_last15=home_xg,
        away_xg_last15=away_xg,
        home_shots_last15=home_shots,
        away_shots_last15=away_shots,
    )


def goal_in_window(shots: list[dict], cutoff_minute: int, horizon: int, team: str) -> bool:
    """Did `team` score a goal in (cutoff_minute, cutoff_minute + horizon]?
    Used only to *label* historical data for backtesting/training — never
    call this with information a live model wouldn't have yet."""
    return any(
        s["team"] == team and s["is_goal"] and cutoff_minute < s["minute"] <= cutoff_minute + horizon
        for s in shots
    )


def compute_ml_features(shots: list[dict], cutoff_minute: int, team: str) -> dict[str, float]:
    """Computes an engineered feature set for predicting whether `team`
    ("home" or "away") scores in the next 15 minutes, from `team`'s own
    perspective (own_* vs opp_*), using only shots with minute <=
    cutoff_minute (no look-ahead) -- the same discipline as
    `compute_state_at_minute`.

    Deliberately asymmetric by design: only `team`'s own recent-form trend
    and time-since-last-shot are included, not the opponent's -- keeping
    the feature count modest relative to the available training data (see
    this plan's Global Constraints on overfitting risk at this data scale).
    The opponent's own trend/recency is captured when this function is
    called again with `team` set to the opponent for that team's own
    prediction row.
    """
    opponent = "away" if team == "home" else "home"
    past_shots = [s for s in shots if s["minute"] <= cutoff_minute]
    own_shots = [s for s in past_shots if s["team"] == team]
    opp_shots = [s for s in past_shots if s["team"] == opponent]

    own_goals = sum(1 for s in own_shots if s["is_goal"])
    opp_goals = sum(1 for s in opp_shots if s["is_goal"])

    own_xg_total = sum(s["xg"] for s in own_shots)
    opp_xg_total = sum(s["xg"] for s in opp_shots)

    minutes_elapsed = max(cutoff_minute, 1)
    minutes_remaining = float(max(90 - cutoff_minute, 0))

    own_xg_rate = own_xg_total / minutes_elapsed
    opp_xg_rate = opp_xg_total / minutes_elapsed

    own_max_shot_xg = max((s["xg"] for s in own_shots), default=0.0)
    opp_max_shot_xg = max((s["xg"] for s in opp_shots), default=0.0)

    own_big_chances = float(sum(1 for s in own_shots if s["xg"] > 0.2))
    opp_big_chances = float(sum(1 for s in opp_shots if s["xg"] > 0.2))

    recent_window_start = cutoff_minute - 15
    own_recent_xg = sum(s["xg"] for s in own_shots if s["minute"] > recent_window_start)
    opp_recent_xg = sum(s["xg"] for s in opp_shots if s["minute"] > recent_window_start)
    own_recent_rate = own_recent_xg / 15.0
    own_trend = own_recent_rate / own_xg_rate if own_xg_rate > 0 else 0.0

    own_last_shot_minute = max((s["minute"] for s in own_shots), default=0)
    own_time_since_shot = float(cutoff_minute - own_last_shot_minute)

    goal_minutes = [s["minute"] for s in past_shots if s["is_goal"]]
    time_since_goal = float(cutoff_minute - max(goal_minutes)) if goal_minutes else float(cutoff_minute)

    score_diff = float(own_goals - opp_goals)

    return {
        "is_home": 1.0 if team == "home" else 0.0,
        "minute": float(cutoff_minute),
        "minutes_remaining": minutes_remaining,
        "score_diff": score_diff,
        "score_diff_x_minutes_remaining": score_diff * minutes_remaining,
        "own_xg_total": own_xg_total,
        "opp_xg_total": opp_xg_total,
        "xg_diff": own_xg_total - opp_xg_total,
        "own_xg_rate": own_xg_rate,
        "opp_xg_rate": opp_xg_rate,
        "own_max_shot_xg": own_max_shot_xg,
        "opp_max_shot_xg": opp_max_shot_xg,
        "own_big_chances": own_big_chances,
        "opp_big_chances": opp_big_chances,
        "own_recent_xg": own_recent_xg,
        "opp_recent_xg": opp_recent_xg,
        "own_trend": own_trend,
        "own_time_since_shot": own_time_since_shot,
        "time_since_goal": time_since_goal,
    }
